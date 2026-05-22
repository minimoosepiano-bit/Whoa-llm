"""Detect available hardware and emit a recommended training preset.

Kept dependency-light: ``torch`` is imported lazily so the rest of the
package (CLI, settings) is usable on machines where torch isn't installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import psutil


@dataclass
class GPUInfo:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: tuple[int, int] | None = None
    supports_bf16: bool = False


@dataclass
class HardwarePreset:
    has_cuda: bool
    gpus: list[GPUInfo] = field(default_factory=list)
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    recommended_precision: str = "fp32"  # "bf16" | "fp16" | "fp32"
    recommended_device_map: str = "cpu"  # "auto" | "cuda" | "cpu"

    @property
    def primary_vram_gb(self) -> float:
        return self.gpus[0].total_vram_gb if self.gpus else 0.0

    def summary(self) -> str:
        lines = [
            f"CUDA available: {self.has_cuda}",
            f"RAM: {self.available_ram_gb:.1f} / {self.total_ram_gb:.1f} GB free",
        ]
        for g in self.gpus:
            lines.append(
                f"GPU {g.index}: {g.name} | "
                f"{g.free_vram_gb:.1f}/{g.total_vram_gb:.1f} GB free | "
                f"bf16={g.supports_bf16}"
            )
        lines.append(
            f"Recommended: precision={self.recommended_precision}, "
            f"device_map={self.recommended_device_map}"
        )
        return "\n".join(lines)


def _gather_gpus() -> list[GPUInfo]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / (1024**3)
        try:
            free_bytes, _ = torch.cuda.mem_get_info(i)
            free = free_bytes / (1024**3)
        except Exception:
            free = total
        cc = (props.major, props.minor)
        # bf16 is reliable on Ampere (8.0) and newer.
        supports_bf16 = cc >= (8, 0)
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                total_vram_gb=round(total, 2),
                free_vram_gb=round(free, 2),
                compute_capability=cc,
                supports_bf16=supports_bf16,
            )
        )
    return gpus


def detect() -> HardwarePreset:
    """Probe the local machine and return a :class:`HardwarePreset`."""
    vm = psutil.virtual_memory()
    total_ram = vm.total / (1024**3)
    avail_ram = vm.available / (1024**3)

    gpus = _gather_gpus()
    has_cuda = bool(gpus)

    if has_cuda:
        precision = "bf16" if gpus[0].supports_bf16 else "fp16"
        device_map = "auto"
    else:
        precision = "fp32"
        device_map = "cpu"

    return HardwarePreset(
        has_cuda=has_cuda,
        gpus=gpus,
        total_ram_gb=round(total_ram, 2),
        available_ram_gb=round(avail_ram, 2),
        recommended_precision=precision,
        recommended_device_map=device_map,
    )


# ---------------------------------------------------------------------------
# Memory recommender
# ---------------------------------------------------------------------------
@dataclass
class MemoryRecommendation:
    """A concrete memory plan for a given model on the local hardware."""

    method: Literal["full", "lora", "qlora"]
    quantization: Literal["none", "8bit", "4bit"]
    cpu_offload: bool
    disk_offload_dir: str | None
    gradient_checkpointing: bool
    attn_impl: Literal["sdpa", "flash_attention_2", "eager"] | None
    max_memory: dict[str, str] | None
    precision: Literal["bf16", "fp16", "fp32"]
    # Human-readable breakdown for the UI.
    breakdown: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_memory_config(self) -> dict:
        """Return the dict that matches :class:`whoa_llm.training.sft.MemoryConfig`."""
        return {
            "quantization": self.quantization,
            "cpu_offload": self.cpu_offload,
            "disk_offload_dir": self.disk_offload_dir,
            "attn_impl": self.attn_impl,
            "max_memory": self.max_memory,
        }


# Bytes per parameter at various dtypes / quantizations.
# 4-bit is 0.5 bytes; we add a small overhead for the double-quant scales (~0.05).
_BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "8bit": 1.0,
    "4bit": 0.55,
}


def _activation_budget_gb(params_b: float, seq_len: int, batch: int) -> float:
    """Rough activation-memory estimate (with gradient checkpointing on)."""
    # Empirical fit: ~2 * params * seq_len * batch * 2 bytes / 1e9 / sqrt(layers)
    # We don't know layers here, so use a simple linear approximation.
    return (params_b * seq_len * batch * 2 * 2) / 1e9 / 16.0


def estimate_param_count(model_id: str, *, trust_remote_code: bool = False) -> float | None:
    """Estimate parameter count (in billions) for an HF model without downloading weights.

    Reads only ``config.json`` via ``AutoConfig.from_pretrained`` and computes
    parameter count from the architecture's hidden size / num layers / vocab.
    Returns ``None`` when the architecture isn't recognised.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        return None

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        return None

    # Prefer the model's own reporting when present.
    if hasattr(cfg, "num_parameters") and isinstance(cfg.num_parameters, (int, float)):
        return float(cfg.num_parameters) / 1e9

    h = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    n_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
    )
    vocab = getattr(cfg, "vocab_size", None)
    intermediate = getattr(cfg, "intermediate_size", None) or (4 * h if h else None)

    if not (h and n_layers and vocab):
        return None

    # Approximate parameter count for a transformer decoder:
    #   embeddings  = vocab * h
    #   per layer   = 4 * h^2 (attn) + 3 * h * intermediate (MLP, gated)
    per_layer = 4 * h * h + 3 * h * intermediate
    total = vocab * h + n_layers * per_layer + h  # +final layer norm-ish
    return total / 1e9


def recommend_memory(
    model_size_b: float,
    preset: HardwarePreset | None = None,
    *,
    method: Literal["full", "lora", "qlora"] = "lora",
    seq_len: int = 1024,
    batch: int = 1,
    disk_offload_dir: str = "outputs/.offload",
) -> MemoryRecommendation:
    """Recommend memory settings for *model_size_b* (in billions of params).

    Heuristic:

    * VRAM headroom ≥ 2 × weights at fp16 → no quantisation, no offload.
    * VRAM ≥ 0.6 × model_size (4-bit weights fit + activations) →
      ``qlora`` with 4-bit, gradient checkpointing, no offload.
    * Else → 4-bit + ``cpu_offload=True`` + ``disk_offload_dir`` and a
      ``max_memory`` map that caps GPU/CPU usage.
    """
    if preset is None:
        preset = detect()

    vram = preset.primary_vram_gb
    free_vram = preset.gpus[0].free_vram_gb if preset.gpus else 0.0
    ram = preset.available_ram_gb

    weights_bf16 = model_size_b * _BYTES_PER_PARAM["bf16"]
    weights_4bit = model_size_b * _BYTES_PER_PARAM["4bit"]
    act = _activation_budget_gb(model_size_b, seq_len, batch)

    breakdown = {
        "model_size_b": round(model_size_b, 3),
        "weights_bf16_gb": round(weights_bf16, 2),
        "weights_4bit_gb": round(weights_4bit, 2),
        "activations_gb_est": round(act, 2),
        "free_vram_gb": round(free_vram, 2),
        "available_ram_gb": round(ram, 2),
    }

    precision = preset.recommended_precision  # bf16 on Ampere+; fp32 on CPU
    attn_impl: Literal["sdpa", "flash_attention_2", "eager"] | None = (
        "sdpa" if preset.has_cuda else None
    )

    # ---- Plenty of VRAM: full/LoRA at bf16, no offload ----------------------
    if free_vram >= weights_bf16 * 2 + act:
        return MemoryRecommendation(
            method=method,
            quantization="none",
            cpu_offload=False,
            disk_offload_dir=None,
            gradient_checkpointing=method in {"qlora", "full"},
            attn_impl=attn_impl,
            max_memory=None,
            precision=precision,
            breakdown=breakdown,
            rationale=(
                f"Free VRAM ({free_vram:.1f} GB) ≥ 2× weights at bf16 "
                f"({weights_bf16:.1f} GB) + activations. No quantisation needed."
            ),
        )

    # ---- Tight but workable: QLoRA 4-bit on GPU -----------------------------
    if free_vram >= weights_4bit + act + 1.0:
        return MemoryRecommendation(
            method="qlora",
            quantization="4bit",
            cpu_offload=False,
            disk_offload_dir=None,
            gradient_checkpointing=True,
            attn_impl=attn_impl,
            max_memory=None,
            precision=precision,
            breakdown=breakdown,
            rationale=(
                f"Free VRAM ({free_vram:.1f} GB) fits 4-bit weights "
                f"({weights_4bit:.1f} GB) + activations. Forcing QLoRA."
            ),
        )

    # ---- Doesn't fit: QLoRA + CPU/disk offload -------------------------------
    if free_vram > 1.0:  # have *some* GPU
        gpu_cap = max(int(free_vram - 1), 1)
        cpu_cap = max(int(ram - 4), 1)
        max_memory = {0: f"{gpu_cap}GiB", "cpu": f"{cpu_cap}GiB"}
        return MemoryRecommendation(
            method="qlora",
            quantization="4bit",
            cpu_offload=True,
            disk_offload_dir=disk_offload_dir,
            gradient_checkpointing=True,
            attn_impl=attn_impl,
            max_memory=max_memory,
            precision=precision,
            breakdown=breakdown,
            rationale=(
                f"Free VRAM ({free_vram:.1f} GB) can't fit 4-bit weights "
                f"({weights_4bit:.1f} GB); spilling to CPU "
                f"({cpu_cap} GiB) and disk ({disk_offload_dir})."
            ),
        )

    # ---- No GPU: CPU-only LoRA at fp32 --------------------------------------
    return MemoryRecommendation(
        method="lora" if method != "full" else "full",
        quantization="none",
        cpu_offload=False,
        disk_offload_dir=None,
        gradient_checkpointing=True,
        attn_impl=None,
        max_memory=None,
        precision="fp32",
        breakdown=breakdown,
        rationale=(
            f"No CUDA GPU detected. Running on CPU at fp32 — "
            f"viable only for very small models (<=1B)."
        ),
    )


def estimate_memory(
    model_id: str,
    *,
    method: Literal["full", "lora", "qlora"] = "lora",
    seq_len: int = 1024,
    batch: int = 1,
    trust_remote_code: bool = False,
    preset: HardwarePreset | None = None,
) -> MemoryRecommendation | None:
    """One-call helper for the UI: read model config, then recommend memory.

    Returns ``None`` if the model's config can't be read (offline / private
    repo) — the UI should fall back to manual settings in that case.
    """
    size = estimate_param_count(model_id, trust_remote_code=trust_remote_code)
    if size is None:
        return None
    return recommend_memory(
        size, preset, method=method, seq_len=seq_len, batch=batch,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke
    print(detect().summary())
