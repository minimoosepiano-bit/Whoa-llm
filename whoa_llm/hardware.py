"""Detect available hardware and emit a recommended training preset.

Kept dependency-light: ``torch`` is imported lazily so the rest of the
package (CLI, settings) is usable on machines where torch isn't installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


if __name__ == "__main__":  # pragma: no cover - manual smoke
    print(detect().summary())
