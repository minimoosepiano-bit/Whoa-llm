"""Unsloth engine: fast, memory-efficient loading for supported model families.

Raises :class:`EngineUnavailable` if Unsloth is not installed or the model
fails to load through it; the registry catches this and falls back to HF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """Raised when the Unsloth engine cannot handle a request."""


def load_model_and_tokenizer(
    model_id: str,
    *,
    quantization: str = "none",  # "none" | "4bit" | "8bit"
    dtype: str = "auto",         # "auto" | "bf16" | "fp16" | "fp32"
    device_map: str = "auto",    # ignored by Unsloth but kept for API compat
    offload_dir: str | Path | None = None,  # not supported by Unsloth; logged
    max_memory: dict[Any, str] | None = None,  # not supported; logged
    cpu_offload: bool = False,   # not supported; logged
    attn_impl: str | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    max_seq_length: int = 2048,
) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer using Unsloth's ``FastLanguageModel``.

    Returns
    -------
    (model, tokenizer)
        Unsloth-wrapped model and HF tokenizer.

    Raises
    ------
    EngineUnavailable
        If Unsloth is not installed or raises during loading.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise EngineUnavailable("Unsloth is not installed.") from exc

    if offload_dir is not None or cpu_offload:
        logger.warning(
            "Unsloth engine does not support CPU/disk offloading; "
            "ignoring offload_dir=%s cpu_offload=%s. "
            "Use the HF engine for low-VRAM offload.",
            offload_dir, cpu_offload,
        )
    if max_memory is not None:
        logger.warning(
            "Unsloth engine does not support max_memory; ignoring. "
            "Use the HF engine if you need per-device memory caps."
        )

    load_in_4bit = quantization == "4bit"
    load_in_8bit = quantization == "8bit"

    # Unsloth handles dtype internally; map our strings to its expectations.
    dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "auto": None}
    unsloth_dtype = dtype_map.get(dtype)

    logger.info(
        "Unsloth engine: loading %s (4bit=%s, 8bit=%s, seq_len=%s)",
        model_id, load_in_4bit, load_in_8bit, max_seq_length,
    )
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=unsloth_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception as exc:
        raise EngineUnavailable(
            f"Unsloth failed to load {model_id!r}: {exc}"
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Unsloth engine: model loaded successfully")
    return model, tokenizer


def get_peft_model(
    model: Any,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    use_gradient_checkpointing: bool = True,
) -> Any:
    """Wrap a model loaded by Unsloth with LoRA adapters."""
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise EngineUnavailable("Unsloth is not installed.") from exc

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=42,
    )
