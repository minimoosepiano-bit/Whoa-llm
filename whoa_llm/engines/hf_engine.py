"""HF engine: load any causal LM via ``transformers``.

Supports optional 4-bit / 8-bit quantisation via ``bitsandbytes``, and
CPU / disk offloading via ``accelerate``'s ``device_map="auto"``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _supports_dtype_kwarg() -> bool:
    """transformers >=4.45 accepts ``dtype=``; older versions need ``torch_dtype=``."""
    try:
        import inspect

        from transformers import AutoModelForCausalLM

        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        return "dtype" in sig.parameters
    except Exception:
        return False


def load_model_and_tokenizer(
    model_id: str,
    *,
    quantization: str = "none",  # "none" | "4bit" | "8bit"
    dtype: str = "auto",         # "auto" | "bf16" | "fp16" | "fp32"
    device_map: str = "auto",
    offload_dir: str | Path | None = None,
    max_memory: dict[Any, str] | None = None,
    attn_impl: str | None = None,  # "sdpa" | "flash_attention_2" | "eager" | None
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer using the plain HF stack.

    Returns
    -------
    (model, tokenizer)
        Both are standard ``transformers`` objects.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ---------- dtype -------------------------------------------------------
    _dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if dtype not in _dtype_map:
        raise ValueError(f"dtype must be one of {list(_dtype_map)}, got {dtype!r}")
    torch_dtype = _dtype_map[dtype]

    # ---------- quantisation config -----------------------------------------
    bnb_config: BitsAndBytesConfig | None = None
    load_in_4bit = False
    load_in_8bit = False

    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_in_4bit = True
        logger.info("HF engine: loading in 4-bit NF4 (QLoRA-ready)")
    elif quantization == "8bit":
        load_in_8bit = True
        logger.info("HF engine: loading in 8-bit")
    elif quantization != "none":
        raise ValueError(f"quantization must be 'none', '4bit', or '8bit', got {quantization!r}")

    # ---------- attention implementation ------------------------------------
    model_kwargs: dict[str, Any] = {}
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    # ---------- offload / device map ----------------------------------------
    if offload_dir is not None:
        offload_dir = Path(offload_dir)
        offload_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_dir)
        model_kwargs["offload_state_dict"] = True

    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    # ---------- load --------------------------------------------------------
    logger.info("HF engine: loading tokenizer for %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("HF engine: loading model %s (quantization=%s, device_map=%s)",
                model_id, quantization, device_map)
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    elif load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    # transformers >=4.40 prefers the ``dtype`` kwarg over ``torch_dtype``.
    dtype_kwarg = "dtype" if _supports_dtype_kwarg() else "torch_dtype"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **{dtype_kwarg: torch_dtype},
        **model_kwargs,
    )

    logger.info("HF engine: model loaded successfully")
    return model, tokenizer
