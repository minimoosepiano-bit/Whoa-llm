"""SFT training core: Full / LoRA / QLoRA.

A single :func:`run_sft` function dispatches between the three methods,
loads the model via :mod:`whoa_llm.engines`, applies PEFT wrappers, and
hands off to ``trl.SFTTrainer``.

All ML libraries are imported lazily so importing this module does not
require ``torch``/``transformers``/``trl`` to be installed — useful for
help text, tests, and config validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config schema
# ----------------------------------------------------------------------------
class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None  # auto-detected when None
    bias: Literal["none", "all", "lora_only"] = "none"


class TrainConfig(BaseModel):
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"  # overridden to adamw_torch for "full" on CPU
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 2
    max_steps: int = -1  # -1 means use epochs
    gradient_checkpointing: bool = True


class MemoryConfig(BaseModel):
    """Memory knobs.  Phase 4 fills these in via the recommender."""
    quantization: Literal["none", "4bit", "8bit"] = "none"
    cpu_offload: bool = False
    disk_offload_dir: str | None = None
    attn_impl: Literal["sdpa", "flash_attention_2", "eager"] | None = None
    max_memory: dict[str, str] | None = None


class DatasetConfig(BaseModel):
    name: str
    config: str | None = None
    split: str = "train"
    format: Literal["alpaca", "sharegpt", "completion", "raw_text", "chat_template"] = "alpaca"
    column_map: dict[str, str] | None = None
    max_samples: int | None = None


class SFTConfig(BaseModel):
    model_id: str
    dataset: DatasetConfig
    method: Literal["full", "lora", "qlora"] = "lora"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    seed: int = 42
    output_dir: str = "outputs/sft"
    run_name: str | None = None
    engine: Literal["auto", "unsloth", "hf"] = "auto"
    trust_remote_code: bool = False

    @field_validator("method")
    @classmethod
    def _qlora_implies_4bit(cls, v: str, info):
        # QLoRA implies 4-bit quantization; we set it in __post_init__-like fashion below.
        return v

    def normalised(self) -> SFTConfig:
        """Return a copy with cross-field invariants enforced."""
        cfg = self.model_copy(deep=True)
        if cfg.method == "qlora":
            cfg.memory.quantization = "4bit"
        if cfg.method == "full" and cfg.train.optim == "paged_adamw_8bit":
            # bnb optimisers don't help for full FT (and break on CPU).
            cfg.train.optim = "adamw_torch"
        return cfg


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _default_lora_targets(model: Any) -> list[str]:
    """Best-effort target-module discovery for LoRA.

    Returns the union of common attention/MLP linear names found in the model.
    Falls back to a Llama-style default if nothing matches.
    """
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",     # Llama / Mistral / Qwen
        "gate_proj", "up_proj", "down_proj",        # ditto
        "Wqkv", "out_proj",                          # GPT-NeoX, Phi
        "query_key_value", "dense",                  # Falcon
        "c_attn", "c_proj",                          # GPT-2
    }
    found: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates and module.__class__.__name__.lower().endswith("linear"):
            found.add(leaf)
    if found:
        return sorted(found)
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _build_training_args(cfg: SFTConfig, output_dir: Path) -> Any:
    """Build TRL's SFTConfig (subclass of HF TrainingArguments)."""
    import inspect

    from trl import SFTConfig as TrlSFTConfig

    bf16 = cfg.precision == "bf16"
    fp16 = cfg.precision == "fp16"

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        optim=cfg.train.optim,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        max_steps=cfg.train.max_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        seed=cfg.seed,
        report_to="none",
        run_name=cfg.run_name or f"sft-{cfg.method}",
        packing=False,
    )

    sig = inspect.signature(TrlSFTConfig.__init__).parameters

    # Newer trl uses ``max_length``; older versions used ``max_seq_length``.
    if "max_length" in sig:
        kwargs["max_length"] = cfg.train.max_seq_length
    elif "max_seq_length" in sig:
        kwargs["max_seq_length"] = cfg.train.max_seq_length

    # ``warmup_ratio`` is being phased out in transformers v5 — prefer it when
    # available, else translate to warmup_steps lazily.
    if "warmup_ratio" in sig:
        kwargs["warmup_ratio"] = cfg.train.warmup_ratio
    if cfg.train.gradient_checkpointing and "gradient_checkpointing_kwargs" in sig:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrlSFTConfig(**kwargs)


def _wrap_with_peft(model: Any, cfg: SFTConfig, engine: str) -> Any:
    """Apply LoRA / QLoRA adapters; return the wrapped model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.train.gradient_checkpointing,
        )

    target_modules = cfg.lora.target_modules or _default_lora_targets(model)
    logger.info("LoRA target modules: %s", target_modules)

    if engine == "unsloth":
        from whoa_llm.engines.unsloth_engine import get_peft_model as unsloth_peft
        return unsloth_peft(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=cfg.train.gradient_checkpointing,
        )

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def run_sft(
    cfg: SFTConfig,
    *,
    metrics_callback: Any | None = None,
    extra_callbacks: list[Any] | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    """Run a single SFT job and return a summary dict.

    Parameters
    ----------
    cfg:
        Validated :class:`SFTConfig`.
    metrics_callback:
        Optional :class:`~whoa_llm.training.callbacks.MetricsCallback` instance;
        the UI passes one in to receive live metrics.

    Returns
    -------
    dict
        ``{"output_dir": str, "final_loss": float, "steps": int}``.
    """
    cfg = cfg.normalised()

    from trl import SFTTrainer

    from whoa_llm.data.formatting import make_formatting_func
    from whoa_llm.data.hf_datasets import load_dataset_split
    from whoa_llm.engines import hf_engine, unsloth_engine
    from whoa_llm.engines.registry import pick_engine

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config in both JSON and YAML for resume/reproducibility.
    (output_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    try:
        import yaml
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump({"type": "sft", **cfg.model_dump()}, sort_keys=False)
        )
    except ImportError:
        pass

    # ---------------- engine selection & model load ------------------------
    engine = pick_engine(cfg.model_id, force=None if cfg.engine == "auto" else cfg.engine)
    logger.info("Selected engine: %s", engine)

    load_kwargs = dict(
        quantization=cfg.memory.quantization,
        dtype=cfg.precision,
        device_map="auto",
        offload_dir=cfg.memory.disk_offload_dir,
        max_memory=cfg.memory.max_memory,
        attn_impl=cfg.memory.attn_impl,
        cpu_offload=cfg.memory.cpu_offload,
        trust_remote_code=cfg.trust_remote_code,
    )

    if engine == "unsloth":
        try:
            model, tokenizer = unsloth_engine.load_model_and_tokenizer(
                cfg.model_id,
                max_seq_length=cfg.train.max_seq_length,
                **load_kwargs,
            )
        except unsloth_engine.EngineUnavailable as exc:
            logger.warning("Unsloth load failed (%s); falling back to HF engine.", exc)
            engine = "hf"
            model, tokenizer = hf_engine.load_model_and_tokenizer(cfg.model_id, **load_kwargs)
    else:
        model, tokenizer = hf_engine.load_model_and_tokenizer(cfg.model_id, **load_kwargs)

    # ---------------- PEFT wrapper -----------------------------------------
    if cfg.method in {"lora", "qlora"}:
        model = _wrap_with_peft(model, cfg, engine)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:  # full
        logger.info("Full fine-tune: no PEFT wrapper.")

    # ---------------- dataset & formatting ---------------------------------
    dataset = load_dataset_split(
        cfg.dataset.name,
        config=cfg.dataset.config,
        split=cfg.dataset.split,
        max_samples=cfg.dataset.max_samples,
    )
    formatting_func = make_formatting_func(
        cfg.dataset.format, tokenizer, column_map=cfg.dataset.column_map
    )

    # ---------------- trainer ----------------------------------------------
    args = _build_training_args(cfg, output_dir)

    import inspect

    sftt_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=dataset,
        formatting_func=formatting_func,
    )
    # ``processing_class`` (TRL >=0.11) or ``tokenizer`` (older).
    if "processing_class" in sftt_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sftt_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    callbacks: list[Any] = []
    if metrics_callback is not None:
        callbacks.append(metrics_callback)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SFTTrainer(**trainer_kwargs)

    # ---------------- run ---------------------------------------------------
    train_kwargs: dict[str, Any] = {}
    if resume_from_checkpoint is not None:
        train_kwargs["resume_from_checkpoint"] = resume_from_checkpoint
        logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)
    result = trainer.train(**train_kwargs)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "output_dir": str(output_dir),
        "final_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "engine": engine,
        "method": cfg.method,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("SFT done: %s", summary)
    return summary
