"""GRPO (Group Relative Policy Optimization) trainer wrapper.

Mirrors the SFT pipeline (engine selection, quantisation, offload, PEFT)
and hands off to ``trl.GRPOTrainer``.  The user supplies one or more
reward functions; they can be built-ins (``"length_reward"``), dotted
paths (``"pkg.mod.func"``), file-pasted code in ``~/.whoa_llm/user_rewards/``,
or callables passed directly to :func:`run_grpo`.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator

from whoa_llm.training.sft import (
    DatasetConfig,
    LoRAConfig,
    MemoryConfig,
    TrainConfig,
    _wrap_with_peft,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
class RewardSpec(BaseModel):
    """A single reward function reference plus optional kwargs/weight."""
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class GRPOConfig(BaseModel):
    """Configuration for a GRPO run.

    GRPO is RL-style fine-tuning that doesn't need preference pairs; you
    just give it a *prompt* dataset and one or more reward functions
    (callables returning ``list[float]`` per generation group).
    """

    model_id: str
    dataset: DatasetConfig
    method: Literal["lora", "qlora", "full"] = "lora"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    seed: int = 42
    output_dir: str = "outputs/grpo"
    run_name: str | None = None
    engine: Literal["auto", "unsloth", "hf"] = "auto"
    trust_remote_code: bool = False

    # ---- GRPO-specific knobs ------------------------------------------------
    rewards: list[RewardSpec] = Field(default_factory=list)
    num_generations: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 256
    beta: float = 0.04
    temperature: float = 1.0
    top_p: float = 1.0
    use_vllm: bool = False
    prompt_column: str = "prompt"

    @field_validator("rewards")
    @classmethod
    def _require_rewards(cls, v: list[RewardSpec]) -> list[RewardSpec]:
        # Empty list is allowed for config validation tests but run_grpo will refuse.
        return v

    def normalised(self) -> GRPOConfig:
        cfg = self.model_copy(deep=True)
        if cfg.method == "qlora":
            cfg.memory.quantization = "4bit"
        if cfg.method == "full" and cfg.train.optim == "paged_adamw_8bit":
            cfg.train.optim = "adamw_torch"
        return cfg


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _build_grpo_args(cfg: GRPOConfig, output_dir: Path) -> Any:
    """Construct ``trl.GRPOConfig`` from our pydantic config.

    Only forwards kwargs the installed ``GRPOConfig`` actually accepts so we
    don't break across TRL versions. We do *not* reuse the SFT args builder
    because SFT/GRPO disagree on defaults for fields like ``loss_type``.
    """
    from trl import GRPOConfig as TrlGRPOConfig

    sig = inspect.signature(TrlGRPOConfig.__init__).parameters
    bf16 = cfg.precision == "bf16"
    fp16 = cfg.precision == "fp16"

    candidates: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.train.epochs,
        "learning_rate": cfg.train.learning_rate,
        "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "warmup_ratio": cfg.train.warmup_ratio,
        "weight_decay": cfg.train.weight_decay,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "optim": cfg.train.optim,
        "logging_steps": cfg.train.logging_steps,
        "save_steps": cfg.train.save_steps,
        "save_total_limit": cfg.train.save_total_limit,
        "max_steps": cfg.train.max_steps,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
        "bf16": bf16,
        "fp16": fp16,
        "seed": cfg.seed,
        "report_to": "none",
        "run_name": cfg.run_name or f"grpo-{cfg.method}",
        # GRPO-specific
        "num_generations": cfg.num_generations,
        "max_completion_length": cfg.max_completion_length,
        "max_prompt_length": cfg.max_prompt_length,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "beta": cfg.beta,
        "use_vllm": cfg.use_vllm,
        "reward_weights": [r.weight for r in cfg.rewards] or None,
    }
    if cfg.train.gradient_checkpointing and "gradient_checkpointing_kwargs" in sig:
        candidates["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    final = {k: v for k, v in candidates.items() if k in sig and v is not None}
    return TrlGRPOConfig(**final)


def _resolve_reward_callables(
    rewards: list[RewardSpec] | list[Callable],
) -> list[Callable[..., list[float]]]:
    from whoa_llm.training.rewards.loader import resolve_reward

    out: list[Callable[..., list[float]]] = []
    for r in rewards:
        if isinstance(r, RewardSpec):
            out.append(resolve_reward(r.name, **r.kwargs))
        else:
            out.append(resolve_reward(r))
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def run_grpo(
    cfg: GRPOConfig,
    *,
    metrics_callback: Any | None = None,
    extra_callbacks: list[Any] | None = None,
    reward_funcs_override: list[Callable] | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    """Run a GRPO job and return a summary dict.

    Parameters
    ----------
    cfg:
        Validated :class:`GRPOConfig`.
    metrics_callback / extra_callbacks:
        Same role as for :func:`whoa_llm.training.sft.run_sft`.
    reward_funcs_override:
        If provided, bypass ``cfg.rewards`` resolution and use these
        callables directly (handy for tests and for UI-pasted rewards).

    Returns
    -------
    dict
        ``{"output_dir", "final_loss", "steps", "engine", "method"}``.
    """
    cfg = cfg.normalised()

    from trl import GRPOTrainer

    from whoa_llm.data.hf_datasets import load_dataset_split
    from whoa_llm.engines import hf_engine, unsloth_engine
    from whoa_llm.engines.registry import pick_engine

    if reward_funcs_override is not None:
        reward_funcs = list(reward_funcs_override)
    else:
        if not cfg.rewards:
            raise ValueError("At least one reward function is required for GRPO.")
        reward_funcs = _resolve_reward_callables(cfg.rewards)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    try:
        import yaml
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump({"type": "grpo", **cfg.model_dump()}, sort_keys=False)
        )
    except ImportError:
        pass

    # ---------------- engine + model ----------------------------------------
    engine = pick_engine(cfg.model_id, force=None if cfg.engine == "auto" else cfg.engine)
    logger.info("GRPO engine: %s", engine)

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
                cfg.model_id, max_seq_length=cfg.train.max_seq_length, **load_kwargs,
            )
        except unsloth_engine.EngineUnavailable as exc:
            logger.warning("Unsloth load failed (%s); falling back to HF.", exc)
            engine = "hf"
            model, tokenizer = hf_engine.load_model_and_tokenizer(cfg.model_id, **load_kwargs)
    else:
        model, tokenizer = hf_engine.load_model_and_tokenizer(cfg.model_id, **load_kwargs)

    # ---------------- PEFT --------------------------------------------------
    if cfg.method in {"lora", "qlora"}:
        # Reuse the SFT helper; SFTConfig and GRPOConfig share the relevant fields.
        from whoa_llm.training.sft import SFTConfig as _SFTConfig

        sft_view = _SFTConfig(
            model_id=cfg.model_id,
            dataset=cfg.dataset,
            method=cfg.method,
            lora=cfg.lora,
            train=cfg.train,
            memory=cfg.memory,
            precision=cfg.precision,
            seed=cfg.seed,
            output_dir=cfg.output_dir,
            engine=cfg.engine,
        )
        model = _wrap_with_peft(model, sft_view, engine)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    elif cfg.method == "full":
        logger.warning(
            "Full GRPO is heavy and rarely what you want. Consider LoRA/QLoRA."
        )

    # ---------------- dataset -----------------------------------------------
    dataset = load_dataset_split(
        cfg.dataset.name,
        config=cfg.dataset.config,
        split=cfg.dataset.split,
        max_samples=cfg.dataset.max_samples,
    )
    # GRPO expects a "prompt" column; rename if the user pointed at another.
    if cfg.prompt_column != "prompt" and hasattr(dataset, "rename_column"):
        if cfg.prompt_column in dataset.column_names:  # type: ignore[attr-defined]
            dataset = dataset.rename_column(cfg.prompt_column, "prompt")
        else:
            raise ValueError(
                f"prompt_column={cfg.prompt_column!r} not found in dataset columns "
                f"{list(dataset.column_names)}"  # type: ignore[attr-defined]
            )

    # ---------------- trainer -----------------------------------------------
    args = _build_grpo_args(cfg, output_dir)

    trainer_sig = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
    )
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    callbacks: list[Any] = []
    if metrics_callback is not None:
        callbacks.append(metrics_callback)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = GRPOTrainer(**trainer_kwargs)

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
        "num_rewards": len(reward_funcs),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("GRPO done: %s", summary)
    return summary
