"""Tests for SFTConfig validation and normalisation (no ML imports)."""

import pytest

from whoa_llm.training.sft import (
    DatasetConfig,
    LoRAConfig,
    MemoryConfig,
    SFTConfig,
    TrainConfig,
)


def _base_cfg(**kwargs) -> SFTConfig:
    return SFTConfig(
        model_id="sshleifer/tiny-gpt2",
        dataset=DatasetConfig(name="tatsu-lab/alpaca", format="alpaca", max_samples=10),
        **kwargs,
    )


def test_defaults_are_lora():
    cfg = _base_cfg()
    assert cfg.method == "lora"
    assert cfg.precision == "bf16"
    assert cfg.engine == "auto"


def test_qlora_forces_4bit_quant_after_normalise():
    cfg = _base_cfg(method="qlora").normalised()
    assert cfg.memory.quantization == "4bit"


def test_full_swaps_paged_optimizer():
    cfg = _base_cfg(method="full", train=TrainConfig(optim="paged_adamw_8bit")).normalised()
    assert cfg.train.optim == "adamw_torch"


def test_full_keeps_user_supplied_optim():
    cfg = _base_cfg(method="full", train=TrainConfig(optim="adamw_bnb_8bit")).normalised()
    assert cfg.train.optim == "adamw_bnb_8bit"


def test_lora_keeps_default_quantization_none():
    cfg = _base_cfg(method="lora").normalised()
    assert cfg.memory.quantization == "none"


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        _base_cfg(method="rl")


def test_invalid_format_rejected():
    with pytest.raises(ValueError):
        SFTConfig(
            model_id="x",
            dataset=DatasetConfig(name="y", format="not-a-format"),  # type: ignore[arg-type]
        )


def test_round_trip_json():
    cfg = _base_cfg(method="qlora", lora=LoRAConfig(r=8, alpha=16))
    blob = cfg.model_dump_json()
    restored = SFTConfig.model_validate_json(blob)
    assert restored.lora.r == 8
    assert restored.lora.alpha == 16
    assert restored.method == "qlora"


def test_memory_config_defaults():
    m = MemoryConfig()
    assert m.quantization == "none"
    assert m.cpu_offload is False
    assert m.attn_impl is None
