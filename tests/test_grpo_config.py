"""Tests for GRPOConfig validation and normalisation (no ML imports)."""

import pytest

from whoa_llm.training.grpo import GRPOConfig, RewardSpec
from whoa_llm.training.sft import DatasetConfig, TrainConfig


def _base(**kwargs) -> GRPOConfig:
    kwargs.setdefault("rewards", [RewardSpec(name="length_reward")])
    return GRPOConfig(
        model_id="x",
        dataset=DatasetConfig(name="d", format="raw_text"),
        **kwargs,
    )


def test_defaults():
    cfg = _base()
    assert cfg.method == "lora"
    assert cfg.num_generations == 4
    assert cfg.beta == 0.04
    assert cfg.use_vllm is False


def test_qlora_normalises_to_4bit():
    cfg = _base(method="qlora").normalised()
    assert cfg.memory.quantization == "4bit"


def test_full_swaps_paged_optimizer():
    cfg = _base(method="full", train=TrainConfig(optim="paged_adamw_8bit")).normalised()
    assert cfg.train.optim == "adamw_torch"


def test_reward_spec_round_trip():
    cfg = _base(rewards=[
        RewardSpec(name="regex_format_reward", kwargs={"pattern": r"\d+"}, weight=2.0),
        RewardSpec(name="length_reward", kwargs={"target_tokens": 128}),
    ])
    blob = cfg.model_dump_json()
    restored = GRPOConfig.model_validate_json(blob)
    assert len(restored.rewards) == 2
    assert restored.rewards[0].kwargs == {"pattern": r"\d+"}
    assert restored.rewards[0].weight == 2.0


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        _base(method="rl")


def test_empty_rewards_allowed_at_config_level():
    cfg = _base(rewards=[])
    assert cfg.rewards == []
