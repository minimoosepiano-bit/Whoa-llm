"""Tests for estimate_param_count (mocks transformers.AutoConfig)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from whoa_llm.hardware import estimate_param_count


@pytest.fixture()
def fake_autoconfig():
    """Patch transformers.AutoConfig.from_pretrained to return our SimpleNamespace."""
    def _patcher(config):
        return patch("transformers.AutoConfig.from_pretrained", return_value=config)
    return _patcher


def test_llama_like_config(fake_autoconfig):
    # ~7B Llama-style config.
    cfg = SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=32000,
        intermediate_size=11008,
    )
    with fake_autoconfig(cfg):
        size = estimate_param_count("fake/llama-7b")
    # Should land within 20% of 7B.
    assert 5.5 < size < 8.5, f"got {size}B"


def test_tiny_config(fake_autoconfig):
    cfg = SimpleNamespace(
        hidden_size=32, num_hidden_layers=2, vocab_size=1000, intermediate_size=64,
    )
    with fake_autoconfig(cfg):
        size = estimate_param_count("fake/tiny")
    assert size is not None
    assert size < 0.01  # well under 10M params


def test_unrecognised_config_returns_none(fake_autoconfig):
    cfg = SimpleNamespace()  # no hidden_size / n_embd / vocab
    with fake_autoconfig(cfg):
        size = estimate_param_count("fake/weird")
    assert size is None


def test_failure_returns_none():
    with patch(
        "transformers.AutoConfig.from_pretrained",
        side_effect=OSError("offline"),
    ):
        assert estimate_param_count("does/not/exist") is None


def test_uses_num_parameters_when_available(fake_autoconfig):
    cfg = SimpleNamespace(num_parameters=7_500_000_000)
    with fake_autoconfig(cfg):
        size = estimate_param_count("fake/declared")
    assert size == pytest.approx(7.5)


def test_gpt2_naming(fake_autoconfig):
    # GPT-2 uses n_embd / n_layer instead of hidden_size / num_hidden_layers.
    cfg = SimpleNamespace(n_embd=768, n_layer=12, vocab_size=50257)
    with fake_autoconfig(cfg):
        size = estimate_param_count("fake/gpt2")
    assert size is not None
    # GPT-2 small is ~125M, our estimate uses gated-MLP assumption so it'll
    # overshoot a bit, but it should be within an order of magnitude.
    assert 0.05 < size < 0.5
