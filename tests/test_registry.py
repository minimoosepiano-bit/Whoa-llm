"""Tests for the engine registry (no ML downloads needed)."""

import pytest
from unittest.mock import patch

from whoa_llm.engines.registry import pick_engine


def test_llama_picks_unsloth_when_available():
    with patch("whoa_llm.engines.registry._unsloth_available", return_value=True):
        assert pick_engine("meta-llama/Llama-3.2-1B") == "unsloth"


def test_llama_falls_back_to_hf_without_unsloth():
    with patch("whoa_llm.engines.registry._unsloth_available", return_value=False):
        assert pick_engine("meta-llama/Llama-3.2-1B") == "hf"


def test_gpt2_always_hf():
    with patch("whoa_llm.engines.registry._unsloth_available", return_value=True):
        assert pick_engine("gpt2") == "hf"


def test_force_hf():
    with patch("whoa_llm.engines.registry._unsloth_available", return_value=True):
        assert pick_engine("meta-llama/Llama-3.2-1B", force="hf") == "hf"


def test_force_unsloth_without_package_warns_and_falls_back():
    with patch("whoa_llm.engines.registry._unsloth_available", return_value=False):
        result = pick_engine("meta-llama/Llama-3.2-1B", force="unsloth")
        assert result == "hf"


def test_force_invalid_raises():
    with pytest.raises(ValueError, match="force must be"):
        pick_engine("gpt2", force="invalid")


@pytest.mark.parametrize("model_id,expected_when_unsloth", [
    ("mistralai/Mistral-7B-v0.1", "unsloth"),
    ("Qwen/Qwen2-0.5B", "unsloth"),
    ("google/gemma-2b", "unsloth"),
    ("microsoft/phi-2", "unsloth"),
    ("TinyLlama/TinyLlama-1.1B", "unsloth"),
    ("openai-community/gpt2", "hf"),
    ("facebook/opt-125m", "hf"),
])
def test_family_routing(model_id, expected_when_unsloth):
    with patch("whoa_llm.engines.registry._unsloth_available", return_value=True):
        assert pick_engine(model_id) == expected_when_unsloth
