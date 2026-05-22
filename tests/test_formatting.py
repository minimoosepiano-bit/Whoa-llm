"""Tests for dataset formatting templates (no network, no torch)."""

import pytest
from unittest.mock import MagicMock

from whoa_llm.data.formatting import (
    SUPPORTED_FORMATS,
    get_formatter,
    make_formatting_func,
)


@pytest.fixture()
def dummy_tokenizer():
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.eos_token = "</s>"
    tok.apply_chat_template = MagicMock(
        side_effect=lambda msgs, tokenize=False, add_generation_prompt=False: (
            "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        )
    )
    return tok


def test_supported_formats_list():
    assert set(SUPPORTED_FORMATS) == {"alpaca", "sharegpt", "completion", "raw_text", "chat_template"}


def test_get_formatter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        get_formatter("nonexistent")


class TestAlpaca:
    def test_with_input(self, dummy_tokenizer):
        example = {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
        result = get_formatter("alpaca")(example, dummy_tokenizer)
        assert "Translate" in result
        assert "Hello" in result
        assert "Bonjour" in result
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result

    def test_without_input(self, dummy_tokenizer):
        example = {"instruction": "Write a poem", "input": "", "output": "Roses are red"}
        result = get_formatter("alpaca")(example, dummy_tokenizer)
        assert "### Input:" not in result
        assert "Write a poem" in result
        assert "Roses are red" in result


class TestCompletion:
    def test_basic(self, dummy_tokenizer):
        example = {"prompt": "Hello, ", "completion": "world!"}
        result = get_formatter("completion")(example, dummy_tokenizer)
        assert result == "Hello, world!"

    def test_response_fallback(self, dummy_tokenizer):
        example = {"prompt": "Q: ", "response": "A."}
        result = get_formatter("completion")(example, dummy_tokenizer)
        assert result == "Q: A."


class TestRawText:
    def test_passthrough(self, dummy_tokenizer):
        example = {"text": "The quick brown fox"}
        assert get_formatter("raw_text")(example, dummy_tokenizer) == "The quick brown fox"


class TestShareGPT:
    def test_uses_chat_template_when_available(self, dummy_tokenizer):
        example = {"conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]}
        result = get_formatter("sharegpt")(example, dummy_tokenizer)
        assert "Hi" in result
        assert "Hello!" in result

    def test_fallback_when_no_template(self):
        tok = MagicMock(spec=[])  # no apply_chat_template attr
        example = {"conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]}
        result = get_formatter("sharegpt")(example, tok)
        assert "Hi" in result
        assert "Hello!" in result


class TestChatTemplate:
    def test_applies_tokenizer_template(self, dummy_tokenizer):
        example = {"messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]}
        result = get_formatter("chat_template")(example, dummy_tokenizer)
        assert "What is 2+2?" in result
        assert "4" in result

    def test_raises_without_apply_chat_template(self):
        tok = MagicMock(spec=["pad_token"])
        example = {"messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(ValueError, match="apply_chat_template"):
            get_formatter("chat_template")(example, tok)


class TestMakeFormattingFunc:
    def test_batch_formatting(self, dummy_tokenizer):
        fn = make_formatting_func("raw_text", dummy_tokenizer)
        batch = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        result = fn(batch)
        assert result == ["a", "b", "c"]

    def test_single_example_formatting(self, dummy_tokenizer):
        """TRL >=0.11 passes a single dict, not a batch."""
        fn = make_formatting_func("raw_text", dummy_tokenizer)
        result = fn({"text": "hello"})
        assert result == "hello"

    def test_column_map(self, dummy_tokenizer):
        fn = make_formatting_func("completion", dummy_tokenizer, column_map={"prompt_text": "prompt"})
        batch = [{"prompt_text": "Hello ", "completion": "world"}]
        result = fn(batch)
        assert result == ["Hello world"]
