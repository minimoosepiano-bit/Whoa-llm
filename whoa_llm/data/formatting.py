"""Dataset formatting utilities.

Each formatter is a pure function with signature::

    formatter(example: dict, tokenizer) -> str

The returned string is a *text* representation of the training example
(pre-tokenisation).  ``SFTTrainer`` can then tokenise it via
``dataset_text_field`` or a ``formatting_func``.

Supported templates:
- ``chat_template`` – uses ``tokenizer.apply_chat_template`` (best for
  instruction-tuned base models with a built-in template).
- ``alpaca`` – classic instruction / input / output format.
- ``sharegpt`` – list of ``{"role": ..., "content": ...}`` turns.
- ``completion`` – raw ``prompt`` + ``completion`` concatenation.
- ``raw_text`` – uses a single ``text`` column as-is.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for a formatting function.
Formatter = Callable[[dict[str, Any], Any], str]

_ALPACA_PROMPT = (
    "Below is an instruction that describes a task"
    "{input_section}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{output}"
)


def _alpaca(example: dict[str, Any], tokenizer: Any) -> str:  # noqa: ARG001
    instruction = example.get("instruction", "")
    inp = example.get("input", "").strip()
    output = example.get("output", "")

    if inp:
        input_section = ", paired with an input that provides further context"
        input_block = f"### Input:\n{inp}\n\n"
    else:
        input_section = ""
        input_block = ""

    return _ALPACA_PROMPT.format(
        input_section=input_section,
        instruction=instruction,
        input_block=input_block,
        output=output,
    )


def _sharegpt(example: dict[str, Any], tokenizer: Any) -> str:
    """Format ShareGPT-style conversations (list of role/content dicts)."""
    conversations = example.get("conversations", example.get("messages", []))
    if not conversations:
        return ""

    # Try the tokenizer's built-in chat template first.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # ShareGPT uses "human"/"gpt"; remap to "user"/"assistant".
            mapped = []
            for turn in conversations:
                role = turn.get("role") or turn.get("from", "")
                content = turn.get("content") or turn.get("value", "")
                if role in {"human", "user"}:
                    role = "user"
                elif role in {"gpt", "assistant"}:
                    role = "assistant"
                mapped.append({"role": role, "content": content})
            return tokenizer.apply_chat_template(
                mapped, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass  # fall through to manual formatting

    lines = []
    for turn in conversations:
        role = (turn.get("role") or turn.get("from", "")).capitalize()
        content = turn.get("content") or turn.get("value", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _completion(example: dict[str, Any], tokenizer: Any) -> str:  # noqa: ARG001
    prompt = example.get("prompt", "")
    completion = example.get("completion", example.get("response", ""))
    return prompt + completion


def _raw_text(example: dict[str, Any], tokenizer: Any) -> str:  # noqa: ARG001
    return example.get("text", "")


def _chat_template(example: dict[str, Any], tokenizer: Any) -> str:
    """Use the tokenizer's own chat template on a ``messages`` column."""
    messages = example.get("messages", example.get("conversations", []))
    if not messages:
        logger.warning("chat_template format: no 'messages' key found in example.")
        return ""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "The selected tokenizer does not have apply_chat_template; "
            "choose a different format or a chat-capable model."
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


_FORMATTERS: dict[str, Formatter] = {
    "alpaca": _alpaca,
    "sharegpt": _sharegpt,
    "completion": _completion,
    "raw_text": _raw_text,
    "chat_template": _chat_template,
}

SUPPORTED_FORMATS = list(_FORMATTERS.keys())


def get_formatter(name: str) -> Formatter:
    """Return the formatting function for the given template name.

    Parameters
    ----------
    name:
        One of ``"alpaca"``, ``"sharegpt"``, ``"completion"``,
        ``"raw_text"``, ``"chat_template"``.

    Raises
    ------
    ValueError
        If *name* is not a known template.
    """
    if name not in _FORMATTERS:
        raise ValueError(
            f"Unknown format {name!r}. Choose from: {SUPPORTED_FORMATS}"
        )
    return _FORMATTERS[name]


def make_formatting_func(
    template: str,
    tokenizer: Any,
    column_map: dict[str, str] | None = None,
) -> Callable[[Any], Any]:
    """Return a ``formatting_func`` compatible with ``trl.SFTTrainer``.

    The returned callable handles both calling conventions:

    * single example (dict) → returns ``str`` (TRL >=0.11)
    * batch (list of dicts) → returns ``list[str]`` (older TRL)

    Parameters
    ----------
    template:
        Format name (see :func:`get_formatter`).
    tokenizer:
        The model tokenizer (needed for chat-template formats).
    column_map:
        Optional remapping of column names, e.g.
        ``{"instruction": "prompt", "output": "response"}``.
    """
    formatter = get_formatter(template)

    def _apply(ex: dict[str, Any]) -> str:
        if column_map:
            ex = {column_map.get(k, k): v for k, v in ex.items()}
        return formatter(ex, tokenizer)

    def _func(examples: Any) -> Any:
        # Batch mode: a plain list of examples → list of strings.
        if isinstance(examples, list):
            return [_apply(ex) for ex in examples]
        # Single-example mode: dict or dict-like (e.g. ``datasets.LazyRow``).
        return _apply(examples)

    return _func
