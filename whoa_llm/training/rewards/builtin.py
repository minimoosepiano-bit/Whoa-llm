"""Built-in reward functions shipped with whoa-llm."""

from __future__ import annotations

import re
from typing import Any


def length_reward(
    prompts: list[Any], completions: list[Any], *,
    target_tokens: int = 256, **kwargs: Any,
) -> list[float]:
    """Reward completions whose length (in whitespace tokens) is near *target_tokens*.

    Returns 1.0 at the target, falling off linearly to 0.0 at distance == target.
    Negative rewards are clipped to 0.

    Example::

        rewards = length_reward(prompts, completions, target_tokens=128)
    """
    rewards: list[float] = []
    for c in completions:
        text = _to_text(c)
        n = len(text.split())
        distance = abs(n - target_tokens)
        reward = max(0.0, 1.0 - distance / max(1, target_tokens))
        rewards.append(float(reward))
    return rewards


def regex_format_reward(
    prompts: list[Any], completions: list[Any], *,
    pattern: str = r".*", flags: int = 0, **kwargs: Any,
) -> list[float]:
    """1.0 when the completion matches *pattern* (re.search), else 0.0.

    Useful for enforcing structured outputs like JSON or ``<answer>…</answer>``.
    """
    rx = re.compile(pattern, flags)
    return [1.0 if rx.search(_to_text(c)) else 0.0 for c in completions]


def contains_reward(
    prompts: list[Any], completions: list[Any], *,
    keywords: list[str] | None = None, **kwargs: Any,
) -> list[float]:
    """Reward = fraction of *keywords* that appear (case-insensitive) in each completion."""
    if not keywords:
        return [0.0] * len(completions)
    kws = [k.lower() for k in keywords]
    rewards: list[float] = []
    for c in completions:
        text = _to_text(c).lower()
        hits = sum(1 for k in kws if k in text)
        rewards.append(hits / len(kws))
    return rewards


def _to_text(item: Any) -> str:
    """GRPO sometimes hands completions in as chat-message lists; flatten to a string."""
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        parts: list[str] = []
        for elem in item:
            if isinstance(elem, dict):
                parts.append(str(elem.get("content", "")))
            else:
                parts.append(str(elem))
        return "\n".join(parts)
    if isinstance(item, dict):
        return str(item.get("content", item))
    return str(item)


def discover_builtins() -> dict[str, Any]:
    """Return ``{name: callable}`` for every built-in reward function."""
    return {
        "length_reward": length_reward,
        "regex_format_reward": regex_format_reward,
        "contains_reward": contains_reward,
    }
