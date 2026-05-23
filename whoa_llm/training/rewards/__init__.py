"""Built-in reward functions for GRPO.

Each reward function has the signature::

    func(prompts, completions, **kwargs) -> list[float]

where ``prompts`` and ``completions`` are lists of strings of equal length
(one per group sample) and the returned list contains the reward (higher
is better) for each completion.

Custom rewards live in ``whoa_llm.training.rewards.<name>`` (built-in) or
``~/.whoa_llm/user_rewards/<name>.py`` (user-pasted, see Phase 7 UI).
"""

from . import builtin
from .builtin import (
    contains_reward,
    discover_builtins,
    length_reward,
    regex_format_reward,
)
from .loader import resolve_reward, resolve_rewards

__all__ = [
    "builtin",
    "length_reward",
    "regex_format_reward",
    "contains_reward",
    "discover_builtins",
    "resolve_reward",
    "resolve_rewards",
]
