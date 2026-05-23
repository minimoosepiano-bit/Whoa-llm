"""Resolve reward functions from dotted paths or built-in names.

Accepts:
    * ``"length_reward"``                          — built-in by short name
    * ``"whoa_llm.training.rewards.builtin.length_reward"``  — full dotted path
    * ``"user_rewards.my_module:my_func"``         — file-relative for pasted code
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable


def _named_partial(func: Callable, /, **kwargs: Any) -> Callable:
    """Like ``functools.partial`` but preserves ``__name__`` (TRL needs it)."""
    if not kwargs:
        return func
    p = functools.partial(func, **kwargs)
    p.__name__ = getattr(func, "__name__", func.__class__.__name__)  # type: ignore[attr-defined]
    return p

logger = logging.getLogger(__name__)


def _load_from_user_dir(name: str) -> Callable[..., list[float]] | None:
    """Try ``~/.whoa_llm/user_rewards/<name>.py`` (used by UI-pasted rewards)."""
    user_dir = Path.home() / ".whoa_llm" / "user_rewards"
    target = user_dir / f"{name}.py"
    if not target.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"whoa_user_reward_{name}", target)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    func = getattr(module, name, None) or getattr(module, "reward", None)
    if not callable(func):
        logger.warning("User reward %s has no callable named %s or 'reward'.", target, name)
        return None
    return func


def resolve_reward(spec: str | dict | Callable, **default_kwargs: Any) -> Callable[..., list[float]]:
    """Resolve a single reward spec into a callable.

    ``spec`` can be:

    * a callable — returned as-is (with default kwargs bound)
    * a string — built-in short name OR dotted path OR ``"module:func"``
    * a dict ``{"name": "regex_format_reward", "kwargs": {"pattern": "..."}}``
    """
    if callable(spec):
        return _named_partial(spec, **default_kwargs) if default_kwargs else spec

    if isinstance(spec, dict):
        name = spec["name"]
        kwargs = {**default_kwargs, **spec.get("kwargs", {})}
        return resolve_reward(name, **kwargs)

    if not isinstance(spec, str):
        raise TypeError(f"reward spec must be str|dict|callable, got {type(spec).__name__}")

    # Built-in short name
    from .builtin import discover_builtins
    builtins = discover_builtins()
    if spec in builtins:
        func = builtins[spec]
        return _named_partial(func, **default_kwargs) if default_kwargs else func

    # "module:func" form
    if ":" in spec:
        mod_path, func_name = spec.split(":", 1)
    elif "." in spec:
        mod_path, func_name = spec.rsplit(".", 1)
    else:
        # Try user dir.
        func = _load_from_user_dir(spec)
        if func is None:
            raise ValueError(
                f"Unknown reward {spec!r}. Built-ins: {list(builtins)}. "
                f"Or pass a dotted path like 'pkg.mod.func'."
            )
        return _named_partial(func, **default_kwargs) if default_kwargs else func

    try:
        module = importlib.import_module(mod_path)
    except ImportError as exc:
        raise ValueError(f"Could not import {mod_path!r} for reward {spec!r}: {exc}") from exc
    func = getattr(module, func_name, None)
    if func is None:
        raise ValueError(f"{mod_path!r} has no attribute {func_name!r}")
    return _named_partial(func, **default_kwargs) if default_kwargs else func


def resolve_rewards(
    specs: list[str | dict | Callable], **default_kwargs: Any,
) -> list[Callable[..., list[float]]]:
    """Resolve a list of reward specs."""
    return [resolve_reward(s, **default_kwargs) for s in specs]
