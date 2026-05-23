"""Resume a training run from a saved ``config.yaml`` + checkpoint dir."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-N`` subdirectory, or None."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        m = _CHECKPOINT_RE.match(child.name)
        if m:
            candidates.append((int(m.group(1)), child))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resume_run(
    output_dir: str | Path,
    *,
    metrics_callback: Any | None = None,
    extra_callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Resume the run saved in *output_dir*.

    Reads ``config.yaml`` (a copy is written by ``run_sft`` / ``run_grpo``)
    to recover the run type (``sft`` | ``grpo``) and the full config, then
    calls the appropriate trainer with ``resume_from_checkpoint`` pointing
    at the latest ``checkpoint-N`` subdirectory.

    Returns the same dict shape as ``run_sft`` / ``run_grpo``.
    """
    import yaml

    output_dir = Path(output_dir)
    cfg_path = output_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.yaml in {output_dir}; can't resume. "
            "Was this directory produced by whoa-llm?"
        )

    data = yaml.safe_load(cfg_path.read_text())
    kind = (data.pop("type", None) or "sft").lower()
    ckpt = find_latest_checkpoint(output_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint-* subdirs in {output_dir}; nothing to resume from."
        )
    logger.info("Resuming %s run from %s", kind, ckpt)

    if kind == "sft":
        from whoa_llm.training.sft import SFTConfig, run_sft

        cfg = SFTConfig.model_validate(data)
        return run_sft(
            cfg,
            metrics_callback=metrics_callback,
            extra_callbacks=extra_callbacks,
            resume_from_checkpoint=str(ckpt),
        )
    if kind == "grpo":
        from whoa_llm.training.grpo import GRPOConfig, run_grpo

        cfg = GRPOConfig.model_validate(data)
        return run_grpo(
            cfg,
            metrics_callback=metrics_callback,
            extra_callbacks=extra_callbacks,
            resume_from_checkpoint=str(ckpt),
        )
    raise ValueError(f"Unknown training type {kind!r}; expected 'sft' or 'grpo'.")
