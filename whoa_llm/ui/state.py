"""Background training run state — shared across UI tabs.

A single ``RunState`` instance holds the live ``MetricsCallback``, a ring
buffer of the most recent ``(prompt, completion, reward)`` triples seen
during a GRPO run, the worker thread, and synchronisation primitives so
only one run executes at a time.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class LogTail:
    """Bounded ring buffer of log lines surfaced in the UI."""
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.lines.append(f"[{ts}] {line}")

    def text(self) -> str:
        with self.lock:
            return "\n".join(self.lines)


@dataclass
class RewardSample:
    step: int
    prompt: str
    completion: str
    reward: float


class RewardSampleBuffer:
    """Thread-safe ring buffer of the last *N* (prompt, completion, reward) triples."""

    def __init__(self, max_samples: int = 20):
        self._samples: deque[RewardSample] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._step = 0

    def add(self, prompt: str, completion: str, reward: float) -> None:
        with self._lock:
            self._samples.append(
                RewardSample(step=self._step, prompt=prompt, completion=completion, reward=reward)
            )

    def bump_step(self, step: int) -> None:
        self._step = step

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._step = 0

    def snapshot(self) -> list[RewardSample]:
        with self._lock:
            return list(self._samples)

    def as_table(self) -> list[list[Any]]:
        return [
            [s.step, s.prompt[:120], s.completion[:160], round(s.reward, 3)]
            for s in self.snapshot()
        ]


def wrap_reward_for_capture(
    fn: Callable, buffer: RewardSampleBuffer, *, max_per_step: int = 4
) -> Callable:
    """Wrap a reward function so each call also pushes triples into *buffer*.

    ``__name__`` is preserved because TRL's GRPOTrainer reads it.
    """
    def _wrapped(prompts, completions, **kwargs):
        rewards = fn(prompts, completions, **kwargs)
        try:
            for p, c, r in list(zip(prompts, completions, rewards))[:max_per_step]:
                buffer.add(_flatten(p), _flatten(c), float(r))
        except Exception:  # noqa: BLE001 — never let UI capture break training
            logger.exception("Reward capture failed")
        return rewards

    _wrapped.__name__ = getattr(fn, "__name__", "wrapped_reward")
    return _wrapped


def _flatten(item: Any) -> str:
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


class RunState:
    """Coordinates a single background training run (SFT or GRPO)."""

    def __init__(self) -> None:
        from whoa_llm.training.callbacks import MetricsCallback

        self.metrics = MetricsCallback()
        self.log = LogTail()
        self.reward_samples = RewardSampleBuffer()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._start_lock = threading.Lock()
        self.last_summary: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_config: Any = None
        self.last_output_dir: Path | None = None
        self.kind: str | None = None  # "sft" | "grpo"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: Any, *, kind: str = "sft") -> bool:
        """Spawn a worker that calls ``run_sft`` (kind='sft') or ``run_grpo`` (kind='grpo')."""
        with self._start_lock:
            if self.is_running:
                return False
            self.metrics.reset()
            self.reward_samples.reset()
            self._cancel.clear()
            self.last_summary = None
            self.last_error = None
            self.last_config = cfg
            self.last_output_dir = Path(cfg.output_dir)
            self.kind = kind
            self.log.write(f"Starting {kind.upper()} {cfg.method} run for {cfg.model_id}")
            self._thread = threading.Thread(
                target=self._worker, args=(cfg, kind),
                name=f"whoa-llm-{kind}", daemon=True,
            )
            self._thread.start()
            return True

    def cancel(self) -> None:
        """Request the worker to stop. The trainer checks via callback."""
        self._cancel.set()
        self.log.write("Cancellation requested — will stop after current step.")

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self, cfg: Any, kind: str) -> None:
        cancel_cb = _CancelCallback(self._cancel)
        try:
            if kind == "sft":
                from whoa_llm.training.sft import run_sft
                summary = run_sft(
                    cfg, metrics_callback=self.metrics, extra_callbacks=[cancel_cb],
                )
            elif kind == "grpo":
                from whoa_llm.training.grpo import run_grpo
                from whoa_llm.training.rewards.loader import resolve_reward

                # Resolve + wrap rewards so the UI can show (prompt, completion, reward) triples.
                wrapped: list = []
                for spec in cfg.rewards:
                    fn = resolve_reward(spec.name, **spec.kwargs)
                    wrapped.append(wrap_reward_for_capture(fn, self.reward_samples))
                step_cb = _StepTrackingCallback(self.reward_samples)
                summary = run_grpo(
                    cfg,
                    metrics_callback=self.metrics,
                    extra_callbacks=[cancel_cb, step_cb],
                    reward_funcs_override=wrapped,
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}; expected 'sft' or 'grpo'.")
            self.last_summary = summary
            self.log.write(f"Run finished: {summary}")
        except Exception as exc:  # noqa: BLE001 — surface anything to the UI
            self.last_error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            self.log.write(f"Run failed: {self.last_error}")
            logger.exception("Training thread crashed")
            for line in tb.splitlines()[-8:]:
                self.log.write(line)


class _StepTrackingCallback:
    """Bump the reward-sample buffer's step counter on each Trainer step."""

    def __init__(self, buffer: RewardSampleBuffer):
        self._buffer = buffer

    def on_init_end(self, *a, **k): pass  # noqa: E704
    def on_train_begin(self, *a, **k): pass  # noqa: E704
    def on_train_end(self, *a, **k): pass  # noqa: E704
    def on_epoch_begin(self, *a, **k): pass  # noqa: E704
    def on_epoch_end(self, *a, **k): pass  # noqa: E704
    def on_substep_end(self, *a, **k): pass  # noqa: E704
    def on_evaluate(self, *a, **k): pass  # noqa: E704
    def on_predict(self, *a, **k): pass  # noqa: E704
    def on_save(self, *a, **k): pass  # noqa: E704
    def on_log(self, *a, **k): pass  # noqa: E704
    def on_prediction_step(self, *a, **k): pass  # noqa: E704
    def on_step_end(self, *a, **k): pass  # noqa: E704

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if state is not None:
            self._buffer.bump_step(int(getattr(state, "global_step", 0)))


class _CancelCallback:
    """TrainerCallback that flips ``control.should_training_stop`` on cancel."""

    def __init__(self, cancel_event: threading.Event):
        self._event = cancel_event

    # Match transformers.TrainerCallback's hook surface (no-ops elsewhere).
    def on_init_end(self, *a, **k): pass  # noqa: E704
    def on_train_begin(self, *a, **k): pass  # noqa: E704
    def on_train_end(self, *a, **k): pass  # noqa: E704
    def on_epoch_begin(self, *a, **k): pass  # noqa: E704
    def on_epoch_end(self, *a, **k): pass  # noqa: E704
    def on_substep_end(self, *a, **k): pass  # noqa: E704
    def on_evaluate(self, *a, **k): pass  # noqa: E704
    def on_predict(self, *a, **k): pass  # noqa: E704
    def on_save(self, *a, **k): pass  # noqa: E704
    def on_log(self, *a, **k): pass  # noqa: E704
    def on_prediction_step(self, *a, **k): pass  # noqa: E704
    def on_step_begin(self, *a, **k): pass  # noqa: E704

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if self._event.is_set() and control is not None:
            control.should_training_stop = True
        return control
