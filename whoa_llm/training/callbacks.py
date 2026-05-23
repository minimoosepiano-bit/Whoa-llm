"""Training callbacks for live metrics and graceful OOM recovery."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    step: int
    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    epoch: float | None = None
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
        }


try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:  # pragma: no cover — fallback used when transformers isn't installed
    class _TrainerCallback:  # type: ignore[no-redef]
        """Stand-in for ``transformers.TrainerCallback``.

        Provides no-op implementations of the hook methods the Trainer
        actually invokes, so unit tests can exercise the callback without
        a transformers install.
        """

        def on_init_end(self, *args, **kwargs): pass  # noqa: E704
        def on_train_begin(self, *args, **kwargs): pass  # noqa: E704
        def on_train_end(self, *args, **kwargs): pass  # noqa: E704
        def on_epoch_begin(self, *args, **kwargs): pass  # noqa: E704
        def on_epoch_end(self, *args, **kwargs): pass  # noqa: E704
        def on_step_begin(self, *args, **kwargs): pass  # noqa: E704
        def on_step_end(self, *args, **kwargs): pass  # noqa: E704
        def on_substep_end(self, *args, **kwargs): pass  # noqa: E704
        def on_evaluate(self, *args, **kwargs): pass  # noqa: E704
        def on_predict(self, *args, **kwargs): pass  # noqa: E704
        def on_save(self, *args, **kwargs): pass  # noqa: E704
        def on_log(self, *args, **kwargs): pass  # noqa: E704
        def on_prediction_step(self, *args, **kwargs): pass  # noqa: E704


class MetricsCallback(_TrainerCallback):
    """Thread-safe ring-buffer of training metrics, polled by the UI.

    Inherits from ``transformers.TrainerCallback`` when available so the
    Trainer's introspection (``on_init_end`` etc.) finds all hooks.
    """

    def __init__(self, max_points: int = 10_000):
        super().__init__()
        self._lock = threading.Lock()
        self._points: deque[MetricPoint] = deque(maxlen=max_points)
        self._latest: MetricPoint | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Read API (called by UI)
    # ------------------------------------------------------------------
    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of all collected points (cheap, list-of-dicts)."""
        with self._lock:
            return [p.as_dict() for p in self._points]

    @property
    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest.as_dict() if self._latest else None

    @property
    def is_running(self) -> bool:
        return self._running

    def reset(self) -> None:
        with self._lock:
            self._points.clear()
            self._latest = None
            self._running = False

    # ------------------------------------------------------------------
    # transformers.TrainerCallback hooks
    # ------------------------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        with self._lock:
            self._points.clear()
            self._latest = None
            self._running = True
        logger.info("Training started")

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        self._running = False
        logger.info("Training ended (final step=%d)", getattr(state, "global_step", -1))

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        if not logs:
            return
        point = MetricPoint(
            step=int(getattr(state, "global_step", 0)),
            loss=logs.get("loss") or logs.get("train_loss"),
            learning_rate=logs.get("learning_rate"),
            grad_norm=logs.get("grad_norm"),
            epoch=logs.get("epoch") or getattr(state, "epoch", None),
        )
        with self._lock:
            self._points.append(point)
            self._latest = point


class OOMRecoveryCallback:
    """On the first CUDA OOM, halve batch size and double grad-accum, then re-raise on the next.

    Wraps ``on_step_end`` to detect a marker set by an outer exception handler;
    practical OOM recovery requires the trainer to be re-built so we expose a
    ``handle_oom()`` helper that callers (``run_sft``) invoke from a try/except.
    """

    def __init__(self):
        self.recoveries = 0
        self.max_recoveries = 1

    def handle_oom(self, args: Any) -> bool:
        """Mutate ``args`` to lower memory pressure; return True if recovery was applied."""
        if self.recoveries >= self.max_recoveries:
            logger.error("OOM after %d recovery attempts — giving up.", self.recoveries)
            return False

        old_bs = getattr(args, "per_device_train_batch_size", 1)
        old_ga = getattr(args, "gradient_accumulation_steps", 1)

        new_bs = max(1, old_bs // 2)
        new_ga = old_ga * 2

        if new_bs == old_bs:
            # Already at batch size 1 — try gradient checkpointing as a last resort.
            args.gradient_checkpointing = True
            logger.warning("OOM recovery: batch size already 1; enabling gradient checkpointing.")
        else:
            args.per_device_train_batch_size = new_bs
            args.gradient_accumulation_steps = new_ga
            logger.warning(
                "OOM recovery #%d: batch_size %d→%d, grad_accum %d→%d",
                self.recoveries + 1, old_bs, new_bs, old_ga, new_ga,
            )

        self.recoveries += 1
        return True
