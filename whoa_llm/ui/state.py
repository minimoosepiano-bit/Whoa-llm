"""Background training run state — shared across UI tabs.

A single ``RunState`` instance holds the currently configured ``SFTConfig``
draft, the live ``MetricsCallback``, the worker thread (when running) and
synchronisation primitives so only one run executes at a time.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


class RunState:
    """Coordinates a single background training run."""

    def __init__(self) -> None:
        from whoa_llm.training.callbacks import MetricsCallback

        self.metrics = MetricsCallback()
        self.log = LogTail()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._start_lock = threading.Lock()
        self.last_summary: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_config: Any = None  # SFTConfig — typed via duck-typing.
        self.last_output_dir: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: Any) -> bool:
        """Spawn a worker that calls ``run_sft(cfg)``; return True on success."""
        with self._start_lock:
            if self.is_running:
                return False
            self.metrics.reset()
            self._cancel.clear()
            self.last_summary = None
            self.last_error = None
            self.last_config = cfg
            self.last_output_dir = Path(cfg.output_dir)
            self.log.write(f"Starting {cfg.method} run for {cfg.model_id}")
            self._thread = threading.Thread(
                target=self._worker, args=(cfg,), name="whoa-llm-train", daemon=True
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
    def _worker(self, cfg: Any) -> None:
        from whoa_llm.training.sft import run_sft

        try:
            # Inject a cancel-aware callback alongside the metrics one.
            cancel_cb = _CancelCallback(self._cancel)
            summary = run_sft(
                cfg, metrics_callback=self.metrics, extra_callbacks=[cancel_cb],
            )
            self.last_summary = summary
            self.log.write(f"Run finished: {summary}")
        except Exception as exc:  # noqa: BLE001 — surface anything to the UI
            self.last_error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            self.log.write(f"Run failed: {self.last_error}")
            logger.exception("Training thread crashed")
            for line in tb.splitlines()[-8:]:
                self.log.write(line)


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
