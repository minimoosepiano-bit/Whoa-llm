"""Tests for the UI RunState background-run coordinator."""

import time
from types import SimpleNamespace

import pytest

from whoa_llm.ui.state import LogTail, RunState


def test_log_tail_writes_and_caps():
    tail = LogTail()
    tail.lines.clear()
    for i in range(600):
        tail.write(f"line {i}")
    text = tail.text()
    # Bounded at 500.
    assert text.count("\n") < 500
    assert "line 599" in text
    assert "line 0" not in text  # rolled off


def test_run_state_starts_idle():
    state = RunState()
    assert state.is_running is False
    assert state.last_summary is None
    assert state.last_error is None


def test_run_state_starts_thread_and_captures_summary(monkeypatch):
    state = RunState()
    seen: dict = {}

    def fake_run_sft(cfg, *, metrics_callback=None, extra_callbacks=None):
        seen["cfg"] = cfg
        seen["metrics_callback"] = metrics_callback
        seen["extra_callbacks"] = extra_callbacks
        return {"final_loss": 0.5, "steps": 10}

    monkeypatch.setattr("whoa_llm.training.sft.run_sft", fake_run_sft)

    cfg = SimpleNamespace(
        model_id="x", method="lora", output_dir="/tmp/whoa-test-run-state",
    )
    assert state.start(cfg) is True
    state._thread.join(timeout=2)

    assert state.last_summary == {"final_loss": 0.5, "steps": 10}
    assert state.last_error is None
    assert seen["cfg"] is cfg
    assert seen["metrics_callback"] is state.metrics
    # extra_callbacks should contain our cancel hook.
    assert seen["extra_callbacks"] and len(seen["extra_callbacks"]) == 1


def test_run_state_rejects_concurrent_start(monkeypatch):
    state = RunState()

    def slow_run_sft(cfg, *, metrics_callback=None, extra_callbacks=None):
        time.sleep(0.3)
        return {"ok": True}

    monkeypatch.setattr("whoa_llm.training.sft.run_sft", slow_run_sft)

    cfg = SimpleNamespace(
        model_id="x", method="lora", output_dir="/tmp/whoa-test-concurrent",
    )
    assert state.start(cfg) is True
    # Second start while first is running should be rejected.
    assert state.start(cfg) is False
    state._thread.join(timeout=2)
    # After it's done, a new run should be accepted.
    assert state.start(cfg) is True
    state._thread.join(timeout=2)


def test_run_state_records_errors(monkeypatch):
    state = RunState()

    def bad_run_sft(cfg, *, metrics_callback=None, extra_callbacks=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("whoa_llm.training.sft.run_sft", bad_run_sft)

    cfg = SimpleNamespace(
        model_id="x", method="lora", output_dir="/tmp/whoa-test-error",
    )
    state.start(cfg)
    state._thread.join(timeout=2)
    assert state.last_error is not None
    assert "boom" in state.last_error
    assert state.last_summary is None
    assert "Run failed" in state.log.text()


def test_cancel_sets_event(monkeypatch):
    state = RunState()
    monkeypatch.setattr(
        "whoa_llm.training.sft.run_sft",
        lambda cfg, *, metrics_callback=None, extra_callbacks=None: {"steps": 0},
    )
    cfg = SimpleNamespace(model_id="x", method="lora", output_dir="/tmp/x")
    state.start(cfg)
    state.cancel()
    assert "Cancellation requested" in state.log.text()
    state._thread.join(timeout=2)
