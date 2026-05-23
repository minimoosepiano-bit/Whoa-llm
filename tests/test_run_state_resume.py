"""Tests for RunState.resume() wiring."""

from unittest.mock import patch

import yaml

from whoa_llm.ui.state import RunState


def _seed_run_dir(path):
    (path / "checkpoint-2").mkdir()
    (path / "config.yaml").write_text(yaml.safe_dump({
        "type": "sft",
        "model_id": "x",
        "dataset": {"name": "d", "format": "alpaca"},
        "output_dir": str(path),
    }))


def test_resume_with_no_config_yaml(tmp_path):
    state = RunState()
    ok, msg = state.resume(tmp_path)
    assert ok is False
    assert "config.yaml" in msg


def test_resume_with_no_checkpoint(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "type": "sft", "model_id": "x",
        "dataset": {"name": "d", "format": "alpaca"},
    }))
    state = RunState()
    ok, msg = state.resume(tmp_path)
    assert ok is False
    assert "checkpoint" in msg


def test_resume_spawns_thread(tmp_path):
    _seed_run_dir(tmp_path)
    state = RunState()
    captured = {}
    def fake_resume_run(out_dir, *, metrics_callback=None, extra_callbacks=None):
        captured["out_dir"] = out_dir
        captured["metrics_callback"] = metrics_callback
        captured["extra_callbacks"] = extra_callbacks
        return {"steps": 5}
    with patch("whoa_llm.training.resume.resume_run", side_effect=fake_resume_run):
        ok, msg = state.resume(tmp_path)
        assert ok is True
        state._thread.join(timeout=3)
    assert captured["out_dir"] == tmp_path
    assert captured["metrics_callback"] is state.metrics
    assert state.last_summary == {"steps": 5}
    assert state.kind == "resume"


def test_resume_refuses_when_running(tmp_path):
    _seed_run_dir(tmp_path)
    state = RunState()
    import time
    def slow(out_dir, *, metrics_callback=None, extra_callbacks=None):
        time.sleep(0.3)
        return {"steps": 0}
    with patch("whoa_llm.training.resume.resume_run", side_effect=slow):
        state.resume(tmp_path)
        ok, msg = state.resume(tmp_path)
    assert ok is False
    assert "in progress" in msg
    state._thread.join(timeout=2)
