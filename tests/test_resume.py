"""Tests for checkpoint discovery and resume helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from whoa_llm.training.resume import find_latest_checkpoint, resume_run


class TestFindLatestCheckpoint:
    def test_returns_highest_numbered(self, tmp_path):
        for n in [10, 100, 50]:
            (tmp_path / f"checkpoint-{n}").mkdir()
        ckpt = find_latest_checkpoint(tmp_path)
        assert ckpt is not None
        assert ckpt.name == "checkpoint-100"

    def test_none_when_empty(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_none_when_dir_missing(self, tmp_path):
        assert find_latest_checkpoint(tmp_path / "does-not-exist") is None

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        (tmp_path / "checkpoint-5").mkdir()
        (tmp_path / "random_dir").mkdir()
        (tmp_path / "checkpoint-foo").mkdir()
        ckpt = find_latest_checkpoint(tmp_path)
        assert ckpt.name == "checkpoint-5"

    def test_ignores_files_named_checkpoint(self, tmp_path):
        (tmp_path / "checkpoint-5").write_text("hello")  # file, not dir
        assert find_latest_checkpoint(tmp_path) is None


class TestResumeRun:
    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            resume_run(tmp_path)

    def test_missing_checkpoint_raises(self, tmp_path):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "type": "sft",
            "model_id": "x",
            "dataset": {"name": "d", "format": "alpaca"},
        }))
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            resume_run(tmp_path)

    def test_dispatches_to_sft(self, tmp_path):
        (tmp_path / "checkpoint-3").mkdir()
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "type": "sft",
            "model_id": "x",
            "dataset": {"name": "d", "format": "alpaca"},
            "output_dir": str(tmp_path),
        }))
        captured = {}
        def fake_run_sft(cfg, *, metrics_callback=None, extra_callbacks=None,
                         resume_from_checkpoint=None):
            captured["resume"] = resume_from_checkpoint
            captured["model_id"] = cfg.model_id
            return {"steps": 1}
        with patch("whoa_llm.training.sft.run_sft", side_effect=fake_run_sft):
            summary = resume_run(tmp_path)
        assert summary == {"steps": 1}
        assert captured["model_id"] == "x"
        assert "checkpoint-3" in captured["resume"]

    def test_dispatches_to_grpo(self, tmp_path):
        (tmp_path / "checkpoint-7").mkdir()
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "type": "grpo",
            "model_id": "x",
            "dataset": {"name": "d", "format": "raw_text"},
            "output_dir": str(tmp_path),
            "rewards": [{"name": "length_reward"}],
        }))
        captured = {}
        def fake_run_grpo(cfg, *, metrics_callback=None, extra_callbacks=None,
                          reward_funcs_override=None, resume_from_checkpoint=None):
            captured["resume"] = resume_from_checkpoint
            return {"steps": 1}
        with patch("whoa_llm.training.grpo.run_grpo", side_effect=fake_run_grpo):
            summary = resume_run(tmp_path)
        assert summary == {"steps": 1}
        assert "checkpoint-7" in captured["resume"]

    def test_unknown_kind_raises(self, tmp_path):
        (tmp_path / "checkpoint-1").mkdir()
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "type": "weird",
            "model_id": "x",
            "dataset": {"name": "d", "format": "alpaca"},
        }))
        with pytest.raises(ValueError, match="Unknown training type"):
            resume_run(tmp_path)

    def test_defaults_to_sft_when_no_type(self, tmp_path):
        (tmp_path / "checkpoint-1").mkdir()
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "model_id": "x",
            "dataset": {"name": "d", "format": "alpaca"},
            "output_dir": str(tmp_path),
        }))
        with patch("whoa_llm.training.sft.run_sft", return_value={"ok": 1}) as m:
            resume_run(tmp_path)
        assert m.called
