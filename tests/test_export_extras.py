"""Tests for GGUF helper + model-card-from-config."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from whoa_llm.ui.export import (
    card_from_config,
    convert_to_gguf,
    find_gguf_converter,
    write_model_card,
)


class TestFindGgufConverter:
    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        # Patch home() / Path("/opt/...") so they look empty.
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert find_gguf_converter() is None

    def test_finds_via_which(self, monkeypatch):
        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/local/bin/convert_hf_to_gguf.py"
            if name == "convert_hf_to_gguf.py" else None,
        )
        assert find_gguf_converter() == "/usr/local/bin/convert_hf_to_gguf.py"


class TestConvertToGguf:
    def test_missing_converter_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("whoa_llm.ui.export.find_gguf_converter", lambda: None)
        with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf"):
            convert_to_gguf(tmp_path, tmp_path / "x.gguf")

    def test_invokes_converter(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "whoa_llm.ui.export.find_gguf_converter",
            lambda: "/fake/convert_hf_to_gguf.py",
        )
        calls = {}
        def fake_run(cmd, check):
            calls["cmd"] = cmd
            calls["check"] = check
            return subprocess.CompletedProcess(cmd, 0)
        monkeypatch.setattr("subprocess.run", fake_run)
        out = convert_to_gguf(tmp_path, tmp_path / "model.gguf", quant="q5_k_m")
        assert out == tmp_path / "model.gguf"
        assert calls["check"] is True
        assert "--outtype" in calls["cmd"]
        assert "q5_k_m" in calls["cmd"]


class TestCardFromConfig:
    def test_renders_fields_from_config_dict(self, tmp_path):
        config = {
            "model_id": "meta-llama/Llama-3.2-1B",
            "method": "qlora",
            "precision": "bf16",
            "dataset": {"name": "tatsu-lab/alpaca"},
            "train": {
                "epochs": 1, "learning_rate": 2e-4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
            },
            "lora": {"r": 16, "alpha": 32},
            "memory": {"quantization": "4bit"},
        }
        path = card_from_config(tmp_path, config)
        text = path.read_text()
        assert "meta-llama/Llama-3.2-1B" in text
        assert "qlora" in text
        assert "tatsu-lab/alpaca" in text
        assert "lora_r" in text
        assert "4bit" in text

    def test_handles_missing_optional_fields(self, tmp_path):
        path = card_from_config(tmp_path, {"model_id": "x"})
        assert path.exists()
        text = path.read_text()
        assert "x" in text


class TestWriteModelCardYaml:
    def test_includes_yaml_frontmatter(self, tmp_path):
        p = write_model_card(
            tmp_path, base_model="b", method="lora", dataset="d", hyperparams={"r": 16},
        )
        text = p.read_text()
        assert text.startswith("---")
        assert "base_model: b" in text
        assert "library_name: peft" in text
