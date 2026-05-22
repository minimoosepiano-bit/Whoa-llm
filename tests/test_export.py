"""Tests for the export helpers (mocks all heavy ML calls)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whoa_llm.ui.export import push_to_hub, write_model_card


def test_write_model_card(tmp_path):
    p = write_model_card(
        tmp_path,
        base_model="meta-llama/Llama-3.2-1B",
        method="lora",
        dataset="alpaca",
        hyperparams={"r": 16},
    )
    assert p.exists()
    body = p.read_text()
    assert "meta-llama/Llama-3.2-1B" in body
    assert "lora" in body
    assert "alpaca" in body


def test_push_to_hub_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        push_to_hub(tmp_path / "does-not-exist", "user/repo")


def test_push_to_hub_calls_hf_api(tmp_path):
    # Make a real folder so the FileNotFoundError check passes.
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "weights.safetensors").write_bytes(b"\x00\x00")

    with patch("huggingface_hub.create_repo") as create, \
         patch("huggingface_hub.HfApi") as api_cls:
        api = MagicMock()
        api_cls.return_value = api
        url = push_to_hub(folder, "user/myrepo", private=True)
    create.assert_called_once_with("user/myrepo", private=True, exist_ok=True)
    api.upload_folder.assert_called_once()
    assert url == "https://huggingface.co/user/myrepo"
