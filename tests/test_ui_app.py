"""UI smoke tests: build the Gradio app + assemble configs from form dicts."""

import pytest

gr = pytest.importorskip("gradio")


def test_build_app_constructs_blocks():
    from whoa_llm.ui.app import build_app
    app = build_app()
    assert app is not None
    # Gradio Blocks expose .blocks (mapping of component ids).
    assert hasattr(app, "blocks")


def test_build_config_from_form():
    from whoa_llm.ui.app import _build_config
    form = {
        "model_id": "meta-llama/Llama-3.2-1B",
        "method": "qlora",
        "engine": "auto",
        "precision": "bf16",
        "dataset_name": "tatsu-lab/alpaca",
        "dataset_split": "train[:100]",
        "dataset_format": "alpaca",
        "epochs": 1,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "grad_accum": 4,
        "max_seq_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "quantization": "4bit",
        "cpu_offload": False,
        "output_dir": "/tmp/whoa-ui-test",
        "seed": 7,
    }
    cfg = _build_config(form)
    assert cfg.model_id == "meta-llama/Llama-3.2-1B"
    assert cfg.method == "qlora"
    assert cfg.lora.r == 16
    assert cfg.train.per_device_train_batch_size == 2
    # QLoRA normalisation kicks in.
    assert cfg.memory.quantization == "4bit"
    assert cfg.dataset.name == "tatsu-lab/alpaca"
    assert cfg.seed == 7


def test_update_form_handler():
    from whoa_llm.ui.app import _update_form
    h = _update_form("model_id")
    new = h({}, "abc/def")
    assert new == {"model_id": "abc/def"}
    # Existing keys preserved.
    new2 = h({"method": "lora"}, "x/y")
    assert new2 == {"method": "lora", "model_id": "x/y"}


def test_estimate_memory_handler_empty_returns_message():
    from whoa_llm.ui.app import _estimate_memory_handler
    md, rows = _estimate_memory_handler("", "lora", 1024)
    assert "Enter" in md
    assert rows == []


def test_estimate_memory_handler_unreadable_config(monkeypatch):
    from whoa_llm.ui import app as app_mod
    monkeypatch.setattr("whoa_llm.hardware.estimate_memory", lambda *a, **k: None)
    md, rows = app_mod._estimate_memory_handler("nope/nope", "lora", 1024)
    assert "Could not read" in md
    assert rows == []
