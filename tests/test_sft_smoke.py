"""End-to-end SFT smoke test on a tiny model + tiny dataset.

Skipped unless torch / transformers / trl / peft / datasets are installed.
The test builds a tiny GPT-2 in-process and saves it to a local path so it
runs **fully offline** — no Hugging Face Hub access required.

Asserts:
    * ``run_sft`` completes,
    * ``adapter_model.safetensors`` (or ``adapter_model.bin``) is written,
    * the MetricsCallback captured at least one log point.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trl", reason="trl not installed — skipping SFT smoke test")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")


def _make_tiny_gpt2(path) -> str:
    """Create + save a tiny GPT-2 model and tokenizer at *path*; return path."""
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    import os

    cfg = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(str(path))

    # Use the real GPT2 tokenizer files if cached locally; otherwise build a
    # minimal vocab so we never need network access.
    vocab = {f"<{i}>": i for i in range(900)}
    vocab.update({chr(c): 900 + (c - 32) for c in range(32, 127)})
    merges = "#version: 0.2\n"

    import json
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(f"{path}/merges.txt", "w") as f:
        f.write(merges)

    tok = GPT2Tokenizer(vocab_file=f"{path}/vocab.json", merges_file=f"{path}/merges.txt")
    tok.pad_token = tok.eos_token = "<0>"
    tok.save_pretrained(str(path))
    return str(path)


def test_sft_lora_smoke(tmp_path, monkeypatch):
    """Run a few steps of LoRA on a tiny offline GPT-2; check adapter was saved."""
    from datasets import Dataset

    from whoa_llm.training.callbacks import MetricsCallback
    from whoa_llm.training.sft import (
        DatasetConfig,
        LoRAConfig,
        SFTConfig,
        TrainConfig,
        run_sft,
    )

    model_path = _make_tiny_gpt2(tmp_path / "model")

    def _fake_load(name, config=None, split="train", streaming=False, max_samples=None):
        rows = [
            {"instruction": f"Say hi {i}", "input": "", "output": f"Hi {i}!"}
            for i in range(16)
        ]
        return Dataset.from_list(rows)

    monkeypatch.setattr("whoa_llm.data.hf_datasets.load_dataset_split", _fake_load)
    monkeypatch.setenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    cfg = SFTConfig(
        model_id=model_path,
        dataset=DatasetConfig(name="dummy", format="alpaca", max_samples=16),
        method="lora",
        lora=LoRAConfig(r=4, alpha=8, target_modules=["c_attn"]),
        train=TrainConfig(
            epochs=1,
            max_steps=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_seq_length=32,
            logging_steps=1,
            save_steps=100,
            optim="adamw_torch",
            gradient_checkpointing=False,
        ),
        precision="fp32",
        output_dir=str(tmp_path / "out"),
        engine="hf",
    )

    callback = MetricsCallback()
    summary = run_sft(cfg, metrics_callback=callback)

    assert summary["steps"] >= 1, summary
    assert summary["method"] == "lora"
    assert summary["engine"] == "hf"

    out_dir = tmp_path / "out"
    adapter_files = list(out_dir.glob("adapter_*"))
    assert adapter_files, f"No adapter_* files in {out_dir}: {list(out_dir.iterdir())}"

    snap = callback.snapshot()
    assert len(snap) >= 1
    assert any(p["loss"] is not None for p in snap)
