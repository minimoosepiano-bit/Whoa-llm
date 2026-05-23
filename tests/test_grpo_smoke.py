"""End-to-end GRPO smoke test on a tiny offline model.

Builds a tiny GPT-2 in-process (no HF Hub access), runs a handful of GRPO
steps with the built-in ``length_reward``, and asserts the trainer
produced a non-empty reward/loss history and saved a checkpoint.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trl", reason="trl not installed — skipping GRPO smoke test")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")


def _make_tiny_gpt2(path) -> str:
    """Create + save a tiny GPT-2 model and a word-level fast tokenizer."""
    import os

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    os.makedirs(path, exist_ok=True)

    vocab_words = [
        "<pad>", "<bos>", "<eos>", "<unk>",
        "Tell", "me", "a", "story", "about", "the", "and", "is", "of",
        "in", "to", "for", "with", "on", "at", "this", "that", "it",
    ] + [str(i) for i in range(100)]
    vocab = {w: i for i, w in enumerate(vocab_words)}

    tok_inner = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_inner.pre_tokenizer = Whitespace()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_inner,
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
    )
    tok.save_pretrained(str(path))

    cfg = GPT2Config(
        vocab_size=len(vocab),
        n_positions=128, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=vocab["<bos>"], eos_token_id=vocab["<eos>"],
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(str(path))
    return str(path)


def test_grpo_smoke(tmp_path, monkeypatch):
    from datasets import Dataset

    from whoa_llm.training.callbacks import MetricsCallback
    from whoa_llm.training.grpo import GRPOConfig, RewardSpec, run_grpo
    from whoa_llm.training.sft import DatasetConfig, LoRAConfig, TrainConfig

    model_path = _make_tiny_gpt2(tmp_path / "model")

    def _fake_load(name, config=None, split="train", streaming=False, max_samples=None):
        rows = [{"prompt": f"Tell me a story about {i}"} for i in range(8)]
        return Dataset.from_list(rows)

    monkeypatch.setattr("whoa_llm.data.hf_datasets.load_dataset_split", _fake_load)
    monkeypatch.setenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    cfg = GRPOConfig(
        model_id=model_path,
        dataset=DatasetConfig(name="dummy", format="raw_text"),
        method="lora",
        lora=LoRAConfig(r=4, alpha=8, target_modules=["c_attn"]),
        rewards=[RewardSpec(name="length_reward", kwargs={"target_tokens": 8})],
        num_generations=2,
        max_completion_length=16,
        max_prompt_length=16,
        train=TrainConfig(
            epochs=1, max_steps=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_seq_length=32,
            logging_steps=1, save_steps=100,
            optim="adamw_torch",
            gradient_checkpointing=False,
        ),
        precision="fp32",
        output_dir=str(tmp_path / "out"),
        engine="hf",
    )

    callback = MetricsCallback()
    summary = run_grpo(cfg, metrics_callback=callback)

    assert summary["steps"] >= 1, summary
    assert summary["num_rewards"] == 1
    assert summary["method"] == "lora"

    out_dir = tmp_path / "out"
    adapter_files = list(out_dir.glob("adapter_*"))
    assert adapter_files, f"No adapter files in {out_dir}: {list(out_dir.iterdir())}"

    snap = callback.snapshot()
    assert len(snap) >= 1
