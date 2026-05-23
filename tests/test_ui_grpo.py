"""Tests for the GRPO UI handlers (no Gradio launch needed)."""


import pytest

gr = pytest.importorskip("gradio")


def test_list_available_rewards_includes_builtins():
    from whoa_llm.ui.app import _list_available_rewards
    rewards = _list_available_rewards()
    assert "length_reward" in rewards
    assert "regex_format_reward" in rewards
    assert "contains_reward" in rewards


def test_list_available_rewards_picks_up_user_rewards(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    user_dir = tmp_path / ".whoa_llm" / "user_rewards"
    user_dir.mkdir(parents=True)
    (user_dir / "my_custom.py").write_text("def my_custom(p, c, **kw): return [1.0]\n")
    from whoa_llm.ui.app import _list_available_rewards
    rewards = _list_available_rewards()
    assert "my_custom" in rewards


def test_save_custom_reward_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from whoa_llm.ui.app import _save_custom_reward
    code = "def foo(p, c, **kw):\n    return [0.5]\n"
    status = _save_custom_reward("foo", code)
    assert "✅" in status or "Saved" in status
    assert (tmp_path / ".whoa_llm" / "user_rewards" / "foo.py").exists()


def test_save_custom_reward_rejects_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from whoa_llm.ui.app import _save_custom_reward
    status = _save_custom_reward("not-an-identifier", "def x(p, c): return [0]")
    assert "❌" in status


def test_save_custom_reward_rejects_empty_code(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from whoa_llm.ui.app import _save_custom_reward
    status = _save_custom_reward("foo", "")
    assert "❌" in status


def test_build_grpo_config_from_form():
    from whoa_llm.training.grpo import RewardSpec
    from whoa_llm.ui.app import _build_grpo_config

    form = {
        "model_id": "meta-llama/Llama-3.2-1B",
        "method": "lora",
        "dataset_name": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_split": "train[:100]",
        "prompt_column": "question",
        "num_generations": 4,
        "max_prompt_length": 256,
        "max_completion_length": 128,
        "beta": 0.04,
        "temperature": 0.9,
        "top_p": 1.0,
        "use_vllm": False,
        "lora_r": 8,
        "quantization": "4bit",
        "output_dir": "/tmp/whoa-grpo-test",
    }
    rewards = [
        RewardSpec(name="regex_format_reward", kwargs={"pattern": r"\d+"}, weight=1.0),
        RewardSpec(name="length_reward", kwargs={"target_tokens": 64}, weight=0.5),
    ]
    cfg = _build_grpo_config(form, rewards)
    assert cfg.model_id == "meta-llama/Llama-3.2-1B"
    assert cfg.prompt_column == "question"
    assert cfg.num_generations == 4
    assert cfg.beta == 0.04
    assert len(cfg.rewards) == 2
    assert cfg.rewards[0].weight == 1.0
    assert cfg.memory.quantization == "4bit"


def test_build_grpo_config_accepts_dict_rewards():
    from whoa_llm.ui.app import _build_grpo_config
    form = {
        "model_id": "x",
        "method": "lora",
        "dataset_name": "d",
        "output_dir": "/tmp/grpo",
    }
    rewards = [{"name": "length_reward", "kwargs": {"target_tokens": 10}}]
    cfg = _build_grpo_config(form, rewards)
    assert len(cfg.rewards) == 1
    assert cfg.rewards[0].name == "length_reward"
    assert cfg.rewards[0].kwargs == {"target_tokens": 10}


def test_build_app_includes_grpo_tab():
    """Smoke test: building the app shouldn't blow up after adding the GRPO tab."""
    from whoa_llm.ui.app import build_app
    app = build_app()
    assert app is not None
