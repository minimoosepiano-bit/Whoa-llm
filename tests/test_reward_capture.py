"""Tests for reward-sample capture and GRPO-aware RunState."""

import time
from types import SimpleNamespace

import pytest

from whoa_llm.ui.state import (
    RewardSample,
    RewardSampleBuffer,
    RunState,
    wrap_reward_for_capture,
)


class TestRewardSampleBuffer:
    def test_add_and_snapshot(self):
        buf = RewardSampleBuffer(max_samples=5)
        buf.add("p1", "c1", 0.5)
        buf.add("p2", "c2", 1.0)
        snap = buf.snapshot()
        assert [s.reward for s in snap] == [0.5, 1.0]

    def test_ring_buffer_caps(self):
        buf = RewardSampleBuffer(max_samples=3)
        for i in range(5):
            buf.add(f"p{i}", f"c{i}", float(i))
        snap = buf.snapshot()
        assert len(snap) == 3
        assert [s.prompt for s in snap] == ["p2", "p3", "p4"]

    def test_as_table_truncates_long_strings(self):
        buf = RewardSampleBuffer()
        buf.bump_step(7)
        buf.add("a" * 200, "b" * 300, 0.42)
        row = buf.as_table()[0]
        assert row[0] == 7
        assert len(row[1]) <= 120
        assert len(row[2]) <= 160
        assert row[3] == 0.42

    def test_reset(self):
        buf = RewardSampleBuffer()
        buf.add("p", "c", 1.0)
        buf.bump_step(10)
        buf.reset()
        assert buf.snapshot() == []
        # New samples should be at step 0 again.
        buf.add("p", "c", 1.0)
        assert buf.snapshot()[0].step == 0


class TestWrapRewardForCapture:
    def test_preserves_returned_rewards(self):
        buf = RewardSampleBuffer()
        def reward_fn(prompts, completions, **kw):
            return [0.3, 0.7]
        wrapped = wrap_reward_for_capture(reward_fn, buf)
        out = wrapped(["p1", "p2"], ["c1", "c2"])
        assert out == [0.3, 0.7]
        snap = buf.snapshot()
        assert len(snap) == 2
        assert snap[0].prompt == "p1"
        assert snap[1].reward == 0.7

    def test_preserves_name(self):
        buf = RewardSampleBuffer()
        def my_reward(p, c, **kw):
            return [1.0]
        wrapped = wrap_reward_for_capture(my_reward, buf)
        assert wrapped.__name__ == "my_reward"

    def test_handles_chat_message_completions(self):
        buf = RewardSampleBuffer()
        wrapped = wrap_reward_for_capture(lambda p, c, **kw: [0.5], buf)
        wrapped(["q"], [[{"role": "assistant", "content": "answer here"}]])
        snap = buf.snapshot()
        assert "answer here" in snap[0].completion

    def test_capture_failure_doesnt_break_training(self, monkeypatch):
        buf = RewardSampleBuffer()
        # Force the capture path to blow up.
        def boom(*a, **kw):
            raise RuntimeError("capture failed")
        monkeypatch.setattr(buf, "add", boom)
        wrapped = wrap_reward_for_capture(lambda p, c, **kw: [1.0], buf)
        # Should still return rewards normally.
        assert wrapped(["p"], ["c"]) == [1.0]

    def test_caps_samples_per_step(self):
        buf = RewardSampleBuffer(max_samples=100)
        wrapped = wrap_reward_for_capture(
            lambda p, c, **kw: [0.0] * 10, buf, max_per_step=3,
        )
        wrapped([f"p{i}" for i in range(10)], [f"c{i}" for i in range(10)])
        assert len(buf.snapshot()) == 3


class TestRunStateGRPODispatch:
    def test_kind_sft_calls_run_sft(self, monkeypatch):
        state = RunState()
        seen: dict = {}
        def fake_sft(cfg, *, metrics_callback=None, extra_callbacks=None):
            seen["sft"] = True
            return {"steps": 1}
        def fake_grpo(*a, **kw):
            seen["grpo"] = True
        monkeypatch.setattr("whoa_llm.training.sft.run_sft", fake_sft)
        monkeypatch.setattr("whoa_llm.training.grpo.run_grpo", fake_grpo)

        cfg = SimpleNamespace(model_id="x", method="lora", output_dir="/tmp/x")
        state.start(cfg, kind="sft")
        state._thread.join(timeout=2)
        assert seen == {"sft": True}
        assert state.kind == "sft"

    def test_kind_grpo_calls_run_grpo(self, monkeypatch):
        state = RunState()
        captured: dict = {}
        def fake_grpo(cfg, *, metrics_callback=None, extra_callbacks=None,
                      reward_funcs_override=None):
            captured["override"] = reward_funcs_override
            captured["extra"] = extra_callbacks
            return {"steps": 1, "num_rewards": len(reward_funcs_override or [])}
        monkeypatch.setattr("whoa_llm.training.grpo.run_grpo", fake_grpo)

        from whoa_llm.training.grpo import RewardSpec
        cfg = SimpleNamespace(
            model_id="x", method="lora", output_dir="/tmp/x",
            rewards=[RewardSpec(name="length_reward", kwargs={"target_tokens": 10})],
        )
        state.start(cfg, kind="grpo")
        state._thread.join(timeout=3)
        assert state.kind == "grpo"
        assert captured["override"] is not None and len(captured["override"]) == 1
        # extra_callbacks should include cancel + step-tracking.
        assert len(captured["extra"]) == 2

    def test_unknown_kind_records_error(self, monkeypatch):
        state = RunState()
        cfg = SimpleNamespace(model_id="x", method="lora", output_dir="/tmp/x")
        state.start(cfg, kind="weird")
        state._thread.join(timeout=2)
        assert state.last_error is not None
        assert "weird" in state.last_error or "Unknown" in state.last_error
