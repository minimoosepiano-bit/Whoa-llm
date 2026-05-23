"""Tests for built-in reward functions and the loader (offline)."""

import pytest

from whoa_llm.training.rewards import (
    contains_reward,
    discover_builtins,
    length_reward,
    regex_format_reward,
    resolve_reward,
    resolve_rewards,
)


class TestLengthReward:
    def test_peak_at_target(self):
        c = ["word " * 100]
        out = length_reward([""], c, target_tokens=100)
        assert out == [1.0]

    def test_falls_off_with_distance(self):
        c = ["word " * 50, "word " * 100, "word " * 150]
        out = length_reward(["", "", ""], c, target_tokens=100)
        assert out[1] == 1.0
        assert out[0] == pytest.approx(0.5)
        assert out[2] == pytest.approx(0.5)

    def test_clipped_to_zero(self):
        out = length_reward([""], ["a"] * 1, target_tokens=10)
        # 1 word vs target 10 → distance 9 → 1 - 9/10 = 0.1
        assert 0.05 < out[0] < 0.15

    def test_handles_message_list_form(self):
        c = [[{"role": "assistant", "content": "hi " * 50}]]
        out = length_reward([""], c, target_tokens=50)
        assert out[0] == 1.0


class TestRegexFormatReward:
    def test_match(self):
        out = regex_format_reward([""], ["<answer>42</answer>"], pattern=r"<answer>\d+</answer>")
        assert out == [1.0]

    def test_no_match(self):
        out = regex_format_reward([""], ["nope"], pattern=r"<answer>\d+</answer>")
        assert out == [0.0]

    def test_mixed_batch(self):
        out = regex_format_reward(
            [""] * 3,
            ["<answer>1</answer>", "no", "prefix <answer>9</answer> suffix"],
            pattern=r"<answer>\d+</answer>",
        )
        assert out == [1.0, 0.0, 1.0]


class TestContainsReward:
    def test_fraction(self):
        out = contains_reward([""], ["The quick brown fox"], keywords=["quick", "fox", "slow"])
        assert out[0] == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        out = contains_reward([""], ["HELLO world"], keywords=["hello"])
        assert out == [1.0]

    def test_no_keywords(self):
        out = contains_reward([""], ["anything"], keywords=[])
        assert out == [0.0]


class TestDiscoverBuiltins:
    def test_lists_expected(self):
        b = discover_builtins()
        assert "length_reward" in b
        assert "regex_format_reward" in b
        assert "contains_reward" in b
        for v in b.values():
            assert callable(v)


class TestResolveReward:
    def test_resolve_builtin_short_name(self):
        f = resolve_reward("length_reward")
        out = f([""], ["a b c"], target_tokens=3)
        assert out == [1.0]

    def test_resolve_callable_passthrough(self):
        sentinel = lambda p, c, **kw: [42.0]  # noqa: E731
        out = resolve_reward(sentinel)([""], [""])
        assert out == [42.0]

    def test_resolve_dotted_path(self):
        f = resolve_reward("whoa_llm.training.rewards.builtin.length_reward")
        out = f([""], ["a b"], target_tokens=2)
        assert out == [1.0]

    def test_resolve_dict_with_kwargs(self):
        f = resolve_reward({"name": "regex_format_reward", "kwargs": {"pattern": r"^x$"}})
        assert f([""], ["x"]) == [1.0]
        assert f([""], ["y"]) == [0.0]

    def test_kwargs_binding_via_partial(self):
        f = resolve_reward("regex_format_reward", pattern=r"\d+")
        assert f([""], ["abc 123"]) == [1.0]
        assert f([""], ["abc"]) == [0.0]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown reward"):
            resolve_reward("nonexistent_reward_xyz")

    def test_bad_dotted_path_raises(self):
        with pytest.raises(ValueError, match="Could not import"):
            resolve_reward("not.a.real.module.func")


class TestResolveRewards:
    def test_resolves_list(self):
        specs = [
            "length_reward",
            {"name": "regex_format_reward", "kwargs": {"pattern": r"\d"}},
        ]
        fns = resolve_rewards(specs)
        assert len(fns) == 2
        assert callable(fns[0]) and callable(fns[1])
