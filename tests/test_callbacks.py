"""Tests for MetricsCallback and OOMRecoveryCallback (no torch needed)."""

from types import SimpleNamespace

from whoa_llm.training.callbacks import MetricsCallback, OOMRecoveryCallback


class TestMetricsCallback:
    def test_starts_empty(self):
        cb = MetricsCallback()
        assert cb.snapshot() == []
        assert cb.latest is None
        assert cb.is_running is False

    def test_on_log_appends(self):
        cb = MetricsCallback()
        cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
        cb.on_log(
            None,
            SimpleNamespace(global_step=10, epoch=0.1),
            None,
            logs={"loss": 1.23, "learning_rate": 1e-4, "grad_norm": 0.5},
        )
        snap = cb.snapshot()
        assert len(snap) == 1
        assert snap[0]["loss"] == 1.23
        assert snap[0]["step"] == 10
        assert cb.latest["loss"] == 1.23

    def test_reset_clears(self):
        cb = MetricsCallback()
        cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
        cb.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": 2.0})
        cb.reset()
        assert cb.snapshot() == []
        assert cb.latest is None

    def test_ring_buffer_caps(self):
        cb = MetricsCallback(max_points=3)
        cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
        for i in range(5):
            cb.on_log(
                None, SimpleNamespace(global_step=i), None, logs={"loss": float(i)}
            )
        snap = cb.snapshot()
        assert len(snap) == 3
        assert [p["step"] for p in snap] == [2, 3, 4]

    def test_on_train_end_clears_running(self):
        cb = MetricsCallback()
        cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
        assert cb.is_running
        cb.on_train_end(None, SimpleNamespace(global_step=100), None)
        assert not cb.is_running


class TestOOMRecovery:
    def test_first_oom_halves_batch(self):
        cb = OOMRecoveryCallback()
        args = SimpleNamespace(
            per_device_train_batch_size=8, gradient_accumulation_steps=2,
            gradient_checkpointing=False,
        )
        ok = cb.handle_oom(args)
        assert ok is True
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 4

    def test_second_oom_gives_up(self):
        cb = OOMRecoveryCallback()
        args = SimpleNamespace(
            per_device_train_batch_size=8, gradient_accumulation_steps=2,
            gradient_checkpointing=False,
        )
        cb.handle_oom(args)
        ok = cb.handle_oom(args)
        assert ok is False

    def test_batch_size_one_enables_grad_checkpointing(self):
        cb = OOMRecoveryCallback()
        args = SimpleNamespace(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            gradient_checkpointing=False,
        )
        cb.handle_oom(args)
        assert args.gradient_checkpointing is True
