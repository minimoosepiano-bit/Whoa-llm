from whoa_llm.hardware import HardwarePreset, detect


def test_detect_returns_preset():
    p = detect()
    assert isinstance(p, HardwarePreset)
    assert p.total_ram_gb > 0
    assert p.available_ram_gb > 0
    assert p.recommended_precision in {"bf16", "fp16", "fp32"}
    assert p.recommended_device_map in {"auto", "cuda", "cpu"}
    if not p.has_cuda:
        assert p.gpus == []
        assert p.recommended_device_map == "cpu"
        assert p.recommended_precision == "fp32"


def test_summary_contains_key_fields():
    s = detect().summary()
    assert "CUDA available" in s
    assert "RAM:" in s
    assert "Recommended:" in s
