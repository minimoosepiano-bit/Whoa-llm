"""Tests for the memory recommender (pure function, no torch needed)."""

from dataclasses import replace

import pytest

from whoa_llm.hardware import (
    GPUInfo,
    HardwarePreset,
    MemoryRecommendation,
    recommend_memory,
)


def _gpu(vram: float, free: float | None = None, bf16: bool = True) -> GPUInfo:
    return GPUInfo(
        index=0,
        name="MockGPU",
        total_vram_gb=vram,
        free_vram_gb=free if free is not None else vram,
        compute_capability=(8, 0) if bf16 else (7, 5),
        supports_bf16=bf16,
    )


def _preset(gpus, ram: float = 32.0) -> HardwarePreset:
    return HardwarePreset(
        has_cuda=bool(gpus),
        gpus=gpus,
        total_ram_gb=ram,
        available_ram_gb=ram,
        recommended_precision="bf16" if gpus else "fp32",
        recommended_device_map="auto" if gpus else "cpu",
    )


class TestRecommendMemory:
    def test_no_gpu_falls_back_to_cpu_fp32(self):
        rec = recommend_memory(0.5, _preset(gpus=[], ram=8.0))
        assert rec.precision == "fp32"
        assert rec.quantization == "none"
        assert rec.cpu_offload is False
        assert rec.attn_impl is None
        assert "No CUDA GPU" in rec.rationale

    def test_huge_vram_picks_full_precision(self):
        # 80 GB GPU, 7B model → plenty of headroom.
        rec = recommend_memory(7.0, _preset([_gpu(80, 78)]))
        assert rec.quantization == "none"
        assert rec.cpu_offload is False
        assert rec.precision == "bf16"
        assert rec.attn_impl == "sdpa"

    def test_tight_vram_picks_qlora_4bit(self):
        # 8 GB consumer GPU, 7B model → 4-bit weights ≈ 3.85 GB → fits with
        # activations; recommender should pick QLoRA 4-bit on-GPU (no offload).
        rec = recommend_memory(7.0, _preset([_gpu(8, 7.5)], ram=16.0))
        assert rec.method == "qlora"
        assert rec.quantization == "4bit"
        assert rec.cpu_offload is False
        assert rec.gradient_checkpointing is True

    def test_too_small_vram_enables_cpu_offload(self):
        # 6 GB GPU, 13B model → 4-bit weights ≈ 7.15 GB, doesn't fit → offload.
        rec = recommend_memory(13.0, _preset([_gpu(6, 5.5)], ram=32.0))
        assert rec.method == "qlora"
        assert rec.quantization == "4bit"
        assert rec.cpu_offload is True
        assert rec.disk_offload_dir is not None
        assert rec.max_memory is not None
        assert 0 in rec.max_memory
        assert "cpu" in rec.max_memory

    def test_breakdown_populated(self):
        rec = recommend_memory(7.0, _preset([_gpu(24, 22)]))
        assert rec.breakdown["model_size_b"] == 7.0
        assert rec.breakdown["weights_bf16_gb"] > 0
        assert rec.breakdown["weights_4bit_gb"] > 0
        assert rec.breakdown["free_vram_gb"] == 22.0

    def test_to_memory_config_shape(self):
        rec = recommend_memory(7.0, _preset([_gpu(24, 22)]))
        mc = rec.to_memory_config()
        assert set(mc.keys()) == {
            "quantization", "cpu_offload", "disk_offload_dir", "attn_impl", "max_memory"
        }

    def test_rationale_is_human_readable(self):
        rec = recommend_memory(7.0, _preset([_gpu(8, 7.5)]))
        assert len(rec.rationale) > 20
        assert "VRAM" in rec.rationale

    def test_pre_ampere_gpu_gets_fp16(self):
        rec = recommend_memory(7.0, _preset([_gpu(24, 22, bf16=False)]))
        # The preset's recommended_precision is bf16 because we set it that way;
        # but if we build a non-bf16 preset, the recommender should respect it.
        preset = HardwarePreset(
            has_cuda=True,
            gpus=[_gpu(24, 22, bf16=False)],
            total_ram_gb=32, available_ram_gb=32,
            recommended_precision="fp16", recommended_device_map="auto",
        )
        rec = recommend_memory(7.0, preset)
        assert rec.precision == "fp16"
