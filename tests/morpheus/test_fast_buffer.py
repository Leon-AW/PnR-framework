"""
Unit Tests — Fast Adaptation Buffer (System 3)
================================================

Tests for the hippocampal fast adaptation buffer:
- Sample management and capacity limits
- Training step tracking and fill level
- Distribution shift detection
- Pattern separation noise
- Loss statistics computation
- Buffer reset after consolidation
"""

import pytest
import numpy as np
import torch

from src.morpheus.fast_buffer import FastBuffer, BufferSample
from src.morpheus.config import FastBufferConfig


class TestSampleManagement:
    """Tests for adding and retrieving buffer samples."""

    def test_add_and_get_samples(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        buf.add_sample("Hello world", domain_signal="general")
        buf.add_sample("Second sample", domain_signal="medical")
        assert buf.num_samples == 2

    def test_get_training_texts(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        buf.add_sample("text_a")
        buf.add_sample("text_b")
        texts = buf.get_training_texts()
        assert texts == ["text_a", "text_b"]

    def test_get_limited_samples(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        for i in range(10):
            buf.add_sample(f"sample_{i}")
        texts = buf.get_training_texts(n=3)
        assert len(texts) == 3
        assert texts[-1] == "sample_9"


class TestCapacityAndFillLevel:
    """Tests for capacity tracking and fill level."""

    def test_fill_level_starts_at_zero(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        assert buf.fill_level == 0.0

    def test_fill_level_increases_with_steps(self):
        config = FastBufferConfig(max_capacity_steps=100, checkpoint_dir="/tmp/test_buf")
        buf = FastBuffer(config)
        for _ in range(50):
            buf.record_step(loss=0.5)
        assert buf.fill_level == pytest.approx(0.5, abs=0.01)

    def test_is_full_at_capacity(self):
        config = FastBufferConfig(max_capacity_steps=10, checkpoint_dir="/tmp/test_buf")
        buf = FastBuffer(config)
        for _ in range(10):
            buf.record_step(loss=0.5)
        assert buf.is_full

    def test_not_full_before_capacity(self):
        config = FastBufferConfig(max_capacity_steps=100, checkpoint_dir="/tmp/test_buf")
        buf = FastBuffer(config)
        for _ in range(5):
            buf.record_step(loss=0.5)
        assert not buf.is_full

    def test_fill_level_capped_at_one(self):
        config = FastBufferConfig(max_capacity_steps=10, checkpoint_dir="/tmp/test_buf")
        buf = FastBuffer(config)
        for _ in range(20):
            buf.record_step(loss=0.5)
        assert buf.fill_level == 1.0


class TestBufferReset:
    """Tests for buffer reset after consolidation."""

    def test_reset_clears_samples(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        buf.add_sample("text")
        buf.record_step(0.5)
        buf.reset()
        assert buf.num_samples == 0
        assert buf.fill_level == 0.0
        assert not buf.is_full

    def test_reset_preserves_total_steps(self):
        buf = FastBuffer(FastBufferConfig(
            max_capacity_steps=10,
            checkpoint_dir="/tmp/test_buf",
        ))
        for _ in range(10):
            buf.record_step(loss=0.5)
        buf.reset()
        assert buf._total_steps == 10
        assert buf._steps_since_consolidation == 0


class TestDistributionShiftDetection:
    """Tests for distribution shift detection from loss dynamics."""

    def test_no_shift_with_stable_loss(self):
        config = FastBufferConfig(checkpoint_dir="/tmp/test_buf")
        buf = FastBuffer(config)
        for _ in range(200):
            buf.record_step(loss=0.5)
        shift = buf.detect_distribution_shift(window=50)
        assert shift < 0.5

    def test_shift_detected_on_loss_spike(self):
        config = FastBufferConfig(checkpoint_dir="/tmp/test_buf")
        buf = FastBuffer(config)
        for _ in range(50):
            buf.record_step(loss=0.5)
        for _ in range(50):
            buf.record_step(loss=2.0)
        # window=50 with 100 entries: old=losses[0:50] (0.5), new=losses[50:100] (2.0)
        shift = buf.detect_distribution_shift(window=50)
        assert shift > 1.0

    def test_insufficient_data_returns_zero(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        buf.record_step(loss=0.5)
        assert buf.detect_distribution_shift() == 0.0


class TestLossStatistics:
    """Tests for loss statistics computation."""

    def test_empty_buffer_stats(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        stats = buf.get_loss_statistics()
        assert stats["mean"] == 0.0
        assert stats["trend"] == 0.0

    def test_stats_with_data(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        for i in range(100):
            buf.record_step(loss=float(i) / 100)
        stats = buf.get_loss_statistics()
        assert stats["mean"] > 0.0
        assert "std" in stats
        assert "recent" in stats
        assert "trend" in stats

    def test_increasing_loss_positive_trend(self):
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_buf"))
        for i in range(100):
            buf.record_step(loss=0.01 * i)
        stats = buf.get_loss_statistics()
        assert stats["trend"] > 0.0


class TestPatternSeparation:
    """Tests for hippocampal pattern separation noise."""

    def test_noise_modifies_embeddings(self):
        config = FastBufferConfig(
            pattern_separation_noise=0.1,
            checkpoint_dir="/tmp/test_buf",
        )
        buf = FastBuffer(config)
        emb = torch.ones(10, 64)
        separated = buf.apply_pattern_separation(emb)
        assert not torch.allclose(emb, separated)

    def test_no_noise_when_disabled(self):
        config = FastBufferConfig(
            pattern_separation_noise=0.0,
            checkpoint_dir="/tmp/test_buf",
        )
        buf = FastBuffer(config)
        emb = torch.ones(10, 64)
        separated = buf.apply_pattern_separation(emb)
        assert torch.allclose(emb, separated)

    def test_noise_is_sparse(self):
        """Only ~10% of values should be perturbed."""
        config = FastBufferConfig(
            pattern_separation_noise=1.0,
            checkpoint_dir="/tmp/test_buf",
        )
        buf = FastBuffer(config)
        torch.manual_seed(42)
        emb = torch.zeros(100, 64)
        separated = buf.apply_pattern_separation(emb)
        changed_fraction = (separated != 0).float().mean().item()
        assert 0.05 < changed_fraction < 0.2


class TestLoraConfig:
    """Tests for buffer LoRA configuration."""

    def test_lora_config_properties(self):
        config = FastBufferConfig(
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.1,
            checkpoint_dir="/tmp/test_buf",
        )
        buf = FastBuffer(config)
        lora = buf.get_lora_config()
        assert lora.r == 32
        assert lora.lora_alpha == 64
