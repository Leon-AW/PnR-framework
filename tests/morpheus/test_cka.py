"""
Unit Tests — CKA (Centered Kernel Alignment)
=============================================

Tests that linear CKA and mini-batch CKA produce correct similarity
scores, satisfy expected mathematical properties, and that
compute_representation_shift correctly inverts CKA.
"""

import pytest
import torch
import numpy as np

from src.morpheus.cka import linear_cka, minibatch_cka, compute_representation_shift


class TestLinearCKA:
    """Tests for the linear CKA closed-form computation."""

    def test_identical_representations_give_one(self):
        X = torch.randn(100, 64)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-4)

    def test_random_representations_less_than_one(self):
        X = torch.randn(100, 64)
        Y = torch.randn(100, 64)
        cka = linear_cka(X, Y)
        assert 0.0 <= cka < 1.0

    def test_symmetry(self):
        X = torch.randn(80, 32)
        Y = torch.randn(80, 32)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-6)

    def test_orthogonal_representations_near_zero(self):
        torch.manual_seed(0)
        X = torch.zeros(200, 40)
        Y = torch.zeros(200, 40)
        X[:, :20] = torch.randn(200, 20)
        Y[:, 20:] = torch.randn(200, 20)
        cka = linear_cka(X, Y)
        assert cka < 0.25

    def test_linearly_transformed_representations(self):
        """CKA should be high when Y = X @ A for invertible A."""
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        A = torch.eye(32) + 0.1 * torch.randn(32, 32)
        Y = X @ A
        cka = linear_cka(X, Y)
        assert cka > 0.5

    def test_different_dimensionalities(self):
        X = torch.randn(80, 32)
        Y = torch.randn(80, 64)
        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_single_sample_returns_one(self):
        X = torch.randn(1, 16)
        Y = torch.randn(1, 16)
        assert linear_cka(X, Y) == 1.0

    def test_sample_count_mismatch_raises(self):
        X = torch.randn(50, 32)
        Y = torch.randn(60, 32)
        with pytest.raises(AssertionError):
            linear_cka(X, Y)

    def test_output_range(self):
        for _ in range(10):
            X = torch.randn(50, 16)
            Y = torch.randn(50, 16)
            cka = linear_cka(X, Y)
            assert 0.0 <= cka <= 1.0

    def test_scaled_representations(self):
        """CKA should be invariant to isotropic scaling."""
        X = torch.randn(80, 32)
        Y = X * 5.0
        cka = linear_cka(X, Y)
        assert cka == pytest.approx(1.0, abs=1e-4)


class TestMinibatchCKA:
    """Tests for stochastically unbiased mini-batch CKA."""

    def test_converges_to_full_cka(self):
        X = torch.randn(200, 32)
        Y = torch.randn(200, 32)
        full = linear_cka(X, Y)
        mb = minibatch_cka(X, Y, batch_size=150, n_batches=30)
        assert mb == pytest.approx(full, abs=0.15)

    def test_identical_gives_one(self):
        X = torch.randn(100, 32)
        assert minibatch_cka(X, X, batch_size=50, n_batches=10) == pytest.approx(1.0, abs=0.02)

    def test_deterministic_with_seed(self):
        X = torch.randn(100, 32)
        Y = torch.randn(100, 32)
        a = minibatch_cka(X, Y, seed=123)
        b = minibatch_cka(X, Y, seed=123)
        assert a == b

    def test_different_seeds_give_similar(self):
        X = torch.randn(200, 32)
        Y = torch.randn(200, 32)
        a = minibatch_cka(X, Y, batch_size=100, n_batches=20, seed=1)
        b = minibatch_cka(X, Y, batch_size=100, n_batches=20, seed=2)
        assert abs(a - b) < 0.1


class TestRepresentationShift:
    """Tests for compute_representation_shift."""

    def test_identical_gives_zero(self):
        X = torch.randn(100, 64)
        shift = compute_representation_shift(X, X)
        assert shift == pytest.approx(0.0, abs=1e-4)

    def test_random_gives_positive(self):
        X = torch.randn(100, 64)
        Y = torch.randn(100, 64)
        shift = compute_representation_shift(X, Y)
        assert shift > 0.0

    def test_shift_is_one_minus_cka(self):
        X = torch.randn(100, 32)
        Y = torch.randn(100, 32)
        cka = linear_cka(X, Y)
        shift = compute_representation_shift(X, Y)
        assert shift == pytest.approx(1.0 - cka, abs=1e-6)

    def test_minibatch_mode(self):
        X = torch.randn(100, 32)
        shift = compute_representation_shift(X, X, use_minibatch=True, batch_size=50)
        assert shift == pytest.approx(0.0, abs=0.02)
