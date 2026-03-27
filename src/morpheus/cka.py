"""
Centered Kernel Alignment (CKA)
================================

Implements Linear CKA for measuring representational similarity between
core versions (Kornblith et al., 2019).

Linear CKA operates on feature matrices directly with O(n * d^2) complexity
instead of the O(n^2 * d) of kernel CKA, making it feasible for large
representation spaces. Mini-batch CKA provides stochastically unbiased
estimates at constant memory cost.

Used by System 1 (Stable Core) to enforce bounded representation drift
during structural distillation.
"""

from __future__ import annotations

import torch
import numpy as np


def _center_matrix(K: torch.Tensor) -> torch.Tensor:
    """Center a matrix by removing row and column means."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
    return H @ K @ H


def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """Compute Linear CKA between two representation matrices.

    Linear CKA measures the similarity between two sets of representations
    using their linear kernel matrices, centered to remove trivial
    correlations.

    Args:
        X: Representations from model version v, shape (n, d1).
        Y: Representations from model version v+1, shape (n, d2).

    Returns:
        CKA similarity in [0, 1]. 1.0 means identical representation spaces.
    """
    assert X.shape[0] == Y.shape[0], (
        f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
    )
    n = X.shape[0]
    if n < 2:
        return 1.0

    X = X.float()
    Y = Y.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear CKA closed-form: HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    # where K = X @ X^T, L = Y @ Y^T
    # Efficient form avoids building n x n matrices:
    # HSIC_linear(X, Y) = ||Y^T X||_F^2 / (n - 1)^2
    YtX = Y.T @ X                   # (d2, d1)
    hsic_xy = (YtX * YtX).sum()     # ||Y^T X||_F^2

    XtX = X.T @ X                   # (d1, d1)
    hsic_xx = (XtX * XtX).sum()     # ||X^T X||_F^2

    YtY = Y.T @ Y                   # (d2, d2)
    hsic_yy = (YtY * YtY).sum()     # ||Y^T Y||_F^2

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 1.0

    cka = (hsic_xy / denom).item()
    return float(np.clip(cka, 0.0, 1.0))


def minibatch_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 256,
    n_batches: int = 10,
    seed: int = 42,
) -> float:
    """Compute mini-batch CKA for large probe sets.

    Provides a stochastically unbiased estimate of CKA at constant memory
    cost by averaging over random subsets of samples.

    Args:
        X: Representations from model version v, shape (n, d1).
        Y: Representations from model version v+1, shape (n, d2).
        batch_size: Samples per mini-batch.
        n_batches: Number of mini-batches to average.
        seed: Random seed for reproducibility.

    Returns:
        Estimated CKA similarity in [0, 1].
    """
    n = X.shape[0]
    batch_size = min(batch_size, n)
    rng = np.random.RandomState(seed)
    cka_values = []

    for _ in range(n_batches):
        indices = rng.choice(n, size=batch_size, replace=False)
        idx = torch.tensor(indices, device=X.device)
        cka_val = linear_cka(X[idx], Y[idx])
        cka_values.append(cka_val)

    return float(np.mean(cka_values))


def compute_representation_shift(
    X: torch.Tensor,
    Y: torch.Tensor,
    use_minibatch: bool = False,
    **kwargs,
) -> float:
    """Compute representation shift as 1 - CKA.

    A shift of 0.0 means identical representations; 1.0 means completely
    different. The stable core constrains this to stay below cka_threshold.

    Args:
        X: Old representations, shape (n, d).
        Y: New representations, shape (n, d).
        use_minibatch: Use mini-batch CKA for large probe sets.
        **kwargs: Passed to minibatch_cka if use_minibatch=True.

    Returns:
        Representation shift in [0, 1].
    """
    if use_minibatch:
        cka = minibatch_cka(X, Y, **kwargs)
    else:
        cka = linear_cka(X, Y)
    return 1.0 - cka
