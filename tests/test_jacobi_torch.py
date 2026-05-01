"""Correctness of the pure-torch Jacobi eigendecomposition.

NOTE: This implementation is correct but NOT competitive on MPS due to
Python kernel-launch overhead (see docs/equivalence.md, V1.1 section).
It is kept as a portable reference and as a likely-useful path on CUDA
where launch overhead is much lower. The Metal kernel
(:mod:`asrpy_gpu._jacobi_metal`) is the production path for MPS.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asrpy_gpu._jacobi_torch import _tournament_pairs, jacobi_eigh


def _make_spd(n: int, batch: int | None, *, seed: int, dtype=torch.float64):
    """Random symmetric positive-definite matrices."""
    g = torch.Generator().manual_seed(seed)
    if batch is None:
        A = torch.randn(n, n, dtype=dtype, generator=g)
        A = A @ A.T + 1e-3 * torch.eye(n, dtype=dtype)
    else:
        A = torch.randn(batch, n, n, dtype=dtype, generator=g)
        A = A @ A.transpose(-2, -1) + 1e-3 * torch.eye(n, dtype=dtype)
    return A


def test_tournament_pairs_covers_all_pairs():
    for n in [2, 4, 6, 8, 16]:
        rounds = _tournament_pairs(n)
        assert len(rounds) == n - 1, f"expected {n - 1} rounds, got {len(rounds)}"
        seen = set()
        for rnd in rounds:
            assert len(rnd) == n // 2, "each round must have n/2 pairs"
            # Within a round: indices must be unique (non-overlapping rotations)
            flat = [i for pair in rnd for i in pair]
            assert len(set(flat)) == n
            for p, q in rnd:
                assert p < q
                seen.add((p, q))
        # All C(n, 2) pairs must be covered exactly once
        expected = {(i, j) for i in range(n) for j in range(i + 1, n)}
        assert seen == expected


def test_tournament_pairs_odd():
    # For odd n: each round has (n-1)/2 pairs (one bye per round).
    rounds = _tournament_pairs(5)
    assert len(rounds) == 5  # n+1 - 1 = n rounds for n+1 = 6
    seen = set()
    for rnd in rounds:
        for p, q in rnd:
            seen.add((p, q))
    expected = {(i, j) for i in range(5) for j in range(i + 1, 5)}
    assert seen == expected


def test_jacobi_eigh_unbatched_small():
    A = _make_spd(8, None, seed=0)
    D_ref, V_ref = torch.linalg.eigh(A)
    D, V = jacobi_eigh(A, max_sweeps=20)
    np.testing.assert_allclose(D.numpy(), D_ref.numpy(), rtol=1e-10, atol=1e-12)
    # A V = V D
    np.testing.assert_allclose(
        (A @ V).numpy(), (V * D.unsqueeze(-2)).numpy(), rtol=1e-9, atol=1e-10
    )
    # Orthogonality
    np.testing.assert_allclose((V @ V.T).numpy(), np.eye(8), rtol=1e-10, atol=1e-12)


def test_jacobi_eigh_batched_cpu_f64():
    A = _make_spd(16, 4, seed=1)
    D_ref, V_ref = torch.linalg.eigh(A)
    D, V = jacobi_eigh(A, max_sweeps=25)
    np.testing.assert_allclose(D.numpy(), D_ref.numpy(), rtol=1e-9, atol=1e-11)


def test_jacobi_eigh_diagonal_input():
    """Diagonal input must converge in zero sweeps."""
    diag = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    A = torch.diag(diag)
    D, V = jacobi_eigh(A, max_sweeps=1)
    np.testing.assert_allclose(D.numpy(), diag.numpy(), rtol=1e-12)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_jacobi_eigh_mps_f32():
    """Float32 on MPS: looser tolerance but algorithmically equivalent."""
    A = _make_spd(8, 2, seed=2, dtype=torch.float32).to("mps")
    D_ref, V_ref = torch.linalg.eigh(A.cpu().to(torch.float64))
    D, V = jacobi_eigh(A, max_sweeps=20)
    np.testing.assert_allclose(
        D.cpu().to(torch.float64).numpy(), D_ref.numpy(), rtol=1e-4, atol=1e-5
    )
