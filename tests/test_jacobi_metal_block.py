"""Correctness of the block-Jacobi Metal kernel.

NOTE: this kernel is correct but currently SLOWER than the standard
Jacobi (see docs/equivalence.md / commit log). Kept as documented
experimental work; tests guard against regression in the algorithm
itself, not in performance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asrpy_gpu._jacobi_metal import METAL_AVAILABLE

if not METAL_AVAILABLE:
    pytest.skip(
        "pyobjc-framework-Metal not installed; install asrpy_gpu[metal]",
        allow_module_level=True,
    )

from asrpy_gpu._jacobi_metal_block import jacobi_eigh_block  # noqa: E402


def _make_spd(B: int | None, n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if B is None:
        A = rng.standard_normal((n, n)).astype(np.float32)
        A = A @ A.T + 1e-3 * np.eye(n, dtype=np.float32)
    else:
        A = rng.standard_normal((B, n, n)).astype(np.float32)
        A = A @ np.swapaxes(A, -2, -1) + 1e-3 * np.eye(n, dtype=np.float32)[None]
    return np.ascontiguousarray(A)


def _check_eigh(A: np.ndarray, D: np.ndarray, V: np.ndarray, *, atol_rel=2e-4):
    """Validate via residual ||AV - VD|| / ||A||."""
    AV = np.einsum("bij,bjk->bik", A, V) if A.ndim == 3 else A @ V
    if A.ndim == 3:
        VD = V * D[:, None, :]
    else:
        VD = V * D[None, :]
    err = np.abs(AV - VD).max() / np.abs(A).max()
    assert err < atol_rel, f"||AV - VD|| / ||A|| = {err:.3e}"
    if A.ndim == 3:
        VVt = np.einsum("bij,bjk->bik", V, np.swapaxes(V, -2, -1))
        I = np.eye(V.shape[-1], dtype=V.dtype)[None]
        np.testing.assert_allclose(VVt, np.broadcast_to(I, V.shape), atol=2e-3)


def test_block_jacobi_n64():
    """n=64 → 2 blocks, the smallest case where block Jacobi runs."""
    A = _make_spd(4, 64, seed=0)
    D, V = jacobi_eigh_block(A, max_block_sweeps=8, max_inner_sweeps=12)
    _check_eigh(A, D, V)


def test_block_jacobi_n128():
    A = _make_spd(8, 128, seed=1)
    D, V = jacobi_eigh_block(A, max_block_sweeps=8, max_inner_sweeps=12)
    _check_eigh(A, D, V)


@pytest.mark.slow
def test_block_jacobi_n256():
    A = _make_spd(50, 256, seed=2)
    D, V = jacobi_eigh_block(A, max_block_sweeps=8, max_inner_sweeps=12)
    _check_eigh(A, D, V)


def test_block_jacobi_rejects_misaligned_n():
    # n must be a multiple of SUB_DIM = 2 * BLOCK_SIZE = 64.
    # Multiples: 64, 128, 192, 256, …
    A_bad = _make_spd(2, 96, seed=3)  # 96 not multiple of 64
    with pytest.raises(ValueError, match="multiple of SUB_DIM"):
        jacobi_eigh_block(A_bad)
    A_bad = _make_spd(2, 100, seed=4)
    with pytest.raises(ValueError, match="multiple of SUB_DIM"):
        jacobi_eigh_block(A_bad)
    # Multiples of 64 work.
    A_ok = _make_spd(2, 192, seed=5)
    D, V = jacobi_eigh_block(A_ok)
    _check_eigh(A_ok, D, V)


def test_block_jacobi_eigvals_ascending():
    A = _make_spd(3, 64, seed=5)
    D, _ = jacobi_eigh_block(A)
    diffs = np.diff(D, axis=-1)
    assert (diffs >= -1e-3).all()
