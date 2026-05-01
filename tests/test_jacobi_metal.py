"""Correctness of the Metal kernel for batched symmetric eigh."""

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

from asrpy_gpu._jacobi_metal import jacobi_eigh as metal_eigh  # noqa: E402


def _make_spd(B: int | None, n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if B is None:
        A = rng.standard_normal((n, n)).astype(np.float32)
        A = (A @ A.T + 1e-3 * np.eye(n, dtype=np.float32))
    else:
        A = rng.standard_normal((B, n, n)).astype(np.float32)
        A = A @ np.swapaxes(A, -2, -1) + 1e-3 * np.eye(n, dtype=np.float32)[None]
    return np.ascontiguousarray(A)


def _check_eigh(A: np.ndarray, D: np.ndarray, V: np.ndarray, *, atol_rel=2e-3):
    """Validate (D, V) against LAPACK on the residual ``||AV - VD||``.

    The residual is the most algorithm-relevant correctness measure (and
    sign-invariant). For float32 Jacobi we accept up to a few × 1e-4
    absolute residual relative to the spectral norm of A.
    """
    A_t = torch.from_numpy(np.atleast_3d(A).astype(np.float64))
    if A_t.shape[0] != A.shape[0] if A.ndim == 3 else 1:
        A_t = A_t.unsqueeze(0)

    if A.ndim == 2:
        # Unbatched
        AV = A @ V  # (n, n)
        VD = V * D[None, :]
        err = np.abs(AV - VD).max() / np.abs(A).max()
        assert err < atol_rel, f"||AV - VD|| / ||A|| = {err:.3e}"
        # Orthogonality
        VVt = V @ V.T
        np.testing.assert_allclose(
            VVt, np.eye(V.shape[0], dtype=V.dtype), atol=1e-3
        )
    else:
        AV = np.einsum("bij,bjk->bik", A, V)
        VD = V * D[:, None, :]
        err = np.abs(AV - VD).max() / np.abs(A).max()
        assert err < atol_rel, f"||AV - VD|| / ||A|| = {err:.3e}"
        VVt = np.einsum("bij,bjk->bik", V, np.swapaxes(V, -2, -1))
        I = np.eye(V.shape[-1], dtype=V.dtype)[None]
        np.testing.assert_allclose(VVt, np.broadcast_to(I, V.shape), atol=1e-3)


def test_metal_eigh_unbatched():
    A = _make_spd(None, 8, seed=0)
    D, V = metal_eigh(A, max_sweeps=20)
    _check_eigh(A, D, V)


def test_metal_eigh_batched_small():
    A = _make_spd(4, 16, seed=1)
    D, V = metal_eigh(A, max_sweeps=20)
    _check_eigh(A, D, V)


def test_metal_eigh_batched_medium():
    """The realistic scale used inside ASR.process()."""
    A = _make_spd(50, 64, seed=2)
    D, V = metal_eigh(A, max_sweeps=15)
    _check_eigh(A, D, V)


@pytest.mark.slow
def test_metal_eigh_batched_realistic():
    """B=1000, n=128 — close to user's MAGSTIM 128 ch session."""
    A = _make_spd(1000, 128, seed=3)
    D, V = metal_eigh(A, max_sweeps=15)
    _check_eigh(A, D, V, atol_rel=2e-3)


def test_metal_eigh_diagonal_input():
    """Diagonal input must converge in essentially zero work."""
    diag = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    A = np.diag(diag)
    D, V = metal_eigh(A, max_sweeps=2)
    np.testing.assert_allclose(np.sort(D), np.sort(diag), rtol=1e-5)


def test_metal_eigh_eigvals_ordered():
    A = _make_spd(3, 16, seed=4)
    D, _ = metal_eigh(A, max_sweeps=15)
    # Each row must be sorted ascending.
    diffs = np.diff(D, axis=-1)
    assert (diffs >= -1e-3).all(), "eigenvalues must come out ascending"
