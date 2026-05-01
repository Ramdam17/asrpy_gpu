"""Batched symmetric eigendecomposition via parallel Jacobi rotations.

Implements a fully MPS-compatible alternative to ``torch.linalg.eigh``,
which is not (yet) supported on Apple's MPS backend (torch 2.11). Uses
only operations that ARE supported on MPS: matrix multiply, broadcasting,
indexing (``index_select`` / advanced indexing), elementwise arithmetic.

Algorithm — parallel cyclic Jacobi
----------------------------------

The classical cyclic Jacobi method zeros off-diagonal entries one pair
``(p, q)`` at a time, applying a Givens rotation that diagonalises the
2×2 block. After ``O(n²)`` rotations (one *sweep*) the matrix is closer
to diagonal; convergence is roughly quadratic so ``~5–8`` sweeps suffice
for float32 precision.

Naive cyclic Jacobi visits ``n*(n-1)/2`` pairs per sweep sequentially,
which dominates Python launch overhead on the GPU. Parallel Jacobi
(Brent & Luk 1985) covers all pairs in only ``n-1`` *parallel* steps:
each step applies ``n/2`` non-overlapping rotations simultaneously.
Pair scheduling follows a round-robin tournament (one fixed index, the
remaining ``n-1`` indices rotate around a "circle"), which yields
non-conflicting rotation pairs.

References
----------
- Jacobi, C. G. J. (1846). *Ueber ein leichtes Verfahren die in der
  Theorie der Säcularstörungen vorkommenden Gleichungen numerisch
  aufzulösen.* J. Reine Angew. Math.
- Brent, R. P., & Luk, F. T. (1985). *The solution of singular-value
  and symmetric eigenvalue problems on multiprocessor arrays.* SIAM
  J. Sci. Stat. Comput., 6(1), 69–84.
- Demmel, J. (1997). *Applied Numerical Linear Algebra.* SIAM. Chapter 5.
"""

from __future__ import annotations

import torch


def _tournament_pairs(n: int) -> list[list[tuple[int, int]]]:
    """Round-robin tournament schedule for ``n`` (even) indices.

    Returns ``n - 1`` rounds, each with ``n // 2`` pairs of indices that
    do not share any element. Across all rounds, every pair ``(i, j)``
    with ``i < j`` appears exactly once, covering all ``C(n, 2)``.
    """
    if n % 2 != 0:
        # For odd n, append a phantom index n; rounds where a pair
        # contains n are skipped (one pair per round becomes a "bye").
        rounds = _tournament_pairs(n + 1)
        return [
            [(p, q) for p, q in rnd if p != n and q != n] for rnd in rounds
        ]

    indices = list(range(n))
    fixed = indices[0]
    rotating = indices[1:]
    rounds: list[list[tuple[int, int]]] = []
    for _ in range(n - 1):
        pairs = [(fixed, rotating[0])]
        # Pair the rest in mirror.
        for i in range(1, n // 2):
            a, b = rotating[i], rotating[n - 1 - i]
            pairs.append((min(a, b), max(a, b)))
        rounds.append(pairs)
        # Rotate by one position to the right.
        rotating = rotating[-1:] + rotating[:-1]
    return rounds


def _apply_givens_batched(
    A: torch.Tensor,
    V: torch.Tensor,
    ps: torch.Tensor,
    qs: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``n//2`` non-overlapping Givens rotations in parallel.

    Updates ``A`` (in-place via copies) and ``V``. ``ps`` and ``qs`` are
    1-D LongTensors of length ``k = n // 2`` containing the row/column
    indices for each rotation. ``c, s`` have shape ``(B, k)``.
    """
    B, n, _ = A.shape

    # Read the rows we are about to mix BEFORE writing anywhere — otherwise
    # the second write would see the first write's output.
    A_p_rows = A.index_select(1, ps).clone()  # (B, k, n)
    A_q_rows = A.index_select(1, qs).clone()  # (B, k, n)

    c_e = c.unsqueeze(-1)
    s_e = s.unsqueeze(-1)
    new_p_rows = c_e * A_p_rows - s_e * A_q_rows
    new_q_rows = s_e * A_p_rows + c_e * A_q_rows

    A.index_copy_(1, ps, new_p_rows)
    A.index_copy_(1, qs, new_q_rows)

    # Now the symmetric column update — read first, write second.
    A_p_cols = A.index_select(2, ps).clone()  # (B, n, k)
    A_q_cols = A.index_select(2, qs).clone()
    c_e2 = c.unsqueeze(1)
    s_e2 = s.unsqueeze(1)
    new_p_cols = c_e2 * A_p_cols - s_e2 * A_q_cols
    new_q_cols = s_e2 * A_p_cols + c_e2 * A_q_cols
    A.index_copy_(2, ps, new_p_cols)
    A.index_copy_(2, qs, new_q_cols)

    # V columns (one-sided rotation).
    V_p = V.index_select(2, ps).clone()
    V_q = V.index_select(2, qs).clone()
    new_V_p = c_e2 * V_p - s_e2 * V_q
    new_V_q = s_e2 * V_p + c_e2 * V_q
    V.index_copy_(2, ps, new_V_p)
    V.index_copy_(2, qs, new_V_q)

    return A, V


def jacobi_eigh(
    A: torch.Tensor,
    *,
    max_sweeps: int = 12,
    tol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched symmetric eigendecomposition via parallel Jacobi.

    Parameters
    ----------
    A : Tensor, shape (B, n, n) or (n, n)
        Symmetric input. The input is **not** modified (we work on a
        copy). Must be float32 or float64. ``A`` is assumed symmetric;
        upper / lower triangle inconsistency is silently averaged out by
        the algorithm but, for reproducibility, callers should pass a
        properly symmetric tensor.
    max_sweeps : int, optional
        Maximum number of full sweeps. ``8`` is enough for float32 with
        well-conditioned inputs; we default to ``12`` for safety.
    tol : float, optional
        Convergence threshold on the off-diagonal Frobenius norm
        (relative to the diagonal). When ``None``, uses
        ``8 * eps * ||A||_F`` for the dtype.

    Returns
    -------
    eigvals : Tensor, shape (B, n)
        Eigenvalues, sorted ascending.
    eigvecs : Tensor, shape (B, n, n)
        Eigenvectors as columns.

    Notes
    -----
    Float32 accuracy is around ``rtol=1e-5`` for the eigenvalues and
    ``rtol=1e-3`` for the eigenvectors of well-separated spectra.
    Sign-invariant quantities (``A V = V D``, ``V V.T = I``) match LAPACK
    to better than ``1e-5`` in float32.
    """
    if A.ndim == 2:
        A = A.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B, n, _ = A.shape
    if A.shape[-2] != A.shape[-1]:
        raise ValueError(f"jacobi_eigh requires square matrices, got {A.shape}")

    A = A.clone()
    V = (
        torch.eye(n, dtype=A.dtype, device=A.device)
        .unsqueeze(0)
        .expand(B, n, n)
        .contiguous()
    )

    # Pre-compute the tournament schedule — independent of B.
    rounds = _tournament_pairs(n)
    rounds_idx = [
        (
            torch.tensor([p for p, _ in rnd], dtype=torch.long, device=A.device),
            torch.tensor([q for _, q in rnd], dtype=torch.long, device=A.device),
        )
        for rnd in rounds
    ]

    if tol is None:
        eps = torch.finfo(A.dtype).eps
        # Frobenius norm scales with ||A||_F; relative target.
        a_fro = torch.linalg.matrix_norm(A, ord="fro")  # (B,)
        target = 8 * eps * a_fro
    else:
        target = torch.full((B,), tol, dtype=A.dtype, device=A.device)

    safe_eps = torch.tensor(torch.finfo(A.dtype).tiny, dtype=A.dtype, device=A.device)

    for _sweep in range(max_sweeps):
        for ps, qs in rounds_idx:
            # Pull diagonal-block entries for this round of pairs.
            # gather along last two dims.
            B_idx = torch.arange(B, device=A.device).unsqueeze(-1)
            a_pp = A[B_idx, ps.unsqueeze(0), ps.unsqueeze(0)]  # (B, k)
            a_qq = A[B_idx, qs.unsqueeze(0), qs.unsqueeze(0)]
            a_pq = A[B_idx, ps.unsqueeze(0), qs.unsqueeze(0)]

            # Stable Givens (c, s):
            #   theta = (a_qq - a_pp) / (2 * a_pq)
            #   t = sign(theta) / (|theta| + sqrt(theta^2 + 1))
            #   c = 1 / sqrt(1 + t^2),  s = t * c
            small = a_pq.abs() < safe_eps
            denom = torch.where(small, safe_eps, 2 * a_pq)
            theta = (a_qq - a_pp) / denom
            t = torch.sign(theta) / (theta.abs() + torch.sqrt(theta * theta + 1))
            t = torch.where(small, torch.zeros_like(t), t)
            c = 1.0 / torch.sqrt(1 + t * t)
            s = t * c

            A, V = _apply_givens_batched(A, V, ps, qs, c, s)

        # Convergence test: off-diagonal Frobenius norm per matrix.
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        a_fro_sq = (A * A).sum(dim=(-2, -1))
        d_fro_sq = (diag * diag).sum(dim=-1)
        off = torch.sqrt(torch.clamp(a_fro_sq - d_fro_sq, min=0.0))
        if (off < target).all():
            break

    # Eigenvalues = diagonal; sort ascending.
    D = torch.diagonal(A, dim1=-2, dim2=-1).clone()
    sort_idx = torch.argsort(D, dim=-1)
    D = torch.gather(D, -1, sort_idx)
    V = torch.gather(V, -1, sort_idx.unsqueeze(-2).expand_as(V))

    if squeeze_out:
        return D.squeeze(0), V.squeeze(0)
    return D, V
