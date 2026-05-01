"""torch backend for ASR.

Mirrors the public API of :mod:`asrpy_gpu._backend_numpy` (``calibrate``,
``process``, ``clean_windows``) using ``torch`` for the heavy linear-algebra
operations. Targets Apple MPS first (the project's primary use-case), but
also runs on CUDA or CPU.

Design notes
------------

**Where the GPU helps.** The big win is in :func:`process`: the original
asrpy loops in Python over windows and calls ``np.linalg.eigh`` once per
window. Here we stack the per-window covariances and do a single
``torch.linalg.eigh`` on the batched tensor. Reconstruction matrices ``R``
are also computed batched.

**Where we stay on CPU.** Two pieces stay on CPU:

* ``scipy.signal.lfilter`` (the Yule-Walker IIR) is sequential; we keep it
  on CPU and only push the filtered signal back to the GPU. V5 of the
  roadmap covers a Metal parallel-scan IIR if profiling later shows this
  transfer is the residual bottleneck.
* ``torch.linalg.eigh`` is **not implemented on MPS** as of torch 2.11
  (op ``aten::_linalg_eigh.eigenvalues``). We handle this explicitly: each
  ``eigh`` call moves its input to CPU, runs in **float64** (better
  precision than the float32 we use on MPS), and moves the result back.
  ``torch.linalg.pinv`` uses SVD internally and silently falls back to CPU
  on MPS — we accept this and document it. Once MPS gains native eigh,
  the helper :func:`_eigh_native_or_cpu` can be specialised in V2.

**Precision.** MPS only supports float32; CPU and CUDA support float64. The
backend picks the dtype based on the resolved device; tests use the
tolerance table in ``config/test_tolerances.yaml``. The cleaned signal is
cast back to float64 at the boundary so that downstream MNE pipelines see
the usual dtype.

**Sign invariance.** ASR is structurally invariant to eigenvector sign
flips (the algorithm only uses ``(T @ V)**2`` and ``V @ … @ V.T``-style
products). We therefore do not align signs across implementations — only
``M``, ``T`` and the cleaned signal are compared in tests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from . import _backend_numpy as _np_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------


def _torch_dtype_for(device: str) -> torch.dtype:
    """MPS is float32-only; CPU and CUDA may use float64."""
    return torch.float32 if device == "mps" else torch.float64


def _to_torch(arr: np.ndarray, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _to_numpy(t: torch.Tensor, *, dtype: np.dtype = np.float64) -> np.ndarray:
    return t.detach().to("cpu").to(dtype=torch.float64).numpy().astype(dtype)


def _sync(device: str) -> None:
    """Synchronize the device — required before timing benchmarks."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


_EIGH_STRATEGY: str = "auto"


def set_eigh_strategy(strategy: str) -> None:
    """Override the eigh strategy used by the torch backend.

    Valid strategies:

    * ``"auto"`` — pick the best available for the current device. MPS →
      ``"jacobi_metal"`` if pyobjc-Metal is installed, else
      ``"cpu_fallback"``. CUDA / CPU → ``"native"``.
    * ``"native"`` — call ``torch.linalg.eigh`` directly (fails on MPS).
    * ``"cpu_fallback"`` — move to CPU, compute in float64, move back.
    * ``"jacobi_torch"`` — pure-torch parallel Jacobi (portable; correct
      but not competitive on MPS due to Python kernel-launch overhead).
    * ``"jacobi_metal"`` — Metal compute kernel batched Jacobi (MPS only).
    """
    valid = {"auto", "native", "cpu_fallback", "jacobi_torch", "jacobi_metal"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {valid}, got {strategy!r}")
    global _EIGH_STRATEGY
    _EIGH_STRATEGY = strategy


def get_eigh_strategy() -> str:
    return _EIGH_STRATEGY


def _resolve_eigh_strategy(device_type: str) -> str:
    if _EIGH_STRATEGY != "auto":
        return _EIGH_STRATEGY
    if device_type == "mps":
        from ._jacobi_metal import METAL_AVAILABLE

        return "jacobi_metal" if METAL_AVAILABLE else "cpu_fallback"
    return "native"


def _eigh_cpu_fallback(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if A.device.type == "mps":
        A_cpu64 = A.detach().to("cpu").to(torch.float64)
    else:
        A_cpu64 = A.detach().to(torch.float64)
    D_cpu, V_cpu = torch.linalg.eigh(A_cpu64)
    D = D_cpu.to(A.dtype).to(A.device)
    V = V_cpu.to(A.dtype).to(A.device)
    return D, V


def _eigh_jacobi_torch_strategy(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ._jacobi_torch import jacobi_eigh

    return jacobi_eigh(A)


def _eigh_jacobi_metal_strategy(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hybrid Metal dispatch: block Jacobi for n ≥ 256, std Jacobi otherwise.

    Block Jacobi (V3.2B, double-tiled apply-Q) wins on n ≥ 256 by
    1.5–1.75× over the std cyclic Jacobi (because there are enough
    block-pairs to amortise the per-pair fixed cost). For smaller n the
    fixed cost dominates and std Jacobi is faster — the dispatch
    chooses the right kernel per call.
    """
    from ._jacobi_metal import jacobi_eigh as metal_eigh

    A_np = A.detach().to("cpu").to(torch.float32).numpy()
    n = A_np.shape[-1]
    SUB_DIM = 64  # 2 * BLOCK_SIZE in _jacobi_metal_block

    if n >= 256 and n % SUB_DIM == 0:
        from ._jacobi_metal_block import jacobi_eigh_block

        D_np, V_np = jacobi_eigh_block(A_np)
    else:
        D_np, V_np = metal_eigh(A_np)

    D = torch.from_numpy(D_np).to(A.device).to(A.dtype)
    V = torch.from_numpy(V_np).to(A.device).to(A.dtype)
    return D, V


def _eigh_native_or_cpu(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hermitian eigendecomposition with the configured strategy.

    See :func:`set_eigh_strategy`.
    """
    strategy = _resolve_eigh_strategy(A.device.type)
    if strategy == "native":
        return torch.linalg.eigh(A)
    if strategy == "cpu_fallback":
        return _eigh_cpu_fallback(A)
    if strategy == "jacobi_torch":
        return _eigh_jacobi_torch_strategy(A)
    if strategy == "jacobi_metal":
        return _eigh_jacobi_metal_strategy(A)
    raise RuntimeError(f"Unknown eigh strategy {strategy!r}")


def pinv_via_eigh(A: torch.Tensor, *, rcond: float | None = None) -> torch.Tensor:
    """Moore-Penrose pseudoinverse of square ``A`` via eigh of ``A.T @ A``.

    For a square matrix ``A`` (potentially rank-deficient),

        pinv(A) = V diag(1/D⁺) Vᵀ Aᵀ

    where ``V, D = eigh(Aᵀ A)`` and ``1/D⁺`` zeroes out eigenvalues that
    fall below ``rcond * max(D)``. ``Aᵀ A`` is symmetric positive
    semi-definite, so it is the perfect input for our Jacobi-Metal eigh
    kernel — using this formula keeps the whole computation on MPS,
    bypassing the silent CPU-SVD fallback that ``torch.linalg.pinv``
    triggers on Apple's MPS backend.

    Parameters
    ----------
    A : Tensor, shape (..., n, n)
        Batched square matrix. Must be float32 or float64.
    rcond : float, optional
        Relative cutoff for the small eigenvalues; defaults to
        ``max(n) * eps`` for the dtype.

    Returns
    -------
    P : Tensor, shape (..., n, n)
        Pseudoinverse, same dtype and device as ``A``.

    Notes
    -----
    For numerical accuracy this routine is comparable to ``torch.linalg.
    pinv`` for well-conditioned matrices and slightly worse for very
    rank-deficient / ill-conditioned ones (the squaring ``Aᵀ A`` doubles
    the condition number). For our use case in :func:`process` ``A`` is
    a row-masked rotation of an SPD matrix, so its non-zero singular
    values stay well-separated from zero.
    """
    if rcond is None:
        rcond = A.shape[-1] * torch.finfo(A.dtype).eps

    # B = Aᵀ A — symmetric positive semi-definite.
    B = A.transpose(-2, -1) @ A
    D, V = _eigh_native_or_cpu(B)  # batched-aware

    # tol per-batch = rcond * max(D, dim=-1)
    tol = rcond * D.amax(dim=-1, keepdim=True)
    D_inv = torch.where(
        D > tol, 1.0 / D.clamp(min=torch.finfo(A.dtype).tiny), torch.zeros_like(D)
    )

    # pinv = V diag(D_inv) Vᵀ Aᵀ
    return V @ (D_inv.unsqueeze(-1) * V.transpose(-2, -1)) @ A.transpose(-2, -1)


# ---------------------------------------------------------------------------
# GPU-side primitives
# ---------------------------------------------------------------------------


def _sqrtm_spd_torch(A: torch.Tensor) -> torch.Tensor:
    """Matrix square root for symmetric positive-(semi)definite ``A``.

    For SPD ``A = V diag(D) V.T``, ``sqrtm(A) = V diag(sqrt(D)) V.T`` is the
    principal square root. Tiny negative eigenvalues from numerical noise
    are clamped to zero. Uses :func:`_eigh_native_or_cpu` so it works on MPS.
    """
    D, V = _eigh_native_or_cpu(A)
    D = torch.clamp(D, min=0.0)
    return V @ torch.diag(torch.sqrt(D)) @ V.T


def _block_covariance_torch(data: torch.Tensor, window: int) -> torch.Tensor:
    """GPU port of :func:`asrpy_gpu._backend_numpy._block_covariance`.

    Returns a tensor of shape ``(n_blocks, n_channels**2)``.
    """
    n_ch, n_times = data.shape
    n_blocks = len(np.arange(0, n_times - 1, window))

    U = torch.zeros((n_blocks, n_ch * n_ch), dtype=data.dtype, device=data.device)
    data_T = data.T  # (n_times, n_ch)

    for k in range(window):
        idx = np.minimum(n_times - 1, np.arange(k, n_times + k - 2, window))
        idx_t = torch.as_tensor(idx, device=data.device, dtype=torch.long)
        block = data_T.index_select(0, idx_t)  # (n_blocks, n_ch)
        outer = block.unsqueeze(2) * block.unsqueeze(1)  # (n_blocks, n_ch, n_ch)
        U = U + outer.reshape(n_blocks, n_ch * n_ch)
    return U


def _geometric_median_torch(
    X: torch.Tensor, tol: float = 1e-5, max_iter: int = 500
) -> torch.Tensor:
    """Vardi-Zhang geometric median (Weiszfeld) on torch tensors.

    Same algorithm as :func:`asrpy_gpu._backend_numpy._geometric_median`. The
    convergence test forces a small device→host sync via ``.item()``; this
    is acceptable because typical convergence is < 100 iterations.

    References
    ----------
    Vardi, Y., & Zhang, C. H. (2000). The multivariate L1-median and
    associated data depth. PNAS 97(4), 1423-1426.
    """
    y = X.mean(dim=0)
    n_obs = X.shape[0]

    for _ in range(max_iter):
        D = torch.linalg.vector_norm(X - y, dim=1)  # (n_obs,)
        nonzero = D != 0
        D_nz = D[nonzero]

        Dinv = 1.0 / D_nz
        Dinvs = Dinv.sum()
        W = (Dinv / Dinvs).unsqueeze(1)
        T = (W * X[nonzero]).sum(dim=0)

        n_zeros = n_obs - int(nonzero.sum().item())
        if n_zeros == 0:
            y1 = T
        elif n_zeros == n_obs:
            return y
        else:
            R = (T - y) * Dinvs
            r = torch.linalg.vector_norm(R)
            rinv = 0.0 if r.item() == 0.0 else n_zeros / r.item()
            y1 = max(0.0, 1.0 - rinv) * T + min(1.0, rinv) * y

        if torch.linalg.vector_norm(y - y1).item() < tol:
            return y1
        y = y1

    logger.warning(
        "Geometric median did not converge in %d iterations (tol=%g).",
        max_iter,
        tol,
    )
    return y


def _ma_filter_torch(
    N: int, X: torch.Tensor, Zi: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Streaming moving-average via cumsum, on torch tensors.

    Direct port of :func:`asrpy_gpu._backend_numpy._ma_filter`.
    """
    if Zi is None:
        Zi = torch.zeros(X.shape[0], N, dtype=X.dtype, device=X.device)

    Y = torch.cat([Zi, X], dim=-1)  # (n_ch, N + n_samples)
    M = Y.shape[-1]

    # Build interleaved index pattern -1/+1 / N as in the numpy version.
    I_a = torch.arange(M - N, device=X.device)
    I_b = torch.arange(N, M, device=X.device)
    idx = torch.empty(2 * (M - N), dtype=torch.long, device=X.device)
    idx[0::2] = I_a
    idx[1::2] = I_b
    sign = torch.empty(2 * (M - N), dtype=X.dtype, device=X.device)
    sign[0::2] = -1.0 / N
    sign[1::2] = 1.0 / N

    Xc = torch.cumsum(Y[:, idx] * sign, dim=-1)
    Xc = Xc[:, 1::2]  # take every other column starting at 1

    Zf_first = -(Xc[:, -1] * N - Y[:, -N]).unsqueeze(1)
    Zf = torch.cat([Zf_first, Y[:, -N + 1 :]], dim=-1)
    return Xc, Zf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate(
    X: np.ndarray,
    sfreq: float,
    *,
    cutoff: float = 20.0,
    blocksize: int = 100,
    win_len: float = 0.5,
    win_overlap: float = 0.66,
    max_dropout_fraction: float = 0.1,
    min_clean_fraction: float = 0.25,
    ab: tuple[np.ndarray, np.ndarray] | None = None,
    device: str = "mps",
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate ASR on the GPU. See :func:`_backend_numpy.calibrate`."""
    dtype = _torch_dtype_for(device)
    n_channels, n_samples = X.shape

    # Yule-Walker IIR filter — stays on CPU (sequential).
    X_filt_np, _ = _np_backend._yulewalk_filter(X, sfreq, ab=ab)

    # Push filtered signal to GPU.
    X_t = _to_torch(X_filt_np, device=device, dtype=dtype)

    N = int(np.round(win_len * sfreq))

    # Block covariance.
    U = _block_covariance_torch(X_t, window=blocksize)
    Uavg_flat = _geometric_median_torch(U / blocksize)
    Uavg = Uavg_flat.reshape(n_channels, n_channels)

    # Mixing matrix M = sqrtm(Uavg). Uavg is SPD by construction here.
    M_t = _sqrtm_spd_torch(Uavg)

    # Eigendecomposition (already sorted ascending by torch's eigh).
    D, Vtmp = _eigh_native_or_cpu(M_t)
    sort_idx = torch.argsort(D)
    V_t = Vtmp[:, sort_idx]

    # |V.T @ X|, per-channel grid-search distribution fit.
    abs_proj = (V_t.T @ X_t).abs()  # (n_channels, n_samples)
    abs_proj_np = _to_numpy(abs_proj)  # back to CPU for fit_eeg_distribution

    offsets = np.int_(np.arange(0, n_samples - N, np.round(N * (1 - win_overlap))))
    mu = np.zeros(n_channels)
    sig = np.zeros(n_channels)
    for ichan in reversed(range(n_channels)):
        rms = abs_proj_np[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(rms[o : o + N]) / N) for o in offsets])
        mu[ichan], sig[ichan], _, _ = _np_backend._fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction
        )

    # T = diag(mu + cutoff * sig) @ V.T  — back on CPU as float64 for return.
    diag_t = torch.as_tensor(mu + cutoff * sig, dtype=dtype, device=device)
    T_t = torch.diag(diag_t) @ V_t.T

    return _to_numpy(M_t), _to_numpy(T_t)


def process(
    data: np.ndarray,
    sfreq: float,
    M: np.ndarray,
    T: np.ndarray,
    *,
    win_len: float = 0.5,
    lookahead: float = 0.25,
    stepsize: int = 32,
    maxdims: float | int = 0.66,
    ab: tuple[np.ndarray, np.ndarray] | None = None,
    R: np.ndarray | None = None,
    Zi: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    carry: np.ndarray | None = None,
    return_states: bool = False,
    mem_splits: int = 3,
    device: str = "mps",
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Apply ASR cleaning on the GPU. See :func:`_backend_numpy.process`."""
    dtype = _torch_dtype_for(device)
    n_channels, n_samples = data.shape

    if maxdims < 1:
        maxdims = int(np.round(n_channels * maxdims))

    # Initial Yule-Walker state — CPU.
    if Zi is None:
        _, Zi = _np_backend._yulewalk_filter(
            data, ab=ab, sfreq=sfreq, zi=np.ones([n_channels, 8])
        )

    N = int(np.round(win_len * sfreq))
    P = int(np.round(lookahead * sfreq))

    # Reflect-pad the start (asrpy convention).
    if carry is None:
        carry = (
            np.tile(2 * data[:, 0], (P, 1)).T
            - data[:, np.mod(np.arange(P, 0, -1), n_samples)]
        )
    data = np.concatenate([carry, data], axis=-1)

    # We work in-place on a torch tensor to avoid CPU/GPU ping-pong.
    data_t = _to_torch(data, device=device, dtype=dtype)
    M_t = _to_torch(M, device=device, dtype=dtype)
    T_t = _to_torch(T, device=device, dtype=dtype)
    cov_t: torch.Tensor | None = (
        _to_torch(cov, device=device, dtype=dtype) if cov is not None else None
    )

    splits = mem_splits
    last_trivial = False
    last_R: torch.Tensor | None = None
    R_out: torch.Tensor | None = None

    eye_C = torch.eye(n_channels, dtype=dtype, device=device)

    for i in range(splits):
        i_range = np.arange(
            i * n_samples // splits,
            min((i + 1) * n_samples // splits, n_samples),
            dtype=int,
        )

        # Yule-Walker filter on CPU (sequential).
        chunk_np = data_t[:, i_range + P].detach().to("cpu").numpy().astype(np.float64)
        X_filt_np, Zi = _np_backend._yulewalk_filter(
            chunk_np, sfreq=sfreq, zi=Zi, ab=ab, axis=-1
        )
        X = _to_torch(X_filt_np, device=device, dtype=dtype)

        # Outer-product time series.
        XX = (X.unsqueeze(0) * X.unsqueeze(1)).reshape(n_channels * n_channels, -1)
        Xcov_flat, cov_t = _ma_filter_torch(N, XX, cov_t)

        update_at = np.arange(stepsize, Xcov_flat.shape[-1] + stepsize - 2, stepsize)
        update_at = np.minimum(update_at, Xcov_flat.shape[-1]) - 1

        if last_R is None:
            update_at = np.concatenate([[0], update_at])
            last_R = eye_C.clone()

        # Stack window covariances: (n_windows, C, C).
        Xcov = (
            Xcov_flat[:, update_at]
            .reshape(n_channels, n_channels, -1)
            .permute(2, 0, 1)
            .contiguous()
        )

        # Batched eigh — the main GPU win. Uses CPU fallback on MPS (see
        # _eigh_native_or_cpu); even so, batched eigh on CPU outperforms a
        # Python loop of per-window eigh calls thanks to BLAS amortisation.
        D_all, V_all = _eigh_native_or_cpu(Xcov)  # (W, C), (W, C, C)

        # keep[w, c] = (D[w, c] < sum((T @ V)^2 along axis=0))
        TV = T_t @ V_all  # (W, C, C)
        thresh = (TV**2).sum(dim=1)  # (W, C)
        idx_C = torch.arange(n_channels, device=device).unsqueeze(0)
        keep = (D_all < thresh) | (idx_C + 1 < (n_channels - maxdims))
        # trivial[w] is True when ALL components are kept (R = I).
        trivial_all = keep.all(dim=1)  # (W,)

        # Skip-trivial optimisation: we only need to compute R for windows
        # where keep is NOT all True.
        non_trivial = ~trivial_all
        n_windows = D_all.shape[0]

        # Default R = I for every window.
        R_all = (
            eye_C.unsqueeze(0).expand(n_windows, n_channels, n_channels).contiguous()
        )

        if non_trivial.any():
            nt_idx = non_trivial.nonzero(as_tuple=False).squeeze(-1)
            keep_nt = keep[nt_idx]  # (W_nt, C)
            V_nt = V_all[nt_idx]  # (W_nt, C, C)
            VtM_nt = V_nt.transpose(-2, -1) @ M_t  # (W_nt, C, C)
            masked_nt = keep_nt.unsqueeze(-1) * VtM_nt  # (W_nt, C, C)

            if device == "mps":
                pinv_nt = pinv_via_eigh(masked_nt)
            else:
                pinv_nt = torch.linalg.pinv(masked_nt)
            R_nt = M_t @ pinv_nt @ V_nt.transpose(-2, -1)
            R_all = R_all.index_copy(0, nt_idx, R_nt)

        # Sequential blending loop (intrinsically sequential — depends on
        # last_R). Operations stay on device; only Python orchestrates.
        # We pre-fetch the trivial_all flag array to CPU once (one sync) and
        # iterate over the numpy view, instead of `.item()`-syncing each
        # iteration. For B≈300 windows that turns 300 mps→cpu syncs into 1.
        trivial_arr = trivial_all.cpu().numpy() if trivial_all.is_mps or trivial_all.is_cuda else trivial_all.numpy()
        last_n = 0
        for j in range(len(update_at) - 1):
            trivial = bool(trivial_arr[j])
            R_j = eye_C if trivial else R_all[j]

            n = int(update_at[j]) + 1
            if (not trivial) or (not last_trivial):
                subrange = i_range[np.arange(last_n, n)]
                blend_x = np.pi * np.arange(1, n - last_n + 1) / (n - last_n)
                blend_np = (1 - np.cos(blend_x)) / 2
                blend = _to_torch(blend_np, device=device, dtype=dtype)

                tmp = data_t[:, subrange]
                data_t[:, subrange] = blend * (R_j @ tmp) + (1 - blend) * (last_R @ tmp)

            last_n, last_R, last_trivial, R_out = n, R_j, trivial, R_j

    # Strip the carry padding and return numpy float64.
    cleaned = _to_numpy(data_t[:, :-P])

    if return_states:
        carry_out = np.concatenate([carry, _to_numpy(data_t[:, -P:])], axis=-1)
        carry_out = carry_out[:, -P:]
        state = {
            "M": M,
            "T": T,
            "R": _to_numpy(R_out) if R_out is not None else None,
            "Zi": Zi,
            "cov": _to_numpy(cov_t) if cov_t is not None else None,
            "carry": carry_out,
        }
        return cleaned, state
    return cleaned


def clean_windows(
    X: np.ndarray,
    sfreq: float,
    *,
    max_bad_chans: float = 0.2,
    zthresholds: tuple[float, float] = (-3.5, 5.0),
    win_len: float = 0.5,
    win_overlap: float = 0.66,
    min_clean_fraction: float = 0.25,
    max_dropout_fraction: float = 0.1,
    device: str = "mps",  # noqa: ARG001 — kept for API symmetry
) -> tuple[np.ndarray, np.ndarray]:
    """Window-based bad-data rejection.

    The grid-search ``fit_eeg_distribution`` is sequential per channel and
    not memory-bound; it stays on CPU. Pushing it to GPU does not help and
    would only add transfer cost. We delegate to the numpy backend.
    """
    return _np_backend.clean_windows(
        X,
        sfreq=sfreq,
        max_bad_chans=max_bad_chans,
        zthresholds=zthresholds,
        win_len=win_len,
        win_overlap=win_overlap,
        min_clean_fraction=min_clean_fraction,
        max_dropout_fraction=max_dropout_fraction,
    )
