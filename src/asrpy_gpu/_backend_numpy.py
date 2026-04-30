"""Numpy reference backend for ASR.

This module is a faithful port of `asrpy <https://github.com/DiGyt/asrpy>`__
(BSD-3, Dirk Gütlin & Nicolas Barascud) into the asrpy_gpu package. It is the
**reference implementation**: clarity over speed, used for numerical
equivalence tests against ``asrpy`` and as the ground truth for the torch
backend.

The algorithm is documented in:

- Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S.,
  Jung, T.-P., & Cauwenberghs, G. (2015). *Real-time Neuroimaging and Cognitive
  Monitoring Using Wearable Dry EEG.* IEEE Trans Biomed Eng, 62(11), 2553–2567.
- Kothe, C. A. E., & Jung, T.-P. (2016). *U.S. Patent Application No.
  14/895,440.*

Riemannian variant (not in V1):

- Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019).
  *A Riemannian Modification of Artifact Subspace Reconstruction for EEG
  Artifact Handling.* Frontiers in Human Neuroscience, 13.
  https://doi.org/10.3389/fnhum.2019.00141
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.linalg import pinv
from scipy import linalg, signal
from scipy.linalg import toeplitz
from scipy.spatial.distance import cdist, euclidean
from scipy.special import gamma, gammaincinv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate(
    X: np.ndarray,
    sfreq: float,
    cutoff: float = 20.0,
    blocksize: int = 100,
    win_len: float = 0.5,
    win_overlap: float = 0.66,
    max_dropout_fraction: float = 0.1,
    min_clean_fraction: float = 0.25,
    ab: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate ASR on clean reference data (Euclidean variant).

    Direct port of ``asrpy.asr_calibrate``. See module docstring for the
    references.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        High-pass filtered, reasonably clean calibration data (typically
        ≥ 30 s of resting EEG).
    sfreq : float
        Sampling frequency, in Hz.
    cutoff : float
        Standard-deviation cutoff for artifact rejection. Lower is more
        aggressive. Defaults to 20 (recommended; the original EEGLAB default
        of 5 is widely considered too aggressive for research data).
    blocksize : int
        Block size used to estimate the robust block covariance.
    win_len : float
        Window length in seconds used to compute thresholds.
    win_overlap : float
        Fractional overlap between successive windows.
    max_dropout_fraction : float
        Maximum fraction of windows allowed to be near-zero (e.g., unplugged
        electrodes).
    min_clean_fraction : float
        Minimum fraction of windows that must be clean.
    ab : 2-tuple of ndarray, optional
        ``(A, B)`` IIR coefficients of the spectrum-shaping Yule-Walker
        filter. If ``None``, defaults reproducing EEGLAB's choice are built.

    Returns
    -------
    M : ndarray, shape (n_channels, n_channels)
        Mixing matrix.
    T : ndarray, shape (n_channels, n_channels)
        Threshold matrix.
    """
    n_channels, n_samples = X.shape
    logger.debug("[ASR-numpy] Calibrating: shape=%s sfreq=%g", X.shape, sfreq)

    # Spectrum-shaping IIR filter (Yule-Walker).
    X, _zf = _yulewalk_filter(X, sfreq, ab=ab)

    # Window length in samples for threshold estimation.
    N = int(np.round(win_len * sfreq))

    # Block-wise covariance.
    U = _block_covariance(X, window=blocksize)

    # Robust covariance via geometric median.
    Uavg = _geometric_median(U.reshape(-1, n_channels * n_channels) / blocksize)
    Uavg = Uavg.reshape(n_channels, n_channels)

    # Mixing matrix M = sqrtm(Uavg). Uavg is SPD here, so sqrtm is well-defined.
    M = linalg.sqrtm(np.real(Uavg))

    # Eigendecomposition of M (sorted ascending).
    D, Vtmp = linalg.eigh(M)
    V = Vtmp[:, np.argsort(D)]

    # Threshold matrix T: per-channel mean+cutoff*std of the rectified
    # eigen-projected signal.
    x = np.abs(V.T @ X)
    offsets = np.int_(np.arange(0, n_samples - N, np.round(N * (1 - win_overlap))))

    mu = np.zeros(n_channels)
    sig = np.zeros(n_channels)
    for ichan in reversed(range(n_channels)):
        rms = x[ichan, :] ** 2
        Y = np.array(
            [np.sqrt(np.sum(rms[o : o + N]) / N) for o in offsets]
        )
        mu[ichan], sig[ichan], _, _ = _fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction
        )

    T = np.diag(mu + cutoff * sig) @ V.T
    logger.debug("[ASR-numpy] Calibration done.")
    return M, T


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
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Apply ASR cleaning to a continuous data array.

    Direct port of ``asrpy.asr_process``. The algorithm slides a window over
    the data, performs an eigendecomposition of the local covariance,
    projects out components whose variance exceeds the calibrated threshold,
    and reconstructs the signal with cosine-blended boundaries.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Continuous data to clean. Should be the same montage as the
        calibration data.
    sfreq : float
        Sampling frequency, in Hz.
    M, T : ndarray, shape (n_channels, n_channels)
        Mixing and threshold matrices from :func:`calibrate`.
    win_len : float
        Window length in seconds.
    lookahead : float
        Look-ahead in seconds. Recommended: ``win_len / 2``.
    stepsize : int
        Update stride, in samples.
    maxdims : float or int
        If < 1, fraction of channels that may be removed per window. If ≥ 1,
        absolute number.
    ab : 2-tuple of ndarray, optional
        IIR coefficients matching :func:`calibrate`.
    R, Zi, cov, carry : ndarray, optional
        Streaming-mode state from a previous call.
    return_states : bool
        If True, also return a dict of streaming states.
    mem_splits : int
        Number of memory-saving segments to split the input into.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_samples)
        Cleaned data.
    state : dict, only if ``return_states`` is True
    """
    if maxdims < 1:
        maxdims = np.round(len(data) * maxdims)

    if Zi is None:
        _, Zi = _yulewalk_filter(
            data, ab=ab, sfreq=sfreq, zi=np.ones([len(data), 8])
        )

    n_channels, n_samples = data.shape
    N = np.round(win_len * sfreq).astype(int)
    P = np.round(lookahead * sfreq).astype(int)

    # Reflect-pad the start to avoid edge artefacts; this is the asrpy
    # convention (see asrpy/asr.py:574-576).
    if carry is None:
        carry = (
            np.tile(2 * data[:, 0], (P, 1)).T
            - data[:, np.mod(np.arange(P, 0, -1), n_samples)]
        )
    data = np.concatenate([carry, data], axis=-1)

    splits = mem_splits

    last_trivial = False
    last_R: np.ndarray | None = None
    R_out: np.ndarray | None = None

    for i in range(splits):
        i_range = np.arange(
            i * n_samples // splits,
            min((i + 1) * n_samples // splits, n_samples),
            dtype=int,
        )

        X, Zi = _yulewalk_filter(
            data[:, i_range + P], sfreq=sfreq, zi=Zi, ab=ab, axis=-1
        )

        # Moving-average covariance (vectorised).
        Xcov, cov = _ma_filter(
            N,
            np.reshape(
                np.multiply(
                    np.reshape(X, (1, n_channels, -1)),
                    np.reshape(X, (n_channels, 1, -1)),
                ),
                (n_channels * n_channels, -1),
            ),
            cov,
        )

        update_at = np.arange(stepsize, Xcov.shape[-1] + stepsize - 2, stepsize)
        update_at = np.minimum(update_at, Xcov.shape[-1]) - 1

        if last_R is None:
            update_at = np.concatenate([[0], update_at])
            last_R = np.eye(n_channels)

        Xcov = np.reshape(Xcov[:, update_at], (n_channels, n_channels, -1))

        last_n = 0
        for j in range(len(update_at) - 1):
            D, V = np.linalg.eigh(Xcov[:, :, j])

            keep = np.logical_or(
                D < np.sum((T @ V) ** 2, axis=0),
                np.arange(n_channels) + 1 < (n_channels - maxdims),
            )
            trivial = bool(np.all(keep))

            if not trivial:
                inv_ = pinv(np.multiply(keep[:, np.newaxis], V.T @ M))
                R_out = np.real(M @ inv_ @ V.T)
            else:
                R_out = np.eye(n_channels)

            n = update_at[j] + 1
            if (not trivial) or (not last_trivial):
                subrange = i_range[np.arange(last_n, n)]
                blend_x = np.pi * np.arange(1, n - last_n + 1) / (n - last_n)
                blend = (1 - np.cos(blend_x)) / 2
                tmp = data[:, subrange]
                data[:, subrange] = np.multiply(
                    blend, R_out @ tmp
                ) + np.multiply(1 - blend, last_R @ tmp)

            last_n, last_R, last_trivial = n, R_out, trivial

    carry = np.concatenate([carry, data[:, -P:]])
    carry = carry[:, -P:]

    if return_states:
        return data[:, :-P], {
            "M": M,
            "T": T,
            "R": R_out,
            "Zi": Zi,
            "cov": cov,
            "carry": carry,
        }
    return data[:, :-P]


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
) -> tuple[np.ndarray, np.ndarray]:
    """Drop windows containing too many bad channels.

    Direct port of ``asrpy.clean_windows``.
    """
    if not 0 < max_bad_chans < 1:
        raise ValueError("max_bad_chans must be a fraction in (0, 1)")

    truncate_quant = (0.0220, 0.6000)
    step_sizes = (0.01, 0.01)
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    n_channels, n_samples = X.shape
    N = int(win_len * sfreq)
    offsets = np.int_(
        np.round(np.arange(0, n_samples - N, N * (1 - win_overlap)))
    )

    wz = np.zeros((n_channels, len(offsets)))
    for ichan in range(n_channels):
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o : o + N]) / N) for o in offsets])
        mu, sig, _, _ = _fit_eeg_distribution(
            Y,
            min_clean_fraction,
            max_dropout_fraction,
            truncate_quant,
            step_sizes,
            shape_range,
        )
        wz[ichan] = (Y - mu) / sig

    wz[np.isnan(wz)] = np.inf
    swz = np.sort(wz, axis=0)

    mask1 = np.zeros(swz.shape[1], dtype=bool)
    mask2 = np.zeros(swz.shape[1], dtype=bool)
    if max(zthresholds) > 0:
        mask1 = swz[-(int(max_bad_chans) + 1), :] > max(zthresholds)
    if min(zthresholds) < 0:
        mask2 = swz[1 + int(max_bad_chans - 1), :] < min(zthresholds)

    remove_mask = np.logical_or(mask1, mask2)
    removed_wins = np.where(remove_mask)[0]

    sample_maskidx: list[np.ndarray] = []
    for w in removed_wins:
        sample_maskidx.append(np.arange(offsets[w], offsets[w] + N))
    sample_mask2remove = (
        np.unique(np.concatenate(sample_maskidx)) if sample_maskidx else np.array([], dtype=int)
    )

    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, axis=1)
        sample_mask = np.ones((1, n_samples), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
    else:
        sample_mask = np.ones((1, n_samples), dtype=bool)
        clean = X
        logger.info(
            "clean_windows: nothing to remove; calibration data is already clean."
        )

    return clean, sample_mask


# ---------------------------------------------------------------------------
# Internal helpers (private API; ports of asrpy.asr_utils)
# ---------------------------------------------------------------------------


def _fit_eeg_distribution(
    X: np.ndarray,
    min_clean_fraction: float = 0.25,
    max_dropout_fraction: float = 0.1,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    step_sizes: tuple[float, float] = (0.01, 0.01),
    shape_range: np.ndarray = np.arange(1.7, 3.5, 0.15),
) -> tuple[float, float, float, float]:
    """Estimate (mu, sig, alpha, beta) of clean EEG amplitude distribution.

    Port of ``asrpy.asr_utils.fit_eeg_distribution``. Grid-search fit of a
    truncated generalized Gaussian.
    """
    X = np.sort(X)
    n = len(X)

    quants = np.array(fit_quantiles)
    zbounds: list[np.ndarray] = []
    rescale: list[float] = []
    for b in range(len(shape_range)):
        gam = gammaincinv(
            1 / shape_range[b], np.sign(quants - 0.5) * (2 * quants - 1)
        )
        zbounds.append(np.sign(quants - 0.5) * gam ** (1 / shape_range[b]))
        rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

    lower_min = float(np.min(quants))
    max_width = float(quants[1] - quants[0])
    min_width = min_clean_fraction * max_width

    cols = np.arange(
        lower_min,
        lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
        step_sizes[0],
    )
    cols = np.round(n * cols).astype(int)
    rows = np.arange(0, int(np.round(n * max_width)))
    newX = np.zeros((len(rows), len(cols)))
    for i in range(len(rows)):
        newX[i] = X[i + cols]

    X1 = newX[0, :]
    newX = newX - X1

    opt_val: float = np.inf
    opt_lu: list[float] = [np.inf, np.inf]
    opt_bounds: np.ndarray = np.array([np.inf, np.inf])
    opt_beta: float = np.inf
    gridsearch = np.round(
        n * np.arange(max_width, min_width, -step_sizes[1])
    )
    for m in gridsearch.astype(int):
        mcurr = m - 1
        nbins = int(np.round(3 * np.log2(1 + m / 2)))
        col_scale = nbins / newX[mcurr]
        H = newX[:m] * col_scale

        hist_all = np.empty((nbins + 1, len(col_scale)), dtype=int)
        for ih in range(len(col_scale)):
            hist_all[:nbins, ih], _ = np.histogram(
                H[:, ih], bins=np.arange(0, nbins + 1)
            )
        hist_all[nbins, :] = 0
        logq = np.log(hist_all + 0.01)

        for k, _ in enumerate(shape_range):
            bounds = zbounds[k]
            bounds_width = float(bounds[1] - bounds[0])
            x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * bounds_width
            p = np.exp(-np.abs(x) ** shape_range[k]) * rescale[k]
            p = p / np.sum(p)

            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)
            min_val = float(np.min(kl))
            idx = int(np.argmin(kl))
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = float(shape_range[k])
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + newX[m - 1, idx]]

    alpha = (opt_lu[1] - opt_lu[0]) / float(opt_bounds[1] - opt_bounds[0])
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta
    sig = np.sqrt((alpha**2) * gamma(3 / beta) / gamma(1 / beta))
    return float(mu), float(sig), float(alpha), float(beta)


def _yulewalk(order: int, F: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recursive least-squares filter design (Yule-Walker).

    Port of ``asrpy.asr_utils.yulewalk``. Reference: Friedlander & Porat,
    *The Modified Yule-Walker Method of ARMA Spectral Estimation*, IEEE TAES,
    1984.
    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = int(np.fix(npt / 25))
    mf = F.size
    npt = npt + 1
    Ht = np.zeros((1, npt))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - lap // 2
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb : ne + 1] = inc * M[i + 1] + (1 - inc) * M[i]
        nb = ne + 1

    Ht_full = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht_full.size
    n2 = int(np.fix((n + 1) / 2))
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    R = np.real(np.fft.ifft(Ht_full * Ht_full))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))

    Rwindow = np.concatenate(
        (np.array([0.5]), np.ones(n2 - 1), np.zeros(n - n2)),
        axis=None,
    )
    A = _polystab(_denf(R, order))
    Qh = _numf(np.concatenate((np.array([R[0] / 2]), R[1:nr]), axis=None), A, order)
    _, Ss = 2 * np.real(signal.freqz(Qh, A, worN=n, whole=True))
    hh = np.fft.ifft(
        np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=np.complex128))))
    )
    B = np.real(_numf(hh[0:nr], A, nb))
    return B, A


def _yulewalk_filter(
    X: np.ndarray,
    sfreq: float,
    zi: np.ndarray | None = None,
    ab: tuple[np.ndarray, np.ndarray] | None = None,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the Yule-Walker spectrum-shaping IIR filter."""
    if ab is None:
        F = (
            np.array(
                [
                    0,
                    2,
                    3,
                    13,
                    16,
                    40,
                    np.minimum(80.0, (sfreq / 2.0) - 1.0),
                    sfreq / 2.0,
                ]
            )
            * 2.0
            / sfreq
        )
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = _yulewalk(8, F, M)
    else:
        A, B = ab

    if zi is None:
        out = signal.lfilter(B, A, X, axis=axis)
        zf = None
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)
    return out, zf


def _ma_filter(
    N: int, X: np.ndarray, Zi: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming moving-average filter implemented via cumsum."""
    if Zi is None:
        Zi = np.zeros([len(X), N])

    Y = np.concatenate([Zi, X], axis=1)
    M = Y.shape[-1]
    I_ = np.stack([np.arange(M - N), np.arange(N, M)]).astype(int)
    S = np.stack([-np.ones(M - N), np.ones(M - N)]) / N
    Xc = np.cumsum(
        np.multiply(Y[:, np.reshape(I_.T, -1)], np.reshape(S.T, [-1])), axis=-1
    )
    Xc = Xc[:, 1::2]
    Zf = np.concatenate(
        [-(Xc[:, -1] * N - Y[:, -N])[:, np.newaxis], Y[:, -N + 1 :]], axis=-1
    )
    return Xc, Zf


def _geometric_median(
    X: np.ndarray, tol: float = 1e-5, max_iter: int = 500
) -> np.ndarray:
    """Vardi-Zhang geometric median (Weiszfeld algorithm).

    Reference: Vardi, Y., & Zhang, C. H. (2000). *The multivariate L1-median
    and associated data depth.* PNAS, 97(4), 1423-1426.
    https://doi.org/10.1073/pnas.97.4.1423
    """
    y = np.mean(X, 0)

    for _ in range(max_iter):
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1.0 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < tol:
            return y1
        y = y1

    warnings.warn(
        f"Geometric median did not converge in {max_iter} iterations "
        f"(tol={tol}); returning last iterate.",
        RuntimeWarning,
        stacklevel=2,
    )
    return y


def _polystab(a: np.ndarray) -> np.ndarray:
    """Stabilize polynomial roots wrt the unit circle."""
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)
    if not np.sum(np.imag(a)):
        b = np.real(b)
    return b


def _numf(h: np.ndarray, a: np.ndarray, nb: int) -> np.ndarray:
    """Solve for numerator B given impulse response h of B/A and denominator A."""
    nh = h.size
    xn = np.concatenate((np.array([1.0]), np.zeros(nh - 1)), axis=None)
    impr = signal.lfilter(np.array([1.0]), a, xn)
    b = np.linalg.lstsq(
        toeplitz(impr, np.concatenate((np.array([1.0]), np.zeros(nb)), axis=None)),
        h.T,
        rcond=None,
    )[0].T
    return b


def _denf(R: np.ndarray, na: int) -> np.ndarray:
    """Compute order-na denominator A from covariances R(0)..R(nr)."""
    nr = R.size
    Rm = toeplitz(R[na : nr - 1], R[na:0:-1])
    Rhs = -R[na + 1 : nr]
    A = np.concatenate(
        (np.array([1.0]), np.linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T),
        axis=None,
    )
    return A


def _block_covariance(data: np.ndarray, window: int = 128) -> np.ndarray:
    """Block-wise covariance, summed over windowed shifts.

    Returns an array of shape (n_blocks, n_channels**2) (flattened cov), as
    required by :func:`_geometric_median`.
    """
    n_ch, n_times = data.shape
    n_blocks = len(np.arange(0, n_times - 1, window))
    U = np.zeros([n_blocks, n_ch * n_ch])
    data_T = data.T
    for k in range(window):
        idx_range = np.minimum(
            n_times - 1, np.arange(k, n_times + k - 2, window)
        )
        U = U + np.reshape(
            data_T[idx_range].reshape([-1, 1, n_ch])
            * data_T[idx_range].reshape(-1, n_ch, 1),
            U.shape,
        )
    return U
