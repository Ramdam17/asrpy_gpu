"""Compatibility shim around `asrpy` for the equivalence tests.

`asrpy` (DiGyt/asrpy, BSD-3) was written against numpy < 2 and breaks at
``fit_eeg_distribution`` on numpy 2.x:

    rows = np.arange(0, int(np.round(n * max_width)))
    # TypeError: only 0-dimensional arrays can be converted to Python scalars

This shim monkey-patches the offending lines so we can use the upstream
package as a numerical reference without pinning the whole project to
numpy < 2. The patch is **purely a numpy-2 compatibility shim**: it does not
change the algorithm. Verified line-for-line against
``asrpy.asr_utils.fit_eeg_distribution`` (commit
5a99169ff59faa7bb967ace78987bd00cf39f334).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaincinv

import asrpy
import asrpy.asr_utils as _asr_utils


def _patched_fit_eeg_distribution(  # noqa: PLR0913
    X,
    min_clean_fraction=0.25,
    max_dropout_fraction=0.1,
    fit_quantiles=(0.022, 0.6),
    step_sizes=(0.01, 0.01),
    shape_range=np.arange(1.7, 3.5, 0.15),
):
    """Numpy-2 compatible re-implementation, byte-equivalent to asrpy."""
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

    opt_val = np.inf
    opt_lu = [np.inf, np.inf]
    opt_bounds = np.array([np.inf, np.inf])
    opt_beta = np.inf
    gridsearch = np.round(n * np.arange(max_width, min_width, -step_sizes[1]))
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
    return mu, sig, alpha, beta


# Apply the patch at module import.
_asr_utils.fit_eeg_distribution = _patched_fit_eeg_distribution
# `asrpy.asr` imports `fit_eeg_distribution` by name into its namespace, so
# we have to patch there too — otherwise the old (broken) reference is
# captured in the module dict.
asrpy.asr.fit_eeg_distribution = _patched_fit_eeg_distribution

# Re-export the upstream API so tests can write
# `from tests.reference.asrpy_reference import asr_calibrate` etc.
asr_calibrate = asrpy.asr_calibrate
asr_process = asrpy.asr_process
clean_windows = asrpy.clean_windows
ASR = asrpy.ASR

__all__ = ["ASR", "asr_calibrate", "asr_process", "clean_windows"]
