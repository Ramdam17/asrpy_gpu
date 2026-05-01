"""Public ``ASR`` class.

Drop-in compatible with ``asrpy.ASR``: the constructor signature and the
``fit(raw)`` / ``transform(raw)`` methods match the upstream API. The only
addition is a ``backend`` argument that selects ``"numpy"``, ``"torch"``, or
``"auto"``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import _backend_numpy as _np_backend
from ._device import resolve_device

logger = logging.getLogger(__name__)


class ASR:
    """Artifact Subspace Reconstruction.

    GPU-accelerated implementation of the Mullen et al. (2015) / Kothe &
    Jung (2016) ASR. The numpy backend is a faithful port of
    :mod:`asrpy` (used as the reference); the torch backend runs on Apple
    MPS, NVIDIA CUDA, or CPU.

    Parameters
    ----------
    sfreq : float
        Sampling rate, in Hz.
    cutoff : float, optional
        Standard-deviation cutoff for rejection. Defaults to 20.
    blocksize : int, optional
        Block size for the robust covariance and threshold estimation.
    win_len : float, optional
        Window length in seconds.
    win_overlap : float, optional
        Window overlap fraction.
    max_dropout_fraction : float, optional
        Maximum fraction of dropout windows.
    min_clean_fraction : float, optional
        Minimum fraction of windows that must be clean.
    ab : 2-tuple of ndarray, optional
        IIR coefficients ``(A, B)`` for the spectrum-shaping Yule-Walker
        filter. If ``None``, the EEGLAB default is built.
    max_bad_chans : float, optional
        Max fraction of bad channels per retained window.
    method : {"euclid"}, optional
        Only Euclidean ASR is implemented in V1; ``"riemann"`` is reserved
        for V2 (see ``docs/roadmap.md``).
    backend : {"auto", "numpy", "torch"}, optional
        Compute backend. ``"auto"`` resolves to ``"torch"`` if torch is
        installed, otherwise ``"numpy"``.

    Attributes
    ----------
    M, T : ndarray, shape (n_channels, n_channels)
        Mixing and threshold matrices, populated after :meth:`fit`.
    backend : str
        Resolved backend (``"numpy"`` or ``"torch"``).
    device : str
        Resolved device (``"cpu"``, ``"mps"``, or ``"cuda"``).

    References
    ----------
    .. [1] Mullen, T. R. et al. (2015). Real-time Neuroimaging and Cognitive
       Monitoring Using Wearable Dry EEG. IEEE Trans Biomed Eng, 62(11).
    .. [2] Kothe, C. A. E. & Jung, T.-P. (2016). U.S. Patent Application No.
       14/895,440.
    .. [3] Blum, S. et al. (2019). A Riemannian Modification of Artifact
       Subspace Reconstruction for EEG Artifact Handling. Frontiers in
       Human Neuroscience. https://doi.org/10.3389/fnhum.2019.00141
    """

    def __init__(
        self,
        sfreq: float,
        cutoff: float = 20.0,
        blocksize: int = 100,
        win_len: float = 0.5,
        win_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        ab: tuple[np.ndarray, np.ndarray] | None = None,
        max_bad_chans: float = 0.1,
        method: str = "euclid",
        backend: str = "auto",
    ) -> None:
        if method != "euclid":
            raise NotImplementedError(
                f"method={method!r} is not implemented in V1. Riemannian ASR "
                "is on the roadmap as V2 — see docs/roadmap.md."
            )

        self.sfreq = sfreq
        self.cutoff = cutoff
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = max_bad_chans
        self.method = method

        # Resolve backend / device.
        self.backend, self.device = resolve_device(backend)
        self._impl = self._load_impl(self.backend)

        # Build the default Yule-Walker IIR if not provided.
        if ab is None:
            yw_f = (
                np.array(
                    [
                        0,
                        2,
                        3,
                        13,
                        16,
                        40,
                        np.minimum(80.0, (self.sfreq / 2.0) - 1.0),
                        self.sfreq / 2.0,
                    ]
                )
                * 2.0
                / self.sfreq
            )
            yw_m = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
            self.B, self.A = _np_backend._yulewalk(8, yw_f, yw_m)
        else:
            self.A, self.B = ab

        self._reset()

    # ------------------------------------------------------------------ utils

    def _reset(self) -> None:
        self.M: np.ndarray | None = None
        self.T: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self.carry: np.ndarray | None = None
        self.Zi: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self._fitted = False

    @staticmethod
    def _load_impl(backend: str):
        """Return the backend module — imported lazily to keep torch optional."""
        if backend == "torch":
            from . import _backend_torch as impl

            return impl
        return _np_backend

    def _backend_kwargs(self) -> dict[str, Any]:
        return {"device": self.device} if self.backend == "torch" else {}

    # ------------------------------------------------------------------ API

    def fit(
        self,
        raw,
        picks: str | list | slice | None = "eeg",
        start: int = 0,
        stop: int | None = None,
        return_clean_window: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Calibrate ASR on clean reference data.

        Same semantics as ``asrpy.ASR.fit``: drops bad windows with
        :func:`clean_windows`, then estimates ``M`` and ``T`` from the
        remaining clean data.
        """
        X = raw.get_data(picks=picks, start=start, stop=stop)

        clean, sample_mask = self._impl.clean_windows(
            X,
            sfreq=self.sfreq,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_bad_chans=self.max_bad_chans,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction,
            **self._backend_kwargs(),
        )

        self.M, self.T = self._impl.calibrate(
            clean,
            sfreq=self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            ab=(self.A, self.B),
            **self._backend_kwargs(),
        )
        self._fitted = True
        logger.info(
            "[ASR] Fitted (backend=%s device=%s). M.shape=%s, T.shape=%s",
            self.backend,
            self.device,
            self.M.shape,
            self.T.shape,
        )
        if return_clean_window:
            return clean, sample_mask
        return None

    def transform(
        self,
        raw,
        picks: str | list | slice | None = "eeg",
        lookahead: float = 0.25,
        stepsize: int = 32,
        maxdims: float | int = 0.66,
        return_states: bool = False,
        mem_splits: int = 3,
    ):
        """Apply ASR cleaning to ``raw``.

        Returns a copy of ``raw`` with the picked channels replaced by their
        ASR-cleaned counterparts. Same semantics as ``asrpy.ASR.transform``.
        """
        if not self._fitted:
            raise RuntimeError("ASR.fit must be called before ASR.transform.")

        X = raw.get_data(picks=picks)

        # asrpy convention: zero-pad lookahead at the end.
        lookahead_samples = int(self.sfreq * lookahead)
        X = np.concatenate(
            [X, np.zeros([X.shape[0], lookahead_samples])],
            axis=1,
        )

        result = self._impl.process(
            X,
            self.sfreq,
            self.M,
            self.T,
            win_len=self.win_len,
            lookahead=lookahead,
            stepsize=stepsize,
            maxdims=maxdims,
            ab=(self.A, self.B),
            R=self.R,
            Zi=self.Zi,
            cov=self.cov,
            carry=self.carry,
            return_states=return_states,
            mem_splits=mem_splits,
            **self._backend_kwargs(),
        )

        if return_states:
            X_clean, _ = result
        else:
            X_clean = result

        # Strip the lookahead pad.
        X_clean = X_clean[:, lookahead_samples:]

        out = raw.copy()
        out.apply_function(lambda _x: X_clean, picks=picks, channel_wise=False)
        return out
