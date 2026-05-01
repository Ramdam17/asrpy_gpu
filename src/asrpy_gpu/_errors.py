"""Domain-specific exceptions for asrpy_gpu.

Lives in its own module so that both backends and the public API can import
without creating a cycle (``asr.py`` imports the backends, the backends import
this).
"""

from __future__ import annotations


class InsufficientCalibrationDataError(ValueError):
    """Raised when too few samples remain for ASR calibration.

    ASR estimates a per-channel amplitude distribution by sliding a window
    of length ``win_len`` over the calibration data; if the data has fewer
    samples than one such window, the estimator silently produces an empty
    array and crashes downstream. This exception is raised explicitly at
    that threshold so the failure mode is informative rather than an
    inscrutable ``IndexError`` deep in :func:`_fit_eeg_distribution`.

    The most common cause is :func:`clean_windows` rejecting too much of
    the calibration data (e.g. a saturated channel inflating the rejection
    mask). Suggestions in the error message guide the user toward
    pipeline-level remedies (relax thresholds, merge tasks, skip subject).

    Parameters
    ----------
    n_samples : int
        Number of samples that survived ``clean_windows`` (i.e. the
        ``n_samples`` axis of the array passed to :func:`calibrate`).
    sfreq : float
        Sampling frequency in Hz.
    n_required : int
        Minimum number of samples required (typically ``win_len * sfreq``
        plus a margin so that at least one window-step exists).
    win_len : float
        Calibration window length in seconds.

    Attributes
    ----------
    n_samples, sfreq, n_required, win_len : as above
        Stored on the instance so callers (e.g. a batch pipeline) can
        introspect the failure programmatically.
    """

    def __init__(
        self,
        n_samples: int,
        sfreq: float,
        n_required: int,
        win_len: float,
    ) -> None:
        self.n_samples = int(n_samples)
        self.sfreq = float(sfreq)
        self.n_required = int(n_required)
        self.win_len = float(win_len)

        actual_s = self.n_samples / self.sfreq
        required_s = self.n_required / self.sfreq

        msg = (
            f"ASR calibration requires at least {self.n_required} samples "
            f"(~{required_s:.2f} s at {self.sfreq:g} Hz, i.e. one "
            f"{self.win_len:g}-s window plus one step), "
            f"but only {self.n_samples} ({actual_s:.2f} s) remained after "
            f"window rejection.\n"
            "Likely causes:\n"
            "  - one or more pathological channels inflated the bad-window "
            "mask in clean_windows;\n"
            "  - the calibration recording is too short or too contaminated.\n"
            "Possible remedies:\n"
            "  - drop pathological channels before fit (raw.drop_channels(...));\n"
            "  - relax max_bad_chans or min_clean_fraction;\n"
            "  - concatenate multiple resting/baseline tasks for calibration;\n"
            "  - skip this subject by passing on_insufficient_data='warn_skip'."
        )
        super().__init__(msg)
