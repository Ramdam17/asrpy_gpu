"""Robustness tests for ASR calibration on degenerate / borderline inputs.

Regression coverage for the IndexError previously raised when
``clean_windows`` removed enough data that fewer than one calibration
window remained — see ``_fit_eeg_distribution`` ``newX[0, :]``.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import pytest

from asrpy_gpu import ASR, InsufficientCalibrationDataError
from asrpy_gpu import _backend_numpy as _np_backend
from asrpy_gpu._errors import InsufficientCalibrationDataError as ErrCls

from .conftest import needs_torch

# ---------------------------------------------------------------------------
# Low-level: _fit_eeg_distribution contract
# ---------------------------------------------------------------------------


def test_fit_eeg_distribution_rejects_empty_input():
    """Empty input used to crash with an opaque IndexError; now ValueError."""
    with pytest.raises(ValueError, match="empty array"):
        _np_backend._fit_eeg_distribution(np.array([]))


# ---------------------------------------------------------------------------
# Backend-level: calibrate() and clean_windows() raise our typed exception
# ---------------------------------------------------------------------------


def _too_short_array(n_samples: int, n_channels: int = 4, seed: int = 0) -> np.ndarray:
    """Synthetic EEG with fewer samples than one calibration window."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples))


def test_numpy_calibrate_raises_typed_error_below_one_window():
    sfreq = 500.0
    # win_len=0.5 → N=250. We pass exactly 250 samples; arange(0, 0, step) is
    # empty, so we should fail loud.
    X = _too_short_array(n_samples=250)
    with pytest.raises(InsufficientCalibrationDataError) as excinfo:
        _np_backend.calibrate(X, sfreq=sfreq)
    assert excinfo.value.n_samples == 250
    assert excinfo.value.sfreq == sfreq
    assert excinfo.value.n_required == 251
    # Drop-in compatibility: the exception is also a ValueError so existing
    # except-ValueError handlers still catch it.
    assert isinstance(excinfo.value, ValueError)


def test_numpy_calibrate_passes_just_above_threshold():
    """Smoke test: 251 samples is the smallest input that should not raise."""
    X = _too_short_array(n_samples=260)
    # Will likely produce a degenerate fit, but should not raise our typed
    # error. Other numerical warnings are OK.
    _np_backend.calibrate(X, sfreq=500.0)


def test_numpy_clean_windows_raises_typed_error_below_one_window():
    sfreq = 500.0
    X = _too_short_array(n_samples=250)
    with pytest.raises(InsufficientCalibrationDataError):
        _np_backend.clean_windows(X, sfreq=sfreq)


@needs_torch
def test_torch_calibrate_raises_typed_error_below_one_window():
    from asrpy_gpu import _backend_torch as _t_backend

    sfreq = 500.0
    X = _too_short_array(n_samples=250)
    with pytest.raises(InsufficientCalibrationDataError):
        _t_backend.calibrate(X, sfreq=sfreq, device="cpu")


# ---------------------------------------------------------------------------
# ASR.fit policy: raise vs warn_skip
# ---------------------------------------------------------------------------


def _too_short_raw(duration_s: float, sfreq: float = 500.0) -> mne.io.RawArray:
    rng = np.random.default_rng(0)
    n_chan = 4
    n_samples = int(duration_s * sfreq)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_chan)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    return mne.io.RawArray(
        rng.standard_normal((n_chan, n_samples)) * 1e-5, info, verbose=False
    )


def test_asr_fit_raises_by_default_on_too_short_signal():
    raw = _too_short_raw(duration_s=0.4)  # < 0.5 s = win_len
    asr = ASR(sfreq=raw.info["sfreq"], backend="numpy")
    with pytest.raises(InsufficientCalibrationDataError):
        asr.fit(raw)
    assert asr.M is None
    assert asr.T is None


def test_asr_fit_warn_skip_does_not_raise(caplog):
    raw = _too_short_raw(duration_s=0.4)
    asr = ASR(
        sfreq=raw.info["sfreq"],
        backend="numpy",
        on_insufficient_data="warn_skip",
    )
    with caplog.at_level(logging.WARNING, logger="asrpy_gpu.asr"):
        result = asr.fit(raw)
    assert result is None
    assert asr.M is None
    assert asr.T is None
    # The transform call must then raise a clear RuntimeError, not a
    # cryptic AttributeError on `None.shape`.
    with pytest.raises(RuntimeError, match="warn_skip"):
        asr.transform(raw)
    assert any("Skipping calibration" in r.message for r in caplog.records)


def test_asr_constructor_rejects_invalid_policy():
    with pytest.raises(ValueError, match="on_insufficient_data"):
        ASR(sfreq=500.0, on_insufficient_data="lol_skip")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Soft warning: short-but-fittable calibration data
# ---------------------------------------------------------------------------


def test_asr_fit_warns_below_min_calibration_seconds(caplog):
    """5 s of clean data should fit but warn: under the 30-s default."""
    raw = _too_short_raw(duration_s=5.0)
    asr = ASR(sfreq=raw.info["sfreq"], backend="numpy")
    with caplog.at_level(logging.WARNING, logger="asrpy_gpu.asr"):
        asr.fit(raw)
    assert asr._fitted
    assert any("min_calibration_seconds" in r.message for r in caplog.records)


def test_asr_fit_no_warn_when_threshold_disabled(caplog):
    raw = _too_short_raw(duration_s=5.0)
    asr = ASR(
        sfreq=raw.info["sfreq"],
        backend="numpy",
        min_calibration_seconds=0.0,
    )
    with caplog.at_level(logging.WARNING, logger="asrpy_gpu.asr"):
        asr.fit(raw)
    assert not any("min_calibration_seconds" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Sanity: the alias used in conftest is the same object
# ---------------------------------------------------------------------------


def test_error_class_is_a_value_error_subclass():
    assert issubclass(ErrCls, ValueError)
    assert ErrCls is InsufficientCalibrationDataError
