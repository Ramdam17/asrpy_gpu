"""API surface tests — make sure asrpy_gpu.ASR is drop-in for asrpy.ASR."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from asrpy_gpu import ASR


def test_asr_accepts_asrpy_signature(synthetic_raw):
    """The constructor must accept all asrpy.ASR keyword arguments."""
    sfreq = synthetic_raw.info["sfreq"]
    asr = ASR(
        sfreq=sfreq,
        cutoff=20,
        blocksize=100,
        win_len=0.5,
        win_overlap=0.66,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        ab=None,
        max_bad_chans=0.1,
        method="euclid",
    )
    assert asr.sfreq == sfreq
    assert asr.method == "euclid"
    assert asr.M is None
    assert asr.T is None


def test_asr_rejects_riemann_in_v1(synthetic_raw):
    sfreq = synthetic_raw.info["sfreq"]
    with pytest.raises(NotImplementedError, match="V2"):
        ASR(sfreq=sfreq, method="riemann")


def test_fit_transform_returns_raw(synthetic_raw):
    sfreq = synthetic_raw.info["sfreq"]
    asr = ASR(sfreq=sfreq, backend="numpy")
    asr.fit(synthetic_raw, picks="eeg")
    cleaned = asr.transform(synthetic_raw, picks="eeg")
    assert cleaned.get_data().shape == synthetic_raw.get_data().shape


def test_transform_before_fit_errors(synthetic_raw):
    sfreq = synthetic_raw.info["sfreq"]
    asr = ASR(sfreq=sfreq, backend="numpy")
    with pytest.raises(RuntimeError, match="fit"):
        asr.transform(synthetic_raw)


def test_fit_populates_M_and_T(synthetic_raw):
    sfreq = synthetic_raw.info["sfreq"]
    asr = ASR(sfreq=sfreq, backend="numpy")
    asr.fit(synthetic_raw, picks="eeg")
    n_eeg = len(synthetic_raw.copy().pick("eeg").ch_names)
    assert asr.M.shape == (n_eeg, n_eeg)
    assert asr.T.shape == (n_eeg, n_eeg)
    assert np.isfinite(asr.M).all()
    assert np.isfinite(asr.T).all()


def test_signature_compatible_with_asrpy():
    """Every asrpy.ASR.__init__ kwarg must exist in our ASR (with same name)."""
    import asrpy

    asrpy_params = inspect.signature(asrpy.ASR.__init__).parameters
    ours_params = inspect.signature(ASR.__init__).parameters

    for name in asrpy_params:
        if name == "self":
            continue
        assert name in ours_params, (
            f"asrpy.ASR.__init__ has '{name}' but asrpy_gpu.ASR doesn't"
        )
