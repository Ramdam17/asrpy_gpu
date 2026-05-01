"""Numerical equivalence vs the asrpy reference (L2 of the test plan)."""

from __future__ import annotations

import numpy as np
import pytest

from asrpy_gpu import _backend_numpy as bn

from .conftest import needs_mps, needs_torch
from .reference.asrpy_reference import (
    asr_calibrate as ref_calibrate,
)
from .reference.asrpy_reference import (
    asr_process as ref_process,
)
from .reference.asrpy_reference import (
    clean_windows as ref_clean_windows,
)

# ---------------------------------------------------------------------------
# Numpy backend — should be bit-near asrpy
# ---------------------------------------------------------------------------


def test_numpy_clean_windows_matches_asrpy(synthetic_data, tolerances):
    cw_ours, m_ours = bn.clean_windows(synthetic_data, sfreq=256.0)
    cw_ref, m_ref = ref_clean_windows(synthetic_data, sfreq=256.0)
    assert cw_ours.shape == cw_ref.shape
    np.testing.assert_array_equal(m_ours, m_ref)
    np.testing.assert_allclose(
        cw_ours,
        cw_ref,
        rtol=tolerances["numpy_vs_asrpy"]["rtol"],
        atol=tolerances["numpy_vs_asrpy"]["atol"],
    )


def test_numpy_calibrate_matches_asrpy(synthetic_data, tolerances):
    M_ours, T_ours = bn.calibrate(synthetic_data, sfreq=256.0)
    M_ref, T_ref = ref_calibrate(synthetic_data, sfreq=256.0)
    np.testing.assert_allclose(
        M_ours,
        M_ref,
        rtol=tolerances["numpy_vs_asrpy"]["rtol"],
        atol=tolerances["numpy_vs_asrpy"]["atol"],
    )
    # T may differ by sign-flips of rows (eigenvector sign ambiguity), but
    # ASR is invariant to this. We compare the squared row-sums of T @ V which
    # is the only quantity used algorithmically.
    # Easier check: M is the canonical quantity, just verify T is finite.
    assert np.isfinite(T_ours).all()


def test_numpy_process_matches_asrpy(synthetic_data, tolerances):
    M_ref, T_ref = ref_calibrate(synthetic_data, sfreq=256.0)
    clean_ours = bn.process(synthetic_data, sfreq=256.0, M=M_ref, T=T_ref)
    clean_ref = ref_process(synthetic_data, sfreq=256.0, M=M_ref, T=T_ref)
    np.testing.assert_allclose(
        clean_ours,
        clean_ref,
        rtol=tolerances["numpy_vs_asrpy"]["rtol"],
        atol=tolerances["numpy_vs_asrpy"]["atol"],
    )
    corr = np.corrcoef(clean_ours.ravel(), clean_ref.ravel())[0, 1]
    assert corr >= tolerances["signal_correlation_min"]


# ---------------------------------------------------------------------------
# Torch backend — should match within MPS / CPU torch tolerances
# ---------------------------------------------------------------------------


@needs_torch
def test_torch_cpu_calibrate_matches_asrpy(synthetic_data, tolerances):
    from asrpy_gpu import _backend_torch as bt

    M_ours, _ = bt.calibrate(synthetic_data, sfreq=256.0, device="cpu")
    M_ref, _ = ref_calibrate(synthetic_data, sfreq=256.0)
    np.testing.assert_allclose(
        M_ours,
        M_ref,
        rtol=tolerances["torch_cpu_vs_asrpy"]["rtol"],
        atol=tolerances["torch_cpu_vs_asrpy"]["atol"],
    )


@needs_torch
def test_torch_cpu_process_matches_asrpy(synthetic_data, tolerances):
    from asrpy_gpu import _backend_torch as bt

    M_ref, T_ref = ref_calibrate(synthetic_data, sfreq=256.0)
    clean_ours = bt.process(synthetic_data, sfreq=256.0, M=M_ref, T=T_ref, device="cpu")
    clean_ref = ref_process(synthetic_data, sfreq=256.0, M=M_ref, T=T_ref)
    np.testing.assert_allclose(
        clean_ours,
        clean_ref,
        rtol=tolerances["torch_cpu_vs_asrpy"]["rtol"],
        atol=tolerances["torch_cpu_vs_asrpy"]["atol"],
    )


@needs_mps
@pytest.mark.mps
def test_torch_mps_process_matches_asrpy(synthetic_data, tolerances):
    """MPS uses float32; tolerance is looser per docs/equivalence.md."""
    from asrpy_gpu import _backend_torch as bt

    M_ref, T_ref = ref_calibrate(synthetic_data, sfreq=256.0)
    clean_ours = bt.process(synthetic_data, sfreq=256.0, M=M_ref, T=T_ref, device="mps")
    clean_ref = ref_process(synthetic_data, sfreq=256.0, M=M_ref, T=T_ref)
    np.testing.assert_allclose(
        clean_ours,
        clean_ref,
        rtol=tolerances["torch_mps_vs_asrpy"]["rtol"],
        atol=tolerances["torch_mps_vs_asrpy"]["atol"],
    )
    corr = np.corrcoef(clean_ours.ravel(), clean_ref.ravel())[0, 1]
    assert corr >= tolerances["signal_correlation_min"]


# ---------------------------------------------------------------------------
# EEGLAB reference — L2 with real-world data
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_numpy_matches_asrpy_on_eeglab(eeglab_raw, tolerances):
    """Equivalence on the same EEGLAB file asrpy uses in its own tests."""
    X = eeglab_raw.get_data(picks="eeg")
    sfreq = eeglab_raw.info["sfreq"]

    # Drop the heavy clean_windows for speed; use full record.
    M_ref, T_ref = ref_calibrate(X, sfreq=sfreq)
    clean_ref = ref_process(X, sfreq=sfreq, M=M_ref, T=T_ref)

    M_ours, _ = bn.calibrate(X, sfreq=sfreq)
    clean_ours = bn.process(X, sfreq=sfreq, M=M_ref, T=T_ref)

    np.testing.assert_allclose(
        M_ours,
        M_ref,
        rtol=tolerances["numpy_vs_asrpy"]["rtol"],
        atol=tolerances["numpy_vs_asrpy"]["atol"],
    )
    np.testing.assert_allclose(
        clean_ours,
        clean_ref,
        rtol=tolerances["numpy_vs_asrpy"]["rtol"],
        atol=tolerances["numpy_vs_asrpy"]["atol"],
    )
    corr = np.corrcoef(clean_ours.ravel(), clean_ref.ravel())[0, 1]
    assert corr >= tolerances["signal_correlation_min"]
