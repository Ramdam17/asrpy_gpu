"""Determinism: two identical runs must produce bit-identical output."""

from __future__ import annotations

import numpy as np
import pytest

from asrpy_gpu import ASR

from .conftest import needs_mps, needs_torch


def _run_twice(synthetic_raw, backend: str) -> tuple[np.ndarray, np.ndarray]:
    asr1 = ASR(sfreq=synthetic_raw.info["sfreq"], backend=backend)
    asr1.fit(synthetic_raw)
    out1 = asr1.transform(synthetic_raw).get_data()

    asr2 = ASR(sfreq=synthetic_raw.info["sfreq"], backend=backend)
    asr2.fit(synthetic_raw)
    out2 = asr2.transform(synthetic_raw).get_data()
    return out1, out2


def test_numpy_deterministic(synthetic_raw):
    out1, out2 = _run_twice(synthetic_raw, "numpy")
    np.testing.assert_array_equal(out1, out2)


@needs_torch
def test_torch_deterministic(synthetic_raw):
    out1, out2 = _run_twice(synthetic_raw, "torch")
    # Torch on CPU should be exact; on MPS may have tiny noise from BLAS
    # thread scheduling but is generally deterministic for SPD inputs.
    np.testing.assert_allclose(out1, out2, rtol=0, atol=1e-10)


@needs_mps
@pytest.mark.mps
def test_mps_deterministic(synthetic_raw):
    """MPS-specific check (only runs when MPS is available)."""
    out1, out2 = _run_twice(synthetic_raw, "torch")
    np.testing.assert_allclose(out1, out2, rtol=0, atol=1e-6)
