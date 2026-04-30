"""Shared test fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest
import yaml

from asrpy_gpu import CUDA_AVAILABLE, MPS_AVAILABLE, TORCH_AVAILABLE

CONFIG_PATH = Path(__file__).parents[1] / "config" / "test_tolerances.yaml"


# ---------------------------------------------------------------------------
# Tolerances loaded from config/test_tolerances.yaml
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tolerances() -> dict:
    """Numerical tolerances used across equivalence tests."""
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)["equivalence"]


# ---------------------------------------------------------------------------
# Synthetic signal — L1 of the test plan
# ---------------------------------------------------------------------------


def make_synthetic_signal(
    n_channels: int = 16,
    duration_s: float = 30.0,
    sfreq: float = 256.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a reproducible EEG-like signal with injected artifacts.

    The signal is intentionally simple in V1: pink-ish noise + a handful of
    spike-like transients. It exists to exercise the non-trivial
    reconstruction branch of ASR (where ``trivial=False``) so we don't only
    test the identity case.

    .. note::

       Future enrichment (good first contribution): use realistic artifact
       templates — eyeblink shapes (low-freq, frontal-dominant), muscle
       bursts (broadband > 30 Hz, temporal-dominant), electrode pops
       (sharp DC-shifts). See literature in :mod:`asrpy_gpu._backend_numpy`.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sfreq)

    # 1/f-ish noise via filtering white noise.
    white = rng.standard_normal((n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sfreq)
    safe_freqs = np.where(freqs > 0, freqs, 1.0)
    pink_filter = np.where(freqs > 0, 1.0 / np.sqrt(safe_freqs), 0.0)
    pink = np.fft.irfft(spectrum * pink_filter, n=n_samples, axis=-1)

    # Scale to ~10 µV (typical EEG amplitude).
    sig = pink * (1e-5 / pink.std())

    # Inject ~30 spike artifacts, scattered in time and channels, ±100 µV.
    n_spikes = 30
    spike_times = rng.integers(int(0.5 * sfreq), n_samples - int(0.5 * sfreq), n_spikes)
    spike_chans = rng.integers(0, n_channels, n_spikes)
    spike_amp = rng.choice([-1.0, 1.0], n_spikes) * 1e-4
    for t, c, a in zip(spike_times, spike_chans, spike_amp, strict=True):
        sig[c, t] += a

    return sig


@pytest.fixture(scope="session")
def synthetic_data() -> np.ndarray:
    """Raw numpy array of synthetic EEG data."""
    return make_synthetic_signal()


@pytest.fixture(scope="session")
def synthetic_raw() -> mne.io.RawArray:
    """MNE :class:`Raw` wrapper around the synthetic signal."""
    sfreq = 256.0
    data = make_synthetic_signal(sfreq=sfreq)
    n_chan = data.shape[0]
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_chan)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# EEGLAB reference dataset — L2 of the test plan
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def eeglab_raw() -> mne.io.Raw:
    """The EEGLAB ``test_raw.set`` from ``mne.datasets.testing``.

    This is the same fixture used by ``asrpy``'s own test suite, which makes
    apples-to-apples comparisons trivial.
    """
    from mne.datasets import testing

    data_path = Path(testing.data_path(download=True)) / "EEGLAB"
    return mne.io.read_raw_eeglab(
        data_path / "test_raw.set", preload=True, verbose=False
    )


# ---------------------------------------------------------------------------
# Backend-availability markers
# ---------------------------------------------------------------------------


needs_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)
needs_mps = pytest.mark.skipif(
    not MPS_AVAILABLE, reason="MPS not available on this host"
)
needs_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA not available on this host"
)
