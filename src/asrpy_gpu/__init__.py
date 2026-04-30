"""asrpy_gpu — GPU-accelerated Artifact Subspace Reconstruction.

Drop-in replacement for ``asrpy`` (https://github.com/DiGyt/asrpy) with a
torch backend that runs on Apple MPS (default), NVIDIA CUDA, or CPU. Numpy
remains the reference (used for tests).

Heritage and licensing: BSD-3, see ``LICENSE``.
"""

from __future__ import annotations

from ._device import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    TORCH_AVAILABLE,
    resolve_device,
)
from .asr import ASR

__version__ = "0.1.0"

__all__ = [
    "ASR",
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "TORCH_AVAILABLE",
    "__version__",
    "resolve_device",
]
