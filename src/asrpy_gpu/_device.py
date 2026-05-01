"""Device resolution for the torch backend.

Detects available compute backends (Apple MPS, NVIDIA CUDA, CPU) and exposes a
``resolve_device`` helper. Pattern adapted from the project's GPU-optimization
skill (priority: MPS > CUDA > CPU).
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = bool(torch.backends.mps.is_available())
    CUDA_AVAILABLE = bool(torch.cuda.is_available())
except ImportError:  # pragma: no cover - tested in CI without torch
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    CUDA_AVAILABLE = False


def resolve_device(backend: str = "auto") -> tuple[str, str]:
    """Resolve the compute backend and device for ASR.

    Parameters
    ----------
    backend : {"auto", "numpy", "torch"}
        Requested backend.

        * ``"auto"`` — torch (MPS > CUDA > CPU) if available, else numpy.
        * ``"numpy"`` — force the pure-numpy reference backend.
        * ``"torch"`` — force torch; falls back to numpy with a warning if torch
          is not installed.

    Returns
    -------
    backend : {"numpy", "torch"}
        The actually-resolved backend.
    device : {"cpu", "mps", "cuda"}
        The device string. ``"cpu"`` for the numpy backend; ``"mps"`` or
        ``"cuda"`` or ``"cpu"`` for the torch backend.

    Notes
    -----
    The numpy backend is the reference implementation: it stays bit-near to
    ``asrpy`` but does not benefit from the GPU. Use it for validation, not
    for production runs on large EEG sessions.
    """
    if backend == "numpy":
        return "numpy", "cpu"

    if backend == "torch":
        if not TORCH_AVAILABLE:
            warnings.warn(
                "torch is not installed; falling back to numpy backend. "
                "Install torch with: `uv add 'asrpy_gpu[torch]'`",
                UserWarning,
                stacklevel=2,
            )
            return "numpy", "cpu"
        return "torch", _best_torch_device()

    if backend == "auto":
        if TORCH_AVAILABLE:
            return "torch", _best_torch_device()
        return "numpy", "cpu"

    raise ValueError(
        f"Unknown backend {backend!r}. Expected one of {{'auto', 'numpy', 'torch'}}."
    )


def _best_torch_device() -> str:
    """Pick the best torch device: MPS > CUDA > CPU."""
    if MPS_AVAILABLE:
        return "mps"
    if CUDA_AVAILABLE:
        return "cuda"
    warnings.warn(
        "torch is installed but no GPU is available; running on CPU. "
        "Performance will be similar to the numpy backend.",
        UserWarning,
        stacklevel=2,
    )
    return "cpu"
