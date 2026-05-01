"""PyObjC wrapper for the batched Jacobi eigendecomposition Metal kernel.

This is the production path for symmetric eigendecomposition on Apple
MPS, where ``torch.linalg.eigh`` is not yet implemented (torch 2.11) and
the pure-torch Jacobi (:mod:`asrpy_gpu._jacobi_torch`) is dominated by
Python-level kernel-launch overhead.

The Metal kernel does **all sweeps and rounds inside a single dispatch**,
so only one Python-to-GPU launch occurs per call to :func:`jacobi_eigh`.
The cost moves to global memory bandwidth, which on Apple Silicon's
unified memory comfortably handles our problem sizes (B ≤ a few thousand,
n ≤ a few hundred).

Optional dependency
-------------------
Requires ``pyobjc-framework-Metal``. The module sets ``METAL_AVAILABLE``
accordingly; importing it never raises, but calling :func:`jacobi_eigh`
when Metal is unavailable does.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:
    import Metal  # type: ignore[import-not-found]

    METAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    METAL_AVAILABLE = False

from ._jacobi_torch import _tournament_pairs

_KERNEL_SOURCE_PATH = Path(__file__).parent / "_jacobi_metal.metal"

# Lazily-initialised compiled kernel + Metal device, cached at module level.
_state: dict | None = None


def _build_state():
    """Compile the kernel and create a command queue (called once)."""
    if not METAL_AVAILABLE:
        raise RuntimeError(
            "pyobjc-framework-Metal is not installed; "
            "Metal kernel path unavailable."
        )

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device found.")

    source = _KERNEL_SOURCE_PATH.read_text()
    options = Metal.MTLCompileOptions.new()
    library, err = device.newLibraryWithSource_options_error_(
        source, options, None
    )
    if err is not None:
        raise RuntimeError(f"Metal compilation failed: {err}")

    fn = library.newFunctionWithName_("jacobi_eigh_kernel")
    if fn is None:
        raise RuntimeError("Kernel 'jacobi_eigh_kernel' not found in library.")

    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err is not None:
        raise RuntimeError(f"Pipeline creation failed: {err}")

    queue = device.newCommandQueue()
    return {
        "device": device,
        "library": library,
        "pipeline": pipeline,
        "queue": queue,
        # Schedule cache keyed by `n` (tournament structure depends only on n).
        "schedule_cache": {},
    }


def _get_state():
    global _state
    if _state is None:
        _state = _build_state()
    return _state


def _get_schedule(n: int) -> tuple[np.ndarray, int, int]:
    """Cache-friendly tournament schedule for size n.

    Returns
    -------
    schedule : (n_rounds, n_pairs, 2) int32 array (flat, C-contiguous)
    n_rounds : int
    n_pairs : int (always n // 2 for even n; padded with -1 sentinels for odd)
    """
    st = _get_state()
    cache = st["schedule_cache"]
    if n in cache:
        return cache[n]

    rounds = _tournament_pairs(n)
    n_rounds = len(rounds)
    n_pairs = n // 2  # parallel rotations per round (even n: exact; odd n: bye)

    sched = np.full((n_rounds, n_pairs, 2), -1, dtype=np.int32)
    for r, rnd in enumerate(rounds):
        for i, (p, q) in enumerate(rnd):
            sched[r, i, 0] = p
            sched[r, i, 1] = q

    cache[n] = (sched, n_rounds, n_pairs)
    return cache[n]


def jacobi_eigh(
    A: np.ndarray, *, max_sweeps: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Batched symmetric eigendecomposition on the GPU via Metal.

    Parameters
    ----------
    A : ndarray, shape (B, n, n) or (n, n), float32
        Symmetric input. Will be cast to float32 if not already (Metal
        does not support float64). The input is **not** modified.
    max_sweeps : int, optional
        Maximum number of full Jacobi sweeps. ``8–10`` is enough for
        float32 precision on well-conditioned inputs.

    Returns
    -------
    eigvals : ndarray, shape (B, n) or (n,), float32, sorted ascending.
    eigvecs : ndarray, shape (B, n, n) or (n, n), float32, columns are
        eigenvectors.
    """
    if not METAL_AVAILABLE:
        raise RuntimeError(
            "pyobjc-framework-Metal is not installed. Install via "
            "`uv pip install pyobjc-framework-Metal`."
        )

    if A.ndim == 2:
        A = A[None]
        squeeze_out = True
    else:
        squeeze_out = False

    B, n, m = A.shape
    if n != m:
        raise ValueError(f"Square matrices required, got shape {A.shape}.")

    A32 = np.ascontiguousarray(A, dtype=np.float32)
    A_buf_in = A32.copy()  # we'll let the kernel overwrite this in place
    V_buf_out = np.empty_like(A_buf_in)

    sched, n_rounds, n_pairs = _get_schedule(n)
    sched_buf = np.ascontiguousarray(sched, dtype=np.int32)

    st = _get_state()
    device = st["device"]
    pipeline = st["pipeline"]
    queue = st["queue"]

    # MTLBuffer creation. Shared mode gives zero-copy with numpy on
    # Apple Silicon's unified memory.
    options = Metal.MTLResourceStorageModeShared
    buf_A = device.newBufferWithBytes_length_options_(
        A_buf_in.tobytes(), A_buf_in.nbytes, options
    )
    buf_V = device.newBufferWithLength_options_(V_buf_out.nbytes, options)
    buf_sched = device.newBufferWithBytes_length_options_(
        sched_buf.tobytes(), sched_buf.nbytes, options
    )
    buf_n = device.newBufferWithBytes_length_options_(
        struct.pack("I", n), 4, options
    )
    buf_max_sweeps = device.newBufferWithBytes_length_options_(
        struct.pack("I", max_sweeps), 4, options
    )
    buf_n_rounds = device.newBufferWithBytes_length_options_(
        struct.pack("I", n_rounds), 4, options
    )
    buf_n_pairs = device.newBufferWithBytes_length_options_(
        struct.pack("I", n_pairs), 4, options
    )

    # Threadgroup memory for the (c, s) of the n_pairs rotations of a round.
    threadgroup_bytes = n_pairs * 2 * 4  # 2 floats per pair, 4 bytes each
    threadgroup_bytes = max(threadgroup_bytes, 16)  # Metal alignment

    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(buf_A, 0, 0)
    enc.setBuffer_offset_atIndex_(buf_V, 0, 1)
    enc.setBuffer_offset_atIndex_(buf_n, 0, 2)
    enc.setBuffer_offset_atIndex_(buf_max_sweeps, 0, 3)
    enc.setBuffer_offset_atIndex_(buf_sched, 0, 4)
    enc.setBuffer_offset_atIndex_(buf_n_rounds, 0, 5)
    enc.setBuffer_offset_atIndex_(buf_n_pairs, 0, 6)
    enc.setThreadgroupMemoryLength_atIndex_(threadgroup_bytes, 0)

    # Use as many threads per threadgroup as we can (typically 1024 on
    # Apple Silicon). We have n_pairs * n total work units per step, so
    # using a large threadgroup keeps everyone busy and reduces the
    # latency of the strided loop in the kernel.
    max_threads = pipeline.maxTotalThreadsPerThreadgroup()
    threads_per_group = min(n_pairs * n, max_threads)
    threads_per_group = max(threads_per_group, n)  # at least n threads

    grid = Metal.MTLSize(B * threads_per_group, 1, 1)
    tgsz = Metal.MTLSize(threads_per_group, 1, 1)
    enc.dispatchThreads_threadsPerThreadgroup_(grid, tgsz)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    # Copy results back. as_buffer() exposes the MTLBuffer contents as a
    # python buffer-protocol object; np.frombuffer wraps it cheaply.
    a_out = np.frombuffer(
        buf_A.contents().as_buffer(A_buf_in.nbytes),
        dtype=np.float32,
    ).copy().reshape(B, n, n)
    v_out = np.frombuffer(
        buf_V.contents().as_buffer(V_buf_out.nbytes),
        dtype=np.float32,
    ).copy().reshape(B, n, n)

    # Eigenvalues = diagonal of A; sort ascending and reorder V columns.
    D = np.diagonal(a_out, axis1=-2, axis2=-1).copy()
    sort_idx = np.argsort(D, axis=-1)
    D_sorted = np.take_along_axis(D, sort_idx, axis=-1)
    V_sorted = np.take_along_axis(
        v_out, sort_idx[:, None, :], axis=-1
    )

    if squeeze_out:
        return D_sorted[0], V_sorted[0]
    return D_sorted, V_sorted
