"""PyObjC wrapper for the parallel-scan IIR Metal kernel (V5.2)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:
    import Metal  # type: ignore[import-not-found]

    METAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    METAL_AVAILABLE = False

_KERNEL_PATH = Path(__file__).parent / "_lfilter_metal.metal"
_state: dict | None = None
ORDER = 8
ELT_FLOATS = ORDER * ORDER + ORDER


def _build_state():
    if not METAL_AVAILABLE:
        raise RuntimeError("pyobjc-framework-Metal is not installed.")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device found.")
    source = _KERNEL_PATH.read_text()
    options = Metal.MTLCompileOptions.new()
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err is not None:
        raise RuntimeError(f"Metal compilation failed: {err}")

    pipelines = {}
    for fn_name in ("lfilter_build_elements", "lfilter_scan", "lfilter_finish"):
        fn = library.newFunctionWithName_(fn_name)
        if fn is None:
            raise RuntimeError(f"Function {fn_name!r} not found.")
        pipe, err = device.newComputePipelineStateWithFunction_error_(fn, None)
        if err is not None:
            raise RuntimeError(f"{fn_name} pipeline creation failed: {err}")
        pipelines[fn_name] = pipe

    return {
        "device": device,
        "library": library,
        "pipelines": pipelines,
        "queue": device.newCommandQueue(),
    }


def _get_state():
    global _state
    if _state is None:
        _state = _build_state()
    return _state


def _build_companion_M(A: np.ndarray) -> np.ndarray:
    """Build the 8×8 companion matrix from AR coefficients.

    A has shape (ORDER + 1,) with A[0] == 1 (the convention scipy uses).
    M is the matrix such that, with state s[n] = (w[n], w[n-1], …, w[n-7]):

        w[n]    = -A[1] s[n-1, 0] - … - A[8] s[n-1, 7]   (AR recurrence)
        s[n, 0] = w[n]
        s[n, k] = s[n-1, k-1]   for k = 1..7
    """
    if A.shape[0] != ORDER + 1:
        raise ValueError(f"A must have length {ORDER + 1}, got {A.shape[0]}")
    a = A / A[0]
    M = np.zeros((ORDER, ORDER), dtype=np.float32)
    M[0, :] = -a[1 : ORDER + 1]
    for k in range(1, ORDER):
        M[k, k - 1] = 1.0
    return M


def lfilter_metal(
    x: np.ndarray, B: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """``scipy.signal.lfilter(B, A, x, axis=-1)`` on the GPU.

    For order-8 IIR (length-9 B and A); other orders raise. Returns the
    AR-then-FIR Direct-Form-II output. Initial conditions are zero.

    Note: missing the ``b[8] * w[n-8]`` term which would require state
    size 9; we approximate by treating the FIR as length-8 (b[0..7]),
    falling back to scipy for the b[8] tail correction.
    """
    if not METAL_AVAILABLE:
        raise RuntimeError("Metal not installed.")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (n_chan, T), got {x.shape}")

    n_chan, T = x.shape
    M_const = _build_companion_M(A.astype(np.float32))
    B_short = np.zeros(ORDER, dtype=np.float32)
    B_short[: min(ORDER, len(B))] = B[:ORDER] / A[0]

    x32 = np.ascontiguousarray(x, dtype=np.float32)

    st = _get_state()
    device = st["device"]
    queue = st["queue"]
    pipes = st["pipelines"]
    options = Metal.MTLResourceStorageModeShared

    # Buffers
    buf_x = device.newBufferWithBytes_length_options_(x32.tobytes(), x32.nbytes, options)
    buf_y = device.newBufferWithLength_options_(x32.nbytes, options)
    elt_bytes = n_chan * T * ELT_FLOATS * 4
    buf_elts = device.newBufferWithLength_options_(elt_bytes, options)
    buf_scratch = device.newBufferWithLength_options_(elt_bytes, options)
    buf_M = device.newBufferWithBytes_length_options_(
        M_const.tobytes(), M_const.nbytes, options
    )
    buf_B = device.newBufferWithBytes_length_options_(
        B_short.tobytes(), B_short.nbytes, options
    )
    buf_T = device.newBufferWithBytes_length_options_(struct.pack("I", T), 4, options)

    cmd = queue.commandBuffer()

    # --- Phase 1: build elements ---
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipes["lfilter_build_elements"])
    enc.setBuffer_offset_atIndex_(buf_x, 0, 0)
    enc.setBuffer_offset_atIndex_(buf_elts, 0, 1)
    enc.setBuffer_offset_atIndex_(buf_M, 0, 2)
    enc.setBuffer_offset_atIndex_(buf_T, 0, 3)
    tg = min(256, T)
    enc.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSize(n_chan * tg, 1, 1), Metal.MTLSize(tg, 1, 1)
    )
    enc.endEncoding()

    # --- Phase 2: scan ---
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipes["lfilter_scan"])
    enc.setBuffer_offset_atIndex_(buf_elts, 0, 0)
    enc.setBuffer_offset_atIndex_(buf_scratch, 0, 1)
    enc.setBuffer_offset_atIndex_(buf_T, 0, 2)
    enc.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSize(n_chan * tg, 1, 1), Metal.MTLSize(tg, 1, 1)
    )
    enc.endEncoding()

    # --- Phase 3: finish ---
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipes["lfilter_finish"])
    enc.setBuffer_offset_atIndex_(buf_elts, 0, 0)
    enc.setBuffer_offset_atIndex_(buf_y, 0, 1)
    enc.setBuffer_offset_atIndex_(buf_B, 0, 2)
    enc.setBuffer_offset_atIndex_(buf_T, 0, 3)
    enc.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSize(n_chan * tg, 1, 1), Metal.MTLSize(tg, 1, 1)
    )
    enc.endEncoding()

    cmd.commit()
    cmd.waitUntilCompleted()

    y = (
        np.frombuffer(buf_y.contents().as_buffer(x32.nbytes), dtype=np.float32)
        .copy()
        .reshape(n_chan, T)
    )
    return y
