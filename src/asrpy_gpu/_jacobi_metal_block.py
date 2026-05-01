"""PyObjC wrapper for the block-Jacobi Metal kernel.

Block Jacobi (b = 32) for symmetric eigendecomposition. Targets the
n = 256 case where standard Jacobi runs into the unified-memory
bandwidth wall: working in 2b × 2b sub-matrices that stay in
threadgroup (= L1-equivalent) memory amortises the bandwidth cost
over many internal rotations before any device-memory traffic.

For n that does not fit a clean 32-block partition, this wrapper
falls back to the standard Jacobi via :mod:`_jacobi_metal`.
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

_KERNEL_SOURCE_PATH = Path(__file__).parent / "_jacobi_metal_block.metal"

BLOCK_SIZE = 32

_state: dict | None = None


def _build_state():
    if not METAL_AVAILABLE:
        raise RuntimeError("pyobjc-framework-Metal is not installed.")

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device found.")

    source = _KERNEL_SOURCE_PATH.read_text()
    options = Metal.MTLCompileOptions.new()
    library, err = device.newLibraryWithSource_options_error_(source, options, None)
    if err is not None:
        raise RuntimeError(f"Metal compilation failed: {err}")

    fn = library.newFunctionWithName_("block_jacobi_eigh_kernel")
    if fn is None:
        raise RuntimeError("Kernel 'block_jacobi_eigh_kernel' not found.")

    pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err is not None:
        raise RuntimeError(f"Pipeline creation failed: {err}")

    return {
        "device": device,
        "library": library,
        "pipeline": pipeline,
        "queue": device.newCommandQueue(),
        "schedule_cache": {},
    }


def _get_state():
    global _state
    if _state is None:
        _state = _build_state()
    return _state


def _block_schedule(num_blocks: int) -> tuple[np.ndarray, int, int]:
    """Tournament schedule over num_blocks block columns."""
    rounds = _tournament_pairs(num_blocks)
    n_rounds = len(rounds)
    n_pairs = num_blocks // 2
    sched = np.full((n_rounds, n_pairs, 2), -1, dtype=np.int32)
    for r, rnd in enumerate(rounds):
        for i, (p, q) in enumerate(rnd):
            sched[r, i, 0] = p
            sched[r, i, 1] = q
    return sched, n_rounds, n_pairs


def _sub_schedule() -> np.ndarray:
    """Tournament schedule for the 2b × 2b sub-matrix Jacobi."""
    sub_dim = 2 * BLOCK_SIZE
    rounds = _tournament_pairs(sub_dim)
    n_rounds = len(rounds)
    n_pairs = sub_dim // 2
    sched = np.full((n_rounds, n_pairs, 2), -1, dtype=np.int32)
    for r, rnd in enumerate(rounds):
        for i, (p, q) in enumerate(rnd):
            sched[r, i, 0] = p
            sched[r, i, 1] = q
    return sched


def jacobi_eigh_block(
    A: np.ndarray,
    *,
    max_block_sweeps: int = 8,
    max_inner_sweeps: int = 12,
    rel_tol: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-Jacobi symmetric eigendecomposition on the GPU.

    Parameters
    ----------
    A : ndarray, shape (B, n, n) or (n, n), float32
        Symmetric input. n must be a multiple of ``BLOCK_SIZE`` (32).
    max_block_sweeps : int, optional
        Number of OUTER block sweeps. Each sweep walks every C(num_blocks, 2)
        pair of block columns once.
    max_inner_sweeps : int, optional
        Number of INNER sweeps (cyclic Jacobi on the 2b × 2b sub-matrix).
    rel_tol : float, optional
        Relative tolerance for the inner convergence check.

    Returns
    -------
    eigvals : ndarray, shape (B, n) or (n,) — sorted ascending.
    eigvecs : ndarray, shape (B, n, n) or (n, n).
    """
    if not METAL_AVAILABLE:
        raise RuntimeError("pyobjc-framework-Metal not installed.")

    if A.ndim == 2:
        A = A[None]
        squeeze_out = True
    else:
        squeeze_out = False

    B, n, m = A.shape
    if n != m:
        raise ValueError(f"Square required, got {A.shape}")
    if n % BLOCK_SIZE != 0:
        raise ValueError(f"n={n} must be a multiple of BLOCK_SIZE={BLOCK_SIZE}")

    num_blocks = n // BLOCK_SIZE
    if num_blocks < 2:
        raise ValueError(
            f"Block Jacobi needs num_blocks ≥ 2 (n ≥ {2 * BLOCK_SIZE}). "
            f"Use _jacobi_metal.jacobi_eigh for smaller matrices."
        )

    A32 = np.ascontiguousarray(A, dtype=np.float32).copy()

    block_sched, b_rounds, b_pairs = _block_schedule(num_blocks)
    sub_sched = _sub_schedule()

    a_fro = float(np.linalg.norm(A32.reshape(B, -1), axis=-1).mean())
    tol_abs = rel_tol * a_fro

    st = _get_state()
    device = st["device"]
    pipeline = st["pipeline"]
    queue = st["queue"]
    options = Metal.MTLResourceStorageModeShared

    nb = A32.nbytes
    buf_A = device.newBufferWithBytes_length_options_(A32.tobytes(), nb, options)
    buf_V = device.newBufferWithLength_options_(nb, options)
    buf_block_sched = device.newBufferWithBytes_length_options_(
        np.ascontiguousarray(block_sched).tobytes(),
        block_sched.nbytes, options,
    )
    buf_sub_sched = device.newBufferWithBytes_length_options_(
        np.ascontiguousarray(sub_sched).tobytes(),
        sub_sched.nbytes, options,
    )

    buf_n = device.newBufferWithBytes_length_options_(struct.pack("I", n), 4, options)
    buf_nb_blocks = device.newBufferWithBytes_length_options_(struct.pack("I", num_blocks), 4, options)
    buf_max_bs = device.newBufferWithBytes_length_options_(struct.pack("I", max_block_sweeps), 4, options)
    buf_b_rounds = device.newBufferWithBytes_length_options_(struct.pack("I", b_rounds), 4, options)
    buf_b_pairs = device.newBufferWithBytes_length_options_(struct.pack("I", b_pairs), 4, options)
    buf_tol = device.newBufferWithBytes_length_options_(struct.pack("f", tol_abs), 4, options)
    buf_max_is = device.newBufferWithBytes_length_options_(struct.pack("I", max_inner_sweeps), 4, options)

    # Threadgroup memory: sub_A + sub_Q + sub_cs + off_sum
    SUB_DIM = 2 * BLOCK_SIZE
    SUB_NN = SUB_DIM * SUB_DIM
    SUB_NPAIRS = SUB_DIM // 2
    threadgroup_bytes = (SUB_NN + SUB_NN + SUB_NPAIRS * 2 + 1) * 4

    cmd = queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(buf_A, 0, 0)
    enc.setBuffer_offset_atIndex_(buf_V, 0, 1)
    enc.setBuffer_offset_atIndex_(buf_n, 0, 2)
    enc.setBuffer_offset_atIndex_(buf_nb_blocks, 0, 3)
    enc.setBuffer_offset_atIndex_(buf_max_bs, 0, 4)
    enc.setBuffer_offset_atIndex_(buf_block_sched, 0, 5)
    enc.setBuffer_offset_atIndex_(buf_b_rounds, 0, 6)
    enc.setBuffer_offset_atIndex_(buf_b_pairs, 0, 7)
    enc.setBuffer_offset_atIndex_(buf_sub_sched, 0, 8)
    enc.setBuffer_offset_atIndex_(buf_tol, 0, 9)
    enc.setBuffer_offset_atIndex_(buf_max_is, 0, 10)
    enc.setThreadgroupMemoryLength_atIndex_(threadgroup_bytes, 0)

    max_threads = pipeline.maxTotalThreadsPerThreadgroup()
    threads_per_group = min(SUB_DIM * SUB_DIM, max_threads)
    threads_per_group = max(threads_per_group, SUB_DIM)

    grid = Metal.MTLSize(B * threads_per_group, 1, 1)
    tgsz = Metal.MTLSize(threads_per_group, 1, 1)
    enc.dispatchThreads_threadsPerThreadgroup_(grid, tgsz)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    a_out = np.frombuffer(buf_A.contents().as_buffer(nb), dtype=np.float32).copy().reshape(B, n, n)
    v_out = np.frombuffer(buf_V.contents().as_buffer(nb), dtype=np.float32).copy().reshape(B, n, n)

    D = np.diagonal(a_out, axis1=-2, axis2=-1).copy()
    sort_idx = np.argsort(D, axis=-1)
    D = np.take_along_axis(D, sort_idx, axis=-1)
    V = np.take_along_axis(v_out, sort_idx[:, None, :], axis=-1)

    if squeeze_out:
        return D[0], V[0]
    return D, V
