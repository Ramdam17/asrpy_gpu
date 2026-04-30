# Numerical equivalence — design notes

This document explains the **equivalence guarantees** between `asrpy_gpu`
and the upstream `asrpy`, and the **tolerances** chosen for the test
suite. Tolerances are stored centrally in
[`config/test_tolerances.yaml`](../config/test_tolerances.yaml).

## What we compare

ASR's outputs of interest are:

| Quantity | Where | Comparable? |
|---|---|---|
| `M` (mixing matrix) | `calibrate` output | **Yes**, directly |
| `T` (threshold matrix) | `calibrate` output | Indirectly — sign-ambiguous |
| Cleaned signal | `process` output | **Yes**, directly |

`T = diag(mu + cutoff * sig) @ V.T` where `V` comes from `eigh(M)`. Any
column-sign flip of `V` flips a row of `T`. So `T` may differ
component-wise across implementations, even though the algorithm gives
identical results downstream. We therefore compare `M` and the cleaned
signal directly, and treat `T` as a sanity check (finite, correct shape).

## Why sign-invariance is structural

Two facts make ASR invariant to eigenvector signs:

1. The keep-mask in `process` is computed via `(T @ V_window) ** 2` —
   squaring kills the sign.
2. The reconstruction matrix `R = M @ pinv(keep[:, None] * (V_window.T @
   M)) @ V_window.T` couples `V_window.T @ M` and `V_window.T` with the
   same column-sign choice; a sign flip on a column of `V_window` flips
   one row of `V_window.T @ M` and one row of `V_window.T` consistently,
   so `R` is invariant.

The test suite therefore relies on `M` and the cleaned signal as the
authoritative quantities.

## Tolerances and where they come from

| Backend / device | dtype | rtol | atol | Reason |
|---|---|---|---|---|
| `numpy_backend` ↔ `asrpy` | float64 | 1e-6 | 1e-8 | Same code path; differences are pure API translation. Empirically: bit-exact on synthetic data. |
| torch CPU ↔ `asrpy` | float64 | 1e-9 | 1e-10 | Different BLAS path through torch but stays float64. |
| **torch MPS ↔ `asrpy`** | float32 | **1e-4** | **1e-5** | MPS forbids float64. Float32 cumulative error across the Yule-Walker filter, block covariance, geometric median, eigendecomposition, batched pinv, and reconstruction. Empirically: ~3e-5 on synthetic data. |
| torch CUDA ↔ `asrpy` | float64 | 1e-9 | 1e-10 | IEEE-compliant float64 on GPU. (V4.) |
| End-to-end signal correlation | — | — | — | `corr(asrpy_clean, gpu_clean) ≥ 0.999` is required regardless of backend. |

These tolerances are also documented as defaults in the project's
`gpu-optimization` skill.

## What can break determinism

Once seeded, the algorithm itself is deterministic — confirmed by reading
the upstream source: no `np.random` / `RandomState` / seed in
`asrpy/asr.py` or `asr_utils.py`. The only sources of non-determinism in
practice are:

1. **BLAS thread scheduling**. `OpenBLAS` / `MKL` may sum reductions in
   different orders across threads, producing ~1e-12 jitter in float64.
   Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for the strictest
   reproducibility. Tests do not pin threading; the bit-exact tests use
   the numpy backend where intra-process determinism is preserved by
   numpy.
2. **MPS BLAS**. The Apple MPS backend's matmul kernels are also
   reduction-ordered and can produce ~1e-7 float32 jitter. Test
   tolerances are loose enough to absorb this.
3. **`scipy.linalg.sqrtm` (numpy backend)**. Schur-based; deterministic
   for the same input on the same machine.
4. **`scipy.linalg.eigh` vs `numpy.linalg.eigh`** (mixed in `asrpy`):
   typically the same LAPACK routine but may differ at the last bits.

## Known V1 limitations

* `torch.linalg.eigh` is not implemented on MPS in torch 2.11
  (`aten::_linalg_eigh.eigenvalues`). We handle this by moving to CPU,
  computing in float64, and moving back. Performance is still better
  than the numpy backend's per-window Python loop because the eigh is
  batched.
* `torch.linalg.pinv` uses SVD internally, which falls back from MPS to
  CPU silently. Same trade-off as eigh.
* `scipy.signal.lfilter` is sequential and stays on CPU. Profiling will
  decide whether V5 (Metal parallel-scan IIR) is worth the effort.

## What we _do not_ guarantee

* Bit-equivalence between MPS and CPU. Float32 vs float64 makes this
  impossible.
* Bit-equivalence between two MPS runs across machines (different unified
  memory bandwidth and reduction ordering may differ). The intra-machine
  determinism test does pass within `atol=1e-6` for MPS, however.

## Re-running the equivalence suite

```bash
uv run pytest tests/test_equivalence.py -v
```

The MPS and CUDA cases are auto-skipped on CI runners that lack the
hardware (markers `@needs_mps`, `@needs_cuda`).
