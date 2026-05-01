# ASRPy-GPU — Roadmap

> Perf-first versioning: every V is a measurable speedup over the
> previous one or a new feature. The original numbering put Riemann
> at V2; the actual sequence of optimisations pushed Riemann to V6.

## V1 — Euclid + torch+MPS baseline (done)

**Scope.** Drop-in replacement for `asrpy` (Euclidean ASR) with a torch
backend on Apple MPS, NVIDIA CUDA, or CPU. Numpy is the reference.

**Key technique.** Replace `asrpy`'s per-window Python loop over
`np.linalg.eigh` with a single batched `torch.linalg.eigh` (CPU
fallback inside one call on MPS).

**Result.** ~6–10× over numpy on a typical 128 ch × 120 s session.

---

## V2 — MPS-native eigh + pinv (done)

**Scope.** Eliminate the silent CPU fallbacks in the hot loop:
- Custom **Metal compute kernel** (cyclic parallel Jacobi with
  per-matrix early-exit) for `eigh`.
- **`pinv` via `eigh(AᵀA)`** so the second decomposition uses the
  same Metal kernel.
- **Skip-trivial windows** (R = I) before either eigh runs.

**Result.** 10–15× over numpy on 128 ch × 120 s.

**Things tried but not shipped.**
- *Pure-torch Jacobi.* Algorithm correct, ~80× slower on MPS due to
  the launch overhead of thousands of small kernels. Kept as portable
  reference and likely-good CUDA path.
- *Fused eigh+pinv kernel.* Built and verified, but no end-to-end
  speedup once skip-trivial captures the workload.

---

## V3 — float4 + double-tile block Jacobi (done)

**Scope.** Push past the unified-memory bandwidth wall on bigger
configurations:
- ``float4`` row-side updates in the std Jacobi kernel (cols stay
  scalar — non-contiguous in row-major).
- **Double-tile block Jacobi** for n ≥ 256: 2b × 2b sub-matrix in
  threadgroup memory, double-tiled apply-Q step that keeps every
  device-memory read amortised over 2b multiply-adds.
- Hybrid dispatch: std Jacobi for n ≤ 128, block Jacobi for n ≥ 256.

**Result.** 18× on 128 ch × 120 s (up from ~10× in V2); 18× on
256 ch × 120 s (up from ~5× in V2).

---

## V4 — tile-based memory access deeper (in flight)

**Scope.** Take the cache-blocking idea further at very high channel
counts (n ≥ 256). Candidate ideas:
- Triangular storage of the symmetric matrix (half the bandwidth).
- Outer + inner tiling for the col + V updates of the std Jacobi.
- Mixed-precision accumulation (compute in fp32, accumulate in
  thread-local fp64-emulated to recover precision at no bandwidth
  cost).

**Reference.** V3.

**Status.** Profile-gated. The 256ch case still has headroom — the
double-tile block Jacobi at 18× isn't yet hitting the bandwidth
ceiling (~30 GB/s short).

---

## V5 — lfilter on GPU (parallel-scan IIR) (deferred)

**Scope.** Replace `scipy.signal.lfilter` with a Metal parallel-scan
IIR (Blelloch up-down sweep on the order-8 transition matrix).

**Why deferred.** Profiling shows lfilter accounts for ~3% of the
runtime in V3. Custom parallel-scan IIR is sensitive to float32
stability, ~1–2 days of work for a ~3% gain. Will revisit when other
bottlenecks fall below this threshold.

**Reference.** V3 std `scipy.signal.lfilter` output.

---

## V6 — Riemann CPU reference

**Scope.** Add `method="riemann"` to the numpy backend, following
Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019).
*A Riemannian Modification of Artifact Subspace Reconstruction for EEG
Artifact Handling.* Frontiers in Human Neuroscience, 13.
https://doi.org/10.3389/fnhum.2019.00141

**Why CPU first.** `asrpy` does not implement Riemannian ASR — the
docstring is there but the implementation falls back to Euclidean. We
have no Python reference to validate against. So we build the numpy
reference ourselves, validate piecewise against `pyriemann`'s
Karcher mean and against the Blum 2019 MATLAB code, then port to
GPU in V7.

**Reference.**
- Karcher mean: `pyriemann.utils.mean_covariance(metric='riemann')`.
- End-to-end pipeline: Blum 2019 MATLAB code.

**Milestones.**
- Karcher mean implemented in numpy, tested against pyriemann at
  `rtol=1e-7`.
- PGA in tangent space, end-to-end test at corr > 0.99 against the
  MATLAB reference.

---

## V7 — Riemann torch+MPS

**Scope.** Port V6 to torch+MPS using the same Metal eigh kernel from
V2 plus matrix exp / log built from `eigh` on SPD matrices.

**Reference.** V6 numpy.

**Milestones.**
- Riemann torch backend matches V6 numpy at `rtol=1e-4` (MPS float32).
- Full V1+V2+V3 (Euclid) and V6+V7 (Riemann) matrix of backends green.

---

## V8 — torch+CUDA support

**Scope.** Same code as V7, exposing `device='cuda'` for NVIDIA GPUs
(lab cluster, Compute Canada). The `_device.resolve_device()` already
considers CUDA — V8 just extends test coverage and CI to GPU runners
when available.

**Reference.** V1+V3+V7.

**Milestones.**
- CI passes on a CUDA runner (GitHub Actions self-hosted or
  Compute Canada-hosted).
- Numerical equivalence between MPS float32 and CUDA float32 at
  `rtol=1e-5`.
- Numerical equivalence between CUDA float64 and asrpy CPU at
  `rtol=1e-9`.

---

## Out of scope (no version planned)

- Real-time / streaming ASR (`asrpy` itself is offline).
- Other artifact removal methods (ICA, autoreject, RANSAC) — separate
  libraries.
- Distributed multi-GPU inference (single dyad fits comfortably on one
  device).
