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

## V4 — outer-exit + batched trivial sync (done)

**Scope.** Two micro-opts on top of V3:
- Block Jacobi outer-sweep early-exit (sub_A re-used as reduction
  scratch; the check is essentially free even when it doesn't trigger).
- Batched ``trivial_all.cpu()`` before the blending Python loop,
  replacing 300 per-iteration ``.item()`` MPS→CPU syncs with one.

**Result.** +13% on 128 ch × 120 s (15× vs numpy); +5% on 256 ch.

---

## V5 — three perf candidates investigated (no shipped gain)

Three paths tried, none shipped:

* **V5.1 — Convergence on real ASR Xcov vs random.** Both regimes
  converge at the same rate (the Yule-Walker pre-filter spreads energy
  across the spectrum so eigvals aren't clustered). max_block_sweeps
  default of 8 stays.
* **V5.2 — Parallel-scan IIR Metal kernel.** Hillis–Steele scan over
  (8×8 M, 8-vec c) state-space pairs. Correct (corr 0.99999 first
  samples). 8–12× **slower** than scipy.signal.lfilter on our problem
  sizes — order=8 is too small to amortise the per-combine 8×8 matmul
  against scipy's tight C loop. Code preserved as evidence.
* **V5.3 — Lanczos eigh prototype (pure torch).** Correct (max abs err
  vs LAPACK = 1.1e-7). Speed roughly tied with block Jacobi due to
  Python orchestration of 256 iterations. To win 3–5× on eigh would
  require writing the whole loop inside a single Metal compute kernel.

---

## V6 — pinv-via-solve (hybrid, n ≥ 256) (done)

**Scope.** ``torch.linalg.solve`` is native MPS, beats the block-Jacobi
inside ``pinv_via_eigh`` by 1.06–1.4× at n ≥ 256. Hybrid dispatch:
solve for n ≥ 256, eigh for n < 256 (where std-Jacobi inside
pinv_via_eigh wins by 3–10×).

**Result.** Neutral on the realistic 128/256 ch pipelines (the
solve advantage at the typical batch sizes saves ~150 ms of a 7.7 s
pipeline = ~2-3%). Helper kept for future use (CUDA path).

**Tikhonov ε** sweep-validated: `eps=1e-12` (well below float32 noise)
preserves bit-equivalence with `asrpy`. Looser eps degrades corr
sharply.

---

## Final state of the perf series

| Config | numpy ref | asrpy_gpu | speedup |
|---|---|---|---|
| 128 ch × 120 s | ~13 s | 0.87 s | **15×** |
| 256 ch × 120 s | ~70-150 s | 7.6 s | **9-20×** |

Diminishing-returns regime reached. Each V from V3 onward gained
5-15%. The remaining ambitious lever is a **full Metal Lanczos
kernel** (potential 1.5-2× pipeline speedup, ~1-2 days of focused MSL
work, deferred).

---

## V7 — Riemann CPU reference (next session)

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

## V8 — Riemann torch+MPS

**Scope.** Port V7 to torch+MPS using the same Metal eigh kernel from
V2 plus matrix exp / log built from `eigh` on SPD matrices.

**Reference.** V7 numpy.

**Milestones.**
- Riemann torch backend matches V7 numpy at `rtol=1e-4` (MPS float32).
- Full Euclid (V1–V6) and Riemann (V7+V8) matrix of backends green.

---

## V9 — torch+CUDA support

(Was V8 in the previous numbering.)

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
