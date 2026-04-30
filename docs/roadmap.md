# ASRPy-GPU — Roadmap

> Each version closes with a tagged release and a section appended to `docs/benchmark.md`.

## V1 — Euclid + torch+MPS (current)

**Scope.** Drop-in replacement for `asrpy` (Euclidean ASR) with a torch+MPS backend on Apple Silicon and a numpy reference backend. Numerical equivalence vs `asrpy` validated.

**Reference.** `asrpy` (CPU, scipy/numpy).

**Milestones.**
- Numpy backend reaches `rtol=1e-9` vs `asrpy` on EEGLAB test file.
- Torch+MPS backend reaches `rtol=1e-4` (float32 cumulative bound) on the same file.
- Benchmark documents a measurable speedup on transform for ≥64 channels.

---

## V2 — Riemann (CPU numpy reference first)

**Scope.** Add `method="riemann"` to the numpy backend, following Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019). *A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling.* Frontiers in Human Neuroscience, 13. https://doi.org/10.3389/fnhum.2019.00141

**Why CPU first.** `asrpy` does not implement Riemannian ASR — it has the docstring and a `warnings.warn` fallback to Euclid. We have no Python reference to validate against. So we build the numpy reference ourselves, validate it piecewise against `pyriemann` (Karcher mean) and the original MATLAB code (end-to-end), then move it to GPU in V3.

**Reference.**
- Karcher mean: `pyriemann.utils.mean_covariance(metric='riemann')`.
- End-to-end pipeline: Blum 2019 MATLAB code (or our own numpy translation re-validated piecewise).

**Milestones.**
- Karcher mean implemented in numpy, tested against pyriemann at `rtol=1e-7`.
- PGA in tangent space, end-to-end test at correlation > 0.99 against MATLAB reference.

---

## V3 — Riemann on torch+MPS

**Scope.** Port V2 numpy implementation to torch+MPS using `torch.linalg.eigh` (already used in V1) plus `torch.linalg.matrix_exp` and a `matrix_log` built from eigh on SPD matrices.

**Reference.** V2 numpy (already tested in V2).

**Milestones.**
- Riemann torch backend matches V2 numpy at `rtol=1e-4` (MPS float32).
- Full V1+V3 (Euclid + Riemann × numpy + torch+MPS) matrix of backends green.

---

## V4 — torch+CUDA support

**Scope.** Same code as V3, exposing `device='cuda'` for NVIDIA GPUs (lab cluster, Compute Canada). The `_device.resolve_device()` already considers CUDA — V4 just extends test coverage and CI to GPU runners when available.

**Reference.** V1+V3.

**Milestones.**
- CI passes on a CUDA runner (GitHub Actions self-hosted or Compute Canada-hosted).
- Numerical equivalence between MPS float32 and CUDA float32 at `rtol=1e-5`.
- Numerical equivalence between CUDA float64 and asrpy CPU at `rtol=1e-9`.

---

## V5 — Custom Metal kernel for parallel-scan IIR (profile-gated)

**Scope.** Replace `scipy.signal.lfilter` (sequential CPU IIR for `yulewalk_filter`) with a Metal compute shader implementing a parallel scan IIR (Blelloch-style or recursive doubling) — but **only** if profiling in V1–V4 shows that `lfilter` is the residual bottleneck after the rest of the pipeline is on GPU.

**Why gated.** `lfilter` is called a small fixed number of times per fit/transform, not per window. It may not justify the engineering cost.

**Reference.** V1 numpy lfilter output as ground truth.

**Milestones.**
- Profiling report shows `lfilter` >25% of total wall-clock time on representative load.
- Metal kernel matches numpy `lfilter` at `rtol=1e-3` (float32).
- Optional dependency (`pyobjc-framework-Metal`), `UserWarning` fallback.

---

## Out of scope (no version planned)

- Real-time / streaming ASR (asrpy itself is offline).
- Other artifact removal methods (ICA, autoreject, RANSAC) — separate libraries.
- Distributed multi-GPU inference (single dyad fits comfortably on one device).
