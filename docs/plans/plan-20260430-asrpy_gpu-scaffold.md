# Plan: ASRPy-GPU — Initial Scaffold (V1)

**Date:** 2026-04-30
**Author:** Rémy Ramadour
**Sprint:** Sprint-01 (V1 scaffold)
**Estimated complexity:** L (multi-file, multi-decision, port + tests + benchmark + docs)

---

## Problem Statement

The reference implementation `asrpy` (DiGyt/asrpy, BSD-3) is a faithful Python port of EEGLAB's MATLAB `clean_rawdata` ASR. It is the de-facto Python ASR. However, on large configurations (128-channel MAGSTIM EGI, long recordings ≥30 min, hyperscanning sessions), it produces out-of-memory errors and is single-threaded on CPU. We need a GPU-accelerated fork that:

1. Resolves the OOM on big configurations.
2. Stays numerically equivalent to `asrpy` (so we can re-run pipelines without changing analysis decisions).
3. Keeps the same high-level API (`asr.fit(raw)`, `asr.transform(raw)`).
4. Targets Apple Silicon (MPS) first — extending later to CUDA for the lab's HPC nodes.

---

## Scientific Rationale

**ASR — Artifact Subspace Reconstruction**

- Original method: Mullen et al., 2015. *Real-time Neuroimaging and Cognitive Monitoring Using Wearable Dry EEG.* IEEE Trans Biomed Eng.
- Patent: Kothe & Jung, 2016. *U.S. Patent Application No. 14/895,440.*
- Riemannian variant: Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019). *A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling.* Frontiers in Human Neuroscience, 13. https://doi.org/10.3389/fnhum.2019.00141
- Validation in BCI/EEG: Chang et al., 2018. *Evaluation of Artifact Subspace Reconstruction for Automatic EEG Artifact Removal.*

**Determinism analysis (verified 2026-04-30 against asrpy source):**

- No `np.random` / `RandomState` / seed in `asrpy/asr.py` or `asrpy/asr_utils.py`.
- All operations are: deterministic linear algebra (`scipy.linalg.eigh`, `sqrtm`, `pinv`), deterministic iterative algorithms (Weiszfeld geometric median with fixed init `np.mean(X, 0)`, tol=1e-5, max_iter=500), and deterministic grid searches (`fit_eeg_distribution` over fixed `shape_range` and `step_sizes`).
- The algorithm is **structurally invariant to eigenvector sign flips**: critical operations use `(T @ V)**2` (asr.py:625), `V V.T`-style products, or `abs(V.T @ X)` (asr.py:448) — all sign-invariant. Therefore numerical equivalence tests on `R`, `M`, and the cleaned signal do not require sign alignment.
- Caveats: BLAS thread non-associativity can produce ~1e-12 relative jitter; Apple MPS is float32-only, so end-to-end equivalence is bounded by float32 precision (~1e-5 relative).

**Why GPU = torch+MPS, not Metal kernels (V1):**

- ASR is dominated by batched dense linear algebra (`eigh`, `pinv`) — exactly what `torch.linalg` already optimizes.
- No `sign()`-before-reduction, no per-element loops requiring custom kernels, no custom accumulation patterns.
- The biggest gain is moving the per-window Python loop `for j: np.linalg.eigh(Xcov[:, :, j])` (asr.py:618–622) to a single batched `torch.linalg.eigh` call.
- `torch.linalg.sqrtm` does not exist, but `Uavg` is SPD so `sqrtm(Uavg) = V @ diag(sqrt(D)) @ V.T` via `eigh` — same primitive as the rest.

---

## Implementation Approach

### Data flow

```
Input: mne.io.Raw (n_channels, n_samples) — float64 numpy
  ↓ Backend resolution: auto / numpy / torch (mps preferred)
  ↓ ASR.fit:
  ↓   clean_windows()       — per-channel threshold (vectorisable)
  ↓   yulewalk_filter()     — IIR, scipy.signal.lfilter (CPU, sequential)
  ↓   block_covariance()    — torch.einsum
  ↓   geometric_median()    — Weiszfeld via torch.cdist
  ↓   sqrtm(Uavg)           — via torch.linalg.eigh on SPD
  ↓   eigh(M)               — torch.linalg.eigh
  ↓   fit_eeg_distribution  — vectorised grid search per channel
  ↓ → store M, T (mixing + threshold matrices)
  ↓ ASR.transform:
  ↓   yulewalk_filter()     — CPU IIR
  ↓   ma_filter (cumsum)    — torch.cumsum
  ↓   batched eigh(Xcov)    — single GPU call (THE big gain)
  ↓   pinv (batched)        — torch.linalg.pinv
  ↓   cosine blending       — torch broadcast
Output: mne.io.Raw with cleaned data — float64 numpy
```

### Repository structure

```
asrpy_gpu/                         # working directory = current sandbox
├── pyproject.toml                 # uv-managed, torch as optional dep
├── uv.lock
├── README.md                      # heritage, motivation, usage
├── LICENSE                        # BSD-3 (inherits from asrpy)
├── .gitignore
├── .github/workflows/
│   └── ci.yml                     # lint + test (numpy backend, MPS skipped on CI)
├── docs/
│   ├── plans/
│   │   └── plan-20260430-asrpy_gpu-scaffold.md   # this file
│   ├── roadmap.md                 # V1–V5 (per user request)
│   ├── benchmark.md               # protocol, results, plots
│   └── equivalence.md             # tolerance choices, methodology
├── src/asrpy_gpu/
│   ├── __init__.py                # public API: ASR, asr_calibrate, asr_process
│   ├── _device.py                 # resolve_device() per gpu-optimization skill
│   ├── _backend_numpy.py          # reference numpy implementation (baseline)
│   ├── _backend_torch.py          # torch+MPS implementation
│   ├── asr.py                     # high-level ASR class (backend-agnostic)
│   └── _utils.py                  # signal generation for tests, helpers
├── tests/
│   ├── conftest.py                # fixtures: synthetic signal, EEGLAB raw
│   ├── test_determinism.py        # same input → identical output (both backends)
│   ├── test_equivalence.py        # asrpy vs asrpy_gpu (numpy then torch)
│   ├── test_api.py                # ASR.fit/transform signature compatibility
│   └── reference/
│       └── asrpy_reference.py     # thin wrapper around asrpy for clean comparisons
└── benchmarks/
    ├── bench_calibrate.py         # numpy vs torch-mps, varying n_chan / duration
    ├── bench_process.py           # idem for transform
    └── results/
        └── README.md              # placeholders for plots
```

### Public API (drop-in compatibility with asrpy)

```python
from asrpy_gpu import ASR

asr = ASR(
    sfreq=raw.info["sfreq"],
    cutoff=20,
    blocksize=100,
    win_len=0.5,
    win_overlap=0.66,
    max_dropout_fraction=0.1,
    min_clean_fraction=0.25,
    max_bad_chans=0.1,
    method="euclid",          # only euclid in V1; "riemann" in V2+
    backend="auto",           # NEW: "auto" | "numpy" | "torch"
)
asr.fit(raw)
clean_raw = asr.transform(raw)
```

`backend="auto"` resolves via `_device.resolve_device()`: torch+MPS if available, else torch+CUDA, else numpy. Falls back with `UserWarning` if torch is requested but unavailable.

---

## Roadmap (per user request — to be also written to `docs/roadmap.md`)

| Version | Scope | Reference target | Effort |
|---|---|---|---|
| **V1** | euclid, torch+MPS, numpy fallback | `asrpy` (CPU, scipy/numpy) | This plan |
| **V2** | + Riemann (Blum 2019), CPU numpy backend | `pyriemann` for Karcher mean; numpy traduction of Blum's MATLAB code | Medium |
| **V3** | Riemann on torch+MPS | V2 numpy ref | Medium |
| **V4** | torch + CUDA support (HPC: Compute Canada / lab cluster) | V1+V3 (same code, different device) | Small |
| **V5** | Custom Metal kernel for parallel-scan IIR (`yulewalk_filter` GPU-native) — only if profiling shows it's the residual bottleneck | V1 numpy lfilter | High, profile-gated |

Each version closes with a tagged release (`v1.0.0`, `v2.0.0`, ...) and adds a section to `docs/benchmark.md`.

---

## Config / parameters

Following `config-first` philosophy: tolerance defaults, benchmark sizes, and test fixtures are stored in YAML, not hardcoded.

```yaml
# config/test_tolerances.yaml
equivalence:
  numpy_vs_asrpy:
    rtol: 1.0e-6     # same backend, only API translation
    atol: 1.0e-8
  torch_cpu_vs_asrpy:
    rtol: 1.0e-9
    atol: 1.0e-10
  torch_mps_vs_asrpy:
    rtol: 1.0e-4     # float32 cumulative bound across whole pipeline
    atol: 1.0e-5
  signal_correlation_min: 0.999  # corrcoef(asrpy_clean, gpu_clean) >= 0.999

benchmark:
  channels: [32, 64, 128, 256]
  durations_s: [60, 300, 1800]
  sfreq: 256
  warmup_runs: 2
  measured_runs: 5
```

---

## Risks & Unknowns

- **R1 — `torch.linalg.eigh` MPS quirks**: documented to work on SPD/Hermitian matrices in float32, but batched usage on large stacks can fall back to CPU silently. **Mitigation**: explicit assert that the result tensor remains on `device='mps'`; benchmark detects regression.
- **R2 — `lfilter` is sequential**: `scipy.signal.lfilter` cannot trivially go on GPU. **Mitigation**: keep on CPU with explicit transfer, document the trade-off, profile to confirm it's not the dominant cost. If it is, V5 unlocks Metal parallel-scan IIR.
- **R3 — float32 cumulative error in pipeline**: many operations chained may push end-to-end error past 1e-4. **Mitigation**: empirically measure with the 3-tier test plan; if a specific intermediate is the source, evaluate selective float64 (CPU) for that step only.
- **R4 — pre-existing bugs in asrpy** (asr.py:417 and 555: `method == "euclid"` instead of `method = "euclid"` — silent comparison). **Mitigation**: **flag to user, do not silently fix in fork** (per CLAUDE.md "never make unsolicited changes" rule). Open an upstream issue, decide explicitly.
- **R5 — MNE Raw I/O float precision**: `raw.get_data()` returns float64 by default. Casting to float32 for MPS, back to float64 for the output Raw, may introduce a known 1e-7 round-trip floor. **Mitigation**: document in `docs/equivalence.md`.
- **R6 — Test data licensing**: `mne.datasets.testing` (EEGLAB sample) is BSD-licensed for redistribution. ✓ OK for public repo.

### Unknowns that need empirical answers (gated, will not block scaffold)

- **U1**: Does batched `torch.linalg.eigh` on MPS actually outperform a Python loop with single `np.linalg.eigh` calls beyond a threshold size? **To answer**: micro-benchmark in V1 dev.
- **U2**: Does `geometric_median` benefit from GPU at realistic block counts (~30–100 blocks of 128×128 covariance)? **To answer**: profile both implementations.

---

## Verification Plan

### L1 — Synthetic signal (CI, fast, deterministic)

- Reproducible 30 s × 32 ch signal at 256 Hz, seeded RNG, with injected artifacts (eyeblinks, muscle bursts, electrode pop).
- Tests: shape, no NaN/inf, M and T have correct shape, output rms within expected band.
- Tolerance: `rtol=1e-5, atol=1e-5` (MPS float32).

### L2 — `asrpy` reference equivalence (CI, slower, gold standard)

- Use the same `mne.datasets.testing` EEGLAB `test_raw.set` that asrpy and autoreject both use → apples-to-apples comparison.
- Run asrpy CPU and asrpy_gpu (numpy backend then torch backend) on the **identical** raw.
- Compare: `M`, `T`, cleaned signal — under tolerances from `config/test_tolerances.yaml`.
- Bonus: signal correlation `np.corrcoef(asrpy.clean, gpu.clean)[0,1] >= 0.999`.

### L3 — Realistic load (benchmark, not CI)

- `mne.datasets.sample` (~300 MB, 60 ch MEG/EEG) + Rémy's MAGSTIM 128-ch local files.
- Targets: throughput, peak memory, OOM handling (does the GPU version actually clear OOM cases that fail on asrpy CPU?).
- Visualize: time-series snippets pre/post cleaning for both implementations to detect any qualitative divergence.

### Determinism check

```python
def test_determinism(synthetic_raw):
    asr = ASR(sfreq=256, backend="torch")
    asr.fit(synthetic_raw)
    out1 = asr.transform(synthetic_raw).get_data()
    asr.fit(synthetic_raw)
    out2 = asr.transform(synthetic_raw).get_data()
    np.testing.assert_array_equal(out1, out2)
```

### Skill checklist applied per file

- All Python source: `logging-standards`, `modular-code`, `science-rigor` (cite sources in docstrings).
- Test files: `numerical-testing`, `science-rigor`.
- Benchmark scripts: `benchmarking` (warmup, sync, regression detection).
- Commit messages: `git-discipline` (imperative, scoped).
- README + docs: `doc-writer` style (warm, pedagogical).

---

## Definition of Done (V1)

- [ ] Repo initialized with `uv`, `pyproject.toml` correctly declares torch as optional extra (`asrpy_gpu[torch]`).
- [ ] Public API matches asrpy signature for `ASR.fit(raw)` and `ASR.transform(raw)`.
- [ ] Numpy backend passes L1 + L2 tests at full precision (`rtol=1e-9`).
- [ ] Torch backend passes L1 + L2 tests on MPS at documented tolerances.
- [ ] Determinism test passes for both backends.
- [ ] Benchmark script produces a reproducible CSV + plot for `bench_calibrate` and `bench_process`.
- [ ] README explains: heritage, motivation (OOM on 128 ch big sessions), usage, install, citations.
- [ ] `docs/roadmap.md` lists V1–V5 with scope and target reference.
- [ ] `docs/equivalence.md` documents tolerance reasoning, MPS float32 implications, sign-invariance argument.
- [ ] Git history is clean: scoped imperative commits, plan referenced in each commit.
- [ ] Public GitHub repo (`ramdam17/asrpy_gpu`) created and pushed.
- [ ] CI green: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check`.
- [ ] `verification-before-completion` checklist passed.

---

## Plan history

- **2026-04-30** — Initial draft.
