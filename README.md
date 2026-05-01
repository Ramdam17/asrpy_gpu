# asrpy_gpu

GPU-accelerated **Artifact Subspace Reconstruction** (ASR) for EEG, drop-in
compatible with [`asrpy`](https://github.com/DiGyt/asrpy).

```python
from asrpy_gpu import ASR

asr = ASR(sfreq=raw.info["sfreq"], cutoff=20)   # backend="auto" by default
asr.fit(raw)
clean = asr.transform(raw)
```

The `asr.fit / asr.transform` API is identical to `asrpy`; the constructor
adds a single `backend` argument (`"auto" | "numpy" | "torch"`).

---

## Install

End-user install (recommended for using the library):

```bash
# core (numpy backend only — works everywhere, no GPU)
uv add git+https://github.com/Ramdam17/asrpy_gpu

# core + torch (CPU / MPS / CUDA)
uv add 'asrpy_gpu[torch] @ git+https://github.com/Ramdam17/asrpy_gpu'

# core + torch + the Apple Metal kernel for full MPS-native eigh / pinv
uv add 'asrpy_gpu[torch,metal] @ git+https://github.com/Ramdam17/asrpy_gpu'
```

Or with `pip`:

```bash
pip install 'asrpy_gpu[torch,metal] @ git+https://github.com/Ramdam17/asrpy_gpu'
```

Minimum: Python 3.12, numpy ≥ 1.26, scipy ≥ 1.11, mne ≥ 1.6. The `torch`
extra pulls torch ≥ 2.2, `metal` adds `pyobjc-framework-Metal` (Apple
Silicon only).

Developer install (for contributing — clones the repo, fetches `asrpy`
from upstream as a numerical reference, installs ruff and pytest):

```bash
git clone https://github.com/Ramdam17/asrpy_gpu.git
cd asrpy_gpu
uv sync --all-groups --extra torch --extra metal
uv run pytest
```

---

## Where this comes from

`asrpy_gpu` is a fork of [`asrpy`](https://github.com/DiGyt/asrpy) by Dirk
Gütlin & Nicolas Barascud (BSD-3), which itself is a Python port of
EEGLAB's `clean_rawdata` ASR. The algorithm is described in:

- Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S.,
  Jung, T.-P., & Cauwenberghs, G. (2015). *Real-time Neuroimaging and
  Cognitive Monitoring Using Wearable Dry EEG.* IEEE TBME, 62(11),
  2553–2567.
- Kothe, C. A. E., & Jung, T.-P. (2016). *U.S. Patent Application No.
  14/895,440.* https://patents.google.com/patent/US20160113587A1/en

The Riemannian variant (planned for V6 of this fork) is from Blum et al.
(2019), *A Riemannian Modification of ASR*,
[Frontiers in Human Neuroscience](https://doi.org/10.3389/fnhum.2019.00141).

---

## Why GPU

`asrpy` is single-threaded numpy + scipy. On large EEG configurations (128
channels, ≥ 30 min recordings, hyperscanning sessions), two pain-points
appear:

1. **Memory pressure / OOM.** The per-window covariance stack `Xcov` of
   shape `(n_channels, n_channels, n_windows)` blows past available RAM
   when blocks are pre-allocated. `asrpy` mitigates with `mem_splits`, but
   peaks remain large.
2. **Wall-clock cost.** `asr_process` calls `np.linalg.eigh` once per
   window inside a Python loop. For a 30 min × 128 ch session at 256 Hz
   with `stepsize=32`, that's ~14 000 sequential eigh calls.

`asrpy_gpu` keeps the algorithm equivalent to `asrpy` (see
[`docs/equivalence.md`](docs/equivalence.md)) but routes the
linear-algebra-heavy paths to a torch backend with custom Metal kernels
on Apple Silicon. The lab targets M-series Macs (MAGSTIM EGI 128 ch
sessions); CUDA is on the roadmap.

---

## Speedup vs the original `asrpy`

Real numbers from `benchmarks/bench_process.py` on an Apple M4 Max,
seeded random data, `process()` only:

| channels | duration | numpy (s) | asrpy_gpu (MPS) | speedup |
|---------:|---------:|----------:|----------------:|--------:|
| 64  | 60 s  |  1.46 |  0.13 | **11×** |
| 64  | 120 s |  3.0  |  0.24 | **12×** |
| 128 | 60 s  | 23.0  |  0.63 | **36×** |
| 128 | 120 s | 38.1  |  1.0  | **38×** |
| 256 | 60 s  | 35.2  |  3.9  | **9×**  |
| 256 | 120 s | 146.0 |  8.2  | **18×** |

(*your numbers will vary with system load, signal characteristics, and
artifact density: ASR's "trivial" branch is much cheaper than the
artifact branch, and the speedup grows with both channel count and
duration*)

---

## How the GPU port works

The torch backend follows two simple rules:

* **Stay on device.** All matmul, einsum, cumsum, indexing, and the
  per-window eigh / pinv happen on the resolved device (MPS / CUDA / CPU).
* **Cross only when necessary.** ``scipy.signal.lfilter`` (the
  Yule-Walker IIR) is sequential and stays on CPU; we round-trip the
  filtered signal back to the GPU after that. The optional Metal
  kernels eliminate the previous CPU fallback for `eigh` and `pinv`.

Six perf tiers stack:

1. **V1 — torch+MPS baseline.** Batched `torch.linalg.eigh` (CPU
   fallback inside a single call) replaces `asrpy`'s per-window Python
   loop. ~6–10× over numpy.
2. **V2 — Metal Jacobi kernel + pinv-via-eigh + skip-trivial.** Custom
   Metal compute kernel for batched symmetric eigh (cyclic parallel
   Jacobi, per-matrix early-exit). `pinv(masked) = V diag(1/D⁺) Vᵀ Aᵀ`
   reuses the same kernel. Trivial windows skip the reconstruction.
3. **V3 — float4 + double-tile block Jacobi.** Row updates use
   ``float4`` reads/writes; for n ≥ 256 channels, switch to a
   double-tiled block Jacobi that keeps the active 2b × 2b sub-matrix
   in 32 KB threadgroup memory.
4. **V4 — outer-exit + batched trivial sync.** Block-Jacobi outer
   convergence early-exit; one MPS→CPU sync for the trivial flag
   instead of one per window.
5. **V5 — investigations** (no shipped change). Real Xcov vs random,
   parallel-scan IIR, Lanczos prototype — all measured, none viable.
6. **V6 — pinv-via-solve hybrid (n ≥ 256).** `torch.linalg.solve` is
   native MPS and beats the block-Jacobi inside pinv-via-eigh at large
   n; small-n stays on the Jacobi path.

See [`docs/roadmap.md`](docs/roadmap.md) for the full perf-first
versioning and what's still on the table.

---

## Numerical equivalence

Tested via three tiers (`docs/equivalence.md`, `tests/`):

| Tier | Data | Tolerance |
|------|------|-----------|
| **L1 — synthetic** | Pink-noise + transient spikes, seeded RNG | numpy: bit-near `asrpy`; MPS: `rtol ≤ 1e-4` |
| **L2 — `asrpy` reference** | EEGLAB `test_raw.set` from `mne.datasets.testing` | same as above |
| **L3 — realistic** (benchmark only) | `mne.datasets.sample` + private MAGSTIM 128 ch files | corr ≥ 0.999 |

Empirical residuals:

* numpy backend ↔ `asrpy`: **bit-near** on synthetic data.
* torch CPU backend ↔ `asrpy`: max abs diff ~5e-14 (float64).
* torch MPS backend ↔ `asrpy`: max abs diff ~3e-5 to ~6e-5 (float32
  cumulative across the pipeline).

---

## Roadmap

[`docs/roadmap.md`](docs/roadmap.md). Current state:

```
V1 — Euclid + torch+MPS baseline                            done
V2 — Metal Jacobi kernel + pinv-via-eigh + skip-trivial     done
V3 — float4 + hybrid block Jacobi for n ≥ 256               done
V4 — outer-exit + batched trivial sync                      done
V5 — investigations (real Xcov / IIR scan / Lanczos)        no gain shipped
V6 — pinv-via-solve hybrid for n ≥ 256                      done
V7 — Riemann CPU reference                                  next
V8 — Riemann torch+MPS
V9 — torch+CUDA
```

## License

BSD-3-Clause. See [`LICENSE`](LICENSE) — copyright is shared with the
original `asrpy` authors.

## Citing

If you use `asrpy_gpu` in published work, please cite both the original
ASR papers (Mullen 2015, Kothe & Jung 2016) and this fork. A
`CITATION.cff` file will be added at the next tagged release.
