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

The Riemannian variant (planned for V2 of this fork) is from:

- Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019).
  *A Riemannian Modification of Artifact Subspace Reconstruction for EEG
  Artifact Handling.* Frontiers in Human Neuroscience, 13.
  https://doi.org/10.3389/fnhum.2019.00141

---

## Why GPU

`asrpy` is single-threaded numpy + scipy. On large EEG configurations (128
channels, ≥30 min recordings, hyperscanning sessions), two pain-points
appear:

1. **Memory pressure / OOM.** The per-window covariance stack `Xcov` of
   shape `(n_channels, n_channels, n_windows)` blows past available RAM
   when blocks are pre-allocated. `asrpy` mitigates with `mem_splits`, but
   peaks remain large.
2. **Wall-clock cost.** `asr_process` calls `np.linalg.eigh` once per
   window inside a Python loop. For a 30 min × 128 ch session at 256 Hz
   with `stepsize=32`, that's ~14 000 sequential eigh calls.

`asrpy_gpu` keeps the algorithm bit-near `asrpy` (see
[`docs/equivalence.md`](docs/equivalence.md)) but routes the
linear-algebra-heavy paths to a torch backend. Apple MPS is the primary
target (the lab uses M4 Max and M-series workstations); CUDA is on the
roadmap as V4.

---

## How the GPU port works

The torch backend follows two simple rules:

* **Stay on device.** All matmul, einsum, cumsum, indexing happens on the
  resolved device (MPS / CUDA / CPU).
* **Cross only when necessary.** Two operations cross back to CPU:
  * `scipy.signal.lfilter` for the Yule-Walker IIR filter (sequential by
    nature). V5 may replace this with a Metal parallel-scan IIR if
    profiling demands it.
  * `torch.linalg.eigh`, which is not yet implemented on MPS as of
    torch 2.11. We do an explicit CPU round-trip in **float64** (rather
    than the float32 we use on MPS), which actually improves precision.

The biggest win is in `transform`: the per-window eigh loop is replaced
by a single batched `torch.linalg.eigh` call (CPU-side on MPS for now,
device-native on CUDA). Reconstruction matrices are batched too. The
sequential cosine-blending loop stays Python-orchestrated because of the
`last_R` / `last_trivial` carry, but only orchestrates — the math runs on
device.

---

## Numerical equivalence

Tested via three tiers (see [`docs/equivalence.md`](docs/equivalence.md)
and `tests/`):

| Tier | Data | Tolerance |
|------|------|-----------|
| **L1 — synthetic** | Pink-noise + 30 transient spikes, seeded RNG | numpy: bit-near `asrpy`; MPS: `rtol≤1e-4` |
| **L2 — `asrpy` reference** | EEGLAB `test_raw.set` from `mne.datasets.testing` | same as above |
| **L3 — realistic** (benchmark only, not CI) | `mne.datasets.sample` + private MAGSTIM 128 ch files | corr ≥ 0.999 |

Currently (V1, scaffold):

* numpy backend ↔ `asrpy`: **bit-near** on synthetic data.
* torch CPU backend ↔ `asrpy`: max abs diff ~5e-14 (float64).
* torch MPS backend ↔ `asrpy`: max abs diff ~3e-5 (float32 cumulative).

---

## Install

This project uses [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/ramdam17/asrpy_gpu.git
cd asrpy_gpu
uv sync --extra torch        # core + torch
# or, for development (asrpy reference, tests, ruff):
uv sync --all-groups --extra torch
```

Run the tests:

```bash
uv run pytest                 # full suite (downloads MNE testing dataset)
uv run pytest -m "not slow"   # quick subset
```

Run the benchmarks:

```bash
uv run python benchmarks/bench_calibrate.py
uv run python benchmarks/bench_process.py
# Results land in benchmarks/results/.
```

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md). Current scope is **V1 — Euclid +
torch+MPS**. V2 adds Riemannian ASR (CPU numpy reference first), V3 ports
it to MPS, V4 brings CUDA, V5 explores a Metal parallel-scan IIR.

## License

BSD-3-Clause. See [`LICENSE`](LICENSE) — copyright is shared with the
original `asrpy` authors.

## Citing

If you use `asrpy_gpu` in published work, please cite both the original
ASR papers (Mullen 2015, Kothe & Jung 2016) and this fork. A
`CITATION.cff` file will be added at the V1 release tag.
