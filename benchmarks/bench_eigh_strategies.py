"""Compare the three eigh strategies inside the ASR pipeline.

Strategies under test (see :mod:`asrpy_gpu._backend_torch.set_eigh_strategy`):

1. ``cpu_fallback`` — V1 baseline. Move to CPU, eigh in float64, back to MPS.
2. ``jacobi_torch`` — pure-torch parallel Jacobi (portable; expected slow on MPS).
3. ``jacobi_metal`` — Metal compute kernel.

We report wall-clock for ``process()`` (where the bottleneck lives), as well
as a sanity check on numerical agreement vs the asrpy reference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from asrpy_gpu import _backend_numpy as bn
from asrpy_gpu import _backend_torch as bt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bench_eigh")


def make_signal(n_chan: int, dur_s: float, sfreq: float = 256.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(dur_s * sfreq)
    return rng.standard_normal((n_chan, n)).astype(np.float64)


def time_strategy(
    strategy: str,
    X: np.ndarray,
    M: np.ndarray,
    T: np.ndarray,
    sfreq: float,
    *,
    warmup: int = 1,
    runs: int = 3,
) -> tuple[float, np.ndarray]:
    bt.set_eigh_strategy(strategy)
    sync = torch.mps.synchronize

    for _ in range(warmup):
        out = bt.process(X, sfreq=sfreq, M=M, T=T, device="mps")
        sync()

    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        out = bt.process(X, sfreq=sfreq, M=M, T=T, device="mps")
        sync()
        times.append(time.perf_counter() - t0)
    return float(np.min(times)), out


def main() -> None:
    sfreq = 256.0
    cases = [
        (32, 30),
        (64, 30),
        (128, 30),
        (128, 120),
    ]

    strategies = ["cpu_fallback", "jacobi_metal"]
    # jacobi_torch is correct but ~80x slower than cpu_fallback on MPS;
    # we include it on small cases only to stay under a sane budget.
    strategies_small = strategies + ["jacobi_torch"]

    print("=" * 90)
    print(
        f"{'channels':>8} {'duration':>10} {'strategy':>15} "
        f"{'time (s)':>10} {'vs cpu_fallback':>18} {'max abs vs CPU':>18}"
    )
    print("-" * 90)

    rows: list[dict] = []
    for n_chan, dur in cases:
        X = make_signal(n_chan, dur, sfreq)
        M, T = bn.calibrate(X, sfreq=sfreq)

        # cpu_fallback first (reference)
        t_cpu, out_cpu = time_strategy("cpu_fallback", X, M, T, sfreq)
        case_strategies = strategies_small if n_chan <= 32 and dur <= 30 else strategies
        for strat in case_strategies:
            if strat == "cpu_fallback":
                t = t_cpu
                err = 0.0
            else:
                try:
                    t, out = time_strategy(strat, X, M, T, sfreq)
                    err = float(np.max(np.abs(out - out_cpu)))
                except Exception as ex:
                    t = float("nan")
                    err = float("nan")
                    logger.error("%s failed: %s", strat, ex)
            speedup = t_cpu / t if t > 0 else float("nan")
            print(
                f"{n_chan:>8} {dur:>9}s {strat:>15} "
                f"{t:>9.3f}s {speedup:>17.2f}x {err:>18.3e}"
            )
            rows.append(
                {
                    "n_channels": n_chan,
                    "duration_s": dur,
                    "strategy": strat,
                    "time_s": round(t, 4),
                    "speedup_vs_cpu_fallback": round(speedup, 3),
                    "max_abs_err_vs_cpu": float(err),
                }
            )

    # Save CSV
    import csv

    out_path = Path(__file__).parent / "results" / "bench_eigh_strategies.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
