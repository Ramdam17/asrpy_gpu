"""Shared helpers for benchmarks.

Implements the warmup + sync + measured-run pattern from the project's
``benchmarking`` skill. Reads sizes from ``config/test_tolerances.yaml``.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "config" / "test_tolerances.yaml"
RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_bench_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)["benchmark"]


def make_signal(
    n_channels: int, duration_s: float, sfreq: float, seed: int = 0
) -> np.ndarray:
    """Reproducible Gaussian signal — benchmarks measure throughput, not
    cleaning quality, so a bare Gaussian is fine."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    return rng.standard_normal((n_channels, n)).astype(np.float64)


def time_call(
    fn: Callable[[], Any],
    *,
    warmup: int,
    measured: int,
    sync: Callable[[], None] | None = None,
) -> tuple[float, float, float]:
    """Run ``fn`` ``warmup + measured`` times; return (mean, std, min) in seconds.

    ``sync`` is a no-arg callable that blocks until any in-flight async work
    finishes (e.g. ``torch.mps.synchronize``). It is called BEFORE starting
    each timer to ensure we don't include leftover work, and AFTER the call
    to wait for results.
    """
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    times: list[float] = []
    for _ in range(measured):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(arr.min())


def write_results_csv(rows: list[dict[str, Any]], filename: str) -> Path:
    """Persist benchmark rows next to the script."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def get_sync_for(device: str) -> Callable[[], None] | None:
    """Return the proper torch sync function for ``device``."""
    if device == "cpu":
        return None
    import torch

    if device == "mps":
        return torch.mps.synchronize
    if device == "cuda":
        return torch.cuda.synchronize
    return None
