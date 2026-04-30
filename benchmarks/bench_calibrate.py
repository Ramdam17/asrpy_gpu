"""Benchmark ASR.fit / calibrate across backends."""

from __future__ import annotations

import logging

from _bench_utils import (
    get_sync_for,
    load_bench_config,
    make_signal,
    time_call,
    write_results_csv,
)

from asrpy_gpu import _backend_numpy as bn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bench_calibrate")


def bench_one(
    n_channels: int,
    duration_s: float,
    sfreq: float,
    backend: str,
    device: str,
    warmup: int,
    measured: int,
) -> dict:
    X = make_signal(n_channels, duration_s, sfreq)
    sync = get_sync_for(device) if backend == "torch" else None

    if backend == "numpy":

        def fn():
            bn.calibrate(X, sfreq=sfreq)
    else:
        from asrpy_gpu import _backend_torch as bt

        def fn():
            bt.calibrate(X, sfreq=sfreq, device=device)

    mean, std, best = time_call(fn, warmup=warmup, measured=measured, sync=sync)
    return {
        "backend": backend,
        "device": device,
        "n_channels": n_channels,
        "duration_s": duration_s,
        "sfreq": sfreq,
        "mean_s": round(mean, 4),
        "std_s": round(std, 4),
        "min_s": round(best, 4),
    }


def main() -> None:
    cfg = load_bench_config()
    rows: list[dict] = []

    targets = [
        ("numpy", "cpu"),
        ("torch", "cpu"),
    ]
    # Add MPS / CUDA only if available.
    try:
        import torch

        if torch.backends.mps.is_available():
            targets.append(("torch", "mps"))
        if torch.cuda.is_available():
            targets.append(("torch", "cuda"))
    except ImportError:
        pass

    for n_chan in cfg["channels"]:
        for dur in cfg["durations_s"]:
            for backend, device in targets:
                logger.info(
                    "benchmarking calibrate: backend=%s device=%s n_chan=%d duration=%ds",
                    backend,
                    device,
                    n_chan,
                    dur,
                )
                row = bench_one(
                    n_chan,
                    dur,
                    cfg["sfreq"],
                    backend,
                    device,
                    warmup=cfg["warmup_runs"],
                    measured=cfg["measured_runs"],
                )
                rows.append(row)
                logger.info("  → mean=%.3fs  min=%.3fs", row["mean_s"], row["min_s"])

    path = write_results_csv(rows, "bench_calibrate.csv")
    logger.info("saved %d rows to %s", len(rows), path)


if __name__ == "__main__":
    main()
