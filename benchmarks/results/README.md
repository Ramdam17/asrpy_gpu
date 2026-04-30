# Benchmark results

Results produced by `benchmarks/bench_calibrate.py` and `benchmarks/bench_process.py`
land here as CSV + PNG files. They are git-ignored (see `.gitignore`); only this
README and the `.gitkeep` marker are tracked.

To reproduce locally:

```bash
uv run python benchmarks/bench_calibrate.py
uv run python benchmarks/bench_process.py
```

A summary is appended to `docs/benchmark.md` after each run.
