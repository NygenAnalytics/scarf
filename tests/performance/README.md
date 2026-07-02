# Performance benchmarks

Opt-in benchmarks for the shard-parallel processing primitives
(`scarf/parallel.py`, `scarf/storage/budget.py`, `scarf/chunked.py`,
`scarf/storage/zarr_store.py`). Not just for this feature: add new `bench_*.py`
scripts here for any future perf work.

## Why these don't run in the normal test suite

pytest only collects files matching `test_*.py`/`*_test.py` (see
`[tool.pytest.ini_options]` in `pyproject.toml`). Every script in this
directory is named `bench_*.py`, so a plain `uv run pytest` never sees or
imports them, and CI stays unaffected. They are standalone scripts, each with
its own `argparse` CLI and a `__main__` guard, run directly with `uv run`.

## Benchmarks

All local benchmarks default to a synthetic 60,000 x 1,000 float32 matrix in
10,000-row shards (6 shards), written to a real local-disk Zarr store in a
temp directory so timings include decompression, not just in-memory math.
Pass `--n-cells`/`--n-cols`/`--shard-rows` to change the dataset size, and
`--workers` (comma-separated, e.g. `1,2,4,8`) to change the sweep.

- `bench_reductions.py`: `ChunkedArray` axis=0 `sum`/`var`/`mean_and_std` via
  `map_shards`. Sweeps worker counts, prints wall-time and speedup per step,
  and asserts every reduction is bit-identical across worker counts.

  ```bash
  uv run python -m tests.performance.bench_reductions
  uv run python -m tests.performance.bench_reductions --n-cells 100000 --workers 1,2,4,8,16
  ```

- `bench_projection.py`: the per-block reducer `AnnStream` uses for
  embeddings/ANN fitting (z-score against fixed `mu`/`sigma`, then `dot` with
  a fixed loadings matrix) applied via `map_blocks`. Sweeps worker counts and
  asserts the projected output is identical across worker counts.

  ```bash
  uv run python -m tests.performance.bench_projection
  uv run python -m tests.performance.bench_projection --dims 30 --workers 1,2,4,8
  ```

- `bench_write.py`: `write_dense_in_shard_rows` (parallel produce, single
  writer) against a naive fully-serial row-band loop, reporting the speedup.

  ```bash
  uv run python -m tests.performance.bench_write
  uv run python -m tests.performance.bench_write --workers 8
  ```

- `bench_r2.py` (optional, needs R2 credentials): uploads a small synthetic
  sharded array to Cloudflare R2, then sweeps across-shard depth and Zarr
  `async.concurrency` independently while reading it back, showing that the
  budget split bounds total in-flight requests instead of multiplying them.
  Skips cleanly (prints a message, exits 0) if credentials are missing, and
  always deletes the probe data it uploaded, even if a step fails.

  ```bash
  uv run python -m tests.performance.bench_r2
  uv run python -m tests.performance.bench_r2 --across 1,2,4,8 --io-concurrency 1,4,8
  ```

## R2 setup (for `bench_r2.py`)

Copy `tests/.env.example` to `tests/.env` and fill in:

- `R2_BUCKET`, `R2_PREFIX`: where probe data is written and cleaned up
  (same variables `tests/zarr_cloud_exp` uses).
- `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`: credentials.
  `tests/r2_profile.py` (reused here for `storage_options()`) prefers
  `SCARF_R2_ENDPOINT`/`SCARF_R2_ACCESS_KEY_ID`/`SCARF_R2_SECRET_ACCESS_KEY`
  if set, falling back to the `R2_*` names above.

Each run writes under a fresh `R2_PREFIX/scarf-perf/<random-id>/` key so
concurrent runs don't collide, and removes everything under that key in a
`finally` block regardless of outcome.

## Expected findings (recorded from the original design benchmarks)

See the "Benchmark findings" section of the shard-parallel-processing plan
for the numbers these benchmarks reproduce. In short: across-shard depth is
the dominant lever, extra within-block BLAS threads add little (so
`withinBlockThreads=1`), throughput flattens past roughly 4-8 concurrent
shards on both local disk and R2 (`ACROSS_SHARD_CAP=8`), and `io_concurrency`
is a modest secondary lever, not a multiplier on top of worker count.
