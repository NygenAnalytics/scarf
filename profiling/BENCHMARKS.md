# Scarf profiling references

This file records the latest completed full-scale Scarf measurements. It does
not preserve superseded experiments or operational call identifiers.

Where a configuration has multiple completed replicates, the main tables report
the mean and sample standard deviation, with `n` shown explicitly. Individual
replicate totals are kept in a compact summary so run-to-run drift stays
visible. Student-t 95% confidence intervals are not shown at `n = 2`: with one
degree of freedom the critical value is about 12.7, so even modest drift
produces an interval wider than the mean. Prefer `n >= 3` before publishing a
t-based CI.

## Scope

The measurements were completed on 2026-08-02 from commit
`ba6dc04d7f4e18e441e07d1f503722ef1018f1ff`. Each run downloaded an H5AD, used a
fresh object-store-backed Zarr v3 store, and completed the same sixteen-stage CPU
workflow:

1. cell-major count conversion
2. `countsT` construction
3. datastore initialization and quality-control metrics
4. datastore reopen
5. cell filtering
6. highly variable feature selection
7. normalization
8. PCA
9. embedding initialization
10. approximate-neighbour index construction
11. neighbour queries
12. connectivity-map construction
13. UMAP
14. Leiden clustering
15. Paris clustering
16. marker search using the Leiden groups

The source was the public CELLxGENE H5AD with dataset ID
`dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3` and version ID
`1bc30289-9565-4099-abf9-3326328c11ac`. The profiler selected deterministic,
nested samples with seed 0 for every target size.

## Settings

These are the recorded measurement settings. `config.example.toml` is an
evolving operational template and is not an exact reproduction of these runs.

- 2,000 highly variable features and 50 PCA dimensions
- 11 neighbours and 1,000 embedding centroids
- graph seed 4466
- UMAP and Leiden seed 4444
- 300 UMAP epochs and Leiden resolution 1.0
- parallel ANN and UMAP enabled
- embedding-initialization sample fraction 0.1 and minibatch size 10,000
- 1st and 99th percentile cell filtering
- minimum 10 features per cell and 20 cells per feature
- 1,000-cell H5AD conversion batches
- S3-compatible object storage in the Modal EU region
- Scarf memory budget set to 75% of the container memory limit

The Scarf memory budget controls planned block sizes and concurrency. It is not
a hard process-memory limit.

Machine classes differed by size, so the rows are not a same-machine scaling
curve:

| Input cells | CPU | Container memory | Scarf budget |
| ---: | ---: | ---: | ---: |
| 10,000 to 100,000 | 4 | 16 GiB | 12 GiB |
| 500,000 to 1,000,000 | 8 | 32 GiB | 24 GiB |
| 5,000,000 to 10,000,000 | 16 | 64 GiB | 48 GiB |

## End-to-end results

Peak memory uses sampled `memory.current` (`peakCgroupBytes`). Modal did not
expose a cgroup peak scope for these runs, so short spikes may be missed.

| Input cells | CPU | Container memory | Scarf budget | n | Wall time | Peak cgroup memory | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4 | 16 GiB | 12 GiB | 2 | 854.1 ± 175.6 s (14.2 min); range 729.9–978.3 | 2.6 GiB | 2.6 GiB |
| 50,000 | 4 | 16 GiB | 12 GiB | 2 | 930.6 ± 495.7 s (15.5 min); range 580.1–1,281.1 | 5.2 GiB | 5.2 GiB |
| 100,000 | 4 | 16 GiB | 12 GiB | 2 | 1,107.9 ± 168.0 s (18.5 min); range 989.1–1,226.7 | 6.9 GiB | 7.0 GiB |
| 500,000 | 8 | 32 GiB | 24 GiB | 2 | 1,838.6 ± 312.1 s (30.6 min); range 1,617.9–2,059.3 | 26.8 GiB | 27.0 GiB |
| 1,000,000 | 8 | 32 GiB | 24 GiB | 2 | 3,157.8 ± 630.4 s (52.6 min); range 2,712.0–3,603.5 | 28.9 GiB | 29.1 GiB |
| 5,000,000 | 16 | 64 GiB | 48 GiB | 2 | 11,636.9 ± 1,878.1 s (3.23 h); range 10,308.9–12,964.9 | 57.2 GiB | 57.3 GiB |
| 10,000,000 | 16 | 64 GiB | 48 GiB | 1 | 25,897.9 s (7.19 h) | 56.4 GiB | 56.6 GiB |

Wall-time `±` values are sample standard deviations, not confidence intervals.

### Replicate totals

Individual completed totals for matching configurations. The 10M second
replicate is still running.

| Input cells | Replicate | Wall time (s) | Peak cgroup (GiB) | Peak RSS (GiB) |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | r1 | 729.9 | 2.7 | 2.7 |
| 10,000 | r2 | 978.3 | 2.6 | 2.6 |
| 50,000 | r1 | 580.1 | 5.2 | 5.3 |
| 50,000 | r2 | 1,281.1 | 5.2 | 5.1 |
| 100,000 | r1 | 989.1 | 6.8 | 7.0 |
| 100,000 | r2 | 1,226.7 | 7.0 | 7.0 |
| 500,000 | r1 | 1,617.9 | 26.6 | 27.0 |
| 500,000 | r2 | 2,059.3 | 27.0 | 27.0 |
| 1,000,000 | r1 | 2,712.0 | 28.8 | 29.2 |
| 1,000,000 | r2 | 3,603.5 | 29.0 | 29.1 |
| 5,000,000 | r1 | 10,308.9 | 57.3 | 57.4 |
| 5,000,000 | r2 | 12,964.9 | 57.1 | 57.2 |
| 10,000,000 | r1 | 25,897.9 | 56.4 | 56.6 |

Stage values below use stage `seconds`, which includes recorded stage work but
excludes the shared funnel download and orchestration. The funnel total includes
the initial download and a few seconds of orchestration.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 2.4 | 95.3 | 31.8 | 92.5 | 242.1 | 1,335.4 | 2,174.9 |
| Create count store | 5.6 | 18.0 | 40.4 | 178.8 | 521.7 | 1,310.3 | 3,481.2 |
| Write `countsT` | 8.2 | 19.4 | 40.5 | 100.2 | 190.4 | 619.1 | 1,384.8 |
| Initialize datastore | 30.4 | 29.5 | 40.1 | 63.5 | 99.8 | 369.7 | 683.0 |
| Reopen datastore | 10.1 | 8.8 | 9.7 | 10.5 | 9.2 | 10.2 | 10.3 |
| Filter cells | 30.7 | 25.5 | 31.4 | 30.3 | 30.4 | 41.0 | 54.0 |
| Mark HVGs | 61.0 | 60.4 | 89.3 | 142.9 | 246.1 | 648.7 | 1,366.9 |
| Normalize | 36.6 | 38.5 | 63.2 | 135.7 | 218.2 | 948.5 | 1,296.4 |
| PCA | 43.8 | 35.4 | 49.4 | 70.7 | 101.9 | 264.0 | 570.9 |
| Build embedding initialization | 12.8 | 11.0 | 14.2 | 18.5 | 27.8 | 111.4 | 246.5 |
| Build ANN index | 23.3 | 21.6 | 30.6 | 74.0 | 125.1 | 476.3 | 831.5 |
| Query neighbours | 26.8 | 21.2 | 24.5 | 37.0 | 53.0 | 142.2 | 299.8 |
| Build connectivity map | 53.1 | 55.7 | 51.6 | 52.4 | 55.8 | 64.7 | 101.8 |
| UMAP | 66.0 | 72.9 | 104.8 | 182.9 | 323.7 | 681.0 | 1,772.2 |
| Leiden | 55.6 | 55.5 | 65.3 | 143.1 | 257.6 | 1,380.2 | 3,486.5 |
| Paris | 112.8 | 98.8 | 116.2 | 133.6 | 156.5 | 353.6 | 769.2 |
| Marker search | 115.4 | 118.4 | 139.0 | 181.2 | 282.2 | 2,231.9 | 6,117.9 |
| **Funnel total** | **854.1** | **930.6** | **1,107.9** | **1,838.6** | **3,157.8** | **11,636.9** | **25,897.9** |

The 10k through 5M stage columns are means of two replicates. Several large
wall-time SDs are dominated by download drift (50k: 12.4 s vs 178.3 s; 1M:
59.2 s vs 425.0 s; 5M: 1,123.8 s vs 1,546.9 s), not by a proportional
slowdown across every pipeline stage.

## Interpretation

The 10M row is still one completed run (`n = 1`) while its second replicate
runs. The 10k through 5M rows have two replicates and report mean ± sample SD
plus the observed range. That is enough to show drift at `n = 2`. A Student-t
95% CI will wait until there are at least three matching replicates.

These runs replace the previous July 2026 1M and 10M reference rows. The older
10M measurement used 16 CPU and 128 GiB; the current 10M measurement uses 16 CPU
and 64 GiB with a 48 GiB Scarf budget, so the totals are not directly comparable
as a software regression check.

Peak values come from sampled `memory.current`, so very short spikes may be
missed. The profiler did not force garbage collection between stages. At 5M and
10M, marker search and Leiden dominate wall time. At 10k, fixed overheads are
large relative to useful work, which is why that row can look slower than 50k
even on the same machine class.

These results establish execution and resource use for the recorded dataset,
code revision, settings, and cloud conditions. They do not establish biological
correctness, a hardware guarantee, or superiority over another package.

## Running the profiler

Copy `profiling/config.example.toml` to the ignored
`profiling/config.toml`, configure your own object store and Modal secret, and
set a fresh `runTag`. Deployment is a user action. After the app is deployed,
prepare the deterministic samples, then spawn the current funnel:

```bash
uv run --group profiling modal run --env scarf_profiling \
  -m profiling.modal_app -- prepare \
  --config profiling/config.toml

uv run --group profiling modal run --env scarf_profiling \
  -m profiling.modal_app -- run-e2e \
  --config profiling/config.toml --size 1000000
```

The prepare command spawns work and returns immediately. Confirm that the
requested H5AD exists before starting `run-e2e`.

Each stage writes a result JSON, and the complete invocation writes
`funnel.json`. Compare runs only when their dataset, code revision, workflow
settings, storage conditions, and resource envelope are stated.
