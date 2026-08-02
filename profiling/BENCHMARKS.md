# Scarf profiling references

This file records the latest completed full-scale Scarf measurements. It does
not preserve superseded experiments or operational call identifiers.

Where a configuration has multiple completed replicates, the main tables report
the mean and a Student-t 95% confidence interval, with `n` shown explicitly.
Until additional replicates finish, every row below is `n = 1`, so the reported
value is the single completed run and no confidence interval is shown. A compact
replicate summary holds individual totals so run-to-run drift remains visible as
more measurements arrive.

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
| 10,000 | 4 | 16 GiB | 12 GiB | 2 | 854.1 s (14.2 min), 95% CI ±1,577.7 s | 2.6 GiB ±0.7 | 2.6 GiB ±0.4 |
| 50,000 | 4 | 16 GiB | 12 GiB | 2 | 930.6 s (15.5 min), 95% CI ±4,453.8 s | 5.2 GiB ±0.4 | 5.2 GiB ±0.7 |
| 100,000 | 4 | 16 GiB | 12 GiB | 1 | 989.1 s (16.5 min) | 6.8 GiB | 7.0 GiB |
| 500,000 | 8 | 32 GiB | 24 GiB | 1 | 1,617.9 s (27.0 min) | 26.6 GiB | 27.0 GiB |
| 1,000,000 | 8 | 32 GiB | 24 GiB | 1 | 2,712.0 s (45.2 min) | 28.8 GiB | 29.2 GiB |
| 5,000,000 | 16 | 64 GiB | 48 GiB | 1 | 10,308.9 s (2.86 h) | 57.3 GiB | 57.4 GiB |
| 10,000,000 | 16 | 64 GiB | 48 GiB | 1 | 25,897.9 s (7.19 h) | 56.4 GiB | 56.6 GiB |

### Replicate totals

Individual completed totals for matching configurations. Second replicates for
100k through 5M are still running or queued; 10M remains `n = 1`. With `n = 2`,
the Student-t 95% CI is provisional and can be very wide.

| Input cells | Replicate | Wall time (s) | Peak cgroup (GiB) | Peak RSS (GiB) |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | r1 | 729.9 | 2.7 | 2.7 |
| 10,000 | r2 | 978.3 | 2.6 | 2.6 |
| 50,000 | r1 | 580.1 | 5.2 | 5.3 |
| 50,000 | r2 | 1,281.1 | 5.2 | 5.1 |
| 100,000 | r1 | 989.1 | 6.8 | 7.0 |
| 500,000 | r1 | 1,617.9 | 26.6 | 27.0 |
| 1,000,000 | r1 | 2,712.0 | 28.8 | 29.2 |
| 5,000,000 | r1 | 10,308.9 | 57.3 | 57.4 |
| 10,000,000 | r1 | 25,897.9 | 56.4 | 56.6 |

Stage values below use stage `seconds`, which includes recorded stage work but
excludes the shared funnel download and orchestration. The funnel total includes
the initial download and a few seconds of orchestration.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 2.4 | 95.3 | 26.4 | 118.8 | 59.2 | 1,123.8 | 2,174.9 |
| Create count store | 5.6 | 18.0 | 47.1 | 206.8 | 613.8 | 1,300.9 | 3,481.2 |
| Write `countsT` | 8.2 | 19.4 | 39.3 | 72.7 | 144.9 | 512.7 | 1,384.8 |
| Initialize datastore | 30.4 | 29.5 | 39.9 | 59.6 | 83.6 | 311.8 | 683.0 |
| Reopen datastore | 10.1 | 8.8 | 7.3 | 8.1 | 6.2 | 5.4 | 10.3 |
| Filter cells | 30.7 | 25.5 | 28.5 | 25.6 | 20.4 | 25.0 | 54.0 |
| Mark HVGs | 61.0 | 60.4 | 71.8 | 95.5 | 183.5 | 619.4 | 1,366.9 |
| Normalize | 36.6 | 38.5 | 62.9 | 134.6 | 166.4 | 926.1 | 1,296.4 |
| PCA | 43.8 | 35.4 | 44.7 | 60.8 | 79.1 | 223.2 | 570.9 |
| Build embedding initialization | 12.8 | 11.0 | 12.9 | 17.0 | 26.2 | 109.1 | 246.5 |
| Build ANN index | 23.3 | 21.6 | 27.4 | 56.8 | 86.5 | 341.4 | 831.5 |
| Query neighbours | 26.8 | 21.2 | 21.6 | 30.7 | 45.4 | 108.2 | 299.8 |
| Build connectivity map | 53.1 | 55.7 | 49.1 | 43.0 | 51.1 | 43.0 | 101.8 |
| UMAP | 66.0 | 72.9 | 102.4 | 137.4 | 284.4 | 699.3 | 1,772.2 |
| Leiden | 55.6 | 55.5 | 57.7 | 128.7 | 259.6 | 1,348.4 | 3,486.5 |
| Paris | 112.8 | 98.8 | 106.7 | 115.2 | 132.8 | 261.1 | 769.2 |
| Marker search | 115.4 | 118.4 | 105.3 | 150.0 | 293.0 | 1,712.7 | 6,117.9 |
| **Funnel total** | **854.1** | **930.6** | **989.1** | **1,617.9** | **2,712.0** | **10,308.9** | **25,897.9** |

The 10k and 50k stage columns are means of two replicates. End-to-end wall-time
95% CI half-widths are ±1,577.7 s (10k) and ±4,453.8 s (50k). Stage-level
half-widths are omitted because `n = 2` intervals are too wide to be useful at a
glance. The large 50k interval is dominated by download drift (12.4 s vs
178.3 s).

## Interpretation

Most published rows are still one completed run (`n = 1`). The 10k and 50k rows
now have two replicates and report mean wall time with a Student-t 95%
confidence interval. With `n = 2`, that interval remains wide and should be
treated as provisional drift capture rather than a tight uncertainty bound.

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
