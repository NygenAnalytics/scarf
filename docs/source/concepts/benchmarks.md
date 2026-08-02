(benchmarks)=
# Benchmarks

These measurements are empirical references for a fixed workflow, dataset,
revision, and cloud envelope. They establish execution and resource use. They
are not hardware guarantees, biological validation, or a claim of superiority
over another package.

The source record, including operational profiling notes, is
[profiling/BENCHMARKS.md](https://github.com/parashardhapola/scarf/blob/master/profiling/BENCHMARKS.md).
Resource planning controls are explained in {doc}`scale_and_memory`.

## What was measured

Measurements completed on 2026-08-02 from commit
`ba6dc04d7f4e18e441e07d1f503722ef1018f1ff`. Each run downloaded a public
CELLxGENE H5AD, wrote a fresh object-store-backed Zarr v3 store, and completed
the same sixteen-stage CPU workflow on nested deterministic samples (seed 0):

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

Source dataset ID `dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3`, version ID
`1bc30289-9565-4099-abf9-3326328c11ac`.

### Shared analysis settings

- 2,000 highly variable features and 50 PCA dimensions
- 11 neighbours and 1,000 embedding centroids
- graph seed 4466; UMAP and Leiden seed 4444
- 300 UMAP epochs and Leiden resolution 1.0
- parallel ANN and UMAP enabled
- embedding-initialization sample fraction 0.1 and minibatch size 10,000
- 1st and 99th percentile cell filtering
- minimum 10 features per cell and 20 cells per feature
- 1,000-cell H5AD conversion batches
- S3-compatible object storage in the Modal EU region
- Scarf memory budget set to 75% of the container memory limit

`mem_budget` controls planned block sizes and concurrency. It is not a hard
process-memory limit.

### Machine classes

Machine size grew with input size, so the rows are not a same-machine scaling
curve:

| Input cells | CPU | Container memory | Scarf budget |
| ----------: | --: | ---------------: | -----------: |
| 10,000 to 100,000 | 4 | 16 GiB | 12 GiB |
| 500,000 to 1,000,000 | 8 | 32 GiB | 24 GiB |
| 5,000,000 to 10,000,000 | 16 | 64 GiB | 48 GiB |

## End-to-end results

Peak memory uses sampled `memory.current` (`peakCgroupBytes`). Short spikes may
be missed. Every published row is currently one completed run (`n = 1`), so no
confidence interval is shown.

| Input cells | CPU | Container | Budget | Wall time | Peak cgroup | Peak RSS |
| ----------: | --: | --------: | -----: | --------: | ----------: | -------: |
| 10,000 | 4 | 16 GiB | 12 GiB | 12.2 min | 2.7 GiB | 2.7 GiB |
| 50,000 | 4 | 16 GiB | 12 GiB | 9.7 min | 5.2 GiB | 5.3 GiB |
| 100,000 | 4 | 16 GiB | 12 GiB | 16.5 min | 6.8 GiB | 7.0 GiB |
| 500,000 | 8 | 32 GiB | 24 GiB | 27.0 min | 26.6 GiB | 27.0 GiB |
| 1,000,000 | 8 | 32 GiB | 24 GiB | 45.2 min | 28.8 GiB | 29.2 GiB |
| 5,000,000 | 16 | 64 GiB | 48 GiB | 2.86 h | 57.3 GiB | 57.4 GiB |
| 10,000,000 | 16 | 64 GiB | 48 GiB | 7.19 h | 56.4 GiB | 56.6 GiB |

At 10k, fixed overheads are large relative to useful work, which is why that row
can look slower than 50k on the same machine class. At 5M and 10M, marker search
and Leiden dominate wall time.

## Stage timings

Stage values are stage `seconds` and exclude shared funnel download and
orchestration. The funnel total includes the initial download and a few seconds
of orchestration. Times are shown in seconds.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 2.7 | 12.4 | 26.4 | 118.8 | 59.2 | 1,123.8 | 2,174.9 |
| Create count store | 6.0 | 17.7 | 47.1 | 206.8 | 613.8 | 1,300.9 | 3,481.2 |
| Write `countsT` | 7.6 | 16.8 | 39.3 | 72.7 | 144.9 | 512.7 | 1,384.8 |
| Initialize datastore | 26.5 | 18.7 | 39.9 | 59.6 | 83.6 | 311.8 | 683.0 |
| Reopen datastore | 7.8 | 5.8 | 7.3 | 8.1 | 6.2 | 5.4 | 10.3 |
| Filter cells | 27.0 | 14.1 | 28.5 | 25.6 | 20.4 | 25.0 | 54.0 |
| Mark HVGs | 48.6 | 38.0 | 71.8 | 95.5 | 183.5 | 619.4 | 1,366.9 |
| Normalize | 30.4 | 25.9 | 62.9 | 134.6 | 166.4 | 926.1 | 1,296.4 |
| PCA | 40.6 | 23.7 | 44.7 | 60.8 | 79.1 | 223.2 | 570.9 |
| Build embedding initialization | 12.0 | 7.2 | 12.9 | 17.0 | 26.2 | 109.1 | 246.5 |
| Build ANN index | 18.9 | 12.4 | 27.4 | 56.8 | 86.5 | 341.4 | 831.5 |
| Query neighbours | 20.9 | 12.6 | 21.6 | 30.7 | 45.4 | 108.2 | 299.8 |
| Build connectivity map | 53.7 | 49.0 | 49.1 | 43.0 | 51.1 | 43.0 | 101.8 |
| UMAP | 57.7 | 52.4 | 102.4 | 137.4 | 284.4 | 699.3 | 1,772.2 |
| Leiden | 45.1 | 34.9 | 57.7 | 128.7 | 259.6 | 1,348.4 | 3,486.5 |
| Paris | 99.0 | 65.7 | 106.7 | 115.2 | 132.8 | 261.1 | 769.2 |
| Marker search | 92.7 | 78.5 | 105.3 | 150.0 | 293.0 | 1,712.7 | 6,117.9 |
| **Funnel total** | **729.9** | **580.1** | **989.1** | **1,617.9** | **2,712.0** | **10,308.9** | **25,897.9** |

## How to read these numbers

- Compare runs only when dataset, code revision, workflow settings, storage
  conditions, and resource envelope are stated.
- Peak values are sampled, and the profiler did not force garbage collection
  between stages.
- When a second matching replicate finishes, that size will switch to mean wall
  time with a Student-t 95% confidence interval. With `n = 2`, that interval
  remains wide and should be treated as provisional drift capture.
- These runs replace earlier July 2026 1M and 10M reference rows. The older 10M
  measurement used 16 CPU and 128 GiB; the current 10M measurement uses 16 CPU
  and 64 GiB with a 48 GiB Scarf budget, so the totals are not directly
  comparable as a software regression check.
