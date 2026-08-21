# Scarf profiling references

This file records the latest completed full-scale Scarf measurements. It does
not preserve superseded experiments or operational call identifiers.

Where a configuration has multiple completed replicates, the main tables report
the mean. Individual replicate totals are kept below so run-to-run drift stays
visible.

## Scope

The measurements were completed on 2026-08-21 from commit `84ab362`. Each run
downloaded an H5AD, used a fresh object-store-backed Zarr v3 store, and
completed the same sixteen-stage CPU workflow in one non-preemptible
container:

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

- 1,000 highly variable features and 21 PCA dimensions
- 11 neighbours and 1,000 embedding centroids
- graph seed 4466
- UMAP and Leiden seed 4444
- 300 UMAP epochs and Leiden resolution 1.0
- parallel ANN and UMAP enabled
- embedding-initialization sample fraction 0.1 and minibatch size 10,000
- 1st and 99th percentile cell filtering
- minimum 10 features per cell and 20 cells per feature
- 1,000-cell H5AD conversion batches
- 1 GB count-matrix units and 100 MB chunks (product defaults)
- S3-compatible object storage in the Modal EU region
- Scarf memory budget set to 75% of the container memory limit
- worker count equal to the container CPU count
- same CPU, memory, budget, and worker count for every stage in a run

The Scarf memory budget controls planned block sizes and concurrency. It is not
a hard process-memory limit. Machine classes differed by size, so the
end-to-end rows are not a same-machine scaling curve.

## End-to-end results

Peak RSS is sampled process RSS. Short spikes may be missed.

| Input cells | CPU | Container memory | n | Wall time | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4 | 16 GiB | 3 | 417.5 s (7.0 min) | 2.7 GiB |
| 50,000 | 4 | 16 GiB | 3 | 483.2 s (8.1 min) | 7.0 GiB |
| 100,000 | 4 | 16 GiB | 3 | 522.5 s (8.7 min) | 8.5 GiB |
| 500,000 | 8 | 32 GiB | 3 | 1,218.1 s (20.3 min) | 15.0 GiB |
| 1,000,000 | 8 | 32 GiB | 3 | 1,825.1 s (30.4 min) | 12.9 GiB |
| 5,000,000 | 16 | 64 GiB | 3 | 6,506.8 s (1.81 h) | 32.1 GiB |
| 10,000,000 | 16 | 64 GiB | 1 | 10,326.7 s (2.87 h) | 33.8 GiB |

Wall time and peak RSS for n = 3 are means of the completed replicates. The
10M row is the first completed replicate.

### Replicate totals

| Input cells | Replicate | Wall time (s) | Peak cgroup (GiB) | Peak RSS (GiB) |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | r1 | 437.4 | 2.60 | 2.63 |
| 10,000 | r2 | 433.8 | 2.68 | 2.71 |
| 10,000 | r3 | 381.2 | 2.82 | 2.84 |
| 50,000 | r1 | 395.8 | 7.06 | 7.11 |
| 50,000 | r2 | 541.5 | 6.22 | 6.67 |
| 50,000 | r3 | 512.1 | 7.17 | 7.22 |
| 100,000 | r1 | 513.2 | 8.66 | 8.70 |
| 100,000 | r2 | 463.3 | 8.76 | 8.81 |
| 100,000 | r3 | 591.0 | 8.01 | 8.06 |
| 500,000 | r1 | 1,229.9 | 16.02 | 16.12 |
| 500,000 | r2 | 1,202.7 | 13.46 | 13.52 |
| 500,000 | r3 | 1,221.6 | 15.17 | 15.26 |
| 1,000,000 | r1 | 1,974.5 | 13.21 | 13.26 |
| 1,000,000 | r2 | 1,748.2 | 12.93 | 12.99 |
| 1,000,000 | r3 | 1,752.6 | 12.62 | 12.55 |
| 5,000,000 | r1 | 6,745.7 | 31.26 | 31.44 |
| 5,000,000 | r2 | 6,792.7 | 32.19 | 32.40 |
| 5,000,000 | r3 | 5,982.1 | 32.22 | 32.43 |
| 10,000,000 | r1 | 10,326.7 | 33.22 | 33.75 |
| 10,000,000 | r2 |  |  |  |
| 10,000,000 | r3 |  |  |  |

Stage values below use stage `seconds`, which includes recorded stage work but
excludes the shared funnel download and orchestration. The funnel total is
mean `wholeFunctionSeconds` (or the single completed 10M run) and includes
download plus orchestration.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 1.2 | 1.8 | 4.5 | 14.9 | 20.7 | 143.4 | 322.4 |
| Create count store | 5.4 | 19.6 | 29.2 | 88.2 | 161.3 | 489.4 | 723.1 |
| Write `countsT` | 7.5 | 20.2 | 31.6 | 172.6 | 259.5 | 1,041.8 | 1,080.6 |
| Initialize datastore | 17.8 | 19.6 | 20.2 | 55.4 | 75.8 | 370.4 | 575.9 |
| Reopen datastore | 6.3 | 5.8 | 5.7 | 6.1 | 6.5 | 8.0 | 9.1 |
| Filter cells | 17.2 | 16.0 | 14.6 | 16.7 | 18.8 | 32.1 | 53.5 |
| Mark HVGs | 35.5 | 36.9 | 36.9 | 67.0 | 93.7 | 158.0 | 453.0 |
| Normalize | 19.5 | 21.8 | 23.5 | 53.8 | 84.2 | 212.6 | 383.2 |
| PCA | 20.8 | 21.5 | 21.3 | 39.5 | 51.6 | 181.6 | 293.0 |
| Build embedding initialization | 6.2 | 6.7 | 6.3 | 10.6 | 14.3 | 62.6 | 66.9 |
| Build ANN index | 10.9 | 11.3 | 13.4 | 32.7 | 63.0 | 210.6 | 443.9 |
| Query neighbours | 12.6 | 11.9 | 12.7 | 22.6 | 31.5 | 94.9 | 179.6 |
| Build connectivity map | 34.3 | 34.7 | 32.2 | 41.5 | 43.0 | 54.5 | 70.0 |
| UMAP | 39.3 | 51.3 | 58.6 | 131.8 | 226.5 | 565.3 | 994.0 |
| Leiden | 29.2 | 34.5 | 36.6 | 113.4 | 233.3 | 1,223.4 | 2,228.1 |
| Paris | 61.6 | 62.5 | 59.6 | 92.4 | 111.7 | 333.4 | 500.6 |
| Marker search | 57.5 | 68.0 | 78.1 | 188.6 | 226.3 | 783.7 | 1,277.0 |
| **Funnel total** | **417.5** | **483.2** | **522.5** | **1,218.1** | **1,825.1** | **6,506.8** | **10,326.7** |

The 10k through 5M stage columns are means of three replicates. The 10M column
is the first completed replicate. Download time is a small fraction of the
funnel. Larger spread across replicates is compute-side: 50k and 100k drift
across several stages, 1M is led by r1 (1,974.5 s vs about 1,750 s), and 5M is
led by marker search in r3 (483.4 s vs 912.4 s and 955.3 s).

## Interpretation

These runs replace the previous reference tables. HVG count and PCA
dimensionality follow the current product defaults, so older tables with
different analysis settings are not a same-settings regression check.

Peak values are sampled, so very short spikes may be missed. The profiler did
not force garbage collection between stages. Sampled RSS stayed well inside
each container: 2.7 GiB of 16 GiB at 10k, 8.5 GiB of 16 GiB at 100k, 12.9 GiB
of 32 GiB at 1M, 32.1 GiB of 64 GiB at 5M, and 33.8 GiB of 64 GiB at 10M. The
1M mean peak is slightly below the 500k mean peak on the same 8 CPU / 32 GiB
class; that is an observed sample, not a claim that 1M is cheaper than 500k in
general.

At 5M and 10M, Leiden is the largest stage, then marker search and `countsT`.
At 10k through 100k, fixed work (Paris, markers, connectivity, UMAP) is large
relative to conversion, which is why those three totals stay in a narrow band
on the 4 CPU / 16 GiB class.

Parallel ANN and UMAP were enabled for this timing series, so graph-derived
embeddings and cluster assignments are not bitwise reproducible. The timing
result does not claim otherwise.

These results establish execution and resource use for the recorded dataset,
code revision, settings, and cloud conditions. They do not establish biological
correctness, a hardware guarantee, or superiority over another package.

## Running the profiler

Copy `profiling/config.example.toml` to the ignored
`profiling/config.toml`, configure your own object store and Modal secret, and
set a fresh `runTag`. For a same-box timing run, prefer `fixedResources` so
every stage uses the same CPU, memory, Scarf budget, and worker count.
Deployment is a user action. After the app is deployed, prepare the
deterministic samples, then spawn the current funnel:

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
