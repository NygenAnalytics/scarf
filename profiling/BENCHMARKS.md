# Scarf benchmark results

These reference measurements cover the full Scarf workflow from H5AD
conversion through marker search at 10,000 to 10 million input cells. The
current 10M reference is 2.78 hours with 31.9 GiB mean peak memory on a 16 CPU,
64 GiB container.

## End-to-end results

| Input cells | CPU | Container memory | n | Wall time | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4 | 16 GiB | 3 | 360.0 s (6.0 min) | 2.7 GiB |
| 50,000 | 4 | 16 GiB | 3 | 426.9 s (7.1 min) | 7.0 GiB |
| 100,000 | 4 | 16 GiB | 3 | 464.9 s (7.7 min) | 8.5 GiB |
| 500,000 | 8 | 32 GiB | 3 | 1,065.7 s (17.8 min) | 15.0 GiB |
| 1,000,000 | 8 | 32 GiB | 3 | 1,558.8 s (26.0 min) | 12.9 GiB |
| 5,000,000 | 16 | 64 GiB | 3 | 5,206.5 s (1.45 h) | 32.1 GiB |
| 10,000,000 | 16 | 64 GiB | 3 | 10,013.7 s (2.78 h) | 31.9 GiB |

## Stage breakdown

Values are mean elapsed seconds. Stage times exclude orchestration; dataset
download is shown separately.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 1.2 | 1.8 | 4.5 | 14.9 | 20.7 | 143.4 | 328.2 |
| Create count store | 5.4 | 19.6 | 29.2 | 88.2 | 161.3 | 489.4 | 1,018.5 |
| Write `countsT` | 7.5 | 20.2 | 31.6 | 172.6 | 259.5 | 1,041.8 | 1,668.7 |
| Initialize datastore | 17.8 | 19.6 | 20.2 | 55.4 | 75.8 | 370.4 | 686.7 |
| Reopen datastore | 6.3 | 5.8 | 5.7 | 6.1 | 6.5 | 8.0 | 12.5 |
| Filter cells | 17.2 | 16.0 | 14.6 | 16.7 | 18.8 | 32.1 | 62.6 |
| Mark HVGs | 35.5 | 36.9 | 36.9 | 67.0 | 93.7 | 158.0 | 492.4 |
| Normalize | 19.5 | 21.8 | 23.5 | 53.8 | 84.2 | 212.6 | 481.0 |
| PCA | 20.8 | 21.5 | 21.3 | 39.5 | 51.6 | 181.6 | 370.6 |
| Build embedding initialization | 6.2 | 6.7 | 6.3 | 10.6 | 14.3 | 62.6 | 109.8 |
| Build ANN index | 10.9 | 11.3 | 13.4 | 32.7 | 63.0 | 210.6 | 469.3 |
| Query neighbours | 12.6 | 11.9 | 12.7 | 22.6 | 31.5 | 94.9 | 207.4 |
| Build connectivity map | 34.3 | 34.7 | 32.2 | 41.5 | 43.0 | 54.5 | 84.9 |
| UMAP | 39.3 | 51.3 | 58.6 | 131.8 | 226.5 | 565.3 | 1,233.8 |
| Leiden | 33.4 | 40.7 | 38.6 | 53.4 | 78.7 | 256.5 | 501.5 |
| Marker search | 57.5 | 68.0 | 78.1 | 188.6 | 226.3 | 783.7 | 1,280.4 |
| **Reference total** | **360.0** | **426.9** | **464.9** | **1,065.7** | **1,558.8** | **5,206.5** | **10,013.7** |

## Replicate totals

| Input cells | Replicate | Wall time (s) | Peak memory (GiB) |
| ---: | --- | ---: | ---: |
| 10,000 | r1 | 367.4 | 2.63 |
| 10,000 | r2 | 377.6 | 2.71 |
| 10,000 | r3 | 335.1 | 2.84 |
| 50,000 | r1 | 360.0 | 7.11 |
| 50,000 | r2 | 472.8 | 6.67 |
| 50,000 | r3 | 448.1 | 7.22 |
| 100,000 | r1 | 450.1 | 8.70 |
| 100,000 | r2 | 416.1 | 8.81 |
| 100,000 | r3 | 528.6 | 8.06 |
| 500,000 | r1 | 1,069.0 | 16.12 |
| 500,000 | r2 | 1,056.5 | 13.52 |
| 500,000 | r3 | 1,071.7 | 15.26 |
| 1,000,000 | r1 | 1,717.6 | 13.26 |
| 1,000,000 | r2 | 1,489.8 | 12.99 |
| 1,000,000 | r3 | 1,468.8 | 12.55 |
| 5,000,000 | r1 | 5,421.4 | 31.44 |
| 5,000,000 | r2 | 5,530.7 | 32.40 |
| 5,000,000 | r3 | 4,667.6 | 32.43 |
| 10,000,000 | r1 | 8,075.3 | 33.75 |
| 10,000,000 | r2 | 11,211.4 | 29.87 |
| 10,000,000 | r3 | 10,754.4 | 32.12 |

## Measurement record

The original full funnels completed on 2026-08-21 from commit `84ab362`.
The source was the public CELLxGENE H5AD with dataset ID
`dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3` and version ID
`1bc30289-9565-4099-abf9-3326328c11ac`. Samples were deterministic and nested,
using seed 0 at every target size.

- Analysis: 1,000 highly variable features, 21 PCA dimensions, 11 neighbours,
  1,000 embedding centroids, 300 UMAP epochs, and igraph Leiden at resolution
  1.0.
- Filtering: 1st and 99th percentile cell filters, at least 10 features per
  cell, and at least 20 cells per feature.
- Execution: parallel ANN and UMAP, one worker per CPU, a Scarf memory budget
  equal to 75% of container memory, 1 GB count-matrix units, and 100 MB chunks.
- Seeds: graph 4466; UMAP and Leiden 4444.

Parallel ANN and UMAP mean graph-derived outputs are not bitwise reproducible.
These measurements establish execution and resource use for this recorded
configuration. They are not a hardware guarantee or biological validation.

## Running the profiler

`profiling/config.example.toml` is an operational template, not an exact
reproduction of these measurements. Copy it to the ignored
`profiling/config.toml`, configure the object store and Modal secret, set a
fresh `runTag`, and use `fixedResources` for a same-box timing run. Deployment
is a user action.

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
`funnel.json`.
