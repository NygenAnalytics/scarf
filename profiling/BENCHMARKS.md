# Scarf profiling references

This file records the latest completed full-scale Scarf measurements. It does
not preserve superseded experiments or operational call identifiers.

## Scope

The measurements were completed on 2026-07-29 and 2026-07-30 from commit
`ce50bbced52e3f8177ebfb492a04ef474dff6918`. Both runs downloaded an H5AD,
used a fresh object-store-backed Zarr v3 store, and completed the same
sixteen-stage CPU workflow:

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
nested samples with seed 0. The prepared inputs contained exactly 1,000,000 and
10,000,000 cells.

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

The Scarf memory budget controls planned block sizes and concurrency. It is not
a hard process-memory limit.

## End-to-end results

| Input cells | Cells after QC | CPU | Container memory | Scarf budget | Wall time | Peak cgroup memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 889,974 | 8 | 32 GiB | 24 GiB | 2,458.5 s (41.0 min) | 28.5 GiB |
| 10,000,000 | 8,902,268 | 16 | 128 GiB | 96 GiB | 37,118.9 s (10.31 h) | 105.0 GiB |

The source H5AD files were 3.748 GiB and 37.434 GiB. The ten-million-cell run
used twice the CPUs and four times the container and software memory, so these
rows are not a same-machine scaling curve.

Stage values below use `wholeFunctionSeconds`, which includes recorded stage
setup and validation. The funnel total also includes the initial download and
about six seconds of orchestration.

| Stage | 1M seconds | 10M seconds | 10M / 1M |
| --- | ---: | ---: | ---: |
| Dataset download | 90.1 | 2,992.2 | 33.2 |
| Create count store | 404.6 | 7,316.0 | 18.1 |
| Write `countsT` | 180.1 | 2,768.7 | 15.4 |
| Initialize datastore | 101.1 | 1,842.5 | 18.2 |
| Reopen datastore | 8.1 | 10.4 | 1.3 |
| Filter cells | 77.5 | 124.4 | 1.6 |
| Mark HVGs | 172.2 | 3,444.5 | 20.0 |
| Normalize | 168.3 | 2,201.2 | 13.1 |
| PCA | 86.1 | 1,135.9 | 13.2 |
| Build embedding initialization | 33.8 | 341.8 | 10.1 |
| Build ANN index | 95.7 | 1,060.5 | 11.1 |
| Query neighbours | 51.9 | 311.7 | 6.0 |
| Build connectivity map | 57.1 | 87.7 | 1.5 |
| UMAP | 314.6 | 2,051.6 | 6.5 |
| Leiden | 269.3 | 3,446.5 | 12.8 |
| Paris | 114.0 | 793.3 | 7.0 |
| Marker search | 228.2 | 7,183.1 | 31.5 |
| **Funnel total** | **2,458.5** | **37,118.9** | **15.1** |

## Interpretation

Each row is one completed run, not a median of repeated runs. Peak cgroup values
were sampled from `memory.current`, so very short spikes may be missed. The
profiler did not force garbage collection between stages.

Marker search set the peak-memory value in both runs. At ten million cells, its
block reads took 6,481.4 seconds and its rank computation took 613.0 seconds.
The planned feature blocks consumed enough of the 96 GiB software budget that
inner read concurrency fell to one. The measured 105.0 GiB peak is therefore a
throughput choice, not a fixed memory requirement for ten million cells.

Storage throughput also varied. The input size grew by 9.99 times, while
download wall time grew by 33.2 times because average throughput fell from
42.6 MiB/s to 12.8 MiB/s. Excluding download, total wall time scaled by 14.4
times.

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
