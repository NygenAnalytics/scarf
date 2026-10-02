(benchmarks)=
# Benchmarks

These empirical reference runs measure one fixed object-store workflow, dataset, software
revision, and cloud resource envelope. Execution used S3-compatible object storage in the Modal
EU region. The largest completed size processed 10 million input cells through conversion,
quality control, normalization, graph construction, embedding, clustering, and marker search in
88.8 minutes on average, with 38.0 GiB mean sampled peak memory on a 16 CPU, 64 GiB container.

The results establish execution and resource use for this recorded configuration. No local
filesystem baseline was collected, so they do not measure an object-store penalty or compare
remote and local execution. They are not hardware guarantees, biological validation, proof for
another object-store environment, or a comparison with another package.

## 1. End-to-end results

The table reports end-to-end wall time and sampled peak memory for each input
size.

| Input cells | CPU | Container | n | Wall time | Peak memory |
| ----------: | --: | --------: | -: | --------: | ----------: |
| 10,000 | 4 | 16 GiB | 3 | 5.1 ± 1.8 min | 2.7 GiB |
| 50,000 | 4 | 16 GiB | 3 | 4.6 ± 0.4 min | 7.1 GiB |
| 100,000 | 4 | 16 GiB | 3 | 5.7 ± 0.7 min | 8.7 GiB |
| 500,000 | 8 | 32 GiB | 3 | 9.7 ± 0.3 min | 15.7 GiB |
| 1,000,000 | 8 | 32 GiB | 3 | 15.3 ± 1.7 min | 16.5 GiB |
| 5,000,000 | 16 | 64 GiB | 3 | 41.4 ± 2.6 min | 46.6 GiB |
| 10,000,000 | 16 | 64 GiB | 3 | 88.8 ± 0.1 min | 38.0 GiB |

`±` is the sample standard deviation. Peak memory is the sampled resident
memory of the process tree. Individual replicate totals are in the table below.

| Input cells | Replicate | Wall time (s) | Peak memory (GiB) |
| ----------: | --- | ------------: | ----------------: |
| 10,000 | r1 | 430.8 | 2.74 |
| 10,000 | r2 | 237.7 | 2.69 |
| 10,000 | r3 | 252.1 | 2.74 |
| 50,000 | r1 | 294.8 | 6.84 |
| 50,000 | r2 | 280.6 | 8.06 |
| 50,000 | r3 | 248.2 | 6.42 |
| 100,000 | r1 | 365.5 | 9.03 |
| 100,000 | r2 | 372.4 | 7.92 |
| 100,000 | r3 | 294.3 | 9.04 |
| 500,000 | r1 | 601.0 | 15.95 |
| 500,000 | r2 | 581.6 | 15.36 |
| 500,000 | r3 | 562.2 | 15.90 |
| 1,000,000 | r1 | 907.7 | 16.64 |
| 1,000,000 | r2 | 1,027.3 | 16.07 |
| 1,000,000 | r3 | 818.2 | 16.79 |
| 5,000,000 | r1 | 2,667.2 | 51.60 |
| 5,000,000 | r2 | 2,379.4 | 43.12 |
| 5,000,000 | r3 | 2,412.9 | 45.16 |
| 10,000,000 | r1 | 5,319.5 | 38.27 |
| 10,000,000 | r2 | 5,322.8 | 36.81 |
| 10,000,000 | r3 | 5,334.4 | 38.97 |

(umap-gallery)=
## 2. UMAPs across scale

These embeddings show saved output from three runs of the earlier 2026-08-21
measurement at commit `84ab362`, colored by CELLxGENE development stage.

::::{container} benchmark-gallery
:::{figure} ../_static/benchmarks/umap_development_stage_100000.png
:alt: UMAP from the 100,000-cell reference run colored from early to late Theiler stage
:width: 100%

**100k input:** 88,955 filtered cells shown
:::
:::{figure} ../_static/benchmarks/umap_development_stage_1000000.png
:alt: UMAP from the 1,000,000-cell reference run colored from early to late Theiler stage
:width: 100%

**1M input:** 500,000 of 889,974 filtered cells shown
:::
:::{figure} ../_static/benchmarks/umap_development_stage_10000000.png
:alt: UMAP from the 10,000,000-cell reference run colored from early to late Theiler stage
:width: 100%

**10M input:** 500,000 of 8,902,268 filtered cells shown
:::
::::

:::{image} ../_static/benchmarks/umap_development_stage_legend.png
:alt: Development-stage color key from Theiler stage 12 through Theiler stage 27
:class: benchmark-gallery-legend
:width: 92%
:align: center
:::

Each input size has an independently fitted UMAP, so coordinates are not
aligned across panels. Development stages came from the source CELLxGENE
metadata using the deterministic sample rows, with cell IDs used to verify the
mapping. These panels provide visual context, not biological validation.

(stage-timings)=
## 3. Stage breakdown

Values are elapsed seconds for each stage. All columns are means of three
replicates. Stage values exclude orchestration, while dataset download is shown
separately. In this measurement the profiling funnel ran UMAP on a background
thread while Leiden ran in a child process, so the two stages overlapped in time
and shared their memory window. `DataStore.pipeline` runs every stage in
sequence, so its totals include both stages.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 1.9 | 1.6 | 16.1 | 31.6 | 17.5 | 139.1 | 323.2 |
| Create count store | 9.4 | 17.6 | 26.1 | 56.3 | 131.5 | 346.6 | 644.4 |
| Write `countsT` | 12.8 | 14.0 | 23.0 | 46.8 | 78.6 | 156.4 | 429.2 |
| Initialize datastore | 15.4 | 24.3 | 20.2 | 17.6 | 23.0 | 35.7 | 50.6 |
| Reopen datastore | 3.0 | 2.7 | 2.9 | 2.9 | 3.2 | 3.4 | 4.5 |
| Filter cells | 15.8 | 5.6 | 6.2 | 6.6 | 9.0 | 13.9 | 19.7 |
| Select HVGs | 50.6 | 18.0 | 21.8 | 32.5 | 55.2 | 110.0 | 334.3 |
| Normalize | 9.1 | 10.4 | 12.3 | 20.3 | 37.6 | 142.6 | 321.9 |
| PCA | 17.8 | 13.9 | 17.3 | 33.3 | 55.8 | 123.7 | 277.3 |
| Build embedding initialization | 12.0 | 9.0 | 11.5 | 16.1 | 23.4 | 74.0 | 140.6 |
| Build ANN index | 10.9 | 9.1 | 10.6 | 16.3 | 26.6 | 61.9 | 121.0 |
| Query neighbours | 10.8 | 19.6 | 12.8 | 18.7 | 36.1 | 67.7 | 119.1 |
| Build connectivity map | 59.8 | 40.3 | 65.1 | 70.2 | 62.1 | 57.9 | 60.0 |
| UMAP | 18.4 | 27.1 | 42.3 | 107.2 | 194.5 | 501.9 | 1,125.3 |
| Leiden | 8.1 | 20.9 | 10.6 | 22.7 | 40.8 | 183.6 | 404.3 |
| Marker search | 41.7 | 32.1 | 32.3 | 47.0 | 72.8 | 149.8 | 409.9 |

At 10M, UMAP was the largest stage and creating the count store was next.
Leiden ran within the UMAP window of the funnel, so its seconds overlap UMAP's
instead of adding to the total. At the smallest sizes, fixed work dominates, so the 10k,
50k, and 100k totals are similar despite the difference in cell count.

(what-was-measured)=
(shared-analysis-settings)=
(machine-classes)=
(how-to-read-these-numbers)=
## 4. Method and limits

| Item | Recorded setting |
| --- | --- |
| Measurement | Completed 2026-09-29 from commit `7291ed45106ec1478750d7363392f0d4020f8ba4` |
| Source | CELLxGENE dataset `dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3`, version `1bc30289-9565-4099-abf9-3326328c11ac` |
| Sampling | Nested deterministic samples, seed 0 |
| Analysis | 1,000 highly variable features, 21 PCA dimensions, 11 neighbours, 1,000 embedding centroids |
| Graph and clustering | Graph seed 4466; 300 UMAP epochs; UMAP and Leiden seed 4444; igraph Leiden at resolution 1.0 |
| Filtering | 1st and 99th cell quantiles; minimum 10 features per cell and 20 cells per feature |
| Execution | Parallel ANN and UMAP on S3-compatible object storage in the Modal EU region; one worker per CPU; 1 GB count-matrix units and 100 MB chunks; funnel UMAP on a background thread beside a Leiden child process |
| Memory planning | Scarf budget set to 75% of each container memory limit |

- Machine size grew with input size. Compare rows only with their recorded
  resource envelope.
- Peak values are sampled, so short memory spikes may be missed.
- Memory still held from earlier stages adds to what an operation plans, so a
  run's peak can exceed the budget. The 5M r1 peak of 51.6 GiB exceeded its
  48 GiB budget; marker search alone added about 32 GiB.
- Three replicates show run-to-run drift but are insufficient for a useful
  confidence interval.
- Parallel ANN and UMAP mean graph-derived outputs are not bitwise
  reproducible.
- These measurements establish execution and resource use for this
  configuration. They do not establish biological correctness or a general
  hardware guarantee.

See {doc}`memory_and_execution` for how `mem_budget` controls planned block
sizes and concurrency. It is not a hard process-memory limit. See
{doc}`../tutorials/remote_stores` for mounted source/target mechanics and non-executed direct-store
templates. That guide's executable example downloads first and is separate from these measured
object-store runs.
