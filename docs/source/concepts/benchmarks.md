(benchmarks)=
# Benchmarks

These empirical reference runs measure one fixed workflow, dataset, software
revision, and cloud resource envelope. The largest completed run processed
10 million input cells through conversion, quality control, normalization,
graph construction, embedding, clustering, and marker search in 2.78 hours,
with 31.9 GiB mean peak memory on a 16 CPU, 64 GiB container.

The results establish execution and resource use for this configuration. They
are not hardware guarantees, biological validation, or a comparison with
another package.

## 1. End-to-end results

The table reports end-to-end wall time and sampled peak memory for each input
size.

| Input cells | CPU | Container | n | Wall time | Peak memory |
| ----------: | --: | --------: | -: | --------: | ----------: |
| 10,000 | 4 | 16 GiB | 3 | 6.0 ± 0.4 min | 2.7 GiB |
| 50,000 | 4 | 16 GiB | 3 | 7.1 ± 1.0 min | 7.0 GiB |
| 100,000 | 4 | 16 GiB | 3 | 7.7 ± 1.0 min | 8.5 GiB |
| 500,000 | 8 | 32 GiB | 3 | 17.8 ± 0.1 min | 15.0 GiB |
| 1,000,000 | 8 | 32 GiB | 3 | 26.0 ± 2.3 min | 12.9 GiB |
| 5,000,000 | 16 | 64 GiB | 3 | 1.45 ± 0.13 h | 32.1 GiB |
| 10,000,000 | 16 | 64 GiB | 3 | 2.78 ± 0.47 h | 31.9 GiB |

`±` is the sample standard deviation. Peak memory is sampled
`memory.current`. Individual replicate totals are in the table below.

| Input cells | Replicate | Wall time (s) | Peak memory (GiB) |
| ----------: | --- | ------------: | ----------------: |
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

(umap-gallery)=
## 2. UMAPs across scale

These embeddings show the saved output from three reference runs, colored by
CELLxGENE development stage.

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
separately.

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

At 10M, writing `countsT` was the largest stage; marker search and UMAP were
next. At the smallest sizes, fixed work makes the 10k and 50k totals similar
despite the difference in cell count.

(what-was-measured)=
(shared-analysis-settings)=
(machine-classes)=
(how-to-read-these-numbers)=
## 4. Method and limits

| Item | Recorded setting |
| --- | --- |
| Measurement | Completed 2026-08-21 from commit `84ab362345c66d34a0083af5e2609c7762d321a5` |
| Source | CELLxGENE dataset `dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3`, version `1bc30289-9565-4099-abf9-3326328c11ac` |
| Sampling | Nested deterministic samples, seed 0 |
| Analysis | 1,000 highly variable features, 21 PCA dimensions, 11 neighbours, 1,000 embedding centroids |
| Graph and clustering | Graph seed 4466; 300 UMAP epochs; UMAP and Leiden seed 4444; igraph Leiden at resolution 1.0 |
| Filtering | 1st and 99th cell quantiles; minimum 10 features per cell and 20 cells per feature |
| Execution | Parallel ANN and UMAP on S3-compatible object storage in the Modal EU region; one worker per CPU; 1 GB count-matrix units and 100 MB chunks |
| Memory planning | Scarf budget set to 75% of each container memory limit |

- Machine size grew with input size. Compare rows only with their recorded
  resource envelope.
- Peak values are sampled, so short memory spikes may be missed.
- Three replicates show run-to-run drift but are insufficient for a useful
  confidence interval.
- Parallel ANN and UMAP mean graph-derived outputs are not bitwise
  reproducible.
- These measurements establish execution and resource use for this
  configuration. They do not establish biological correctness or a general
  hardware guarantee.

See {doc}`memory_and_execution` for how `mem_budget` controls planned block
sizes and concurrency. It is not a hard process-memory limit.
