(scale_and_memory)=
# Scale, memory, and execution

Scarf keeps large matrices in Zarr and streams planned blocks through memory.
It does not load the full count matrix simply because a `DataStore` is opened.
Memory use still depends on the operation: graph construction, clustering, and
marker batches can hold structures in addition to the streamed count blocks.

## Resource controls

Set a budget when sharing a machine, submitting a batch job, or testing how an
analysis behaves under a smaller memory allowance:

```python
ds = scarf.DataStore(
    "data.zarr",
    mem_budget="16G",
    nthreads=8,
)
```

`mem_budget` accepts bytes, a size such as `"8G"`, or a fraction of detected
system memory such as `"0.6"`. It bounds planned streaming blocks, concurrent
writes, and automatically sized feature batches. It is a software planning
budget, not a hard cap on total process resident memory. Python, native
libraries, graph structures, and allocator overhead can take total RSS above
the configured value, so leave host headroom.

`nthreads` controls I/O and compute concurrency used by methods that support it.
More workers can increase throughput when storage and CPU have capacity, but
they can also increase concurrent buffers or contend for a slow remote store.
Measure the complete workflow rather than assuming that the largest value is
best.

## Count orientations

`counts` is the primary cell-major count array. It supports cell-wise scans used
by normalization and graph construction. A store may also contain `countsT`, a
derived feature-major orientation used by gene-wise stages such as HVG and
marker calculations.

These are two orientations of the same assay matrix, not independent
datastores. Writing `countsT` costs time and storage during conversion, but can
reduce repeated strided reads later. A store without `countsT` remains valid.

## Storage profiles

New writers use Zarr v3 and choose one of two profiles:

- `fast_local` uses LZ4 with bit-shuffle and is selected automatically for
  local filesystem, memory, and `file://` targets.
- `cloud` uses Zstandard level 3 and is selected automatically for remote URI
  targets and other non-local stores.

Override automatic selection with a writer's `profile=`, a datastore's
`zarrProfile=`, or `SCARF_ZARR_PROFILE=fast_local|cloud`. The profile determines
the physical encoding when arrays are written. Changing it while reopening an
existing store does not rewrite the arrays.

Credentials, mounted stores, repacking, and `local_cache` scratch are covered
in {doc}`../tutorials/remote_stores`.

## Batch and HPC output

Disable animated progress in non-interactive logs and add timestamps:

```python
import scarf

scarf.configure_output(progress=False, timestamps=True)
```

Progress rendering and log severity are independent. To also select a log level
or file:

```python
scarf.set_verbosity(level="INFO", filepath="scarf-run.log")
```

See {doc}`../reference/api/utilities` for the exact output contract.

## Measured scaling references

The following measurements are empirical references, not hardware guarantees.
They cover all active cells through store preparation, QC, HVGs, graph, UMAP,
Leiden, and markers. Dataset sparsity, selected features, graph parameters,
storage latency, software version, and cache state all affect the result.

One staged object-storage series used feature-major `countsT` and right-sized
resources for each stage:

| Cells | End-to-end wall time | Peak process or cgroup memory |
|---:|---:|---:|
| 100,000 | 881 s | 6.7 GiB |
| 500,000 | 2,825 s | 16.6 GiB |
| 5,000,000 | 29,465 s (8.2 h) | 33.0 GiB |
| 10,000,000 | about 22.8 h | 36.2 GiB |

The 10-million-cell total replaces one cached HVG timing with a full-compute
estimate. The series used staged, right-sized resources and should not be read
as a fixed-machine benchmark.

More recent same-container, current-code references used different resource
envelopes:

| Cells | CPU | Host memory | Scarf budget | Wall time | Peak cgroup memory |
|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 8 | 32 GiB | 24 GiB | 2,458 s (41.0 min) | 28.5 GiB |
| 10,000,000 | 16 | 128 GiB | 96 GiB | 37,119 s (10.31 h) | 105.0 GiB |

The marker stage drove both peaks and tracks the configured software budget.
The 105 GiB result is therefore not a fixed memory floor for ten million cells.
Reducing the budget should shrink marker batches but can increase wall time.
These rows must not be combined with the staged series as one scaling curve.

For one bounded storage comparison at 100,000 cells, the same 8-CPU,
32-GiB-host, 24-GiB-budget workflow took 421 seconds on ephemeral local disk and
735 seconds against the prepared object-store layout. Local disk is a useful
ceiling for that experiment, not a promise for a cloud service or network.
