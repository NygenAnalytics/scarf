(memory_and_execution)=
# Scale, memory, and execution

Scarf keeps large matrices in Zarr and streams planned blocks through memory.
It does not load the full count matrix simply because a `DataStore` is opened.
Memory use still depends on the operation: graph construction, clustering, and marker batches can hold structures in addition to the streamed count blocks.

## Resource controls

### Memory budget

Set a memory budget when sharing a machine, submitting a batch job, or testing how an analysis behaves under a smaller memory allowance:

```python
ds = scarf.DataStore(
    "data.zarr",
    mem_budget="16G",
)
```

`mem_budget` accepts bytes, a size such as `"8G"`, or a fraction of detected system memory such as `"0.6"`.
It bounds planned streaming blocks, concurrent writes, and automatically sized feature batches.
It is a software planning budget, not a hard cap on total process resident memory.
Python, native libraries, graph structures, and allocator overhead can take total RSS above the configured value, so leave host headroom.

### Worker concurrency

Worker concurrency is auto-detected from the process environment.
On shared hosts, multiprocess jobs, or remote object stores, set `SCARF_WORKERS` or pass the advanced `nthreads` constructor argument to bound the maximum worker budget.
More workers can increase concurrent buffers or remote requests, so pair a large worker budget with an explicit `mem_budget`.
Opt-in parallel UMAP, tSNE, and ANN index builds record the resolved worker count in artifact provenance; pass `nthreads` explicitly when the same parallel request must stay reuse-eligible across machines.

## Count orientations

Most analysis steps walk the matrix by cell.
Quality control, library-size normalization, and graph construction read rows of `counts`.
Gene-wise steps such as highly variable gene selection and marker search walk the matrix by feature.

Those two access patterns fight each other on a single layout.
Scarf therefore stores RNA counts twice: `counts` is cell-major, and `countsT` is the same values in gene-major order.
They are two orientations of one assay matrix, not two datastores.
Import, subset, merge, and `repack_zarr` write both on Zarr v3.
The extra copy roughly doubles stored RNA counts.
ATAC, ADT, and other non-RNA assays keep only `counts`.

Opening an RNA assay fails if `countsT` is missing, incomplete, not a Zarr v3 sharded array, or does not match `counts`.
There is no silent rewrite on open.
Rebuild the store from the source, or run `python -m scarf.tools.repack_zarr`.
After a rewrite, recompute HVG, normalization, PCA, graph, and marker results.
Do not resume those artefacts from the pre-rewrite store.

## Storage profiles

New writers use Zarr v3 and choose one of two profiles:

- `fast_local` uses LZ4 with bit-shuffle and is selected automatically for local filesystem, memory, and `file://` targets.
- `cloud` uses Zstandard level 3 and is selected automatically for remote URI targets and other non-local stores.

Override automatic selection with a writer's `profile=`, a datastore's `zarrProfile=`, or `SCARF_ZARR_PROFILE=fast_local|cloud`.
The profile determines the physical encoding when arrays are written.
Changing it while reopening an existing store does not rewrite the arrays.

Credentials, mounted stores, repacking, and `local_cache` scratch are covered in {doc}`../tutorials/remote_stores`.

## Batch and HPC output

Disable animated progress in non-interactive logs and add timestamps:

```python
import scarf

scarf.configure_output(progress=False, timestamps=True)
```

Progress rendering and log severity are independent.
To also select a log level or file:

```python
scarf.set_verbosity(level="INFO", filepath="scarf-run.log")
```

See {doc}`../reference/api/utilities` for the exact output contract.

## Measured scaling references

Measured end-to-end wall times, peak memory, machine classes, and per-stage timings for a fixed object-store workflow are published in {doc}`benchmarks`.
Dataset sparsity, selected features, graph parameters, storage latency, software version, and cache state all affect the result.
Those rows use different machine sizes and must not be read as one same-machine scaling curve.
