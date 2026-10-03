(memory_and_execution)=
# Scale, memory, and execution

Scarf keeps large matrices in Zarr and streams planned blocks through memory.
It does not load the full count matrix simply because a `DataStore` is opened.
Memory use still depends on the operation: graph construction and clustering can hold structures in addition to the streamed count blocks.

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
Each operation reserves within it the blocks it reads and what Zarr holds while it reads them: for a sharded array, such as the counts, a shard-level copy of the selection and the compressed chunks the read touches in a shard, and for every array the chunks it decodes.
For a block transformed by a chain of element-wise steps, such as library-size scaling followed by a logarithm, it reserves the step of the chain that holds the most, one step's input beside its output.
It also reserves its kernel scratch and results, and limits its concurrent reads and writes and the width of automatically sized feature batches to fit.
Imports write the default count layout unless they are given a `policy`, and an import whose default count shards do not fit stops before it writes, naming the smaller `policy` that fits; subset, repack, merge, and grouped assays instead fit the count layout to the budget, writing smaller count shards. The layout changes how the counts are stored, never the store's identity.
It is an operation budget, not a hard cap on total process resident memory.
Memory the process already holds when an operation starts, such as the interpreter, native libraries, graph structures, and earlier results, is not subtracted, so a process can peak at its resident memory plus the budget.
Outside these reservations, a Zarr codec thread can keep a decoded chunk for a moment after its read, and decoding a chunk of an unsharded array, such as normalized data, also holds that chunk's compressed bytes; leave host headroom.
glibc can also keep freed buffers of 32 MiB or less resident after an operation; setting the `MALLOC_MMAP_THRESHOLD_` environment variable, for example to `131072`, before Python starts returns them to the system.

### Worker concurrency

Worker concurrency is auto-detected from the process environment.
On shared hosts, multiprocess jobs, or remote object stores, set `SCARF_WORKERS` or pass the advanced `nthreads` constructor argument to bound the maximum worker budget.
More workers can increase concurrent buffers or remote requests, so pair a large worker budget with an explicit `mem_budget`.
Opt-in parallel UMAP, tSNE, and ANN index builds record the resolved worker count in artifact provenance; pass `nthreads` explicitly when the same parallel request must stay reuse-eligible across machines.

## Why RNA stores have two count orientations

Most analysis steps walk the matrix by cell.
Quality control, library-size normalization, and graph construction read rows of `counts`.
Gene-wise steps such as highly variable gene selection and marker search walk the matrix by feature.

Those two access patterns compete on a single physical layout. Scarf therefore stores RNA counts
twice: `counts` is cell-major, and `countsT` contains the same values in gene-major order. They are
two orientations of one assay matrix, not two datastores. The second orientation roughly doubles
stored RNA counts. ATAC, ADT, and other non-RNA assays keep only `counts`.

Current RNA stores require a matching current-layout `countsT`. Store inspection, import, and
offline rewrite procedures belong to {doc}`../tutorials/import_and_export` and the
{doc}`../reference/faq`, rather than the memory-planning model on this page.

## Storage profiles

New writers use Zarr v3 and choose one of two profiles:

- `fast_local` uses LZ4 with bit-shuffle and is selected automatically for local filesystem, memory, and `file://` targets.
- `cloud` uses Zstandard level 3 and is selected automatically for remote URI targets and other non-local stores.

Override automatic selection with a writer's `profile=`, a datastore's `zarrProfile=`, or `SCARF_ZARR_PROFILE=fast_local|cloud`.
The profile determines the physical encoding when arrays are written.
Changing it while reopening an existing store does not rewrite the arrays.

Direct object-store templates, the distinction between a mounted count source and analysis
target, and `local_cache` scratch behavior are covered in
{doc}`../tutorials/remote_stores`. Its executable example downloads first and mounts two local
paths; it is not evidence of remote execution.

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

Measured end-to-end wall times, sampled peak memory, machine classes, and per-stage timings for a
fixed S3-compatible object-store workflow are published in {doc}`benchmarks`. That page records
the exact dataset, software revision, cloud region, resource envelope, and analysis settings.

Dataset sparsity, selected features, graph parameters, storage latency, software version, and
cache state all affect the result. The rows use different machine sizes and must not be read as a
same-machine scaling curve, a remote-versus-local comparison, or a comparison with another
package.
