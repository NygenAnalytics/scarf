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

Some steps hold whole structures in memory beside their streamed blocks, and compare an estimate of them with the budget before they read a coordinate or graph:

- an approximate nearest-neighbour index holds every cell's coordinates and links, about `4 * dims + 8 * ann_m + 130` bytes per cell on x86-64 Linux, while `build_ann_index` builds and saves it and while `query_neighbors` loads and queries it; a query also holds the indices and distances of every cell's neighbours, 8 bytes per neighbour;
- an in-memory k-means embedding initialization holds the coordinates of every cell;
- cluster selection holds the coordinates of its sample of at most 10,000 cells, in their stored dtype and as float64, beside the silhouette's distance chunks, which use a quarter of the budget up to 1 GiB;
- Harmony holds the float64 coordinates several times over, float64 matrices of soft cluster assignments with `nclust` values per cell, and the one-hot design of its batch levels, about 5 KB per cell with 30 dimensions and 100 clusters, beside the batch labels it read, about 60 bytes per cell for each column of short labels;
- WNN integration holds every assay's neighbours and coordinates and the integrated graph;
- SNN integration holds every source graph and a float64 matrix of shared-neighbour fractions per graph.

Over the budget, such a step raises `MemoryError` before it reads anything or creates or loads an index, naming the bytes, the cells and other sizes, and the limit; it never changes its algorithm on its own.
The estimates are near-exact counts of what each step allocates, calibrated against the peaks of its traced allocations (with `tracemalloc`, or for the hnswlib index with glibc's count of allocated bytes), plus stated allowances for buffers the trace does not see, such as scikit-learn's per-thread buffers.
The budget compares them with its limit; they are not a cap on the memory of the process, which also holds the interpreter, native libraries and their own buffers, and earlier results.
A coordinate stream's reads reserve what its consumer holds while it reads, and a consumer releases each block before the stream reads the next.
Cluster selection reads its sampled cells in blocks of the stored coordinates, so it decodes every stored chunk that holds one of them, which for a sample spread over a large normalized matrix is most of it.

`build_ann_index` saves the index through a temporary file of about `4 * dims + 8 * ann_m + 20` bytes per cell, and `query_neighbors` loads it through another one of the same size, in the system's temporary directory (`TMPDIR` on Linux and macOS); leave that much free space there.

A pipeline run with `pca_dims=0` builds its graph on the normalized values of the selected features, one coordinate per highly variable gene, so the memory and time of the ANN index, the neighbour queries, the embedding initialization, and cluster selection scale with the HVG count instead of the PCA dimensions.
At 1,000,000 cells the index alone holds about 4.5 GB with 1,000 HVGs and 8.5 GB with 2,000, and the embedding initialization holds every cell's coordinates, 4 bytes per HVG per cell, when one stored band holds every cell; pass `params={"embedding_initialization": {"batch_size": ...}}` with a batch size below the cell count to select its streamed fit, which holds a sample of the cells and one block of coordinates at a time.

Outside these reservations, a Zarr codec thread can keep a decoded chunk for a moment after its read, and decoding a chunk of an unsharded array, such as normalized data, also holds that chunk's compressed bytes; leave host headroom.
The identity checks of stored cell and feature selections, which most operations run once on their inputs, read the row identifiers in bands of whole chunks up to 64 MiB and hold one band, with its selected rows where they check a row order, outside the budget.
glibc can also keep freed buffers of 32 MiB or less resident after an operation; setting the `MALLOC_MMAP_THRESHOLD_` environment variable, for example to `131072`, before Python starts returns them to the system.

### Worker concurrency

Worker concurrency is auto-detected from the process environment.
On shared hosts, multiprocess jobs, or remote object stores, set `SCARF_WORKERS` or pass the advanced `nthreads` constructor argument to bound the maximum worker budget.
More workers can increase concurrent buffers or remote requests, so pair a large worker budget with an explicit `mem_budget`.
Opt-in parallel UMAP layouts and ANN index builds are faster but not reproducible: two runs on several threads can differ even with the same thread count.
Their artifact identities therefore record only the `parallel` or `ann_parallel` flag, and the thread counts are execution options, so the same request reuses its result on any machine.
t-SNE always runs on one thread through the optional `sgtsnepi` package, so its artifacts record no worker count.

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
