(scale_and_memory)=
# Scale, memory, and execution

Scarf streams counts from Zarr and bounds working-set size with an explicit
memory budget. The figures below are measured end-to-end funnels on
feature-major `countsT` stores (create store through marker search), not
projections. Hardware, storage latency, and parameters all move wall time.

## Measured anchors

Cloud funnel on 8 CPU cores with the countsT speed pack (object-store backed
Zarr). Peak RAM is the stage max under that campaign; graph construction sets
most of the ceiling at large N.

| Cells | Peak RAM | End-to-end wall time |
|---|---|---|
| 100k | ~7 GiB | ~15 min |
| 500k | ~25 GiB | ~47 min |
| 5M | ~33 GiB | ~8.2 h |
| 10M | ~37 GiB | ~22.8 h |

The 10M total uses a corrected HVG estimate; one recorded HVG stage JSON was a
cache hit and must not be treated as a full compute. See
`profiling/LEARNINGS.md` in the repository for stage-level tables.

These runs analyze **all active cells** under the fixed budget: QC, HVGs,
graph, UMAP, Leiden, and markers on the full selected set. That is a different
job from sketch-and-project workflows that fit neighbors only on a subset.

## Memory budget and workers

Open a store with an explicit budget when you share a machine or want
predictable streaming tiles:

```python
ds = scarf.DataStore(
    "data.zarr",
    mem_budget="16G",
    nthreads=8,
    working_copies=8,
)
```

Three knobs matter:

| Knob | Role |
|---|---|
| `mem_budget` | Total software memory budget. Accepts bytes, a size like `"8G"`, or a fraction of system RAM like `"0.6"`. Drives write-time chunk geometry and auto marker batch size. |
| `nthreads` / workers | Read concurrency and async IO parallelism. Also the default thread count for multi-threaded steps. |
| `working_copies` | How many concurrent in-memory copies the budget is divided across when sizing tiles. Default is 8. Treat it as a copy-count model, not a speed dial. |

Once arrays are written, reads follow on-disk chunk and shard geometry. Raising
host RAM without raising `mem_budget` does not enlarge auto marker batches.

Within a reduction block, BLAS threads stay pinned so numeric results remain
stable across worker counts where that contract is tested. Extra BLAS threads
are not used as a silent parallelism knob.

## Cloud profile and remote overhead

New stores use Zarr v3. Choose a storage profile with `zarrProfile=` or
`SCARF_ZARR_PROFILE`:

- `fast_local`: local SSD-oriented sharding
- `cloud`: larger count chunks (default target around 128 MiB) for object storage

At 100k cells, the same countsT funnel took about **421 s** on ephemeral local
disk versus about **735 s** on a reorganized R2 store (~1.75× remote overhead).
Remote analysis is still practical; expect IO wait, especially on gene-wise
stages, and prefer `zarrProfile="cloud"` plus `local_cache` for multi-pass graph
steps. See {doc}`../tutorials/remote_stores`.

## Practical sizing

Rough machine floors from measured peaks (countsT, with headroom):

| Cells | Suggested host class for the graph-heavy stages |
|---|---|
| 100k | 16 to 32 GiB |
| 500k | 32 GiB |
| 5M to 10M | 64 GiB for create-store / graph; smaller stages can run leaner |

Gene-wise wall time drops sharply with feature-major `countsT` at mid scale;
create-store pays for writing that layout. Graph RAM still sets the ceiling.

For remote stores, credentials, and scratch cleanup, continue with
{doc}`../tutorials/remote_stores`. For how results are reused instead of
recomputed, see {doc}`provenance`.
