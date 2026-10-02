(zarr_internals)=
# Zarr internals

This page is for contributors and advanced users who need the on-disk layout.
Analysts should start with {doc}`../tutorials/data_organization`.
Why RNA stores two orientations, and what to do when an older store will not open, is in {doc}`../concepts/memory_and_execution`.

## Layout overview

A Scarf Zarr store is a directory hierarchy.
Typical top-level groups include:

- cell metadata
- one or more assay groups (for example `RNA`, `ADT`, `ATAC`)
- per-assay feature metadata
- immutable feature selections, summaries, normalized matrices, reductions, and graph artifacts under assay-specific artifact paths
- durable pipeline records under `pipeline/runs/{runId}` with ordered stage records below each run

Artifact group names use kind and identity, not encoded graph-stage parameters.
Prefer inspecting a store with `zarr.open` or Scarf's `DataStore` summary rather than hard-coding internal paths in analysis scripts.

Analytical outputs are artifacts only. Feature selection, embedding, clustering, score, and marker
operations leave metadata tables unchanged. Consumers validate exact refs,
artifact completion, lineage, and ordered-axis identity.

## Pipeline records and snapshots

Pipeline run and stage documents use one strict unversioned shape. A run has exactly
`runId`, `recipe`, `requestedLabel`, `label`, `assay`, `startedAtNs`, `finishedAtNs`, `status`,
`complete`, `scarfVersion`, `config`, `stageOrder`, `outputs`, `fields`, `error`, and
`interruption`. A stage has exactly `stage`, `ordinal`, `startedAtNs`, `finishedAtNs`, `status`,
`complete`, `outputs`, `plans`, `metrics`, `error`, and `interruption`. Unknown or missing fields
fail closed.

The run record is created before `input_snapshot`; final outputs and field descriptors are written
only when the complete recipe succeeds. A handled failure or interruption first commits terminal
stage and run details. A hard process death can leave a run or stage incomplete. There is no
on-disk resume, repair, or same-ID retry protocol.

Requested run labels use append-only atomic claims below
`pipeline/runs/.label-claims/{labelDigest}` while a run is finalized. The completed run record is
the public label owner. Failed, interrupted, and removed predecessors can be bypassed by a later
claim. A live or unclean incomplete predecessor blocks the same label and fails closed. After the
operator has confirmed that its process stopped, the exact owner can be marked interrupted with
`pipeline.abandon_label_claim(label=..., run_id=..., reason=...)`; Scarf never infers abandonment
from elapsed time. A storage backend without atomic conditional creation rejects a labeled run
before its run record or any computation is started.

Catalog scans skip malformed or torn run children so one crash cannot hide healthy runs. Opening
an exact malformed `run_id` remains strict and reports the bad record.

The first stage stores a cell-selection artifact, a `feature_universe` all-feature selection, and
full-axis cell and feature metadata snapshots. Frozen `run.features["I"]` is backed by that
immutable feature selection rather than the live feature `I` column. Stored selection integrity
compares its Boolean payload and current ordered row-ID fingerprint, not the later value of the
source metadata column. Frozen run views therefore survive live `I` changes but fail if row
identities are replaced or reordered.

Each completed or failed stage stores exact nested artifact-plan dispositions and sampled
process-tree RSS metrics. Artifact reuse comes from planning receipts, never timestamp inference.
Timing, memory, run identity, and reuse state remain in the run ledger and never mutate an artifact.
New artifacts store immutable creation time and creator Scarf version for diagnostics only.

## Removed assay state

The former `{assay}/state` analysis document is not part of this layout. `DataStore` rejects a
store containing that group before it initializes datastore metadata. Scarf does not inspect its
contents, migrate it, or use it to recover a current graph. Re-import or rebuild the dataset with
the current release.

`repack_zarr` copies run records and their append-only label claims because it preserves the axes.
This applies to the root datastore and nested workspaces. Subset and merge outputs do not copy
source runs.
The explicit rewrite omits retired `{assay}/state` groups. Recompute analysis artifacts after the
rewrite; the removed document is never translated into current lineage.
An overwriting merge clears pipeline records and datastore-scoped artifacts in its destination
workspace while preserving unrelated root siblings.

(mounted_targets)=

## Mounted targets

`mount_datastore` creates a target whose root attribute `matrixSource` binds it to the source
that owns the counts. The record holds exactly `location` (an absolute path or URI),
`workspace`, and `assays`, and each assay entry holds exactly `datasetFingerprint`,
`countsFingerprint`, and `requiresTranspose`. Every open checks the source against these
identities. Any other shape, including a field that a later release adds, is rejected rather than
ignored, and a source that is itself a mount is refused.

The target holds copied cell and feature tables, its own artifacts, and its own pipeline runs.
Opening it reopens the target root through a namespace that resolves keys in this order:

1. Keys outside the artifact roots, `[workspace/]artifacts` and `[workspace/]{assay}/artifacts`
   of the mounted assays, belong to the target alone. Cell and feature tables, counts, and
   `pipeline/` never fall back to the source.
2. A key inside an artifact group `{root}/{kind}/{id}` is read from the target when the target
   holds the group's `zarr.json`, and otherwise from the source when the source holds it.
3. Listings of an artifact root and of its kind directories are the union of both stores, and a
   directory document that the target lacks is read from the source.

Every write goes to the target. Writing or deleting inside a source group raises
`PermissionError`, so the source is never modified. Artifact IDs are random 256-bit tokens, so a
group that the source holds is exactly the artifact a copy in the target would be. The target
lists, loads, traces, and reuses source artifacts with their identities unchanged, and a recipe
whose provenance matches the published results creates nothing. Pipeline runs and their atomic
label claims stay per store: open a source run on the source datastore and pass its refs to the
mount.

Refs, provenance, and run records never record a location, and nothing in the target records
which store holds an artifact. Results that use source artifacts therefore need the source, as
counts do, and a source artifact removed by an outside edit fails closed as a missing input. A
target group opened directly with `zarr.open_group` shows only the target's own artifacts.
`repack_zarr` reads a mount through the same namespace, so its output holds the artifacts of both
stores, the target's runs, and copied counts, and no longer needs the source.

## Count arrays

`counts` is the cell-major assay matrix (`n_cells` × `n_features`).
RNA assays (`RNAassay` and aliases such as `GeneActivity` / `URNA`) also store `countsT`, the same values in gene-major order (`n_features` × `n_cells`).
Import, subset, merge, and `repack_zarr` write both arrays together on Zarr v3.
Non-RNA assays (ATAC, ADT, and similar) write `counts` only.

The two RNA arrays are orientations of one matrix, not independent datastores.
New count arrays are sharded Zarr v3 arrays.
Normalized arrays stay unsharded (`shards=None`).

The matrix group, `counts`, and `countsT` each carry a `scarf:countMatrixLayout` attribute.
Scarf checks that those records agree with each other and with the live arrays.
Missing or mismatched layout metadata is an error.
There is no silent rewrite on open and no automatic upgrade of older count layouts.

Writers compute the raw-count identity while they write `counts`, so no second pass reads the
matrix back. `counts` records it as `content_fingerprint`. The sibling `countSummaries` group holds
per-cell totals (`rowSums`), per-cell detected-feature counts (`rowPositive`), and per-feature
detected-cell counts (`columnPositive`), and its `source_fingerprint` attribute names the counts
it describes. Preparation reads these summaries instead of streaming the matrix. Summaries that are
missing, malformed, or bound to other counts are an error; rebuild the store with `--data-only`.

### Count dtype

`counts` and `countsT` share one dtype, which `scarf.storage.count_dtype` resolves from the
canonical (duplicate-summed) values of the assay. Counts are stored unsigned, in the narrowest of
`uint8`, `uint16`, `uint32`, and `uint64` that holds the largest value, exactly when every value
is a non-negative integer. Other counts keep their source dtype: `float32` or `float64`, or a
signed integer dtype when values are negative. `float16` is not a count storage dtype. The dtype
depends only on the assay's own values, not on the reader, source encoding, index order,
orientation, memory budget, or the other assays of a source matrix, so the same counts imported
from H5AD, 10x HDF5, Matrix Market, CSV, an in-memory sparse matrix, or a Seurat object give the
same store identity. Imports that split one source matrix into assays (10x HDF5 and Matrix Market
feature types, H5AD `assay_split_key`) resolve the dtype of each assay from its own features.
Every import reads all its counts once before it creates the destination to find their ranges:
readers of 10x HDF5, Matrix Market, and H5AD files report the range of each group of features
over the selected cells (`count_value_ranges`), the CSV reader finds it in its first pass, and
`SparseToZarr` and `SeuratToZarr` scan their sources. The scans sum duplicate coordinates as the
writers do, integers exactly in 64 bits and floats in source order. No import takes a count dtype
argument. Count matrices hold finite values: writers reject NaN and infinity, and every import
rejects them before it creates the destination. Writers cast through a checked cast, so a count
that the stored dtype cannot hold raises instead of wrapping.

Writers that rebuild or combine existing counts follow their sources instead. Subset and repack
keep the source dtype, because they rebuild an existing dataset whose identity and copied
artifacts must stay valid. A merge stores the common type of its source count dtypes, widened so
that features summed by name cannot overflow it; integer sources without a common integer dtype
are rejected rather than stored as floats. Derived assays (grouped and melded) keep their
`float64` values.

### Count layout

The layout policy (`unitBytes` and `chunkBytes`) is recorded in `scarf:countMatrixLayout`, and
readers and `countsT` writers replay it. Every count writer (the H5AD, 10x HDF5, Matrix Market,
CSV, sparse, and Seurat imports, subset, `repack_zarr`, merge, and `add_grouped_assay`) fits the
policy to `mem_budget` before it creates its destination: when the default policy's counts write
or `countsT` transpose does not fit, it halves `unitBytes` and `chunkBytes` together, keeping the
chunks per shard, down to count shards of one row. Sparse writers admit the band writes of their
sparse sources, and dense writers the dense row bands they write. Writers that choose their source
batches (the sparse imports, the Seurat imports, and merge) admit batches of one destination row
band, the batch their write starts from, so a fitted layout never leaves the write narrower
batches than its bands. An explicit policy is used exactly or refused before the destination
exists. `add_melded_assay` sizes its shards to the melding band that fits `mem_budget`. Writers
that write their assays one at a time (Seurat, subset, repack, and merge) fit each assay on its
own. A resumed merge keeps the layout persisted with its completed counts, so a budget change
between attempts cannot block it, and fits the layout of the counts it rewrites. A store built
with a small budget therefore has smaller shards and more objects. The layout never changes
identity: `content_fingerprint`, the counts fingerprint, and the dataset fingerprint are computed
from the stored values, so only the layout fingerprint differs between budgets.

Existing stores are never rewritten under a new dtype or layout rule and keep their dtype, layout,
and identity. A re-import of the same source can store a different dtype, and so get different
fingerprints and artifact ids; mounts and mapping references bound to the replaced store fail
closed.

## Opening an RNA assay

`RNAassay` construction requires a complete `countsT` on Zarr v3 plus matching layout metadata.
The open fails if `countsT` is missing, incomplete, unsharded, a Zarr v2 array, or out of agreement with `counts`.

`repack_zarr --data-only` reads the raw store without constructing `RNAassay`.
The user-facing repair is to re-import the source or run:

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local --data-only
```

Without `--data-only`, `repack_zarr` accepts only a prepared source and preserves its results after
verifying that the destination has the same dataset identity.

After a rewrite, recompute HVG, normalization, PCA, graph, and marker artefacts.
Do not resume them from pre-rewrite lineage.

Non-RNA assays in a multi-assay store remain usable when tooling opens without forcing every assay class to construct.

## Zarr versions

New stores default to Zarr v3.
RNA assays require v3 because sharded `countsT` is a v3 feature.
A Zarr v2 RNA store will not open as `RNAassay`.

## Memory controls

`DataStore(..., mem_budget='8G')` bounds streaming and concurrency (blocks, concurrent work, feature batches).
A Zarr read holds more than its result. A read of a sharded array, such as `counts` or `countsT`, fills a shard-level copy of its selection, holds the compressed bytes of every inner chunk it touches in a shard, and decodes those chunks next to them. Plans charge this through `ArrayGeometry.readBytes`, at the decoded size of the compressed bytes because a plan cannot know them before it reads: a `counts` row block of a whole shard reserves twice its bytes, the shard's chunks, and the one chunk that row-block streams decode at a time, and a `countsT` cell band reserves twice its bytes and two of each chunk it touches. An unsharded read, such as one of normalized data, is charged its result and the chunks it decodes; the compressed bytes of a chunk it decodes and a decoded chunk that a codec thread still holds after a read stay outside the plan.
Environment variables used by the documentation executor (`SCARF_MEM_BUDGET`, `SCARF_WORKERS`, …) are also useful for local large runs.

## Related guides

- {doc}`../tutorials/data_organization`
- {doc}`../concepts/memory_and_execution`
- {doc}`contributing`
