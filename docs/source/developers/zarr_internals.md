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

## Opening an RNA assay

`RNAassay` construction requires a complete `countsT` on Zarr v3 plus matching layout metadata.
The open fails if `countsT` is missing, incomplete, unsharded, a Zarr v2 array, or out of agreement with `counts`.

Inspect and `repack_zarr` can open the store without constructing `RNAassay`.
The user-facing repair is to re-import the source or run:

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

After a rewrite, recompute HVG, normalization, PCA, graph, and marker artefacts.
Do not resume them from pre-rewrite lineage.

Non-RNA assays in a multi-assay store remain usable when tooling opens without forcing every assay class to construct.

## Zarr versions

New stores default to Zarr v3.
RNA assays require v3 because sharded `countsT` is a v3 feature.
A Zarr v2 RNA store will not open as `RNAassay`.

## Memory controls

`DataStore(..., mem_budget='8G')` bounds streaming and concurrency (blocks, concurrent work, feature batches).
Environment variables used by the documentation executor (`SCARF_MEM_BUDGET`, `SCARF_WORKERS`, …) are also useful for local large runs.

## Related guides

- {doc}`../tutorials/data_organization`
- {doc}`../concepts/memory_and_execution`
- {doc}`contributing`
