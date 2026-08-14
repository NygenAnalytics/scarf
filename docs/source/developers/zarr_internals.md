(zarr_internals)=
# Zarr internals

This page is for contributors and advanced users who need the on-disk layout.
Analysts should start with {doc}`../tutorials/data_organization`.

## Layout overview

A Scarf Zarr store is a directory hierarchy.
Typical top-level groups include:

- cell metadata
- one or more assay groups (for example `RNA`, `ADT`, `ATAC`)
- per-assay feature metadata
- normalized matrices, reductions, and graph artefacts under assay-specific paths

Exact group names depend on the assay and the graph-stage parameters.
Prefer inspecting a store with `zarr.open` or Scarf's `DataStore` summary rather than hard-coding internal paths in analysis scripts.

## RNA `countsT` (mandatory, strip-sharded)

RNA assays (`RNAassay` and aliases such as `GeneActivity` / `URNA`) **require** a complete
feature-major `countsT` array with the strip-sharded Zarr v3 layout:

- Written automatically after `counts` on RNA ingest, subset dump, merge, and `repack_zarr`
- Whole gene strips (~1 GiB target) with ~128 MiB chunks; unsharded `countsT` is invalid
- `RNAassay` load fails closed if `countsT` is missing, incomplete, unsharded, or the store is
  Zarr v2 (strip shards need v3)
- Non-RNA assays (ATAC, ADT, …) do **not** use `countsT`

After rewriting an older store to strip `countsT` / v3, recompute HVG, normalization, PCA, graph,
and marker artefacts. Do not resume them from pre-rewrite lineage.

Repair or migrate with `repack_zarr` or `write_counts_t`. Inspect and repack tooling can open a
store without constructing a failing `RNAassay`.

## Zarr versions

New stores default to Zarr v3.
Assay count matrices are sharded; normalized arrays are unsharded (`shards=None`).

RNA assays require Zarr v3 because strip-sharded `countsT` is a v3 feature.
Non-RNA assays in a multi-assay store remain usable when tooling opens without forcing every
assay class to construct.

To convert an older store:

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

## Memory controls

`DataStore(..., mem_budget='8G')` bounds streaming and concurrency (blocks/bands, concurrent work, feature batches).
Environment variables used by the documentation executor (`SCARF_MEM_BUDGET`, `SCARF_WORKERS`, …) are also useful for local large runs.

## Related guides

- {doc}`../tutorials/data_organization`
- {doc}`contributing`
