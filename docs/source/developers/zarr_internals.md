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

## Zarr versions

Existing Zarr v2 stores can still be opened.
New stores default to Zarr v3.
Assay count matrices are sharded; normalized arrays are unsharded (`shards=None`).

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
