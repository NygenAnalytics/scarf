(migration_notes)=
# Migration notes

Notes for developers updating older Scarf workflows or documentation.

## Plotting

Prefer `scarf.plotting` (`embedding`, `marker_heatmap`, `cluster_tree`, …).
`DataStore.plot_*` and `scarf.plots` remain available compatibility helpers.

## Merge APIs

Prefer `AssayMerge` / `DatasetMerge`. `ZarrMerge` remains documented as a compatibility
class where still present.

## Cell keys

Filtering still marks cells inactive via boolean cell keys (default `I`) rather than
deleting rows. Custom keys remain the supported way to subset for reclustering and mapping.

## Mapping calls

Existing mapping calls remain accepted for one deprecation cycle:

- Projection groups written before provenance schemas remain readable when their neighbor arrays are structurally valid. Scarf emits `DeprecationWarning`; rerun `run_mapping` to write full provenance.
- A writable legacy Harmony graph without a mapping artifact is rebuilt automatically the first time `run_mapping` needs it. For a read-only store, reopen it with `zarr_mode='r+'` and call `build_mapping_reference(..., batch_columns=[...])` once.
- `ref_mu=False` and `ref_sigma=False` no longer select query-derived statistics. They emit `DeprecationWarning` and use reference statistics. Remove these arguments.
- `exclude_missing=True` remains an alias for `missing_feature_policy='intersection'`.
- `run_coral=True` remains available with a deprecation warning.

Recomputed results can differ from earlier Scarf releases. Recalibrate downstream thresholds after rebuilding.

## Documentation execution

Executable pages live under `docs/source/` (especially `tutorials/`). Refresh the myst-nb
cache with:

```bash
cd docs && make execute-page PAGE=scrna_seq
# or
cd docs && make execute-docs JOBS=1
```

Timeout per page is 600 seconds (`nb_execution_timeout` in `conf.py`).
