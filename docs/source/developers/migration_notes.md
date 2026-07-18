(migration_notes)=
# Migration notes

Notes for developers updating older Scarf workflows or documentation.

## Plotting

The plotting API is a clean break from earlier releases. Older plotting entry points have
been removed, so existing plotting calls must be rewritten. Import `scarf.plotting` as `splt`
and call its functions directly:

`scarf.plots` and the `DataStore.plot_*` plotting methods no longer exist. Use
`splt.embedding`, `splt.distribution`, `splt.marker_heatmap`, `splt.cluster_tree`,
`splt.pseudotime_heatmap`, and `splt.unified_embedding` instead.

```python
import scarf.plotting as splt

splt.embedding(ds, layout_key="RNA_UMAP", color_by="clusters")
```

Plot functions render by default with `show=True`. Pass `show=False` when the returned
`PlotResult` must remain available for figure access or saving.

`mark_hvgs(..., show_plot=True, **plot_kwargs)` now forwards keyword arguments to
`splt.highly_variable_features`. Rename older mean-variance plot kwargs as follows:
`ax_label_fs` to `label_size`, `fig_size` to `figsize`, `ss` to `point_sizes`, and
`cmaps` to `colormaps`.

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
