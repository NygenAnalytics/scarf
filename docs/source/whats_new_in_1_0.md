(whats_new_in_1_0)=
# What is new in Scarf 1.0

User-facing changes when moving from Scarf 0.x to 1.0. Internal import-path
moves stay in {doc}`developers/migration_notes`.

## Installation

- Requires **Python 3.12+**
- Install analysis and plotting extras: `uv pip install "scarf[extra]"` (or pip)
- New stores use **Zarr v3** with sharding; existing Zarr v2 stores remain readable

## Dataset fetching

`scarf.fetch_dataset` and `scarf.show_available_datasets` are removed. Use
Cytebase:

```python
# Before
# scarf.fetch_dataset("tenx_5K_pbmc_rnaseq", save_path="scarf_datasets")

# After
datasets = scarf.cytebase.connect("scarf_docs")
datasets.list_datasets()
datasets.download_dataset("tenx_5K_pbmc_rnaseq", destination="scarf_datasets")
# Optional prepared Zarr: download_dataset(..., zarr=True)
```

## Graph construction

`make_graph` has been removed. Use the standard recipe or atomic steps:

```python
# Before
# ds.mark_hvgs(...)
# ds.make_graph(feat_key="hvgs", k=11, dims=15)

# After: standard recipe
artifacts = ds.pipeline.run(
    filtering={"method": "manual", "attrs": [...], "highs": [...], "lows": [...]},
    cell_cycle_scoring=False,
    highly_variable_features={"top_n": 500},
    pca={"dims": 15},
    neighbors={"k": 11},
    paris=False,
    doublet_scoring=False,
    markers=False,
)

# After: atomic chain
ds.run_normalization(feat_key="hvgs")
ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()
```

Datastores containing graphs written by earlier releases remain readable.

See {doc}`concepts/graph_and_state` and
{doc}`tutorials/atomic_graph_operations`.

## Provenance-backed artifacts

Analysis outputs are stored as artifacts with provenance, reuse, and
`invalidate_cache`. Inspect with `list_artifacts`, `inspect_artifact`, and
`load_artifact`. Metadata columns may carry `source_artifact` links.

See {doc}`concepts/provenance` and {doc}`tutorials/provenance_and_reuse`.

## Paris clustering

- Call `run_paris_clustering` directly (`run_clustering` forwarding is gone)
- Default `n_clusters="auto"` uses a branch-adaptive cut with a modularity guard
- Labels can differ from earlier Scarf releases even at the same integer cut:
  the hierarchy now fits the additive graph `A + A.T`, and cut behavior changed
- Balanced-cut mode on `DataStore` was removed
- `scikit-network` is no longer a runtime dependency

## Metric renames

| Removed name | Use instead |
|---|---|
| `metric_integration` | `metric_label_concordance` (ARI / NMI) |
| `metric_batch_mixing` | `metric_proportional_batch_mixing` or `metric_ilisi` |
| `metric_silhouette` | `metric_graph_silhouette` |
| `scarf.metrics.integration_score` | removed |

`metric_proportional_batch_mixing` and `metric_ilisi` answer different batch
questions; they are not aliases.

## Mapping

Compatibility kwargs remain through 1.x with warnings (`ref_mu` / `ref_sigma`,
`exclude_missing`, `run_coral`). Prefer current arguments and rebuild mapping
references when you need full provenance. Details:
{doc}`developers/migration_notes`.

## Plotting and merge

- Plots: `ds.plots.*` or `import scarf.plotting as splt` (old `scarf.plots` /
  `plot_*` methods are gone)
- Merge: construct `AssayMerge` / `DatasetMerge` (`ZarrMerge` alias removed)

## New capabilities to know about

- `ds.pipeline.run` for `basic_rna_analysis`
- Atomic graph API and `AssayState`
- Direct object-store `DataStore` with `storage_options` and `zarrProfile`
- Explicit `mem_budget` and `nthreads` resource controls
- `python -m scarf.tools.repack_zarr` for layout migration

## Next steps

- {ref}`Quick start <quickstart>`
- {doc}`concepts/scale_and_memory`
- {doc}`scarf_and_scanpy`
- {doc}`developers/migration_notes` (internal paths)
