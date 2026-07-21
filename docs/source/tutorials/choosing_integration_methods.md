(integration_guide)=
# Choosing integration methods

Scarf includes several integration and batch-correction approaches. This page maps goals to
APIs. For worked examples, follow the linked tutorials.

## Prerequisites

- Know whether the data are separate batches or multiple assays from the same cells
- Identify whether a fixed reference atlas is available

## What you will learn

- Select a correction, multimodal integration, or mapping approach
- Recognize the constraints of SNN and WNN integration
- Evaluate integration beyond visual mixing in UMAP

## Decision guide

| Goal | Recommended approach | Tutorial |
|------|---------------------|----------|
| Merge two scRNA-seq batches in one object | `AssayMerge` then Harmony or partial PCA | {ref}`merging tutorial <harmony_batch_correction>` |
| Correct batch effects after merge | `make_graph(harmonize=True, batch_columns=[...])` | {ref}`Harmony batch correction <harmony_batch_correction>` |
| Lightweight batch correction with a reference sample | `make_graph(pca_cell_key='reference_subset')` | {ref}`merging tutorial <harmony_batch_correction>` |
| Integrate RNA + ADT (CITE-seq) in the same cells | `integrate_assays(assays=[...], label='...', method='snn' or 'wnn')` | {ref}`WNN integration <wnn_integration>` |
| Map query cells onto a reusable harmonized atlas | `build_mapping_reference` then `MappingReference.map_query` | {ref}`reference atlas mapping <reference_atlas_mapping>` |
| Map query cells onto a fixed PCA reference | `run_mapping` | {ref}`data projection <data_projection>` |
| Co-embed reference and query for exploration | `run_unified_umap` or `run_unified_tsne` | {ref}`data projection <data_projection>` |
| Measure integration quality | `metric_ilisi`, `metric_clisi`, `metric_graph_connectivity` | {ref}`LISI metrics <lisi_metrics>` |

## Harmony vs partial PCA

Both operate on a merged dataset in a single `DataStore`.

**Harmony** (`make_graph(harmonize=True, batch_columns=['sample_id'])`) runs iterative correction on the PCA embedding before KNN construction. Use when multiple batches need to mix while preserving biology, especially when no single sample should define the embedding.

**Partial PCA** (`make_graph(pca_cell_key='is_ctrl')`) trains PCA on a subset of cells (for example one control sample). Use when one batch is a trusted reference and you want a lightweight correction that down-weights batch-specific variance.

After either method, quantify results with {ref}`LISI metrics <lisi_metrics>`.

## Measuring integration quality

After correction, quantify the result rather than relying on the UMAP alone:

- **`metric_lisi`** returns per-cell LISI for any label. Run it on the batch column to check neighborhood mixing and on the cell-type column to check that biology is preserved.
- **`metric_ilisi`** summarizes batch mixing with scIB median scaling.
- **`metric_clisi`** summarizes biological-label conservation with scIB scaling.
- **`metric_proportional_batch_mixing`** uses mean LISI and adjusts for observed batch sizes.
- **`metric_graph_connectivity`** measures whether cells with each biological label remain connected.
- **`metric_graph_silhouette`** scores how separated each cluster is from its nearest neighbor cluster.
- **`metric_label_concordance`** compares two labelings with ARI or NMI. It measures label agreement, not batch mixing.

A useful pattern is to compute these metrics on the naive, partial PCA, and Harmony graphs, then compare. See {doc}`integration_metrics`.

## SNN vs WNN (multimodal)

Both merge per-modality KNN graphs from the same cells:

- **SNN** (default): shared-nearest-neighbor graph merge. Supports two or more assays.
- **WNN**: weighted nearest neighbors (Hao et al., Cell 2021). Exactly two assays only.

See {ref}`WNN integration <wnn_integration>`.

## Projection vs merge

**Merge** combines raw counts from multiple datasets into one Zarr store. Use when you will analyze all cells together.

**Projection** (`run_mapping`) maps external cells onto an existing PCA reference graph without merging stores. Use it for one-off label transfer when the reference does not need batch correction.

**Reference atlas mapping** (`build_mapping_reference` then `MappingReference.map_query`) stores a content-addressed RNA/PCA reference with Harmony state for Symphony-style fixed-reference correction. Use it for repeated mapping into a harmonized atlas.

CORAL (`run_mapping(run_coral=True)`) is deprecated in favor of Symphony-style mapping references.

Developer migration notes for older mapping calls live in {doc}`../developers/migration_notes`.

## Not supported

Scarf does not include Scanorama, BBKNN, scVI, ComBat, or other external integration packages. Export subsets with `to_anndata` or `SubsetZarr` if you need those tools.

## Further reading

- Korsunsky et al. 2019, Harmony: https://doi.org/10.1038/s41592-019-0619-0
- Hao et al. 2021, weighted nearest neighbor analysis: https://doi.org/10.1016/j.cell.2021.04.048
- [Seurat WNN vignette](https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis)
- Kang et al. 2021, Symphony: https://doi.org/10.1038/s41467-021-25957-x

## Related pages

- {ref}`Quick start <quickstart>`
- {doc}`data_organization`
- {doc}`../reference/api`
