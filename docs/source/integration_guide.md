(integration_guide)=
# Integration methods guide

Scarf offers several integration and batch-correction approaches. This page helps you choose the right one. For worked examples, follow the linked vignettes.

## Decision guide

| Goal | Recommended approach | Vignette |
|------|---------------------|----------|
| Merge two scRNA-seq batches in one object | `AssayMerge` then Harmony or partial PCA | {ref}`merging vignette <harmony_batch_correction>` |
| Correct batch effects after merge | `make_graph(harmonize=True, batch_columns=[...])` | {ref}`Harmony batch correction <harmony_batch_correction>` |
| Lightweight batch correction with a reference sample | `make_graph(pca_cell_key='reference_subset')` | {ref}`merging vignette <harmony_batch_correction>` |
| Integrate RNA + ADT (CITE-seq) in the same cells | `integrate_assays(method='snn')` or `method='wnn'` | {ref}`WNN integration <wnn_integration>` |
| Map query cells onto a reference atlas | `run_mapping` with optional CORAL | {ref}`data projection <data_projection>` |
| Co-embed reference and query on one layout | `run_unified_umap` or `run_unified_tsne` | {ref}`data projection <data_projection>` |
| Measure integration quality | `metric_lisi`, `metric_batch_mixing`, `metric_silhouette` | {ref}`LISI metrics <lisi_metrics>` |

## Harmony vs partial PCA

Both operate on a merged dataset in a single `DataStore`.

**Harmony** (`make_graph(harmonize=True, batch_columns=['sample_id'])`) runs iterative correction on the PCA embedding before KNN construction. Use when multiple batches need to mix while preserving biology, especially when no single sample should define the embedding.

**Partial PCA** (`make_graph(pca_cell_key='is_ctrl')`) trains PCA on a subset of cells (for example one control sample). Use when one batch is a trusted reference and you want a lightweight correction that down-weights batch-specific variance.

After either method, quantify results with {ref}`LISI metrics <lisi_metrics>`.

## Measuring integration quality

After correction, quantify the result rather than relying on the UMAP alone. Scarf exposes four metrics with different purposes:

- **`metric_lisi`** returns per-cell LISI for any label. Run it on the batch column to check neighborhood mixing and on the cell-type column to check that biology is preserved. Good integration raises batch LISI while keeping cell-type LISI low.
- **`metric_batch_mixing`** condenses batch LISI into a single value in `[0, 1]` by rescaling the mean against the mixing that perfectly integrated data would reach for the given batch sizes. Use it to compare graphs and datasets on a common scale. Higher is better.
- **`metric_silhouette`** scores how separated each cluster is from its nearest neighbor cluster, from -1 to 1. Values near 1 mean distinct clusters. Read it together with the batch metrics, since over-correction can mix genuinely distinct cell types.
- **`metric_label_concordance`** compares two labelings with ARI or NMI, for example predicted clusters against imported annotations. It measures label agreement, not batch mixing.

A useful pattern is to compute these metrics on the naive, partial PCA, and Harmony graphs, then compare. Better integration shows higher batch LISI and batch-mixing scores without collapsing cell-type separation. See the worked example in the {ref}`merging vignette <lisi_metrics>`.

## SNN vs WNN (multimodal)

Both merge per-modality KNN graphs from the same cells:

- **SNN** (default): shared-nearest-neighbor graph merge. Supports two or more assays.
- **WNN**: weighted nearest neighbors (Hao et al., Cell 2022). **Exactly two assays only.** Often helps when modalities differ in sparsity or signal strength.

See {ref}`WNN integration <wnn_integration>`.

## Projection vs merge

**Merge** combines raw counts from multiple datasets into one Zarr store. Use when you will analyze all cells together.

**Projection** (`run_mapping`) maps external cells onto an existing reference graph without merging stores. Use for label transfer, atlas mapping, or when query cells should not alter the reference embedding. CORAL can align feature distributions between reference and query.

## Not supported

Scarf does not include Scanorama, BBKNN, scVI, ComBat, or other external integration packages. Export subsets with `to_anndata` or `SubsetZarr` if you need those tools.

## Related pages

- {ref}`Quick start <quickstart>`
- {ref}`Data organization in Scarf <data_organization>`
- {doc}`api`
