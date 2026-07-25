---
description: Marker tables, known markers, cell labels, and subclustering with cell keys in Scarf.
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(annotation)=

# Annotation

This chapter starts from a clustered PBMC store and shows how to read marker tables,
plot known markers, assign labels, and recluster a subset with a custom cell key.

## Prerequisites

- {doc}`scrna_seq` (or an equivalent clustered Zarr store)
- Familiarity with cell keys (`I` and custom boolean columns)

## What you will learn

- Retrieve markers with `get_markers`
- Color embeddings by gene expression
- Write annotation columns into cell metadata
- Subset cells with a custom `cell_key` and recluster

## Dataset

Use the same 5K PBMC Zarr store. Rebuild a short analysis path so this page runs alone.

```{code-cell} ipython3
import numpy as np
import scarf

scarf.set_verbosity('WARNING')

scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
normalized = ds.run_normalization(feat_key='hvgs')
pca = ds.run_pca(normalized, dims=15)
ds.build_embedding_initialization(pca)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
ds.build_connectivity_map(neighbors)
ds.run_umap(n_epochs=150, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)
ds.run_marker_search(group_key='RNA_leiden_cluster', gene_batch_size=100)
```


## 1) Marker tables

`get_markers` returns genes ranked by marker score. Pass a `group_id` for one cluster, or
`group_id=None` for every cluster in one long table with a `group_id` column. Columns include
scores, expression fractions, fold change, and Mann-Whitney `p_value`. These are not
FDR-corrected DE results.

```{code-cell} ipython3
markers = ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id='1',
    min_score=-1,
    min_frac_exp=-1,
)
markers.head(10)
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key='RNA_leiden_cluster',
    topn=5,
    figsize=(5, 9),
)
```

Rows are top markers per cluster; use them with known lineage genes, not as FDR DE.

## 2) Known markers on the embedding

Confirm panel genes on the UMAP before assigning labels.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['CD14', 'MS4A1', 'CD3D'],
    n_columns=3,
    sort_values=True,
)
```

CD14, MS4A1, and CD3D mark monocyte-, B-, and T-cell-like regions when those lineages
are present.

## 3) Assign labels

Map Leiden clusters to names using the marker UMAPs and marker tables. Cluster IDs are not
stable across parameter changes, so this cell picks the cluster where each lineage gene
ranks highest among markers, then leaves other clusters as `Cluster_<id>`.

```{code-cell} ipython3
cluster_ids = ds.cells.fetch_all('RNA_leiden_cluster')
unique = sorted(
    {str(c) for c in cluster_ids},
    key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
)

markers = ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id=None,
    min_score=-1,
    min_frac_exp=-1,
)

label_map = {c: f'Cluster {c}' for c in unique}
for gene, name in [('CD14', 'Monocytes'), ('MS4A1', 'B cells'), ('CD3D', 'T cells')]:
    hit = markers[markers['feature_name'].astype(str) == gene]
    if hit.empty:
        continue
    cid = str(hit.sort_values('score', ascending=False).iloc[0]['group_id'])
    label_map[cid] = name

labels = np.array([label_map[str(c)] for c in cluster_ids], dtype=object)
ds.cells.insert(column_name='cell_type', values=labels, overwrite=True)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='cell_type',
)
```

Assigned labels replace numeric cluster IDs where the panel genes ranked highest.

## 4) Relabel clusters with `smart_label`

`smart_label` renames values in one categorical column from the most frequent overlap with
another column. Here Leiden cluster IDs are rewritten from the `cell_type` labels just
assigned. Shared base labels get letter suffixes when more than one Leiden cluster maps to
the same type.

```{code-cell} ipython3
ds.smart_label(
    to_relabel='RNA_leiden_cluster',
    base_label='cell_type',
    new_col_name='leiden_by_type',
)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='leiden_by_type',
)
```

`leiden_by_type` is a convenience labeling of clusters, not an automated ontology annotation.

## 5) Subset and recluster

Create a boolean cell key for one population, then recompute the graph with that key.

```{code-cell} ipython3
clusters = ds.cells.to_pandas_dataframe(
    columns=['RNA_leiden_cluster'],
    key='I',
)['RNA_leiden_cluster'].astype(str)
focus = str(clusters.value_counts().index[0])
subset = np.array([str(c) == focus for c in ds.cells.fetch_all('RNA_leiden_cluster')])
active = ds.cells.fetch_all('I').astype(bool)
ds.cells.insert('focus_cells', active & subset, overwrite=True)
# Feature keys are resolved as <cell_key>__<feat_key>. Recompute HVGs for the subset.
ds.mark_hvgs(
    cell_key='focus_cells',
    min_cells=10,
    top_n=500,
    show_plot=False,
)

normalized = ds.run_normalization(cell_key='focus_cells', feat_key='hvgs')
pca = ds.run_pca(normalized, dims=15)
ds.build_embedding_initialization(pca)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
ds.build_connectivity_map(neighbors)
ds.run_umap(
    cell_key='focus_cells',
    n_epochs=100,
    spread=5,
    min_dist=1,
    parallel=True,
    label='UMAP',
)
ds.run_leiden_clustering(
    cell_key='focus_cells',
    resolution=0.4,
    label='leiden_cluster',
)
# With cell_key != 'I', columns are RNA_<cell_key>_<label>
ds.plots.embedding(
    layout_key='RNA_focus_cells_UMAP',
    color_by='RNA_focus_cells_leiden_cluster',
    cell_key='focus_cells',
)
```

The subset UMAP and Leiden labels apply only to cells marked in `focus_cells`.

## Common mistakes and limitations

- Treating cluster IDs as biologically stable across resolutions
- Overwriting annotation columns without keeping the clustering key you used
- Claiming FDR-corrected DE from `run_marker_search` alone

## Summary of saved results

| Kind | Keys |
|---|---|
| Markers | from `run_marker_search` / `get_markers` |
| Labels | user columns such as `cell_type`, `leiden_by_type` |
| Subset key | e.g. `focus_cells` |
| Subcluster results | e.g. `RNA_focus_cells_UMAP*`, `RNA_focus_cells_leiden_cluster` |

## Further reading

- [Single-cell best practices: cell-type annotation](https://www.sc-best-practices.org/cellular_structure/annotation.html)
- Scarf does not ship automated ontology annotators; labels here are assigned from markers and known genes

## Next steps

- {doc}`gene_set_enrichment`
- {doc}`pseudobulk_and_differential_expression`
- {doc}`mapping_and_label_transfer`
- {doc}`plotting`
