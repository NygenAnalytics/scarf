---
description: End-to-end scRNA-seq analysis in Scarf from counts through clustering and marker genes.
jupytext:
  cell_metadata_filter: -all
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

(scrna_seq_workflow)=

# scRNA-seq analysis

This chapter runs a full scRNA-seq workflow on a 5K PBMC dataset: import, quality control,
highly variable genes, neighbourhood graph, UMAP, clustering, and marker genes. For a
minimal pipeline see {ref}`Quick start <quickstart>`. For Scanpy equivalents see
{doc}`../scarf_and_scanpy`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies ({ref}`installation <installation>`)
- Basic familiarity with count matrices

## What you will learn

- Convert Cell Ranger H5 to Zarr and open a `DataStore`
- Filter cells without deleting them from the store
- Select highly variable genes and build a neighbourhood graph with atomic ops
- Run UMAP and Leiden clustering on that graph
- Rank marker genes per cluster and plot them
- Optionally impute sparse features with graph diffusion


## Dataset

`tenx_5K_pbmc_rnaseq` is a public 10x Genomics PBMC dataset distributed through the
`scarf_docs` Cytebase repository.

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.configure_output(level='WARNING', progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
)
```

## 1) Import

Read the Cell Ranger H5 file, inspect cell and feature counts, then write a Zarr store.

```{note}
A Zarr "file" is a directory hierarchy on disk, not a single HDF5-style file.
```

```{code-cell} ipython3
reader = scarf.CrH5Reader(f'{dataset}/data.h5')
reader.nCells, reader.nFeatures
```

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc=f'{dataset}/data.zarr',
)
writer.dump(batch_size=1000)
```

Open a `DataStore`. On first open Scarf streams the count matrix once to compute
initialization statistics: per-cell QC columns such as `RNA_nCounts` and
`RNA_nFeatures`, mito/ribo fractions when gene-name patterns match, and the
per-feature cell counts used for feature filtering.
`min_features_per_cell` marks cells inactive when they have fewer non-zero features
than the threshold. Feature filtering uses `min_cells_per_feature` (default 20).

Opening this store prints a message that the smallest cell count is below the RNA size
factor of 1000. It refers to normalization, which is covered in step 3, and the QC filter
in step 2 removes those cells.

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    min_features_per_cell=10
)
ds
```

## 2) Quality control

Inspect QC distributions, then set dataset-specific thresholds. Thresholds below are chosen
for this PBMC dataset; other datasets need their own cutoffs. Deeper QC options, including
`auto_filter_cells` and doublet detection, are in {doc}`quality_control`.

```{code-cell} ipython3
qc_cols = [
    c for c in (
        'RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito', 'RNA_percentRibo'
    )
    if c in ds.cells.columns
]
ds.plots.distribution(
    keys=qc_cols,
    kind='violin',
    max_points=2000,
)
```

Each violin is one QC metric before filtering; set thresholds from the tails of these
distributions.

```{code-cell} ipython3
n_before = int(ds.cells.fetch_all('I').sum())
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0]
)
n_after = int(ds.cells.fetch_all('I').sum())
print(f'Active cells before filter: {n_before}')
print(f'Active cells after filter: {n_after}')
```

```{note}
Filtered cells are marked inactive in the boolean cell key `I`, not deleted. Most
`DataStore` methods default to `cell_key='I'`. See {doc}`data_organization`.
```

```{code-cell} ipython3
ds.plots.distribution(
    keys=qc_cols,
    kind='violin',
    max_points=2000,
    color='coral',
)
```

After filtering, the same metrics are restricted to active cells (`I=True`).

## 3) Feature selection

Library-size normalization for RNA assays uses a scalar factor (`ds.RNA.sf`, default 1000).
Because cells with `RNA_nCounts` below 1000 were filtered out, that default is safe here.

```{code-cell} ipython3
ds.RNA.sf
```

`mark_hvgs` ranks genes by corrected variance and marks highly variable genes. The feature
column is named with the cell key prefix (here `I__hvgs`). Pass `feat_key='hvgs'` later;
Scarf resolves the prefix.

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=20,
    top_n=500,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=True,
)
print('Selected genes:', int(ds.RNA.feats.fetch_all('I__hvgs').sum()))
ds.RNA.feats.to_pandas_dataframe(
    ['names', 'nCells', 'I__hvgs']
).head()
```

## 4) Neighbourhood graph

Cells are linked into a k-nearest-neighbour graph in five steps: normalize the selected
genes, reduce them with PCA, index the reduced coordinates, query each cell's neighbours,
and turn those neighbours into a weighted graph. Every step reads the previous result from
the store, so run them in this order.

Important parameters:

- `feat_key`: feature column to use (`hvgs` here)
- `dims`: PCA dimensions
- `k`: neighbours per cell

```{code-cell} ipython3
ds.run_normalization(feat_key='hvgs')
ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()

ds.load_graph()
```

`load_graph` returns the result as a sparse cell-by-cell matrix, which is a quick way to
confirm the graph covers the active cells.

```{seealso}
Each step also returns a reference to the artifact it wrote. Capturing those references
lets you branch the chain, for example to compare two values of `k` or to insert Harmony
batch correction between PCA and the neighbour index. That style is covered in
{doc}`atomic_graph_operations`. To run the whole recipe in one call, see
{ref}`Quick start <quickstart>`.
```


## 5) Dimensionality reduction

Run UMAP on the latest graph. Results are stored as `RNA_UMAP1` and `RNA_UMAP2`.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
ds.cells.to_pandas_dataframe(
    columns=['RNA_UMAP1', 'RNA_UMAP2'],
    key='I'
).head()
```

```{code-cell} ipython3
ds.plots.embedding(layout_key='RNA_UMAP')
```

Cells are placed by neighbourhood-graph proximity on the UMAP.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_nCounts',
    color_scale=splt.ColorScale(cmap='coolwarm'),
)
```

Library size varies across the embedding; check whether high-count cells dominate one region.

Alternatives (densMAP, tSNE, Paris trees) are covered in
{doc}`dimensionality_reduction_and_clustering`.

## 6) Clustering

Leiden clustering runs on the same neighbourhood graph. Cluster labels are saved as
`RNA_leiden_cluster`.

```{code-cell} ipython3
ds.run_leiden_clustering(resolution=0.5)
ds.cells.to_pandas_dataframe(
    columns=['RNA_leiden_cluster'],
    key='I'
)['RNA_leiden_cluster'].value_counts().sort_index()
```

Cluster sizes are worth a look before plotting: a resolution that is too high splits one
cell type into several small clusters.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

Each colour is a Leiden partition on the same UMAP coordinates.

## 7) Marker genes

`run_marker_search` ranks genes per group. Results include specificity-oriented scores,
Mann-Whitney U test p-values (`p_value`), AUC effect sizes, and within-group
Benjamini-Hochberg adjusted values (`p_value_adjusted`). The adjusted column is a cell-level
marker correction for that group, not replicate-aware differential expression. For
condition-level DE with full workflows, export counts (see
{doc}`pseudobulk_and_differential_expression`) and use an external tool.

```{code-cell} ipython3
ds.run_marker_search(group_key='RNA_leiden_cluster')
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key='RNA_leiden_cluster',
    topn=5,
    figsize=(5, 9)
)
```

Rows are top marker genes per cluster; stronger scores mark more cluster-specific genes.

```{code-cell} ipython3
df = ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id='1',
    min_score=-1,
    min_frac_exp=-1
)
df.head()
```

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

Annotation from markers, known gene panels, and subclustering is covered in
{doc}`annotation`.

(imputation)=

## 8) Feature imputation

Scarf can impute feature values by diffusing expression along the KNN graph
(MAGIC-style). Use `get_imputed` after the neighbourhood graph exists.


```{code-cell} ipython3
imputed_cd4 = ds.get_imputed(feature_name='CD4', t=2)
ds.cells.insert('CD4_imputed', imputed_cd4, overwrite=True)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['CD4', 'CD4_imputed'],
    n_columns=2,
    sort_values=True,
)
```

The `t` parameter controls diffusion depth. Higher values smooth more. The diffusion
operator is cached under the graph location in the Zarr store.

Raw CD4 is sparse; the imputed panel should keep the same high-expression neighborhoods
while filling gaps inside them. Do not use imputed values as input counts for differential
expression or as evidence that a gene was detected in a cell.

## Common mistakes and limitations

- Reusing QC thresholds from another dataset without inspecting distributions
- Calling `run_umap` or clustering before building the neighbourhood graph
- Treating marker `p_value` or within-group `p_value_adjusted` columns as replicate-aware DE results
- Expecting filtered cells to disappear from `ds.cells` (they remain, with `I=False`)
- Treating imputed expression as a replacement for observed counts


## Summary of saved results

| Kind | Keys / location |
|---|---|
| QC columns | `RNA_nCounts`, `RNA_nFeatures`, `RNA_percentMito`, … |
| Active cells | cell key `I` |
| HVGs | feature column `I__hvgs` (pass as `hvgs`) |
| Embedding | `RNA_UMAP1`, `RNA_UMAP2` |
| Clusters | `RNA_leiden_cluster` |
| Markers | marker tables from `run_marker_search` / `get_markers` |
| Imputed values | cell columns such as `CD4_imputed` after `insert` |

## Further reading

- [Single-cell best practices: quality control](https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html)
- [Single-cell best practices: clustering](https://www.sc-best-practices.org/cellular_structure/clustering.html)
- [Scanpy clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)
- van Dijk et al. 2018, MAGIC (algorithmic ancestry for graph diffusion imputation; not feature parity with Scarf): https://doi.org/10.1016/j.cell.2018.05.061
- Scarf paper: https://doi.org/10.1038/s41467-022-32097-3

## Next steps

- {doc}`quality_control`
- {doc}`annotation`
- {doc}`gene_set_enrichment`
- {doc}`plotting`
- {doc}`data_integration`

