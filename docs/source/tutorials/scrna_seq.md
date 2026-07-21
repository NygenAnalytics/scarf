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
- Select highly variable genes and build a neighbourhood graph with `make_graph`
- Run UMAP and Leiden clustering on that graph
- Rank marker genes per cluster and plot them

## Dataset

`tenx_5K_pbmc_rnaseq` is a public 10x Genomics PBMC dataset distributed through Scarf's
`fetch_dataset` catalog.

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

scarf.fetch_dataset(
    'tenx_5K_pbmc_rnaseq',
    save_path='scarf_datasets'
)
```

## 1) Import

Read the Cell Ranger H5 file, inspect cell and feature counts, then write a Zarr store.

```{note}
A Zarr "file" is a directory hierarchy on disk, not a single HDF5-style file.
```

```{code-cell} ipython3
reader = scarf.CrH5Reader('scarf_datasets/tenx_5K_pbmc_rnaseq/data.h5')
reader.nCells, reader.nFeatures
```

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc='scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    chunk_size=(2000, 1000)
)
writer.dump(batch_size=1000)
```

Open a `DataStore`. On first open Scarf computes per-cell QC columns such as `RNA_nCounts`
and `RNA_nFeatures`, and mito/ribo fractions when gene-name patterns match.
`min_features_per_cell` marks cells inactive when they have fewer non-zero features
than the threshold. Feature filtering uses `min_cells_per_feature` (default 20).

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
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

```{code-cell} ipython3
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0]
)
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
ds.cells.head()
```

## 3) Feature selection

Library-size normalization for RNA assays uses a scalar factor (`ds.RNA.sf`, default 1000).
Because we filtered cells with `RNA_nCounts` at least 1000, that default is safe here.

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
    max_var=6
)
ds.RNA.feats.head()
```

## 4) Neighbourhood graph

`make_graph` is the central step. It normalizes selected features, runs PCA, builds an ANN
index, queries K nearest neighbours, computes edge weights, and fits MiniBatch KMeans
centroids used for embedding initialization.

Important parameters:

- `feat_key`: feature column to use (`hvgs` here)
- `k`: neighbours per cell
- `dims`: PCA dimensions
- `n_centroids`: KMeans centroids

```{code-cell} ipython3
ds.make_graph(
    feat_key='hvgs',
    k=11,
    dims=15,
    n_centroids=100
)
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
ds.cells.head()
```

```{code-cell} ipython3
ds.plots.embedding(layout_key='RNA_UMAP')
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_nCounts',
    color_scale=splt.ColorScale(cmap='coolwarm'),
)
```

Alternatives (densMAP, tSNE, Paris trees) are covered in
{doc}`dimensionality_reduction_and_clustering`.

## 6) Clustering

Leiden clustering runs on the same neighbourhood graph. Cluster labels are saved as
`RNA_leiden_cluster`.

```{code-cell} ipython3
ds.run_leiden_clustering(resolution=0.5)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

```{code-cell} ipython3
leiden_clusters = ds.cells.to_pandas_dataframe(
    columns=['RNA_leiden_cluster'],
    key='I'
)
leiden_clusters.head()
```

## 7) Marker genes

`run_marker_search` ranks genes per group. Results include specificity-oriented scores and
Mann-Whitney U statistics (`p_value`). Scarf does not apply multiple-testing correction
(FDR). For condition-level differential expression with full DE workflows, export counts
(see {doc}`pseudobulk_and_differential_expression`) and use an external tool.

```{code-cell} ipython3
ds.run_marker_search(
    group_key='RNA_leiden_cluster',
    gene_batch_size=100
)
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key='RNA_leiden_cluster',
    topn=5,
    figsize=(5, 9)
)
```

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
    color_by='CD14',
    sort_values=True,
)
```

Annotation from markers, known gene panels, and subclustering is covered in
{doc}`annotation`.

## Common mistakes and limitations

- Reusing QC thresholds from another dataset without inspecting distributions
- Calling `run_umap` or clustering before `make_graph`
- Treating marker `p_value` columns as FDR-corrected DE results
- Expecting filtered cells to disappear from `ds.cells` (they remain, with `I=False`)

## Summary of saved results

| Kind | Keys / location |
|---|---|
| QC columns | `RNA_nCounts`, `RNA_nFeatures`, `RNA_percentMito`, … |
| Active cells | cell key `I` |
| HVGs | feature column `I__hvgs` (pass as `hvgs`) |
| Embedding | `RNA_UMAP1`, `RNA_UMAP2` |
| Clusters | `RNA_leiden_cluster` |
| Markers | marker tables from `run_marker_search` / `get_markers` |

## Further reading

- [Single-cell best practices: quality control](https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html)
- [Single-cell best practices: clustering](https://www.sc-best-practices.org/cellular_structure/clustering.html)
- Scarf paper: https://doi.org/10.1038/s41467-022-32097-3

## Next steps

- {doc}`quality_control`
- {doc}`annotation`
- {doc}`gene_set_enrichment`
- {doc}`plotting`
- {doc}`data_integration`
