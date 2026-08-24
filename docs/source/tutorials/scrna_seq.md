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

# Cellular heterogeneity with scRNA-seq

This tutorial follows one recommended path from a 5K PBMC count matrix to broad cellular populations and marker genes.
The dataset is small enough for teaching and has familiar immune-cell structure.
It is not evidence for Scarf's scaling claims; see {doc}`../concepts/memory_and_execution` for measured resource profiles.

## Prerequisites

- Scarf installed with the `extra` optional dependencies ({ref}`installation <installation>`)
- Basic familiarity with count matrices

## What you will learn

- Convert Cell Ranger H5 to Zarr and open a `DataStore`
- Filter cells without deleting them from the store
- Select informative genes and build a neighbourhood graph step by step
- Run UMAP and Leiden clustering on that graph
- Compare the Leiden result with Scarf's hierarchical Paris alternative
- Rank marker genes per cluster and inspect known immune markers


## Dataset

`tenx_5K_pbmc_rnaseq` is a public 10x Genomics PBMC dataset distributed through the `scarf_docs` Cytebase repository.

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
)
```

## 1. Import

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
writer.dump()
```

Open a `DataStore`.
On first open Scarf streams the count matrix once to compute initialization statistics: per-cell QC columns such as `RNA_nCounts` and `RNA_nFeatures`, mito/ribo fractions when gene-name patterns match, and the per-feature cell counts used for feature filtering.
`min_features_per_cell` marks cells inactive when they have fewer non-zero features than the threshold.
Feature filtering uses `min_cells_per_feature` (default 20).

Opening this store prints a message that the smallest cell count is below the RNA size factor of 1000.
It refers to normalization, which is covered in step 3, and the QC filter in step 2 removes those cells.

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    min_features_per_cell=10
)
ds
```

Active cells, assay feature counts, and the QC column names computed on open.

## 2. Quality control

Inspect QC distributions, then set dataset-specific thresholds.
Thresholds below are chosen for this PBMC dataset; other datasets need their own cutoffs.
Deeper QC options, including `auto_filter_cells` and doublet detection, are in {doc}`quality_control`.

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

Each violin is one QC metric before filtering; set thresholds from the tails of these distributions.

```{code-cell} ipython3
n_before = int(ds.cells.fetch_all('I').sum())
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0]
)
I = ds.cells.fetch_all('I')
print(f'Active cells before filter: {n_before}')
print(f'Active cells after filter: {int(I.sum())}')
print(f'Inactive cells (I=False): {int((~I).sum())}; total in store: {len(I)}')
ds.cells.to_pandas_dataframe(columns=['I'])['I'].value_counts()
```

```{note}
Filtered cells are marked inactive in the boolean cell key `I`, not deleted.
Most `DataStore` methods default to `cell_key='I'`.
See {doc}`data_organization`.
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

## 3. Feature selection

`mark_hvgs` ranks genes by corrected variance and marks highly variable genes.
The feature column is named with the {term}`cell key` prefix (here `I__hvgs`).
Pass `feat_key='hvgs'` later; Scarf resolves the prefix.
See {term}`feat_key`.

By default, Scarf excludes common mitochondrial, ribosomal, cell-cycle, HLA/H2, histone, and sex-linked gene-name patterns, together with genes detected in nearly every selected cell.
These defaults reduce technical and broadly shared signals in this teaching workflow.
Use `blacklist=""` to keep all names, pass a custom regular expression for a dataset-specific exclusion, or set `max_cells=np.inf` to disable the ubiquitous-gene filter.
The {doc}`feature_selection` guide explains the exact patterns and how to compare feature sets.

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
```

The selected genes should span the fitted mean-variance trend rather than being concentrated among only the most abundant genes.
Very few retained genes or a selection dominated by one gene family warrants inspection before continuing.

## 4. Normalization

By default, `run_normalization` scales each selected cell profile by the sum of its selected features (here the HVGs; `renormalize_subset=True`), multiplies by `ds.RNA.sf`, and applies log1p (`log_transform=True`).
Full `RNA_nCounts` library size is used only when `renormalize_subset=False`.
The default size factor is 1000, and the earlier filter removes cells below that count.

```{code-cell} ipython3
normalized = ds.run_normalization(feat_key='hvgs')
opts = ds.inspect_artifact(normalized).execution_options or {}
print('Size factor (ds.RNA.sf):', ds.RNA.sf)
print('cell_key:', opts.get('cell_key'), 'feat_key:', opts.get('feat_key'))
```

The normalized {term}`artifact` records both the active cell selection and the `hvgs` feature selection.

## 5. PCA

PCA represents the dominant axes of variation among the selected genes.
Fifteen components are sufficient for this controlled PBMC example; the elbow plot shows explained variance flattening after the early components.

```{code-cell} ipython3
ds.run_pca(dims=15, show_elbow_plot=True)
```

Choosing a component count is a scientific decision on new data.
The {doc}`dimensionality_reduction` guide shows how to compare choices.

## 6. Graph construction

The remaining steps index the PCA coordinates, find nearby cells, and turn those neighbours into a weighted graph.
Downstream layouts and clusters consume this graph rather than the count matrix directly.

```{code-cell} ipython3
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()

graph = ds.load_graph()
degrees = graph.getnnz(axis=1)
print(graph.shape, graph.nnz)
print(
    'Degree min / median / max:',
    int(degrees.min()),
    int(np.median(degrees)),
    int(degrees.max()),
)
```

`load_graph` returns the result as a sparse cell-by-cell matrix.
Shape and nnz confirm the graph covers the active cells; the degree summary checks that neighbourhood sizes stay near the requested `k`.

```{seealso}
Each step also returns a reference to the artifact it wrote.
Capturing those references allows branches and partial recomputation without changing the recommended path here.
See {doc}`graph_construction`.
```


## 7. UMAP

Run UMAP on the latest graph.
Results are stored as `RNA_UMAP1` and `RNA_UMAP2`.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
```

```{code-cell} ipython3
ds.plots.embedding(layout_key='RNA_UMAP')
```

Cells are placed by neighbourhood-graph proximity on the UMAP.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_nCounts',
)
```

Library size varies across the embedding; check whether high-count cells dominate one region.

UMAP preserves local neighbourhood evidence but its global distances and empty space are not quantitative measurements.
Parameter choice, densMAP, and t-SNE are covered in {doc}`dimensionality_reduction`.

## 8. Clustering

Leiden clustering runs on the same neighbourhood graph.
Cluster labels are saved as `RNA_leiden_cluster`.

```{code-cell} ipython3
ds.run_leiden_clustering(resolution=0.5)
ds.cells.to_pandas_dataframe(
    columns=['RNA_leiden_cluster'],
    key='I'
)['RNA_leiden_cluster'].value_counts().sort_index()
```

Cluster sizes are worth a look before plotting: a resolution that is too high splits one cell type into several small clusters.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

Each colour is a Leiden partition on the same UMAP coordinates.

Paris provides a hierarchical view of the same graph.
Its automatic cut is a useful second checkpoint, not a replacement for biological validation.

```{code-cell} ipython3
paris = ds.run_paris_clustering()
ds.cells.to_pandas_dataframe(
    columns=[paris.label_key],
    key='I'
)[paris.label_key].value_counts().sort_index()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=paris.label_key,
)
```

```{code-cell} ipython3
pd.crosstab(
    pd.Series(ds.cells.fetch('RNA_leiden_cluster', key='I'), name='Leiden'),
    pd.Series(ds.cells.fetch(paris.label_key, key='I'), name='Paris'),
)
```

Both partitions should preserve broad monocyte, B-cell, and T-cell structure.
The sizes and Leiden×Paris crosstab make that concordance readable before the marker step.
Tiny isolated clusters dominated by low-count cells are a reason to revisit QC before interpreting markers.
Resolution sweeps, cluster confidence, graph connectivity, and the Paris tree are covered in {doc}`clustering`.

## 9. Marker genes

`run_marker_search` ranks genes per group.
Results include specificity-oriented scores, Mann-Whitney U test p-values (`p_value`), AUC effect sizes, and within-group Benjamini-Hochberg adjusted values (`p_value_adjusted`).
The adjusted column is a cell-level marker correction for that group, not replicate-aware differential expression.
For condition-level DE with full workflows, export counts (see {doc}`pseudobulk_and_differential_expression`) and use an external tool.

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

Rank the same three lineage genes across all Leiden groups before plotting them on the embedding.

```{code-cell} ipython3
markers = ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id=None,
    min_score=-1,
    min_frac_exp=-1,
)
panel = markers[markers['feature_name'].isin(['CD14', 'MS4A1', 'CD3D'])]
panel.sort_values(
    ['feature_name', 'score'], ascending=[True, False]
).groupby('feature_name', sort=False).head(2)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['CD14', 'MS4A1', 'CD3D'],
    n_columns=3,
    sort_values=True,
)
```

CD14, MS4A1, and CD3D mark monocyte-, B-, and T-cell-like regions when those lineages are present.
The lookup above names which Leiden clusters rank each gene highest.

Annotation from markers, known gene panels, and subclustering is covered in {doc}`annotation`.

```{raw} html
<span id="imputation"></span>
```

## 10. Feature imputation

Graph diffusion is optional and is not part of this default workflow.
See {doc}`imputation` for a focused comparison of observed and imputed expression, including the limits on interpretation.

## Common mistakes and limitations

- Reusing QC thresholds from another dataset without inspecting distributions
- Selecting too few genes to represent rare populations, or so many that technical variation dominates
- Choosing PCA dimensions or neighbours without checking whether the resulting graph is connected and biologically plausible
- Interpreting UMAP distances or empty space as measured biological distances
- Treating every extra cluster at a higher resolution as a distinct cell type
- Requesting marker tests for groups with fewer than two target or reference cells
- Treating marker `p_value` or within-group `p_value_adjusted` columns as replicate-aware DE results
- Expecting filtered cells to disappear from `ds.cells` (they remain, with `I=False`)

