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
On first open Scarf streams the count matrix once to compute initialization statistics: per-cell QC columns such as `RNA_nCounts` and `RNA_nFeatures`, mito/ribo fractions when gene-name patterns match, and per-feature `nCells` and `dropOuts` columns.
`min_features_per_cell` marks cells inactive when they have fewer non-zero features than the threshold.
The physical feature column `I` remains all true; feature filtering is an explicit artifact-producing step.

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
qc_cell_selection = ds.snapshot_cell_selection('I')
ds.plots.distribution(
    keys=qc_cols,
    cell_selection=qc_cell_selection,
    kind='violin',
    max_points=2000,
)
```

Each violin is one QC metric before filtering; set thresholds from the tails of these distributions.

```{code-cell} ipython3
n_before = int(ds.cells.fetch_all('I').sum())
cell_selection = ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0]
)
cell_mask = np.asarray(ds.load_artifact(cell_selection)['values'][:], dtype=bool)
print(f'Cells in input selection: {n_before}')
print(f'Cells after filter: {int(cell_mask.sum())}')
print(f'Excluded cells: {int((~cell_mask).sum())}; total in store: {len(cell_mask)}')
```

```{note}
Filtering returns an immutable selection artifact and does not change the boolean cell key `I` or
delete rows. Pass that exact ref to downstream analytical producers. See {doc}`data_organization`.
```

```{code-cell} ipython3
pd.DataFrame({key: ds.cells.fetch_all(key)[cell_mask] for key in qc_cols}).describe()
```

The summary now covers only the cells selected by the returned artifact.

## 3. Feature selection

`select_hvgs` ranks genes by corrected variance and returns an immutable {term}`feature selection`
artifact and leaves feature metadata unchanged.

By default, Scarf excludes common mitochondrial, ribosomal, cell-cycle, HLA/H2, histone, and sex-linked gene-name patterns, together with genes detected in nearly every selected cell.
These defaults reduce technical and broadly shared signals in this teaching workflow.
Use `blacklist=""` to keep all names, pass a custom regular expression for a dataset-specific exclusion, or set `max_cells=np.inf` to disable the ubiquitous-gene filter.
The {doc}`feature_selection` guide explains the exact patterns and how to compare feature sets.

```{code-cell} ipython3
hvg_ref = ds.select_hvgs(
    cell_selection,
    min_cells=20,
    top_n=500,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=True,
)
print(
    'Selected genes:',
    int(np.asarray(ds.load_artifact(hvg_ref)['values'][:]).sum()),
)
hvg_ref
```

The selected genes should span the fitted mean-variance trend rather than being concentrated among only the most abundant genes.
Very few retained genes or a selection dominated by one gene family warrants inspection before continuing.

## 4. Normalization

By default, `run_normalization` scales each selected cell profile by the sum of its selected features (here the HVGs; `renormalize_subset=True`), multiplies by `ds.RNA.sf`, and applies log1p (`log_transform=True`).
Full `RNA_nCounts` library size is used only when `renormalize_subset=False`.
The default size factor is 1000, and the earlier filter removes cells below that count.

```{code-cell} ipython3
normalized = ds.run_normalization(cell_selection, hvg_ref)
status = ds.inspect_artifact(normalized)
print('Size factor (ds.RNA.sf):', ds.RNA.sf)
print('cell selection:', status.inputs['cell_selection'])
print('feature selection:', status.inputs['feature_selection'])
```

The normalized {term}`artifact` records both exact selection refs.

## 5. PCA

PCA represents the dominant axes of variation among the selected genes.
Fifteen components are sufficient for this controlled PBMC example; the elbow plot shows explained variance flattening after the early components.

```{code-cell} ipython3
pca = ds.run_pca(normalized, dims=15, show_elbow_plot=True)
```

Choosing a component count is a scientific decision on new data.
The {doc}`dimensionality_reduction` guide shows how to compare choices.

## 6. Graph construction

The remaining steps index the PCA coordinates, find nearby cells, and turn those neighbours into a weighted graph.
Downstream layouts and clusters consume this graph rather than the count matrix directly.

```{code-cell} ipython3
initialization = ds.build_embedding_initialization(pca)
ann_index = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann_index, k=11)
graph_ref = ds.build_connectivity_map(neighbors)

graph = ds.load_graph(graph_ref)
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

Run UMAP on the explicit graph and its matching initialization artifact.
The result is an immutable embedding artifact.

```{code-cell} ipython3
umap_ref = ds.run_umap(
    graph_ref,
    initialization,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
```

```{code-cell} ipython3
ds.plots.embedding(layout=umap_ref)
```

Cells are placed by neighbourhood-graph proximity on the UMAP.

```{code-cell} ipython3
ds.plots.embedding(layout=umap_ref, color_by="RNA_nCounts")
```

Library size varies across the embedding; check whether high-count cells dominate one region.

UMAP preserves local neighbourhood evidence but its global distances and empty space are not quantitative measurements.
Parameter choice, densMAP, and t-SNE are covered in {doc}`dimensionality_reduction`.

## 8. Clustering

Leiden clustering runs on the same neighbourhood graph.
This manual call returns its exact cluster-label artifact. A pipeline run keeps the same artifact
refs in frozen run-local fields.

```{code-cell} ipython3
leiden = ds.run_leiden_clustering(graph_ref, resolution=0.5)
leiden_values = np.asarray(ds.load_artifact(leiden)['values'][:])
pd.Series(leiden_values).value_counts().sort_index()
```

Cluster sizes are worth a look before plotting: a resolution that is too high splits one cell type into several small clusters.

```{code-cell} ipython3
ds.plots.embedding(layout=umap_ref, color_by=leiden)
```

Each colour is a Leiden partition on the same UMAP coordinates.

Paris provides a hierarchical view of the same graph.
Its automatic cut is a useful second checkpoint, not a replacement for biological validation.

```{code-cell} ipython3
paris = ds.run_paris_clustering(graph_ref)
paris_values = np.asarray(ds.load_artifact(paris)['labels'][:])
pd.Series(paris_values).value_counts().sort_index()
```

```{code-cell} ipython3
ds.plots.embedding(layout=umap_ref, color_by=paris)
```

```{code-cell} ipython3
pd.crosstab(
    pd.Series(leiden_values, name='Leiden'),
    pd.Series(paris_values, name='Paris'),
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
all_features = ds.set_feature_selection(
    from_assay='RNA',
    feature_indexes=range(ds.RNA.feats.N),
)
marker_ref = ds.run_marker_search(
    leiden,
    features=all_features,
)
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    marker=marker_ref,
    topn=5,
    figsize=(5, 9)
)
```

Rows are top marker genes per cluster; stronger scores mark more cluster-specific genes.

```{code-cell} ipython3
df = ds.get_markers(
    marker=marker_ref,
    group_id='1',
    min_score=-1,
    min_frac_exp=-1
)
df.head()
```

Rank the same three lineage genes across all Leiden groups before plotting them on the embedding.

```{code-cell} ipython3
markers = ds.get_markers(
    marker=marker_ref,
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
ds.plots.dotplot(
    features=["CD14", "MS4A1", "CD3D"],
    groups=leiden,
)
```

CD14, MS4A1, and CD3D provide monocyte-, B-, and T-cell evidence when those lineages are present.
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
- Expecting filtering to delete rows or change live `I` instead of returning a selection artifact
