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
- Run the standard RNA pipeline and inspect its exact outputs
- Check feature selection, normalization, PCA, graph, UMAP, and Leiden results
- Compare the Leiden result with Scarf's hierarchical Paris alternative
- Rank marker genes per cluster and inspect known immune markers


## Dataset

`tenx_5K_pbmc_rnaseq` is a public 10x Genomics PBMC dataset distributed through the `scarf_docs` Cytebase repository.

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
)
```

## 1. Import

Read the Cell Ranger H5 file, inspect cell and feature counts, then write a Zarr store.

```{note}
A Zarr "file" is a directory hierarchy on disk, not a single HDF5-style file.
```

```{code-cell} ipython3
reader = scarf.CrH5Reader(f"{dataset}/data.h5")
reader.nCells, reader.nFeatures
```

```{code-cell} ipython3
scarf.CrToZarr(
    reader,
    zarr_loc=f"{dataset}/data.zarr",
).dump()
```

Open a `DataStore`.
On first open Scarf streams the count matrix once to compute initialization statistics: per-cell QC columns such as `RNA_nCounts` and `RNA_nFeatures`, mito/ribo fractions when gene-name patterns match, and per-feature `nCells` and `dropOuts` columns.
`min_features_per_cell` marks cells inactive when they have fewer non-zero features than the threshold.
The physical feature column `I` remains all true; feature filtering is an explicit artifact-producing step.

Opening this store prints a message that the smallest cell count is below the RNA size factor of 1000.
It refers to normalization, which is covered in step 3, and the QC filter in step 2 removes those cells.

```{code-cell} ipython3
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4, min_features_per_cell=10)
ds
```

Active cells, assay feature counts, and the QC column names computed on open.

## 2. Quality control

Inspect QC distributions, then set dataset-specific thresholds.
Thresholds below are chosen for this PBMC dataset; other datasets need their own cutoffs.
Deeper QC options, including `auto_filter_cells` and doublet detection, are in {doc}`quality_control`.

```{code-cell} ipython3
qc_cols = [
    c
    for c in ("RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo")
    if c in ds.cells.columns
]
qc_cell_selection = ds.snapshot_cell_selection("I")
ds.plots.distribution(
    keys=qc_cols,
    cell_selection=qc_cell_selection,
    kind="violin",
    max_points=2000,
)
```

Each violin is one QC metric before filtering; set thresholds from the tails of these distributions.

```{code-cell} ipython3
manual_filter = {
    "method": "manual",
    "attrs": ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    "highs": [15000, 4000, 15],
    "lows": [1000, 500, 0],
}
```

## 3. Run the standard RNA pipeline

Pass the QC bounds to one pipeline run. This produces one durable record whose outputs retain the
exact selection, feature, reduction, graph, clustering, and marker refs used together. The QC
columns are frozen with the run so later plots do not depend on mutable live metadata.

```{code-cell} ipython3
run = ds.pipeline.run(
    filtering=manual_filter,
    hvg_count=500,
    pca_dims=15,
    neighbors_k=11,
    leiden={"partitions": [0.5]},
    cell_cycle=False,
    paris=False,
    doublets=False,
    markers=True,
    snapshot_columns=qc_cols,
)

graph_ref = run["connectivity_map"]
leiden = run["clusters"]
marker_ref = run["markers"]

cell_mask = run.cells.fetch_all("I")
print(f"Cells in input selection: {int(ds.cells.fetch_all('I').sum())}")
print(f"Cells after filter: {int(cell_mask.sum())}")
print(f"Excluded cells: {int((~cell_mask).sum())}; total in store: {len(cell_mask)}")
run.cells.to_pandas_dataframe(qc_cols).describe()
```

Filtering returns an immutable selection artifact and does not change live `I` or delete rows. The
summary covers only cells selected by the run. See {doc}`data_organization` for branching from
exact output refs and {doc}`reuse_and_tracing` for reopening and comparing runs.

## 4. Inspect feature selection and normalization

`select_hvgs` ranks genes by corrected variance and returns an immutable {term}`feature selection`
artifact and leaves feature metadata unchanged.

By default, Scarf excludes common mitochondrial, ribosomal, cell-cycle, HLA/H2, histone, and sex-linked gene-name patterns, together with genes detected in nearly every selected cell.
These defaults reduce technical and broadly shared signals in this teaching workflow.
Use `blacklist=""` to keep all names, pass a custom regular expression for a dataset-specific exclusion, or set `max_cells=np.inf` to disable the ubiquitous-gene filter.
The {doc}`feature_selection` guide explains the exact patterns and how to compare feature sets.

```{code-cell} ipython3
print(
    "Selected genes:",
    int(np.count_nonzero(run.features.fetch("highly_variable_features"))),
)
print("HVG artifact:", run["highly_variable_features"])
```

The selected genes should span the fitted mean-variance trend rather than being concentrated among only the most abundant genes.
Very few retained genes or a selection dominated by one gene family warrants inspection before continuing.

By default, `run_normalization` scales each selected cell profile by the sum of its selected features (here the HVGs; `renormalize_subset=True`), multiplies by `ds.RNA.sf`, and applies log1p (`log_transform=True`).
Full `RNA_nCounts` library size is used only when `renormalize_subset=False`.
The default size factor is 1000, and the earlier filter removes cells below that count.

```{code-cell} ipython3
status = ds.inspect_artifact(run["normalized"])
print("Size factor (ds.RNA.sf):", ds.RNA.sf)
print("cell selection:", status.inputs["cell_selection"])
print("feature selection:", status.inputs["feature_selection"])
```

The normalized {term}`artifact` records both exact selection refs.

## 5. Inspect PCA and the graph

PCA represents the dominant axes of variation among the selected genes. Fifteen components are
sufficient for this controlled PBMC example. The pipeline indexes those coordinates, finds nearby
cells, and turns the neighbours into a weighted graph. Downstream layouts and clusters consume
that exact graph rather than the count matrix directly.

```{code-cell} ipython3
pca_status = ds.inspect_artifact(run["pca"])
pca_shape = ds.load_artifact(run["pca"])["data"].shape
graph = ds.load_graph(graph_ref)
degrees = graph.getnnz(axis=1)
print("PCA input:", pca_status.inputs["normalized"])
print("PCA coordinate shape:", pca_shape)
print(graph.shape, graph.nnz)
print(
    "Degree min / median / max:",
    int(degrees.min()),
    int(np.median(degrees)),
    int(degrees.max()),
)
```

`load_graph` returns the result as a sparse cell-by-cell matrix.
Shape and nnz confirm the graph covers the selected cells; the degree summary checks that
neighbourhood sizes stay near the requested `k`.

```{seealso}
Choosing a component count and checking graph construction are scientific decisions on new data.
See {doc}`dimensionality_reduction` and {doc}`graph_construction` for atomic branches and deeper
diagnostics.
```

## 6. Inspect UMAP

The pipeline's UMAP consumes the graph and its matching initialization artifact. Plotting through
the run uses its frozen selection and fields.

```{code-cell} ipython3
ds.plots.embedding(run=run)
```

Cells are placed by neighbourhood-graph proximity on the UMAP.

```{code-cell} ipython3
ds.plots.embedding(run=run, color_by="RNA_nCounts")
```

Library size varies across the embedding; check whether high-count cells dominate one region.

UMAP preserves local neighbourhood evidence but its global distances and empty space are not quantitative measurements.
Parameter choice, densMAP, and t-SNE are covered in {doc}`dimensionality_reduction`.

## 7. Inspect clustering

This run requested one Leiden resolution, so its selected `clusters` output is that exact Leiden
artifact. Cluster sizes are worth checking before plotting.

```{code-cell} ipython3
leiden_values = np.asarray(run.cells.fetch("clusters"))
pd.Series(leiden_values).value_counts().sort_index()
```

Cluster sizes are worth a look before plotting: a resolution that is too high splits one cell type into several small clusters.

```{code-cell} ipython3
ds.plots.embedding(run=run, color_by="clusters")
```

Each colour is a Leiden partition on the same UMAP coordinates.

Paris is an optional hierarchical alternative. Run its atomic producer on the pipeline's exact
graph so the comparison changes only the clustering method.

```{code-cell} ipython3
paris = ds.run_paris_clustering(graph_ref)
paris_values = np.asarray(ds.load_artifact(paris)["labels"][:])
pd.Series(paris_values).value_counts().sort_index()
```

```{code-cell} ipython3
ds.plots.embedding(layout=run["umap"], color_by=paris)
```

```{code-cell} ipython3
pd.crosstab(
    pd.Series(leiden_values, name="Leiden"),
    pd.Series(paris_values, name="Paris"),
)
```

Both partitions should preserve broad monocyte, B-cell, and T-cell structure.
The sizes and Leiden×Paris crosstab make that concordance readable before the marker step.
Tiny isolated clusters dominated by low-count cells are a reason to revisit QC before interpreting markers.
Resolution sweeps, cluster confidence, graph connectivity, and the Paris tree are covered in {doc}`clustering`.

## 8. Marker genes

`run_marker_search` ranks genes per group.
Results include specificity-oriented scores, Mann-Whitney U test p-values (`p_value`), AUC effect sizes, and within-group Benjamini-Hochberg adjusted values (`p_value_adjusted`).
The adjusted column is a cell-level marker correction for that group, not replicate-aware differential expression.
For condition-level DE with full workflows, export counts (see {doc}`pseudobulk_and_differential_expression`) and use an external tool.

The pipeline already ran marker search against its selected Leiden artifact and frozen feature
universe, so the plots and lookups below reuse that exact result.

```{code-cell} ipython3
ds.plots.marker_heatmap(marker=marker_ref, topn=5, figsize=(5, 9))
```

Rows are top marker genes per cluster; stronger scores mark more cluster-specific genes.

```{code-cell} ipython3
ds.get_markers(marker=marker_ref, group_id="1", min_score=-1, min_frac_exp=-1).head()
```

Plot three lineage genes across all Leiden groups.

```{code-cell} ipython3
ds.plots.dotplot(
    features=["CD14", "MS4A1", "CD3D"],
    groups=leiden,
)
```

CD14, MS4A1, and CD3D provide monocyte-, B-, and T-cell evidence when those lineages are present.

Annotation from markers, known gene panels, and subclustering is covered in {doc}`annotation`.

```{raw} html
<span id="imputation"></span>
```

## 9. Feature imputation

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
