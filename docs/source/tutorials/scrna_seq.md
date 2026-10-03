---
description: Interpret a prepared scRNA-seq workflow with marker-supported PBMC labels.
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

# Identify PBMC populations with scRNA-seq

A cluster number tells us which cells have similar RNA profiles. To give a cluster a biological
name, we need evidence from its genes. Here we will inspect a prepared blood-cell analysis,
compare a small marker panel, and assign broad cell types.

Start with the {ref}`quick start <quickstart>` if you want to run the standard pipeline on your
own counts. This page uses saved results so we can concentrate on interpreting them.

## Open the analysis and look at the clusters

```{code-cell} ipython3
# Open count stores and run Scarf analyses.
import scarf

# Keep routine execution messages out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr")
# Open the saved analysis and its exact results.
run = ds.pipeline.open(label="docs_default")
# Inspect the opened store's cells and features.
ds
```

The prepared run contains the selected cells, clusters, UMAP coordinates, and marker results.
Its name is `docs_default`, but it used dataset-specific filtering, 500 variable genes, and
15 PCs. Those settings preserve the example we will interpret; they are not the current pipeline
defaults.

```{code-cell} ipython3
# Locate the numbered clusters in the prepared PBMC UMAP.
ds.plots.embedding(run=run, color_by="clusters")
```

Look for the main groups and the smaller populations. Nearby cells have similar profiles in
this view, but the size of a gap between clusters does not establish how different their cell
types are. We will use marker expression to investigate that.

## Compare markers before naming the groups

The panel below includes several genes for each broad lineage. In a dot plot, a larger dot means
more cells express the gene; its colour shows the mean expression in the cluster.

```{code-cell} ipython3
# Choose several markers for each broad lineage.
marker_panel = {
    "Monocyte": ["LST1", "S100A8", "FCGR3A"],
    "B cell": ["MS4A1", "CD79A"],
    "T cell": ["CD3D", "IL7R"],
    "NK cell": ["NKG7", "GNLY"],
    "pDC-like": ["GZMB", "JCHAIN"],
}
# Compare marker expression and detection across the selected groups.
ds.plots.dotplot(features=marker_panel, groups=run["clusters"])
```

MS4A1 and CD79A support B-cell identities; CD3D and IL7R support T cells. NKG7 and GNLY help
identify cytotoxic populations, so compare them with the T-cell markers before calling a group
NK cells. LST1, S100A8, and FCGR3A help distinguish the monocyte populations.

GZMB together with JCHAIN suggests a small pDC-like population. This is a provisional label:
the {doc}`annotation` tutorial checks IL3RA and LILRA4 and examines competing interpretations.

For a gene with much lower expression than the others, it can help to scale each gene separately.
The next plot changes only the colour scale: values are now relative within each gene, so colours
cannot be used to compare absolute expression between different genes.

```{code-cell} ipython3
# Compare marker expression and detection across the selected groups.
ds.plots.dotplot(features=marker_panel, groups=run['clusters'], standardize='feature')
```

## Give the clusters broad names

The marker evidence supports the following broad labels for this prepared result. Several
clusters share a label because they belong to the same lineage. These cluster numbers are
specific to this analysis and must not be copied to another dataset.

```{code-cell} ipython3
# Assign broad names supported by the marker evidence.
cell_type_by_cluster = {
    "1": "CD14 monocytes",
    "2": "monocytes",
    "3": "B cells",
    "4": "T cells",
    "5": "T cells",
    "6": "NK cells",
    "7": "T cells",
    "8": "T cells",
    "9": "B cells",
    "10": "pDC-like cells",
}
# Review the names assigned to the prepared clusters.
cell_type_by_cluster
```

Save the names in the cell table. The run contains only the analyzed cells, while the cell table
contains every cell, so fill the other rows with an explicit label before inserting the column.

```{code-cell} ipython3
# Work with numeric arrays and cell masks.
import numpy as np

# Locate the analyzed cells within the full cell table.
analysis_cells = run.cells.fetch_all("I").astype(bool)
# Read cluster labels in the selected cells' order.
cluster_values = run.cells.fetch("clusters").astype(str)
# Give cells outside the analysis an explicit label.
cell_types = np.full(ds.cells.N, "Not analyzed", dtype=object)
# Fill analyzed rows with their cluster's chosen cell-type label.
cell_types[analysis_cells] = [cell_type_by_cluster[value] for value in cluster_values]
# Save the calculated values in the cell table.
ds.cells.insert("pbmc_cell_type", cell_types, overwrite=True)
# Count analyzed cells assigned to each cell type.
labels, counts = np.unique(cell_types[analysis_cells], return_counts=True)
# Show the number of analyzed cells assigned to each cell type.
{str(label): int(count) for label, count in zip(labels, counts, strict=True)}
```

This writes our labels to `pbmc_cell_type`; rerunning the cell replaces that column. The saved
clusters remain available. Use their UMAP to display the new names:

```{code-cell} ipython3
# Show the assigned broad cell types on the saved UMAP.
ds.plots.embedding(layout=run["umap"], color_by="pbmc_cell_type")
```

## Build on this analysis

Continue with {doc}`annotation` to distinguish finer cell types and states using supporting and
negative markers. Before interpreting your own data, review {doc}`quality_control` and check
that the retained cells make sense for your study.

When the first result raises a specific question, use {doc}`feature_selection`,
{doc}`dimensionality_reduction`, or {doc}`clustering` to compare one choice at a time.
{doc}`graph_construction` explains the individual computation steps, and
{doc}`reuse_and_tracing` shows how to reopen and compare saved analyses.

Marker tests here compare cells, not biological replicates. For differences between conditions,
use a suitable study design and the {doc}`pseudobulk_and_differential_expression` or
{doc}`condition_comparisons` guide.
