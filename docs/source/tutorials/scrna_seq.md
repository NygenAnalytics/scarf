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

# scRNA-seq primer

Single-cell RNA sequencing captures mRNA counts from thousands of individual cells, so each cell is described by which genes it expresses and how strongly. In peripheral blood mononuclear cells (PBMCs), T cells, B cells, monocytes, NK cells, and plasmacytoid dendritic cells often circulate side by side throughout the body. Sequencing allows us to separate them by their transcriptional profile, and then go further downstream to potentially see differences in these cells across diseases. Furthermore, our sequencing can separate them based on the transcriptional profile, but a cluster of similar cells is not yet a cell type, and needs a baseline definiton based marker-gene evidence that is present

# Identify PBMC populations with scRNA-seq

A cluster number tells us which cells have similar RNA profiles, but there is no use in knowing that we have multiple clusters without identifying their identity. To give a cluster a biological name, we need evidence from its genes expression (transcriptional profile). Here we will inspect a prepared blood-cell analysis, compare a small marker-gene panel, and assign broad cell types.

## Open the analysis and look at the clusters

```{code-cell} ipython3
import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr")
run = ds.pipeline.open(label="docs_default")
ds
```

The prepared run contains the selected cells, clusters, UMAP coordinates, and marker results.

```{code-cell} ipython3
ds.plots.embedding(run=run, color_by="clusters")
```

We can visualize our existings clusters and distributions of "transcriptionally similar cells" in a 2 dimensional space. Look for the main groups and the smaller populations; A gap between clusters does not establish how different their cell types are. We will use marker expression to investigate this next

## Compare markers before naming the groups

With our clusters visualized, we can now use several different genes as proxies for determing the broad lineage of each cluster. In a dot plot, a larger dot means more cells express the gene in that cluster, with its colour shows the mean expression in the cluster.

```{code-cell} ipython3
marker_panel = {
    "Monocyte": ["LST1", "S100A8", "FCGR3A"],
    "B cell": ["MS4A1", "CD79A"],
    "T cell": ["CD3D", "IL7R"],
    "NK cell": ["NKG7", "GNLY"],
    "pDC-like": ["GZMB", "JCHAIN"],
}
ds.plots.dotplot(features=marker_panel, groups=run["clusters"])
```

MS4A1 and CD79A support B-cell identities; CD3D and IL7R support T cells. NKG7 and GNLY help
identify cytotoxic populations, so compare them with the T-cell markers before calling a group
NK cells. LST1, S100A8, and FCGR3A help distinguish the monocyte populations.

GZMB together with JCHAIN suggests a small pDC-like population. This is a provisional label:
the {doc}`annotation` tutorial checks IL3RA and LILRA4 and examines competing interpretations, along with how to go into further depth to confirm annotations.

For a gene with much lower expression than the others, it can help to scale each gene separately.
The next plot changes only the colour scale: values are now relative within each gene, so colours
cannot be used to compare absolute expression between different genes.

```{code-cell} ipython3
ds.plots.dotplot(features=marker_panel, groups=run['clusters'], standardize='feature')
```

## Give the clusters broad names

With the dot plot giving us marker evidence as to what genes dominate expression inside a cluster, we can assign broad labels for the result. Several clusters share the same label because they belong to the same lineage. These cluster numbers are specific to this analysis.

```{code-cell} ipython3
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
cell_type_by_cluster
```

Save the names in the cell table. The run contains only the analyzed cells, while the cell table
contains every cell, so fill the other rows with an explicit label before inserting the column.

```{code-cell} ipython3
import numpy as np

analysis_cells = run.cells.fetch_all("I").astype(bool)
cluster_values = run.cells.fetch("clusters").astype(str)
cell_types = np.full(ds.cells.N, "Not analyzed", dtype=object)
cell_types[analysis_cells] = [cell_type_by_cluster[value] for value in cluster_values]
ds.cells.insert("pbmc_cell_type", cell_types, overwrite=True)
labels, counts = np.unique(cell_types[analysis_cells], return_counts=True)
{str(label): int(count) for label, count in zip(labels, counts, strict=True)}
```

This writes our labels to `pbmc_cell_type`; rerunning the cell replaces that column. The saved
clusters remain available, and then we can use their UMAP to display the new names:

```{code-cell} ipython3
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
