---
description: Build an assay graph stage by stage and branch parameters with explicit artifact references.
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

(graph_construction_guide)=

# Building neighbourhood graphs step by step

Embeddings, clustering, imputation, and trajectories consume a cell graph.
Scarf builds that graph from a selected cell population and feature set through separate persisted stages.
Calling the stages directly is useful when you need to branch one parameter, insert batch correction, or {term}`reuse` an expensive reduction.
Start with {doc}`scrna_seq` if you have not yet run an analysis. This page explains the individual
steps behind that workflow.

Feature selection is covered in {doc}`feature_selection`.
This guide begins once a feature-selection artifact exists.

## 1. The standard graph workflow

For RNA, the graph stages are:

1. Normalize the selected genes.
2. Reduce them with PCA.
3. Optionally correct the reduced coordinates with Harmony.
4. Build an approximate-neighbour index.
5. Query `k` neighbours per cell.
6. Convert neighbour distances into weighted connectivity.
7. Build a separate initialization for UMAP and t-SNE.

ATAC follows the same shape but uses TF-IDF normalization and LSI.
The initialization does not define graph edges; it only supplies starting coordinates for a layout.

`ds.pipeline.run()` orchestrates the standard RNA path and can continue through UMAP, clustering, doublet scoring, and markers.
Use it when the defaults match the analysis.
The stage methods below expose the same persisted results with more control.

## 2. Build an RNA graph explicitly

The rebuilt store carries a completed `docs_default` pipeline run. This page starts from that
run's frozen cell and feature selections, then calls every graph stage explicitly. Identical calls
reuse the completed baseline artifacts; later sections create only the branches they discuss.

```{code-cell} ipython3
# Arrange and save Matplotlib figures.
import matplotlib.pyplot as plt
# Work with numeric arrays and cell masks.
import numpy as np
# Summarize cells and results in tables.
import pandas as pd

# Open count stores and run Scarf analyses.
import scarf
# Use Scarf plotting options and diagnostics.
import scarf.plotting as splt

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Open the saved analysis that supplies the starting selections.
baseline = ds.pipeline.open(label="docs_default")
# Inspect the opened store's cells and features.
ds
```

Retrieve the cell and gene selections used by the saved analysis:

```{code-cell} ipython3
# Keep the cells used by the saved analysis.
cell_selection = baseline["analysis_cell_selection"]
# Keep the variable genes used by the saved analysis.
hvg_ref = baseline["highly_variable_features"]
# Use the cell metadata frozen with this analysis.
run_cells = baseline.cells
# Inspect the saved cell and gene selections.
{"cells": cell_selection, "genes": hvg_ref}
```

Each method returns a reference to its saved result. Pass it to the next method to keep the
steps connected. We use 15 PCs to match the prepared example; `run_pca` defaults to 21.

```{code-cell} ipython3
# Normalize counts over the selected features.
normalized = ds.run_normalization(cell_selection, hvg_ref)
# Reduce the normalized expression to principal components.
pca = ds.run_pca(normalized, dims=15)
# Inspect the PCA result that will supply neighbor coordinates.
pca
```

Build the neighbor index from PCA, then turn neighbor distances into connectivity:

```{code-cell} ipython3
# Build the nearest-neighbor search index from the reduction.
ann_index = ds.build_ann_index(pca)
# Find neighbors of each cell in the reduced space.
neighbors = ds.query_neighbors(ann_index)
# Convert neighbor distances into weighted connectivity.
graph = ds.build_connectivity_map(neighbors)
# Inspect the saved connectivity graph.
graph
```

The default neighbour count is 11. We will change only that value in the comparison below.

`load_graph` returns a sparse cell-by-cell connectivity matrix. Its default keeps the directed
neighbour edges. For the diagnostics below, use `symmetric=True` to include a connection when
either cell selects the other. This lets us count each cell's neighbours in either direction.

```{code-cell} ipython3
# Load a symmetric sparse graph for diagnostics.
loaded_graph = ds.load_graph(graph, symmetric=True)
# Check the number of cells and nonzero graph edges.
loaded_graph.shape, loaded_graph.nnz
```

```{code-cell} ipython3
# Inspect graph degree and edge-weight distributions.
splt.graph_qc(loaded_graph)
```

Check isolation and whether degree tracks QC metrics before treating the graph as ready for clustering:

```{code-cell} ipython3
# Count the neighbors connected to each cell.
degrees = np.asarray((loaded_graph != 0).sum(axis=0)).ravel()
# Summarize graph coverage and isolated cells.
pd.Series(
    {
        "active cells": int(loaded_graph.shape[0]),
        "isolated cells": int((degrees == 0).sum()),
        "median degree": float(np.median(degrees)),
        "min degree": int(degrees.min()),
        "max degree": int(degrees.max()),
    },
    name="graph coverage",
)
```

```{code-cell} ipython3
# Align graph degrees with the same cells' quality measurements.
degree_vs_qc = pd.DataFrame(
    {
        "degree": degrees,
        "RNA_nCounts": run_cells.fetch("RNA_nCounts"),
        "RNA_nFeatures": run_cells.fetch("RNA_nFeatures"),
    }
)
# Check whether graph degree tracks the quality measurements.
degree_vs_qc.corr(numeric_only=True)
```

The graph should include every active cell and have finite nonzero connectivities.
A disconnected graph, many isolated cells, or degree structure driven by a QC metric warrants revisiting features, PCA dimensions, or `k`.

## 3. Use the graph for a layout and clustering

`run_umap` and `run_tsne` require both the graph and its matching initialization.
Leiden and Paris require the graph.
Each returns an immutable artifact without adding cell-metadata columns. Use
`load_paris_clustering(ref)` only when hierarchy diagnostics are needed.
Resolution 0.5 matches the prepared PBMC analysis; Leiden's default is 1.0.

```{code-cell} ipython3
# Build starting coordinates for the embedding.
initialization = ds.build_embedding_initialization(pca)
# Calculate the UMAP coordinates from the graph.
umap = ds.run_umap(graph, initialization)
# Find groups of cells in the graph.
clusters = ds.run_leiden_clustering(graph, resolution=0.5)
# Read cluster labels in the selected cells' order.
cluster_values = np.asarray(ds.load_artifact(clusters)["values"][:])
# Color the graph-derived UMAP by its Leiden clusters.
ds.plots.embedding(layout=umap, color_by=clusters)
```

## 4. Branch by retaining both references

Suppose the PCA and ANN index are expensive but two neighbour counts need to be compared.
Reuse the same index and retain both returned graph references.

```{code-cell} ipython3
# Query the same index with 21 neighbors per cell.
neighbors_k21 = ds.query_neighbors(ann_index, k=21)
# Build connectivity for the 21-neighbor comparison.
graph_k21 = ds.build_connectivity_map(neighbors_k21)
# Check that the changed parameters produced a distinct result.
graph_k21 != graph
```

Both branches remain complete, addressable artifacts.
Downstream calls must receive one of them explicitly, so a parameter experiment cannot silently replace another branch.

Degree and edge weight both shift when every cell sees more neighbours:

```{code-cell} ipython3
# Load the comparison graph with symmetric edges.
loaded_graph_k21 = ds.load_graph(graph_k21, symmetric=True)
# Compare the number of stored edges in the two graphs.
pd.Series(
    {
        "k=11 nnz": int(loaded_graph.nnz),
        "k=21 nnz": int(loaded_graph_k21.nnz),
    },
    name="edges",
)
```

```{code-cell} ipython3
# Inspect graph degree and edge-weight distributions.
splt.graph_qc(loaded_graph_k21)
```

To analyse the side branch, pass its exact graph reference.
Retain both returned refs so neither branch replaces the other.

```{code-cell} ipython3
# Cluster the 21-neighbor graph at the same resolution.
clusters_k21 = ds.run_leiden_clustering(graph_k21, resolution=0.5)
# Read the comparison cluster labels.
cluster_values_k21 = np.asarray(ds.load_artifact(clusters_k21)["values"][:])
# Count cells in each reported category.
pd.Series(cluster_values_k21, name="cluster").value_counts().sort_index()
```

Place both partitions on the shared `k=11` UMAP so changes in group boundaries are visible,
then compare their assignments with a crosstab:

```{code-cell} ipython3
# Create axes for the comparison panels.
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
# Draw each result on its comparison axes.
for axis, cluster_ref, title in zip(
    axes,
    (clusters, clusters_k21),
    ("k=11 Leiden", "k=21 Leiden"),
    strict=True,
):
    # Compare the two clusterings on the same k=11 UMAP.
    ds.plots.embedding(
        layout=umap,
        color_by=cluster_ref,
        target=axis,
        show_titles=False,
        show=False,
    )
    # Label the panel with the result it shows.
    axis.set_title(title)
# Adjust spacing between the comparison panels.
figure.tight_layout()
# Display the completed comparison figure.
figure
```

```{code-cell} ipython3
# Count cells shared by each pair of cluster assignments.
pd.crosstab(
    pd.Series(cluster_values, name="k=11"),
    pd.Series(cluster_values_k21, name="k=21"),
)
```

Cluster numbers can change even when the groups stay the same. Look for a row spread across
several columns, or a column collecting several rows, to find splits or merges that depend on
`k`. Review marker evidence before accepting those boundaries.

## 5. Recompute only what changed

Artifact identity includes the operation, scientific parameters, and upstream inputs.
Calling an identical stage reuses its completed result.
Changing `k` reuses normalization, PCA, and the ANN index but creates new neighbour and connectivity artifacts.
Changing the cell or feature selection requires new downstream results. The saved results of
the earlier analysis remain available.

Harmony fits between PCA and the ANN index:

```python
# Correct PCA coordinates using a technical batch column.
corrected = ds.run_harmony(pca, ["technical_batch"])
# Build the neighbor index from corrected coordinates.
corrected_index = ds.build_ann_index(corrected)
# Find neighbors in the corrected space.
corrected_neighbors = ds.query_neighbors(corrected_index, k=21)
# Build connectivity from the corrected neighbors.
corrected_graph = ds.build_connectivity_map(corrected_neighbors)
```

Use {doc}`../concepts/provenance` to inspect complete lineage and {doc}`reuse_and_tracing` for reuse and invalidation patterns.
