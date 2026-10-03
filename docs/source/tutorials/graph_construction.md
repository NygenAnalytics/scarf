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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
import scarf.plotting as splt

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
baseline = ds.pipeline.open(label="docs_default")
cell_selection = baseline["analysis_cell_selection"]
hvg_ref = baseline["highly_variable_features"]
run_cells = baseline.cells
```

Each method returns a reference to its saved result. Pass it to the next method to keep the
steps connected. We use 15 PCs to match the prepared example; `run_pca` defaults to 21.

```{code-cell} ipython3
normalized = ds.run_normalization(cell_selection, hvg_ref)
pca = ds.run_pca(normalized, dims=15)
initialization = ds.build_embedding_initialization(pca)
ann_index = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann_index)
graph = ds.build_connectivity_map(neighbors)
graph
```

The default neighbour count is 11. We will change only that value in the comparison below.

`load_graph` returns a sparse cell-by-cell connectivity matrix. Its default keeps the directed
neighbour edges. For the diagnostics below, use `symmetric=True` to include a connection when
either cell selects the other. This lets us count each cell's neighbours in either direction.

```{code-cell} ipython3
loaded_graph = ds.load_graph(graph, symmetric=True)
loaded_graph.shape, loaded_graph.nnz
```

```{code-cell} ipython3
splt.graph_qc(loaded_graph)
```

Check isolation and whether degree tracks QC metrics before treating the graph as ready for clustering:

```{code-cell} ipython3
degrees = np.asarray((loaded_graph != 0).sum(axis=0)).ravel()
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
degree_vs_qc = pd.DataFrame(
    {
        "degree": degrees,
        "RNA_nCounts": run_cells.fetch("RNA_nCounts"),
        "RNA_nFeatures": run_cells.fetch("RNA_nFeatures"),
    }
)
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
umap = ds.run_umap(graph, initialization)
clusters = ds.run_leiden_clustering(graph, resolution=0.5)
cluster_values = np.asarray(ds.load_artifact(clusters)["values"][:])
ds.plots.embedding(layout=umap, color_by=clusters)
```

## 4. Branch by retaining both references

Suppose the PCA and ANN index are expensive but two neighbour counts need to be compared.
Reuse the same index and retain both returned graph references.

```{code-cell} ipython3
neighbors_k21 = ds.query_neighbors(ann_index, k=21)
graph_k21 = ds.build_connectivity_map(neighbors_k21)
graph_k21 != graph
```

Both branches remain complete, addressable artifacts.
Downstream calls must receive one of them explicitly, so a parameter experiment cannot silently replace another branch.

Degree and edge weight both shift when every cell sees more neighbours:

```{code-cell} ipython3
loaded_graph_k21 = ds.load_graph(graph_k21, symmetric=True)
pd.Series(
    {
        "k=11 nnz": int(loaded_graph.nnz),
        "k=21 nnz": int(loaded_graph_k21.nnz),
    },
    name="edges",
)
```

```{code-cell} ipython3
splt.graph_qc(loaded_graph_k21)
```

To analyse the side branch, pass its exact graph reference.
Retain both returned refs so neither branch replaces the other.

```{code-cell} ipython3
clusters_k21 = ds.run_leiden_clustering(graph_k21, resolution=0.5)
```

Place both partitions on the shared `k=11` UMAP so changes in group boundaries are visible,
then compare their assignments with a crosstab:

```{code-cell} ipython3
cluster_values_k21 = np.asarray(ds.load_artifact(clusters_k21)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, cluster_ref, title in zip(
    axes,
    (clusters, clusters_k21),
    ("k=11 Leiden", "k=21 Leiden"),
    strict=True,
):
    ds.plots.embedding(
        layout=umap,
        color_by=cluster_ref,
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

```{code-cell} ipython3
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
corrected = ds.run_harmony(pca, ["technical_batch"])
corrected_index = ds.build_ann_index(corrected)
corrected_neighbors = ds.query_neighbors(corrected_index, k=21)
corrected_graph = ds.build_connectivity_map(corrected_neighbors)
```

Use {doc}`../concepts/provenance` to inspect complete lineage and {doc}`reuse_and_tracing` for reuse and invalidation patterns.
