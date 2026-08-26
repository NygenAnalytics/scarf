---
description: Build an assay graph stage by stage, branch parameters, and control the current analysis chain.
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
(graph_and_state)=

# Building neighbourhood graphs step by step

Embeddings, clustering, imputation, and trajectories consume a cell graph.
Scarf builds that graph from a selected cell population and feature set through separate persisted stages.
Calling the stages directly is useful when you need to branch one parameter, insert batch correction, or {term}`reuse` an expensive reduction.

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

The downloaded store is structurally repacked and mounted as a count source so this page builds a complete current analysis chain without reading its persisted analysis state.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
import scarf.plotting as splt
from scarf.embeddings import initial_embedding
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = str(Path(analysis_directory.name) / "counts.zarr")
repack_store(
    f"{dataset}/data.zarr",
    repacked_counts,
    nthreads=2,
)
ds = scarf.mount_datastore(
    repacked_counts,
    at=str(Path(analysis_directory.name) / "graph_analysis.zarr"),
    default_assay="RNA",
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures"],
    highs=[15000, 4000],
    lows=[1000, 500],
    reset_previous=True,
)
hvg_ref = ds.mark_hvgs(
    min_cells=20,
    top_n=500,
    show_plot=False,
)
```

Each method returns an {term}`ArtifactRef`.
Passing it to the next method makes the dependency explicit.

```{code-cell} ipython3
normalized = ds.run_normalization(features=hvg_ref)
pca = ds.run_pca(normalized, dims=15)
initialization = ds.build_embedding_initialization(
    pca,
    n_centroids=100,
)
ann_index = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann_index, k=11)
graph = ds.build_connectivity_map(neighbors)
graph
```

The intermediate stages are addressable artifacts, not only progress messages:

```{code-cell} ipython3
stage_rows = []
for name, ref in (
    ("normalized", normalized),
    ("pca", pca),
    ("initialization", initialization),
    ("ann_index", ann_index),
    ("neighbors", neighbors),
    ("graph", graph),
):
    status = ds.inspect_artifact(ref)
    stage_rows.append(
        {
            "stage": name,
            "operation": status.operation,
            "complete": status.complete,
        }
    )
pd.DataFrame(stage_rows)
```

`load_graph` returns the sparse cell-by-cell connectivity matrix for supported custom graph analyses.

```{code-cell} ipython3
loaded_graph = ds.load_graph()
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
        "RNA_nCounts": ds.cells.fetch("RNA_nCounts"),
        "RNA_nFeatures": ds.cells.fetch("RNA_nFeatures"),
    }
)
degree_vs_qc.corr(numeric_only=True)
```

The graph should include every active cell and have finite nonzero connectivities.
A disconnected graph, many isolated cells, or degree structure driven by a QC metric warrants revisiting features, PCA dimensions, or `k`.

## 3. Understand the current analysis chain

Successful stages normally update a small pointer set for the assay.
In plain language, this is the {term}`analysis chain` that downstream calls should use when no explicit input is supplied.
The public class representing it is `AssayState`.

```{code-cell} ipython3
state = ds.get_assay_state("RNA")
normalized_status = ds.inspect_artifact(state.normalized)
{
    "cell selection": state.cell_key,
    "feature selection": normalized_status.inputs["feature_selection"],
    "normalization": state.normalized,
    "reduction": state.reduction,
    "embedding initialization": state.embedding_initialization,
    "neighbours": state.neighbors,
    "connectivity": state.connectivity_map,
}
```

The initialization artifact stores K-means centers and labels used only as UMAP and t-SNE starting coordinates.
Inspect it, then plot the projected seed layout:

```{code-cell} ipython3
init_status = ds.inspect_artifact(initialization)
init_group = ds.load_artifact(initialization)
centers = np.asarray(init_group["cluster_centers"][:])
labels = np.asarray(init_group["cluster_labels"][:])
{
    "operation": init_status.operation,
    "n_centroids": init_status.parameters.get("n_centroids"),
    "cluster_centers": centers.shape,
    "n_labels": int(pd.Series(labels).nunique()),
    "complete": init_status.complete,
}
```

```{code-cell} ipython3
ini_coords = initial_embedding(centers, labels, 2)
figure, axis = plt.subplots(figsize=(4.5, 4))
axis.scatter(ini_coords[:, 0], ini_coords[:, 1], s=3, alpha=0.4, linewidths=0)
axis.set_xlabel("initialization 1")
axis.set_ylabel("initialization 2")
axis.set_title("Layout seed (not graph edges)")
figure.tight_layout()
figure
```

`run_umap` and `run_tsne` resolve the current connectivity and initialization from this chain when `graph=None`.
Leiden and Paris resolve only the connectivity graph.
They write their cell-metadata columns as usual.
UMAP, t-SNE, and Leiden return the artifact they wrote; Paris returns a `ParisClusteringResult` whose `.ref` holds that artifact.
Either way, a result can be inspected or reused without looking the column up first.

```{code-cell} ipython3
umap = ds.run_umap(
    n_epochs=100,
    parallel=True,
)
clusters = ds.run_leiden_clustering(resolution=0.5)
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_leiden_cluster",
)
```

```{code-cell} ipython3
ds.inspect_artifact(clusters).parameters
```

## 4. Branch without changing the current chain

Suppose the PCA and ANN index are expensive but two neighbour counts need to be compared.
Reuse the same index and set `update_state=False` on the side branch.

```{code-cell} ipython3
neighbors_k21 = ds.query_neighbors(
    ann_index,
    k=21,
    update_state=False,
)
graph_k21 = ds.build_connectivity_map(
    neighbors_k21,
    update_state=False,
)

current_state = ds.get_assay_state("RNA")
current_state.connectivity_map == graph, graph_k21 != graph
```

The side branch remains a complete, addressable artifact, while downstream calls without explicit inputs still use the `k=11` graph.
This prevents a parameter experiment from silently replacing the selected chain.

Degree and edge weight both shift when every cell sees more neighbours:

```{code-cell} ipython3
loaded_graph_k21 = ds.load_graph(
    graph=graph_k21,
)
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
The current chain is untouched, so both partitions stay available for comparison.

```{code-cell} ipython3
ds.run_leiden_clustering(
    graph=graph_k21,
    resolution=0.5,
    label="leiden_k21",
)
```

Place both partitions on the shared `k=11` UMAP so absorption is visible, then quantify agreement with a crosstab:

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, color_by, title in zip(
    axes,
    ("RNA_leiden_cluster", "RNA_leiden_k21"),
    ("k=11 Leiden", "k=21 Leiden"),
    strict=True,
):
    ds.plots.embedding(
        layout_key="RNA_UMAP",
        color_by=color_by,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

```{code-cell} ipython3
pd.crosstab(
    pd.Series(ds.cells.fetch("RNA_leiden_cluster"), name="k=11"),
    pd.Series(ds.cells.fetch("RNA_leiden_k21"), name="k=21"),
)
```

Most cells keep their group.
The off-diagonal mass shows which splits depend on `k`: the smallest `k=11` clusters are absorbed once every cell sees more neighbours, so treat those boundaries as provisional.

## 5. Recompute only what changed

Artifact identity includes the operation, scientific parameters, and upstream inputs.
Calling an identical stage reuses its completed result.
Changing `k` reuses normalization, PCA, and the ANN index but creates new neighbour and connectivity artifacts.
Changing the cell or feature selection invalidates all dependent stages.

Harmony fits between PCA and the ANN index:

```python
corrected = ds.run_harmony(["technical_batch"], pca)
corrected_index = ds.build_ann_index(corrected)
corrected_neighbors = ds.query_neighbors(corrected_index, k=21)
corrected_graph = ds.build_connectivity_map(corrected_neighbors)
```

Use {doc}`../concepts/provenance` to inspect complete lineage and {doc}`reuse_and_tracing` for reuse and invalidation patterns.
