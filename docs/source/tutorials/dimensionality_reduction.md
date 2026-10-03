---
description: Choose PCA dimensions and compare UMAP, densMAP, and t-SNE without over-interpreting layouts.
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

(dimensionality_reduction_and_clustering)=

# Choosing dimensionality reductions

PCA compresses selected features into the coordinates used to find neighbours.
UMAP, densMAP, and t-SNE then turn the resulting graph into a two-dimensional view.
They are visual summaries, not alternative cluster assignments.

```{raw} html
<span id="clustering"></span>
```

Clustering guidance from the former combined page now lives in {doc}`clustering`.

## 1. Standalone setup

```{code-cell} ipython3
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
baseline = ds.pipeline.open(label="docs_default")
normalized = baseline["normalized"]
umap = baseline["umap"]
```

The saved analysis used 15 PCs; Scarf's current default is 21. We keep the example's cells,
genes, and normalization fixed so we can explore what changing the number of PCs does.

```{code-cell} ipython3
ds.plots.embedding(run=baseline, color_by="clusters")
```

## 2. Compare PCA dimension counts

Build each candidate from the same normalized data and cluster each graph by passing it explicitly.
Retain the 15-component graph and initialization for the layout comparisons below.

```{code-cell} ipython3
dimension_counts = (10, 15, 30)
graph_15 = baseline["connectivity_map"]
cluster_refs = {15: baseline["leiden_0.5"]}
for dimensions in (10, 30):
    pca = ds.run_pca(
        normalized,
        dims=dimensions,
        show_elbow_plot=dimensions == 30,
    )
    ann = ds.build_ann_index(pca)
    neighbors = ds.query_neighbors(ann, k=11)
    graph = ds.build_connectivity_map(neighbors)
    cluster_refs[dimensions] = ds.run_leiden_clustering(
        graph,
        resolution=0.5,
    )

initialization_15 = baseline["embedding_initialization"]
cluster_values = {
    dimensions: np.asarray(ds.load_artifact(cluster_refs[dimensions])["values"][:])
    for dimensions in dimension_counts
}
```

PCA axes represent decreasing amounts of variation in the selected genes.
An early bend in the elbow plot means later axes each add less variance. It suggests a range to
investigate, rather than a single correct cutoff. Scarf fits one extra component for this plot;
when reusing a saved PCA result, it may warn that the plot is unavailable.
Too few axes can merge distinct populations; too many can restore technical variation and noise.
Compare graph connectivity, cluster stability, and marker coherence when the choice is uncertain.

Compare cluster sizes as well as the number of clusters. Cluster numbers can change between
analyses, so matching row numbers do not necessarily identify the same cells.

```{code-cell} ipython3
pd.DataFrame(
    {
        dimensions: pd.Series(cluster_values[dimensions]).value_counts()
        for dimensions in dimension_counts
    }
).fillna(0).astype(int)
```

```{code-cell} ipython3
pd.Series(
    {
        f"{first} vs {second} PCs": ds.metric_label_concordance(
            cluster_refs[first], cluster_refs[second]
        )
        for first, second in combinations(dimension_counts, 2)
    },
    name="adjusted Rand index",
)
```

The adjusted Rand index measures partition agreement without requiring matching cluster numbers.
It does not identify the biologically correct dimension count. Inspect markers and QC metrics
where the partitions disagree. See {doc}`clustering` for more on cluster evidence.

## 3. Compare UMAP packing

The layout below uses the explicit 15-component graph.
Colouring by each Leiden partition shows how the 10-, 15-, and 30-component cuts land on the same coordinates.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, dimensions in zip(
    axes,
    dimension_counts,
    strict=True,
):
    ds.plots.embedding(
        layout=umap,
        color_by=cluster_refs[dimensions],
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(f"Leiden on {dimensions} PCs")
figure.tight_layout()
figure
```

`min_dist` controls how tightly local groups can pack, while `spread` controls the overall scale.
These parameters change appearance without changing the input graph.
A second UMAP with a smaller `min_dist` shows packing on the same neighbours.

```{code-cell} ipython3
umap_tight = ds.run_umap(graph_15, initialization_15, min_dist=0.1)
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, layout, title in zip(
    axes,
    (umap, umap_tight),
    ("min_dist=1", "min_dist=0.1"),
    strict=True,
):
    ds.plots.embedding(
        layout=layout,
        color_by=cluster_refs[15],
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

## Optional: compare densMAP and t-SNE

```{code-cell} ipython3
densmap = ds.run_umap(graph_15, initialization_15, use_density_map=True)
```

densMAP adds a density-preservation objective.
Relative packing can differ from UMAP; plot area is still not a direct estimate of cell frequency.

Scarf's t-SNE consumes the same neighbourhood graph.
Computing a new embedding requires `sys.platform` in `posix` or `linux`; macOS (`darwin`) and Windows are unsupported.

```{code-cell} ipython3
tsne = ds.run_tsne(graph_15, initialization_15, verbose=False)
```

### Read the layouts cautiously

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
layout_comparisons = (
    ("UMAP", umap),
    ("densMAP", densmap),
    ("t-SNE", tsne),
)
for axis, (title, layout) in zip(
    axes,
    layout_comparisons,
    strict=True,
):
    ds.plots.embedding(
        layout=layout,
        color_by=cluster_refs[15],
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

The three panels share cells, graph, and cluster labels.
A useful comparison asks whether local neighbours and known populations remain visible.
Differences in global orientation, distance, empty space, or apparent island size do not demonstrate different biology.
A layout that hides connected transitions or separates obvious technical covariates needs further investigation.

```{raw} html
<span id="run-paris-clustering-and-inspect-the-tree"></span>
```

## Next: clustering

This material moved to {doc}`clustering`.
That guide distinguishes graph connectivity between groups from the Paris hierarchy and covers both adaptive and fixed cuts.
