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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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

The rebuilt store's `docs_default` run freezes the cell selection, feature selection, and
normalization used here. Each dimensionality branch therefore changes only PCA and its downstream
artifacts.

## 2. Compare PCA dimension counts

Build each candidate from the same normalized data and cluster each graph by passing it explicitly.
Retain the 15-component graph and initialization for the layout comparisons below.

```{code-cell} ipython3
dimension_counts = (10, 15, 30)
pca_refs = {15: baseline["pca"]}
graph_15 = baseline["connectivity_map"]
cluster_refs = {15: baseline["leiden_0.5"]}
for dimensions in (10, 30):
    pca = ds.run_pca(
        normalized,
        dims=dimensions,
        show_elbow_plot=dimensions == 30,
    )
    pca_refs[dimensions] = pca
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
The elbow plot uses 31 explained-variance ratios from a dims+1 fit (`dims=30`), not a scree of only the 30 kept components.
On artifact reuse the fit is unavailable, so Scarf may warn and skip the plot.
An early bend means later axes add less variance each; it does not force a stop at the marked component, and it does not make 10, 15, and 30 interchangeable.
The cumulative shares below place those cutoffs on the same 30 kept components.
Too few axes can merge distinct populations; too many can restore technical variation and noise.
Compare graph connectivity, cluster stability, and marker coherence when the choice is uncertain.

```{code-cell} ipython3
scores_30 = np.asarray(ds.load_artifact(pca_refs[30])["data"])
retained_share = scores_30.var(axis=0, ddof=1)
retained_share = 100.0 * retained_share / retained_share.sum()
cumulative_share = np.cumsum(retained_share)
pd.Series(
    {
        dimensions: float(cumulative_share[dimensions - 1])
        for dimensions in dimension_counts
    },
    name="cumulative_share_of_30pc_fit",
).rename_axis("pca_dimensions")
```

```{code-cell} ipython3
pd.Series(
    {
        dimensions: pd.Series(cluster_values[dimensions]).nunique()
        for dimensions in dimension_counts
    },
    name="n_clusters",
).rename_axis("pca_dimensions")
```

Similar cluster counts can still hide size flips.
Per-cluster sizes show whether an extra group is a real split or a tiny fragment.

```{code-cell} ipython3
pd.DataFrame(
    {
        dimensions: pd.Series(cluster_values[dimensions]).value_counts()
        for dimensions in dimension_counts
    }
).fillna(0).astype(int)
```

```{code-cell} ipython3
agreement_rows = []
for first, second in combinations(dimension_counts, 2):
    agreement_rows.append(
        {
            "comparison": f"{first} vs {second} dimensions",
            "ARI": adjusted_rand_score(cluster_values[first], cluster_values[second]),
            "NMI": normalized_mutual_info_score(
                cluster_values[first],
                cluster_values[second],
            ),
        }
    )
pd.DataFrame(agreement_rows)
```

ARI and NMI measure agreement between partitions but do not identify the biologically correct dimension count.
Investigate a low-agreement arm through markers, technical covariates, and graph diagnostics before choosing it or discarding it.

## 3. Compare UMAP packing

The layout below uses the explicit 15-component graph.
Colouring by each Leiden partition shows how the 10-, 15-, and 30-component cuts land on the same coordinates.

```{code-cell} ipython3
umap_values = np.asarray(ds.load_artifact(umap)["values"][:])
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, dimensions in zip(
    axes,
    dimension_counts,
    strict=True,
):
    axis.scatter(
        umap_values[:, 0],
        umap_values[:, 1],
        c=cluster_values[dimensions],
        s=3,
        cmap="tab20",
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
umap_tight_values = np.asarray(ds.load_artifact(umap_tight)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, coordinates, title in zip(
    axes,
    (umap_values, umap_tight_values),
    ("min_dist=1", "min_dist=0.1"),
    strict=True,
):
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=cluster_values[15],
        s=3,
        cmap="tab20",
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

## 4. Preserve local density with densMAP

```{code-cell} ipython3
densmap = ds.run_umap(graph_15, initialization_15, use_density_map=True)
```

densMAP adds a density-preservation objective.
Contours below contrast local cell density on the same cells under UMAP and densMAP.
Relative packing can differ; plot area is still not a direct estimate of cell frequency.

```{code-cell} ipython3
dense_values = np.asarray(ds.load_artifact(densmap)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, coordinates, title in zip(
    axes,
    (umap_values, dense_values),
    ("UMAP", "densMAP"),
    strict=True,
):
    axis.hexbin(coordinates[:, 0], coordinates[:, 1], gridsize=50)
    axis.set_title(title)
figure.tight_layout()
figure
```

## 5. Run graph-based t-SNE

Scarf's t-SNE consumes the same neighbourhood graph.
Computing a new embedding requires `sys.platform` in `posix` or `linux`; macOS (`darwin`) and Windows are unsupported.

```{code-cell} ipython3
tsne = ds.run_tsne(graph_15, initialization_15, verbose=False)
```

## 6. Compare layouts responsibly

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
tsne_values = np.asarray(ds.load_artifact(tsne)["values"][:])
layout_comparisons = (
    ("UMAP", umap_values),
    ("densMAP", dense_values),
    ("t-SNE", tsne_values),
)
for axis, (title, coordinates) in zip(
    axes,
    layout_comparisons,
    strict=True,
):
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=cluster_values[15],
        s=3,
        cmap="tab20",
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

## 7. Paris clustering and its tree

This material moved to {doc}`clustering`.
That guide distinguishes graph connectivity between groups from the Paris hierarchy and covers both adaptive and fixed cuts.
