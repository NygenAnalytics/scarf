---
description: densMAP, tSNE, Paris clustering, and cluster trees as alternatives to the default UMAP and Leiden path.
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

# Dimensionality reduction and clustering

The recommended path in {doc}`scrna_seq` uses UMAP and Leiden. This page covers densMAP,
graph-based tSNE, Paris hierarchical clustering, and cluster trees.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Python 3.12 or newer

## What you will learn

- Build a graph with normalization, PCA, and KNN search
- Compare UMAP, densMAP, and tSNE from that graph
- Inspect Leiden groups with a Paris cluster tree

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

scarf.fetch_dataset(
    'tenx_5K_pbmc_rnaseq',
    save_path='scarf_datasets',
    as_zarr=True,
)
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
ds.run_umap(n_epochs=150, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)
```

## Guided steps

### 1. Run densMAP

`make_graph` normalizes the selected features, calculates PCA coordinates, and builds the KNN
graph used by every step below. Enable density-preserving UMAP when local density structure
matters.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=150,
    spread=5,
    min_dist=1,
    parallel=True,
    use_density_map=True,
    label='densMAP',
)
ds.plots.embedding(layout_key='RNA_densMAP')
```

### 2. Run tSNE

```{note}
Scarf's tSNE runs on the same neighbourhood graph as UMAP. It is not supported on native
Windows. Use a Linux environment such as WSL.
```

```{code-cell} ipython3
ds.run_tsne(
    alpha=10,
    box_h=1,
    early_iter=250,
    max_iter=500,
)
ds.plots.embedding(layout_key='RNA_tSNE')
```

### 3. Run Paris clustering and inspect the tree

Paris builds a hierarchical dendrogram that can be cut to a chosen number of clusters.
`run_clustering` is the Paris entrypoint. Cluster relationships can be drawn with
`ds.plots.cluster_tree`.

```{code-cell} ipython3
n_clusters = int(
    ds.cells.to_pandas_dataframe(
        columns=['RNA_leiden_cluster'], key='I'
    )['RNA_leiden_cluster'].nunique()
)
ds.run_clustering(n_clusters=n_clusters)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_cluster',
)
```

```{code-cell} ipython3
ds.plots.cluster_tree(cluster_key='RNA_cluster', width=1)
```

## Common mistakes

- Comparing UMAP and tSNE as if they used different input graphs (in Scarf they share the graph)
- Expecting Paris cluster IDs to match Leiden IDs one-to-one
- Running tSNE on Windows without handling the platform limitation

## Saved results

UMAP, densMAP, and tSNE coordinates are written to `RNA_UMAP*`, `RNA_densMAP*`, and
`RNA_tSNE*`. Leiden labels are stored in `RNA_leiden_cluster`; Paris labels are stored in
`RNA_cluster`, and its dendrogram is saved with the graph.

## Next steps

- {doc}`scrna_seq`
- {doc}`annotation`
- {doc}`plotting`
