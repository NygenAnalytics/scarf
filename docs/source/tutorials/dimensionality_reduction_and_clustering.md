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
- Inspect Paris groups with a Paris cluster tree

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
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

# HVG columns are stored as `{cell_key}__{hvg_key_name}` (here `I__hvgs`)
if 'I__hvgs' not in ds.RNA.feats.columns:
    ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
normalized = ds.run_normalization(feat_key='hvgs')
pca = ds.run_pca(normalized, dims=15, show_elbow_plot=True)
ds.build_embedding_initialization(pca)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
ds.build_connectivity_map(neighbors)
# Always run layout and clustering on this page so prepared stores still demonstrate the APIs
ds.run_umap(n_epochs=150, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)
```

`show_elbow_plot=True` on `run_pca` plots explained variance and marks a detected elbow.
Use it when choosing `dims`, not as a unique truth for the number of components.


Inspect the loaded KNN graph degree and edge-weight distributions:

```{code-cell} ipython3
import scarf.plotting as splt

graph = ds.load_graph(
    from_assay='RNA',
    cell_key='I',
    feat_key='hvgs',
    symmetric=False,
    upper_only=False,
)
splt.graph_qc(graph)
```

## Guided steps

### 1. Run densMAP

The setup above built the KNN graph used by every step below. Enable density-preserving
UMAP when local density structure matters.


```{code-cell} ipython3
ds.run_umap(
    n_epochs=150,
    spread=5,
    min_dist=1,
    parallel=True,
    use_density_map=True,
    label='densMAP',
)
```

densMAP keeps local density structure that standard UMAP may flatten.

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
```

tSNE uses the same KNN graph; cluster separation can look sharper than on UMAP.

### 3. Compare layouts side by side

```{code-cell} ipython3
layout_keys = [
    key for key in ['RNA_UMAP', 'RNA_densMAP', 'RNA_tSNE']
    if f'{key}1' in ds.cells.columns
]
ds.plots.embedding(
    layout_key=layout_keys,
    color_by='RNA_leiden_cluster',
    n_columns=len(layout_keys),
)
```

The three layouts share the same neighbourhood graph and Leiden labels; differences are
layout geometry, not a second clustering.

### 4. Run Paris clustering and inspect the tree

Paris builds a hierarchical dendrogram. `run_paris_clustering` selects a
branch-adaptive cut by default, or accepts an integer cluster count for a fixed cut.
The automatic cut keeps persistent branches only when splitting them also adds
topological structure beyond a degree-preserving null model. Its minimum cluster size
defaults to the graph's `k + 1` and can be set with `min_cluster_size`.
Cluster relationships can be drawn with `ds.plots.cluster_tree`.

```{code-cell} ipython3
paris = ds.run_paris_clustering()
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=paris.label_key,
)
```

Paris labels (`RNA_paris_cluster`) need not match Leiden IDs one-to-one.

```{code-cell} ipython3
ds.plots.cluster_tree(cluster_key='RNA_paris_cluster', width=1)
```

Internal nodes are successive merges of Paris clusters; nearby leaves share a longer
common branch and are more closely related in the hierarchical cut.

## Common mistakes

- Comparing UMAP and tSNE as if they used different input graphs (in Scarf they share the graph)
- Expecting Paris cluster IDs to match Leiden IDs one-to-one
- Running tSNE on Windows without handling the platform limitation

## Saved results

UMAP, densMAP, and tSNE coordinates are written to `RNA_UMAP*`, `RNA_densMAP*`, and
`RNA_tSNE*`. Leiden labels are stored in `RNA_leiden_cluster`; Paris labels are stored in
`RNA_paris_cluster`, and its hierarchy is saved with the graph.

## Further reading

- [UMAP documentation](https://umap-learn.readthedocs.io/)
- [Scanpy clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)
- [Single-cell best practices: clustering](https://www.sc-best-practices.org/cellular_structure/clustering.html)

## Next steps

- {doc}`annotation`
- {doc}`plotting`
- {doc}`scrna_seq`
