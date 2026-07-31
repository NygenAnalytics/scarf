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
UMAP, densMAP, and t-SNE then turn the resulting graph into a two-dimensional
view. They are visual summaries, not alternative cluster assignments.

```{raw} html
<span id="clustering"></span>
```

Clustering guidance from the former combined page now lives in
{doc}`clustering`.

## Build a graph and inspect PCA

```{code-cell} ipython3
import matplotlib.pyplot as plt

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
if "I__hvgs" not in ds.RNA.feats.columns:
    ds.mark_hvgs(
        min_cells=20,
        top_n=500,
        show_plot=False,
    )
ds.run_normalization(feat_key="hvgs")
ds.run_pca(dims=15, show_elbow_plot=True)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()
ds.run_leiden_clustering(resolution=0.5)
```

PCA axes represent decreasing amounts of variation in the selected genes.
An elbow plot helps identify where additional axes contribute progressively
less variance, but it does not define one correct dimension count. Too few axes
can merge distinct populations; too many can restore technical variation and
noise. Compare graph connectivity, cluster stability, and marker coherence when
the choice is uncertain.

## Run UMAP

```{code-cell} ipython3
ds.run_umap(
    n_epochs=150,
    spread=5,
    min_dist=1,
    parallel=True,
    label="UMAP",
)
```

`min_dist` controls how tightly local groups can pack, while `spread` controls
the overall scale. These parameters change appearance without changing the
input graph. Use a fixed random seed and graph when comparing settings.

## Preserve local density with densMAP

```{code-cell} ipython3
ds.run_umap(
    n_epochs=150,
    spread=5,
    min_dist=1,
    parallel=True,
    use_density_map=True,
    label="densMAP",
)
```

densMAP adds a density-preservation objective. It is useful when relative local
density matters, but it does not turn plot area into a direct estimate of cell
frequency.

## Run graph-based t-SNE

Scarf's t-SNE consumes the same neighbourhood graph. It is not supported on
native Windows; use Linux or WSL.

```{code-cell} ipython3
ds.run_tsne(
    alpha=10,
    box_h=1,
    early_iter=250,
    max_iter=500,
    verbose=False,
)
```

## Compare layouts responsibly

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
layout_comparisons = (
    ("UMAP", "RNA_UMAP"),
    ("densMAP", "RNA_densMAP"),
    ("t-SNE", "RNA_tSNE"),
)
for index, (axis, (title, layout_key)) in enumerate(
    zip(axes, layout_comparisons, strict=True)
):
    ds.plots.embedding(
        layout_key=layout_key,
        color_by="RNA_leiden_cluster",
        legend_loc="right",
        show_legend=index == 2,
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

The three panels share cells, graph, and cluster labels. A useful comparison
asks whether local neighbours and known populations remain visible. Differences
in global orientation, distance, empty space, or apparent island size do not
demonstrate different biology. A layout that hides connected transitions or
separates obvious technical covariates needs further investigation.

```{raw} html
<span id="run-paris-clustering-and-inspect-the-tree"></span>
```

## Paris clustering and its tree

This material moved to {doc}`clustering`. That guide distinguishes graph
connectivity between groups from the Paris hierarchy and covers both adaptive
and fixed cuts.
