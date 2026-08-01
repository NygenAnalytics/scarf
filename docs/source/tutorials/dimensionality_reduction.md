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

## Standalone setup

```{code-cell} ipython3
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=True)

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
normalized = ds.run_normalization(feat_key="hvgs")
```

This section reconstructs the selected cells, features, and normalization from
{doc}`scrna_seq` so the page can run independently.

## Compare PCA dimension counts

Build each candidate from the same normalized data and cluster each graph by
passing it explicitly. The 15-component chain runs last, so it is the one left
in the assay's {term}`analysis chain`, and the layout comparisons below pick it
up when no graph is named.

```{code-cell} ipython3
cluster_keys = {}
for dimensions in (10, 30, 15):
    pca = ds.run_pca(
        normalized,
        dims=dimensions,
        show_elbow_plot=dimensions == 30,
    )
    if dimensions == 15:
        ds.build_embedding_initialization(pca)
    ann = ds.build_ann_index(pca)
    neighbors = ds.query_neighbors(ann, k=11)
    graph = ds.build_connectivity_map(neighbors)

    label = f"leiden_pca_{dimensions}"
    ds.run_leiden_clustering(
        graph,
        resolution=0.5,
        label=label,
    )
    cluster_keys[dimensions] = f"RNA_{label}"
```

PCA axes represent decreasing amounts of variation in the selected genes.
An elbow plot helps identify where additional axes contribute progressively
less variance, but it does not define one correct dimension count. Too few axes
can merge distinct populations; too many can restore technical variation and
noise. Compare graph connectivity, cluster stability, and marker coherence when
the choice is uncertain.

```{code-cell} ipython3
cluster_counts = pd.Series(
    {
        dimensions: pd.Series(
            ds.cells.fetch(cluster_key, key="I")
        ).nunique()
        for dimensions, cluster_key in cluster_keys.items()
    },
    name="n_clusters",
).rename_axis("pca_dimensions")
cluster_counts
```

```{code-cell} ipython3
agreement_rows = []
for first, second in combinations(cluster_keys, 2):
    columns = [cluster_keys[first], cluster_keys[second]]
    agreement_rows.append(
        {
            "comparison": f"{first} vs {second} dimensions",
            "ARI": ds.metric_label_concordance(
                columns,
                metric="ari",
            ),
            "NMI": ds.metric_label_concordance(
                columns,
                metric="nmi",
            ),
        }
    )
pd.DataFrame(agreement_rows)
```

ARI and NMI measure agreement between partitions but do not identify the
biologically correct dimension count. Investigate a low-agreement arm through
markers, technical covariates, and graph diagnostics before choosing it or
discarding it.

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
        color_by=cluster_keys[15],
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
