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
import numpy as np
import pandas as pd

import scarf
import scarf.plotting as splt

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
pca_refs = {}
for dimensions in (10, 30, 15):
    pca = ds.run_pca(
        normalized,
        dims=dimensions,
        show_elbow_plot=dimensions == 30,
    )
    pca_refs[dimensions] = pca
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
The elbow plot above is for the 30-component fit. An early bend means later
axes add less variance each; it does not force a stop at the marked component,
and it does not make 10, 15, and 30 interchangeable. The cumulative shares
below place those cutoffs on the same 30-component fit. Too few axes can merge
distinct populations; too many can restore technical variation and noise.
Compare graph connectivity, cluster stability, and marker coherence when the
choice is uncertain.

```{code-cell} ipython3
scores_30 = np.asarray(ds.load_artifact(pca_refs[30])["data"])
retained_share = scores_30.var(axis=0, ddof=1)
retained_share = 100.0 * retained_share / retained_share.sum()
cumulative_share = np.cumsum(retained_share)
pd.Series(
    {
        dimensions: float(cumulative_share[dimensions - 1])
        for dimensions in (10, 15, 30)
    },
    name="cumulative_share_of_30pc_fit",
).rename_axis("pca_dimensions")
```

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

Similar cluster counts can still hide size flips. Per-cluster sizes show whether
an extra group is a real split or a tiny fragment.

```{code-cell} ipython3
cluster_sizes = pd.DataFrame(
    {
        dimensions: pd.Series(
            ds.cells.fetch(cluster_key, key="I")
        ).value_counts()
        for dimensions, cluster_key in cluster_keys.items()
    }
).fillna(0).astype(int)
cluster_sizes
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

The layout below uses the active 15-component graph. Colouring by each Leiden
partition shows how the 10-, 15-, and 30-component cuts land on the same
coordinates.

```{code-cell} ipython3
pca_partition_view = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[cluster_keys[dimensions] for dimensions in (10, 15, 30)],
    n_columns=3,
    legend_loc="on_data",
    show_titles=False,
    show=False,
)
for axis, dimensions in zip(
    pca_partition_view.axes.values(),
    (10, 15, 30),
    strict=True,
):
    axis.set_title(f"Leiden on {dimensions} PCs")
pca_partition_view.figure.set_size_inches(12, 4)
pca_partition_view.figure
```

`min_dist` controls how tightly local groups can pack, while `spread` controls
the overall scale. These parameters change appearance without changing the
input graph. A second UMAP with a smaller `min_dist` shows packing on the same
neighbours.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=150,
    spread=5,
    min_dist=0.1,
    parallel=True,
    label="UMAP_tight",
)
```

```{code-cell} ipython3
min_dist_view = ds.plots.embedding(
    layout_key=["RNA_UMAP", "RNA_UMAP_tight"],
    color_by=cluster_keys[15],
    n_columns=2,
    legend_loc="on_data",
    show_titles=False,
    show=False,
)
for axis, title in zip(
    min_dist_view.axes.values(),
    ("min_dist=1", "min_dist=0.1"),
    strict=True,
):
    axis.set_title(title)
min_dist_view.figure.set_size_inches(10, 4)
min_dist_view.figure
```

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

densMAP adds a density-preservation objective. Contours below contrast local
cell density on the same cells under UMAP and densMAP. Relative packing can
differ; plot area is still not a direct estimate of cell frequency.

```{code-cell} ipython3
density_contrast = ds.plots.embedding(
    layout_key=["RNA_UMAP", "RNA_densMAP"],
    color_by=None,
    n_columns=2,
    density_overlay=splt.DensityOverlay(statistic="density"),
    show_legend=False,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    density_contrast.axes.values(),
    ("UMAP", "densMAP"),
    strict=True,
):
    axis.set_title(title)
density_contrast.figure.set_size_inches(10, 4)
density_contrast.figure
```

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
