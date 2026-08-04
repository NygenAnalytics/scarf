---
description: Compare Leiden resolutions and Paris cuts, validate partitions, and recluster a selected population.
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

# Choosing and validating clusters

Clustering turns a neighbourhood graph into a discrete partition. It does not
reveal one universally optimal number of cell types. A useful partition should
be stable enough for the question, preserve known biology, avoid groups driven
only by technical covariates, and support interpretable markers.

This guide compares Leiden resolutions and Paris cuts with several diagnostics.

## Build the shared graph

```{code-cell} ipython3
from dataclasses import asdict
from itertools import combinations

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
ds.run_normalization(feat_key="hvgs")
pca = ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()
ds.run_umap(
    n_epochs=150,
    spread=5,
    min_dist=1,
    parallel=True,
)
```

Every partition below consumes this same graph. That keeps differences in
cluster labels separate from differences caused by changing features, PCA, or
neighbours.

## Sweep Leiden resolution

Higher Leiden resolution usually produces more and smaller groups. Keep each
candidate under a distinct label.

```{code-cell} ipython3
leiden_candidates = {
    0.3: "leiden_r03",
    0.5: "leiden_r05",
    0.8: "leiden_r08",
}
for resolution, label in leiden_candidates.items():
    ds.run_leiden_clustering(
        resolution=resolution,
        label=label,
    )
chosen_key = "RNA_leiden_r05"
```

```{code-cell} ipython3
cluster_sizes = pd.DataFrame(
    {
        label: pd.Series(
            ds.cells.fetch(f"RNA_{label}", key="I")
        ).value_counts()
        for label in leiden_candidates.values()
    }
).fillna(0).astype(int)
cluster_sizes
```

```{code-cell} ipython3
leiden_comparison = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[f"RNA_{label}" for label in leiden_candidates.values()],
    n_columns=3,
    legend_loc="on_data",
    show_titles=False,
    show=False,
)
for axis, resolution in zip(
    leiden_comparison.axes.values(), leiden_candidates, strict=True
):
    axis.set_title(f"Leiden resolution {resolution}")
leiden_comparison.figure
```

Reject a resolution when new groups are tiny, spatially diffuse, dominated by
low-count cells, or unsupported by markers. A coarse partition can still be
appropriate when the question concerns broad lineages.

Check whether small groups in the working partition are QC outliers before
treating them as cell types.

```{code-cell} ipython3
chosen_sizes = (
    pd.Series(ds.cells.fetch(chosen_key, key="I"))
    .value_counts()
    .sort_values()
)
chosen_sizes
```

```{code-cell} ipython3
ds.plots.distribution(
    keys=["RNA_nCounts", "RNA_percentMito"],
    group_by=chosen_key,
    kind="violin",
)
```

Tiny groups that also sit at low `RNA_nCounts` or high `RNA_percentMito` need
QC review. Marker support for the smallest group is checked after
`run_marker_search` below.

## Compare partitions and separation

ARI and NMI quantify agreement between pairs of partitions. They do not say
which partition is biologically correct. Graph silhouette asks whether each
group is closer to itself than to neighbouring groups and can penalize
transitional populations that are biologically real.

```{code-cell} ipython3
agreement_rows = []
for first, second in combinations(leiden_candidates.values(), 2):
    columns = [f"RNA_{first}", f"RNA_{second}"]
    agreement_rows.append(
        {
            "comparison": f"{first} vs {second}",
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

```{code-cell} ipython3
silhouette_rows = []
for label in leiden_candidates.values():
    scores = ds.metric_graph_silhouette(res_label=label)
    silhouette_rows.append(
        {
            "partition": label,
            "mean graph silhouette": float(np.nanmean(scores)),
            "minimum graph silhouette": float(np.nanmin(scores)),
        }
    )
pd.DataFrame(silhouette_rows)
```

Per-cluster graph silhouette for the working partition, then the same scores
mapped onto the UMAP so weak groups are visible in embedding space.

```{code-cell} ipython3
graph_scores = ds.metric_graph_silhouette(res_label="leiden_r05")
clusters_all = ds.cells.fetch_all(chosen_key)
active_mask = ds.cells.fetch_all("I").astype(bool)
active_labels = pd.Series(clusters_all[active_mask])
categories = pd.Categorical(active_labels).categories
per_cluster_silhouette = pd.DataFrame(
    {
        "cluster": list(categories),
        "graph_silhouette": graph_scores,
        "n_cells": active_labels.value_counts()
        .reindex(categories)
        .to_numpy(),
    }
)
per_cluster_silhouette
```

```{code-cell} ipython3
score_map = dict(zip(categories, graph_scores, strict=True))
mapped_silhouette = np.full(len(clusters_all), np.nan, dtype=float)
mapped_silhouette[active_mask] = (
    active_labels.map(score_map).to_numpy(dtype=float)
)
ds.cells.insert(
    "graph_silhouette",
    mapped_silhouette,
    overwrite=True,
)
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="graph_silhouette",
)
```

Coordinate separability asks whether the PCA values can recover each partition.
Macro F1 gives each cluster equal influence, weighted F1 follows cluster size,
and coordinate silhouette compares within-cluster and between-cluster
distances.

```{code-cell} ipython3
separability = ds.metric_cluster_separability(
    pca,
    cluster_columns=[
        f"RNA_{label}" for label in leiden_candidates.values()
    ],
)
separability.clustering_scores
```

Per-cluster F1 and the confusion table show which groups drive a weak
aggregate score.

```{code-cell} ipython3
separability.cluster_scores.query(
    "clustering == @chosen_key"
).sort_values("f1_score")
```

```{code-cell} ipython3
separability.confusion.query("clustering == @chosen_key")
```

Graph silhouette evaluates the neighbourhood graph, while these scores evaluate
the PCA coordinates. The labels were derived from a graph built from the same
PCA, so strong separability supports internal coherence but is not independent
biological validation.

The standard pipeline compares several Leiden resolutions and uses PCA
silhouette as one heuristic when it needs a partition for markers or doublet
scoring. Treat that automatic choice as a starting point, not a declared
optimum.

## Inspect per-cell membership strength

Membership strength is the fraction of a cell's graph neighbours assigned to
its most common cluster. Low values often occur near boundaries or in weakly
supported groups.

```{code-cell} ipython3
ds.calc_membership_strength(
    "RNA",
    "I",
    "hvgs",
    chosen_key,
)
membership_key = "RNA_I_cluster_membership_strength"
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=membership_key,
)
ds.plots.distribution(
    keys=membership_key,
    group_by=chosen_key,
    kind="violin",
)
```

Low membership throughout one cluster suggests that its boundary is not well
supported. A few low values between otherwise coherent groups can instead
represent continuous biology.

## Inspect graph connectivity between groups

```{code-cell} ipython3
ds.plots.cluster_connectivity(
    group_by=chosen_key,
    layout_key="RNA_UMAP",
    from_assay="RNA",
    feat_key="hvgs",
    show_cells=True,
)
```

This view summarizes weighted graph connections between groups. It is not a
lineage graph and is distinct from the Paris hierarchy below.

## Compare Paris adaptive and fixed cuts

Paris builds a hierarchy by repeatedly contracting the graph. The automatic cut
retains persistent branches subject to a minimum size. A fixed cut requests an
exact cluster count from the same hierarchy.

```{code-cell} ipython3
paris_auto = ds.run_paris_clustering(
    n_clusters="auto",
    label="paris_auto",
)
pd.DataFrame(
    [asdict(item) for item in paris_auto.diagnostics]
)[
    [
        "label",
        "size",
        "persistence",
        "decision_margin",
        "forced",
    ]
]
```

Persistence describes the resolution interval over which a selected branch
survives. The decision margin measures how strongly the adaptive objective
favoured retaining that branch. Forced groups satisfy structural constraints
and should not be interpreted as strongly supported solely from that flag.

```{code-cell} ipython3
paris_fixed = ds.run_paris_clustering(
    n_clusters=paris_auto.n_clusters,
    label="paris_fixed",
)
ds.metric_label_concordance(
    [paris_auto.label_key, paris_fixed.label_key],
    metric="ari",
)
```

```{code-cell} ipython3
paris_comparison = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[paris_auto.label_key, paris_fixed.label_key],
    n_columns=2,
    legend_loc="on_data",
    show_titles=False,
    show=False,
)
for axis, title in zip(
    paris_comparison.axes.values(),
    ["Paris auto cut", "Paris fixed cut"],
    strict=True,
):
    axis.set_title(title)
paris_comparison.figure
```

```{code-cell} ipython3
ds.plots.cluster_tree(
    cluster_key=paris_auto.label_key,
    width=1.7,
    fontsize=9,
    figsize=(7, 5),
)
```

The tree represents hierarchical merges, while
`plots.cluster_connectivity` represents direct inter-group graph structure.

Optionally place Leiden and Paris on the same embedding frame, and colour the
Paris tree by Leiden membership.

```{code-cell} ipython3
leiden_paris = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[chosen_key, paris_auto.label_key],
    n_columns=2,
    legend_loc="on_data",
    show_titles=False,
    show=False,
)
for axis, title in zip(
    leiden_paris.axes.values(),
    ["Leiden (chosen)", "Paris auto"],
    strict=True,
):
    axis.set_title(title)
print(
    "ARI:",
    ds.metric_label_concordance(
        [chosen_key, paris_auto.label_key],
        metric="ari",
    ),
)
leiden_paris.figure
```

```{code-cell} ipython3
ds.plots.cluster_tree(
    cluster_key=paris_auto.label_key,
    fill_by_value=chosen_key,
    width=1.7,
    fontsize=9,
    figsize=(7, 5),
)
```

## Review marker specificity

```{code-cell} ipython3
ds.run_marker_search(group_key=chosen_key)
largest_group = chosen_sizes.index[-1]
smallest_group = chosen_sizes.index[0]
markers = ds.get_markers(
    group_key=chosen_key,
    group_id=largest_group,
)
print(f"Largest group: {largest_group} (n={int(chosen_sizes.loc[largest_group])})")
markers[
    [
        "feature_name",
        "score",
        "auc",
        "p_value",
        "p_value_adjusted",
    ]
].head(10)
```

```{code-cell} ipython3
small_markers = ds.get_markers(
    group_key=chosen_key,
    group_id=smallest_group,
)
print(
    f"Smallest group: {smallest_group} "
    f"(n={int(chosen_sizes.loc[smallest_group])})"
)
small_markers[
    [
        "feature_name",
        "score",
        "auc",
        "p_value",
        "p_value_adjusted",
    ]
].head(10)
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key=chosen_key,
    topn=5,
    figsize=(5, 9),
)
```

```{code-cell} ipython3
top_genes = markers["feature_name"].astype(str).head(3).tolist()
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=top_genes,
    n_columns=3,
    sort_values=True,
)
```

AUC measures how often a randomly selected cell in the group has a higher value
than a cell outside it. Review AUC, specificity score, expression fraction, and
known biology together. Adjusted p-values are within-group cell-level tests, not
replicate-aware differential expression. Groups with fewer than two target or
reference cells fail marker testing rather than returning a misleading table.

## Subset and recluster

Subclustering should rebuild feature selection and the graph within the selected
population. Reusing the full-dataset graph can preserve boundaries that are
irrelevant inside the subset.

Highlight the focus population on the full embedding before rebuilding the
subset graph.

```{code-cell} ipython3
clusters = ds.cells.fetch_all(chosen_key)
active = ds.cells.fetch_all("I").astype(bool)
focus = pd.Series(clusters[active]).value_counts().index[0]
ds.cells.insert(
    "focus_cells",
    active & (clusters == focus),
    overwrite=True,
)
print(f"Focus cluster: {focus}")
print(f"Focus cells: {int(ds.cells.fetch_all('focus_cells').sum())}")
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=None,
    default_color="#bdbdbd",
    point_alpha=0.4,
    highlight=splt.Highlight(
        by="focus_cells",
        groups=(True,),
        color="#d62728",
        dim_alpha=0.12,
        size_multiplier=1.35,
        halo_width=0.4,
    ),
)
```

```{code-cell} ipython3
ds.mark_hvgs(
    cell_key="focus_cells",
    min_cells=10,
    top_n=500,
    show_plot=False,
)
ds.run_normalization(
    cell_key="focus_cells",
    feat_key="hvgs",
)
ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()
ds.run_umap(
    cell_key="focus_cells",
    n_epochs=100,
    spread=5,
    min_dist=1,
    parallel=True,
    label="UMAP",
)
ds.run_leiden_clustering(
    cell_key="focus_cells",
    resolution=0.4,
    label="leiden_cluster",
)
pd.Series(
    ds.cells.fetch("RNA_focus_cells_leiden_cluster", key="focus_cells")
).value_counts()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_focus_cells_UMAP",
    color_by="RNA_focus_cells_leiden_cluster",
    cell_key="focus_cells",
)
```

The resulting columns apply only to `focus_cells`. Revisit QC, feature
selection, marker support, and stability before assigning finer biological
labels.
