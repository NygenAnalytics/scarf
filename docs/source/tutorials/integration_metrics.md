---
description: Compare batch mixing and biological preservation on uncorrected, partial-PCA, and Harmony graphs.
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

(lisi_metrics)=
(integration_metrics)=

# Integration metrics

An integration method can improve batch mixing while erasing biological
structure. This guide measures both sides of that trade-off on an uncorrected
graph, a partial-PCA graph, and a Harmony graph. No single score determines
whether an integration is scientifically valid.

## Standalone setup

This section reconstructs the merged Kang datastore from
{doc}`data_integration` so the page can run independently.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="ERROR", progress=True)

repository = scarf.cytebase.connect("scarf_docs")
ctrl_path = repository.download_dataset(
    name="kang_15K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
stim_path = repository.download_dataset(
    name="kang_14K_ifnb-pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds_ctrl = scarf.DataStore(f"{ctrl_path}/data.zarr", nthreads=4)
ds_stim = scarf.DataStore(f"{stim_path}/data.zarr", nthreads=4)

merged_path = "scarf_datasets/kang_integration_metrics.zarr"
scarf.AssayMerge(
    zarr_path=merged_path,
    assays=[ds_ctrl.RNA, ds_stim.RNA],
    names=["ctrl", "stim"],
    merge_assay_name="RNA",
    prepend_text="orig",
    reset_cell_filter=False,
    source_column="sample_id",
    overwrite=True,
).dump()
ds = scarf.DataStore(merged_path, nthreads=4)
ds.mark_hvgs(
    min_cells=10,
    top_n=2000,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=False,
)
normalized = ds.run_normalization(feat_key="hvgs")
pca = ds.run_pca(normalized, dims=25)
```

The graph uses `k=21`, so LISI uses `perplexity=7`. A larger perplexity expects
more neighbours. Keep `k` and perplexity consistent when comparing methods.

## Define the metric panel

LISI measures label diversity around each cell. Batch LISI should increase when
sources mix, while cell-type LISI should remain low when annotated populations
remain distinct. iLISI and cLISI summarize those objectives on benchmark-style
scales. Graph connectivity measures whether each imported population remains
connected. ARI compares the new cluster partition with imported labels.
Graph-silhouette values inspect cluster separation and can expose collapse after
over-correction.

```{code-cell} ipython3
def evaluate_graph(cluster_label):
    lisi = ds.metric_lisi(
        label_columns=["sample_id", "orig_cluster_labels"],
        perplexity=7,
    )
    silhouette = ds.metric_graph_silhouette(
        res_label=cluster_label,
    )
    return {
        "batch LISI median": float(np.median(lisi["sample_id"])),
        "cell-type LISI median": float(
            np.median(lisi["orig_cluster_labels"])
        ),
        "iLISI": ds.metric_ilisi(
            batch_colname="sample_id",
            perplexity=7,
        ),
        "cLISI": ds.metric_clisi(
            label_colname="orig_cluster_labels",
            perplexity=7,
        ),
        "proportional mixing": (
            ds.metric_proportional_batch_mixing(
                label_colname="sample_id",
                perplexity=7,
            )
        ),
        "graph connectivity": ds.metric_graph_connectivity(
            label_colname="orig_cluster_labels",
        ),
        "label ARI": ds.metric_label_concordance(
            label_columns=[
                f"RNA_{cluster_label}",
                "orig_cluster_labels",
            ],
            metric="ari",
        ),
        "mean graph silhouette": float(np.nanmean(silhouette)),
    }
```

## Measure the uncorrected graph

```{code-cell} ipython3
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=21)
graph = ds.build_connectivity_map(neighbors)
ds.run_leiden_clustering(
    graph,
    resolution=1.0,
    label="metrics_uncorrected_clusters",
)
uncorrected_metrics = evaluate_graph(
    "metrics_uncorrected_clusters"
)
```

## Measure the partial-PCA graph

Partial PCA learns its loading basis from the control cells and projects every
active cell into that basis. The graph is evaluated without adding a layout.

```{code-cell} ipython3
ds.cells.insert(
    column_name="is_ctrl",
    values=ds.cells.fetch_all("sample_id") == "ctrl",
    overwrite=True,
)
pca_partial = ds.run_pca(
    normalized,
    dims=25,
    pca_cell_key="is_ctrl",
)
ann = ds.build_ann_index(pca_partial)
neighbors = ds.query_neighbors(ann, k=21)
partial_graph = ds.build_connectivity_map(neighbors)
ds.run_leiden_clustering(
    partial_graph,
    resolution=1.0,
    label="metrics_partial_clusters",
)
partial_metrics = evaluate_graph("metrics_partial_clusters")
```

## Measure the Harmony graph

```{code-cell} ipython3
corrected = ds.run_harmony(["sample_id"], pca)
ds.build_embedding_initialization(pca)
ann = ds.build_ann_index(corrected)
neighbors = ds.query_neighbors(ann, k=21)
harmony_graph = ds.build_connectivity_map(neighbors)
ds.run_umap(
    harmony_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="metrics_harmony_UMAP",
)
ds.run_leiden_clustering(
    harmony_graph,
    resolution=1.0,
    label="metrics_harmony_clusters",
)
harmony_metrics = evaluate_graph("metrics_harmony_clusters")
```

## Compare mixing and preservation

```{code-cell} ipython3
metric_frame = pd.DataFrame(
    {
        "uncorrected": uncorrected_metrics,
        "partial PCA": partial_metrics,
        "Harmony": harmony_metrics,
    }
).T
metric_frame.round(3)
```

Raw batch and cell-type LISI medians remain in the table because their ranges
depend on the number of labels. The figure compares normalized metrics. Mixing,
cLISI, connectivity, and ARI are best at one. Graph silhouette ranges from
minus one to one and is best at one.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
metric_frame[
    [
        "iLISI",
        "proportional mixing",
    ]
].plot.bar(ax=axes[0], title="Source mixing")
metric_frame[
    [
        "cLISI",
        "graph connectivity",
        "label ARI",
        "mean graph silhouette",
    ]
].plot.bar(ax=axes[1], title="Structure preservation")
for ax in axes:
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[1].set_ylim(-1, 1)
axes[1].axhline(0, color="0.5", linewidth=0.8)
fig.tight_layout()
```

The corrected graph should improve at least one mixing measure without a
substantial loss across every preservation measure. Absolute values depend on
labels, graph size, `k`, and perplexity. These Kang sources are confounded with
treatment, so a mixing improvement remains descriptive.

Per-cell LISI can reveal where mixing succeeds or fails even when a median looks
acceptable.

```{code-cell} ipython3
harmony_lisi = ds.metric_lisi(
    label_columns=["sample_id"],
    perplexity=7,
)["sample_id"]
ds.cells.insert(
    "harmony_sample_lisi",
    harmony_lisi,
    key="I",
    overwrite=True,
)
ds.plots.embedding(
    layout_key="RNA_metrics_harmony_UMAP",
    color_by="harmony_sample_lisi",
)
```

Do not compare LISI values across graphs built with different neighbourhood
sizes, treat imported labels as error-free ground truth, or optimize one metric
without inspecting the biological populations it is meant to preserve.

See the {doc}`../reference/api/integration` for exact metric signatures and
definitions.
