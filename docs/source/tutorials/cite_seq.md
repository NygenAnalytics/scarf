---
description: Build RNA and ADT graphs and integrate them with immutable SNN and WNN artifacts.
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

(multimodal_integration)=

# CITE-seq and matched RNA plus protein

CITE-seq measures RNA and antibody-derived tags in the same cells. Scarf keeps each assay's
normalization and graph separate, then integrates exact graph artifacts. No analytical stage adds
layout, cluster, or modality-weight columns to shared cell metadata.

## 1. Import the matched assays

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=True)

counts = scarf.cytebase.connect("scarf_docs").download(
    "tenx_8K_pbmc_citeseq/data.h5",
    destination="scarf_datasets",
)[0]
store = counts.with_name("data.zarr")
reader = scarf.CrH5Reader(str(counts))
print(reader.assayFeats)
scarf.CrToZarr(reader, zarr_loc=str(store)).dump()

ds = scarf.DataStore(
    str(store),
    default_assay="RNA",
    nthreads=4,
)
```

## 2. Build the RNA graph and select cells

The standard RNA pipeline handles filtering, feature selection, normalization, PCA, neighbours,
the connectivity map, and the requested Leiden partition. UMAP is disabled in the pipeline so all
four graphs on this page can use the same non-default display settings.

```{code-cell} ipython3
shared_umap_options = {
    "n_epochs": 250,
    "spread": 5,
    "min_dist": 1,
    "parallel": True,
}
rna_run = ds.pipeline.run(
    assay="RNA",
    hvg_count=1000,
    pca_dims=15,
    neighbors_k=21,
    umap=False,
    leiden={"partitions": [1.0]},
    cell_cycle=False,
    paris=False,
    doublets=False,
    markers=False,
)
cell_selection = rna_run["analysis_cell_selection"]
rna_initialization = ds.build_embedding_initialization(rna_run["pca"])
rna_neighbors = rna_run["neighbors"]
rna_graph = rna_run["connectivity_map"]
rna_umap = ds.run_umap(
    rna_graph,
    rna_initialization,
    **shared_umap_options,
)
rna_clusters = rna_run["leiden_1.0"]

cell_mask = np.asarray(
    ds.load_artifact(cell_selection)["values"][:],
    dtype=bool,
)
print(f"Selected cells: {int(cell_mask.sum())} of {len(cell_mask)}")
```

The pipeline's immutable cell selection is shared by both assays because their rows describe the
same cells. The live `I` column remains unchanged. The ADT-specific reduction and the multimodal
integration steps below stay atomic because they are not part of the fixed RNA recipe.

```{code-cell} ipython3
rna_umap_values = np.asarray(ds.load_artifact(rna_umap)["values"][:])
rna_cluster_values = np.asarray(ds.load_artifact(rna_clusters)["values"][:])
plt.scatter(
    rna_umap_values[:, 0],
    rna_umap_values[:, 1],
    c=rna_cluster_values,
    s=3,
)
```

## 3. Build the ADT graph

ADT uses centred-log-ratio normalization. Inspect the panel before excluding controls because
naming conventions vary.

```{code-cell} ipython3
adt_panel = ds.ADT.feats.to_pandas_dataframe(["names"])
adt_panel["is_control"] = adt_panel["names"].str.contains("control")
adt_panel[adt_panel["is_control"]]
```

```{code-cell} ipython3
adt_features = ds.set_feature_selection(
    from_assay="ADT",
    mask=~adt_panel["is_control"].to_numpy(),
)
adt_feature_values = np.asarray(ds.load_artifact(adt_features)["values"][:])
print(f"Selected ADT features: {int(adt_feature_values.sum())} of {len(adt_panel)}")

normalized_adt = ds.run_normalization(cell_selection, adt_features)
n_adt_features = int(ds.load_artifact(normalized_adt)["data"].shape[1])
adt_reduction = ds.run_custom_reduction(
    np.eye(n_adt_features, dtype=np.float64),
    normalized_adt,
)
adt_initialization = ds.build_embedding_initialization(adt_reduction)
adt_ann = ds.build_ann_index(adt_reduction)
adt_neighbors = ds.query_neighbors(adt_ann, k=21)
adt_graph = ds.build_connectivity_map(adt_neighbors)
adt_umap = ds.run_umap(
    adt_graph,
    adt_initialization,
    **shared_umap_options,
)
adt_clusters = ds.run_leiden_clustering(adt_graph, resolution=1)
```

An identity reduction keeps neighbour search in normalized antibody space. PCA is unnecessary for
a panel with only a few dozen features.

```{code-cell} ipython3
adt_umap_values = np.asarray(ds.load_artifact(adt_umap)["values"][:])
adt_cluster_values = np.asarray(ds.load_artifact(adt_clusters)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, coordinates, labels, title in (
    (axes[0], rna_umap_values, rna_cluster_values, "RNA"),
    (axes[1], adt_umap_values, adt_cluster_values, "ADT"),
):
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, s=3)
    axis.set_title(title)
figure.tight_layout()
figure
```

## 4. Shared nearest-neighbour integration

SNN consumes exact connectivity-map refs and requires equal neighbour degree.

```{code-cell} ipython3
snn_graph = ds.integrate_assays([rna_graph, adt_graph], method="snn")
snn_umap = ds.run_umap(
    snn_graph,
    rna_initialization,
    **shared_umap_options,
)
snn_clusters = ds.run_leiden_clustering(snn_graph, resolution=1.75)
```

```{code-cell} ipython3
snn_umap_values = np.asarray(ds.load_artifact(snn_umap)["values"][:])
snn_cluster_values = np.asarray(ds.load_artifact(snn_clusters)["values"][:])
plt.scatter(
    snn_umap_values[:, 0],
    snn_umap_values[:, 1],
    c=snn_cluster_values,
    s=3,
)
```

(wnn_integration)=

## 5. Weighted nearest-neighbour integration

WNN consumes modality-specific neighbour refs and learns a non-negative per-cell weight for each
modality. The weights are stored inside the integrated artifact.

```{code-cell} ipython3
wnn_graph = ds.integrate_assays(
    [rna_neighbors, adt_neighbors],
    method="wnn",
    l2_normalize=True,
)
wnn_umap = ds.run_umap(
    wnn_graph,
    rna_initialization,
    **shared_umap_options,
)
wnn_clusters = ds.run_leiden_clustering(wnn_graph, resolution=1.75)
weight_values = np.asarray(ds.load_artifact(wnn_graph)["modality_weights"][:])

pd.Series(
    {
        "minimum weight": float(weight_values.min()),
        "maximum weight": float(weight_values.max()),
        "mean RNA weight": float(weight_values[:, 0].mean()),
        "mean ADT weight": float(weight_values[:, 1].mean()),
        "maximum row-sum error": float(
            np.abs(weight_values.sum(axis=1) - 1).max()
        ),
    }
)
```

```{code-cell} ipython3
wnn_umap_values = np.asarray(ds.load_artifact(wnn_umap)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, index, title in (
    (axes[0], 0, "RNA weight"),
    (axes[1], 1, "ADT weight"),
):
    points = axis.scatter(
        wnn_umap_values[:, 0],
        wnn_umap_values[:, 1],
        c=weight_values[:, index],
        s=3,
    )
    axis.set_title(title)
    figure.colorbar(points, ax=axis)
figure.tight_layout()
figure
```

WNN is Hao-inspired but not bit-identical to Seurat. Scarf scores the union of persisted KNN rows,
retains the smallest input degree, uses each modality's nearest-to-farthest distance span as its
bandwidth, and stores all source refs plus modality weights in one immutable integrated graph.

Keep RNA, ADT, SNN, and WNN refs when comparing alternatives. No branch becomes an implicit active
graph or writes weights into cell metadata.

## 6. HTO demultiplexing

Hashtag assignment is a separate sample-identification task, not part of the RNA/ADT integration path.
See {doc}`hto_demultiplexing`.

## Common mistakes and limitations

- Filtering cells on one assay and then comparing modalities built from different cell sets
- Integrating per-assay graphs built with different `k`
- Leaving control antibodies active in the ADT panel
- Using WNN with fewer than two assays or with graphs built over different cells
- Reading RNA and ADT clusters as interchangeable labels for the same populations
