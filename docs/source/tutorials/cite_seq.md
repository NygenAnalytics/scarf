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

## 1. Open the matched assays

The downloaded Zarr store was rebuilt with this Scarf version. It contains the complete RNA, ADT,
SNN, and WNN artifacts used below, so calls with the recorded parameters reuse those artifacts.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_8K_pbmc_citeseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", default_assay="RNA", nthreads=4)
ds
```

## 2. Build the RNA graph and select cells

The prepared RNA pipeline run fixes the filtering, feature selection, normalization, PCA,
neighbours, connectivity map, UMAP, and Leiden outputs used together.

```{code-cell} ipython3
rna_run = ds.pipeline.open(label="docs_default")
cell_selection = rna_run["analysis_cell_selection"]
rna_initialization = rna_run["embedding_initialization"]
rna_neighbors = rna_run["neighbors"]
rna_graph = rna_run["connectivity_map"]
rna_umap = rna_run["umap"]
rna_clusters = rna_run["leiden_1.0"]

cell_mask = rna_run.cells.fetch_all("I")
print(f"Selected cells: {int(cell_mask.sum())} of {len(cell_mask)}")
```

The run's immutable cell selection is shared by both assays because their rows describe the same
cells. The live `I` column remains unchanged. ADT reduction and multimodal integration remain
separate atomic chains.

```{code-cell} ipython3
ds.plots.embedding(layout=rna_umap, color_by=rna_clusters)
```

## 3. Build the ADT graph

ADT uses centred-log-ratio normalization. Inspect the panel before excluding controls because
naming conventions vary.

```{code-cell} ipython3
adt_names = ds.ADT.feats.fetch_all("names").astype(str)
adt_panel = pd.DataFrame({"name": adt_names})
adt_panel["is_control"] = adt_panel["name"].str.lower().str.contains("control")
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
adt_umap = ds.run_umap(adt_graph, adt_initialization)
adt_clusters = ds.run_leiden_clustering(adt_graph, resolution=1)
```

An identity reduction keeps neighbour search in normalized antibody space. PCA is unnecessary for
a panel with only a few dozen features.

```{code-cell} ipython3
ds.plots.embedding(layout=adt_umap, color_by=adt_clusters)
```

## 4. Shared nearest-neighbour integration

SNN consumes exact connectivity-map refs and requires equal neighbour degree.

```{code-cell} ipython3
snn_graph = ds.integrate_assays([rna_graph, adt_graph], method="snn")
snn_umap = ds.run_umap(snn_graph, rna_initialization)
snn_clusters = ds.run_leiden_clustering(snn_graph, resolution=1.75)
```

```{code-cell} ipython3
ds.plots.embedding(layout=snn_umap, color_by=snn_clusters)
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
wnn_umap = ds.run_umap(wnn_graph, rna_initialization)
weight_values = np.asarray(ds.load_artifact(wnn_graph)["modality_weights"][:])

pd.Series(
    {
        "minimum weight": float(weight_values.min()),
        "maximum weight": float(weight_values.max()),
        "mean RNA weight": float(weight_values[:, 0].mean()),
        "mean ADT weight": float(weight_values[:, 1].mean()),
        "maximum row-sum error": float(np.abs(weight_values.sum(axis=1) - 1).max()),
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
