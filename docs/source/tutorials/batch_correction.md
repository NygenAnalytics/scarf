---
description: Correct a merged RNA graph with partial PCA or Harmony and inspect what changed.
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

(harmony_batch_correction)=

# Correcting batch effects

Batch correction changes the reduced coordinates used to build a neighbourhood graph.
Counts remain unchanged.
A useful correction should increase source mixing without dissolving biological populations.
The rebuilt teaching store carries one frozen uncorrected run. This guide branches its exact
normalization and PCA artifacts into partial PCA and Harmony comparisons.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
from scarf.plotting import CellField

scarf.configure_output(level="ERROR", progress=False)

repository = scarf.cytebase.connect("scarf_docs")
merged_path = repository.download_dataset(
    name="kang_29K_ctrl-ifnb_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{merged_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
baseline = ds.pipeline.open(label="docs_default")

normalized = baseline["normalized"]
pca_full = baseline["pca"]
uncorrected_neighbors = baseline["neighbors"]
uncorrected_graph = baseline["connectivity_map"]


def integration_scores(neighbors, graph):
    return {
        "iLISI": ds.metric_ilisi(
            batch_colname="sample_id",
            neighbors=neighbors,
            perplexity=7,
        ),
        "cLISI": ds.metric_clisi(
            annotation_column="orig_cluster_labels",
            neighbors=neighbors,
            perplexity=7,
        ),
        "graph connectivity": ds.metric_graph_connectivity(
            annotation_column="orig_cluster_labels",
            graph=graph,
        ),
    }


scores = {
    "Uncorrected": integration_scores(
        uncorrected_neighbors,
        uncorrected_graph,
    )
}
```

The run remains immutable and artifact-only. Its requested metadata and result fields stay in the
frozen run view; no live layout or cluster columns are created.

The baseline artifacts fix the active cells, highly variable features, full PCA, and 21-neighbour graph used by every comparison below.
Source identity and imported cell types on the uncorrected UMAP show the defect this page aims to reduce.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, color_by in zip(
    axes,
    ("sample_id", "orig_cluster_labels"),
    strict=True,
):
    ds.plots.embedding(
        run=baseline,
        layout="umap",
        color_by=color_by,
        target=axis,
        show=False,
    )
figure.tight_layout()
figure
```

```{code-cell} ipython3
pd.crosstab(
    baseline.cells.fetch("clusters"),
    baseline.cells.fetch("sample_id"),
    normalize="index",
)
```

```{code-cell} ipython3
pd.Series(scores["Uncorrected"]).round(3).rename("Uncorrected")
```

```{raw} html
<span id="partial-pca"></span>
<span id="partial-pca-integration"></span>
```

## 1. Learn PCA from a reference subset

`pca_cell_selection` is an explicit immutable subset of the cells represented by the normalized artifact.
PCA fits its loading basis on that subset, then projects every cell represented by `normalized` into the same basis.
Here the control cells define the reference space.
Signals absent from the control subset contribute less to the resulting graph.

```{code-cell} ipython3
baseline_active = baseline.cells.fetch_all("I").astype(bool)
is_ctrl = baseline_active & (baseline.cells.fetch_all("sample_id") == "ctrl")
ds.cells.insert(
    column_name="is_ctrl",
    values=is_ctrl,
    overwrite=True,
)
pd.Series(
    {
        "active cells": int(baseline_active.sum()),
        "reference (is_ctrl)": int(is_ctrl.sum()),
        "reference fraction": round(float(is_ctrl[baseline_active].mean()), 3),
    }
)
```

```{code-cell} ipython3
pca_partial = ds.run_pca(
    normalized,
    dims=25,
    pca_cell_selection=ds.snapshot_cell_selection(cell_key="is_ctrl"),
)
partial_initialization = ds.build_embedding_initialization(pca_partial)
partial_neighbors = ds.query_neighbors(
    ds.build_ann_index(pca_partial),
    k=21,
)
partial_graph = ds.build_connectivity_map(partial_neighbors)
partial_umap = ds.run_umap(
    partial_graph,
    partial_initialization,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
)
partial_clusters = ds.run_leiden_clustering(
    partial_graph,
    resolution=1.0,
)
scores["Partial PCA"] = integration_scores(
    partial_neighbors,
    partial_graph,
)
```

```{code-cell} ipython3
partial_umap_values = np.asarray(ds.load_artifact(partial_umap)["values"][:])
partial_cluster_values = np.asarray(
    ds.load_artifact(partial_clusters)["values"][:]
)
```

```{raw} html
<span id="harmony"></span>
<span id="harmony-batch-correction"></span>
```

## 2. Correct PCA coordinates with Harmony

Harmony adjusts the full PCA coordinates using one or more batch columns before the ANN index is built.
Treat each supplied column as variation to remove.
Do not use a biological condition that the downstream analysis needs to retain.

```{code-cell} ipython3
corrected = ds.run_harmony(pca_full, ["sample_id"])
harmony_initialization = ds.build_embedding_initialization(corrected)
harmony_neighbors = ds.query_neighbors(
    ds.build_ann_index(corrected),
    k=21,
)
harmony_graph = ds.build_connectivity_map(harmony_neighbors)
harmony_umap = ds.run_umap(
    harmony_graph,
    harmony_initialization,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
)
harmony_clusters = ds.run_leiden_clustering(
    harmony_graph,
    resolution=1.0,
)
scores["Harmony"] = integration_scores(
    harmony_neighbors,
    harmony_graph,
)
```

```{code-cell} ipython3
harmony_umap_values = np.asarray(ds.load_artifact(harmony_umap)["values"][:])
harmony_cluster_values = np.asarray(
    ds.load_artifact(harmony_clusters)["values"][:]
)
```

## 3. Compare the three graphs

Read the three exact embedding refs and place them in uncorrected, partial-PCA, and Harmony order.

```{code-cell} ipython3
baseline_frame = baseline.cells.to_pandas_dataframe(
    ["umap_1", "umap_2", "sample_id", "orig_cluster_labels", "clusters"]
)
baseline_umap_values = baseline_frame[["umap_1", "umap_2"]].to_numpy()
layout_values = (baseline_umap_values, partial_umap_values, harmony_umap_values)
sample_codes = pd.factorize(baseline_frame["sample_id"])[0]

figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, coordinates, title in zip(
    axes,
    layout_values,
    ("Uncorrected", "Partial PCA", "Harmony"),
    strict=True,
):
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=sample_codes, s=3)
    axis.set_title(title)
figure.tight_layout()
figure
```

```{code-cell} ipython3
label_codes = pd.factorize(baseline_frame["orig_cluster_labels"])[0]
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, coordinates, title in zip(
    axes,
    layout_values,
    ("Uncorrected", "Partial PCA", "Harmony"),
    strict=True,
):
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=label_codes, s=3)
    axis.set_title(title)
figure.tight_layout()
figure
```

Each method also creates its own Leiden label artifact.
Plotting those labels on the matching layout links the composition bars below to geography on the page.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
cluster_panels = (
    ("Uncorrected", baseline_umap_values, baseline_frame["clusters"].to_numpy()),
    ("Partial PCA", partial_umap_values, partial_cluster_values),
    ("Harmony", harmony_umap_values, harmony_cluster_values),
)
for axis, (title, coordinates, values) in zip(
    axes, cluster_panels, strict=True
):
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=values, s=3)
    axis.set_title(title)
figure.tight_layout()
figure
```

```{code-cell} ipython3
pd.concat(
    {
        title: pd.crosstab(values, baseline_frame["sample_id"], normalize="index")
        for title, _coordinates, values in cluster_panels
    }
)
```

## 4. Treatment response is not batch structure

The two Kang sources are also the control and interferon beta treatment groups.
`ISG15` is an interferon-stimulated gene.
Default `plots.embedding` and `plots.distribution` use assay-normalized expression via `NormalizationSpec(source="assay")` (library-size normalized through `assay.normed()`), not raw counts; use `source="raw"` for counts.
Compare expression on the same frozen cell selection because Harmony changes coordinates, not the
count matrix. Source mixing on the graph is not the same question as removing a treatment effect
from the counts.

```{code-cell} ipython3
ds.plots.distribution(
    keys="ISG15",
    grouping=CellField("sample_id"),
    cell_selection=baseline["analysis_cell_selection"],
    kind="violin",
    max_points=2000,
    seed=0,
)
```

(lisi_metrics)=
(integration_metrics)=

## 5. Quantify mixing and structural preservation

iLISI measures source mixing. cLISI checks whether imported cell-type labels remain locally separated, while graph connectivity checks whether cells with the same imported label remain connected.
All three scores are scaled so higher values are better.

```{code-cell} ipython3
score_frame = pd.DataFrame.from_dict(scores, orient="index")
score_frame.round(3)
```

Because `sample_id` coincides with treatment, iLISI describes source mixing rather than proving removal of a technical effect. cLISI and connectivity provide preservation checks, but they cannot establish that every treatment response was retained.
The `ISG15` comparison above keeps that distinction visible.
Keep the uncorrected counts for condition-level differential expression.

Compare methods only when active cells, selected features, neighbour count, and LISI perplexity match.
Do not choose a method solely because its UMAP appears compact.

When the reference should remain fixed and new samples arrive later, use {doc}`mapping_and_label_transfer` instead of rebuilding a joint graph.

See {doc}`../reference/api/graph_construction` for the PCA and Harmony contracts, and {doc}`../reference/api/integration` for the metric definitions.
