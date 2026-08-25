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
This guide structurally repacks the published count store, reconstructs the uncorrected analysis in a separate mounted store, then compares it with partial PCA and Harmony.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="ERROR", progress=True)

repository = scarf.cytebase.connect("scarf_docs")
merged_path = repository.download_dataset(
    name="kang_29K_ctrl-ifnb_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = str(Path(analysis_directory.name) / "counts.zarr")
repack_store(
    f"{merged_path}/data.zarr",
    repacked_counts,
    nthreads=2,
)
ds = scarf.mount_datastore(
    repacked_counts,
    at=str(Path(analysis_directory.name) / "batch_analysis.zarr"),
    default_assay="RNA",
    nthreads=4,
)
ds.pipeline.run(
    filtering=False,
    cell_cycle_scoring=False,
    highly_variable_features={
        "min_cells": 10,
        "top_n": 2000,
        "min_mean": -3,
        "max_mean": 2,
        "max_var": 6,
    },
    pca={"dims": 25},
    neighbors={"k": 21},
    umap={
        "n_epochs": 250,
        "spread": 5,
        "min_dist": 1,
        "parallel": True,
    },
    leiden={1.0: {"label": "integration_clusters"}},
    paris=False,
    doublet_scoring=False,
    markers=False,
)

baseline = ds.get_assay_state("RNA")
normalized = baseline.normalized
pca_full = baseline.reduction
uncorrected_neighbors = baseline.neighbors
uncorrected_graph = baseline.connectivity_map


def integration_scores(neighbors, graph):
    return {
        "iLISI": ds.metric_ilisi(
            batch_colname="sample_id",
            neighbors=neighbors,
            perplexity=7,
        ),
        "cLISI": ds.metric_clisi(
            label_colname="orig_cluster_labels",
            neighbors=neighbors,
            perplexity=7,
        ),
        "graph connectivity": ds.metric_graph_connectivity(
            label_colname="orig_cluster_labels",
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

The baseline artifacts fix the active cells, highly variable features, full PCA, and 21-neighbour graph used by every comparison below.
Source identity and imported cell types on the uncorrected UMAP show the defect this page aims to reduce.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=["sample_id", "orig_cluster_labels"],
    n_columns=2,
)
```

```{code-cell} ipython3
ds.plots.composition(
    category_by="sample_id",
    sample_by="RNA_clusters",
    kind="stacked",
    show_percent_labels=True,
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

Partial PCA learns its loading basis from cells selected by `pca_cell_key`, then projects every active cell into that basis.
Here the control cells define the reference space.
Signals absent from the control subset contribute less to the resulting graph.

```{code-cell} ipython3
ds.cells.insert(
    column_name="is_ctrl",
    values=ds.cells.fetch_all("sample_id") == "ctrl",
    overwrite=True,
)
active = ds.cells.fetch_all("I").astype(bool)
is_ctrl = ds.cells.fetch_all("is_ctrl")
pd.Series(
    {
        "active cells": int(active.sum()),
        "reference (is_ctrl)": int(is_ctrl[active].sum()),
        "reference fraction": round(float(is_ctrl[active].mean()), 3),
    }
)
```

```{code-cell} ipython3
pca_partial = ds.run_pca(
    normalized,
    dims=25,
    pca_cell_key="is_ctrl",
)
ds.build_embedding_initialization(pca_partial)
partial_neighbors = ds.query_neighbors(
    ds.build_ann_index(pca_partial),
    k=21,
)
partial_graph = ds.build_connectivity_map(partial_neighbors)
ds.run_umap(
    graph=partial_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="partial_UMAP",
)
ds.run_leiden_clustering(
    graph=partial_graph,
    resolution=1.0,
    label="partial_clusters",
)
scores["Partial PCA"] = integration_scores(
    partial_neighbors,
    partial_graph,
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_partial_UMAP",
    color_by=["sample_id", "orig_cluster_labels"],
    n_columns=2,
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
corrected = ds.run_harmony(["sample_id"], pca_full)
ds.build_embedding_initialization(pca_full)
harmony_neighbors = ds.query_neighbors(
    ds.build_ann_index(corrected),
    k=21,
)
harmony_graph = ds.build_connectivity_map(harmony_neighbors)
ds.run_umap(
    graph=harmony_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="harmony_UMAP",
)
ds.run_leiden_clustering(
    graph=harmony_graph,
    resolution=1.0,
    label="harmony_clusters",
)
scores["Harmony"] = integration_scores(
    harmony_neighbors,
    harmony_graph,
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_harmony_UMAP",
    color_by=["sample_id", "orig_cluster_labels"],
    n_columns=2,
)
```

## 3. Compare the three graphs

The plotting facade accepts several layouts directly, so the comparison does not need a custom Matplotlib helper.
Panels appear in uncorrected, partial-PCA, and Harmony order.

```{code-cell} ipython3
layouts = [
    "RNA_UMAP",
    "RNA_partial_UMAP",
    "RNA_harmony_UMAP",
]
ds.plots.embedding(
    layout_key=layouts,
    color_by="sample_id",
    n_columns=3,
    legend_loc="right",
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=layouts,
    color_by="orig_cluster_labels",
    n_columns=3,
    legend_loc="right",
)
```

Each method also writes its own Leiden partition.
Plotting those labels on the matching layout links the composition bars below to geography on the page.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
cluster_panels = (
    ("Uncorrected", "RNA_UMAP", "RNA_clusters"),
    ("Partial PCA", "RNA_partial_UMAP", "RNA_partial_clusters"),
    ("Harmony", "RNA_harmony_UMAP", "RNA_harmony_clusters"),
)
for axis, (title, layout_key, color_by) in zip(
    axes, cluster_panels, strict=True
):
    ds.plots.embedding(
        layout_key=layout_key,
        color_by=color_by,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(14, 4))
composition_panels = (
    ("Uncorrected", "RNA_clusters"),
    ("Partial PCA", "RNA_partial_clusters"),
    ("Harmony", "RNA_harmony_clusters"),
)
for index, (axis, (title, sample_by)) in enumerate(
    zip(axes, composition_panels, strict=True)
):
    ds.plots.composition(
        category_by="sample_id",
        sample_by=sample_by,
        kind="stacked",
        show_percent_labels=True,
        show_legend=index == 2,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
figure
```

## 4. Treatment response is not batch structure

The two Kang sources are also the control and interferon beta treatment groups.
`ISG15` is an interferon-stimulated gene.
Default `plots.embedding` and `plots.distribution` use assay-normalized expression via `NormalizationSpec(source="assay")` (library-size normalized through `assay.normed()`), not raw counts; use `source="raw"` for counts.
Coloring uncorrected and Harmony layouts with those same values shows that Harmony moves cells while expression itself is unchanged.
Source mixing on the graph is therefore not the same question as removing a treatment effect from the counts.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=["RNA_UMAP", "RNA_harmony_UMAP"],
    color_by="ISG15",
    n_columns=2,
    sort_values=True,
    legend_loc="right",
)
```

```{code-cell} ipython3
ds.plots.distribution(
    keys="ISG15",
    group_by="sample_id",
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
The `ISG15` panels above keep that distinction visible.
Keep the uncorrected counts for condition-level differential expression.

Compare methods only when active cells, selected features, neighbour count, and LISI perplexity match.
Do not choose a method solely because its UMAP appears compact.

When the reference should remain fixed and new samples arrive later, use {doc}`mapping_and_label_transfer` instead of rebuilding a joint graph.

See {doc}`../reference/api/graph_construction` for the PCA and Harmony contracts, and {doc}`../reference/api/integration` for the metric definitions.
