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

Batch correction changes the reduced coordinates used to build a neighbourhood
graph. Counts remain unchanged. A useful correction should increase source
mixing without dissolving biological populations. This guide resumes the
persisted uncorrected analysis from {doc}`data_integration` and compares it
with partial PCA and Harmony.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level="ERROR", progress=True)

repository = scarf.cytebase.connect("scarf_docs")
merged_path = repository.download_dataset(
    name="kang_29K_ctrl-ifnb_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{merged_path}/data.zarr", nthreads=4)

baseline = ds.get_assay_state("RNA")
normalized = baseline.normalized
pca_full = baseline.reduction
uncorrected_neighbors = baseline.neighbors
uncorrected_graph = baseline.connectivity_map


def integration_scores(neighbors, graph):
    knn_path = ds.inspect_artifact(neighbors).path
    graph_path = ds.inspect_artifact(graph).path
    return {
        "iLISI": ds.metric_ilisi(
            batch_colname="sample_id",
            use_latest_knn=False,
            knn_loc=knn_path,
            perplexity=7,
        ),
        "cLISI": ds.metric_clisi(
            label_colname="orig_cluster_labels",
            use_latest_knn=False,
            knn_loc=knn_path,
            perplexity=7,
        ),
        "graph connectivity": ds.metric_graph_connectivity(
            label_colname="orig_cluster_labels",
            graph_loc=graph_path,
        ),
    }


scores = {
    "Uncorrected": integration_scores(
        uncorrected_neighbors,
        uncorrected_graph,
    )
}
{"uncorrected iLISI": round(scores["Uncorrected"]["iLISI"], 3)}
```

The baseline artifacts fix the active cells, highly variable features, full
PCA, and 21-neighbour graph used by every comparison below.

```{raw} html
<span id="partial-pca"></span>
<span id="partial-pca-integration"></span>
```

## Learn PCA from a reference subset

Partial PCA learns its loading basis from cells selected by `pca_cell_key`, then
projects every active cell into that basis. Here the control cells define the
reference space. Signals absent from the control subset contribute less to the
resulting graph.

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
ds.build_embedding_initialization(pca_partial)
partial_neighbors = ds.query_neighbors(
    ds.build_ann_index(pca_partial),
    k=21,
)
partial_graph = ds.build_connectivity_map(partial_neighbors)
ds.run_umap(
    partial_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="partial_UMAP",
)
ds.run_leiden_clustering(
    partial_graph,
    resolution=1.0,
    label="partial_clusters",
)
scores["Partial PCA"] = integration_scores(
    partial_neighbors,
    partial_graph,
)
{"partial PCA iLISI": round(scores["Partial PCA"]["iLISI"], 3)}
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_partial_UMAP",
    color_by=["sample_id", "orig_cluster_labels"],
    n_columns=2,
)
```

```{code-cell} ipython3
ds.plots.composition(
    category_by="sample_id",
    sample_by="RNA_partial_clusters",
    kind="stacked",
    show_percent_labels=True,
)
```

```{raw} html
<span id="harmony"></span>
<span id="harmony-batch-correction"></span>
```

## Correct PCA coordinates with Harmony

Harmony adjusts the full PCA coordinates using one or more batch columns before
the ANN index is built. Treat each supplied column as variation to remove. Do
not use a biological condition that the downstream analysis needs to retain.

```{code-cell} ipython3
corrected = ds.run_harmony(["sample_id"], pca_full)
ds.build_embedding_initialization(pca_full)
harmony_neighbors = ds.query_neighbors(
    ds.build_ann_index(corrected),
    k=21,
)
harmony_graph = ds.build_connectivity_map(harmony_neighbors)
ds.run_umap(
    harmony_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="harmony_UMAP",
)
ds.run_leiden_clustering(
    harmony_graph,
    resolution=1.0,
    label="harmony_clusters",
)
scores["Harmony"] = integration_scores(
    harmony_neighbors,
    harmony_graph,
)
{"Harmony iLISI": round(scores["Harmony"]["iLISI"], 3)}
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_harmony_UMAP",
    color_by=["sample_id", "orig_cluster_labels"],
    n_columns=2,
)
```

```{code-cell} ipython3
ds.plots.composition(
    category_by="sample_id",
    sample_by="RNA_harmony_clusters",
    kind="stacked",
    show_percent_labels=True,
)
```

## Compare the three graphs

The plotting facade accepts several layouts directly, so the comparison does
not need a custom Matplotlib helper. Panels appear in uncorrected, partial-PCA,
and Harmony order.

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

(lisi_metrics)=
(integration_metrics)=

## Quantify mixing and structural preservation

iLISI measures source mixing. cLISI checks whether imported cell-type labels
remain locally separated, while graph connectivity checks whether cells with
the same imported label remain connected. All three scores are scaled so
higher values are better.

```{code-cell} ipython3
score_frame = pd.DataFrame.from_dict(scores, orient="index")
score_frame.round(3)
```

The two Kang sources are also the control and interferon beta treatment groups,
so iLISI describes source mixing rather than proving removal of a technical
effect. cLISI and connectivity provide preservation checks, but they cannot
establish that every treatment response was retained. Keep the uncorrected
counts for condition-level differential expression.

Compare methods only when active cells, selected features, neighbour count, and
LISI perplexity match. Do not choose a method solely because its UMAP appears
compact.

When the reference should remain fixed and new samples arrive later, use
{doc}`mapping_and_label_transfer` instead of rebuilding a joint graph.

See {doc}`../reference/api/graph_construction` for the PCA and Harmony
contracts, and {doc}`../reference/api/integration` for the metric definitions.
