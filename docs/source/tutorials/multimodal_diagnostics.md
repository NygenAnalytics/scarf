---
description: Diagnose RNA and ADT agreement, then compare SNN and WNN integration.
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

# Multimodal diagnostics

RNA and ADT can agree on broad cell populations while resolving different local structure.
This guide measures that concordance and compares Scarf's SNN and WNN integration methods.
It assumes the reader already understands the recommended CITE-seq path in {doc}`cite_seq`.

## Standalone setup

The published CITE-seq store carries literal layouts and partitions for visualization plus immutable
SNN and WNN graph artifacts over matched cells. This page reads those values for diagnostics;
{doc}`cite_seq` shows the current explicit-input construction API.

```{code-cell} ipython3
from itertools import combinations

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
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
```

The snapshot retains literal `RNA+ADT` and `RNA+ADT_wnn` layout and partition columns.
The immutable graph artifacts themselves are distinguished by their scientific `method` parameter and exact references.

```{code-cell} ipython3
integrated = []
integrated_refs = {}
integrated_parameters = {}
for ref in ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    complete_only=True,
):
    status = ds.inspect_artifact(ref)
    parameters = status.parameters or {}
    method = parameters["method"]
    if method in integrated_refs:
        raise RuntimeError(f"Expected one complete {method!r} graph")
    integrated_refs[method] = ref
    integrated_parameters[method] = parameters
    integrated.append(
        {
            "method": method,
            "artifact": ref.artifact_id[:12],
        }
    )
pd.DataFrame(integrated).sort_values("method", ignore_index=True)
```

Resolve the four immutable clustering artifacts once. Their graph inputs identify
which partition each artifact contains; the literal metadata columns below remain
useful only for plotting and cross-tabulation.

```{code-cell} ipython3
baseline = ds.pipeline.open(label="docs_default")
graph_names = {
    baseline["connectivity_map"]: "RNA",
    integrated_refs["snn"]: "SNN",
    integrated_refs["wnn"]: "WNN",
}
partition_refs = {"RNA": baseline["clusters"]}
partition_candidates = ds.list_artifacts(
    kind="cluster_labels",
    scope="datastore",
    complete_only=True,
)
for assay in ("RNA", "ADT"):
    partition_candidates.extend(
        ds.list_artifacts(
            kind="cluster_labels",
            from_assay=assay,
            complete_only=True,
        )
    )
for ref in partition_candidates:
    graph_value = (ds.inspect_artifact(ref).inputs or {}).get("graph")
    if not isinstance(graph_value, dict):
        continue
    graph_ref = scarf.ArtifactRef.from_dict(graph_value)
    name = graph_names.get(graph_ref)
    if name is None and ref.assay == "ADT":
        name = "ADT"
    if name is None:
        continue
    if name in partition_refs and partition_refs[name] != ref:
        raise RuntimeError(f"Expected one complete {name!r} partition")
    partition_refs[name] = ref

expected_partitions = {"RNA", "ADT", "SNN", "WNN"}
if set(partition_refs) != expected_partitions:
    raise RuntimeError("The CITE-seq snapshot is missing a required partition")
```

## 1. Measure modality concordance

First compare the cluster partitions on the two layouts.
Agreement at broad scales supports a shared biological signal.
Local disagreement can be useful when one modality resolves a population more clearly.

A normalized cross-tabulation shows which RNA populations contribute to each ADT cluster without letting large clusters dominate the comparison.

```{code-cell} ipython3
overlap = pd.crosstab(
    ds.cells.fetch("RNA_leiden_cluster"),
    ds.cells.fetch("ADT_leiden_cluster"),
    normalize="columns",
)

fig, ax = plt.subplots(figsize=(7, 5))
image = ax.imshow(overlap.to_numpy(), aspect="auto", cmap="magma")
ax.set(
    xlabel="ADT cluster",
    ylabel="RNA cluster",
    xticks=np.arange(overlap.shape[1]),
    yticks=np.arange(overlap.shape[0]),
    xticklabels=overlap.columns,
    yticklabels=overlap.index,
)
fig.colorbar(image, ax=ax, label="Fraction within ADT cluster")
fig.tight_layout()
```

Plot each modality's clusters on the other layout to locate local disagreement that the table averages away.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
cross_panels = (
    ("RNA layout, ADT clusters", "RNA_UMAP", "ADT_leiden_cluster"),
    ("ADT layout, RNA clusters", "ADT_UMAP", "RNA_leiden_cluster"),
)
for axis, (title, layout_key, color_by) in zip(
    axes, cross_panels, strict=True
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
```

Compare an antibody with its coding gene to distinguish genuine modality complementarity from a gross alignment problem.

```{code-cell} ipython3
def compare_signal(feature, assay, label, layouts):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for index, (axis, (layout_label, layout_key)) in enumerate(
        zip(axes, layouts, strict=True)
    ):
        ds.plots.embedding(
            layout_key=layout_key,
            color_by=feature,
            from_assay=assay,
            point_size=5,
            show_legend=index == 1,
            show_titles=False,
            target=axis,
            show=False,
        )
        axis.set_title(f"{layout_label}: {label}")
    # Top colorbars occupy the title slot; label the colorbar axes instead.
    right_title = f"{layouts[-1][0]}: {label}"
    for colorbar_axis in set(figure.axes) - set(axes):
        colorbar_axis.set_title(right_title)
        colorbar_axis.set_xlabel("")
        colorbar_axis.set_ylabel("")
    figure.tight_layout()
    return figure


modality_layouts = (
    ("RNA layout", "RNA_UMAP"),
    ("ADT layout", "ADT_UMAP"),
)
compare_signal(
    "CD16_TotalSeqB",
    "ADT",
    "CD16 protein",
    modality_layouts,
);
```

```{code-cell} ipython3
compare_signal(
    "FCGR3A",
    "RNA",
    "FCGR3A transcript",
    modality_layouts,
);
```

Protein signal is commonly less sparse than the matching transcript, so exact point-wise agreement is not expected.
Large regions with contradictory signal need inspection before integration.

## 2. Compare SNN and WNN partitions

SNN combines shared edge support from one connectivity-map artifact per assay.
WNN accepts one neighbour artifact per assay and learns how strongly each cell should rely on each modality.
Both accept two or more assays and require matched cells.
Matched neighbour counts (`k`) are required for SNN only; WNN warns and keeps `min(k)` when they differ.
The RNA and ADT comparison here is the two-modality special case.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
integration_panels = (
    ("SNN", "RNA+ADT_UMAP", "RNA+ADT_leiden_cluster"),
    ("WNN", "RNA+ADT_wnn_UMAP", "RNA+ADT_wnn_leiden_cluster"),
)
for axis, (title, layout_key, color_by) in zip(
    axes, integration_panels, strict=True
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
```

```{code-cell} ipython3
concordance_rows = []
for first, second in combinations(partition_refs, 2):
    concordance_rows.append(
        {
            "comparison": f"{first} vs {second}",
            "ARI": ds.metric_label_concordance(
                partition_refs[first],
                partition_refs[second],
                metric="ari",
            ),
            "NMI": ds.metric_label_concordance(
                partition_refs[first],
                partition_refs[second],
                metric="nmi",
            ),
        }
    )
pd.DataFrame(concordance_rows)
```

ARI and NMI quantify partition agreement without choosing which modality or integration method is correct.
Compare them with marker coherence and the overlap structure above instead of selecting the cleanest layout.

Plot the same protein and transcript on the two integrated layouts to check whether marker geography stays coherent after merging.

```{code-cell} ipython3
integrated_layouts = (
    ("SNN layout", "RNA+ADT_UMAP"),
    ("WNN layout", "RNA+ADT_wnn_UMAP"),
)
compare_signal(
    "CD16_TotalSeqB",
    "ADT",
    "CD16 protein",
    integrated_layouts,
);
```

```{code-cell} ipython3
compare_signal(
    "FCGR3A",
    "RNA",
    "FCGR3A transcript",
    integrated_layouts,
);
```

## 3. Inspect WNN modality weights

WNN stores how much each cell relies on each modality inside the returned graph artifact.
First check the weight constraints directly.

```{code-cell} ipython3
weight_values = np.asarray(
    ds.load_artifact(integrated_refs["wnn"])["modality_weights"][:],
    dtype=np.float64,
)
weight_columns = [
    f"{assay} weight"
    for assay in integrated_parameters["wnn"]["assays"]
]
weight_frame = pd.DataFrame(
    weight_values,
    columns=weight_columns,
)
pd.Series(
    {
        "all finite": bool(np.isfinite(weight_values).all()),
        "all non-negative": bool((weight_values >= 0).all()),
        "maximum row-sum error": float(
            np.max(np.abs(weight_values.sum(axis=1) - 1))
        ),
    }
)
```

Map the weights onto the WNN layout, then inspect their distribution.
Because the two weights sum to one per cell, the RNA-weight histogram also shows where ADT takes over.

```{code-cell} ipython3
wnn_layout = ds.cells.to_pandas_dataframe(
    ["RNA+ADT_wnn_UMAP1", "RNA+ADT_wnn_UMAP2"],
    key="I",
)
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, title in zip(axes, weight_frame.columns, strict=True):
    points = axis.scatter(
        wnn_layout["RNA+ADT_wnn_UMAP1"],
        wnn_layout["RNA+ADT_wnn_UMAP2"],
        c=weight_frame[title],
        s=5,
    )
    axis.set_title(title)
    figure.colorbar(points, ax=axis)
figure.tight_layout()
```

```{code-cell} ipython3
figure, axis = plt.subplots(figsize=(5, 3.5))
axis.hist(weight_frame["RNA weight"], bins=40, color="C0", alpha=0.85)
axis.set(xlabel="RNA weight", ylabel="Cells")
figure.tight_layout()
```

Use the earlier overlap table to distinguish cells in the dominant RNA cluster for each ADT cluster from other RNA and ADT combinations.

```{code-cell} ipython3
rna_clusters = pd.Series(
    ds.cells.fetch("RNA_leiden_cluster", key="I")
)
adt_clusters = pd.Series(
    ds.cells.fetch("ADT_leiden_cluster", key="I")
)
dominant_rna_by_adt = overlap.idxmax(axis=0)
weight_frame["RNA-ADT overlap"] = np.where(
    rna_clusters == adt_clusters.map(dominant_rna_by_adt),
    "dominant overlap",
    "other overlap",
)
weight_frame.groupby("RNA-ADT overlap").agg(["count", "mean", "median"])
```

The dominant-overlap grouping is descriptive, not a reference annotation.
Weight shifts can identify cells whose local structure is better resolved by one modality, but they can also expose noise or a graph mismatch.
The {doc}`cite_seq` workflow covers how SNN and WNN construct the integrated graph.

Choose between methods from the assay design and evidence, not from the layout that looks cleaner.
WNN is useful when relative local informativeness varies across two or more modalities, while SNN gives all source graphs equal standing.
Compare cluster stability, marker coherence, known populations, and the modality-concordance checks above.

Common failures include integrating graphs built over different cells, using different `k` values with SNN, retaining control antibodies, and interpreting an integrated embedding as proof that all modality-specific disagreement has been resolved.
