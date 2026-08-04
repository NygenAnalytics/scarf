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

RNA and ADT can agree on broad cell populations while resolving different local
structure. This guide measures that concordance and compares Scarf's SNN and
WNN integration methods. It assumes the reader already understands the
recommended CITE-seq path in {doc}`cite_seq`.

## Standalone setup

The published CITE-seq store carries the independent RNA and ADT
{term}`analysis chains <analysis chain>` and
both integrated graphs, built exactly as {doc}`cite_seq` describes: matched
active cells, `k=21` for each modality, control antibodies already marked
inactive. This page reads those results rather than reproducing them.

```{code-cell} ipython3
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=True)

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

The SNN graph is stored under `RNA+ADT` and the WNN graph under
`RNA+ADT_wnn`, each with its own UMAP and Leiden partition. The method is a
parameter, because it changes the result, while the label is an execution
option, because renaming a graph does not.

```{code-cell} ipython3
integrated = []
for ref in ds.list_artifacts(scope="datastore", kind="integrated_graph"):
    status = ds.inspect_artifact(ref)
    integrated.append(
        {
            "label": status.execution_options["label"],
            "method": status.parameters["method"],
            "artifact": ref.artifact_id[:12],
        }
    )
pd.DataFrame(integrated).sort_values("label", ignore_index=True)
```

## Measure modality concordance

First compare the cluster partitions on the two layouts. Agreement at broad
scales supports a shared biological signal. Local disagreement can be useful
when one modality resolves a population more clearly.

A normalized cross-tabulation shows which RNA populations contribute to each
ADT cluster without letting large clusters dominate the comparison.

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

Plot each modality's clusters on the other layout to locate local disagreement
that the table averages away.

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

Compare an antibody with its coding gene to distinguish genuine modality
complementarity from a gross alignment problem.

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

Protein signal is commonly less sparse than the matching transcript, so exact
point-wise agreement is not expected. Large regions with contradictory signal
need inspection before integration.

## Compare SNN and WNN

SNN combines shared edge support and can integrate two or more assays. WNN
also accepts two or more assays and learns how strongly each cell should rely
on each modality. Both consume one graph per named assay, which is why the two
chains above had to use matched cells and neighbour counts. The RNA and ADT
comparison here is the two-modality special case.

## Compare integrated partitions

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
partition_columns = {
    "RNA": "RNA_leiden_cluster",
    "ADT": "ADT_leiden_cluster",
    "SNN": "RNA+ADT_leiden_cluster",
    "WNN": "RNA+ADT_wnn_leiden_cluster",
}
concordance_rows = []
for first, second in combinations(partition_columns, 2):
    columns = [
        partition_columns[first],
        partition_columns[second],
    ]
    concordance_rows.append(
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
pd.DataFrame(concordance_rows)
```

ARI and NMI quantify partition agreement without choosing which modality or
integration method is correct. Compare them with marker coherence and the
overlap structure above instead of selecting the cleanest layout.

Plot the same protein and transcript on the two integrated layouts to check
whether marker geography stays coherent after merging.

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

## Inspect WNN modality weights

WNN records how much each cell relies on each modality. First check the weight
constraints directly.

```{code-cell} ipython3
weight_columns = {
    "RNA weight": "RNA+ADT_wnn_RNA_weight",
    "ADT weight": "RNA+ADT_wnn_ADT_weight",
}
weight_frame = pd.DataFrame(
    {
        label: ds.cells.fetch(column, key="I")
        for label, column in weight_columns.items()
    }
)
weight_values = weight_frame.to_numpy(dtype=np.float64)
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

Map the weights onto the WNN layout, then inspect their distribution. Because
the two weights sum to one per cell, the RNA-weight histogram also shows where
ADT takes over.

```{code-cell} ipython3
weight_view = ds.plots.embedding(
    layout_key="RNA+ADT_wnn_UMAP",
    color_by=list(weight_columns.values()),
    n_columns=2,
    point_size=5,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    weight_view.axes.values(),
    weight_columns,
    strict=True,
):
    axis.set_title(title)
weight_view.figure.set_size_inches(9, 4)
```

```{code-cell} ipython3
figure, axis = plt.subplots(figsize=(5, 3.5))
axis.hist(weight_frame["RNA weight"], bins=40, color="C0", alpha=0.85)
axis.set(xlabel="RNA weight", ylabel="Cells")
figure.tight_layout()
```

Use the earlier overlap table to distinguish cells in the dominant RNA cluster
for each ADT cluster from other RNA and ADT combinations.

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
weight_frame.groupby("RNA-ADT overlap")[
    list(weight_columns)
].agg(["count", "mean", "median"])
```

The dominant-overlap grouping is descriptive, not a reference annotation.
Weight shifts can identify cells whose local structure is better resolved by
one modality, but they can also expose noise or a graph mismatch. The
{doc}`cite_seq` workflow covers how SNN and WNN construct the integrated graph.

Choose between methods from the assay design and evidence, not from the layout
that looks cleaner. WNN is useful when relative local informativeness varies
across two or more modalities, while SNN gives all source graphs equal standing.
Compare cluster stability, marker coherence, known populations, and the
modality-concordance checks above.

Common failures include integrating graphs built over different cells, using
different `k` values, retaining control antibodies, and interpreting an
integrated embedding as proof that all modality-specific disagreement has been
resolved.
