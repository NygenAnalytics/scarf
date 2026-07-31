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

# Multimodal integration

RNA and ADT can agree on broad cell populations while resolving different local
structure. This guide builds both graphs independently, measures their
concordance, and compares Scarf's SNN and WNN integration methods. It assumes
the reader already understands the recommended CITE-seq path in
{doc}`cite_seq`.

## Build independent RNA and ADT graphs

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
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
ds.auto_filter_cells()
```

Each modality needs a graph built over the same active cell selection and with
the same neighbour count. RNA uses HVGs and PCA. ADT uses active antibodies and
an identity reduction because the panel is already low-dimensional.

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=20,
    top_n=1000,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=False,
)
ds.run_normalization(feat_key="hvgs")
ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=21)
ds.build_connectivity_map()
ds.run_umap(n_epochs=250, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=1)
```

```{code-cell} ipython3
adt_panel = ds.ADT.feats.to_pandas_dataframe(["names"])
is_control = adt_panel["names"].str.contains("control")
ds.ADT.feats.update_key(~is_control.values, "I")

normalized_adt = ds.run_normalization(
    from_assay="ADT",
    feat_key="I",
)
n_adt_features = int(
    ds.load_artifact(normalized_adt)["data"].shape[1]
)
ds.run_custom_reduction(
    np.eye(n_adt_features, dtype=np.float64),
    normalized_adt,
    from_assay="ADT",
)
ds.build_embedding_initialization(
    from_assay="ADT",
    n_centroids=100,
)
ds.build_ann_index(from_assay="ADT")
ds.query_neighbors(from_assay="ADT", k=21)
ds.build_connectivity_map(from_assay="ADT")
ds.run_umap(
    from_assay="ADT",
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
)
ds.run_leiden_clustering(from_assay="ADT", resolution=1)
```

## Measure modality concordance

First compare the cluster partitions on the two layouts. Agreement at broad
scales supports a shared biological signal. Local disagreement can be useful
when one modality resolves a population more clearly.

```{code-cell} ipython3
figure, axes = plt.subplots(2, 2, figsize=(9, 8))
modality_panels = (
    ("RNA layout, RNA clusters", "RNA_UMAP", "RNA_leiden_cluster"),
    ("RNA layout, ADT clusters", "RNA_UMAP", "ADT_leiden_cluster"),
    ("ADT layout, RNA clusters", "ADT_UMAP", "RNA_leiden_cluster"),
    ("ADT layout, ADT clusters", "ADT_UMAP", "ADT_leiden_cluster"),
)
for axis, (title, layout_key, color_by) in zip(
    axes.flat, modality_panels, strict=True
):
    ds.plots.embedding(
        layout_key=layout_key,
        color_by=color_by,
        point_size=5,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
```

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

Compare an antibody with its coding gene to distinguish genuine modality
complementarity from a gross alignment problem.

```{code-cell} ipython3
def compare_signal(feature, assay, label):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for index, (axis, (layout_label, layout_key)) in enumerate(
        zip(
            axes,
            (("RNA layout", "RNA_UMAP"), ("ADT layout", "ADT_UMAP")),
            strict=True,
        )
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
    figure.tight_layout()
    return figure


compare_signal("CD16_TotalSeqB", "ADT", "CD16 protein");
```

```{code-cell} ipython3
compare_signal("FCGR3A", "RNA", "FCGR3A transcript");
```

Protein signal is commonly less sparse than the matching transcript, so exact
point-wise agreement is not expected. Large regions with contradictory signal
need inspection before integration.

## Compare SNN and WNN

SNN combines shared edge support and can integrate two or more assays. WNN
accepts exactly two assays and learns how strongly each cell should rely on each
modality. Both methods consume the latest graph for each named assay.

```{code-cell} ipython3
ds.integrate_assays(
    assays=["RNA", "ADT"],
    label="RNA+ADT_snn",
    method="snn",
)
ds.run_umap(
    integrated_graph="RNA+ADT_snn",
    n_epochs=500,
    spread=5,
    min_dist=0.5,
    parallel=True,
)
ds.run_leiden_clustering(
    integrated_graph="RNA+ADT_snn",
    resolution=1.75,
)
```

```{code-cell} ipython3
ds.integrate_assays(
    assays=["RNA", "ADT"],
    label="RNA+ADT_wnn",
    method="wnn",
)
ds.run_umap(
    integrated_graph="RNA+ADT_wnn",
    n_epochs=500,
    spread=5,
    min_dist=0.5,
    parallel=True,
)
ds.run_leiden_clustering(
    integrated_graph="RNA+ADT_wnn",
    resolution=1.75,
)
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
integration_panels = (
    ("SNN", "RNA+ADT_snn_UMAP", "RNA+ADT_snn_leiden_cluster"),
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

Choose between methods from the assay design and evidence, not from the layout
that looks cleaner. SNN is the available choice for more than two modalities.
WNN is useful when the relative informativeness of RNA and ADT varies across
cells. Compare cluster stability, marker coherence, known populations, and the
modality-concordance checks above.

Common failures include integrating graphs built over different cells, using
different `k` values, retaining control antibodies, and interpreting an
integrated embedding as proof that all modality-specific disagreement has been
resolved.
