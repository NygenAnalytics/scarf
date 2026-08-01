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

Batch correction changes the reduced representation used to build a
neighbourhood graph. It should remove unwanted technical structure while
preserving biological structure relevant to the analysis. This guide compares
an uncorrected graph with partial PCA and Harmony.

The Kang control and interferon-stimulated PBMC datasets demonstrate the APIs,
but treatment and source are confounded. Better mixing in this example is not
proof that a technical effect was removed without losing interferon biology.

## Prepare a merged datastore

```{code-cell} ipython3
import matplotlib.pyplot as plt

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

merged_path = "scarf_datasets/kang_batch_correction.zarr"
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
```

## Keep an uncorrected baseline

```{code-cell} ipython3
pca_full = ds.run_pca(normalized, dims=25)
ds.build_embedding_initialization(pca_full)
ann = ds.build_ann_index(pca_full)
neighbors = ds.query_neighbors(ann, k=21)
graph = ds.build_connectivity_map(neighbors)
ds.run_umap(
    graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="uncorrected_UMAP",
)
```

The baseline records the source separation that correction will change. Retain
it for comparison rather than judging the corrected layout in isolation.

## Learn PCA from a reference subset

Partial PCA learns the loading basis from cells selected by `pca_cell_key`, then
projects all active cells into that basis. It is appropriate when one trusted
sample defines the reference space. Signals absent from that subset, including
real condition-specific biology, contribute less to the graph.

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
ann = ds.build_ann_index(pca_partial)
neighbors = ds.query_neighbors(ann, k=21)
partial_graph = ds.build_connectivity_map(neighbors)
ds.run_umap(
    partial_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="partial_UMAP",
)
```

## Correct PCA coordinates with Harmony

Harmony adjusts the full PCA coordinates using one or more batch columns before
the ANN index is built. It treats the supplied column as unwanted variation, so
do not pass a biological condition that the downstream analysis needs to retain.

```{code-cell} ipython3
pca_full = ds.run_pca(normalized, dims=25)
corrected = ds.run_harmony(["sample_id"], pca_full)
ds.build_embedding_initialization(pca_full)
ann = ds.build_ann_index(corrected)
neighbors = ds.query_neighbors(ann, k=21)
harmony_graph = ds.build_connectivity_map(neighbors)
ds.run_umap(
    harmony_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label="harmony_UMAP",
)
```

## Compare the three graphs

```{code-cell} ipython3
layout_comparisons = (
    ("Uncorrected", "RNA_uncorrected_UMAP"),
    ("Partial PCA", "RNA_partial_UMAP"),
    ("Harmony", "RNA_harmony_UMAP"),
)


def compare_graphs(color_by):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for index, (axis, (title, layout_key)) in enumerate(
        zip(axes, layout_comparisons, strict=True)
    ):
        ds.plots.embedding(
            layout_key=layout_key,
            color_by=color_by,
            legend_loc="right",
            show_legend=index == 2,
            show_titles=False,
            target=axis,
            show=False,
        )
        axis.set_title(title)
    figure.tight_layout()
    return figure


compare_graphs("sample_id");
```

```{code-cell} ipython3
compare_graphs("orig_cluster_labels");
```

Correction should increase source mixing without collapsing the imported cell
types. Layouts only suggest that trade-off. Use {doc}`integration_metrics` for
quantitative checks, and retain uncorrected counts for condition-level
differential expression.

Common failures include correcting a confounded biological condition, comparing
methods built with different features or neighbour counts, and selecting the
method that produces the most visually compact UMAP.

See the {doc}`../reference/api/graph_construction` for the exact
{py:meth}`~scarf.DataStore.run_pca` and
{py:meth}`~scarf.DataStore.run_harmony` contracts.
