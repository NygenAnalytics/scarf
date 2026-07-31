---
description: Merge compatible single-cell datasets and inspect their uncorrected joint structure.
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

# Merging datasets for joint analysis

Merging places cells from compatible assays into one datastore. It aligns the
feature order, preserves selected source metadata, and records each cell's
source. It does not remove batch effects. This guide deliberately builds an
uncorrected graph first so the source structure is visible before any
correction.

Use {doc}`choosing_integration_methods` if you are deciding between merging,
batch correction, mapping, and multimodal integration.

## Download compatible source stores

The control and interferon-stimulated Kang PBMC datasets share an RNA feature
space. They also differ in biological treatment, so `sample_id` is not a purely
technical batch variable.

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
```

Check feature identities and assay types before merging real datasets. A shared
gene symbol is not sufficient if genome builds or quantification conventions
differ.

## Merge counts and metadata

`AssayMerge` writes a new Zarr store. `names` supplies the source labels,
`source_column` names their cell-metadata column, and `prepend_text` prevents
source columns from colliding with new analysis results.

```{code-cell} ipython3
merged_path = "scarf_datasets/kang_dataset_merging.zarr"
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
```

The merge preserves each input cell filter because
`reset_cell_filter=False`. It aligns counts and metadata under the same row
permutation, prefixes imported columns with `orig_`, and prefixes cell IDs by
source.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ["ids", "sample_id", "orig_cluster_labels", "I"]
).head()
```

Rows with `I=False` remain in the merged metadata but are inactive. An inactive
row can therefore show a missing imported label without indicating a failed
merge.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ["sample_id"],
    key="I",
)["sample_id"].value_counts()
```

## Build an uncorrected joint graph

The naive graph provides a baseline. If samples separate, that observation
defines what a correction method would need to change. It does not by itself
show whether the separation is technical or biological.

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=10,
    top_n=2000,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=False,
)
ds.run_normalization(feat_key="hvgs")
ds.run_pca(dims=25)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=21)
ds.build_connectivity_map()
ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
)
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
comparison_panels = (
    ("Source", "sample_id"),
    ("Imported cell type", "orig_cluster_labels"),
)
for axis, (title, color_by) in zip(axes, comparison_panels, strict=True):
    ds.plots.embedding(
        layout_key="RNA_UMAP",
        color_by=color_by,
        legend_loc="right",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
```

The sample panel shows how strongly source identity structures the uncorrected
graph. The label panel checks whether imported cell types still occupy coherent
regions. In this experiment, source and interferon treatment are confounded, so
separation cannot be classified as a removable batch effect from these plots
alone.

```{raw} html
<span id="partial-pca"></span>
<span id="partial-pca-integration"></span>
<span id="harmony"></span>
<span id="harmony-batch-correction"></span>
```

## Batch correction

Use {doc}`batch_correction` to compare partial PCA and Harmony, then
{doc}`integration_metrics` to measure mixing and label preservation. Do not use
corrected coordinates as input for condition-level differential expression.
