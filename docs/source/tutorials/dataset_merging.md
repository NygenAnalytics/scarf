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

(integration_guide)=

# Integrating datasets by merging

Dataset integration starts by placing compatible assays in one datastore.
`DataStoreMerge` aligns their feature order, carries selected metadata, and records the source of each cell.
It does not alter expression values or correct the joint representation.
This guide builds that uncorrected baseline first.

## 1. Load compatible source stores

The control and interferon beta stimulated Kang PBMC stores use the same RNA feature space.
Their publication recipe physically removes cells without an imported cell-type label before running source-level quality control.
The remaining `I` cell key records that quality-control selection.

```{code-cell} ipython3
import pandas as pd

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

Confirm assay type, cell counts, and feature counts before merging.
Matching gene symbols alone do not establish compatible measurements; genome builds and quantification conventions still need to match when you bring other datasets.

```{code-cell} ipython3
ctrl_ids = set(ds_ctrl.RNA.feats.fetch_all("ids").astype(str))
stim_ids = set(ds_stim.RNA.feats.fetch_all("ids").astype(str))

pd.DataFrame(
    [
        {
            "source": label,
            "assay": type(store.RNA).__name__,
            "cells": store.cells.N,
            "active cells": int(store.cells.fetch_all("I").sum()),
            "features": store.RNA.feats.N,
        }
        for label, store in (("ctrl", ds_ctrl), ("stim", ds_stim))
    ]
).assign(shared_features=len(ctrl_ids & stim_ids))
```

## 2. Merge counts and metadata

`names` supplies the source labels, `source_column` names their metadata column, and `prepend_text` prevents imported columns from colliding with new analysis results.
`reset_cell_filter=False` preserves the source quality-control selections.

```{code-cell} ipython3
merged_path = "scarf_datasets/kang_dataset_merging.zarr"
scarf.DataStoreMerge(
    datasets=[ds_ctrl, ds_stim],
    zarr_path=merged_path,
    names=["ctrl", "stim"],
    assays=["RNA"],
    prepend_text="orig",
    reset_cell_filter=False,
    source_column="sample_id",
    overwrite=True,
).dump()

ds = scarf.DataStore(merged_path, nthreads=4)
```

`sample_id` records the source label.
Columns imported from the sources keep the `orig_` prefix so later analysis columns can reuse their original names.

```{code-cell} ipython3
orig_cols = [
    column
    for column in ds.cells.columns
    if column == "sample_id" or column.startswith("orig_")
]
ds.cells.to_pandas_dataframe(orig_cols, key="I").head()
```

The merged active population contains labelled cells from both sources.

```{code-cell} ipython3
active_cells = ds.cells.to_pandas_dataframe(
    ["sample_id", "orig_cluster_labels"],
    key="I",
)
active_cells.groupby("sample_id")["orig_cluster_labels"].agg(
    cells="count",
    cell_types="nunique",
)
```

## 3. Build the uncorrected baseline

The standard RNA pipeline records the complete analysis chain as reusable artifacts.
Filtering is disabled because the source selections were retained.
The graph uses 21 neighbours so the same graph parameters can be compared with the correction methods on the next page.

```{code-cell} ipython3
baseline = ds.pipeline.run(
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
sorted(baseline)
```

The return value lists each written {term}`artifact` by name.
The selected Leiden partition is also published as `RNA_clusters` for later plotting.

One plotting call compares source identity with the imported cell types on the same layout.

```{code-cell} ipython3
comparison = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=["sample_id", "orig_cluster_labels"],
    n_columns=2,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    comparison.axes.values(),
    ("Source", "Imported cell type"),
    strict=True,
):
    axis.set_title(title)
comparison.show()
```

The uncorrected Leiden partition that the pipeline selected as `RNA_clusters` should track broad cell-type structure even while sources remain segregated.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_clusters",
)
```

A proportional composition plot makes source dominance within the uncorrected Leiden clusters explicit.

```{code-cell} ipython3
ds.plots.composition(
    category_by="sample_id",
    sample_by="RNA_clusters",
    kind="stacked",
    show_percent_labels=True,
)
```

iLISI summarizes local source mixing on a zero-to-one scale.
Zero means the median neighbourhood effectively contains cells from only one source.
One is the maximum mixing score across the observed sources.

```{code-cell} ipython3
uncorrected_ilisi = ds.metric_ilisi(
    batch_colname="sample_id",
    perplexity=7,
)
{"uncorrected iLISI": round(uncorrected_ilisi, 3)}
```

The value `0.000` therefore indicates essentially no source mixing in the median uncorrected neighbourhood.

The stimulated sample received interferon beta, and PBMC cell types do not all respond identically to that treatment.
Source-associated structure can therefore include biological response as well as technical variation.
An interferon-response gene such as `ISG15` makes that stim-enriched program visible on the same uncorrected layout.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="ISG15",
)
```

This page establishes the uncorrected observation; {doc}`batch_correction` compares how partial PCA and Harmony change it.
Keep uncorrected counts for condition-level differential expression.
