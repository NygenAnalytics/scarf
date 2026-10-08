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

When you have multiple different datasets, and you seek to combine them into one, integration is usually the path to take. Dataset integration starts by placing compatible assays in one datastore. `DataStoreMerge` aligns their genes, carries selected metadata, and records the source of each cell so you can verify what data came from what dataset. It does not alter expression values or correct the joint representation. For correction of the joint representation, refer to {doc}`batch_correction`. This guide builds that uncorrected, merged dataset first.

## Load compatible source stores

The control and interferon beta stimulated Kang PBMC stores use the same cell types and genes; the prepared stores contain cells with existing cell-type labels, and the `I` columns mark the cells that are still used for analysis as they passed the quality control.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level="ERROR", progress=False)

repository = scarf.cytebase.connect("scarf_docs")
ctrl_path = repository.download_dataset(
    name="kang_15K_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
```

Download the stimulated sample from the same repository.

```{code-cell} ipython3
stim_path = repository.download_dataset(
    name="kang_14K_ifnb-pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
```

Open both source stores before checking their axes.

```{code-cell} ipython3
ds_ctrl = scarf.DataStore(f"{ctrl_path}/data.zarr", nthreads=4)
ds_stim = scarf.DataStore(f"{stim_path}/data.zarr", nthreads=4)
```

Confirm assay type, cell counts, and feature counts before merging. Ensure that all of the data you would like to analyze is present in one specific spot. `DataStoreMerge` validates the feature axes; matching gene symbols alone do not establish fully compatible genome builds.

```{code-cell} ipython3
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
)
```

## Merge counts and metadata

When merging, the `names` supply the source labels, `source_column` names their metadata column, and `prepend_text` keeps imported metadata names distinct from columns authored in the merged store. `reset_cell_filter=False` preserves the source quality-control selections, and doesn't merge the cells that were filtered out into the active selection (still merged, just not selected).

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

merged = scarf.DataStore(merged_path, nthreads=4)
merged
```

The `sample_id` records the dataset source label; columns that are imported from the sources keep the `orig_` prefix so their origin remains explicit and interpretable.

The merged active population contains labeled cells from both sources.

```{code-cell} ipython3
merged_labels = merged.cells.to_pandas_dataframe(
    ["sample_id", "orig_cluster_labels"], key="I"
)
merged_labels.groupby("sample_id")["orig_cluster_labels"].agg(
    cells="count", cell_types="nunique"
)
```

## Inspect a prepared joint analysis

Now that the merge is complete, we can see what these datasets look like together by opening the prepared merged store. It uses the same merge recipe and already contains PCA, clustering, and UMAP. The UMAP we visualize has had no batch corrections applied to it, and is thus simply the raw results of merging, and rerunning the analysis pipeline.

For context, this is a separate store from `merged`, so the following plots do not run an analysis on the store we just created.

```{code-cell} ipython3
prepared_path = repository.download_dataset(
    name="kang_29K_ctrl-ifnb_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
ds = scarf.DataStore(f"{prepared_path}/data.zarr", nthreads=4)
baseline = ds.pipeline.open(label="docs_default")
sorted(baseline)
```

The durable run maps each output name to its exact {term}`artifact`. The plot below compares source identity, imported cell types, and the exact clustering artifact on the same layout.

```{code-cell} ipython3
ds.plots.embedding(
    layout=baseline["umap"],
    color_by=["sample_id", "orig_cluster_labels", baseline["clusters"]],
    n_columns=3,
)
```

A table of proportions shows whether each Leiden cluster contains cells from both sources.

```{code-cell} ipython3
pd.crosstab(
    baseline.cells.fetch("clusters"),
    baseline.cells.fetch("sample_id"),
    rownames=["cluster"],
    colnames=["source"],
    normalize="index",
).round(3)
```

The stimulated sample received interferon beta, and PBMC cell types do not all respond identically to that treatment, thus source-associated structure can therefore include biological response as well as technical variation.

The next step would be found in {doc}`batch_correction`, which teaches how to correct technical variation between both datasets with Harmony. It also introduces metrics for batch mixing.
