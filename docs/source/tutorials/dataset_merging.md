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
These prepared stores contain cells with author-provided cell-type labels.
Their `I` columns mark the cells that passed quality control.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level="ERROR", progress=False)

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

Confirm assay type, cell counts, and feature counts before merging. `DataStoreMerge` validates the
feature axes; matching gene symbols alone do not establish compatible genome builds or
quantification conventions.

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

## 2. Merge counts and metadata

`names` supplies the source labels, `source_column` names their metadata column, and `prepend_text`
keeps imported metadata names distinct from columns authored in the merged store.
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

merged = scarf.DataStore(merged_path, nthreads=4)
```

`sample_id` records the source label.
Columns imported from the sources keep the `orig_` prefix so their origin remains explicit.

The merged active population contains labelled cells from both sources.

```{code-cell} ipython3
merged.cells.to_pandas_dataframe(
    ["sample_id", "orig_cluster_labels"],
    key="I",
).groupby("sample_id")["orig_cluster_labels"].agg(
    cells="count",
    cell_types="nunique",
)
```

## 3. Inspect a prepared joint analysis

The merge is complete. To see what these datasets look like together, open the catalog's prepared
merged store. It uses the same merge recipe and already contains PCA, clustering, and UMAP.
This is a separate store from `merged`, so the following plots do not run an analysis on the store
you just created. The saved example run is named `docs_default`.

```{code-cell} ipython3
prepared_path = repository.download_dataset(
    name="kang_29K_ctrl-ifnb_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{prepared_path}/data.zarr", nthreads=4)
baseline = ds.pipeline.open(label="docs_default")
sorted(baseline)
```

The durable run maps each output name to its exact {term}`artifact`.
Requested metadata and results remain in its frozen view.

One plotting call compares source identity, imported cell types, and the exact clustering artifact
on the same layout.

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
    normalize="index",
)
```

The stimulated sample received interferon beta, and PBMC cell types do not all respond identically to that treatment.
Source-associated structure can therefore include biological response as well as technical variation.

The next page, {doc}`batch_correction`, uses a different dataset with measured sequencing batches
to compare an uncorrected analysis with Harmony. It also introduces metrics for batch mixing.
Keep uncorrected counts for condition-level differential expression.
