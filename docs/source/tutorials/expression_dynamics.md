---
description: Group genes with shared expression dynamics along pseudotime and summarize them as an assay.
jupytext:
  formats: ipynb,md:myst
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

# Expression dynamics along pseudotime

Group features with similar pseudotime expression patterns, store module means as a new
assay, and inspect where those programs are active.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and UMAP embedding
- Complete {doc}`pseudotime` through correlated genes, or let the setup below score
  pseudotime when it is missing

## What you will learn

- Aggregate features into pseudotime expression modules
- Create a grouped assay from module means with `add_grouped_assay`
- Inspect module activity on the original embedding

## Dataset

This page uses the Bastidas-Ponce pancreas store from {doc}`pseudotime`. The setup below
is standalone: it downloads the store, opens a `DataStore`, and runs pseudotime scoring
when needed.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)

ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    default_assay='RNA',
)

pseudotime = ds.run_pseudotime_scoring(
    source_sink_key='clusters',
    sources=['Ductal'],
    sinks=['Alpha', 'Beta', 'Delta'],
)
pseudotime_key = pseudotime.pseudotime_key
validity_key = pseudotime.validity_key
```

The sources and sinks come from the provided `clusters` annotation, the same choice made in
{doc}`pseudotime`. The embedding below anchors that trajectory before modules are built.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['clusters', pseudotime_key],
    n_columns=2,
    legend_loc='on_data',
)
```

## 1. Identify feature modules

`run_pseudotime_marker_search` identifies features with a linear relationship to pseudotime. It does not capture every dynamic pattern. Some genes, for example, may peak only in the middle of a trajectory or along one branch.

`run_pseudotime_aggregation` orders cells by pseudotime and creates a smoothed, scaled, binned expression matrix. It then applies KNN and Paris clustering to identify features with similar expression patterns.

```{code-cell} ipython3
modules = ds.run_pseudotime_aggregation(
    cell_key=validity_key,
    pseudotime_key=pseudotime_key,
    cluster_label='pseudotime_clusters',
    n_clusters=15,
    window_size=200,
    chunk_size=100,
)
```

The returned result contains the lazy binned matrix in `modules.data`, the
aligned physical feature indices, and their cluster assignments. It also
exposes the feature column as `modules.cluster_key`.

Features with mean expression below `min_exp` or with no variation along the ordering
are treated as invalid. They are excluded from the clustering and from the heatmap
below, and they receive the unassigned cluster value (`-1`) in the feature table.
Module sizes make that split explicit: `-1` is the unassigned bin.

```{code-cell} ipython3
ptime_feat = ds.RNA.feats.to_pandas_dataframe(
    columns=['names', modules.cluster_key]
)
ptime_feat[modules.cluster_key].value_counts().sort_index()
```

One representative gene per assigned module ties heatmap labels and later UMAP
panels to the same module ids.

```{code-cell} ipython3
assigned = ptime_feat[ptime_feat[modules.cluster_key] != -1]
representatives = (
    assigned.groupby(modules.cluster_key, sort=True)['names']
    .first()
    .rename('representative gene')
)
representatives
```

`ds.plots.pseudotime_heatmap` visualizes the binned matrix along with the feature
clusters. Spaced representatives become row labels.

```{code-cell} ipython3
genes_to_label = representatives.iloc[::3].tolist()

ds.plots.pseudotime_heatmap(
    cell_key=modules.cell_key,
    feat_key=modules.feature_key,
    feature_cluster_key=modules.cluster_key,
    pseudotime_key=modules.pseudotime_key,
    show_features=genes_to_label,
)
```

The heatmap shows expression dynamics as cells progress through pseudotime. Each
row block is one feature module. A useful result contains coherent early,
intermediate, and late patterns rather than one block dominated by uniformly
expressed genes. Module numbers are clustering labels and do not encode temporal
order.

## 2. Create a module assay

The pseudotime-based feature clusters can seed a new assay. `add_grouped_assay` takes the
mean expression of genes in each cluster and stores those means as features in a new assay.
That keeps many related genes out of the cell metadata table while still exposing one summary
value per module. Here we create an assay named `PTIME_MODULES`.

```{code-cell} ipython3
ds.add_grouped_assay(
    group_key=modules.cluster_key,
    assay_label='PTIME_MODULES'
)
```

The new assay has one feature per assigned module and one mean-expression value
per cell. The preview below is the first five cells against the first five
modules.

```{code-cell} ipython3
module_features = ds.PTIME_MODULES.feats.fetch_all('names').tolist()
{
    "module count": len(module_features),
    "cells": int(ds.PTIME_MODULES.rawData.shape[0]),
    "first modules": module_features[:5],
}
```

```{code-cell} ipython3
pd.DataFrame(
    ds.PTIME_MODULES.rawData[:5, :5].compute(),
    columns=module_features[:5],
)
```

Inspect a spaced subset of modules on the original UMAP. These are mean
expression summaries, so a sequential colour scale is appropriate. The first panel
repeats the ordering so module hotspots can be read against the trajectory.

```{code-cell} ipython3
selected_modules = [
    f"group_{module_id}" for module_id in representatives.iloc[::3].index
]
ds.plots.embedding(
    from_assay='PTIME_MODULES',
    layout_key='RNA_UMAP',
    color_by=[pseudotime_key, *selected_modules],
    n_columns=3,
    sort_values=True,
)
```

This figure complements the heatmap by showing where selected module programs
are active in the original cell layout.

## Common mistakes and limitations

- Interpreting linear correlation as the only form of expression dynamics along pseudotime
- Treating module numbers as ordered along pseudotime (they are clustering labels)
- Comparing module gene lists to cluster markers without accounting for unassigned features (`-1`)

Feature labels are stored under `modules.cluster_key`, and `PTIME_MODULES`
contains one mean-expression feature per assigned module. Parameter diagnostics
and comparison with cluster marker genes are covered in
{doc}`trajectory_validation`.
