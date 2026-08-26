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

Group features with similar pseudotime expression patterns, store module means as a new assay, and inspect where those programs are active.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and UMAP embedding
- Complete {doc}`pseudotime` through correlated genes, or run the setup below, which always scores pseudotime

## What you will learn

- Aggregate features into pseudotime expression modules
- Create a grouped assay from module means with `add_grouped_assay`
- Inspect module activity on the original embedding

## Dataset

This page uses the Bastidas-Ponce pancreas store from {doc}`pseudotime`.
The setup below is standalone: it downloads the store, mounts its counts and literal metadata into a temporary writable analysis store, builds a current graph lineage, and always runs pseudotime scoring.
The catalog snapshot is structurally repacked inside the temporary directory first so it has the current RNA count layout; mounting then starts a clean analysis state.

```{code-cell} ipython3
from tempfile import TemporaryDirectory

import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)

analysis_directory = TemporaryDirectory()
repacked_counts = f'{analysis_directory.name}/counts.zarr'
repack_store(
    f'{dataset}/data.zarr',
    repacked_counts,
    nthreads=2,
)
ds = scarf.mount_datastore(
    repacked_counts,
    at=f'{analysis_directory.name}/analysis.zarr',
    nthreads=4,
    default_assay='RNA',
)
hvg_ref = ds.mark_hvgs(
    top_n=2000,
    show_plot=False,
    label='expression_hvgs',
)
normalized = ds.run_normalization(features=hvg_ref)
reduction = ds.run_pca(normalized, dims=15)
ann_index = ds.build_ann_index(reduction)
neighbors = ds.query_neighbors(ann_index, k=11)
graph = ds.build_connectivity_map(neighbors)
all_features = ds.resolve_features('RNA', 'all_features')

pseudotime = ds.run_pseudotime_scoring(
    graph,
    source_sink_key='clusters',
    sources=['Ductal'],
    sinks=['Alpha', 'Beta', 'Delta'],
)
pseudotime_key = pseudotime.pseudotime_key
validity_key = pseudotime.validity_key
```

The sources and sinks come from the provided `clusters` annotation, the same choice made in {doc}`pseudotime`.
The embedding below anchors that trajectory before modules are built.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['clusters', pseudotime_key],
    n_columns=2,
    legend_loc='on_data',
)
```

## 1. Identify feature modules

`run_pseudotime_marker_search` identifies features with a linear relationship to pseudotime.
It does not capture every dynamic pattern.
Some genes, for example, may peak only in the middle of a trajectory or along one branch.

`run_pseudotime_aggregation` orders cells by pseudotime and creates a smoothed, scaled, binned expression matrix.
It then applies KNN and Paris clustering to identify features with similar expression patterns.

```{code-cell} ipython3
modules = ds.run_pseudotime_aggregation(
    features=all_features,
    cell_key=validity_key,
    pseudotime_key=pseudotime_key,
    cluster_label='pseudotime_clusters',
    n_clusters=15,
    window_size=200,
    chunk_size=100,
)
```

The returned result contains the lazy binned matrix in `modules.data`, the aligned physical feature indices, their cluster assignments, and the exact `modules.feature_selection` reference.
It also exposes the feature column as `modules.cluster_key`.

Features with mean expression below `min_exp` or with no variation along the ordering are treated as invalid.
They are excluded from the clustering and from the heatmap below, and they receive the unassigned cluster value (`-1`) in the feature table.
Module sizes make that split explicit: `-1` is the unassigned bin.

```{code-cell} ipython3
ptime_feat = ds.RNA.feats.to_pandas_dataframe(
    columns=['names', modules.cluster_key]
)
ptime_feat[modules.cluster_key].value_counts().sort_index()
```

One representative gene per assigned module ties heatmap labels and later UMAP panels to the same module ids.

```{code-cell} ipython3
assigned = ptime_feat[ptime_feat[modules.cluster_key] != -1]
representatives = (
    assigned.groupby(modules.cluster_key, sort=True)['names']
    .first()
    .rename('representative gene')
)
representatives
```

`ds.plots.pseudotime_heatmap` visualizes the binned matrix along with the feature clusters.
Spaced representatives become row labels.

```{code-cell} ipython3
genes_to_label = representatives.iloc[::3].tolist()

ds.plots.pseudotime_heatmap(
    cell_key=modules.cell_key,
    features=modules.feature_selection,
    feature_cluster_key=modules.cluster_key,
    pseudotime_key=modules.pseudotime_key,
    show_features=genes_to_label,
)
```

The heatmap shows expression dynamics as cells progress through pseudotime.
Each row block is one feature module.
A useful result contains coherent early, intermediate, and late patterns rather than one block dominated by uniformly expressed genes.
After Paris clustering, modules are renumbered by median peak bin along pseudotime, so earlier peaks get lower IDs.
That order reflects when modules peak, not a strict developmental sequence of every gene.

## 2. Create a module assay

The pseudotime-based feature clusters can seed a new assay.
`add_grouped_assay` takes the mean expression of genes in each cluster and stores those means as features in a new assay.
That keeps many related genes out of the cell metadata table while still exposing one summary value per module.
Here we create an assay named `PTIME_MODULES`.

```{code-cell} ipython3
ds.add_grouped_assay(
    group_key=modules.cluster_key,
    assay_label='PTIME_MODULES'
)
```

The new assay has one feature per assigned module and one mean-expression value per cell.
The preview below is the first five cells against the first five modules.

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

Inspect a spaced subset of modules on the original UMAP.
These are mean expression summaries, so a sequential colour scale is appropriate.
The first panel repeats the ordering so module hotspots can be read against the trajectory.

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

This figure complements the heatmap by showing where selected module programs are active in the original cell layout.

## Common mistakes and limitations

- Interpreting linear correlation as the only form of expression dynamics along pseudotime
- Reading module IDs as a strict developmental sequence rather than peak-time order along the binned trajectory
- Comparing module gene lists to cluster markers without accounting for unassigned features (`-1`)

Feature labels are stored under `modules.cluster_key`, and `PTIME_MODULES` contains one mean-expression feature per assigned module.
Parameter diagnostics and comparison with cluster marker genes are covered in {doc}`trajectory_validation`.
