---
description: Tune and validate pseudotime ordering, expression modules, and multi-sink fate probabilities.
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

(trajectory_analysis)=

# Tuning and validating trajectory analyses

Trajectory results depend on the selected graph, the source and sink
annotations, and the features used for downstream tests. This guide checks those
dependencies on the Bastidas-Ponce pancreas dataset after the recommended path
in {doc}`pseudotime`, {doc}`pseudotime_modules`, and {doc}`fate_mapping`.

## Set up an independent analysis

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
import scarf.plotting as splt

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
    default_assay="RNA",
)
```

The prepared store contains a neighbourhood graph, UMAP coordinates, and the
provided `clusters` annotation. A trajectory from ductal cells toward Alpha,
Beta, and Delta states is biologically directed rather than discovered without
supervision.

## Check graph coverage and direction

Population balance analysis treats movement on the graph as a directed process
between selected boundaries. The graph must connect the populations of interest.
By default, Scarf scores the largest connected component and marks other cells
invalid rather than assigning unsupported values.

```{code-cell} ipython3
pseudotime = ds.run_pseudotime_scoring(
    source_sink_key="clusters",
    sources=["Ductal"],
    sinks=["Alpha", "Beta", "Delta"],
    label="validated_pseudotime",
)

coverage = pd.Series(
    {
        "selected cells": int(ds.cells.fetch_all("I").sum()),
        "valid pseudotime cells": int(
            ds.cells.fetch_all(pseudotime.validity_key).sum()
        ),
    }
)
coverage
```

```{code-cell} ipython3
coverage_plot = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[pseudotime.pseudotime_key, "clusters"],
    n_columns=2,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    coverage_plot.axes.values(),
    ("Pseudotime", "Cell type"),
    strict=True,
):
    axis.set_title(title)
```

The valid fraction should cover the annotated endocrine trajectory, and the
score should increase from the source toward all three sinks. Changing source
and sink labels to force a preferred orientation is not a substitute for fixing
a disconnected or biologically unsuitable graph.

## Validate pseudotime marker tests

`run_pseudotime_marker_search` tests features that meet its minimum-cell and
variance requirements. Untested features retain `NaN` for both raw and adjusted
p-values. Benjamini-Hochberg correction is applied only across the tested
features in this search.

```{code-cell} ipython3
markers = ds.run_pseudotime_marker_search(
    cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
)

markers.table[
    ["feature_name", "r_value", "p_value", "p_value_adjusted"]
].head()
```

```{code-cell} ipython3
markers.table[["p_value", "p_value_adjusted"]].isna().sum()
```

A small adjusted p-value supports association with this fitted ordering. It does
not establish a nonlinear pattern, branch specificity, causality, or
replicate-aware differential expression.

## Compare module choices

Aggregation smooths expression over ordered windows and clusters genes with
similar profiles. `window_size` controls smoothing, while `n_clusters` controls
the requested module granularity. Compare results for coherent patterns and
reasonable module sizes rather than selecting a value from the heatmap alone.

```{code-cell} ipython3
modules = ds.run_pseudotime_aggregation(
    cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
    cluster_label="trajectory_modules",
    n_clusters=12,
    window_size=200,
    chunk_size=100,
)

pd.Series(modules.feature_clusters).value_counts().sort_index()
```

```{code-cell} ipython3
module_features = ds.RNA.feats.to_pandas_dataframe(
    ["names", modules.cluster_key]
)
assigned = module_features[module_features[modules.cluster_key] != -1]
representatives = (
    assigned.groupby(modules.cluster_key, sort=True)["names"]
    .first()
)
labels = representatives.iloc[::2].tolist()
ds.plots.pseudotime_heatmap(
    cell_key=modules.cell_key,
    feat_key=modules.feature_key,
    feature_cluster_key=modules.cluster_key,
    pseudotime_key=modules.pseudotime_key,
    show_features=labels,
)
```

Unassigned features use `-1`. A module made mostly of genes without a coherent
ordered pattern is a reason to adjust smoothing or module granularity.

Cluster markers and trajectory modules answer different questions. The first
contrasts a discrete group with the remaining cells; the second groups genes by
their profiles along an ordering. Their overlap can be informative without
being complete.

```{code-cell} ipython3
ds.run_marker_search(group_key="clusters")
beta_markers = set(
    ds.get_markers(group_key="clusters", group_id="Beta").feature_name
)
module_overlap = (
    assigned.groupby(modules.cluster_key)["names"]
    .apply(lambda names: len(set(names) & beta_markers))
    .sort_values(ascending=False)
)
module_overlap.head()
```

## Validate fate probabilities

Fate mapping solves one absorption probability per terminal group on a graph
biased toward increasing pseudotime. For valid cells, probabilities should sum
to one. Cells in a terminal boundary should have probability one for their own
fate.

```{code-cell} ipython3
fate = ds.run_fate_mapping(
    cell_key="I",
    subset_cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
    sink_key="clusters",
    sinks=["Alpha", "Beta", "Delta"],
)

valid_probabilities = fate.values[fate.valid]
simplex_error = float(
    np.max(np.abs(valid_probabilities.sum(axis=1) - 1.0))
)
simplex_error
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(11, 4))
probability_scale = splt.ColorScale(vmin=0, vmax=1)
for index, (axis, sink, fate_key) in enumerate(
    zip(axes, fate.sink_labels, fate.fate_keys, strict=True)
):
    ds.plots.embedding(
        layout_key="RNA_UMAP",
        color_by=fate_key,
        subset_by=fate.validity_key,
        color_scale=probability_scale,
        sort_values=True,
        show_legend=index == 2,
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(f"{sink} fate probability")
figure.tight_layout()
figure
```

```{code-cell} ipython3
selected_labels = ds.cells.fetch(
    fate.sink_key,
    key=fate.result_cell_key,
)
terminal_checks = []
for index, sink in enumerate(fate.sink_labels):
    rows = (selected_labels == sink) & fate.valid
    terminal_checks.append(
        {
            "sink": sink,
            "cells": int(rows.sum()),
            "minimum own probability": float(
                fate.values[rows, index].min()
            ),
        }
    )
pd.DataFrame(terminal_checks)
```

Large simplex errors, terminal probabilities below one, or fate fields that do
not align with the chosen terminal regions indicate a numerical, graph, or
boundary problem. Fate probabilities remain model-based summaries and require
independent biological validation.
