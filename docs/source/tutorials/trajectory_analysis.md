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

# Trajectory validation

Trajectory results depend on the selected graph, the source and sink
annotations, and the features used for downstream tests. This guide checks those
dependencies on the Bastidas-Ponce pancreas dataset after the recommended path
in {doc}`pseudotime`, {doc}`pseudotime_modules`, and {doc}`fate_mapping`.

## Standalone setup

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=True)

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

This section opens the prepared datastore used by {doc}`pseudotime` and fits the
same multi-sink baseline so the page can run independently.

```{code-cell} ipython3
multi_sink = ds.run_pseudotime_scoring(
    source_sink_key="clusters",
    sources=["Ductal"],
    sinks=["Alpha", "Beta", "Delta"],
    label="multi_sink_pseudotime",
)
```

## Compare boundary choices

Population balance analysis directs movement between the selected boundaries.
Changing those boundaries changes the scientific question. Compare the
multi-sink baseline with a narrower ductal-to-Beta question.

```{code-cell} ipython3
beta_sink = ds.run_pseudotime_scoring(
    source_sink_key="clusters",
    sources=["Ductal"],
    sinks=["Beta"],
    label="beta_sink_pseudotime",
)
```

```{code-cell} ipython3
boundary_results = {
    "Alpha, Beta, and Delta sinks": multi_sink,
    "Beta sink": beta_sink,
}
selected_cells = int(ds.cells.fetch_all("I").sum())
coverage = pd.DataFrame(
    {
        name: {
            "valid cells": int(
                ds.cells.fetch_all(result.validity_key).sum()
            ),
            "valid fraction": float(
                ds.cells.fetch_all(result.validity_key).sum()
                / selected_cells
            ),
        }
        for name, result in boundary_results.items()
    }
).T
coverage
```

```{code-cell} ipython3
multi_valid = ds.cells.fetch_all(multi_sink.validity_key).astype(bool)
beta_valid = ds.cells.fetch_all(beta_sink.validity_key).astype(bool)
shared_valid = multi_valid & beta_valid
if int(shared_valid.sum()) < 2:
    raise RuntimeError(
        "Boundary comparison needs at least two jointly valid cells"
    )

multi_values = ds.cells.fetch_all(multi_sink.pseudotime_key)
beta_values = ds.cells.fetch_all(beta_sink.pseudotime_key)
rank_agreement = pd.Series(
    multi_values[shared_valid]
).corr(
    pd.Series(beta_values[shared_valid]),
    method="spearman",
)
pd.Series(
    {
        "jointly valid cells": int(shared_valid.sum()),
        "Spearman rank agreement": rank_agreement,
    }
)
```

```{code-cell} ipython3
figure, axis = plt.subplots(figsize=(5, 4))
density = axis.hexbin(
    multi_values[shared_valid],
    beta_values[shared_valid],
    gridsize=35,
    mincnt=1,
    cmap="viridis",
)
comparison_min = min(
    multi_values[shared_valid].min(),
    beta_values[shared_valid].min(),
)
comparison_max = max(
    multi_values[shared_valid].max(),
    beta_values[shared_valid].max(),
)
axis.plot(
    [comparison_min, comparison_max],
    [comparison_min, comparison_max],
    color="black",
    linestyle="--",
    linewidth=1,
)
axis.set(
    xlabel="Multi-sink pseudotime",
    ylabel="Beta-sink pseudotime",
    title=f"Spearman agreement: {rank_agreement:.3f}",
)
figure.colorbar(density, ax=axis, label="Cells per bin")
figure.tight_layout()
plt.show()
```

```{code-cell} ipython3
population_summary = pd.DataFrame(
    {
        "population": ds.cells.fetch_all("clusters"),
        "multi-sink pseudotime": np.where(
            multi_valid,
            multi_values,
            np.nan,
        ),
        "Beta-sink pseudotime": np.where(
            beta_valid,
            beta_values,
            np.nan,
        ),
    }
).groupby("population").agg(["count", "median"])
population_summary
```

A narrow diagonal band means the boundary change largely preserves cell order;
systematic departures identify cells whose position depends on the selected
terminal states. Disagreement does not show that one ordering is true. Low
coverage can instead indicate disconnected populations or a graph that does not
support the proposed trajectory. Use
{doc}`custom_graph_construction` when the graph itself needs a controlled
sensitivity analysis.

## Validate pseudotime marker tests

`run_pseudotime_marker_search` tests features that meet its minimum-cell and
variance requirements. Untested features retain `NaN` for both raw and adjusted
p-values. Benjamini-Hochberg correction is applied only across the tested
features in this search.

```{code-cell} ipython3
markers = ds.run_pseudotime_marker_search(
    cell_key=multi_sink.validity_key,
    pseudotime_key=multi_sink.pseudotime_key,
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
module_results = {}
module_rows = []
for name, n_clusters in {"coarse": 6, "fine": 12}.items():
    result = ds.run_pseudotime_aggregation(
        cell_key=multi_sink.validity_key,
        pseudotime_key=multi_sink.pseudotime_key,
        cluster_label=f"trajectory_modules_{name}",
        n_clusters=n_clusters,
        window_size=200,
        chunk_size=100,
    )
    module_results[name] = result
    assignments = pd.Series(result.feature_clusters)
    module_sizes = assignments[assignments != -1].value_counts()
    module_rows.append(
        {
            "setting": name,
            "requested modules": n_clusters,
            "assigned features": int(module_sizes.sum()),
            "observed modules": int(len(module_sizes)),
            "smallest module": int(module_sizes.min()),
            "largest module": int(module_sizes.max()),
        }
    )
pd.DataFrame(module_rows).set_index("setting")
```

```{code-cell} ipython3
coarse_modules = module_results["coarse"]
fine_modules = module_results["fine"]
module_features = ds.RNA.feats.to_pandas_dataframe(
    [
        "names",
        coarse_modules.cluster_key,
        fine_modules.cluster_key,
    ]
)
assigned_both = module_features[
    (module_features[coarse_modules.cluster_key] != -1)
    & (module_features[fine_modules.cluster_key] != -1)
]
pd.crosstab(
    assigned_both[coarse_modules.cluster_key],
    assigned_both[fine_modules.cluster_key],
    normalize="index",
).round(2)
```

The cross-tabulation shows how each coarse module divides at finer
granularity. A split can reveal distinct profiles, but a fragmented row with
very small groups can also indicate an unstable setting.

```{code-cell} ipython3
modules = fine_modules
assigned = module_features[
    module_features[modules.cluster_key] != -1
]
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
    subset_cell_key=multi_sink.validity_key,
    pseudotime_key=multi_sink.pseudotime_key,
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
