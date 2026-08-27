---
description: Score an explicit graph for pseudotime and load immutable trajectory marker artifacts.
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

# Pseudotime analysis

Pseudotime is an oriented summary of graph structure. Source and sink choices supervise that
orientation; Scarf does not infer terminal states or causal lineage.

## 1. Build an explicit graph

```{code-cell} ipython3
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = f"{analysis_directory.name}/counts.zarr"
repack_store(f"{dataset}/data.zarr", repacked_counts, nthreads=2)
ds = scarf.mount_datastore(
    repacked_counts,
    at=f"{analysis_directory.name}/analysis.zarr",
    nthreads=4,
    default_assay="RNA",
)

cell_selection = ds.snapshot_cell_selection("I")
features = ds.select_hvgs(
    cell_selection,
    top_n=2000,
    show_plot=False,
)
normalized = ds.run_normalization(cell_selection, features)
reduction = ds.run_pca(normalized, dims=15)
ann = ds.build_ann_index(reduction)
neighbors = ds.query_neighbors(ann, k=11)
graph = ds.build_connectivity_map(neighbors)
all_features = ds.set_feature_selection(
    from_assay="RNA",
    feature_indexes=range(ds.RNA.feats.N),
)
```

The teaching store includes literal `clusters` annotations and an imported UMAP. Build a custom
zero-sum source/sink vector over the exact graph rows. Ductal cells supply positive source mass;
Alpha, Beta, and Delta cells share negative sink mass.

```{code-cell} ipython3
labels = np.asarray(ds.cells.fetch("clusters", key="I"))
source = labels == "Ductal"
sink = np.isin(labels, ["Alpha", "Beta", "Delta"])
source_sink_vector = np.zeros(len(labels), dtype=float)
source_sink_vector[source] = 1.0 / int(source.sum())
source_sink_vector[sink] = -1.0 / int(sink.sum())
float(source_sink_vector.sum())
```

## 2. Score pseudotime

```{code-cell} ipython3
pseudotime_ref = ds.run_pseudotime_scoring(
    graph,
    ss_vec=source_sink_vector,
)
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
{
    "artifact": pseudotime.ref,
    "graph": pseudotime.graph,
    "valid cells": int(pseudotime.valid.sum()),
}
```

The producer returns an artifact. The explicit loader returns values, a validity mask, graph ref,
and cell-selection ref. No pseudotime or validity column is added to live metadata.

```{code-cell} ipython3
umap = ds.cells.to_pandas_dataframe(["RNA_UMAP1", "RNA_UMAP2"], key="I")
valid = pseudotime.valid
figure, axis = plt.subplots(figsize=(5, 4))
points = axis.scatter(
    umap.loc[valid, "RNA_UMAP1"],
    umap.loc[valid, "RNA_UMAP2"],
    c=pseudotime.values[valid],
    s=4,
)
figure.colorbar(points, ax=axis, label="pseudotime")
figure.tight_layout()
figure
```

Values should progress from the ductal region toward endocrine endpoints. A disconnected or
reversed pattern is a reason to revisit the graph and endpoint choices.

```{code-cell} ipython3
pd.DataFrame(
    {
        "cluster": labels[valid],
        "pseudotime": pseudotime.values[valid],
    }
).groupby("cluster")["pseudotime"].describe()
```

## 3. Search for pseudotime-associated features

```{code-cell} ipython3
marker_ref = ds.run_pseudotime_marker_search(
    pseudotime_ref,
    features=all_features,
)
markers = ds.load_pseudotime_markers(marker_ref)
markers.table[["p_value", "p_value_adjusted"]].notna().sum()
```

Untested features retain `NaN` p-values. Benjamini-Hochberg adjustment covers tested features only.

```{code-cell} ipython3
tested = markers.table.loc[
    markers.table["p_value_adjusted"].notna(),
    ["feature_name", "r_value", "p_value_adjusted"],
]
increasing = tested.loc[tested["r_value"] > 0].nlargest(10, "r_value")
decreasing = tested.loc[tested["r_value"] < 0].nsmallest(10, "r_value")
pd.concat({"increasing": increasing, "decreasing": decreasing})
```

Correlation is one form of evidence and can miss nonlinear dynamics. Use {doc}`expression_dynamics`
for smoothed feature profiles and modules, {doc}`fate_mapping` for multiple terminal outcomes, and
{doc}`trajectory_validation` for component and endpoint checks.

## Common mistakes and limitations

- Choosing source or sink groups that do not sit at the intended ends of the graph
- Ignoring the validity mask when the graph has multiple components
- Treating a strong correlation as evidence of causal lineage
- Comparing trajectory refs built from different graphs without recording that difference
