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
# Pseudotime Primer

Pseudotime analysis refers to modeling and testing how molecular features, primarily gene expression, transcription factor activity, or pathway scores, change continuously along a reconstructed trajectory coordinate. Pseudotime allows us to not view cells as a static discrete cluster, but instead enable us to dynamically model cellular processes as continuous gene regulatory programs unfolding across a developmental, activation, or perturbation axis.

Pseudotime more specifically is a summary of the existing graph structure and the distance of a cell's transcriptional profile from a defined starting population along the graph.

## Open the prepared graph

```{code-cell}
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
analysis_run = ds.pipeline.open(label="docs_default")
graph = analysis_run["connectivity_map"]
all_features = analysis_run["feature_universe"]
```

From the completed analysis, we grab the existing computed graph, feature universe, and the UMAP.

For pseudotime analysis, we need to select a starting point for what we can use to calculate the distance of a cell from the starting population. For our dataset, based on the biological context, our starting population would be the ductal progeinerators cells. If you'd like to gain a more technical understanding of what the source & the sink vectors are, refer to {doc}`fate_mapping`.

```{code-cell}
labels = ds.cells.fetch("clusters", key="I")
source = labels == "Ductal"
sink = np.isin(labels, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any():
    raise ValueError("Source and sink labels must both be present")
source_sink_vector = np.zeros(len(labels), dtype=float)
source_sink_vector[source] = -1.0 / source.sum()
source_sink_vector[sink] = 1.0 / sink.sum()
float(source_sink_vector.sum())
```

We define our sink vectors, or our endpoints, as Alpha, Beta, or Delta cells to define the endpoint of our axis, in which ductal cells are the starting point. With this supervised axis, each cell can then be scored: low means Ductal-like, high means terminus-like (Alpha/Beta/Delta). To go a step further and determine if a cell is likely to be a Alpha/Beta/Delta, access {doc}`fate_mapping`.

## Scoring Pseudotime

```{code-cell}
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

After we perform our analysis, we can now visualize the pseudotime results.

```{code-cell}
ds.cells.insert("pseudotime", pseudotime.values, key="I", overwrite=True)
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by="pseudotime",
    sort_values=True,
)
```

With our known biological context, values should progress from the ductal region toward our endocrine cell endpoints. A disconnected or reversed pattern is something thats important to investigate. Remember, to assign our endpoints and our starting points, use gene markers to chracterize these bits.

```{code-cell}
pd.DataFrame(
    {
        "cluster": labels[pseudotime.valid],
        "pseudotime": pseudotime.values[pseudotime.valid],
    }
).groupby("cluster")["pseudotime"].describe()
```

For each cluster, the table above reports how many cells it holds and how their pseudotime values distribute in comparison to the starting population, letting you validate thar your starting ductal cells sit near sit near zero, while our terminal cells like Alpha, Beta, and Delta cells sit near one. A cluster whose mean lands mid-axis, or whose spread spans the full range, deserves digging into the data further.

## 3. Search for pseudotime-associated features

```{code-cell}
marker_ref = ds.run_pseudotime_marker_search(
    pseudotime_ref,
    features=all_features,
)
markers = ds.load_pseudotime_markers(marker_ref)
markers.table[["p_value", "p_value_adjusted"]].notna().sum()
```

Untested features retain `NaN` p-values. Benjamini-Hochberg adjustment covers tested features only.

```{code-cell}
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
