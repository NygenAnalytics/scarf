---
description: Compute and validate immutable multi-sink fate-probability artifacts.
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

# Fate mapping

Fate mapping estimates terminal-outcome probabilities on an oriented graph. It is supervised by a
pseudotime artifact and an exact sink-label artifact. The probabilities summarize the model; they
do not establish causal lineage.

## 1. Reuse the prepared graph and sink labels

```{code-cell} ipython3
import matplotlib.pyplot as plt
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

annotations = ds.cells.fetch("clusters", key="I")
source = annotations == "Ductal"
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any():
    raise ValueError("Source and sink annotations must both be present")
source_sink_vector = np.zeros(len(annotations), dtype=float)
source_sink_vector[source] = -1.0 / source.sum()
source_sink_vector[sink] = 1.0 / sink.sum()

pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
sink_labels_ref = analysis_run["clusters"]
sink_labels = analysis_run.cells.fetch("clusters")
```

The rebuilt catalog store contains the completed `docs_default` pipeline run. This page reuses its
exact graph, selected clustering, and UMAP. The literal `clusters` column contains the published
cell-type annotations used only to orient pseudotime. The new pseudotime and fate results remain
exact artifacts.

For this executable mechanics example, choose the two clusters with the greatest mean valid
pseudotime as candidate terminal labels. A real analysis should choose and validate endpoints from
study-specific evidence.

```{code-cell} ipython3
terminal_labels = (
    pd.Series(
        pseudotime.values[pseudotime.valid],
        index=sink_labels[pseudotime.valid],
    )
    .groupby(level=0)
    .mean()
    .nlargest(2)
    .index.tolist()
)
if len(terminal_labels) != 2:
    raise ValueError("Fate mapping requires exactly two terminal labels in this example")
terminal_labels
```

## 2. Compute fate probabilities

```{code-cell} ipython3
fate_ref = ds.run_fate_mapping(
    pseudotime_ref,
    sink_labels_ref,
    sinks=terminal_labels,
)
fate = ds.load_fate_mapping(fate_ref)
{
    "artifact": fate.ref,
    "pseudotime": fate.pseudotime,
    "sink labels": fate.sink_labels,
    "valid cells": int(fate.valid.sum()),
}
```

The producer writes one artifact containing all probability columns and validity, leaving cell
metadata unchanged.

```{code-cell} ipython3
valid_probabilities = fate.values[fate.valid]
probability_summary = pd.DataFrame(
    valid_probabilities,
    columns=[str(label) for label in fate.sink_labels],
)
probability_summary["row-sum error"] = np.abs(
    probability_summary.sum(axis=1) - 1.0
)
probability_summary.agg(["min", "median", "max"])
```

Probabilities should be finite, non-negative, and sum to one for valid cells.

```{code-cell} ipython3
umap = np.asarray(ds.load_artifact(analysis_run["umap"])["values"][:])
figure, axes = plt.subplots(1, len(fate.sink_labels), figsize=(9, 4))
for index, label in enumerate(fate.sink_labels):
    axis = axes[index]
    points = axis.scatter(
        umap[fate.valid, 0],
        umap[fate.valid, 1],
        c=fate.values[fate.valid, index],
        s=4,
    )
    axis.set_title(f"sink {label}")
    figure.colorbar(points, ax=axis)
figure.tight_layout()
figure
```

## Validation checklist

- Confirm source and sink definitions are supported by independent biological evidence.
- Check graph components and the pseudotime validity mask.
- Verify terminal probabilities peak near their corresponding sink labels.
- Compare plausible endpoint definitions when terminal states are uncertain.
- Retain the pseudotime, sink-label, and fate refs together in the analysis record.

See {doc}`pseudotime` for the ordering and {doc}`trajectory_validation` for broader diagnostics.
