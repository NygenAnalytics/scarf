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

## 1. Build pseudotime and sink labels

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

preparation = ds.pipeline.run(
    filtering=False,
    hvg_count=2000,
    pca_dims=15,
    neighbors_k=11,
    umap=False,
    leiden=False,
    cell_cycle=False,
    paris=False,
    doublets=False,
    markers=False,
)
graph = preparation["connectivity_map"]

annotations = np.asarray(ds.cells.fetch("clusters", key="I"))
source = annotations == "Ductal"
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any() or np.any(source & sink):
    raise ValueError("Source and sink annotations must be non-empty and disjoint")
source_sink_vector = np.zeros(len(annotations), dtype=float)
source_sink_vector[source] = 1.0 / int(source.sum())
source_sink_vector[sink] = -1.0 / int(sink.sum())

pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
sink_labels_ref = ds.run_leiden_clustering(graph, resolution=1.0)
sink_labels = np.asarray(ds.load_artifact(sink_labels_ref)["values"][:])
```

The `clusters` labels and `RNA_UMAP*` columns used on this page are prepared catalog metadata copied
by the mount. The new pseudotime, Leiden, and fate results remain exact artifacts.

For this executable mechanics example, choose the two clusters with the greatest mean valid
pseudotime as candidate terminal labels. A real analysis should choose and validate endpoints from
study-specific evidence.

```{code-cell} ipython3
valid_frame = pd.DataFrame(
    {
        "label": sink_labels[pseudotime.valid],
        "pseudotime": pseudotime.values[pseudotime.valid],
    }
)
terminal_labels = (
    valid_frame.groupby("label")["pseudotime"].mean().nlargest(2).index.tolist()
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
umap = ds.cells.to_pandas_dataframe(["RNA_UMAP1", "RNA_UMAP2"], key="I")
figure, axes = plt.subplots(1, len(fate.sink_labels), figsize=(9, 4))
for axis, index, label in zip(
    np.atleast_1d(axes),
    range(len(fate.sink_labels)),
    fate.sink_labels,
    strict=True,
):
    points = axis.scatter(
        umap.loc[fate.valid, "RNA_UMAP1"],
        umap.loc[fate.valid, "RNA_UMAP2"],
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
