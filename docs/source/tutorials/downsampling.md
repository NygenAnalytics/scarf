---
description: Select representative cells with TopACeDo and export a deliberate subset.
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

# Cell downsampling

A smaller set of representative cells can make a large dataset easier to explore or pass to
another tool. TopACeDo uses the neighbourhood graph and a Paris clustering to choose cells that
cover its structure. It saves the selected cells as an artifact; it does not change the source data.

## 1. Open the required artifacts

TopACeDo requires a Paris cut from the same graph. The rebuilt PBMC store contains a completed
example run labeled `docs_default` and a 15-cluster Paris cut built from its graph. Calling
`run_paris_clustering` with that graph and cut size reuses the exact stored result. A Leiden
partition or a cut from another graph is rejected.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example, including its saved analysis.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
```

Open the downloaded store and its saved analysis.

```{code-cell} ipython3
# Open the datastore for the following analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Open the saved analysis and retain its exact results.
run = ds.pipeline.open(label="docs_default")
# Keep the graph used by the saved analysis.
graph = run["connectivity_map"]
# Reuse the saved UMAP coordinates.
umap = run["umap"]
# Reuse the 15-cluster Paris cut of this graph.
paris = ds.run_paris_clustering(graph, n_clusters=15)
# Inspect the opened assays and their dimensions.
ds
```

`paris` is an exact `cluster_cut` ref. Inspect its labels through the dedicated loader:

```{code-cell} ipython3
# Load the Paris labels and hierarchy diagnostics.
paris_result = ds.load_paris_clustering(paris)
# Count cells in the 15 Paris groups used for sampling.
cluster_counts = pd.Series(paris_result.labels, name="cluster").value_counts()
# Inspect group sizes in cluster-label order.
cluster_counts.sort_index().to_frame("cells")
```

## 2. Choose representative cells

```{code-cell} ipython3
# Select representative cells from the graph and Paris cut.
sampling = ds.run_topacedo_sampler(graph, paris)
# Open the saved sampling result.
sampling_data = ds.load_artifact(sampling)
# Read the sampling mask in the graph's cell order.
sampled = np.asarray(sampling_data["sampled"][:], dtype=bool)
# Compare the sampled-cell count with the full graph size.
{"cells": int(sampled.size), "selected": int(sampled.sum())}
```

The default 5% rate controls seed selection within each cluster, with a minimum number of
seeds per cluster. The sampler then adds cells that connect those seeds through the graph.
The final sample can therefore exceed 5%. Check the selected count above; this is not a
request for an exact sample size.

```{code-cell} ipython3
# Read the saved embedding in that same order.
coordinates = np.asarray(ds.load_artifact(umap)["values"][:])
# Create one plotting axis for each comparison panel.
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
# Plot these cells using the selected coordinates and values.
axes[0].scatter(coordinates[:, 0], coordinates[:, 1], s=3)
# Label the panel with the quantity being compared.
axes[0].set_title("All selected cells")
# Plot these cells using the selected coordinates and values.
axes[1].scatter(coordinates[sampled, 0], coordinates[sampled, 1], s=3)
# Label the panel with the quantity being compared.
axes[1].set_title("TopACeDo sample")
# Adjust spacing so panel labels remain readable.
figure.tight_layout()
```

Downsampling preserves graph coverage. It does not make the sample a statistically interchangeable
replacement for the complete dataset.

## 3. Export the selected cells

The sampler mask follows the graph's compact cell-selection order. Map it back to physical row
indices, then pass those rows directly to `SubsetZarr`. No temporary metadata column is needed.

```{code-cell} ipython3
# Locate graph cells on the original metadata axis.
graph_cell_indices = np.flatnonzero(run.cells.fetch_all("I"))
# Map the sampling mask back to physical cell rows.
sampled_cell_indices = graph_cell_indices[sampled]

# Check how many physical cell rows will be exported.
{"cells to export": len(sampled_cell_indices)}
```

Write those physical rows into a separate store.

```{code-cell} ipython3
# Keep the exported example in a temporary folder.
export_directory = TemporaryDirectory()
# Choose a new path for the selected-cell store.
subset_path = Path(export_directory.name) / "subset.zarr"
# Prepare a writer for the selected counts and metadata.
writer = scarf.SubsetZarr(
    zarr_loc=str(subset_path),
    assays=[ds.RNA],
    cell_idx=sampled_cell_indices,
    reset_cell_filter=False,
    overwrite_existing_file=True,
)
# Write the prepared counts and metadata to the new store.
writer.dump()

# Reopen the exported store to check its dimensions.
subset = scarf.DataStore(str(subset_path))
# Check the exported cell and feature counts.
{"exported cells": subset.cells.N, "retained genes": subset.RNA.feats.N}
```

`SubsetZarr` retains every feature in the listed assays. Use `to_anndata` when you need an in-memory
handoff with both axes constrained.

## 4. Inspect coverage and adjust the sample

Check how many cells were retained from each Paris cluster before choosing a different rate:

```{code-cell} ipython3
# Pair each Paris label with its sampling status.
summary = pd.DataFrame({"cluster": paris_result.labels, "sampled": sampled})
# Compare retained cells with the original size of each Paris group.
summary.groupby("cluster")["sampled"].agg(cells="size", selected="sum")
```

If you need a larger sample, pass `max_sampling_rate=0.1` to raise the maximum seed-selection
rate to 10% per cluster. The minimum-per-cluster rule and added connecting cells still affect
the final size. Compare coverage on the original layout after changing the rate.

The artifact also contains seed cells, density estimates, shared-neighbour summaries, and the
selected graph edges. These are available through `load_artifact` when you need to inspect how
the sampler made its choices.

## Common mistakes

- Passing clusters that do not come from `run_paris_clustering`
- Passing a Paris cut built from a different graph or cell selection
- Looking for sampler-created cell columns instead of loading the returned ref
- Interpreting a topology-preserving sample as an unbiased quantitative subsample
