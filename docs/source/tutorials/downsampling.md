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

TopACeDo selects representative cells from an explicit graph and Paris clustering. It returns one
immutable artifact with the selected mask and diagnostics. Nothing is added to cell metadata unless
you deliberately add a column for an export tool.

## 1. Build the required artifacts

TopACeDo requires the Paris cut for the same graph. A Leiden partition or a cut from another graph
is rejected.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
counts_path = Path(analysis_directory.name) / "counts.zarr"
analysis_path = Path(analysis_directory.name) / "downsampling.zarr"
repack_store(f"{dataset}/data.zarr", str(counts_path), nthreads=2)
ds = scarf.mount_datastore(
    str(counts_path),
    at=str(analysis_path),
    default_assay="RNA",
    nthreads=4,
)

cells = ds.snapshot_cell_selection(cell_key="I")
features = ds.select_hvgs(cells, min_cells=20, top_n=500, show_plot=False)
normalized = ds.run_normalization(cells, features)
pca = ds.run_pca(normalized, dims=15)
initialization = ds.build_embedding_initialization(pca)
neighbors = ds.query_neighbors(ds.build_ann_index(pca), k=11)
graph = ds.build_connectivity_map(neighbors)
umap = ds.run_umap(graph, initialization, n_epochs=100)
paris = ds.run_paris_clustering(graph)
```

`paris` is an exact `cluster_cut` ref. Inspect its labels through the dedicated loader:

```{code-cell} ipython3
paris_result = ds.load_paris_clustering(paris)
pd.Series(paris_result.labels).value_counts().sort_index()
```

## 2. Sample and inspect the artifact

```{code-cell} ipython3
sampling = ds.run_topacedo_sampler(
    graph,
    paris,
    max_sampling_rate=0.1,
)
sampling_data = ds.load_artifact(sampling)
sampled = np.asarray(sampling_data["sampled"][:], dtype=bool)
seeds = np.asarray(sampling_data["seeds"][:], dtype=bool)
density = np.asarray(sampling_data["density"][:])
mean_snn = np.asarray(sampling_data["mean_snn"][:])
{
    "cells": int(sampled.size),
    "selected": int(sampled.sum()),
    "seeds": int(seeds.sum()),
}
```

The payload also contains `edges`, the sampled Steiner-tree edges in graph row coordinates. All
arrays use the captured cell-selection order.

```{code-cell} ipython3
summary = pd.DataFrame(
    {
        "cluster": paris_result.labels,
        "sampled": sampled,
        "seed": seeds,
        "density": density,
        "mean_snn": mean_snn,
    }
)
summary.groupby("cluster", sort=True).agg(
    cells=("sampled", "size"),
    selected=("sampled", "sum"),
    seeds=("seed", "sum"),
)
```

```{code-cell} ipython3
coordinates = np.asarray(ds.load_artifact(umap)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(coordinates[:, 0], coordinates[:, 1], s=3)
axes[0].set_title("All selected cells")
axes[1].scatter(coordinates[sampled, 0], coordinates[sampled, 1], s=3)
axes[1].set_title("TopACeDo sample")
figure.tight_layout()
```

Downsampling preserves graph coverage. It does not make the sample a statistically interchangeable
replacement for the complete dataset.

## 3. Export the selected cells

`SubsetZarr` currently selects cells by a boolean metadata column. Add one explicitly, then export.
This mutation is a user-owned handoff step, not a side effect of TopACeDo.

```{code-cell} ipython3
ds.cells.insert(
    "topacedo_selected",
    sampled,
    key="I",
    fill_value=False,
    overwrite=True,
)

subset_path = Path(analysis_directory.name) / "subset.zarr"
writer = scarf.SubsetZarr(
    zarr_loc=str(subset_path),
    assays=[ds.RNA],
    cell_key="topacedo_selected",
    reset_cell_filter=False,
    overwrite_existing_file=True,
)
writer.dump()

subset = scarf.DataStore(str(subset_path))
subset.cells.N, subset.RNA.feats.N
```

`SubsetZarr` retains every feature in the listed assays. Use `to_anndata` when you need an in-memory
handoff with both axes constrained.

## Common mistakes

- Passing clusters that do not come from `run_paris_clustering`
- Passing a Paris cut built from a different graph or cell selection
- Looking for sampler-created cell columns instead of loading the returned ref
- Interpreting a topology-preserving sample as an unbiased quantitative subsample
