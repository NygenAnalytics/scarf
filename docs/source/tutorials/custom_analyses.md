---
description: Read Scarf graphs and count blocks, add custom results, and choose a supported export path.
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

(custom_analyses)=

# Extending Scarf with custom analyses

Scarf exposes graphs, bounded count streams, metadata tables, and export formats so an external algorithm can participate in an analysis without depending on private storage internals.

## What you will learn

- Load a supported neighbourhood graph and calculate a cell statistic
- Stream selected count blocks when the matrix cannot fit in memory
- Save a custom cell selection
- Choose an exit path for another analysis system

## 1. Prepare a store

The prepared PBMC store supplies counts and a saved example run labeled `docs_default`.
Open the downloaded store directly because the examples write custom artifacts and metadata, then
reuse the run's frozen selection, graph, feature selection, and UMAP. The prepared store's active
`I` matches the run's analysis selection.
See {doc}`graph_construction` to build a graph by hand.

```{code-cell} ipython3
import numpy as np

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
# Keep the reference to the saved connectivity graph.
graph_ref = run["connectivity_map"]
# Inspect the opened assays and their dimensions.
ds
```

## 2. Calculate from the graph

`load_graph` returns the selected neighbourhood graph as a SciPy CSR matrix.
Here the row sum measures each cell's total edge weight in the symmetric graph.
It is a graph statistic, not a biological confidence score.

```{code-cell} ipython3
# Load the symmetric connectivity graph as a sparse matrix.
graph = ds.load_graph(graph=graph_ref, symmetric=True, upper_only=False)
# Sum the edge weights connected to each cell.
graph_strength = np.asarray(graph.sum(axis=1)).ravel()
# Save the values in cell metadata using the stated selection.
ds.cells.insert(
    column_name="customGraphStrength", values=graph_strength, key="I", overwrite=True
)
# Summarize the graph size and the new per-cell connectivity statistic.
{
    "cells": int(graph.shape[0]),
    "edges": int(graph.nnz),
    "customGraphStrength mean": float(graph_strength.mean()),
}
```

The insert writes one value per active cell in graph row order.
The summary shows the graph size and the mean of the new column.

```{code-cell} ipython3
# Locate high and low graph connectivity on the saved embedding.
ds.plots.embedding(layout=run["umap"], color_by="customGraphStrength")
```

The plot asks where cells have stronger or weaker weighted connectivity in this specific graph.
Rebuilds with another feature set or neighbour count need a new statistic.

## 3. Stream count blocks

Avoid `.compute()` on a matrix that may exceed memory.
Slice to the frozen run's cell and highly variable feature indexes, then process ordered row
blocks.
This example counts detected HVGs per active cell:

```{code-cell} ipython3
# Locate the run's cells on the stored count-matrix axis.
cell_index = np.flatnonzero(run.cells.fetch_all("I"))
# Read the run's highly variable gene selection.
hvg_values = np.asarray(run.features.fetch_all("highly_variable_features"), dtype=bool)
# Locate those genes on the stored feature axis.
feature_index = np.flatnonzero(hvg_values)
# Create a lazy view of the selected cells and genes.
selected_counts = ds.RNA.rawData[:, feature_index][cell_index, :]

# Check the selected matrix dimensions before streaming counts.
{"selected cells": len(cell_index), "selected genes": len(feature_index)}
```

Count detected genes one block at a time.

```{code-cell} ipython3
# Collect one small vector of detection counts per block.
detected_blocks = []
# Process a bounded count block without loading the full matrix.
for count_block in selected_counts.stream_blocks(
    nthreads=4, msg="Calculating custom detection statistic"
):
    # Keep this result in its original processing order.
    detected_blocks.append(np.count_nonzero(count_block, axis=1))
# Check how many cell results were collected across the blocks.
sum(len(block) for block in detected_blocks)
```

Collect the per-cell results and save them in metadata.

```{code-cell} ipython3
# Join the block results in their original cell order.
detected_hvgs = np.concatenate(detected_blocks)
# Save the values in cell metadata using the stated selection.
ds.cells.insert(
    column_name="customDetectedHVGs", values=detected_hvgs, key="I", overwrite=True
)
# Summarize the per-cell detected-gene counts.
{
    "cells": int(detected_hvgs.size),
    "customDetectedHVGs mean": float(detected_hvgs.mean()),
    "customDetectedHVGs max": int(detected_hvgs.max()),
}
```

`stream_blocks` preserves row order. Because active `I` is the frozen run selection, inserting with
that key keeps each streamed value aligned with its metadata row.
The summary checks that every streamed cell received a detection count.

## 4. Create custom selections

A boolean cell column can become a `cell_key`.
Use `fill_value=False` when the new key is defined only for currently active cells: the inactive
cells then hold `False`. Without it they are recorded as missing, which a key never selects either.
Here we keep cells above the lowest quarter of graph strength as an example of making a selection.
This is an illustration of the API, not a recommended quality-control filter:

```{code-cell} ipython3
# Keep cells above the lowest quarter of graph strength.
well_connected = graph_strength >= np.quantile(graph_strength, 0.25)
# Save the values in cell metadata using the stated selection.
ds.cells.insert(
    column_name="wellConnected",
    values=well_connected,
    fill_value=False,
    key="I",
    overwrite=True,
)
# Check how many active cells pass the custom selection.
{"selected cells": int(well_connected.sum()), "active cells": len(well_connected)}
```

```{code-cell} ipython3
# Compare detected-gene counts and the custom cell selection.
ds.plots.embedding(
    layout=run["umap"], color_by=["customDetectedHVGs", "wellConnected"], n_columns=2
)
```

The first panel checks where the streamed count statistic varies.
The second shows the lower-quartile graph-strength exclusion created from the same active cell order.

## 5. Pass a small selection to another tool

`to_anndata` creates an in-memory AnnData object. Select the cells and genes you need before
materializing it. Here we pass the custom cell selection and the six-gene panel:

```{code-cell} ipython3
# Choose a small marker panel for the comparison.
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
# Materialize only the selected cells and marker panel.
adata = ds.to_anndata(cell_key="wellConnected", feature_names=panel_genes)
# Check the exported dimensions and feature identifiers.
adata.shape, adata.var_names.tolist()
```

`to_anndata` drops unselected features.
It indexes `var` by gene ids (Ensembl here); gene symbols stay in `var["names"]`.
So `adata.var_names` after `feature_names=panel_genes` lists ids, not the panel symbols.
Check `adata.var["names"]` when you need the symbols.
Use `SubsetZarr` to write selected cells to a Scarf store instead; it retains every feature in
the chosen assays. See {doc}`downsampling` for an example and {doc}`import_and_export` for H5AD
and Matrix Market export.
Use {doc}`remote_stores` when the count source itself must remain remote.

## Extension boundary

Direct arbitrary artifact writing is not a stable public extension API.
Do not mutate `ds.z`, `ds.zw`, `_matrix_z`, or other private storage attributes from analysis code.
Those objects expose implementation layout and can change as storage contracts evolve.

Use public metadata insertion, result-returning methods, graph loading, block streams, and export APIs.
Pipeline callbacks provide read-only execution events; their contract is documented in {doc}`../reference/api/pipeline`.
