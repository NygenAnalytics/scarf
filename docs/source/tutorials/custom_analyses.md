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

With the unique way Scarf handles the data in a memory efficient manner, it would not be an ill assumption to make that SCARF has limited support for external tools and data formats. However, since Scarf is able to expose the graph information, the count streams, and the metadata tables used, exporting to alternative formats so that external algorithms can participate in analysis is not a difficult task. This enables Scarf to be flexible. In the tutorial today, we will generally learn how to perform the analysis in a more memory efficient manner, select a custom group of cells, and then learn how to export various selections/ the dataset in general to other formats for other analysis systems.

## Import an existing store

The tutorial uses a prepared analysis of PBMCs as an example.

```{code-cell} ipython3
import numpy as np

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
graph_ref = run["connectivity_map"]
ds
```

## Calculate from the graph

Say we need to first calculate specific statistics about the graph, for example each cell's total edge weight in the graph. This can tell us which cells sit in dense, well-connected neighborhoods versus cells that are weakly connected and similar to other cells based on the graph. To calculate this, we can use `load_graph`, which returns the selected neighbourhood graph as a SciPy CSR matrix, in which the metric can then actually be calculated. Here the row sum measures each cell's total edge weight in the graph; this is not a biological metric.

```{code-cell} ipython3
graph = ds.load_graph(graph=graph_ref, symmetric=True, upper_only=False)
graph_strength = np.asarray(graph.sum(axis=1)).ravel()
ds.cells.insert(
    column_name="customGraphStrength", values=graph_strength, key="I", overwrite=True
)
{
    "cells": int(graph.shape[0]),
    "edges": int(graph.nnz),
    "customGraphStrength mean": float(graph_strength.mean()),
}
```

The insert writes one value per active cell in graph row order, with the summar showing the  graph size and the mean of the new column.

```{code-cell} ipython3
ds.plots.embedding(layout=run["umap"], color_by="customGraphStrength")
```

The plot simply lets us visualize which cells have stronger or weaker weighted connectivity in this specific graph.

## Stream count blocks

If you are under heavier memory constraints, you can also stream the count information to make SCARF even more memory efficient. One way to do this is to avoid using `.compute()` on a matrix that may exceed memory. We can also slice blocks to the frozen run's cell and highly variable feature indexes, then process the ordered row blocks.
This example counts detected HVGs per active cell:

```{code-cell} ipython3
cell_index = np.flatnonzero(run.cells.fetch_all("I"))
hvg_values = np.asarray(run.features.fetch_all("highly_variable_features"), dtype=bool)
feature_index = np.flatnonzero(hvg_values)
selected_counts = ds.RNA.rawData[:, feature_index][cell_index, :]

{"selected cells": len(cell_index), "selected genes": len(feature_index)}
```

Count detected genes one block at a time.

```{code-cell} ipython3
detected_blocks = []
for count_block in selected_counts.stream_blocks(
    nthreads=4, msg="Calculating custom detection statistic"
):
    detected_blocks.append(np.count_nonzero(count_block, axis=1))
sum(len(block) for block in detected_blocks)
```

Collect the per-cell results and save them in metadata.

```{code-cell} ipython3
detected_hvgs = np.concatenate(detected_blocks)
ds.cells.insert(
    column_name="customDetectedHVGs", values=detected_hvgs, key="I", overwrite=True
)
{
    "cells": int(detected_hvgs.size),
    "customDetectedHVGs mean": float(detected_hvgs.mean()),
    "customDetectedHVGs max": int(detected_hvgs.max()),
}
```

The `stream_blocks` preserves row order. Because active `I` is the frozen run selection, inserting with hat key keeps each streamed value aligned with its metadata row and doesn't make realigning things an issue.

## Create custom selections

It can also be useful during parts of the analysis to create custom selections for a group of cells. To do this, we can create a boolean cell column that is `cell_key`. We can then use `fill_value=False` when the new key is defined only for currently active cells. This is one of the strong benefits of Scarf, is that any cells we select or deselect do not have their information deleted; it is simply no longer read into memory and still saved if we need to revert in the future.

For our example here, we keep cells above the lowest quarter of graph strength as an example.
This is simply an illustration of the API, not a recommended quality-control filter that you can or should implement:

```{code-cell} ipython3
well_connected = graph_strength >= np.quantile(graph_strength, 0.25)
ds.cells.insert(
    column_name="wellConnected",
    values=well_connected,
    fill_value=False,
    key="I",
    overwrite=True,
)
{"selected cells": int(well_connected.sum()), "active cells": len(well_connected)}
```

```{code-cell} ipython3
ds.plots.embedding(
    layout=run["umap"], color_by=["customDetectedHVGs", "wellConnected"], n_columns=2
)
```

The first panel checks where the streamed count statistic varies. The second shows the lower-quartile graph-strength exclusion created from the same active cell order.

## Pass a small selection to another tool

Say for example we have created our selection now, and would liek to pass it to another analysis tool such as Scanpy. We can use the `to_anndata` function create an in-memory AnnData object. Note, by doing this, you often will need memory or need to be careful as this can be memory intensive. Select the cells and genes you need before actually executing it; here we only pass the cells alongside information of the 6 genes listed:

```{code-cell} ipython3
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
adata = ds.to_anndata(cell_key="wellConnected", feature_names=panel_genes)
adata.shape, adata.var_names.tolist()
```

If you persay wanted to export all of the genes, then simply leave out `feature_names`, and it will automatically default to including them all.

When transfering using `to_anndata`, it functions (for transfering genes) by indexing `var` by gene ids (Ensembl here); gene symbols stay in`var["names"]`. Therefore `adata.var_names`after`feature_names=panel_genes` lists ids, not the panel symbols, and would need to be translated back possible. Check`adata.var["names"]`when you need the symbols.

If you want to subset and stay inside of the Scarf ecosystem, instead use `SubsetZarr` to write selected cells to a Scarf store instead.

Write the `wellConnected` cells into a separate Scarf store and confirm what the subset kept.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

export_directory = TemporaryDirectory()
subset_path = Path(export_directory.name) / "well_connected.zarr"
writer = scarf.SubsetZarr(
    zarr_loc=str(subset_path),
    assays=[ds.RNA],
    cell_key="wellConnected",
    reset_cell_filter=False,
    overwrite_existing_file=True,
)
writer.dump()

subset = scarf.DataStore(str(subset_path))
{"exported cells": subset.cells.N, "retained genes": subset.RNA.feats.N}
```

`SubsetZarr` keeps every gene in the listed assays and writes only the selected cells, so the subset stays a full Scarf store rather than a reduced in-memory object.
