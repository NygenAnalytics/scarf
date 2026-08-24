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
- Write custom cell and feature selections
- Register external feature loadings as a reduction branch
- Choose an exit path for another analysis system

## 1. Prepare a store

The published PBMC store is enough for the examples below.
Only the graph-statistic and `wellConnected` path uses `load_graph`.
Streaming, `set_hvgs`, custom reduction, `to_anndata`, and `SubsetZarr` do not.
See {doc}`graph_construction` to build a graph by hand.

```{code-cell} ipython3
import numpy as np
import scarf

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
```

## 2. Calculate from the graph

`load_graph` returns the selected neighbourhood graph as a SciPy CSR matrix.
Here the row sum measures each cell's total edge weight in the symmetric graph.
It is a graph statistic, not a biological confidence score.

```{code-cell} ipython3
graph = ds.load_graph(
    from_assay="RNA",
    cell_key="I",
    feat_key="hvgs",
    symmetric=True,
    upper_only=False,
)
graph_strength = np.asarray(graph.sum(axis=1)).ravel()
ds.cells.insert(
    column_name="customGraphStrength",
    values=graph_strength,
    key="I",
    overwrite=True,
)
{
    "cells": int(graph.shape[0]),
    "edges": int(graph.nnz),
    "customGraphStrength mean": float(graph_strength.mean()),
}
```

The insert writes one value per active cell in graph row order.
The summary confirms the CSR cover and that the new column is populated.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="customGraphStrength",
    sort_values=True,
)
```

The plot asks where cells have stronger or weaker weighted connectivity in this specific graph.
Rebuilds with another feature set or neighbour count need a new statistic.

## 3. Stream count blocks

Avoid `.compute()` on a matrix that may exceed memory.
Slice to the intended cell and feature keys, then process ordered row blocks.
This example counts detected HVGs per active cell:

```{code-cell} ipython3
cell_index = ds.cells.active_index("I")
feature_index = ds.RNA.feats.active_index("I__hvgs")
selected_counts = ds.RNA.rawData[:, feature_index][cell_index, :]

detected_blocks = []
for count_block in selected_counts.stream_blocks(
    nthreads=4,
    msg="Calculating custom detection statistic",
):
    detected_blocks.append(np.count_nonzero(count_block, axis=1))

detected_hvgs = np.concatenate(detected_blocks)
ds.cells.insert(
    column_name="customDetectedHVGs",
    values=detected_hvgs,
    key="I",
    overwrite=True,
)
{
    "cells": int(detected_hvgs.size),
    "customDetectedHVGs mean": float(detected_hvgs.mean()),
    "customDetectedHVGs max": int(detected_hvgs.max()),
}
```

`stream_blocks` preserves row order.
Insert with the same `cell_key` used to construct the view so values align with metadata rows.
The summary checks that every streamed cell received a detection count.

## 4. Create custom selections

A boolean cell column can become a `cell_key`.
Use `fill_value=False` when the new key is defined only for currently active cells:

```{code-cell} ipython3
well_connected = graph_strength >= np.quantile(graph_strength, 0.25)
ds.cells.insert(
    column_name="wellConnected",
    values=well_connected,
    fill_value=False,
    key="I",
    overwrite=True,
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=["customDetectedHVGs", "wellConnected"],
    n_columns=2,
    sort_values=True,
)
```

The first panel checks where the streamed count statistic varies.
The second shows the lower-quartile graph-strength exclusion created from the same active cell order.

Install a supplied RNA feature mask with `set_hvgs`.
This records the cell-selection relationship and produces the feature key `wellConnected__customPanel`.
Downstream methods take the short name `customPanel` as `feat_key`:

```{code-cell} ipython3
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
feature_names = ds.RNA.feats.fetch_all("names").astype(str)
panel_mask = np.isin(feature_names, panel_genes)
custom_feature_key = ds.set_hvgs(
    cell_key="wellConnected",
    mask=panel_mask,
    hvg_key_name="customPanel",
    blacklist="",
)
custom_feature_key, int(ds.RNA.feats.fetch_all(custom_feature_key).sum())
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=panel_genes,
    cell_key="wellConnected",
    n_columns=3,
    sort_values=True,
)
```

The returned key is the full feature column.
The plot checks the same panel on the `wellConnected` cells that the selection is linked to.

## 5. Register external feature loadings

`run_custom_reduction` accepts an external feature-by-dimension loading matrix.
Its rows must match the selected normalized features in order.
It projects Scarf's normalized cell blocks through those loadings and records a reusable reduction artifact.
Here an identity matrix stands in for loadings from an external tool:

```{code-cell} ipython3
branch_normalized = ds.run_normalization(
    cell_key="wellConnected",
    feat_key="customPanel",
    update_state=False,
)
n_features = int(ds.load_artifact(branch_normalized)["data"].shape[1])
external_loadings = np.eye(n_features, dtype=np.float64)

custom_reduction = ds.run_custom_reduction(
    external_loadings,
    branch_normalized,
    update_state=False,
)
custom_ann = ds.build_ann_index(
    custom_reduction,
    update_state=False,
)
ds.load_artifact(custom_reduction)["data"].shape, custom_ann
```

`update_state=False` keeps this experiment as a side branch, outside the assay's {term}`analysis chain`.
Pass returned references explicitly through later graph-construction steps.
Select a branch as current only after checking its outputs.
Replace the identity matrix with real loadings when an external method supplies them; the row count must still match `n_features`.

## 6. Choose an exit path

Confirm a feature-selective in-memory handoff with `to_anndata`, then a cell-selective Scarf store with `SubsetZarr`:

```{code-cell} ipython3
adata = ds.to_anndata(
    cell_key="wellConnected",
    feature_names=panel_genes,
)
adata.shape, adata.var_names.tolist()
```

```{code-cell} ipython3
from pathlib import Path
import shutil

subset_path = Path("scarf_datasets/custom_analyses_subset.zarr")
if subset_path.exists():
    shutil.rmtree(subset_path)

writer = scarf.SubsetZarr(
    zarr_loc=str(subset_path),
    assays=[ds.RNA],
    cell_key="wellConnected",
    reset_cell_filter=False,
    overwrite_existing_file=True,
)
writer.dump()

subset_ds = scarf.DataStore(str(subset_path))
{
    "source wellConnected cells": int(ds.cells.fetch_all("wellConnected").sum()),
    "subset cells": subset_ds.cells.N,
    "subset RNA features": subset_ds.RNA.feats.N,
}
```

`to_anndata` drops unselected features.
It indexes `var` by gene ids (Ensembl here); gene symbols stay in `var["names"]`.
So `adata.var_names` after `feature_names=panel_genes` lists ids, not the panel symbols.
Check `adata.var["names"]` when you need the symbols.
`SubsetZarr` keeps every RNA feature and only the selected cells.
For full-assay `to_h5ad` / `to_mtx` writers, see {doc}`import_and_export`.
Use {doc}`remote_stores` when the count source itself must remain remote.

## Extension boundary

Direct arbitrary artifact writing is not a stable public extension API.
Do not mutate `ds.z`, `ds.zw`, `_matrix_z`, or assay-state attributes from analysis code.
Those objects expose implementation layout and can change as storage contracts evolve.

Use public metadata insertion, result-returning methods, graph loading, block streams, and export APIs.
Pipeline callbacks provide read-only execution events; their contract is documented in {doc}`../reference/api/pipeline`.
