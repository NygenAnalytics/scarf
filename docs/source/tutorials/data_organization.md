---
description: Read cell and feature metadata, understand selections, and inspect saved results.
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
(data_organization)=

# Data organization

A datastore is where everything in a Scarf analysis lives, from the raw counts to the cell and gene metadata, and the graphs built from them, and every saved result. Everything is all stored in one Zarr directory on disk rather than in memory. Scarf keeps the data on disk in chunks and streams only the chunks required at each step to ensure that the analysis can be performed in a memory efficient manner. In the tutorial, you will come to gain a sense of how data organization inside of Scarf works, how to inspect results, and then inspect the Zarr folder hierarchy when you need to pull specific data.

## Dataset

For each new dataset, your `DataStore` is the main entry point of analysis. Each assay contains the counts, the feature metadata, the computationally guided results, and feature selection subset. Cell-level columns are shared across assays. The boolean cell column `I` marks the active cells used by live-metadata methods. Saved runs keep their own frozen cell selection.

```{mermaid}
flowchart TB
    ds["DataStore"]
    cells["Shared cell metadata"]
    rna["RNA assay<br/>feature metadata and results<br/>(default assay)"]
    atac["ATAC assay<br/>feature metadata and results"]
    ds --> cells
    ds --> rna
    ds --> atac
```

By design, Scarf selects the default assay (typically RNA) whenever the user command omits `from_assay`, and this only picks which assay the method acts on; it never merges the assay-specific feature tables, selections, or results into a combined view.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)
```

This page uses the pre-analyzed Bastidas-Ponce pancreas store also used in {doc}`plotting` and {doc}`cell_cycle`. The rebuilt store uses the current layout and contains a completed pipeline run named `docs_default`. Here, we open it directly and reuse the run's exact selections, clustering, UMAP, and
markers to demonstrate how Scarf works.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
```

Open the downloaded store and its saved analysis.

```{code-cell} ipython3
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
analysis_run = ds.pipeline.open(label="docs_default")

ds

cluster_values = analysis_run.cells.fetch("clusters")
cluster_counts = pd.Series(cluster_values, name="cluster").value_counts()
cluster_counts.sort_index().to_frame("cells")
```

The selected clustering is completed and remains in its exact artifact. The catalog's literal `clusters` column is
an imported cell-type annotation.

## Read cell and feature metadata

Cell and feature tables are `MetaData` objects (`ds.cells`, `ds.RNA.feats`), not pandas DataFrames, thus manipulating them takes a different path. Use `head` for a quick look at the information, `to_pandas_dataframe` to export selected columns for more detail, and `fetch` / `fetch_all` to pull data for specific (singular) columns.

```{code-cell} ipython3
ds.cells.head()[["ids", "I", "clusters"]]
```

```{code-cell} ipython3
ds.RNA.feats.head()
```

```{code-cell} ipython3
cell_qc = ds.cells.to_pandas_dataframe(
    columns=["ids", "RNA_nCounts", "RNA_nFeatures", "clusters"]
)
cell_qc.set_index("ids").head()
```

The `insert` function writes a new column and aligns values to the active subset unless you override `key`. Re-inserting an existing column requires `overwrite=True`.
In this prepared store, every cell is active and the saved run contains the same cells, so its
cluster labels align with the values we insert. For another analysis, check the cell selection
before copying results into live metadata to minimize error.

```{code-cell} ipython3
first_run_cluster = cluster_values[0]
is_first_cluster = cluster_values == first_run_cluster
ds.cells.insert(
    column_name="is_first_cluster", values=is_first_cluster, overwrite=True
)
pd.Series(is_first_cluster).value_counts().rename("cells")
```

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=["ids", "clusters", "is_first_cluster"]
).head()
```

```{code-cell} ipython3
ds.plots.embedding(layout=analysis_run["umap"], color_by="is_first_cluster")
```

The new column marks one cluster on the same active cells used for the insert. The `fetch` function returns values for the active subset (default column `I`), whereas `fetch_all` returns every row in the store. With every cell active the lengths match. If you want to select a specific cluster, you can pass the new Boolean column as `key` to select one cluster without changing `I`:

```{code-cell} ipython3
print(
    "fetch:",
    ds.cells.fetch("clusters", key="is_first_cluster").shape,
    "fetch_all:",
    ds.cells.fetch_all("clusters").shape,
)
```

## Select cells from metadata

The `sift` function returns a boolean mask for one numeric range selecting cells; `multi_sift` combines several ranges, and `get_index_by` locates exact categorical values of where cells or values are located:

```{code-cell} ipython3
active_before = int(ds.cells.fetch_all("I").sum())
count_range = ds.cells.sift("RNA_nCounts", min_v=1000, max_v=15000)
{"cells in count range": int(count_range.sum())}
```

Combine count and detected-gene limits.

```{code-cell} ipython3
joint_range = ds.cells.multi_sift(
    columns=["RNA_nCounts", "RNA_nFeatures"],
    lows=[1000, 500],
    highs=[15000, 4000],
)
{"cells in both QC ranges": int(joint_range.sum())}
```

Find rows based on their published annotations.

```{code-cell} ipython3
ductal_rows = ds.cells.get_index_by(["Ductal"], "clusters")

print("Ductal cells:", int(ductal_rows.size))
print("Active cells (I) before:", active_before)
print("Active cells (I) after:", int(ds.cells.fetch_all("I").sum()))
```

```{code-cell} ipython3
ds.cells.insert(column_name="in_count_range", values=count_range, overwrite=True)
ds.plots.embedding(layout=analysis_run["umap"], color_by="in_count_range")
```

These helpers return masks or indexes aligned with the metadata table, and they do not modify `I` until you explicitly insert or update a cell key. They can serve as useful tools when subsetting data. See the `MetaData` API in {doc}`../reference/api/assays` for more information on how the helpers work.

## Count matrices and normalization

Raw counts are a Zarr array (often sharded), exposed as `rawData`, a chunked array with a NumPy-like interface that streams by row. In this store the array is at `RNA/counts`. RNA assays also store `countsT`, a gene-major copy used by HVG and marker stages. 

Normalized values are computed on demand through the lower-level assay `normed()` view from raw counts. `run_normalization(cell_selection, features)` is what saves the normalized counts into the Zarr path and requires exact stored cell- and feature-selection references. The direct `normed()` view follows its explicit or literal metadata indexes; in a newly created store the physical feature `I` column is all true. Inspect the shapes without loading the complete matrix:

```{code-cell} ipython3
print("Raw shape:", ds.RNA.rawData.shape)
print("Normed shape:", ds.RNA.normed().shape)
```

## Inspect saved analysis results

Analysis methods return lightweight references to results stored in Zarr. The completed run retained the HVG and normalization references it created; if you seek to ask for the same normalization results again, Scarf can simply reuse the results rather than recomputing:

```{code-cell} ipython3
reused_normalized = ds.run_normalization(
    analysis_run["analysis_cell_selection"],
    analysis_run["highly_variable_features"],
)
print("Reused:", reused_normalized == analysis_run["normalized"])
reused_normalized
```

Inspect its status and open the underlying group only when a custom method or analysis needs direct access to the information.

```{code-cell} ipython3
status = ds.inspect_artifact(reused_normalized)
print("Complete:", status.complete)
print("Operation:", status.operation)
print("Method:", status.parameters["normalization_method"]["qualname"])
normalization_settings = {
    key: status.parameters[key]
    for key in ("size_factor", "log_transform", "renormalize_subset")
}

group = ds.load_artifact(reused_normalized)
print("Arrays:", list(group.array_keys())[:5])
pd.Series(normalization_settings, name="value").rename_axis("setting").to_frame()
```

## Look inside the store

Scarf uses [Zarr](https://zarr.readthedocs.io/en/stable/) for chunked on-disk arrays, with the central Zarr store being a directory tree with the following tidbits to it: counts, cell and feature attributes, and unchangeable artifacts live
under named groups. Relative to a single HDF5 file, the layout supports parallel reads and writes, fast compression codecs, and automatic persistence of intermediate results.

Running `show_zarr_tree` prints the hierarchy so you can visualize it yourself. Modulating the depth allows you to see the tree in different levels of detail, with `depth=0` you see the top-level assays and `cellData`.

```{code-cell} ipython3
ds.show_zarr_tree(depth=0)
```

Cell statistics computed from an assay are stored under `cellData` with the assay name as a prefix (`RNA_…`, `ADT_…`).

```{code-cell} ipython3
metadata_arrays = {
    name: ds.zw["cellData"][name]
    for name in ("I", "RNA_nCounts", "RNA_nFeatures")
}
pd.DataFrame({
    name: {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "chunks": array.chunks,
    }
    for name, array in metadata_arrays.items()
}).T.rename_axis("cellData column")
```

**The `I` column** is the default {term}`cell key`, which once again, tracks which cells are active for
live-metadata APIs. Values are boolean, so instead of deleting cells, they simply are not loaded in for use. You can see the number of cells by:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(["I"])["I"].value_counts()
```

This store keeps every barcode active (`True`). Analytical filtering returns a separate
selection artifact and leaves this column unchanged. If you deliberately modify a live selection
column, its `False` rows also remain in the table rather than being deleted.

Each assay group holds `featureData` and its persisted artifacts.
Count matrices are Zarr arrays, often sharded. This store keeps RNA counts at `RNA/counts`.

```{code-cell} ipython3
ds.show_zarr_tree(start="RNA", depth=1)
```


## Zarr versions and storage profiles

Current Scarf versions write new datasets as Zarr v3. RNA assays also write a gene-major `countsT` copy next to `counts` (`countsT` is simply a transposed version of the counts). If you have older stores with Scarf, then re-import older RNA stores that use Zarr v2 or lack `countsT` before analysis.

Count matrices from the writers use sharded arrays (default profile `fast_local`). You can modify this with `SCARF_ZARR_PROFILE` (`fast_local` or `cloud`) or `zarrProfile=` when opening a `DataStore`. Selecting cloud simply switches the arrays to a different level of compression regardless of whether the store is on cloud storage or not.

Storage profiles and conversion belong to the physical store, while `mem_budget` and `nthreads` control the budget of executing cells.

## Common mistakes

- Expecting filtering to delete rows or rewrite live `I`. Filtering only returns a separate selection artifact and leaves `I` unchanged, so check the returned artifact or explicitly update a live column with `insert` when you actually want to change the active set.
- Treating `MetaData` as an in-memory pandas DataFrame. `ds.cells` and `ds.RNA.feats` are on-disk metadata objects, so preview them with `head` and pull values out with `to_pandas_dataframe`, `fetch`, or `fetch_all` instead of applying pandas operations directly.
- Using `fetch` when values for inactive cells are also required. `fetch` returns only the active subset while `fetch_all` returns every row in the store, so switch to `fetch_all` whenever your question covers cells outside the current selection.
- Treating a result reference as an in-memory matrix. Analysis methods hand back a lightweight `ArtifactRef`, so pass that reference along or open its values with `load_artifact` instead of indexing it like an array.
- Editing artifact groups directly instead of using Scarf's analysis methods. Direct edits bypass provenance tracking and break reuse, so rerun the corresponding method or producer to create a new recorded artifact.
- Expecting an older RNA Zarr v2 store, or one without `countsT`, to open without re-importing it. The current version requires the Zarr v3 layout with both count orientations, so re-import the source data through the matching writer before analysis.

Metadata changes and artifacts are written into the Zarr store.
Low-level layout details intended for contributors remain in {doc}`../developers/zarr_internals`.
