---
description: Inspect Scarf's Zarr hierarchy, count arrays, metadata, and persisted results.
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

Scarf stores counts, metadata, graphs, and analysis results in a Zarr directory.
This chapter shows how to inspect that hierarchy and how to read or update cell and feature metadata.
Low-level layout details for contributors live in {doc}`../developers/zarr_internals`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Basic familiarity with cell and feature metadata

## What you will learn

- Inspect the Zarr hierarchy
- Read and write cell or feature metadata
- Inspect a persisted normalization result
- Load a marker table through its exact artifact ref

## Dataset

`DataStore` is the main entry point.
Each assay owns feature metadata, normalization, and feature selection.
Cell-level columns are shared across assays.

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

The default assay supplies method defaults when `from_assay` is omitted.
It does not merge assay-specific feature tables or results.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)
```

This page uses the pre-analyzed Bastidas-Ponce pancreas store also used in {doc}`plotting` and {doc}`cell_cycle`.
The rebuilt store uses the current layout and contains a completed pipeline run named
`docs_default`. Open it directly and reuse that run's exact selections, clustering, UMAP, and
markers.

```{code-cell} ipython3
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
marker_ref = analysis_run["markers"]

ds
```

```{code-cell} ipython3
cluster_values = analysis_run.cells.fetch("clusters")
pd.Series(cluster_values).value_counts().sort_index()
```

The selected clustering remains in its exact artifact. The catalog's literal `clusters` column is
an imported cell-type annotation, not a copy of this analytical result. Opening the run does not
modify either one.

## 1. Inspect Zarr trees

Scarf uses [Zarr](https://zarr.readthedocs.io/en/stable/) for chunked on-disk arrays.
The store is a directory tree: counts, cell and feature attributes, and immutable artifacts live
under named groups.
Relative to a single HDF5 file, the layout supports parallel reads and writes, fast compression codecs, and automatic persistence of intermediate results.

`show_zarr_tree` prints the hierarchy.
With `depth=1` you see the top-level assays and `cellData`.

```{code-cell} ipython3
ds.show_zarr_tree(depth=1)
```

Cell statistics computed from an assay are stored under `cellData` with the assay name as a prefix (`RNA_…`, `ADT_…`).

```{code-cell} ipython3
ds.show_zarr_tree(start="cellData")
```

**The `I` column** is the default user-owned {term}`cell key`, tracking which cells are active for
live-metadata APIs. Values are boolean.
Use `snapshot_cell_selection("I")` to capture this live column before passing it to an analytical
producer. Some metadata, mapping, and export utilities still accept `cell_key` directly.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(["I"])["I"].value_counts()
```

This store keeps every barcode active (`True`). Analytical filtering returns a separate immutable
selection artifact and leaves this column unchanged. If you deliberately author a live selection
column, its `False` rows also remain in the table rather than being deleted.

Each assay group holds `featureData` and its persisted artifacts.
Count matrices are Zarr arrays, often sharded. This store keeps RNA counts at `RNA/counts`.

```{code-cell} ipython3
ds.show_zarr_tree(start="RNA", depth=1)
```

```{code-cell} ipython3
ds.show_zarr_tree(start="RNA/featureData", depth=1)
```

Each persisted result is an {term}`artifact`. Assay-scoped results live under
`{assay}/artifacts/{kind}/{artifact_id}`; datastore-scoped selections and integrated results live
under `artifacts/{kind}/{artifact_id}`.
The kind names the operation family and the identifier is derived from the inputs and parameters, which is what lets Scarf recognise an equivalent result instead of recomputing it.
Nothing here encodes parameters in the path, so a second PCA at different dimensionality becomes a sibling entry rather than a new branch of the tree.

```{code-cell} ipython3
ds.show_zarr_tree(start="RNA/artifacts", depth=1)
```

{doc}`../developers/zarr_internals` covers the complete on-disk layout.

## 2. Inspect cell and feature attributes

Cell and feature tables are `MetaData` objects (`ds.cells`, `ds.RNA.feats`), not pandas DataFrames.
Use `head` for a quick look, `to_pandas_dataframe` to export selected columns, and `fetch` / `fetch_all` for single columns.

```{code-cell} ipython3
ds.cells.head()
```

```{code-cell} ipython3
ds.RNA.feats.head()
```

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=["ids", "RNA_nCounts", "RNA_nFeatures", "clusters"]
).set_index("ids")
```

`insert` writes a new column and aligns values to the active subset unless you override `key`.
Re-inserting an existing column requires `overwrite=True`.

```{code-cell} ipython3
first_run_cluster = cluster_values[0]
is_first_cluster = cluster_values == first_run_cluster
ds.cells.insert(
    column_name="is_first_cluster",
    values=is_first_cluster,
    overwrite=True,
)
```

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=["ids", "clusters", "is_first_cluster"]
).head()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by="is_first_cluster",
)
```

The new column marks one cluster on the same active cells used for the insert.

`fetch` returns values for the active subset (default column `I`).
`fetch_all` returns every row in the store.
With every cell active the lengths match. Pass the new Boolean column as `key` to select one
cluster without changing `I`:

```{code-cell} ipython3
print(
    "fetch:", ds.cells.fetch("clusters", key="is_first_cluster").shape,
    "fetch_all:", ds.cells.fetch_all("clusters").shape,
)
```

## 3. Query metadata without materializing a DataFrame

`sift` returns a boolean mask for one numeric range.
`multi_sift` combines several ranges, and `get_index_by` locates exact categorical values:

```{code-cell} ipython3
active_before = int(ds.cells.fetch_all("I").sum())
count_range = ds.cells.sift(
    "RNA_nCounts",
    min_v=1000,
    max_v=15000,
)
joint_range = ds.cells.multi_sift(
    columns=["RNA_nCounts", "RNA_nFeatures"],
    lows=[1000, 500],
    highs=[15000, 4000],
)
ductal_rows = ds.cells.get_index_by(["Ductal"], "clusters")

print(
    "count_range:", int(count_range.sum()),
    "joint_range:", int(joint_range.sum()),
    "ductal_rows:", int(ductal_rows.size),
)
print("Active cells (I) before:", active_before)
print("Active cells (I) after:", int(ds.cells.fetch_all("I").sum()))
```

```{code-cell} ipython3
ds.cells.insert(column_name="in_count_range", values=count_range, overwrite=True)
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by="in_count_range",
)
```

These helpers return masks or indexes aligned with the metadata table.
They do not modify `I` until you explicitly insert or update a cell key.
See the `MetaData` API in {doc}`../reference/api/assays` for update and delete helpers.

## 4. Count matrices and normalization

Raw counts are a Zarr array (often sharded), exposed as `rawData`, a chunked array with a NumPy-like interface that streams by row.
In this store the array is at `RNA/counts`.
RNA assays also store `countsT`, a gene-major copy used by HVG and marker stages.
Routine analysis does not need to touch either array directly.

Normalized values are computed on demand through the lower-level assay `normed()` view from raw counts.
`run_normalization(cell_selection, features)` is the public persisted path and requires exact stored cell- and feature-selection references.
The direct `normed()` view follows its explicit or literal metadata indexes; in a newly created
store the physical feature `I` column is all true. Inspect only small slices when exploring these
lazy arrays:

```{code-cell} ipython3
print("Raw shape:", ds.RNA.rawData.shape)
print("Normed shape:", ds.RNA.normed().shape)
ds.RNA.rawData[:3, :5].compute()
```

```{code-cell} ipython3
ds.RNA.normed()[:3, :5].compute()
```

Override normalization by assigning `normMethod`.
Reassign a custom function each time you open the store.
`scarf.assay.norm_dummy` disables normalization for pre-normalized inputs.

## 5. Inspect persisted analysis results

Analysis methods return lightweight references to results stored in Zarr.
The completed `docs_default` run retained the HVG and normalization references it created.
Asking for the same normalization again reuses that result rather than recomputing:

```{code-cell} ipython3
reused_normalized = ds.run_normalization(
    analysis_run["analysis_cell_selection"],
    analysis_run["highly_variable_features"],
)
print("Reused:", reused_normalized == analysis_run["normalized"])
reused_normalized
```

Inspect its status and open the underlying group only when a custom method needs direct access:

```{code-cell} ipython3
status = ds.inspect_artifact(reused_normalized)
print("Complete:", status.complete)
print("Operation:", status.operation)
print("Parameters:", status.parameters)

group = ds.load_artifact(reused_normalized)
print("Arrays:", list(group.array_keys())[:5])
```

Identical inputs and parameters {term}`reuse` a complete result.
Branching, invalidation, and lineage are covered in {doc}`reuse_and_tracing`.

## 6. Marker features

Marker search writes a `marker_table` artifact like any other result. The pipeline returns its
exact ref as `analysis_run["markers"]`; an explicit `run_marker_search` call follows the same
artifact contract. Pass that ref to table, plot, and export accessors. Different clustering or
feature refs naturally produce distinct artifacts.

Fetch one group with `get_markers`, plot the stored table with `marker_heatmap`, or export all groups with `export_markers_to_csv`.

```{code-cell} ipython3
ds.get_markers(
    marker=marker_ref,
    group_id=cluster_values[0],
    min_score=0.1,
    min_frac_exp=0.1,
).head()
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    marker=marker_ref,
    topn=5,
    figsize=(5, 9),
)
```

```{code-cell} ipython3
markers_csv = "scarf_datasets/pancreas_cluster_markers.csv"
ds.export_markers_to_csv(
    marker=marker_ref,
    csv_filename=markers_csv,
    min_score=0.1,
    min_frac_exp=0.1,
)
pd.read_csv(markers_csv).iloc[:5, :6]
```

## 7. Zarr versions and storage profiles

Current Scarf versions write new datasets as Zarr v3.
RNA assays also write a gene-major `countsT` copy next to `counts`.
The catalog dataset used here was rebuilt from its raw source with this codebase, so it already has
the current layout. Re-import older RNA stores that use Zarr v2 or lack `countsT` before analysis.

Count matrices from the writers use sharded arrays (default profile `fast_local`).
Set the profile with `SCARF_ZARR_PROFILE` (`fast_local` or `cloud`) or `zarrProfile=` when opening a `DataStore`.

Storage profiles and conversion belong to the physical store, while `mem_budget` and `nthreads` control execution.
See {doc}`../concepts/memory_and_execution` for why RNA stores two orientations, and {doc}`remote_stores` for object storage and local scratch.

## Common mistakes

- Expecting filtering to delete rows or rewrite live `I`
- Treating `MetaData` as an in-memory pandas DataFrame
- Using `fetch` when values for inactive cells are also required (`fetch_all`)
- Treating a result reference as an in-memory matrix
- Editing artifact groups directly instead of using Scarf's analysis methods
- Expecting an older RNA Zarr v2 store, or one without `countsT`, to open without re-importing it

Metadata changes and artifacts are written into the Zarr store.
Low-level layout details intended for contributors remain in {doc}`../developers/zarr_internals`.
