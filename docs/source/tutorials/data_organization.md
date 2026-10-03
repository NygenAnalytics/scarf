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

Scarf stores counts, metadata, graphs, and analysis results in a Zarr directory.
Start with cell and feature metadata, then explore how counts and saved results are organized.
Low-level layout details for contributors live in {doc}`../developers/zarr_internals`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Basic familiarity with cell and feature metadata

## What you will learn

- Read and write cell or feature metadata
- Inspect a persisted normalization result
- Inspect the Zarr hierarchy when you need to locate stored data

## Dataset

`DataStore` is the main entry point.
Each assay owns feature metadata, normalization, and feature selection.
Cell-level columns are shared across assays. The Boolean cell column `I` marks the active cells
used by live-metadata methods. Saved runs keep their own frozen cell selection.

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

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)
```

This page uses the pre-analyzed Bastidas-Ponce pancreas store also used in {doc}`plotting` and {doc}`cell_cycle`.
The rebuilt store uses the current layout and contains a completed pipeline run named
`docs_default`. Open it directly and reuse that run's exact selections, clustering, UMAP, and
markers.

```{code-cell} ipython3
# Download the prepared example, including its saved analysis.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
```

Open the downloaded store and its saved analysis.

```{code-cell} ipython3
# Open the datastore for the following analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Reuse the saved run and its frozen cell selection.
analysis_run = ds.pipeline.open(label="docs_default")

# Inspect the store's assays and dimensions.
ds
```

```{code-cell} ipython3
# Read the cluster labels in the run's cell order.
cluster_values = analysis_run.cells.fetch("clusters")
# Count cells in each cluster saved by the run.
cluster_counts = pd.Series(cluster_values, name="cluster").value_counts()
# Inspect cluster sizes in label order.
cluster_counts.sort_index().to_frame("cells")
```

The selected clustering remains in its exact artifact. The catalog's literal `clusters` column is
an imported cell-type annotation, not a copy of this analytical result. Opening the run does not
modify either one.

## 1. Read cell and feature metadata

Cell and feature tables are `MetaData` objects (`ds.cells`, `ds.RNA.feats`), not pandas DataFrames.
Use `head` for a quick look, `to_pandas_dataframe` to export selected columns, and `fetch` / `fetch_all` for single columns.

```{code-cell} ipython3
# Preview cell identity, active status and the published annotation.
ds.cells.head()[["ids", "I", "clusters"]]
```

```{code-cell} ipython3
# Preview the RNA feature metadata.
ds.RNA.feats.head()
```

```{code-cell} ipython3
# Read selected QC columns alongside the published annotations.
cell_qc = ds.cells.to_pandas_dataframe(
    columns=["ids", "RNA_nCounts", "RNA_nFeatures", "clusters"]
)
# Preview the QC table with cell IDs as its row labels.
cell_qc.set_index("ids").head()
```

`insert` writes a new column and aligns values to the active subset unless you override `key`.
Re-inserting an existing column requires `overwrite=True`.
In this prepared store, every cell is active and the saved run contains the same cells, so its
cluster labels align with the values we insert. For another analysis, check the cell selection
before copying results into live metadata.

```{code-cell} ipython3
# Choose the cluster containing the first selected cell.
first_run_cluster = cluster_values[0]
# Mark cells belonging to that cluster.
is_first_cluster = cluster_values == first_run_cluster
# Save the values in cell metadata using the stated selection.
ds.cells.insert(
    column_name="is_first_cluster", values=is_first_cluster, overwrite=True
)
# Check the sizes of the selected cluster and its complement.
pd.Series(is_first_cluster).value_counts().rename("cells")
```

```{code-cell} ipython3
# Preview the newly saved cluster-selection column.
ds.cells.to_pandas_dataframe(
    columns=["ids", "clusters", "is_first_cluster"]
).head()
```

```{code-cell} ipython3
# Locate the selected cluster on the saved UMAP.
ds.plots.embedding(layout=analysis_run["umap"], color_by="is_first_cluster")
```

The new column marks one cluster on the same active cells used for the insert.

`fetch` returns values for the active subset (default column `I`).
`fetch_all` returns every row in the store.
With every cell active the lengths match. Pass the new Boolean column as `key` to select one
cluster without changing `I`:

```{code-cell} ipython3
# Compare a selected fetch with the complete metadata axis.
print(
    "fetch:",
    ds.cells.fetch("clusters", key="is_first_cluster").shape,
    "fetch_all:",
    ds.cells.fetch_all("clusters").shape,
)
```

## 2. Select cells from metadata

`sift` returns a boolean mask for one numeric range.
`multi_sift` combines several ranges, and `get_index_by` locates exact categorical values:

```{code-cell} ipython3
# Record the active-cell count before creating new masks.
active_before = int(ds.cells.fetch_all("I").sum())
# Select cells within the stated count range.
count_range = ds.cells.sift("RNA_nCounts", min_v=1000, max_v=15000)
# Count cells inside the count range.
{"cells in count range": int(count_range.sum())}
```

Combine count and detected-gene limits.

```{code-cell} ipython3
# Apply count and detected-feature ranges together.
joint_range = ds.cells.multi_sift(
    columns=["RNA_nCounts", "RNA_nFeatures"],
    lows=[1000, 500],
    highs=[15000, 4000],
)
# Count cells satisfying both QC ranges.
{"cells in both QC ranges": int(joint_range.sum())}
```

Find rows by their published annotation.

```{code-cell} ipython3
# Find the physical rows annotated as ductal cells.
ductal_rows = ds.cells.get_index_by(["Ductal"], "clusters")

# Count the rows matching the published ductal annotation.
print("Ductal cells:", int(ductal_rows.size))
# Show the active-cell count recorded before creating masks.
print("Active cells (I) before:", active_before)
# Check that creating masks left the active selection unchanged.
print("Active cells (I) after:", int(ds.cells.fetch_all("I").sum()))
```

```{code-cell} ipython3
# Save the values in cell metadata using the stated selection.
ds.cells.insert(column_name="in_count_range", values=count_range, overwrite=True)
# Locate cells inside the count range on the saved UMAP.
ds.plots.embedding(layout=analysis_run["umap"], color_by="in_count_range")
```

These helpers return masks or indexes aligned with the metadata table.
They do not modify `I` until you explicitly insert or update a cell key.
See the `MetaData` API in {doc}`../reference/api/assays` for update and delete helpers.

## 3. Count matrices and normalization

Raw counts are a Zarr array (often sharded), exposed as `rawData`, a chunked array with a NumPy-like interface that streams by row.
In this store the array is at `RNA/counts`.
RNA assays also store `countsT`, a gene-major copy used by HVG and marker stages.
Routine analysis does not need to touch either array directly.

Normalized values are computed on demand through the lower-level assay `normed()` view from raw counts.
`run_normalization(cell_selection, features)` is the public persisted path and requires exact stored cell- and feature-selection references.
The direct `normed()` view follows its explicit or literal metadata indexes; in a newly created
store the physical feature `I` column is all true. Inspect the shapes without loading the
complete matrix:

```{code-cell} ipython3
# Inspect raw matrix dimensions without reading its values.
print("Raw shape:", ds.RNA.rawData.shape)
# Inspect normalized-view dimensions without materializing the matrix.
print("Normed shape:", ds.RNA.normed().shape)
```

For a worked example that reads count blocks for an external calculation, see
{doc}`custom_analyses`.

## 4. Inspect saved analysis results

Analysis methods return lightweight references to results stored in Zarr.
The completed `docs_default` run retained the HVG and normalization references it created.
Asking for the same normalization again reuses that result rather than recomputing:

```{code-cell} ipython3
# Request the saved normalization using the same exact inputs.
reused_normalized = ds.run_normalization(
    analysis_run["analysis_cell_selection"],
    analysis_run["highly_variable_features"],
)
# Check whether this request reused the saved normalization.
print("Reused:", reused_normalized == analysis_run["normalized"])
# Inspect the reference identifying the reused normalization.
reused_normalized
```

Inspect its status and open the underlying group only when a custom method needs direct access:

```{code-cell} ipython3
# Inspect the saved result's completeness and provenance.
status = ds.inspect_artifact(reused_normalized)
# Check that the saved normalization is complete.
print("Complete:", status.complete)
# Show the operation that created this artifact.
print("Operation:", status.operation)
# Show the normalization method without its internal registration details.
print("Method:", status.parameters["normalization_method"]["qualname"])
# Keep the numerical and transformation settings used for this result.
normalization_settings = {
    key: status.parameters[key]
    for key in ("size_factor", "log_transform", "renormalize_subset")
}

# Open the artifact arrays for a low-level inspection.
group = ds.load_artifact(reused_normalized)
# List a few arrays available in the saved artifact.
print("Arrays:", list(group.array_keys())[:5])
# Inspect one normalization setting per row.
pd.Series(normalization_settings, name="value").rename_axis("setting").to_frame()
```

Identical inputs and parameters {term}`reuse` a complete result.
Branching, invalidation, and lineage are covered in {doc}`reuse_and_tracing`.

## 5. Look inside the store

Scarf uses [Zarr](https://zarr.readthedocs.io/en/stable/) for chunked on-disk arrays.
The store is a directory tree: counts, cell and feature attributes, and immutable artifacts live
under named groups.
Relative to a single HDF5 file, the layout supports parallel reads and writes, fast compression codecs, and automatic persistence of intermediate results.

`show_zarr_tree` prints the hierarchy.
With `depth=0` you see the top-level assays and `cellData`.

```{code-cell} ipython3
# Locate the top-level assays, metadata and saved analyses.
ds.show_zarr_tree(depth=0)
```

Cell statistics computed from an assay are stored under `cellData` with the assay name as a prefix (`RNA_…`, `ADT_…`).

```{code-cell} ipython3
# Inspect the storage layout of the active-cell flag and two QC columns.
metadata_arrays = {
    name: ds.zw["cellData"][name]
    for name in ("I", "RNA_nCounts", "RNA_nFeatures")
}
# Compare array shapes, value types and chunk sizes without loading values.
pd.DataFrame({
    name: {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "chunks": array.chunks,
    }
    for name, array in metadata_arrays.items()
}).T.rename_axis("cellData column")
```

**The `I` column** is the default user-owned {term}`cell key`, tracking which cells are active for
live-metadata APIs. Values are boolean.
Use `snapshot_cell_selection("I")` to capture this live column before passing it to an analytical
producer. Some metadata, mapping, and export utilities still accept `cell_key` directly.

```{code-cell} ipython3
# Count active and inactive rows on the complete cell axis.
ds.cells.to_pandas_dataframe(["I"])["I"].value_counts()
```

This store keeps every barcode active (`True`). Analytical filtering returns a separate immutable
selection artifact and leaves this column unchanged. If you deliberately author a live selection
column, its `False` rows also remain in the table rather than being deleted.

Each assay group holds `featureData` and its persisted artifacts.
Count matrices are Zarr arrays, often sharded. This store keeps RNA counts at `RNA/counts`.

```{code-cell} ipython3
# Inspect this part of the store hierarchy without reading counts.
ds.show_zarr_tree(start="RNA", depth=1)
```

For feature arrays, pass `start="RNA/featureData"` to the same method.

Each persisted result is an {term}`artifact`. Assay-scoped results live under
`{assay}/artifacts/{kind}/{artifact_id}`; datastore-scoped selections and integrated results live
under `artifacts/{kind}/{artifact_id}`.
The kind names the operation family and the identifier is derived from the inputs and parameters, which is what lets Scarf recognise an equivalent result instead of recomputing it.
Nothing here encodes parameters in the path, so a second PCA at different dimensionality becomes a sibling entry rather than a new branch of the tree.

To inspect saved result groups, use `ds.show_zarr_tree(start="RNA/artifacts", depth=1)`.

{doc}`../developers/zarr_internals` covers the complete on-disk layout.

## 6. Zarr versions and storage profiles

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
