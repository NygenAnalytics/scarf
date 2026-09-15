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

Scarf keeps one on-disk store for counts, metadata, and results. Live selections such as `I` mark rows rather than deleting them, and each analysis step writes a named result with a record of what produced it. The benefit of this is during analysis, it is possible to easily revert back to a previous version without having to reload your entire object as you do in Scanpy.

This page is for when you inherit a store and ask: where are the counts, which cells are active, where is a result, and how do I undo my edit.
Low-level layout details for contributors live in {doc}`../developers/zarr_internals`.

Each numbered section ends with a checkpoint of what you should see: If you need to revert back to a previous version, the code required shows you how to undo the edit in that section, allowing you to try things safely.

## Short Refresher

Scarf uses one Zarr store, shared cell table, per-assay feature table, and persisted artifacts.

If you previously used Scanpy then general equivalents are: X/obs/var/obsm/uns to counts/cellData/featureData/artifacts table, plus filter-copy versus I-mark contrast.

If you come from Seurat: assays/meta.data/reductions/graphs to same Scarf targets, plus subset-new-object versus I-mark contrast.

If you are new to both: refer to {doc}`../scanpy_and_seurat`.
The general pattern is row means barcode (each individual cell), column means gene or peak, active means I=True, result means named entry with history.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Basic familiarity with cell and feature metadata
- {ref}`Quick start <quickstart>` or {doc}`scrna_seq` completed once, so `DataStore` and cell filtering are familiar

## What you will learn

- Locate counts, cell metadata, feature metadata, and persisted results in the Zarr hierarchy
- Explain the live `I` cell key and restore it after narrowing it
- Use `fetch` for active rows and `fetch_all` for every row
- Tell an `ArtifactRef` handle apart from the live metadata columns around it
- Reuse a persisted normalization instead of recomputing it
- Revert a live selection edit and a practice metadata column

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

Reading guide: `cellData` holds one row per barcode (a cell), `RNA/featureData` holds one row per gene, `RNA/artifacts` holds named results, and RNA counts live at `RNA/counts` in this store.

:::{note} Practice edits on this page
The setup below opens a local copy of the catalog store, so the `insert`, `update_key`, and `drop` calls ahead edit your copy only.
Each edit is paired with its revert, and the Revert boxes in sections 2 and 3 show you how to undo them.
Persisted artifacts are immutable, so there is nothing to undo there: a new parameter simply makes a new entry.
:::

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
Think of the store as a directory tree, not as a single object: counts, cell and feature attributes, and immutable artifacts live under named groups.
This layout is what enables Scarf to perform parallel reads and writes, conduct fast compression, and allow for automatic persistence of intermediate results to improve reproducibility.

`show_zarr_tree` prints the hierarchy.
With `depth=1` you see the top-level assays and `cellData`.

```{code-cell} ipython3
ds.show_zarr_tree(depth=1)
```

Checkpoint 1: you should see `cellData` for shared cell metadata and an `RNA` group for this page's default assay.

Cell statistics computed from an assay are stored under `cellData` with the assay name as a prefix (`RNA_…`, `ADT_…`).

```{code-cell} ipython3
ds.show_zarr_tree(start="cellData")
```

**The `I` column** is the default user-owned {term}`cell key`, tracking which cells are active for
live-metadata APIs (in use for analysis, the rest are excluded). Values are boolean: authored selections
mark cells `False`, not deleted, which is what allows us to go back and revert changes.
Unlike subsetting an in-memory AnnData object, narrowing here keeps every row so thresholds can be inspected, reset, and replaced.
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

Checkpoint 2: two reductions that differ only in `dims` appear as two siblings; identical requests reuse one entry.
See {doc}`reuse_and_tracing` for the reuse demo.

{doc}`../developers/zarr_internals` covers the complete on-disk layout.

## 2. Inspect cell and feature attributes

Cell and feature tables are `MetaData` objects (`ds.cells`, `ds.RNA.feats`), not pandas DataFrames, which makes manipulation slightly different.
They live on disk and every write persists immediately, so each edit below is paired with its revert.

Quick reference you will use in this section:

- `head` glances at the table, `to_pandas_dataframe(columns=...)` exports selected columns.
- `fetch(column)` returns values for active rows only (`I=True`), `fetch_all(column)` returns every row.
- `insert` adds a column, `drop` removes an unprotected column, `update_key` narrows a boolean key, `reset_key` restores a key to all `True`.

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
`I`, `ids`, and `names` are protected: writing `I` needs `force=True` and deleting it is rejected, which guards the active selection from accidents.

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
cluster without changing `I`. To feel the difference, try narrowing `I` itself once on purpose,
compare the two accessors, then revert immediately with the box below:

```{code-cell} ipython3
print(
    "fetch:", ds.cells.fetch("clusters", key="is_first_cluster").shape,
    "fetch_all:", ds.cells.fetch_all("clusters").shape,
)
```

### Revert box: selections and columns

Use one pattern per kind of edit.
Each cell below is safe to run, change, and re-run while you practise.

1. Selection narrowed with `update_key`: restore the backup you saved.
Backup plus restore keeps earlier filters; `update_key` combines with AND, so repeated narrowing without a backup keeps shrinking `I`.

```{code-cell} ipython3
live_backup = ds.cells.fetch_all("I").copy()
ds.cells.update_key(is_first_cluster, "I")
print("Narrowed:", int(ds.cells.fetch_all("I").sum()))
ds.cells.insert(column_name="I", values=live_backup, overwrite=True, force=True)
print("Reverted:", int(ds.cells.fetch_all("I").sum()))
```

2. Selection you want to clear completely: reset the key.
This sets every row to `True` and discards earlier narrowing of `I`.

```{code-cell} ipython3
ds.cells.update_key(is_first_cluster, "I")
print("Narrowed:", int(ds.cells.fetch_all("I").sum()))
ds.cells.reset_key("I")
print("After reset_key:", int(ds.cells.fetch_all("I").sum()))
```

3. Analytical `filter_cells` without losing history: it returns an immutable selection artifact and leaves live `I` untouched.
Compose steps by passing the prior selection explicitly instead of overwriting anything.

```python
# Returns a cell_selection artifact; live I is snapshotted, not modified.
sel = ds.filter_cells(attrs=["RNA_nCounts"], lows=[1000], highs=[15000])
# Compose a second bound on top of the first selection.
sel2 = ds.filter_cells(
    attrs=["RNA_nFeatures"], lows=[500], highs=[4000], cell_selection=sel
)
# Revert: nothing in I changed, so just keep using the earlier ref.
```

4. Practice column you added: delete it.
Run these in order; each one checks first so re-running is safe.

```{code-cell} ipython3
if "in_count_range" in ds.cells.columns:
    ds.cells.drop("in_count_range")
else:
    print("in_count_range not present yet; created in section 3")
print("is_first_cluster" in ds.cells.columns)
```

```{code-cell} ipython3
# Undo the practice column from this section. Re-run the insert cell above to bring it back.
if "is_first_cluster" in ds.cells.columns:
    ds.cells.drop("is_first_cluster")
print("is_first_cluster present after drop:", "is_first_cluster" in ds.cells.columns)
```

```{code-cell} ipython3
# Restore it for the rest of this page. Re-derive from the run so alignment stays correct.
first_run_cluster = cluster_values[0]
is_first_cluster = cluster_values == first_run_cluster
ds.cells.insert(column_name="is_first_cluster", values=is_first_cluster, overwrite=True)
print("Restored:", "is_first_cluster" in ds.cells.columns)
```

Checkpoint 3: after steps 1 to 4, `fetch_all("I").sum()` equals your saved backup count, and only columns you intend to keep remain.

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
That is why `Active cells after` matches `Active cells before` in the printout above.

Revert: nothing to revert here unless you inserted the mask.
To undo `in_count_range`, run `ds.cells.drop("in_count_range")` and re-run the insert cell to bring it back.
See the `MetaData` API in {doc}`../reference/api/assays` for update and delete helpers.

## 4. Count matrices and normalization

Raw counts are a Zarr array (often sharded), exposed as `rawData`, a chunked array with a NumPy-like interface that streams by row.
In this store the array is at `RNA/counts`.
RNA assays also store `countsT`, a gene-major copy used by HVG and marker stages.
Routine analysis does not need to touch either array directly.

Three ideas to keep apart: `rawData` is stored counts, `normed()` is an on-demand view, and `run_normalization(cell_selection, features)` is the persisted reusable result.
Only the third creates an {term}`artifact` you can reuse and trace.

Normalized values are computed on demand through the lower-level assay `normed()` view from raw counts.
`run_normalization(cell_selection, features)` is the public persisted path and requires exact stored cell- and feature-selection references, such as the ones the `docs_default` run holds.
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

Analysis methods return lightweight references to results stored in Zarr, not the matrices themselves.
`analysis_run["highly_variable_features"]` is the exact immutable feature selection the run used.
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
Checkpoint 4: `Reused: True` means no recomputation happened.

Revert: artifacts are immutable, so there is nothing to undo here.
Do not edit artifact groups directly. To try a different parameter, run the method again with new inputs and keep both refs; to compare histories, use lineage in {doc}`reuse_and_tracing`.
Branching, invalidation, and lineage are covered there.

## 6. Marker features

Marker search writes a `marker_table` artifact like any other result. The pipeline returns its
exact ref as `analysis_run["markers"]`; an explicit `run_marker_search` call follows the same
artifact contract. Pass that ref to table, plot, and export accessors. Different clustering or
feature refs naturally produce distinct artifacts.
That is why the cells below pass `marker=marker_ref` explicitly instead of relying on a label.

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

Selections and reads:

- Expecting filtering to delete rows or rewrite live `I`
- Treating `MetaData` as an in-memory pandas DataFrame
- Using `fetch` when values for inactive cells are also required (`fetch_all`)

Results:

- Treating a result reference as an in-memory matrix
- Editing artifact groups directly instead of using Scarf's analysis methods
- Expecting an older RNA Zarr v2 store, or one without `countsT`, to open without re-importing it

Revert mistakes:

- Narrowing `I` twice with `update_key` and expecting the first narrowing to be gone; it ANDs, so restore from backup or use `reset_key`
- Expecting analytical `filter_cells` to rewrite live `I`; it returns a new selection artifact, so just keep using the earlier ref
- Dropping or overwriting a column without a way back; keep the construction expression in a cell so re-running restores it

## Revert cheat-sheet

| You changed | Revert with | Note |
|---|---|---|
| `I` narrowed by `update_key` | `ds.cells.insert(column_name="I", values=backup, overwrite=True, force=True)` | Save `backup = ds.cells.fetch_all("I").copy()` first |
| `I` narrowed and you want it cleared | `ds.cells.reset_key("I")` | Sets every row to `True` |
| Analytical filtering | Keep using the earlier selection ref | `filter_cells` never modifies live `I` |
| Practice column | `ds.cells.drop("column_name")` | `I`, `ids`, `names` cannot be dropped |

Metadata changes and artifacts are written into the Zarr store.
Low-level layout details intended for contributors remain in {doc}`../developers/zarr_internals`.
