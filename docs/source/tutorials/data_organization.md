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
- Locate marker tables through the marker index

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
    source["Optional mounted source<br/>counts and RNA countsT"]
    ds --> cells
    ds --> rna
    ds --> atac
    source -.-> rna
    source -.-> atac
```

The default assay supplies method defaults when `from_assay` is omitted.
It does not merge assay-specific feature tables or results.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level='WARNING', progress=True)
```

This page uses the pre-analyzed Bastidas-Ponce pancreas store also used in {doc}`plotting` and {doc}`cell_cycle`.
The published store contains literal UMAP coordinates and cluster labels together with analysis state from an older Scarf release.
First repack it structurally into a temporary source with the current paired RNA count layout, then mount those count matrices into a fresh writable target.
The published source stays untouched, and every artifact inspected below follows the current contract.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)

analysis_directory = TemporaryDirectory()
repacked_counts_path = Path(analysis_directory.name) / 'counts.zarr'
analysis_path = Path(analysis_directory.name) / 'data_organization.zarr'
repack_store(
    f'{dataset}/data.zarr',
    str(repacked_counts_path),
    nthreads=2,
)
ds = scarf.mount_datastore(
    str(repacked_counts_path),
    at=str(analysis_path),
    default_assay='RNA',
    nthreads=4,
)

hvg_ref = ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
normalized = ds.run_normalization(features=hvg_ref)
all_features = ds.resolve_features('RNA', 'all_features')
marker_ref = ds.run_marker_search(
    group_key='clusters',
    cell_key='I',
    features=all_features,
)

ds
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='clusters',
)
```

Cluster labels already live in cell metadata; the embedding only reads those columns.

## 1. Inspect Zarr trees

Scarf uses [Zarr](https://zarr.readthedocs.io/en/stable/) for chunked on-disk arrays.
The store is a directory tree: counts, cell and feature attributes, and cached intermediates live under named groups.
Relative to a single HDF5 file, the layout supports parallel reads and writes, fast compression codecs, and automatic persistence of intermediate results.

`show_zarr_tree` prints the hierarchy.
With `depth=1` you see the top-level assays and `cellData`.

```{code-cell} ipython3
ds.show_zarr_tree(depth=1)
```

Cell statistics computed from an assay are stored under `cellData` with the assay name as a prefix (`RNA_…`, `ADT_…`).

```{code-cell} ipython3
ds.show_zarr_tree(start='cellData')
```

**The `I` column** is the default {term}`cell key`, tracking which cells are active.
Values are boolean: filtered-out cells are `False`.
Most `DataStore` methods take `cell_key` (default `I`) and operate only on cells marked `True`.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(['I'])['I'].value_counts()
```

This store keeps every barcode active (`True`).
Filtered cells stay in the table as `False` rows; they are not deleted.

Each assay group holds `featureData`, optional `markers`, and its persisted analysis outputs.
Count matrices are Zarr arrays (often sharded): default `{assay}/counts`, workspace `matrices/{assay}/counts`, or still in a mounted source when the assay is mounted.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA', depth=1)
```

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

Each persisted result is an {term}`artifact`, living under `{assay}/artifacts/{kind}/{artifact_id}`.
The kind names the operation family and the identifier is derived from the inputs and parameters, which is what lets Scarf recognise an equivalent result instead of recomputing it.
Nothing here encodes parameters in the path, so a second PCA at different dimensionality becomes a sibling entry rather than a new branch of the tree.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/artifacts', depth=1)
```

For diagnosis only, stores written before this layout may contain nested names such as `RNA/normed__I__hvgs/reduction__pca__15__I/...`.
Current analysis does not resolve those paths or migrate them silently.
Counts and literal metadata remain readable, while legacy analysis state fails closed with `IncompatibleAnalysisStateError` before computation or mutation.
Opening also leaves a legacy feature `I` column byte-for-byte untouched; lazy `all_features` creation covers the complete current feature row order instead of inheriting that older filter.
{doc}`../developers/zarr_internals` covers the on-disk layout and structural Zarr repacking; repacking does not migrate legacy analysis state into artifacts.

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
    columns=['ids', 'RNA_nCounts', 'RNA_nFeatures', 'clusters']
).set_index('ids')
```

`insert` writes a new column and aligns values to the active subset unless you override `key`.
Re-inserting an existing column requires `overwrite=True`.

```{code-cell} ipython3
cluster_labels = ds.cells.fetch('clusters')
first_cluster = str(cluster_labels[0])
is_first_cluster = cluster_labels.astype(str) == first_cluster
ds.cells.insert(column_name='is_first_cluster', values=is_first_cluster, overwrite=True)
```

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=['ids', 'clusters', 'is_first_cluster']
).head()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='is_first_cluster',
)
```

The new column marks one cluster on the same active cells used for the insert.

`fetch` returns values for the active subset (default column `I`).
`fetch_all` returns every row in the store.
With every cell active the lengths match, so temporarily restrict `I` to make the difference visible, then restore the backup:

```{code-cell} ipython3
i_backup = ds.cells.fetch_all('I').copy()
ds.cells.update_key(is_first_cluster, 'I')

print(
    'fetch:', ds.cells.fetch('clusters').shape,
    'fetch_all:', ds.cells.fetch_all('clusters').shape,
)
ds.cells.to_pandas_dataframe(['I'])['I'].value_counts()
```

```{code-cell} ipython3
ds.cells.insert(column_name='I', values=i_backup, overwrite=True, force=True)
print('Restored active cells:', int(ds.cells.fetch_all('I').sum()))
```

```{code-cell} ipython3
try:
    ds.cells.insert(
        column_name='is_first_cluster',
        values=is_first_cluster,
    )
except ValueError:
    print("Expected validation: use overwrite=True to replace an existing column.")
```

## 3. Query metadata without materializing a DataFrame

`sift` returns a boolean mask for one numeric range.
`multi_sift` combines several ranges, and `get_index_by` locates exact categorical values:

```{code-cell} ipython3
active_before = int(ds.cells.fetch_all('I').sum())
count_range = ds.cells.sift(
    'RNA_nCounts',
    min_v=1000,
    max_v=15000,
)
joint_range = ds.cells.multi_sift(
    columns=['RNA_nCounts', 'RNA_nFeatures'],
    lows=[1000, 500],
    highs=[15000, 4000],
)
cluster_rows = ds.cells.get_index_by([first_cluster], 'clusters')

print(
    'count_range:', int(count_range.sum()),
    'joint_range:', int(joint_range.sum()),
    'cluster_rows:', int(cluster_rows.size),
)
print('Active cells (I) before:', active_before)
print('Active cells (I) after:', int(ds.cells.fetch_all('I').sum()))
```

```{code-cell} ipython3
ds.cells.insert(column_name='in_count_range', values=count_range, overwrite=True)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='in_count_range',
)
```

These helpers return masks or indexes aligned with the metadata table.
They do not modify `I` until you explicitly insert or update a cell key.
See the `MetaData` API in {doc}`../reference/api/assays` for update and delete helpers.

## 4. Count matrices and normalization

Raw counts are a Zarr array (often sharded), exposed as `rawData`, a chunked array with a NumPy-like interface that streams by row.
Location depends on layout: default `{assay}/counts`, workspace `matrices/{assay}/counts`, or the mounted source when counts stay there.
RNA assays also store `countsT`, a gene-major copy used by HVG and marker stages.
Routine analysis does not need to touch either array directly.

```{code-cell} ipython3
ds.RNA.rawData
```

Normalized values are computed on demand through the lower-level assay `normed()` view from raw counts.
`run_normalization(features=...)` is the public persisted path and requires an exact feature-selection label or reference.
The direct `normed()` view follows its explicit or literal metadata indexes; in a newly created store the physical feature `I` column is all true:

```{code-cell} ipython3
ds.RNA.normed()
```

```{code-cell} ipython3
print('Raw shape:', ds.RNA.rawData.shape)
print('Normed shape:', ds.RNA.normed().shape)
ds.RNA.rawData[:3, :5].compute()
```

```{code-cell} ipython3
ds.RNA.normed()[:3, :5].compute()
```

Override normalization by assigning `normMethod`.
Reassign a custom function each time you open the store.
`scarf.assay.norm_dummy` disables normalization for pre-normalized inputs.

```{code-cell} ipython3
print('Current method:', ds.RNA.normMethod.__name__)
```

## 5. Inspect persisted analysis results

Analysis methods return lightweight references to results stored in Zarr.
The setup retained the HVG and normalization references it created in the mounted target.
Asking for the same normalization again reuses that result rather than recomputing:

```{code-cell} ipython3
reused_normalized = ds.run_normalization(features=hvg_ref)
print('Reused:', reused_normalized == normalized)
reused_normalized
```

Inspect its status and open the underlying group only when a custom method needs direct access:

```{code-cell} ipython3
status = ds.inspect_artifact(reused_normalized)
print('Complete:', status.complete)
print('Operation:', status.operation)
print('Parameters:', status.parameters)

group = ds.load_artifact(reused_normalized)
print('Arrays:', list(group.array_keys())[:5])
```

Identical inputs and parameters {term}`reuse` a complete result.
Branching, invalidation, lineage, and the current {term}`analysis chain` are covered in {doc}`reuse_and_tracing`.

## 6. Marker features

`run_marker_search` writes a `marker_table` artifact like any other result and returns its exact reference.
The assay also keeps an index under `{assay}/markers`.
In the attrs index layout, that group holds no arrays of its own; its `artifacts` attribute nests refs by cell key, grouping column, and feature-selection artifact id. Legacy `markers/{slot}` subgroups can hold arrays.

```{code-cell} ipython3
index = dict(ds.z['RNA/markers'].attrs.get('artifacts', {}))
index
```

```{code-cell} ipython3
table = scarf.ArtifactRef.from_dict(
    index['I']['clusters'][all_features.artifact_id]
)
print('Stored at:', ds.inspect_artifact(table).path)
```

The nested feature-selection key prevents a marker search over one feature universe from replacing another.
If multiple feature-specific tables exist for the same grouping, pass the exact returned ref to table, plot, and export accessors.

Fetch one group with `get_markers`, plot the stored table with `marker_heatmap`, or export all groups with `export_markers_to_csv`.

```{code-cell} ipython3
ds.get_markers(
    marker=marker_ref,
    cell_key='I',
    group_key='clusters',
    group_id=ds.cells.fetch('clusters')[0],
    min_score=0.1,
    min_frac_exp=0.1,
).head()
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    marker=marker_ref,
    cell_key='I',
    group_key='clusters',
    topn=5,
    figsize=(5, 9),
)
```

```{code-cell} ipython3
markers_csv = 'scarf_datasets/pancreas_cluster_markers.csv'
ds.export_markers_to_csv(
    marker=marker_ref,
    cell_key='I',
    group_key='clusters',
    csv_filename=markers_csv,
    min_score=0.1,
    min_frac_exp=0.1,
)
pd.read_csv(markers_csv).iloc[:5, :6]
```

## 7. Zarr versions and storage profiles

Current Scarf versions write new datasets as Zarr v3.
RNA assays also write a gene-major `countsT` copy next to `counts`.
An older RNA store that is still Zarr v2, or that lacks that copy, will not open until you re-import it or repack it.

Count matrices from the writers use sharded arrays (default profile `fast_local`).
Set the profile with `SCARF_ZARR_PROFILE` (`fast_local` or `cloud`) or `zarrProfile=` when opening a `DataStore`.

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

Storage profiles and conversion belong to the physical store, while `mem_budget` and `nthreads` control execution.
See {doc}`../concepts/memory_and_execution` for why RNA stores two orientations, and {doc}`remote_stores` for object storage and local scratch.

## Common mistakes

- Expecting filtered cells to be deleted instead of marked `False` in `I`
- Treating `MetaData` as an in-memory pandas DataFrame
- Using `fetch` when values for inactive cells are also required (`fetch_all`)
- Treating a result reference as an in-memory matrix
- Editing artifact groups directly instead of using Scarf's analysis methods
- Expecting an older RNA Zarr v2 store, or one without `countsT`, to open without a re-import or `repack_zarr`

Metadata changes and artifacts are written into the Zarr store.
Low-level layout details intended for contributors remain in {doc}`../developers/zarr_internals`.
