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

Scarf stores counts, metadata, graphs, and analysis results in a Zarr directory. This chapter
shows how to inspect that hierarchy and how to read or update cell and feature metadata.
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

`DataStore` is the main entry point. Each assay owns feature metadata, normalization, and
feature selection. Cell-level columns are shared across assays.

```{mermaid}
flowchart TB
    ds["DataStore"]
    cells["Shared cell metadata"]
    rna["RNA assay<br/>feature metadata and results<br/>(default assay)"]
    atac["ATAC assay<br/>feature metadata and results"]
    source["Optional mounted source<br/>counts and countsT"]
    ds --> cells
    ds --> rna
    ds --> atac
    source -.-> rna
    source -.-> atac
```

The default assay supplies method defaults when `from_assay` is omitted. It
does not merge assay-specific feature tables or results.

```{code-cell} ipython3
import pandas as pd

import scarf

scarf.configure_output(level='WARNING', progress=True)
```

This page opens the pre-analyzed Bastidas-Ponce pancreas store also used in
{doc}`plotting` and {doc}`cell_cycle`. It already includes HVGs, a
neighbourhood graph, UMAP coordinates, cluster labels, and a marker table, so
every group shown below is one a real analysis produced.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)

ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    default_assay='RNA',
    nthreads=4,
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

## Guided steps

### 1. Inspect Zarr trees

Scarf uses [Zarr](https://zarr.readthedocs.io/en/stable/) for chunked on-disk arrays. The store
is a directory tree: counts, cell and feature attributes, and cached intermediates live under
named groups. Relative to a single HDF5 file, the layout supports parallel reads and writes,
fast compression codecs, and automatic persistence of intermediate results.

`show_zarr_tree` prints the hierarchy. With `depth=1` you see the top-level assays and
`cellData`.

```{code-cell} ipython3
ds.show_zarr_tree(depth=1)
```

Cell statistics computed from an assay are stored under `cellData` with the assay name as a
prefix (`RNA_…`, `ADT_…`).

```{code-cell} ipython3
ds.show_zarr_tree(start='cellData')
```

**The `I` column** is the default {term}`cell key`, tracking which cells are active.
Values are boolean: filtered-out cells are `False`. Most `DataStore` methods take
`cell_key` (default `I`) and operate only on cells marked `True`.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(['I'])['I'].value_counts()
```

This store keeps every barcode active (`True`). Filtered cells stay in the table as
`False` rows; they are not deleted.

Each assay group holds `counts`, `featureData`, optional `markers`, and its
persisted analysis outputs.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA', depth=1)
```

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

Each persisted result is an {term}`artifact`, living under
`{assay}/artifacts/{kind}/{artifact_id}`. The kind names the operation family and
the identifier is derived from the inputs and parameters, which is what lets
Scarf recognise an equivalent result instead of recomputing it. Nothing here encodes parameters in the path, so a second PCA at different
dimensionality becomes a sibling entry rather than a new branch of the tree.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/artifacts', depth=1)
```

Stores written before this layout encoded the whole chain into nested group
names such as `RNA/normed__I__hvgs/reduction__pca__15__I/...`. Scarf still reads
those. {doc}`../developers/zarr_internals` covers repacking older stores to Zarr
v3 with sharded counts; that does not migrate nested path trees into artifacts.

### 2. Inspect cell and feature attributes

Cell and feature tables are `MetaData` objects (`ds.cells`, `ds.RNA.feats`), not pandas
DataFrames. Use `head` for a quick look, `to_pandas_dataframe` to export selected columns, and
`fetch` / `fetch_all` for single columns.

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

`fetch` returns values for the active subset (default column `I`). `fetch_all` returns every
row in the store. With every cell active the lengths match, so temporarily restrict `I` to
make the difference visible, then restore the backup:

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

### 3. Query metadata without materializing a DataFrame

`sift` returns a boolean mask for one numeric range. `multi_sift` combines
several ranges, and `get_index_by` locates exact categorical values:

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

These helpers return masks or indexes aligned with the metadata table. They do
not modify `I` until you explicitly insert or update a cell key. See the
`MetaData` API in {doc}`../reference/api/assays` for update and delete helpers.

### 4. Count matrices and normalization

Raw counts live under each assay's `counts` group and are exposed as `rawData`, a chunked array
with a NumPy-like interface that streams by row. Routine analysis does not need to touch this
object directly.

```{code-cell} ipython3
ds.RNA.rawData
```

Normalized values are computed on demand through `normed()`. Scarf keeps only raw counts on
disk by default. `normed()` drops inactive features, so its column count is smaller than
`rawData`:

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

Override normalization by assigning `normMethod`. Reassign a custom function each time you
open the store. `scarf.assay.norm_dummy` disables normalization for pre-normalized inputs.

```{code-cell} ipython3
print('Current method:', ds.RNA.normMethod.__name__)
```

### 5. Inspect persisted analysis results

Analysis methods return lightweight references to results stored in Zarr.
Asking for the normalization the store already holds returns a reference to it
rather than recomputing:

```{code-cell} ipython3
normalized = ds.run_normalization(feat_key='hvgs')
normalized
```

Inspect its status and open the underlying group only when a custom method
needs direct access:

```{code-cell} ipython3
status = ds.inspect_artifact(normalized)
print('Complete:', status.complete)
print('Operation:', status.operation)
print('Parameters:', status.parameters)

group = ds.load_artifact(normalized)
print('Arrays:', list(group.array_keys())[:5])
```

Identical inputs and parameters {term}`reuse` a complete result. Branching,
invalidation, lineage, and the current {term}`analysis chain` are covered in
{doc}`reuse_and_tracing`.

### 6. Marker features

`run_marker_search` writes a `marker_table` artifact like any other result. The
assay also keeps an index under `{assay}/markers`, holding no arrays of its own,
whose `artifacts` attribute maps `{cell_key}__{group_key}` slots to those refs.
That indirection is what lets `get_markers` find a table from a group key
without knowing an artifact identifier.

```{code-cell} ipython3
index = dict(ds.z['RNA/markers'].attrs.get('artifacts', {}))
index
```

```{code-cell} ipython3
table = scarf.ArtifactRef.from_dict(index['I__clusters'])
print('Stored at:', ds.inspect_artifact(table).path)
```

Fetch one group with `get_markers`, plot the stored table with `marker_heatmap`,
or export all groups with `export_markers_to_csv`.

```{code-cell} ipython3
ds.get_markers(
    group_key='clusters',
    group_id=ds.cells.fetch('clusters')[0],
    min_score=0.1,
    min_frac_exp=0.1,
).head()
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key='clusters',
    topn=5,
    figsize=(5, 9),
)
```

```{code-cell} ipython3
markers_csv = 'scarf_datasets/pancreas_cluster_markers.csv'
ds.export_markers_to_csv(
    group_key='clusters',
    csv_filename=markers_csv,
    min_score=0.1,
    min_frac_exp=0.1,
)
pd.read_csv(markers_csv).iloc[:5, :6]
```

### 7. Zarr versions and storage profiles

Current Scarf versions write new datasets as Zarr v3. Existing v2 stores remain
readable. Count matrices from the writers use sharded arrays (default profile
`fast_local`). Set the profile with `SCARF_ZARR_PROFILE` (`fast_local` or
`cloud`) or `zarrProfile=` when opening a `DataStore`.

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

Storage profiles and conversion belong to the physical store, while
`mem_budget` and `nthreads` control execution. See
{doc}`../concepts/memory_and_execution` for memory planning and
{doc}`remote_stores` for object storage and local scratch.

## Common mistakes

- Expecting filtered cells to be deleted instead of marked `False` in `I`
- Treating `MetaData` as an in-memory pandas DataFrame
- Using `fetch` when values for inactive cells are also required (`fetch_all`)
- Treating a result reference as an in-memory matrix
- Editing artifact groups directly instead of using Scarf's analysis methods

Metadata changes and artifacts are written into the Zarr store. Low-level
layout details intended for contributors remain in
{doc}`../developers/zarr_internals`.
