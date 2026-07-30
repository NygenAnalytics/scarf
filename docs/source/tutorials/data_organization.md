---
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
- See how released encoded paths and current artifacts coexist
- Create, inspect, reuse, and invalidate a normalization artifact
- Locate marker tables through the marker index

## Dataset

`DataStore` is the main entry point. Each assay owns feature metadata, normalization, and
feature selection. Cell-level columns are shared across assays.

```{code-cell} ipython3
import scarf

scarf.configure_output(level='WARNING', progress=False)
```

This page opens the pre-analyzed Bastidas-Ponce pancreas store also used in
{doc}`plotting` and {doc}`cell_cycle`. The store already includes HVGs, a
neighbourhood graph, UMAP coordinates, and cluster labels, so no graph or
clustering bootstrap is required here.

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

**The `I` column** tracks active cells. Values are boolean: filtered-out cells are `False`.
Most `DataStore` methods take `cell_key` (default `I`) and operate only on cells marked `True`.

Each assay group holds `counts`, `featureData`, optional `markers`, and analysis
outputs. Released tutorial archives often still use encoded paths such as
`normed__I__hvgs`. Current Scarf also writes provenance-backed groups under
`{assay}/artifacts/...`. Both layouts can coexist in one store.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA', depth=1)
```

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

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

`fetch` returns values for the active subset (default column `I`). `fetch_all` returns every
row in the store.

```{code-cell} ipython3
cluster_labels = ds.cells.fetch('clusters')
cluster_labels.shape, ds.cells.fetch_all('clusters').shape
```

`insert` writes a new column and aligns values to the active subset unless you override `key`.
Re-inserting an existing column requires `overwrite=True`.

```{code-cell} ipython3
first_cluster = str(cluster_labels[0])
is_first_cluster = cluster_labels.astype(str) == first_cluster
ds.cells.insert(column_name='is_first_cluster', values=is_first_cluster, overwrite=True)
```

```{code-cell} ipython3
try:
    ds.cells.insert(
        column_name='is_first_cluster',
        values=is_first_cluster,
    )
except ValueError as error:
    print(error)
```

See the `MetaData` API in {doc}`../reference/api/assays` for delete and update helpers.
Useful query helpers:

```python
idx = ds.cells.get_index_by(['3'], 'clusters')
keep = ds.cells.sift('RNA_nCounts', min_v=1000, max_v=15000)
```

### 3. Count matrices and normalization

Raw counts live under each assay's `counts` group and are exposed as `rawData`, a chunked array
with a NumPy-like interface that streams by row. Routine analysis does not need to touch this
object directly.

```{code-cell} ipython3
ds.RNA.rawData
```

Normalized values are computed on demand through `normed()`. Scarf keeps only raw counts on
disk by default.

```{code-cell} ipython3
ds.RNA.normed()
```

Override normalization by assigning `normMethod`. Reassign a custom function each time you
open the store. `scarf.assay.norm_dummy` disables normalization for pre-normalized inputs.

```{code-cell} ipython3
print('Current method:', ds.RNA.normMethod.__name__)
```

### 4. Released paths versus artifacts

Downloaded tutorial archives were often produced before provenance-backed
artifacts. On such a store, assay state and artifact listings can be empty even
when encoded graph groups exist:

```{code-cell} ipython3
print('Published RNA state:', ds.get_assay_state('RNA'))
print('Artifacts:', ds.list_artifacts())
```

`get_normalized_group_path` falls back to the released encoded path when no
matching artifact state is published:

```{code-cell} ipython3
ds.get_normalized_group_path('RNA', 'I', 'hvgs')
```

If that path exists on this archive, inspect it as a plain Zarr group:

```{code-cell} ipython3
encoded = ds.get_normalized_group_path('RNA', 'I', 'hvgs')
if encoded in ds.z:
    ds.show_zarr_tree(start=encoded, depth=1)
```

Create one cheap current-format artifact with atomic normalization. Prefer a
feature key that already exists on the store (`hvgs` here when present):

```{code-cell} ipython3
feat_key = 'hvgs' if 'I__hvgs' in ds.RNA.feats.columns else 'I'
normalized = ds.run_normalization(feat_key=feat_key)
normalized
```

State and listings now include the published artifact. The normalized path
switches to an artifact location:

```{code-cell} ipython3
state = ds.get_assay_state('RNA')
print('Cell and feature keys:', state.cell_key, state.feat_key)
print('Normalized artifacts:', ds.list_artifacts(kind='normalized'))
print('Normalized path:', ds.get_normalized_group_path('RNA', 'I', feat_key))
```

Inspect and load:

```{code-cell} ipython3
status = ds.inspect_artifact(normalized)
print('Complete:', status.complete)
print('Operation:', status.operation)
print('Parameters:', status.parameters)

group = ds.load_artifact(normalized)
print('Arrays:', list(group.array_keys())[:5])
```

Identical provenance reuses the same ref. `invalidate_cache=True` forces a new
artifact ID:

```{code-cell} ipython3
reused = ds.run_normalization(feat_key=feat_key)
forced = ds.run_normalization(feat_key=feat_key, invalidate_cache=True)
print('Reused the same artifact:', reused == normalized)
print('Forced a new artifact:', forced != normalized)
```

`load_graph` still returns a sparse matrix when a published connectivity map (or
compatible released graph) is available. Prefer letting Scarf resolve the graph
internally unless you need the matrix for a custom method.

```{code-cell} ipython3
try:
    ds.load_graph(
        from_assay='RNA',
        cell_key='I',
        feat_key=feat_key,
        symmetric=False,
        upper_only=False,
    )
except Exception as exc:
    print(type(exc).__name__, exc)
```

Concepts: {doc}`../concepts/provenance` and
{doc}`../concepts/graph_and_state`. Hands-on reuse:
{doc}`provenance_and_reuse`.

### 5. Marker features

`run_marker_search` publishes a `marker_table` artifact. The assay keeps a small
index under `{assay}/markers` whose `artifacts` attribute maps
`{cell_key}__{group_key}` slots to those refs. Fetch one group with
`get_markers`, or export all groups with `export_markers_to_csv`.

```{code-cell} ipython3
if 'RNA/markers' not in ds.z:
    ds.run_marker_search(group_key='clusters')
ds.show_zarr_tree(start='RNA/markers', depth=2)
```

```{code-cell} ipython3
index = dict(ds.z['RNA/markers'].attrs.get('artifacts', {}))
index
```

```{code-cell} ipython3
ds.get_markers(
    group_key='clusters',
    group_id=ds.cells.fetch('clusters')[0],
    min_score=0.1,
    min_frac_exp=0.1,
).head()
```

### 6. Zarr v3, memory, and remote stores

Current Scarf versions write new datasets as Zarr v3. Existing v2 stores remain
readable. Count matrices from the writers use sharded arrays (default profile
`fast_local`). Set the profile with `SCARF_ZARR_PROFILE` (`fast_local` or
`cloud`) or `zarrProfile=` when opening a `DataStore`.

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

Bound streaming memory with `mem_budget`:

```python
ds = scarf.DataStore('path/to/data.zarr', mem_budget='8G', nthreads=4)
```

For object storage, use `zarrProfile='cloud'` and pass `local_cache` on atomic
reduction methods (or rely on `"auto"`) to stage normalized data locally before
multi-pass PCA. KNN reads persisted reduced coordinates. See
{doc}`remote_stores`.

## Common mistakes

- Expecting filtered cells to be deleted instead of marked `False` in `I`
- Treating `MetaData` as an in-memory pandas DataFrame
- Using `fetch` when values for inactive cells are also required (`fetch_all`)
- Assuming a downloaded archive already has `AssayState` or `list_artifacts()`
  entries

## Saved results

Metadata changes and artifacts are written to the Zarr store. Marker payloads
live under `{assay}/artifacts/marker_table/...` with an index under
`{assay}/markers`. WAGGR and AUCell results are stored below
`<assay>/enrichment/<label>`; see {doc}`gene_set_enrichment` for the scoring APIs.

## Next steps

- {doc}`provenance_and_reuse`
- {doc}`remote_stores`
- {doc}`import_and_export`
- {doc}`../reference/api`
- {doc}`../developers/zarr_internals`
