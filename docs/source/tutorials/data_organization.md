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
- Locate cached graph and marker results

## Dataset

`DataStore` is the main entry point. Each assay owns feature metadata, normalization, and
feature selection. Cell-level columns are shared across assays.

```{image} ../_static/scarf_organization.png
:alt: Scarf DataStore, assays, and metadata
:width: 800px
```

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')
scarf.__version__
```

This page uses the multimodal CITE-seq store also used in {doc}`cite_seq`. Fresh catalog
downloads need a short RNA graph and clustering pass so the examples below have columns to
inspect. If you already ran the CITE-seq chapter in the same working directory, that work is
reused.

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='tenx_8K_pbmc_citeseq',
    save_path='scarf_datasets',
    as_zarr=True
)

ds = scarf.DataStore(
    'scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr',
    default_assay='RNA',
    nthreads=4,
)

if 'RNA_leiden_cluster' not in ds.cells.columns:
    ds.mark_hvgs(min_cells=20, top_n=500)
    ds.make_graph(feat_key='hvgs', k=11, dims=15)
    ds.run_leiden_clustering(resolution=0.5)
if 'RNA_UMAP1' not in ds.cells.columns:
    ds.run_umap(n_epochs=100, parallel=True)

marker_slot = 'I__RNA_leiden_cluster'
rna_markers = ds.z['RNA']['markers'] if 'markers' in ds.z['RNA'] else None
if rna_markers is None or marker_slot not in rna_markers:
    ds.run_marker_search(group_key='RNA_leiden_cluster', gene_batch_size=100)

ds
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
    show=False,
).figure;
```

## Guided steps

### 1. Inspect Zarr trees

Scarf uses [Zarr](https://zarr.readthedocs.io/en/stable/) for chunked on-disk arrays. The store
is a directory tree: counts, cell and feature attributes, and cached intermediates live under
named groups. Benefits relative to a single HDF5 file include parallel reads and writes, fast
compression codecs, and automatic persistence of intermediate results.

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

Each assay group holds `counts`, `featureData`, optional `markers`, and cached normalized
matrices such as `normed__I__hvgs`.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA', depth=1)
```

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

### 2. Inspect cell and feature attributes

Cell and feature tables are `Metadata` objects (`ds.cells`, `ds.RNA.feats`), not pandas
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
    columns=['ids', 'RNA_nCounts', 'RNA_nFeatures', 'RNA_leiden_cluster']
).set_index('ids')
```

`fetch` returns values for the active subset (default column `I`). `fetch_all` returns every
row in the store.

```{code-cell} ipython3
cluster_labels = ds.cells.fetch('RNA_leiden_cluster')
cluster_labels.shape, ds.cells.fetch_all('RNA_leiden_cluster').shape
```

`insert` writes a new column and aligns values to the active subset unless you override `key`.
Re-inserting an existing column requires `overwrite=True`.

```{code-cell} ipython3
is_cluster_1 = cluster_labels == 1
ds.cells.insert(column_name='is_cluster_1', values=is_cluster_1, overwrite=True)
```

```{code-cell} ipython3
:tags: [raises-exception]

ds.cells.insert(
    column_name='is_cluster_1',
    values=is_cluster_1,
)
```

See the `Metadata` API for delete and update helpers. Useful query helpers:

```python
idx = ds.cells.get_index_by(['3'], 'RNA_leiden_cluster')
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

Override normalization by assigning `normMethod`. Reassign it each time you open the store if
you use a custom function. `scarf.assay.norm_dummy` disables normalization for pre-normalized
inputs.

```{code-cell} ipython3
import inspect

print(inspect.getsource(ds.RNA.normMethod))
```

```{code-cell} ipython3
def my_cool_normalization_method(assay, counts):
    import numpy as np

    lib_size = counts.sum(axis=1).reshape(-1, 1)
    return np.log2(counts / lib_size)

ds.RNA.normMethod = my_cool_normalization_method
ds.RNA.normMethod = scarf.assay.norm_dummy
```

### 4. Graph caching

`make_graph` caches results under a name like `normed__{cell_key}__{feat_key}`. Intermediate
PCA and ANN artefacts are kept so Scarf can skip recomputation when parameters match.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/normed__I__hvgs')
```

`load_graph` returns the latest graph for the given assay, cell key, and feature key as a
sparse matrix. Prefer letting Scarf load graphs internally unless you need the matrix for a
custom method.

```{code-cell} ipython3
ds.load_graph(
    from_assay='RNA',
    cell_key='I',
    feat_key='hvgs',
    symmetric=False,
    upper_only=False,
)
```

### 5. Marker features

`run_marker_search` writes markers under `{assay}/markers/{cell_key}__{group_key}`. Fetch one
group with `get_markers`, or export all groups with `export_markers_to_csv`.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/markers', depth=2)
```

```{code-cell} ipython3
ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id='1',
    min_score=0.1,
    min_frac_exp=0.1,
).head()
```

```{code-cell} ipython3
splt.marker_heatmap(
    ds,
    group_key='RNA_leiden_cluster',
    topn=3,
    figsize=(5, 7),
)
```

```{code-cell} ipython3
ds.export_markers_to_csv(
    group_key='RNA_leiden_cluster',
    csv_filename='test.csv',
    min_score=0.2,
    min_frac_exp=0.1,
)
```

### 6. Zarr v3, memory, and remote stores

Scarf 0.33+ writes new datasets as Zarr v3. Existing v2 stores remain readable. Count matrices
from the writers use sharded arrays (default profile `fast_local`). Set the profile with
`SCARF_ZARR_PROFILE` (`fast_local` or `cloud`) or `zarrProfile=` when opening a `DataStore`.

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

Bound streaming memory with `mem_budget`:

```python
ds = scarf.DataStore('path/to/data.zarr', mem_budget='8G', nthreads=4)
```

For object storage, use `zarrProfile='cloud'`. Pass `local_cache=True` to `make_graph` to stage
normalized data locally before PCA and KNN. After Harmony, corrected embeddings are stored under
the PCA reduction group as `harmonizedData` with `isHarmonized=True`.

## Common mistakes

- Expecting filtered cells to be deleted instead of marked `False` in `I`
- Treating `Metadata` as an in-memory pandas DataFrame
- Using `fetch` when values for inactive cells are also required (`fetch_all`)

## Saved results

Metadata changes, graph caches, and marker results are written to the Zarr store. Exported
marker tables go to the path passed to `csv_filename`.

## Next steps

- {doc}`import_and_export`
- {doc}`plotting`
- {doc}`dimensionality_reduction_and_clustering`
- {doc}`../developers/zarr_internals`
