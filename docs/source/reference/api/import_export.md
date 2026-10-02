# Import and export API reference

## Readers

```{eval-rst}
.. autoclass:: scarf.readers.CrReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CrH5Reader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CrDirReader
    :members:
    :inherited-members:
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: scarf.readers.MtxReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.H5adReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CSVReader
    :members:
```

## H5AD inspection

Use this before `H5adReader` when you do not know which matrix and metadata keys an H5AD file uses.

`H5adReader` checks the matrix shape against the lengths of `obs` and `var` when it is
constructed, and a mismatch raises `ValueError`. Overriding `matrix_key`, for example with
`raw/X`, in the constructor or through `H5adReader.from_inspect`, therefore also needs the
matching `feature_attrs_key`, here `raw/var`. `inspect_h5ad(path, matrix_key="raw/X")` returns
reader arguments that pair them.

`H5adReader` decodes AnnData categorical columns and pandas nullable columns, including the
nullable string arrays that AnnData writes for dataframe indexes. By default it reads the index
that the dataframe's `_index` attribute names.
Missing categorical or object values become `None`; missing numeric values become `NaN`.
Unsupported group encodings and multi-dimensional columns are skipped with a warning.
`H5adToZarr` keeps missing rows under a linked missing mask and stores nullable booleans as
booleans.

Pass `embedding_roles` and `cluster_keys` to select analytical H5AD values for artifact import.
Selected `obsm` arrays and cluster labels are excluded from live metadata and returned as exact
refs by `H5adToZarr.dump()`. Other supported `obs` columns remain literal metadata.

CSC input is converted to temporary row storage on local disk. Pass `temp_dir` to
`H5adReader` or `H5adReader.from_inspect` to choose its parent directory; the default uses
the system temporary directory, including `TMPDIR`. The temporary files are removed when the
reader closes.

```{eval-rst}
.. autofunction:: scarf.inspect_h5ad
```

```{eval-rst}
.. autoclass:: scarf.H5adInspectResult
    :members:
```

```{eval-rst}
.. autoclass:: scarf.H5adImportResult
    :members:
```

## Matrix Market inspection

`inspect_mtx` reports every complete matrix, feature, and cell triplet in a supported source.
Pass one returned candidate to `MtxReader`.

```{eval-rst}
.. autofunction:: scarf.inspect_mtx
```

```{eval-rst}
.. autoclass:: scarf.readers.MtxCandidate
```

## Seurat import

Import a serialized Seurat object from an `.rds` file.
Inspect with `inspect_seurat`, open `SeuratReader`, then write with `SeuratToZarr`.
This path does not attach to a live R session and does not read or write `.h5seurat`.
Sidecar matrices such as BPCells directories and HDF5 files resolve inside `sidecar_root`, which
defaults to the directory of the `.rds` file. A reader opened from a stream needs an explicit
`sidecar_root` before it reads sidecar-backed layers.
See {doc}`../../tutorials/import_and_export` for the worked contract and {doc}`../../seurat` for
workflow mapping.

```{eval-rst}
.. autoclass:: scarf.SeuratReader
    :members:

.. autofunction:: scarf.inspect_seurat

.. autoclass:: scarf.SeuratInspectResult
    :members:

.. autoclass:: scarf.SeuratImportResult
    :members:

.. autoclass:: scarf.SeuratToZarr
    :members:
```

## Storage controls

Writers, merge, subset, and ``DataStore`` accept the same optional storage
controls. ``profile`` chooses the physical encoding. ``policy`` chooses paired
count-matrix geometry, and a writer uses it exactly. Without it, every count
writer (the imports, subset, ``repack_zarr``, merge, and ``add_grouped_assay``)
fits the geometry to ``mem_budget``: it halves the default ``unitBytes`` and
``chunkBytes`` together until the counts write and the ``countsT`` transpose
fit. Writers that choose their source batches (the sparse and Seurat imports,
and merge) admit batches of one destination row band, so a fitted geometry
never leaves the write narrower batches than its bands. A write that does not
fit, with one-row count shards or with its explicit ``policy``, fails before it
creates the destination: writers raise when they are constructed, and merge
when it plans. ``MtxToZarr`` and ``CrToZarr`` take ``lines_in_mem`` in the
constructor, so the fit reserves the Matrix Market parse buffer that the write
uses. A resumed merge keeps the geometry of its completed counts. The geometry
never changes the store's identity. ``io`` overrides automatic read, compute,
and write widths. Unset values stay under automatic planning from
``mem_budget`` and ``nthreads``.

```{eval-rst}
.. autoclass:: scarf.storage.io_policy.StorageIoPolicy
    :members:

.. autoclass:: scarf.storage.count_matrix.CountMatrixPolicy
    :members:
```

### Count storage dtype

Every import stores each assay's counts in a dtype resolved from the values of
that assay, whatever the reader and source encoding, and no import takes a count
dtype argument. When every canonical (duplicate-summed) value is a non-negative
integer, the counts are stored in the narrowest of ``uint8``, ``uint16``,
``uint32``, and ``uint64`` that holds them. Other counts keep their source
dtype in native byte order, with ``float16`` read as ``float32``. Assays split from one source
matrix (10x HDF5 and Matrix Market feature types, H5AD ``assay_split_key``)
each resolve their own dtype. Each import reads every count once before it
creates the destination: ``CrReader`` subclasses and ``H5adReader`` report the
range of each group of features over the selected cells through
``count_value_ranges``, ``CSVReader`` records it as ``countRange`` in its first
pass, and ``SparseToZarr`` and ``SeuratToZarr`` scan their sources. Count
matrices hold finite values, so imports reject NaN and infinity before they
create the destination. Writers cast through a checked cast, so a count that
the stored dtype cannot hold raises instead of wrapping.

Subset and repack keep the source dtype. Merge stores the common type of the
source count dtypes, widened so that features summed by name cannot overflow
it, and rejects integer sources without a common integer dtype. Grouped and
melded assays keep their ``float64`` values.

```{eval-rst}
.. autofunction:: scarf.storage.count_dtype.count_storage_dtype

.. autoclass:: scarf.utils.count_values.CountValueRange
    :members:
```

## Writers

Every writer owns the reserved metadata columns `ids`, `names` and `I` and the
`__scarf_missing__` prefix of missing-value masks. Source cell or feature metadata columns
with these names are skipped with a warning, so imported identifiers are never replaced.
Zarr reads `/` and `\` in a column name as path separators, so writers store a source column
whose name contains either one under the name with `_` in their place and log the rename. A
source name that is already valid keeps its name; a renamed column whose name is taken gets
the first free `_2`, `_3`, and so on. `SeuratToZarr` raises for reserved names instead of
skipping them.
The H5AD, CSV, Matrix Market, and Cell Ranger readers reject missing or repeated cell and
feature IDs.

Import writers take their assay type from the assay name. Pass `assay_type` (or
`assay_types` on `CrToZarr`) to declare a custom-named assay as a preset such as `RNA`.

```{eval-rst}
.. autoclass:: scarf.writers.CrToZarr
    :members:
```

`MtxToZarr` is an alias of `CrToZarr` for use with `MtxReader`.

```{eval-rst}
.. autoclass:: scarf.writers.MtxToZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.H5adToZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.SparseToZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.CSVtoZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.SubsetZarr
    :members:
```

```{eval-rst}
.. autofunction:: scarf.writers.to_h5ad
```

```{eval-rst}
.. autofunction:: scarf.writers.to_mtx
```

```{eval-rst}
.. autofunction:: scarf.writers.chunked_to_zarr
```

```{eval-rst}
.. autofunction:: scarf.writers.create_zarr_dataset
```

```{eval-rst}
.. autofunction:: scarf.writers.create_zarr_obj_array
```

```{eval-rst}
.. autofunction:: scarf.writers.create_zarr_count_assay
```

```{eval-rst}
.. autofunction:: scarf.writers.subset_assay_zarr
```

```{eval-rst}
.. autofunction:: scarf.writers.write_renorm_subset_to_zarr
```

## Selection and layout behavior

{py:meth}`scarf.datastore.datastore.DataStore.to_anndata` supports cell selection plus either `feature_names` or `feature_indexes`.
The two feature selectors are mutually exclusive.
`SubsetZarr` selects cells but retains every feature in each supplied assay.

Without a `run`, `to_h5ad` and `to_mtx` export a complete assay.
For feature-selective disk export outside a pipeline run, call `to_anndata` and use AnnData's
writer.

Nullable metadata columns keep a stored placeholder in each row that their linked missing mask
flags. `to_anndata` and `to_h5ad` export those rows as missing values with or without `run`:
numeric columns become float64 with `NaN`, boolean columns become nullable booleans, and other
columns hold a missing value, which H5AD files store as a missing category.

Pass a completed run to write its frozen cells, feature universe, and result fields directly:

```python
scarf.to_h5ad(ds.RNA, "analysis.h5ad", run=run)
```

The assay must be the exact assay object owned by the datastore that opened the run. Run export
uses {py:meth}`~scarf.datastore.datastore.DataStore.to_anndata` to preserve the frozen view,
writes its UMAP fields to `obsm["X_umap"]`, and keeps `clusters` in `obs`. `embeddings_cols`,
feature-count recalculation, and a writer-specific thread override apply only to ordinary
full-assay export and are rejected when `run` is supplied.

`H5adToZarr.dump()` returns an `H5adImportResult` containing the written assays, analysis assay,
all-cell selection, and imported embedding and clustering refs. Set `analysis_assay` when a
multi-assay import selects analytical values. Use `DataStore.load_artifact(ref)` for payload access
or pass the exact ref to a consumer. Import does not flatten these results into metadata columns.

Ordinary `to_h5ad` export writes a complete assay and live metadata. Run-aware export reads only
the completed run's frozen selections and fields, so export does not require physical result
columns.

`CrToZarr`, `MtxToZarr`, `H5adToZarr`, `SparseToZarr`, and `SeuratToZarr` select source batch
rows automatically when `batch_size` is omitted, starting from the smallest destination
row-shard height, which the fitted count layout admits. An explicit positive `batch_size` is
capped at that height.

## Merge

Use `DataStoreMerge` to merge DataStores.
Pass `assays=["RNA"]` when only one assay type is needed.
Features are matched by exact feature ID by default. Gene symbols are display labels and do not
merge distinct IDs; suffixes such as `_1` are preserved. An assay present in multiple inputs
raises an error if none of its IDs overlap.
When inputs use different identifier conventions for the same genes, for example Ensembl IDs in
one input and gene symbols in another, pass `feature_key="names"` to match features by name.
The merged feature IDs are then the names, and features that share a name within one input are
summed.
Feature annotation columns are merged when the inputs agree. A column whose values differ for a
shared feature, such as per-dataset highly variable gene flags, is left out with a warning. The same
applies when features that share a name within one input disagree.
Merged cell IDs are `{name}__{cell_id}`, so source names cannot contain `__`. Cell metadata columns
keep their attributes when the inputs agree. Differing `levels` of an unordered categorical column
become their union in first-seen order; any other differing attribute, including ordered levels, is
dropped with a warning.
RNA assays write both `counts` and a gene-major `countsT` copy, which roughly doubles stored counts for those assays.
Non-RNA assays never write `countsT`.
The destination must be empty or hold a merge written by `DataStoreMerge`. A destination that
aliases, contains, or lies inside a source store, or that already holds other content, is refused.
`overwrite=True` replaces only a merge-owned destination whose assays have not been prepared by
opening it as a `DataStore`, and it clears the recorded default assay.
Interrupted merges resume at whole-component boundaries (`cellData`, each assay `counts`, and each RNA `countsT`) rather than mid-matrix.
A resume requires the same configuration and the same source counts; a completed `countsT` whose
layout differs from the plan is not rewritten in place. Completed counts keep the layout persisted
with them, so a resume under another `mem_budget` reuses them, while counts that a resume rewrites
get a layout fitted to its budget.

```{eval-rst}
.. autoclass:: scarf.merge.DataStoreMerge
    :members:

.. autoclass:: scarf.merge.MergePlan
    :members:

.. autoclass:: scarf.merge.AssayMergePlan
    :members:

.. autoclass:: scarf.merge.MergeResult
    :members:

.. autoclass:: scarf.merge.ComponentResult
    :members:
```
