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
.. autoclass:: scarf.readers.LoomReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CSVReader
    :members:
```

## H5AD inspection

Use this before `H5adReader` when you do not know which matrix and metadata keys an H5AD file uses.

`H5adReader` decodes AnnData categorical columns and pandas nullable columns.
Missing categorical or object values become `None`; missing numeric values become `NaN`.
Unsupported group encodings are skipped with a warning.

Pass `embedding_roles` and `cluster_keys` to select analytical H5AD values for artifact import.
Selected `obsm` arrays and cluster labels are excluded from live metadata and returned as exact
refs by `H5adToZarr.dump()`. Other supported `obs` columns remain literal metadata.

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
count-matrix geometry. ``io`` overrides automatic read, compute, and write
widths. Unset values stay under automatic planning from ``mem_budget`` and
``nthreads``.

```{eval-rst}
.. autoclass:: scarf.storage.io_policy.StorageIoPolicy
    :members:

.. autoclass:: scarf.storage.count_matrix.CountMatrixPolicy
    :members:
```

## Writers

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
.. autoclass:: scarf.writers.LoomToZarr
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

`CrToZarr`, `MtxToZarr`, `H5adToZarr`, and `SparseToZarr` select source batch rows automatically when `batch_size` is omitted.
The selection starts from the smallest destination row-shard height and shrinks only when required by the operation memory budget.
Explicit positive values remain supported.

## Merge

Use `DataStoreMerge` to merge DataStores.
Pass `assays=["RNA"]` when only one assay type is needed.
RNA assays write both `counts` and a gene-major `countsT` copy, which roughly doubles stored counts for those assays.
Non-RNA assays never write `countsT`.
Interrupted merges resume at whole-component boundaries (`cellData`, each assay `counts`, and each RNA `countsT`) rather than mid-matrix.

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
