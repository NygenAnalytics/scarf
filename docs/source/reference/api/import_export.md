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

Use this before `H5adReader` when you do not know which matrix and metadata keys an
H5AD file uses.

`H5adReader` decodes AnnData categorical columns and pandas nullable columns.
Missing categorical or object values become `None`; missing numeric values
become `NaN`. Unsupported group encodings are skipped with a warning.

Dense `obsm` arrays with one row per cell are flattened into numbered metadata
columns, such as `X_umap1` and `X_umap2`. Sparse arrays, group-encoded slots, and
arrays with an unexpected row count are warned about and skipped.

```{eval-rst}
.. autofunction:: scarf.inspect_h5ad
```

```{eval-rst}
.. autoclass:: scarf.H5adInspectResult
    :members:
```

## Matrix Market inspection

`inspect_mtx` reports every complete matrix, feature, and cell triplet in a
supported source. Pass one returned candidate to `MtxReader`.

```{eval-rst}
.. autofunction:: scarf.inspect_mtx
```

```{eval-rst}
.. autoclass:: scarf.readers.MtxCandidate
```

## Seurat import

Import a serialized Seurat object from an `.rds` file. Inspect with
`inspect_seurat`, open `SeuratReader`, then write with `SeuratToZarr`. This path
does not attach to a live R session and does not read or write `.h5seurat`. See
{doc}`../../tutorials/import_and_export` for the worked contract and
{doc}`../../scanpy_and_seurat` for workflow mapping.

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
.. autofunction:: scarf.writers.dask_to_zarr
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

{py:meth}`scarf.datastore.datastore.DataStore.to_anndata` supports cell selection plus either
`feature_names` or `feature_indexes`. The two feature selectors are mutually
exclusive. `SubsetZarr` selects cells but retains every feature in each supplied
assay.

`to_h5ad` and `to_mtx` export a complete assay. For feature-selective disk
export, call `to_anndata` and use AnnData's writer.

H5AD export recognizes UMAP and t-SNE coordinate pairs and writes them to
`obsm`. H5AD import flattens supported dense `obsm` arrays into cell metadata;
it does not preserve an AnnData-style `obsm` container inside `DataStore`.

`CrToZarr`, `MtxToZarr`, `H5adToZarr`, and `SparseToZarr` select source batch
rows automatically when `batch_size` is omitted. The selection starts from the
smallest destination row-shard height and shrinks only when required by the
operation memory budget. Explicit positive values remain supported.

## Merge

Use `DataStoreMerge` to merge DataStores. Pass `assays=["RNA"]` when only one
assay type is needed. By default RNA assays also receive a feature-major
`countsT` matrix for faster gene-wise reads, which roughly doubles stored
counts for those assays. Interrupted merges resume at whole-component
boundaries (`cellData`, each assay `counts`, and each requested `countsT`)
rather than mid-matrix shards.

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
