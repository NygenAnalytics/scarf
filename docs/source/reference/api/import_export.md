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

## Writers

```{eval-rst}
.. autoclass:: scarf.writers.CrToZarr
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

## Merge

Use `AssayMerge` to merge assay matrices across samples.

```{eval-rst}
.. autoclass:: scarf.merge.AssayMerge
    :members:
```

```{eval-rst}
.. autoclass:: scarf.merge.DatasetMerge
    :members:
```
