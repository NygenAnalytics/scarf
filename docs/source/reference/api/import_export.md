# Import and export

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

## Merge

Use `AssayMerge` for new code. `ZarrMerge` is a deprecated compatibility
subclass retained through Scarf 1.x.

```{eval-rst}
.. autoclass:: scarf.merge.AssayMerge
    :members:
```

```{eval-rst}
.. autoclass:: scarf.merge.DatasetMerge
    :members:
```

```{eval-rst}
.. autoclass:: scarf.merge.ZarrMerge
    :members:
```
