# API

## DataStore classes

### BaseDataStore
```{eval-rst}
.. autoclass:: scarf.datastore.base_datastore.BaseDataStore
    :members:
```

### GraphDataStore
```{eval-rst}
.. autoclass:: scarf.datastore.graph_datastore.GraphDataStore
    :members:
```

### MappingDatastore
```{eval-rst}
.. autoclass:: scarf.datastore.mapping_datastore.MappingDatastore
    :members:
```

### MappingReference
```{eval-rst}
.. autoclass:: scarf.mapping_reference.MappingReference
    :members:
```

```{eval-rst}
.. autoclass:: scarf.mapping_reference.MappingResult
    :members:
```

### DataStore
```{eval-rst}
.. autoclass:: scarf.datastore.datastore.DataStore
    :members:
```

## Assay classes

### Assay
```{eval-rst}
.. autoclass:: scarf.assay.Assay
    :members:
```

### RNAassay
```{eval-rst}
.. autoclass:: scarf.assay.RNAassay
    :members:
```

### ATACassay
```{eval-rst}
.. autoclass:: scarf.assay.ATACassay
    :members:
```

### ADTassay
```{eval-rst}
.. autoclass:: scarf.assay.ADTassay
    :members:
```

## MetaData
```{eval-rst}
.. autoclass:: scarf.metadata.MetaData
    :members:
```

## Harmony
```{eval-rst}
.. autofunction:: scarf.harmony.run_harmony
```

```{eval-rst}
.. autofunction:: scarf.harmony.fit_harmony
```

```{eval-rst}
.. autoclass:: scarf.harmony.HarmonyResult
    :members:
```

## Integration metrics
```{eval-rst}
.. automodule:: scarf.metrics
    :members: compute_lisi, silhouette_scoring, label_concordance_score, lisi_batch_mixing_score, integration_score
```

## Merge classes

### AssayMerge
```{eval-rst}
.. autoclass:: scarf.merge.AssayMerge
    :members:
```

### DatasetMerge
```{eval-rst}
.. autoclass:: scarf.merge.DatasetMerge
    :members:
```

### ZarrMerge (deprecated)
```{eval-rst}
.. autoclass:: scarf.merge.ZarrMerge
    :members:
```

## Meld assay
```{eval-rst}
.. autoclass:: scarf.meld_assay.GffReader
    :members:
```

```{eval-rst}
.. autofunction:: scarf.meld_assay.coordinate_melding
```

## Downloader
```{eval-rst}
.. autofunction:: scarf.downloader.fetch_dataset
```

```{eval-rst}
.. autofunction:: scarf.downloader.show_available_datasets
```

## Utilities
```{eval-rst}
.. autofunction:: scarf.utils.set_verbosity
```

```{eval-rst}
.. autofunction:: scarf.utils.load_zarr
```

```{eval-rst}
.. autofunction:: scarf.utils.controlled_compute
```

## Reader classes

### Cellranger H5 reader
```{eval-rst}
.. autoclass:: scarf.readers.CrH5Reader
    :members:
```

### Cellranger directory (MTX) reader
```{eval-rst}
.. autoclass:: scarf.readers.CrDirReader
    :members:
```

### H5ad (Anndata) reader
```{eval-rst}
.. autoclass:: scarf.readers.H5adReader
    :members:
```

### Loom reader
```{eval-rst}
.. autoclass:: scarf.readers.LoomReader
    :members:
```

### Nabo H5 reader
```{eval-rst}
.. autoclass:: scarf.readers.NaboH5Reader
    :members:
```

### CSV reader
```{eval-rst}
.. autoclass:: scarf.readers.CSVReader
    :members:
```

## Writer classes

### Cellranger to Zarr
```{eval-rst}
.. autoclass:: scarf.writers.CrToZarr
    :members:
```

### H5ad (Anndata) to Zarr
```{eval-rst}
.. autoclass:: scarf.writers.H5adToZarr
    :members:
```

### Nabo H5 to Zarr
```{eval-rst}
.. autoclass:: scarf.writers.NaboH5ToZarr
    :members:
```

### Loom to Zarr
```{eval-rst}
.. autoclass:: scarf.writers.LoomToZarr
    :members:
```

### Sparse matrix to Zarr
```{eval-rst}
.. autoclass:: scarf.writers.SparseToZarr
    :members:
```

### CSV to Zarr
```{eval-rst}
.. autoclass:: scarf.writers.CSVtoZarr
    :members:
```

### Subset Zarr
```{eval-rst}
.. autoclass:: scarf.writers.SubsetZarr
    :members:
```

```{eval-rst}
.. autofunction:: scarf.writers.dask_to_zarr
```

```{eval-rst}
.. autofunction:: scarf.writers.to_h5ad
```

```{eval-rst}
.. autofunction:: scarf.writers.to_mtx
```

## Plots
```{eval-rst}
.. autofunction:: scarf.plots.plot_graph_qc
```
