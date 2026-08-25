# Assays and metadata API reference

`Assay.score_features(feature_names, cell_key, ctrl_size, n_bins, rand_seed)` is computation-only.
It computes full-row-order averages blockwise in memory and is safe on read-only counts; it does not plan artifacts, mount statistics, or write metadata.
Persistent feature summaries and cell-cycle outputs belong to `DataStore.run_cell_cycle_scoring`.

Normalized materialization uses explicit cell and feature indexes through `Assay.save_normalized_data(cell_idx, feat_idx, location, ...)`.
Lower assay methods do not resolve public feature labels.

```{eval-rst}
.. autoclass:: scarf.assay.Assay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.assay.RNAassay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.assay.ATACassay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.assay.ADTassay
    :members:
```

```{eval-rst}
.. autofunction:: scarf.assay.norm_dummy
```

```{eval-rst}
.. autoclass:: scarf.metadata.MetaData
    :members:
```

## ATAC coordinate melding

```{eval-rst}
.. autoclass:: scarf.GffReader
    :members:
```

```{eval-rst}
.. autofunction:: scarf.coordinate_melding
```
