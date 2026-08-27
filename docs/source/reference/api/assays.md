# Assays and metadata API reference

`Assay.score_features(feature_names, cell_key, ctrl_size, n_bins, rand_seed)` is computation-only.
It computes full-row-order averages blockwise in memory and is safe on read-only counts; it does
not plan artifacts or write metadata. Persistent cell-cycle outputs belong to
`DataStore.run_cell_cycle_scoring`.

Feature-count percentages belong to
`DataStore.run_feature_percentage(cell_selection, features)`. It derives the assay from the exact
feature-selection ref and returns an assay-scoped `quality_metric` ref whose `values` array is the
per-cell percentage. It does not add a metadata column.

Persisted normalization belongs to
`DataStore.run_normalization(cell_selection, features)`, which returns an immutable artifact ref.
Assay normalization methods are computation-only and require explicit feature indexes where they
do not accept an `ArtifactRef`.

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
