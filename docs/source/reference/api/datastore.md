# DataStore

`DataStore` is the primary analyst-facing object. It inherits graph, mapping, and assay
helpers from the classes below. Prefer this page over the inheritance appendix unless you
are extending Scarf.

The atomic graph methods are documented on {doc}`graph_ops` and the artifact inspection
methods on {doc}`artifacts`, so they are excluded here rather than repeated.

```{eval-rst}
.. autoclass:: scarf.datastore.datastore.DataStore
    :members:
    :inherited-members:
    :exclude-members: run_normalization, run_pca, run_lsi, run_custom_reduction,
        run_harmony, build_embedding_initialization, build_ann_index, query_neighbors,
        build_connectivity_map, list_artifacts, inspect_artifact, load_artifact,
        get_assay_state
```

## Store-bound plotting

`DataStore.plots` binds the datastore argument for canonical store-first
functions in `scarf.plotting`. Array and DataFrame diagnostics such as
`elbow`, `qc`, `graph_qc`, and `highly_variable_features` remain standalone.

```{eval-rst}
.. autoclass:: scarf.datastore.plot_accessor.DataStorePlotAccessor
    :members:
```

## Analysis results

```{eval-rst}
.. autoclass:: scarf.FateMappingResult
    :members:

.. autoclass:: scarf.PseudotimeScoreResult
    :members:

.. autoclass:: scarf.PseudotimeMarkerResult
    :members:

.. autoclass:: scarf.PseudotimeAggregationResult
    :members:

.. autoclass:: scarf.EnrichmentResult
    :members:

.. autoclass:: scarf.clustering.ParisClusteringResult
    :members:

.. autoclass:: scarf.clustering.ParisClusterDiagnostic
    :members:
```

## Inheritance appendix

These classes exist for implementation structure. Analysts should not need to construct them
directly.

```{eval-rst}
.. autoclass:: scarf.datastore.base_datastore.BaseDataStore
    :members:
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.graph_datastore.GraphDataStore
    :members:
    :inherited-members:
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.mapping_datastore.MappingDatastore
    :members:
    :inherited-members:
    :no-index:
```
