# DataStore

`DataStore` is the primary analyst-facing object. It inherits graph, mapping, and assay
helpers from the classes below. Prefer this page over the inheritance appendix unless you
are extending Scarf.

```{eval-rst}
.. autoclass:: scarf.datastore.datastore.DataStore
    :members:
    :inherited-members:
    :show-inheritance:
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
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.mapping_datastore.MappingDatastore
    :members:
    :no-index:
```
