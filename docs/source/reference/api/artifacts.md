# Artifacts and assay state

Logical artifacts are content-addressed results stored under the Zarr hierarchy. Analysis
code usually talks to them through `DataStore` methods. The types below are the public
contracts for references, status records, and the published assay pointer set.

See {doc}`../../concepts/provenance` and {doc}`../../concepts/graph_and_state`.

## Types

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.ArtifactRef
   scarf.ArtifactStatus
   scarf.AssayState
   scarf.storage.ARTIFACT_KINDS
```

```{eval-rst}
.. autoclass:: scarf.ArtifactRef
    :members:

.. autoclass:: scarf.ArtifactStatus
    :members:

.. autoclass:: scarf.AssayState
    :members:
```


Supported artifact kind names are listed in {py:data}`scarf.storage.ARTIFACT_KINDS`.

```{eval-rst}
.. autodata:: scarf.storage.ARTIFACT_KINDS
    :annotation:
```

## DataStore inspection helpers

Prefer these store-bound methods over calling the storage helpers with a raw Zarr root.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.list_artifacts
   scarf.DataStore.inspect_artifact
   scarf.DataStore.load_artifact
   scarf.DataStore.get_assay_state
```

```{eval-rst}
.. automethod:: scarf.DataStore.list_artifacts
.. automethod:: scarf.DataStore.inspect_artifact
.. automethod:: scarf.DataStore.load_artifact
.. automethod:: scarf.DataStore.get_assay_state
```

## Module-level helpers

These accept a Zarr root group. They are useful in tooling; analysis notebooks should use
the `DataStore` methods above.

```{eval-rst}
.. autofunction:: scarf.storage.list_artifacts
.. autofunction:: scarf.storage.inspect_artifact
```
