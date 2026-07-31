# Artifacts and assay state API reference

Logical artifacts are persisted results with provenance-backed reuse. Analysis
code usually reaches them through `DataStore` methods. The types below are the
public contracts for references, status records, lineage, and the assay's
current analysis chain.

See {doc}`../../concepts/provenance` and
{doc}`../../tutorials/custom_graph_construction`.

## Types

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.ArtifactRef
   scarf.ArtifactStatus
   scarf.ArtifactLineage
   scarf.AssayState
   scarf.storage.ARTIFACT_KINDS
```

```{eval-rst}
.. autoclass:: scarf.ArtifactRef
    :members:

.. autoclass:: scarf.ArtifactStatus
    :members:

.. autoclass:: scarf.ArtifactLineage
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
   scarf.DataStore.lineage
   scarf.DataStore.get_assay_state
```

```{eval-rst}
.. automethod:: scarf.DataStore.list_artifacts
.. automethod:: scarf.DataStore.inspect_artifact
.. automethod:: scarf.DataStore.load_artifact
.. automethod:: scarf.DataStore.lineage
.. automethod:: scarf.DataStore.get_assay_state
```

## Module-level helpers

These accept a Zarr root group. They are useful in tooling; analysis notebooks should use
the `DataStore` methods above.

```{eval-rst}
.. autofunction:: scarf.storage.list_artifacts
.. autofunction:: scarf.storage.inspect_artifact
```
