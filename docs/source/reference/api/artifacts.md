# Artifacts and assay state API reference

An {term}`artifact` is a persisted result, and its {term}`provenance` is what allows Scarf to {term}`reuse` it.
Analysis code usually reaches artifacts through `DataStore` methods.
The types below are the public contracts for a metadata-only store summary, references, status records, lineage, and the assay's current {term}`analysis chain`.

See {doc}`../../concepts/provenance` and {doc}`../../tutorials/graph_construction`.

## Types

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.ArtifactRef
   scarf.ArtifactSelectionError
   scarf.ArtifactStatus
   scarf.ArtifactLineage
   scarf.AssayState
   scarf.DataStoreSummary
   scarf.storage.refs.ExternalArtifactRef
   scarf.storage.ARTIFACT_KINDS
```

```{eval-rst}
.. autoclass:: scarf.ArtifactRef
    :members:

.. autoclass:: scarf.ArtifactSelectionError
    :members:

.. autoclass:: scarf.ArtifactStatus
    :members:

.. autoclass:: scarf.ArtifactLineage
    :members:

.. autoclass:: scarf.AssayState
    :members:

.. autoclass:: scarf.DataStoreSummary
    :members:

.. autoclass:: scarf.storage.refs.ExternalArtifactRef
    :members:
```

Supported artifact kind names are listed in {py:data}`scarf.storage.ARTIFACT_KINDS`.

```{eval-rst}
.. autodata:: scarf.storage.ARTIFACT_KINDS
    :annotation:
```

## Selection validation failures

`ArtifactSelectionError` remains a `ValueError` for existing callers.
Its `code` and JSON-safe `context` attributes distinguish these conditions:

- `artifact_reference_mismatch`, `artifact_missing`, and `artifact_incomplete`
- `selection_table_missing`, `selection_column_missing`, and `selection_row_ids_missing`
- `selection_values_missing`, `row_identity_mismatch`, and `selection_values_changed`

The context identifies the artifact kind and ID, scope, assay, metadata table, and source column when available.
It does not choose a recovery or a scientifically preferred replacement.

## DataStore inspection helpers

Prefer these store-bound methods over calling the storage helpers with a raw Zarr root.
`summary()` scans the literal `I` cell and feature selections in blocks and omits store locations and credentials.
These counts do not follow another selection stored in `AssayState`.
Use the other helpers for deeper inspection of one result.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.summary
   scarf.DataStore.list_artifacts
   scarf.DataStore.inspect_artifact
   scarf.DataStore.load_artifact
   scarf.DataStore.lineage
   scarf.DataStore.get_assay_state
```

```{eval-rst}
.. automethod:: scarf.DataStore.summary
.. automethod:: scarf.DataStore.list_artifacts
.. automethod:: scarf.DataStore.inspect_artifact
.. automethod:: scarf.DataStore.load_artifact
.. automethod:: scarf.DataStore.lineage
.. automethod:: scarf.DataStore.get_assay_state
```

## Module-level helpers

These accept a Zarr root group.
They are useful in tooling; analysis notebooks should use the `DataStore` methods above.

```{eval-rst}
.. autofunction:: scarf.storage.list_artifacts
.. autofunction:: scarf.storage.inspect_artifact
```
