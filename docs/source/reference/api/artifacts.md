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
   scarf.ArtifactResolutionError
   scarf.IncompatibleAnalysisStateError
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

.. autoclass:: scarf.ArtifactResolutionError
    :members:

.. autoclass:: scarf.IncompatibleAnalysisStateError
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

## Feature selections and summaries

Feature-axis artifacts align to the assay's complete current feature-row order and carry ordered-row and payload integrity fingerprints.
Selections store a Boolean `values` array.
Feature summaries store sufficient statistics for one exact cell-selection artifact and are reused by detected-feature, HVG, prevalent-peak, and cell-cycle producers.
New code does not create or read mounted `summary_stats_*` groups.
RNA summaries persist `normed_tot`, `normed_n`, and `sigmas`; average and nonzero mean are derived zero-safely from those arrays and the selected-cell count.
ATAC summaries persist `prevalence` and `document_frequency`.

| Producer | Artifact inputs | Scientific identity |
|---|---|---|
| assay universe | none | dataset and ordered feature IDs |
| manual selection | `all_features` | supplied values fingerprint |
| RNA or ATAC summary | `cell_selection` | normalizer settings |
| detected features | `feature_summary` | `min_cells` |
| highly variable genes | `feature_summary` | resolved variance, mean, binning, blacklist, bounds, and cell-count settings |
| prevalent peaks | `feature_summary` | `top_n` |
| mapping overlap | mapping reference and query `all_features` | exact inputs |

`mark_hvgs`, `mark_prevalent_peaks`, `select_detected_features`, and `set_feature_selection` return their selection reference.
Publishing writes the same values under one exact plain label such as `hvgs`.
Labels use lowercase snake case, contain no double underscore, and are not prefixed with a cell key.
`I`, `ids`, `names`, `nCells`, `dropOuts`, and `all_features` are reserved user labels.
Writable feature operations ensure the internal `all_features` universe exists before resolution.
`resolve_features` is strictly read-only and raises `missing_universe` rather than creating a missing baseline.

Publication uses a store journal so interrupted metadata writes fail closed.
A completed label remains readable if only journal cleanup was interrupted, because the committed column and journal point to the same validated artifact.
An actual in-progress or conflicting publication raises `ArtifactResolutionError` instead of guessing.
Repointing a label never invalidates an older retained `ArtifactRef`.

## Feature projection from graphs

Graph-derived analyses follow named provenance inputs rather than scanning arbitrary ancestors:

```text
connectivity_map -> neighbors -> coordinates and ann_index
ann_index -> the same coordinates
batch_correction -> reduction -> normalized -> feature_selection
```

Native graphs project one selection.
Imported-coordinate graphs project none.
Integrated graphs follow their ordered `source_i` inputs and can project zero, one, or several distinct selections while preserving source order.
WNN integration is stricter than general projection and accepts only native reduction or batch-correction coordinates.

## Selection validation failures

`ArtifactResolutionError` is a `ValueError` with machine-readable failure details.
Its `code` and JSON-safe `context` attributes distinguish these conditions:

- `artifact_reference_mismatch`, `artifact_missing`, and `artifact_incomplete`
- `selection_table_missing`, `selection_column_missing`, and `selection_row_ids_missing`
- `selection_values_missing`, `row_identity_mismatch`, and `selection_values_changed`
- `dimreduc_row_count_mismatch` and `dimreduc_cell_identity_mismatch`
- `wrong_kind`, `wrong_scope`, `wrong_assay`, `missing_artifact`, and `incomplete_artifact`
- `corrupt_payload`, `row_mismatch`, `invalid_label`, `missing_label`, and `unlinked_label`
- `stale_label`, `pending_alias`, `label_collision`, and `missing_universe`
- `missing_current_graph`, `missing_current_neighbors`, and `unsupported_graph_kind`

The context identifies the artifact kind and ID, scope, assay, metadata table, and source column when available.
It does not choose a recovery or a scientifically preferred replacement.

## DataStore inspection helpers

Prefer these store-bound methods over calling the storage helpers with a raw Zarr root.
`summary()` scans the literal `I` cell and feature columns in blocks and omits store locations and credentials.
These counts do not follow another selection stored in `AssayState`.
Use the other helpers for deeper inspection of one result.

`AssayState` stores no feature key or duplicate feature-selection field.
When `normalized` is present, that artifact names its `feature_selection` input.
Imported-coordinate state may have `normalized=None`; graph projection then returns no feature selection.
Use `resolve_features(assay, label_or_ref)` for a strict, read-only lookup.
Plain labels resolve exactly; the reserved `all_features` label names the complete assay feature universe.
Legacy analysis state with feature-key fields raises `IncompatibleAnalysisStateError` before analysis or mutation.
Its code is `legacy_feature_contract`; other malformed or unknown state uses `invalid_analysis_state`.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.summary
   scarf.DataStore.list_artifacts
   scarf.DataStore.inspect_artifact
   scarf.DataStore.load_artifact
   scarf.DataStore.lineage
   scarf.DataStore.get_assay_state
   scarf.DataStore.resolve_features
```

```{eval-rst}
.. automethod:: scarf.DataStore.summary
.. automethod:: scarf.DataStore.list_artifacts
.. automethod:: scarf.DataStore.inspect_artifact
.. automethod:: scarf.DataStore.load_artifact
.. automethod:: scarf.DataStore.lineage
.. automethod:: scarf.DataStore.get_assay_state
.. automethod:: scarf.DataStore.resolve_features
```

## Module-level helpers

These accept a Zarr root group.
They are useful in tooling; analysis notebooks should use the `DataStore` methods above.

```{eval-rst}
.. autofunction:: scarf.storage.list_artifacts
.. autofunction:: scarf.storage.inspect_artifact
```
