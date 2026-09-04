# Artifacts, lineage, and summaries API reference

An {term}`artifact` is an immutable persisted result. Its {term}`provenance` lets Scarf verify its
inputs, inspect its lineage, and {term}`reuse` a completed match. Analysis code normally creates and
consumes artifacts through `DataStore` methods.

See {doc}`../../concepts/provenance`, {doc}`pipeline`, and
{doc}`../../tutorials/graph_construction`.

## Types

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.ArtifactRef
   scarf.ArtifactResolutionError
   scarf.ArtifactStatus
   scarf.ArtifactLineage
   scarf.DataStoreSummary
   scarf.storage.refs.ExternalArtifactRef
   scarf.storage.ARTIFACT_KINDS
```

```{eval-rst}
.. autoclass:: scarf.ArtifactRef
    :members:

.. autoclass:: scarf.ArtifactResolutionError
    :members:

.. autoclass:: scarf.ArtifactStatus
    :members:

.. autoclass:: scarf.ArtifactLineage
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

## Selections and summaries

Cell and feature selections are immutable Boolean artifacts aligned to a complete stored axis.
Their integrity includes the values and the exact ordered row identities. A live metadata column
may be the source of a selection, but changing that column does not change or invalidate the
historical artifact. Replacing, reordering, adding, or removing axis IDs fails closed; Scarf does
not remap a stored result by matching IDs.

Feature summaries hold sufficient statistics for one exact cell selection and can be reused by
detected-feature, HVG, prevalent-peak, and cell-cycle producers. New code does not create or read
mounted `summary_stats_*` groups. RNA summaries persist `normed_tot`, `normed_n`, and `sigmas`;
ATAC summaries persist `prevalence` and `document_frequency`.

| Producer | Artifact inputs | Scientific identity |
|---|---|---|
| assay universe | none | dataset and ordered feature IDs |
| manual selection | `all_features` | supplied values fingerprint |
| RNA or ATAC summary | `cell_selection` | normalizer settings |
| detected features | `feature_summary` | `min_cells` |
| highly variable genes | `feature_summary` | resolved variability settings |
| prevalent peaks | `feature_summary` | `top_n` |
| mapping overlap | mapping reference and query `all_features` | exact inputs |

`select_hvgs`, `select_prevalent_peaks`, `select_detected_features`, and
`set_feature_selection` return exact feature-selection refs. They do not create metadata columns.
`resolve_features` accepts an explicit compatible ref.

## Graph lineage

Graph-derived analyses follow named artifact inputs:

```text
connectivity_map -> neighbors -> coordinates and ann_index
ann_index -> the same coordinates
batch_correction -> reduction -> normalized -> feature_selection
```

Native graphs project one feature selection. Imported-coordinate graphs project none. Integrated
graphs follow their ordered `source_i` refs and can project zero, one, or several distinct
selections. Graph and neighbour consumers require their exact refs. Analytical producers return
artifacts and leave metadata unchanged.

## Inspect and trace results

Use public datastore methods rather than reading private Zarr paths:

```python
refs = ds.list_artifacts(kind="reduction", complete_only=True)
status = ds.inspect_artifact(refs[0])

status.operation
status.parameters
status.inputs
status.execution_options
status.created_at_ns
status.scarf_version
```

`list_artifacts` uses the default assay unless another assay is supplied. Store-level outputs can
be listed with `scope="datastore"`. `load_artifact(ref)` opens the payload only after Scarf confirms
that the artifact exists and is complete.

### Exact provenance filters

`DataStore.list_artifacts` can match the three fields that define artifact provenance:

| Filter | Match rule |
|---|---|
| `operation` | Exact operation name, such as `"run_normalization"` |
| `parameters` | Exact values for each supplied top-level parameter after provenance serialization |
| `inputs` | Exact values for each supplied top-level input after provenance serialization |

Supplied filters are combined with AND. Mapping key order does not matter, and an `ArtifactRef`
matches its serialized `to_dict()` form. Each supplied top-level entry must match exactly, while
the stored provenance may contain other top-level entries. A supplied nested mapping is compared
as a complete value, not as another partial query. Passing `{}` matches an empty mapping; passing
`None` leaves that field unfiltered.

Any `operation`, `parameters`, or `inputs` filter returns only complete artifacts with valid
provenance, even when `complete_only=False`. To search with every top-level provenance entry
available from a known result, inspect it and pass those complete fields back unchanged:

```python
refs = ds.list_artifacts(kind="reduction", complete_only=True)
known = ds.inspect_artifact(refs[0])

matching_refs = ds.list_artifacts(
    kind=known.ref.kind,
    from_assay=known.ref.assay,
    operation=known.operation,
    parameters=known.parameters,
    inputs=known.inputs,
)
```

This still uses containment matching. A stored artifact with the same supplied entries plus
additional top-level entries also matches.

`DataStore.lineage` follows artifact inputs upstream:

```python
lineage = ds.lineage(
    {
        "baselineGraph": baseline_graph,
        "alternativeGraph": alternative_graph,
    }
)

markdown_report = lineage.to_markdown()
mermaid_source = lineage.to_mermaid()
```

This identifies the exact selections, normalization, coordinates, and graph behind a result and
shows where branches diverge.

`ArtifactResolutionError` is a `ValueError` with a machine-readable `code` and JSON-safe
`context`. Failures distinguish missing or incomplete artifacts, wrong kind/scope/assay, changed
row identity or selection values, corrupt payloads, and incompatible artifact contracts. The error
explains what failed; it does not choose a replacement result.

## DataStore summary

`summary()` scans literal live `I` cell and feature columns in blocks and omits store locations and
credentials. It reports artifact inventories, pipeline-run counts by status, and completed labeled
runs. It never selects a pipeline run.
Use `summary.to_dict()` for a deterministic JSON-safe record.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.summary
   scarf.DataStore.list_artifacts
   scarf.DataStore.inspect_artifact
   scarf.DataStore.load_artifact
   scarf.DataStore.lineage
   scarf.DataStore.resolve_features
```

```{eval-rst}
.. automethod:: scarf.DataStore.summary
.. automethod:: scarf.DataStore.list_artifacts
.. automethod:: scarf.DataStore.inspect_artifact
.. automethod:: scarf.DataStore.load_artifact
.. automethod:: scarf.DataStore.lineage
.. automethod:: scarf.DataStore.resolve_features
```

## Module-level helpers

These functions accept a Zarr root group and are useful in tooling. Analysis notebooks should use
the datastore methods above.

```{eval-rst}
.. autofunction:: scarf.storage.list_artifacts
.. autofunction:: scarf.storage.inspect_artifact
```
