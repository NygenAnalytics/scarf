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
   scarf.metadata.CellValues
   scarf.storage.refs.ExternalArtifactRef
   scarf.storage.operation_revisions.OperationRevision
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

.. autoclass:: scarf.metadata.CellValues
    :members:

.. autoclass:: scarf.storage.refs.ExternalArtifactRef
    :members:

.. autoclass:: scarf.storage.operation_revisions.OperationRevision
```

An `ArtifactRef` is a location-free name: a scope, a kind, an assay for assay-scoped artifacts,
and a random 256-bit `artifact_id`. Provenance, pipeline records, and lineage store refs, never
paths or store locations, so the datastore decides which stores resolve a ref. A datastore
resolves its own artifacts; a mounted target resolves its own and then, read only, its source's
(see {ref}`mounted_targets`).

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
status.revision
status.current_revision
status.is_current
status.superseded_by
```

`list_artifacts` uses the default assay unless another assay is supplied. Store-level outputs can
be listed with `scope="datastore"`. `load_artifact(ref)` opens the payload only after Scarf confirms
that the artifact exists and is complete, and `load_cell_values(ref)` reads the per-cell values of a
cell-aligned artifact aligned to their cells ({ref}`cell_aligned_kinds`).

A producer makes its artifact complete with one final write. A write that fails or is interrupted
before that write removes its incomplete artifact. Once the final write is issued, Scarf never
removes the artifact, because the write can persist even when it reports an error. A failure or
interruption during that write can therefore leave an incomplete artifact that
`list_artifacts(complete_only=False)` shows and that nothing reuses.

Results whose values must be finite are checked while they are written: normalized values, reduced
coordinates, loadings, and centers, Harmony corrections and their fitted state, embedding
initializations, embeddings, neighbour distances, and connectivity-map and integrated-graph
weights. A NaN or infinite value, or a finite value too large for the stored dtype, raises
`ValueError` naming the operation, the array, and the first row that holds it; no artifact is
published.

### Operation revisions and stale results

Provenance records an operation revision when it is 2 or more; an artifact without one is revision 1.
A Scarf release that changes what an operation computes adds a revision
({doc}`../../developers/operation_revisions`), and reuse requires the revision that the running
release records, so results of an older revision are computed again. `status.revision` is the
recorded revision, `status.current_revision` the revision that the running release records for the
same provenance, and `status.is_current` whether they agree. `is_current` judges only the
artifact's own revision: an artifact built from a superseded input is current itself.
`status.superseded_by` lists the released revisions after the recorded one that apply to the
artifact, each with the `change` it made. The `scarf_version` attribute names the release that
wrote an artifact for diagnostics and never affects reuse.

A superseded artifact is never reused but stays listable, loadable, and traceable; the one exception
is a marker table of a release before 1.0.0, whose readers refuse it because its `fold_change`
column held sentinels, and name the `run_marker_search` call that recomputes it. When planning finds
one and no exact match, it logs one INFO line that names it and the changes since its revision.
`DataStore.lineage` reports show a superseded artifact with the status `stale`. Results built on
it keep their own current revision, but once it is computed again it has a new reference, so the
results built on it are computed again too. Passing a superseded artifact explicitly as an input
is allowed and records it in lineage as usual.

### Exact provenance filters

`DataStore.list_artifacts` can match the operation, parameters, and inputs that provenance
records. The recorded revision is not a filter, so a listing includes superseded artifacts:

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
shows where branches diverge. Each artifact in the report has one status: `complete`; `stale` when
a later operation revision supersedes it; `incomplete`; `missing`; or `unresolved external` for an
input in another datastore that was not supplied.

An input stored in another datastore, such as the mapping reference behind a query projection, is
recorded as an `ExternalArtifactRef`. It names the artifact's dataset by the prepared dataset
fingerprint of one of its assays; `anchor_assay` names that assay when the artifact is
datastore-scoped or belongs to another assay. The fingerprint names a dataset, not a store, and a
mount shares its source's. Pass the mapping references to
`DataStore.lineage(target, references=...)` to follow such inputs into their datastores; without
them, an external input is shown as unresolved. A mount records the source artifacts it uses as
plain refs, because it resolves them itself.

`ArtifactResolutionError` is a `ValueError` with a machine-readable `code` and JSON-safe
`context`. Failures distinguish missing or incomplete artifacts, wrong kind/scope/assay, changed
row identity or selection values, corrupt payloads, and incompatible artifact contracts. The error
explains what failed; it does not choose a replacement result.

(cell_aligned_kinds)=
### Read per-cell values

`load_cell_values(ref)` reads the per-cell values of a cell-aligned artifact and returns a
{py:class}`scarf.metadata.CellValues` aligned to the `ids` of its cells. A cell-aligned artifact
holds a row for each cell of the cell selection that its provenance records as its `cell_selection`
input. The table lists these kinds, the canonical array that `load_cell_values` reads by default,
whether that array holds labels that group cells or measurements, and the other per-cell arrays
that `value=` may name. Every other kind is refused, among them cell selections, reference labels,
graphs, and per-feature results. Reductions and Harmony corrections hold a row per cell too, but
their rows follow the cell selection of their lineage rather than a recorded input; open them with
`load_artifact`.

| Kind | Canonical array | Holds | Other label arrays | Other measurement arrays |
|---|---|---|---|---|
| `cell_cycle` | `phase` | labels | | `s_score`, `g2m_score` |
| `cluster_cut` | `labels` | labels | | |
| `cluster_labels` | `values` | labels | | |
| `doublet_score` | `values` | measurements | | |
| `embedding` | `values` | measurements | | |
| `enrichment_scores` | `scores` | measurements | | |
| `fate_map` | `probabilities` | measurements | `valid` | |
| `hto_identity` | `values` | labels | | |
| `imported_coordinates` | `data` | measurements | | |
| `label_transfer` | `labels` | labels | `abstention_reason`, `candidate_codes`, `vote_class_codes` | `vote_class_fractions`, `vote_fraction`, `top_two_margin`, `vote_entropy`, `nearest_distance`, `reference_distance_percentile` |
| `membership_strength` | `values` | measurements | | |
| `metadata_snapshot` | `values` | measurements | | |
| `pseudotime` | `pseudotime` | measurements | `valid` | |
| `quality_metric` | `values` | measurements | | |
| `sampling` | `sampled` | labels | `seeds` | `density`, `mean_snn` |
| `smart_label` | `values` | labels | | |

A metadata snapshot is cell-aligned when it records a cell selection, as the custom source and sink
vector of pseudotime scoring does. The metadata snapshot of a pipeline run records none: it holds
whole metadata columns, one row for each row of the cell or feature table, and is valid but not
cell-aligned. `load_cell_values`, plot groupings and colors, and the other readers refuse it with a
`ValueError` that says its rows are not aligned to a recorded cell selection and points to the
run's frozen fields, such as `run.cells.fetch(column)`. Only a snapshot whose recorded cell
selection is malformed raises `ArtifactResolutionError` with code `corrupt_payload`.

The canonical array of each kind is part of the identity of the results that read it. Consumers
such as `select_cells` record the artifact that they read but not which of its arrays, so changing
a kind's canonical array in this table would change what they compute from the same recorded
inputs. Such a change needs an operation revision for each consumer
({doc}`../../developers/operation_revisions`).

`cell_selection=` reads the cells of a selection that the artifact's own selection contains; any
other selection raises `ValueError`. Values come back in cell table order. A row that the artifact
records as missing keeps its stored placeholder and is flagged in `missing`. `to_pandas()` returns
the values indexed by cell id, as a Series, or as a DataFrame with a column for each position of a
row, and shows missing rows as missing. Before it reads anything, the read is charged against the
datastore memory budget, and it raises `MemoryError` when it does not fit: the values and their
missing mask, the cell ids, and the int64 cell rows and positions that it builds, each read chunk
by chunk.

```python
phase = ds.load_cell_values(cell_cycle_ref)  # canonical "phase" labels
s_score = ds.load_cell_values(cell_cycle_ref, value="s_score")
phase.to_pandas()  # Series indexed by cell id
coordinates = ds.load_cell_values(umap_ref).to_pandas()  # one column per dimension
subset = ds.load_cell_values(clusters_ref, cell_selection=t_cells)
```

`select_cells`, artifact groupings and colors in plots, and pseudotime source and sink labels read
the same canonical arrays, and every reader refuses a kind that the table does not list, whichever
array it is asked for. Label consumers, such as `snapshot_cluster_labels`, `make_bulk`, and
`smart_label`, accept the kinds whose canonical array holds labels.

```{eval-rst}
.. autodata:: scarf.metadata.selection.CELL_VALUE_NAMES
    :annotation:
```

## DataStore summary

`summary()` scans literal live `I` cell and feature columns in blocks and omits store locations and
credentials. It reports artifact inventories, pipeline-run counts by status, and completed labeled
runs. It never selects a pipeline run. Each assay's `assay_type` is the type that the open resolved,
`Assay.assayType` ({ref}`assay_types_attribute`), so a read-only open, which cannot record a type,
reports the type of a preset-named assay that the store's `assayTypes` record lacks.
Use `summary.to_dict()` for a deterministic JSON-safe record.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.summary
   scarf.DataStore.list_artifacts
   scarf.DataStore.inspect_artifact
   scarf.DataStore.load_artifact
   scarf.DataStore.load_cell_values
   scarf.DataStore.lineage
   scarf.DataStore.resolve_features
```

```{eval-rst}
.. automethod:: scarf.DataStore.summary
.. automethod:: scarf.DataStore.list_artifacts
.. automethod:: scarf.DataStore.inspect_artifact
.. automethod:: scarf.DataStore.load_artifact
.. automethod:: scarf.DataStore.load_cell_values
.. automethod:: scarf.DataStore.lineage
.. automethod:: scarf.DataStore.resolve_features
```

## Module-level helpers

These functions accept a Zarr root group and are useful in tooling. Analysis notebooks should use
the datastore methods above. The functions inspect exactly the group they are given: the `zw`
group of a mounted datastore also resolves its source's artifacts, while a target group opened
directly with `zarr.open_group` holds only the target's own.

```{eval-rst}
.. autofunction:: scarf.storage.list_artifacts
.. autofunction:: scarf.storage.inspect_artifact
```
