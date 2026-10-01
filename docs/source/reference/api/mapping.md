# Mapping API reference

A mapping workflow retains every boundary explicitly:

```python
reference_ref = reference_ds.build_mapping_reference(neighbors_ref)
reference = reference_ds.get_mapping_reference(reference_ref)
query_cells = query_ds.snapshot_cell_selection("I")
result_ref = query_ds.run_mapping(reference, query_cells)
result = query_ds.get_mapping_result(result_ref, reference=reference)
transfer_ref = query_ds.run_label_transfer(
    result_ref,
    reference=reference,
    reference_labels="cell_type",
)
transfer = query_ds.get_label_transfer(transfer_ref)
```

`MappingReference` pins the exact feature selection and reference model. Query overlap is another
feature-selection artifact. `run_mapping` returns only the projection ref; loaders, label transfer,
score readers, and score plots require that ref plus the explicit reference. There is no named
mapping registry, live `cell_key` routing, omitted-result lookup, or reference fallback.

A handle is validated against its store in full when `get_mapping_reference` loads it. Every later
operation that takes the handle checks the stored artifact record and its attributes against the
handle, validates the cell selection against the current ordered cell IDs, and recomputes the digest
of the handle's arrays. Repeated label transfer and score calls
avoid reading the reference model payload, index and neighbours. Each operation also compares the
prepared dataset fingerprint stored on the reference assay with the fingerprint the handle carries;
an unprepared or different reference dataset is rejected. Reading
reference cell metadata also validates the stored selection against the current ordered cell IDs.
A handle whose arrays or references changed is rejected with `does not match its stored artifact`;
reload it with `get_mapping_reference`.

A query projection records the prepared dataset fingerprint of the query assay and reuses a complete
projection only for the same query dataset, cells, overlap, reference, and options. A query cell is
uninformative when its raw counts are zero in every reference feature that the query measured. Such
a cell keeps its projection row but is flagged in `uninformative`; it receives no transferred label,
adds no mapping score, and does not enter Symphony query-batch statistics or
`queryScaledDispersion`. `diagnostics["uninformativeCellCount"]` counts these cells.
`queryScaledDispersion` averages the squared reference-scaled deviation of informative cells over
the measured reference features only, so it does not depend on `missing_feature_policy`.
Projections written before this contract are rejected with an instruction to re-run `run_mapping`.

## Label transfer

`run_label_transfer` saves a label transfer as two query-owned artifacts. It first reads
`reference_labels`, a cell-metadata column of the reference datastore or a cell-label artifact in
it such as `cluster_labels`, `cluster_cut`, or `smart_label`, and freezes those labels into the
query datastore as a datastore-scoped `reference_labels` artifact. That artifact holds the
reference classes and one class code per selected reference cell. Its identity records the mapping
reference, the source column or the source artifact as an `ExternalArtifactRef`, and a fingerprint
of the labels, so unchanged labels reuse one frozen copy and changed labels create another. A
missing, blank, or masked reference label never votes.

It then saves a `label_transfer` artifact in the projection's assay. Its inputs are the projection,
the frozen reference labels, and the projection's query cell selection; its parameters are
`threshold_fraction`, `max_distance`, and the vote algorithm version. Each query cell takes the
label with the largest share of its neighbours' inverse-distance weight. A cell abstains when it is
uninformative, has no labelled neighbour, ties between labels, falls below `threshold_fraction`, or
has its nearest reference neighbour beyond `max_distance`. Abstention is stored as the linked
missing mask of the `labels` array, never as a placeholder string, so a reference class named
`"NA"` stays a class. The artifact also stores the reference classes, each cell's neighbour votes,
and the evidence below, and a payload fingerprint that loading verifies.

A complete transfer with the same projection, frozen labels, and decision rule is reused. A
read-only query datastore raises `PermissionError` before anything is computed or written when no
match exists. `get_label_transfer` reads only the query datastore, so a saved transfer loads
without the reference and never changes when the reference annotations change. Because
`label_transfer` is a cell-label artifact, it can colour an embedding, group statistical tests and
plots, and seed `select_cells` like a clustering; consumers that need a label for every cell reject
it while it contains abstentions. `DataStore.lineage(transfer_ref, references=reference)` follows
the transfer into the reference datastore.

`LabelTransferResult.evidence` has one row per projected query cell, in the order of
`cell_selection`, with these columns:

| Column | Meaning |
|---|---|
| `label` | Transferred label, missing where the cell abstained |
| `candidateLabel` | Label the vote favoured before the threshold and distance rules |
| `voteFraction` | Share of neighbour weight that supports the winning label |
| `topTwoMargin` | Winning share minus the runner-up share |
| `voteEntropy` | Entropy of the vote among labelled neighbours |
| `nearestDistance` | Distance to the nearest reference neighbour |
| `referenceDistancePercentile` | Percentile of that distance among reference nearest-neighbour distances |
| `abstained` | Whether the cell received no label |
| `abstentionReason` | `uninformative_cell`, `no_labeled_neighbors`, `tied_vote`, `below_threshold`, or `beyond_max_distance` |

Vote metrics are NaN for uninformative cells. Projection-level diagnostics such as
`featureCoverage` and `queryScaledDispersion` stay in `get_mapping_result(...).diagnostics`.

`label_vote_shares(labels)` returns each cell's vote share for given labels, and
`prediction_sets(calibration_nonconformity, alpha)` forms split-conformal prediction sets from the
saved votes, calibrated with one minus those shares on held-out cells. Both need the per-neighbour
vote matrices, which hold one column per saved neighbour, so `get_label_transfer` loads them only
with `load_votes=True`. Each array that a load reads is checked against the digest recorded when the
transfer was written.

```{eval-rst}
.. autoclass:: scarf.MappingReference
    :members:
```

```{eval-rst}
.. autoclass:: scarf.MappingResult
    :members:
```

```{eval-rst}
.. autoclass:: scarf.LabelTransferResult
    :members:
```

## DataStore methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.build_mapping_reference
   scarf.DataStore.get_mapping_reference
   scarf.DataStore.run_mapping
   scarf.DataStore.get_mapping_result
   scarf.DataStore.get_mapping_score
   scarf.DataStore.run_label_transfer
   scarf.DataStore.get_label_transfer
```

```{eval-rst}
.. automethod:: scarf.DataStore.build_mapping_reference
.. automethod:: scarf.DataStore.get_mapping_reference
.. automethod:: scarf.DataStore.run_mapping
.. automethod:: scarf.DataStore.get_mapping_result
.. automethod:: scarf.DataStore.get_mapping_score
.. automethod:: scarf.DataStore.run_label_transfer
.. automethod:: scarf.DataStore.get_label_transfer
```

Mapping diagnostics are documented in the {doc}`plotting` API reference.
