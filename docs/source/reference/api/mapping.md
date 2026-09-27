# Mapping API reference

A mapping workflow retains every boundary explicitly:

```python
reference_ref = reference_ds.build_mapping_reference(neighbors_ref)
reference = reference_ds.get_mapping_reference(reference_ref)
query_cells = query_ds.snapshot_cell_selection("I")
result_ref = query_ds.run_mapping(reference, query_cells)
result = query_ds.get_mapping_result(result_ref, reference=reference)
```

`MappingReference` pins the exact feature selection and reference model. Query overlap is another
feature-selection artifact. `run_mapping` returns only the projection ref; loaders, label-transfer
methods, score readers, and plots require that ref plus the explicit reference. There is no named
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

```{eval-rst}
.. autoclass:: scarf.MappingReference
    :members:
```

```{eval-rst}
.. autoclass:: scarf.MappingResult
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
   scarf.DataStore.get_target_classes
   scarf.DataStore.get_target_label_evidence
   scarf.DataStore.calibrate_label_transfer_threshold
```

```{eval-rst}
.. automethod:: scarf.DataStore.build_mapping_reference
.. automethod:: scarf.DataStore.get_mapping_reference
.. automethod:: scarf.DataStore.run_mapping
.. automethod:: scarf.DataStore.get_mapping_result
.. automethod:: scarf.DataStore.get_mapping_score
.. automethod:: scarf.DataStore.get_target_classes
.. automethod:: scarf.DataStore.get_target_label_evidence
.. automethod:: scarf.DataStore.calibrate_label_transfer_threshold
```

Mapping diagnostics are documented in the {doc}`plotting` API reference.
