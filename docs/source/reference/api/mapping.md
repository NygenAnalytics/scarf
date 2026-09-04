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
