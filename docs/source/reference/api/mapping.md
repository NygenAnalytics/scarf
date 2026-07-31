# Mapping API reference

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
   scarf.DataStore.project_mapping_layout
   scarf.DataStore.load_unified_graph
   scarf.DataStore.run_unified_umap
   scarf.DataStore.run_unified_tsne
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
.. automethod:: scarf.DataStore.project_mapping_layout
.. automethod:: scarf.DataStore.load_unified_graph
.. automethod:: scarf.DataStore.run_unified_umap
.. automethod:: scarf.DataStore.run_unified_tsne
```

Mapping diagnostics are documented in the {doc}`plotting` API reference.
