# DataStore API reference

`DataStore` is the primary analyst-facing object.
It inherits graph, mapping, and assay helpers from the classes below.
Use this page for analyst-facing methods and consult the inheritance appendix when extending Scarf.

Graph-construction methods are documented on {doc}`graph_construction`.
Read-only state inspection, including the metadata-only `DataStore.summary()`, is documented on {doc}`artifacts`.
Mapping methods are on {doc}`mapping`, and integration metrics are on {doc}`integration`.
Those methods are excluded here rather than repeated.

`summary` is reserved for `DataStore.summary()` and cannot be used as an assay name.
Writers and `DataStore` opening reject that name before mutating store-level state.

## Mounting shared count matrices

Use `mount_datastore` when count matrices stay in a read-only source store and analysis artifacts should write to a separate target.
See {doc}`../../tutorials/remote_stores`.

```{eval-rst}
.. autofunction:: scarf.mount_datastore
```

```{eval-rst}
.. autoclass:: scarf.datastore.datastore.DataStore
    :members:
    :inherited-members:
    :exclude-members: run_normalization, run_pca, run_lsi, run_custom_reduction,
        run_harmony, build_embedding_initialization, build_ann_index, query_neighbors,
        build_connectivity_map, load_graph, list_artifacts, inspect_artifact, load_artifact,
        lineage, summary, resolve_features,
        get_assay_state, build_mapping_reference, get_mapping_reference, run_mapping,
        get_mapping_result, get_mapping_score, get_target_classes,
        get_target_label_evidence, calibrate_label_transfer_threshold,
        integrate_assays, metric_lisi, metric_ilisi, metric_clisi,
        metric_proportional_batch_mixing, metric_graph_connectivity,
        metric_graph_silhouette, metric_label_concordance,
        metric_cluster_separability
```

## Store-bound plotting

`DataStore.plots` binds the datastore argument for canonical store-first functions in `scarf.plotting`.
Array and DataFrame diagnostics such as `elbow`, `qc`, `graph_qc`, and `highly_variable_features` remain standalone.

```{eval-rst}
.. autoclass:: scarf.datastore.plot_accessor.DataStorePlotAccessor
    :members:
```

## Selected analysis contracts

Feature producers return immutable {py:class}`~scarf.ArtifactRef` values and publish plain labels.
Use `resolve_features(assay, label_or_ref)` for strict read-only resolution.
`run_normalization` requires keyword-only `features=`, and direct marker, WAGGR, AUCell, and pseudotime feature analyses likewise require an exact label or reference.
Use `all_features` explicitly when the complete assay feature universe is intended.
Graph-derived methods instead accept `graph=` and project feature selections through the graph's named lineage edges.
They do not accept a separate feature selection.

```python
normalized = ds.run_normalization(features=hvg_ref)
imputed = ds.get_imputed("CD4", graph=graph_ref)
ds.calc_membership_strength("clusters", graph=graph_ref)
ds.run_doublet_detection("clusters", graph=graph_ref)
```

`features` is keyword-only for normalization.
`get_imputed` starts with the feature name, while membership strength and doublet detection start with the clustering key; `graph` is an explicit keyword on those graph consumers.

{py:meth}`scarf.datastore.datastore.DataStore.auto_filter_cells` uses pooled Gaussian bounds by default.
Supplying `sample_column` selects per-sample MAD bounds with `n_mads` and `min_cells_per_sample`; `min_p` and `max_p` do not configure that path.

Saved {py:meth}`scarf.datastore.datastore.DataStore.run_marker_search` calls return the exact immutable marker-table reference.
Pass that reference as `get_markers(marker=ref)` to select the exact feature-specific result.
The marker index retains one entry per cell key, grouping column, and feature-selection artifact; an unqualified lookup fails when more than one such result exists.
Fresh marker results include score, expression fractions, fold change, AUC, two-sided Mann-Whitney p-values, and Benjamini-Hochberg values adjusted within each one-versus-rest group over tested features.
These are cell-level marker statistics, not replicate-aware differential expression.

{py:meth}`scarf.datastore.datastore.DataStore.run_pseudotime_marker_search` leaves untested features with `r_value` 0.0 and `NaN` for `p_value` and `p_value_adjusted`, and adjusts p-values over tested features only.

{py:meth}`~scarf.DataStore.integrate_assays` is the public SNN/WNN graph integration entry point.
WNN requires two or more cell-aligned assays and stores one per-cell weight for each assay.
SNN via `merge_graphs` requires one or more graphs.
The recommended workflow is in {doc}`../../tutorials/cite_seq`; method comparison and diagnostics are in {doc}`../../tutorials/multimodal_diagnostics`.

## Analysis results

```{eval-rst}
.. autoclass:: scarf.FateMappingResult
    :members:

.. autoclass:: scarf.PseudotimeScoreResult
    :members:

.. autoclass:: scarf.PseudotimeMarkerResult
    :members:

.. autoclass:: scarf.PseudotimeAggregationResult
    :members:

.. autoclass:: scarf.EnrichmentResult
    :members:

.. autoclass:: scarf.clustering.ParisClusteringResult
    :members:

.. autoclass:: scarf.clustering.ParisClusterDiagnostic
    :members:
```

## Inheritance appendix

These implementation classes are not intended for analysts to construct directly.

```{eval-rst}
.. autoclass:: scarf.datastore.base_datastore.BaseDataStore
    :members:
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.graph_datastore.GraphDataStore
    :members:
    :inherited-members:
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.mapping_datastore.MappingDatastore
    :members:
    :inherited-members:
    :no-index:
```
