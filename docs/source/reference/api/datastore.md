# DataStore API reference

`DataStore` is the primary analyst-facing object.
It inherits graph, mapping, and assay helpers from the classes below.
Use this page for analyst-facing methods and consult the inheritance appendix when extending Scarf.

Graph-construction methods are documented on {doc}`graph_construction`.
Artifact and run inspection, including the metadata-only `DataStore.summary()`, is documented on {doc}`artifacts`.
Mapping methods are on {doc}`mapping`, and integration metrics are on {doc}`integration`.
Those methods are excluded here rather than repeated.

`summary` is reserved for `DataStore.summary()` and cannot be used as an assay name.
Writers and `DataStore` opening reject that name before mutating store-level state.
Opening also rejects the removed `{assay}/state` group. Rebuild such a store with the current
release; Scarf does not read or migrate its former analysis state.

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
        lineage, summary, resolve_features, snapshot_cell_selection,
        build_mapping_reference, get_mapping_reference, run_mapping,
        get_mapping_result, get_mapping_score, get_target_classes,
        get_target_label_evidence, calibrate_label_transfer_threshold,
        integrate_assays, metric_lisi, load_metric_lisi, metric_ilisi, metric_clisi,
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

Feature producers return immutable {py:class}`~scarf.ArtifactRef` values. Use
`resolve_features(assay, ref)` for strict read-only validation.
`select_all_features(from_assay=...)` creates or reuses the canonical immutable all-feature
selection for granular workflows. It does not write a live feature metadata column.
`snapshot_cell_selection(cell_key)` captures the explicit immutable cell input for granular graph
construction. `run_normalization(cell_selection, features)` requires both exact refs. Direct marker,
WAGGR, AUCell, and pseudotime feature analyses likewise require exact refs.
Graph-derived methods require a graph ref and project feature selections through its named lineage edges.
They do not accept a separate feature selection.

```python
cells = ds.snapshot_cell_selection("I")
features = ds.select_all_features(from_assay="RNA")
normalized = ds.run_normalization(cells, features)
diffusion = ds.run_diffusion_operator(graph_ref, t=2)
imputed = ds.get_imputed("CD4", diffusion)
membership = ds.calc_membership_strength(cluster_ref, graph_ref)
doublets = ds.run_doublet_detection(cluster_ref, graph_ref)
```

`get_imputed` starts with the feature name and aligns its returned array to the cell selection in
the explicit diffusion-operator lineage. `load_diffusion_operator(ref)` exposes the validated sparse
operator for direct matrix work. Membership strength and doublet detection consume exact cluster
and graph refs and return artifacts.

`make_bulk(groups, ...)` accepts a categorical cell artifact or a user-owned metadata column name.
Artifact inputs derive their cell selection from lineage; `cell_selection=` can restrict that
selection explicitly. `secondary_groups=` provides an optional nested grouping without writing
artifact labels to a cell column. `add_grouped_assay(groups, assay_label=...)` similarly accepts a
pseudotime-aggregation ref or an explicit feature metadata column when constructing a new assay.

{py:meth}`scarf.datastore.datastore.DataStore.auto_filter_cells` uses pooled Gaussian bounds by default.
Supplying `sample_column` selects per-sample MAD bounds with `n_mads` and `min_cells_per_sample`; `min_p` and `max_p` do not configure that path.
{py:meth}`scarf.datastore.datastore.DataStore.select_cells` thresholds the numeric `values` payload
of an exact cell artifact, or retains categorical values with `include=[...]`, and composes the
result with its stored source selection. An explicit `cell_selection=` may narrow, but never widen,
that source selection.

Saved {py:meth}`scarf.datastore.datastore.DataStore.run_marker_search` calls return the exact immutable marker-table reference.
Pass that reference as `get_markers(marker=ref)` to select the exact feature-specific result.
Fresh marker results include score, expression fractions, fold change, AUC, two-sided Mann-Whitney p-values, and Benjamini-Hochberg values adjusted within each one-versus-rest group over tested features.
These are cell-level marker statistics, not replicate-aware differential expression.

{py:meth}`scarf.datastore.datastore.DataStore.run_pseudotime_marker_search` leaves untested features with `r_value` 0.0 and `NaN` for `p_value` and `p_value_adjusted`, and adjusts p-values over tested features only.

{py:meth}`~scarf.DataStore.integrate_assays` is the public SNN/WNN graph integration entry point.
WNN requires two or more cell-aligned assays and stores one per-cell weight for each assay.
SNN also requires two or more explicit connectivity-map refs.
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

.. autoclass:: scarf.features.statistical.StatisticalTestResult
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
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.graph_datastore.GraphDataStore
    :no-index:
```

```{eval-rst}
.. autoclass:: scarf.datastore.mapping_datastore.MappingDatastore
    :no-index:
```
