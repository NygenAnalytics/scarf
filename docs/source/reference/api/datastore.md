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
the explicit diffusion-operator lineage. It accepts one name, a sequence of names, a
one-dimensional string array, or a Series. A name that exactly matches a live cell metadata
column, including case, diffuses that column; it takes precedence over an assay feature of the same
name. Other names match assay feature names case-insensitively, and duplicates are averaged.
`run_diffusion_operator` raises `MemoryError` before persisting an operator that could not be
formed or loaded again within the datastore memory budget. `load_diffusion_operator(ref)` exposes
the validated sparse operator for direct matrix work. Membership strength and doublet detection consume exact cluster
and graph refs and return artifacts.

`make_bulk(groups, ...)` accepts a categorical cell artifact or a user-owned metadata column name.
Artifact inputs derive their cell selection from lineage; `cell_selection=` can restrict that
selection explicitly. A metadata column without `cell_selection=` snapshots the live `I` column.
`secondary_groups=` provides an optional nested grouping without writing artifact labels to a cell
column. Mean profiles fit the assay normalization once over every selected cell, so ATAC document
frequency and ADT CLR geometric means are shared by all groups. Group values that would produce the
same column name raise an error. `add_grouped_assay(groups, assay_label=...)` similarly accepts a
pseudotime-aggregation ref or an explicit feature metadata column when constructing a new assay.
An aggregation ref groups only the features it clustered, and missing metadata values never form a
group.

`add_grouped_assay` and `add_melded_assay` make the new assay visible only after its counts are
complete. A failed or interrupted write removes the partial assay. A process killed during the
write leaves a pending assay that opening ignores, that repacking refuses, and that blocks the
name until `discard_interrupted_assay(assay_label)` removes it.

{py:meth}`scarf.datastore.datastore.DataStore.auto_filter_cells` defaults to `method="mad"`
over the pooled selection. Supplying `sample_column` estimates MAD bounds separately per sample.
`n_mads` controls the bounds; groups with fewer than `min_cells_per_sample=20` active cells
are retained with a warning, including small pooled selections. Pass `method="gaussian"`
explicitly to use the former pooled Gaussian policy and configure its `min_p` and `max_p`
quantiles. Changing these probabilities with either MAD path raises an error.
`filter_cells` and `auto_filter_cells` apply the same rules as pipeline filtering. A cell whose
metric is recorded as missing in a nullable column or artifact never passes a filter and does not
inform automatic bounds. MAD filtering rejects missing sample labels among active cells. Automatic
bounds reject non-finite metric values, such as the undefined percentages of zero-count cells, and
every filter raises when no cell remains.
{py:meth}`scarf.datastore.datastore.DataStore.select_cells` thresholds the numeric `values` payload
of an exact cell artifact, or retains categorical values with `include=[...]`, and composes the
result with its stored source selection. An explicit `cell_selection=` may narrow, but never widen,
that source selection. Cells whose value the artifact records as missing are never selected, and
doublet detection rejects a clustering with missing labels.

Saved {py:meth}`scarf.datastore.datastore.DataStore.run_marker_search` calls return the exact immutable marker-table reference.
Pass that reference as `get_markers(marker=ref)` to select the exact feature-specific result.
Marker tables report `group_id` as a string and list groups in the same order as plot
categories: numeric labels first in numeric order, so `"2"` precedes `"10"`, then text labels.
`export_markers_to_csv` uses the same column order. An unknown `group_id` raises an error.
Fresh marker results include score, expression fractions, fold change, AUC, two-sided Mann-Whitney p-values, and Benjamini-Hochberg values adjusted within each one-versus-rest group over tested features.
These are cell-level marker statistics, not replicate-aware differential expression.

{py:meth}`scarf.datastore.datastore.DataStore.run_pseudotime_marker_search` leaves untested features with `r_value` 0.0 and `NaN` for `p_value` and `p_value_adjusted`, and adjusts p-values over tested features only.

{py:meth}`scarf.datastore.datastore.DataStore.run_statistical_testing` computes Mann-Whitney
p-values from the exact permutation null, ties included, when the two groups can be formed in at
most 100,000 ways, as in typical sample-level designs. Larger designs keep the marker-search normal
approximation. `StatisticalTestResult.p_value_method` records `"exact"` or `"asymptotic"`, and a
saved Mann-Whitney result without that record must be recomputed. A `StudyDesign` pairing column
applies only to the paired Wilcoxon test; for subjects nested within conditions, pass the subject
column as `sample_by`. Writing a result requires `zarr_mode='r+'` and is checked before any value
is computed, while a matching saved result is still reused from a read-only store.

{py:meth}`scarf.datastore.datastore.DataStore.run_waggr` and
{py:meth}`scarf.datastore.datastore.DataStore.run_aucell` match network targets to active feature
names without case sensitivity. A target that matches several active features, such as a gene
symbol shared by two feature ids, is ambiguous. The default `ambiguous_targets="drop"` removes its
edges before `tmin` pruning, logs a warning, and records the target in the artifact's
`dropped_ambiguous_targets` attribute; `ambiguous_targets="error"` raises instead. Both producers
require `zarr_mode='r+'` and an intact prepared dataset identity.

{py:meth}`scarf.datastore.datastore.DataStore.run_pseudotime_aggregation` aggregates and clusters
features before it starts an artifact. It raises `ValueError`, and leaves no incomplete artifact,
when too few features pass `min_exp` or when `n_clusters` is greater than one but smaller than the
number of disconnected components of the feature neighbour graph. Modules tied at the cut height are merged so that exactly
`n_clusters` modules are returned.
{py:meth}`scarf.datastore.datastore.DataStore.run_fate_mapping` treats `solver_tol` as an absolute
bound on the largest Bellman residual of each solved sink column, so large sink groups do not
loosen the solve.

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
