# DataStore API reference

`DataStore` is the primary analyst-facing object. It inherits graph, mapping, and assay helpers
from the classes below. Use this page for analyst-facing methods and consult the inheritance
appendix when extending Scarf.

Graph-construction methods are documented on {doc}`graph_construction`. Read-only state inspection,
including the metadata-only `DataStore.summary()`, is documented on {doc}`artifacts`. Mapping
methods are on {doc}`mapping`, and integration metrics are on {doc}`integration`. Those methods are
excluded here rather than repeated.

`summary` is reserved for `DataStore.summary()` and cannot be used as an assay name. Writers and
`DataStore` opening reject that name before mutating store-level state.

## Mounting shared count matrices

Use `mount_datastore` when count matrices stay in a read-only source store and analysis
artifacts should write to a separate target. See {doc}`../../tutorials/remote_stores`.

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
        lineage, summary,
        get_assay_state, build_mapping_reference, get_mapping_reference, run_mapping,
        get_mapping_result, get_mapping_score, get_target_classes,
        get_target_label_evidence, calibrate_label_transfer_threshold,
        integrate_assays, metric_lisi, metric_ilisi, metric_clisi,
        metric_proportional_batch_mixing, metric_graph_connectivity,
        metric_graph_silhouette, metric_label_concordance,
        metric_cluster_separability
```

## Store-bound plotting

`DataStore.plots` binds the datastore argument for canonical store-first functions in
`scarf.plotting`. Array and DataFrame diagnostics such as `elbow`, `qc`, `graph_qc`, and
`highly_variable_features` remain standalone.

```{eval-rst}
.. autoclass:: scarf.datastore.plot_accessor.DataStorePlotAccessor
    :members:
```

## Selected analysis contracts

{py:meth}`scarf.datastore.datastore.DataStore.auto_filter_cells` uses pooled Gaussian bounds by
default. Supplying `sample_column` selects per-sample MAD bounds with `n_mads` and
`min_cells_per_sample`; `min_p` and `max_p` do not configure that path.

Fresh {py:meth}`scarf.datastore.datastore.DataStore.run_marker_search` results include score,
expression fractions, fold change, AUC, two-sided Mann-Whitney p-values, and Benjamini-Hochberg
values adjusted within each one-versus-rest group over tested features. These are cell-level marker
statistics, not replicate-aware differential expression.

{py:meth}`scarf.datastore.datastore.DataStore.run_pseudotime_marker_search` stores `NaN` for
features that were not tested and adjusts p-values over tested features only.

{py:meth}`~scarf.DataStore.integrate_assays` is the public SNN/WNN graph integration entry point.
The recommended workflow is in {doc}`../../tutorials/cite_seq`; method comparison and diagnostics
are in {doc}`../../tutorials/multimodal_integration`.

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
