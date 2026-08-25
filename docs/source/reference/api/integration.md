# Integration and metrics API reference

Prefer the `DataStore` methods below in analysis code.

`integrate_assays` captures its inputs from each named assay's `AssayState` before planning.
SNN captures each current connectivity map; WNN captures each current neighbor artifact and its exact native reduction or batch-correction coordinates.
The returned integrated-graph reference is the downstream `graph=` argument.
WNN does not accept imported-coordinate ancestry.
`coordinates` is a named provenance input on neighbor and ANN artifacts, not an artifact kind.

Neighbor-based metrics accept `neighbors=`; `None` resolves the assay's current neighbors.
Graph metrics accept `graph=`; `None` resolves the current connectivity map.
Neither interface accepts storage paths or an implicit latest-result selector.

## DataStore methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.integrate_assays
   scarf.DataStore.metric_lisi
   scarf.DataStore.metric_ilisi
   scarf.DataStore.metric_clisi
   scarf.DataStore.metric_proportional_batch_mixing
   scarf.DataStore.metric_graph_connectivity
   scarf.DataStore.metric_graph_silhouette
   scarf.DataStore.metric_label_concordance
   scarf.DataStore.metric_cluster_separability
```

```{eval-rst}
.. automethod:: scarf.DataStore.integrate_assays
.. automethod:: scarf.DataStore.metric_lisi
.. automethod:: scarf.DataStore.metric_ilisi
.. automethod:: scarf.DataStore.metric_clisi
.. automethod:: scarf.DataStore.metric_proportional_batch_mixing
.. automethod:: scarf.DataStore.metric_graph_connectivity
.. automethod:: scarf.DataStore.metric_graph_silhouette
.. automethod:: scarf.DataStore.metric_label_concordance
.. automethod:: scarf.DataStore.metric_cluster_separability
```

## Harmony

```{eval-rst}
.. autofunction:: scarf.embeddings.run_harmony
```

```{eval-rst}
.. autofunction:: scarf.embeddings.fit_harmony
```

```{eval-rst}
.. autoclass:: scarf.embeddings.HarmonyResult
    :members:
```

## Metrics

```{eval-rst}
.. automodule:: scarf.metrics
    :members: compute_lisi, ilisi_knn, clisi_knn, graph_connectivity, silhouette_scoring, label_concordance_score, lisi_batch_mixing_score, ClusterSeparabilityResult
    :imported-members:
```

Scarf's iLISI and cLISI use the scIB median and scaling definitions over Scarf's self-free persisted KNN arrays.
Graph connectivity follows the original scIB symmetrized-graph definition.
YosefLab `scib-metrics` currently uses directed strong components for graph connectivity, so those values need not match.
See Luecken et al. 2022, doi: 10.1038/s41592-021-01336-8.
