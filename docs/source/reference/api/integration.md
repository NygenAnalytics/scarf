# Integration and metrics

Prefer `DataStore.metric_*` in analysis code. The functions below are the underlying
implementations.

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
    :members: compute_lisi, ilisi_knn, clisi_knn, graph_connectivity, silhouette_scoring, label_concordance_score, lisi_batch_mixing_score
    :imported-members:
```

See also `DataStore.metric_lisi`, `metric_ilisi`, `metric_clisi`,
`metric_proportional_batch_mixing`, `metric_graph_connectivity`,
`metric_graph_silhouette`, and `metric_label_concordance` on {doc}`datastore`.

Scarf's iLISI and cLISI use the scIB median and scaling definitions over
Scarf's self-free persisted KNN arrays. Graph connectivity follows the
original scIB symmetrized-graph definition. YosefLab `scib-metrics` currently
uses directed strong components for graph connectivity, so those values need
not match. See Luecken et al. 2022, doi: 10.1038/s41592-021-01336-8.
