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
    :members: compute_lisi, silhouette_scoring, label_concordance_score, lisi_batch_mixing_score, integration_score
    :imported-members:
```

See also `DataStore.metric_lisi`, `metric_silhouette`, `metric_label_concordance`,
and `metric_batch_mixing` on {doc}`datastore`. `metric_integration` is a
deprecated compatibility name for `metric_label_concordance` and may be
removed in Scarf 2.0.
