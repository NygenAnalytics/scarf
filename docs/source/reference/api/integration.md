# Integration and metrics

Prefer `DataStore.metric_*` in analysis code. The functions below are the underlying
implementations.

## Harmony

```{eval-rst}
.. autofunction:: scarf.harmony.run_harmony
```

```{eval-rst}
.. autofunction:: scarf.harmony.fit_harmony
```

```{eval-rst}
.. autoclass:: scarf.harmony.HarmonyResult
    :members:
```

## Metrics

```{eval-rst}
.. automodule:: scarf.metrics
    :members: compute_lisi, silhouette_scoring, label_concordance_score, lisi_batch_mixing_score, integration_score
```

See also `DataStore.metric_lisi`, `metric_silhouette`, `metric_label_concordance`,
`metric_batch_mixing`, and `metric_integration` on {doc}`datastore`.
