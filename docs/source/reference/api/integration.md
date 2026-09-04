# Integration and metrics API reference

Prefer the `DataStore` methods below in analysis code.

`integrate_assays` accepts two or more exact source refs. WNN is the default; it consumes neighbour
artifacts and follows each one's named native reduction or batch-correction coordinates. SNN is
explicit and consumes connectivity-map artifacts. The returned integrated-graph reference is the
exact downstream `graph` argument.
WNN does not accept imported-coordinate ancestry. `coordinates` is a provenance input on neighbour
and ANN artifacts, not a separate artifact kind.

```python
wnn = ds.integrate_assays([rna_neighbors, adt_neighbors])
snn = ds.integrate_assays([rna_graph, adt_graph], method="snn")
```

The default WNN artifact stores one per-cell weight for each input assay. Plot those values with
{py:func}`scarf.plotting.modality_weights` or the bound
`ds.plots.modality_weights(graph=wnn, layout=wnn_layout)` accessor. An explicit SNN artifact does
not contain modality weights.

Neighbour-based metrics require `neighbors`; graph metrics require `graph`; reduction metrics
require their exact coordinate artifact. Batch-mixing methods and biological-annotation cLISI or
graph connectivity intentionally read imported metadata columns over the rows selected by artifact
lineage. They are not an indirect path for scoring Scarf-produced clusterings. Concordance instead
accepts two exact clustering refs with the same frozen cell selection. None accepts a storage path
or an omitted artifact input.

```python
mixing_ref = ds.metric_lisi(["batch"], rna_neighbors)
mixing = ds.load_metric_lisi(mixing_ref)["batch"]
connectivity = ds.metric_graph_connectivity("cell_type", snn)
agreement = ds.metric_label_concordance(rna_clusters, adt_clusters, metric="ari")
```

Per-cell LISI is axis-aligned analytical data, so `metric_lisi` returns an artifact and
`load_metric_lisi` reads its scores. Dataset-level scalar summaries such as iLISI and graph
connectivity remain direct values.

## DataStore methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.integrate_assays
   scarf.DataStore.metric_lisi
   scarf.DataStore.load_metric_lisi
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
.. automethod:: scarf.DataStore.load_metric_lisi
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
