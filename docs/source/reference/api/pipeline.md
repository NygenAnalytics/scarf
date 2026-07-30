# Analysis pipeline

`DataStore.pipeline` exposes the standard provenance-backed RNA recipe. The public
import path is `scarf.datastore.pipeline_accessor`.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.datastore.pipeline_accessor.PipelineAccessor
   scarf.datastore.pipeline_accessor.PipelineEvent
   scarf.datastore.pipeline_accessor.PipelineAccessor.run
```

## PipelineAccessor

```{eval-rst}
.. autoclass:: scarf.datastore.pipeline_accessor.PipelineAccessor
    :members: run

.. autoclass:: scarf.datastore.pipeline_accessor.PipelineEvent
    :members:
```

## Step options

Most step parameters on {py:meth}`~scarf.datastore.pipeline_accessor.PipelineAccessor.run`
accept:

- `None`: run the step with defaults
- `False`: skip the step
- `dict`: forward keyword arguments to the underlying `DataStore` method

Exceptions:

- `highly_variable_features` is required. Pass options or omit the argument;
  ``False`` is rejected.
- `harmony` is skipped when omitted (`None`). Pass a dict that includes
  `batch_columns` to enable Harmony.
- `leiden` defaults to resolutions `0.5`, `0.75`, `1.0`, and `1.25`. Pass an
  empty mapping to run no Leiden clustering.
- `clustering_concurrency` defaults to `2`. Leiden membership can overlap across
  workers while store publishes stay serialized; Paris joins the same queue.
- When `doublet_scoring` or `markers` omit `clusters`, the Leiden or Paris
  partition with the highest silhouette score on PCA coordinates is selected
  (returned as `selected_clusters`).

Only `pipeline_id="basic_rna_analysis"` is currently supported.

## Returned refs

`run` returns `dict[str, ArtifactRef]` keyed by result name (for example
`normalized`, `pca`, `connectivity_map`, Leiden keys, and any enabled embedding
or cluster outputs). Capture the dict when you need to inspect or branch from a
specific step.

```python
artifacts = ds.pipeline.run(
    pipeline_id="basic_rna_analysis",
    filtering={
        "method": "manual",
        "attrs": ["RNA_nCounts", "RNA_nFeatures"],
        "highs": [15000, 4000],
        "lows": [1000, 500],
    },
    cell_cycle_scoring=False,
    highly_variable_features={"min_cells": 20, "top_n": 500, "show_plot": False},
    pca={"dims": 15, "n_centroids": 100},
    neighbors={"k": 11},
    umap={"n_epochs": 100, "parallel": True},
    leiden={0.5: {}},
    paris=False,
    doublet_scoring=False,
    markers=False,
)
```

See {ref}`Quick start <quickstart>` for a short executable example and
{doc}`graph_ops` for the underlying atomic methods.