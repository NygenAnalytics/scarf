# Analysis pipeline

`DataStore.pipeline` exposes the standard provenance-backed RNA recipe. The public
import path is `scarf.datastore.pipeline_accessor`.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.datastore.pipeline_accessor.PipelineAccessor
   scarf.datastore.pipeline_accessor.PipelineAccessor.run
```

## PipelineAccessor

```{eval-rst}
.. autoclass:: scarf.datastore.pipeline_accessor.PipelineAccessor
    :members: run
```

## Step options

Most step parameters on {py:meth}`~scarf.datastore.pipeline_accessor.PipelineAccessor.run`
accept:

- `None`: run the step with defaults
- `False`: skip the step
- `dict`: forward keyword arguments to the underlying `DataStore` method

Exceptions:

- `harmony` is skipped when omitted (`None`). Pass a dict that includes
  `batch_columns` to enable Harmony.
- `leiden` defaults to one run at resolution `1.0`. Pass an empty mapping to run no
  Leiden clustering, or a mapping such as `{0.5: {}}` for other resolutions.

Only `pipeline_id="basic_rna_analysis"` is currently supported.

## Returned refs

`run` returns `dict[str, ArtifactRef]` keyed by result name (for example
`normalized`, `reduction`, `connectivity_map`, and any enabled embedding or cluster
outputs). Capture the dict when you need to inspect or branch from a specific step.

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
