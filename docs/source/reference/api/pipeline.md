# Analysis pipeline API reference

`DataStore.pipeline` exposes the standard provenance-backed RNA recipe.
The public import path is `scarf.datastore.pipeline_accessor`.

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

### Skip and always-on stages

Skip semantics on {py:meth}`~scarf.datastore.pipeline_accessor.PipelineAccessor.run` are not uniform:

- These stages skip when set to ``False``: `filtering`, `cell_cycle_scoring`, `umap`, `paris`, `doublet_scoring`, and `markers`.
  For those, ``None`` or a dict runs the stage (``None`` uses defaults).
- Always-on stages still run when you pass ``False`` or ``None``: `normalization`, `pca`, `ann_index`, `neighbors`, and `connectivity`.
  Internally ``_options`` maps ``False`` and ``None`` to an empty dict of method kwargs.
- `highly_variable_features` is required.
  Pass options or omit the argument; ``False`` is rejected.
- `harmony` is opt-in: omitted or ``None`` skips it.
  A non-``None`` value must be a dict with a non-empty `batch_columns` list.
  ``False`` is not a skip sentinel.
- `leiden` is ``dict[float, dict] | None``.
  ``None`` uses default resolutions `0.5`, `0.75`, `1.0`, and `1.25`.
  An empty mapping (`{}`) runs no Leiden.
  ``False`` is not a skip sentinel.

### Pipeline-local keys

Step dicts are not always forwarded wholesale to the underlying `DataStore` methods.
Pipeline-local keys are consumed first:

- `filtering`: `method` chooses `auto_filter_cells` vs `filter_cells` (default `"auto"`); remaining keys go to that call.
- `pca`: `n_centroids` and `rand_state` are popped for embedding initialization; remaining keys go to `run_pca`.
- `doublet_scoring` and `markers`: `clusters` is consumed to pick the grouping column; remaining keys go to the detection or marker-search call.

### Other notes

- `clustering_concurrency` defaults to `2`.
  Leiden membership can overlap across workers while store writes stay serialized; Paris joins the same queue.
- Whenever at least one Leiden or Paris partition is available, the one with the highest silhouette score on PCA coordinates is selected.
  Its labels are copied to `{assay}_clusters` and linked to the same artifact, which is also returned as `selected_clusters`.
  A single partition is taken without scoring.
  `doublet_scoring` and `markers` group by that column unless they name a partition through `clusters`.

Only `pipeline_id="basic_rna_analysis"` is currently supported.

HVG selection and marker search read the gene-major `countsT` copy written at RNA import.
An older RNA store that lacks that copy fails when the datastore is opened, before `run` starts.
See {doc}`../../concepts/memory_and_execution`.

## Returned refs

`run` returns `dict[str, ArtifactRef]` keyed by result name (for example `normalized`, `pca`, `connectivity_map`, Leiden keys, and any enabled embedding or cluster outputs).
Capture the dict when you need to inspect or branch from a specific step.

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

## Pipeline callbacks

Callbacks can update an external progress record without changing pipeline control flow:

```python
from scarf.datastore.pipeline_accessor import PipelineEvent

events: list[tuple[str, str]] = []


def record_event(event: PipelineEvent) -> None:
    events.append((event.kind, event.stage))


artifacts = ds.pipeline.run(
    highly_variable_features={
        "min_cells": 20,
        "top_n": 500,
        "show_plot": False,
    },
    callback=record_event,
)
```

Events are serialized on the calling thread.
A skipped stage emits no event, and `stage_completed` means its expected output has finished writing and is available.
Callback exceptions are logged without interrupting the pipeline.
The stable stage-name inventory is part of {py:meth}`~scarf.datastore.pipeline_accessor.PipelineAccessor.run`.

See {ref}`Quick start <quickstart>` for a short executable example and {doc}`graph_construction` for the individual graph-construction methods.
