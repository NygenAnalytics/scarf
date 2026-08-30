# Analysis pipeline API reference

`DataStore.pipeline` runs Scarf's standard RNA recipe and records every invocation in the datastore.
The result is a durable, read-only {py:class}`~scarf.PipelineRun` whose outputs are exact immutable
{py:class}`~scarf.ArtifactRef` values.

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.PipelineRun
   scarf.PipelineExecutionError
   scarf.datastore.pipeline_accessor.PipelineAccessor
   scarf.datastore.pipeline_accessor.PipelineEvent
   scarf.datastore.pipeline_accessor.PipelineAccessor.run
   scarf.datastore.pipeline_accessor.PipelineAccessor.open
   scarf.datastore.pipeline_accessor.PipelineAccessor.list_runs
   scarf.datastore.pipeline_accessor.PipelineAccessor.abandon_label_claim
```

## Run the RNA recipe

```python
run = ds.pipeline.run(assay="RNA", label="baseline")

run["pca"]
run["cluster_selection"]
run["clusters"]
run.cells.to_pandas_dataframe(["umap_1", "umap_2", "clusters"])
ds.plots.embedding(run=run, layout="umap", color_by="clusters")
ds.get_markers(marker=run["markers"], group_id=0)
```

The public signature is:

```python
def run(
    *,
    assay: str | None = None,
    label: str | None = None,
    cell_key: str = "I",
    filtering: bool | Mapping[str, object] = True,
    harmony_batch_columns: Sequence[str] | None = None,
    hvg_count: int = 1000,
    pca_dims: int = 21,
    neighbors_k: int = 11,
    umap: bool = True,
    leiden: Mapping[str, object] | bool = True,
    cell_cycle: bool = True,
    paris: bool = True,
    doublets: bool = True,
    markers: bool = True,
    snapshot_columns: Sequence[str] = (),
    callback: PipelineCallback | None = None,
) -> PipelineRun: ...
```

The default recipe snapshots its inputs, filters cells, scores cell cycle, selects highly variable
genes, normalizes, runs PCA, builds ANN, neighbour, and connectivity artifacts, initializes UMAP,
runs UMAP, evaluates Leiden at `0.5`, `0.75`, `1.0`, and `1.25`, runs Paris, selects a clustering,
scores doublets, and searches for markers. Atomic stages run sequentially so each stage has an
honest memory and cache receipt.

Harmony is enabled by a non-empty `harmony_batch_columns` sequence. The main graph then uses the
Harmony coordinates. When doublet scoring is enabled, its graph branch still uses the uncorrected
PCA coordinates. The branch artifacts appear in their stage receipt, not as convenience entries in
the run's top-level output mapping.

Boolean stage options disable that stage when set to `False`. `True` uses that stage's defaults.
A mapping configures the stage. `None` is rejected. An empty filtering mapping uses the same
automatic defaults as `filtering=True`. An empty Leiden mapping is invalid; a custom Leiden
request contains exactly `partitions`:

```python
run = ds.pipeline.run(leiden={"partitions": [0.4, 0.8]})
```

Doublets and markers require at least one Leiden candidate. Paris can still run as a diagnostic
output when those stages are disabled; a Paris-only run has `run["paris"]` and no
`run["clusters"]`. Setting `umap=False` skips both embedding initialization and UMAP, so neither
artifact appears in the completed run.

`filtering=True` uses automatic filtering over available assay QC columns. Set it to `False` to
retain the captured input selection, or pass a configuration mapping. Automatic filtering accepts
`attrs`, `min_p`, `max_p`, and optionally `sample_column`, `n_mads`, and
`min_cells_per_sample`. Manual filtering requires aligned `attrs`, `lows`, and `highs`, with an
optional Boolean `keep_bounds` value. Probability, MAD, and manual-bound values must be finite
numbers when present; booleans and numeric strings are rejected rather than coerced.
If filtering is requested and none of the default QC columns exists, validation raises instead of
silently analyzing the unfiltered cells. Pass `filtering=False` only when that choice is deliberate.

```python
run = ds.pipeline.run(
    filtering={
        "method": "manual",
        "attrs": ["RNA_nCounts", "RNA_nFeatures"],
        "lows": [1000, 500],
        "highs": [15000, 4000],
    },
)
```

The invocation is validated before a run record is created. Unknown options, missing columns,
invalid stage combinations, reserved snapshot fields, and an already completed label fail without
starting a run. A handled stage failure raises {py:class}`~scarf.PipelineExecutionError`. Use its
`run_id` to inspect the persisted failure, and its `stage` to identify the failing stage. The
original exception is retained as `__cause__`.

```python
from scarf import PipelineExecutionError

try:
    run = ds.pipeline.run(label="baseline")
except PipelineExecutionError as error:
    failed_run = ds.pipeline.open(run_id=error.run_id)
    report = failed_run.report(format="dict")
```

## Automatic cluster selection

The `cluster_selection` stage scores enabled Leiden resolutions in the same PCA or Harmony
coordinates used to build the graph. Paris remains `run["paris"]` for diagnosis and comparison; it
is never the automatic `run["clusters"]` winner. Selection uses one deterministic shared sample of
at most 10,000 selected cells with seed `4466`. The sample reserves up to two seeded cells per
cluster across every Leiden candidate, then fills remaining capacity without replacement. Pairwise
work stays within the datastore memory budget.

This silhouette comparison is a reproducible provisional baseline. It is not biological validation
or ground truth. Keep alternative Leiden refs and Paris when the study question needs other
evidence.

The resulting `cluster_selection` artifact records candidate keys and refs, silhouette scores,
the sample definition (`sampleStrategy="sharedClusterQuota"` and `minClusterQuota=2`),
invalid-candidate reasons, deterministic tie order, and the selected key.
`run["clusters"]` is the selected Leiden candidate's exact ref. It is not a copied artifact or metadata
column. If no Leiden candidate can be scored, the stage fails. A single valid Leiden candidate still
receives a decision artifact.

Persisted cluster selection is intentionally pipeline-only. The fixed recipe owns the candidate
set, deterministic sampling policy, and run-ledger decision. Granular callers can evaluate their
own partitions with public APIs such as `evaluate_cluster_separability` in `scarf.metrics`. There
is no public "best clustering" function and no separate `DataStore` persistence method that could
imply the pipeline policy outside a run. The agent orchestrator is a separate multi-metric
workflow and does not replace this baseline.

## Durable outputs and frozen views

A completed default run exposes these keys in order:

```text
input_cell_selection, analysis_cell_selection, feature_universe, cell_cycle,
highly_variable_features, normalized, pca, ann_index, neighbors,
connectivity_map, embedding_initialization, umap,
leiden_0.5, leiden_0.75, leiden_1.0, leiden_1.25,
paris, cluster_selection, clusters, doublets, markers
```

`harmony` appears after `pca` when enabled. Disabled stages omit their outputs. Normal mapping
operations such as `run["pca"]`, `list(run)`, and `run.items()` are available only on a successfully
completed run.

`run.cells` and `run.features` are narrow read-only views over captured selections, metadata
snapshots, and result artifacts:

```python
view.columns
view.fetch(column)                 # rows selected by the run's stored I
view.fetch_all(column)             # values aligned to the complete stored axis
view.to_pandas_dataframe(columns)  # selected rows
view.head(n=5)
```

Cell `I` is the analysis selection. Feature `I` is backed by the immutable
`run["feature_universe"]` selection created during the input stage, not the live feature `I`
column. The views retain captured names and requested metadata even if live values or live `I`
later change.
They fail closed if ordered cell or feature identities change. There is no live overlay or mutation
surface.

Pipeline execution writes immutable artifacts and run records only. It never inserts result
columns or rewrites live `I`. Plotting, marker loading, and export stay on `DataStore`:

```python
ds.plots.embedding(run=run, layout="umap", color_by="clusters")
markers = ds.get_markers(marker=run["markers"], group_id=0)
adata = ds.to_anndata(run=run)
# adata.obsm["X_umap"] holds frozen UMAP; cluster labels stay in adata.obs
```

## Open, list, and report runs

A successful optional label is an immutable name for one run:

```python
run = ds.pipeline.open(label="baseline")
same_run = ds.pipeline.open(run_id=run.run_id)

recent = ds.pipeline.list_runs(limit=20)
failed = ds.pipeline.list_runs(status="failed")
interrupted = ds.pipeline.list_runs(status="interrupted")
```

`open` requires exactly one of `run_id` or `label`. Only a successfully completed run acquires its
requested label. Failed and interrupted attempts do not reserve it. `list_runs` returns newest
first and includes all statuses unless filtered. Concurrent finalizers use an atomic label claim,
so at most one completed run can acquire a name. A storage backend without atomic conditional
creation rejects `label=` before a run record or computation starts; unlabeled runs are unaffected.
An unclean incomplete finalizer blocks reuse of its requested label and fails closed.

After confirming that the owner process has stopped, explicitly abandon that exact claim before
retrying the label:

```python
interrupted = ds.pipeline.abandon_label_claim(
    label="baseline",
    run_id=stopped_run_id,
    reason="worker terminated after finalization began",
)
replacement = ds.pipeline.run(label="baseline")
```

The recovery call succeeds only for the exact current owner while `complete=False`, including a
torn terminal payload, and records an `abandoned_label_claim` interruption. It refuses every
`complete=True` owner and has no timeout or automatic stale-process heuristic.

Every status exposes identity, status, and `run.report(format="dict" | "markdown")`. Only a
completed run exposes mapping outputs and frozen views. Reports include stage timing, sampled
process-tree RSS, artifact plans with `created` or `reused` dispositions, failures, interruption
details, and signal-guard availability. RSS peaks are sampled lower bounds. An unavailable
measurement is reported as null with a reason.

Run and stage records are strict and unversioned. Unknown or malformed fields fail closed. A hard
process death can leave `complete=False`; this is reported as an unclean incomplete run. There is
no resume, repair, or same-ID retry. A new invocation may reuse only complete artifacts.
Catalog scans and open-by-label skip malformed or torn children so healthy runs remain accessible;
opening the malformed child by its exact run ID remains strict.

## Graceful interruption and callbacks

On the main interpreter thread, a pipeline temporarily installs cooperative handlers for available
`SIGTERM`, `SIGINT`, and `SIGHUP`, while respecting signals already set to ignore. The first signal
requests shutdown at a safe checkpoint. The pipeline persists completed or interrupted stage and
run state, invokes interruption callbacks, restores prior handlers, and propagates the original
signal behavior. A second catchable termination signal escalates immediately. Non-main-thread runs
record that signal protection was unavailable.

`KeyboardInterrupt` and an escaped `asyncio.CancelledError` use the same durable interruption
boundary. Ordinary stage exceptions produce a failed run instead.
If a termination signal races with an ordinary failure or the final successful handoff, the
pipeline preserves that durable outcome and still propagates the pending signal after cleanup.

`SIGKILL`, `SIGSTOP`, out-of-memory termination, power loss, and expired shutdown grace periods
cannot perform cleanup. Their durable contract is an incomplete artifact or run, followed by a new
invocation that reuses only complete artifacts.

Callbacks provide serialized progress notifications without controlling execution:

```python
from scarf.datastore.pipeline_accessor import PipelineEvent

events: list[tuple[str, str]] = []


def record_event(event: PipelineEvent) -> None:
    events.append((event.kind, event.stage))


run = ds.pipeline.run(callback=record_event)
```

Enabled stages emit `stage_started` followed by `stage_completed`, `stage_failed`, or
`stage_interrupted`. A handled interruption also emits `pipeline_interrupted` after durable state is
written. Skipped stages are present in the report but emit no callback. Callback errors are logged
and cannot block durable status.

## Public types

```{eval-rst}
.. autoclass:: scarf.datastore.pipeline_accessor.PipelineAccessor
    :members: run, open, list_runs, abandon_label_claim

.. autoclass:: scarf.datastore.pipeline_accessor.PipelineEvent
    :members:

.. autoclass:: scarf.PipelineRun
    :members:

.. autoclass:: scarf.PipelineExecutionError
    :members:
```

See {ref}`Quick start <quickstart>` for an executable first run and {doc}`graph_construction` for
explicit stage-by-stage APIs.
