# Agent analysis API reference

The optional `scarf[agent]` dependency provides a bounded RNA workflow for prepared local
Scarf stores and Cytebase mounts. Numerical work uses the existing Scarf pipeline; agents
choose among registered options and provide provisional cluster identities.

## Start an analysis

Supply a configured Pydantic AI model and typed study context. Downloading, mounting, and input
conversion happen before this call. The complete audit is saved outside the numerical store,
by default under `agent_runs/<runId>` in the current working directory.

```python
from scarf.agent import AnalysisConfig, Study, analyze_rna

run = analyze_rna(
    "study.zarr",
    model=model,
    study=Study(
        context="Human blood from one healthy donor, with no treatment comparison.",
        objective="Identify major immune-cell populations.",
        organism="Homo sapiens",
        tissue="blood",
    ),
    config=AnalysisConfig(assay="RNA", scoreDoublets=False),
)
print(run.status)
report_path = run.report()
```

Set `run_dir="analysis/blood"` to choose another new directory. Existing directories are
rejected; omission generates a fresh ID matching the saved manifest. The source must be a
prepared local directory with one selected RNA assay. During development, complete numerical
artifacts from earlier analyses prevent admission, including results inherited through a mount.
Imported labels and embeddings remain allowed. The agent does not remove prior results or
prepare a clean copy automatically.

The default QC policy retains the supplied cohort and records outlier flags. Doublet scoring
is opt-in and never removes cells automatically. Harmony requires supported technical-batch
roles, evidence that protects biological differences, and the explicitly enabled doublet
diagnostics needed for correction validation. Missing replication does not block descriptive
population discovery.

Scientific settings in `AnalysisConfig` remain frozen for the run. `RuntimeConfig` controls
provider limits, threads, and memory. Configure credentials on the provider, not in saved
runtime settings. In a notebook with an active event loop, await `analyze_rna_async` instead.

| Input | Selected fields |
| --- | --- |
| `Study` | `context`, `objective`, `organism`, `tissue`, `sampleColumn`, `captureColumn`, `technicalBatchColumns`, `protectedColumns`, `excludedColumns`, `featureExclusions`, `referenceFiles`, `batchCorrectionEvidence` |
| `AnalysisConfig` | `assay`, `workspace`, `cellKey`, `qcPolicy`, `qcBounds`, `interactionMode`, `hvgCount`, `pcaDims`, `neighborsK`, `resolutions`, `maxCandidates`, `maxFinalists`, `scoreDoublets`, `randomSeed` |
| `RuntimeConfig` | `maxRequests`, `maxRequestsPerDecision`, `maxPromptBytes`, `maxOutputTokens`, `decisionTimeout`, `nthreads`, `memBudget`, `modelSettings` |

Pydantic model fields use camel case. New runs require `maxCandidates=4` or `5` and default to
five: four planned native representations and at most one eligible Harmony comparison. The
baseline is 1,000 HVGs, 21 PCs, 11 neighbors, with resolutions `0.5`, `0.75`, `1.0`, `1.25`.
Every graph and clustering uses the full retained cohort. At most two partitions receive
marker assessment before the final pinned recipe adds UMAP and reuses completed artifacts.
See {doc}`../../tutorials/agent_workflow` for the full procedure and diagnostic bounds.

`interactionMode="lenient"` is the default. It records conservative resolutions for optional
metadata uncertainty or model-declared acceptable ties. Strict mode leaves those questions
pending. Essential missing facts, provider failures, invalid output after bounded repair, and
unsupported objectives are not silently resolved. `exploration_coverage` and
`decision_resolutions` expose the recorded trial outcomes and policy choices.

## Inspect and resume

`AnalysisRun.status` is `running`, `needsInput`, `completed`, `failed`, or `interrupted`.
Inspect the status and `pending_questions` before using finalized numerical results. The
workflow records operational failures and unresolved questions; invalid API arguments and
incompatible resume requests can raise exceptions.

```python
from scarf.agent import open_analysis, resume_rna

run = open_analysis(run.run_dir)
print(run.pending_questions)
report_path = run.report()

# When input is requested, answer every exact questionId shown above.
run = resume_rna(run.run_dir, model=model, answers=answers)
```

Opening a run and regenerating its report require neither a model nor access to the source
store. Resume is explicit: repeating `analyze_rna` does not resume an existing directory.
`resume_rna_async` is the notebook counterpart. Unchanged scientific inputs, implementation,
and prompts are required for resume; the provider and operational settings may change.
Relocated stores can be supplied through `source=`, subject to identity and saved-artifact
validation.

## Use finalized results

After completion, `run.pipeline` opens the exact final `PipelineRun`, and `run.artifacts`
returns its artifact references. `get_markers()` reads the saved markers using core filtering
options. `plot_embedding()` plots the final UMAP, and `plot_markers()` shows marker means
and expressing fractions. These operations require the source store and verify its identity.

`save_plots()` saves UMAP and marker previews at 300 DPI. Call `report()` afterward to embed
those images in the offline `report.html`; it also writes `report.md` and `annotations.csv`.
Every saved outcome can be reported, including incomplete runs. `export(output_dir)` writes
aligned cell clusters, UMAP coordinates, markers, annotations, and a summary into a new
directory without recomputing the analysis.

Run requests, events, evidence, and model exchanges live in the external run directory.
Numerical artifacts and pipeline records remain in the Scarf store. Annotations remain
external and provisional; the workflow does not overwrite live cell metadata.

## External audit and compact store result

The external directory contains the complete audit: `run.json`, numbered immutable `events/`,
measured `evidence/`, visible model exchanges in `calls/`, `annotations.csv`, report files,
and saved previews. Do not discard it after completion.

Completion also publishes a compact result under the local store's `agent_results/<runId>`
group. This agent-owned summary is an attribute record, not a new core numerical artifact.
It contains:

- The agent run ID, final core pipeline run ID, assay, workspace, source fingerprint, and
  procedure identity. A `null` workspace means the default workspace; an explicit name preserves
  the location of the core run in a nondefault workspace.
- The exact final `pipelineConfig`, selected candidate and resolution, requested HVG count,
  PCs, neighbors, correction flag, and actual HVG count when measured.
- The saved selection rationale and a relative locator for the external audit directory.

```python
compact = run.compact_result
if compact is not None:
    print(compact["finalPipelineRunId"])
    print(compact["selectedParameters"])
```

This property verifies the source and exact final pipeline before returning the summary. It
does not publish, migrate, or repair a missing record. A missing compact result is `None`.
Published records are immutable and idempotent, with no mutable "latest" pointer. Numerical
truth remains in core artifacts and `PipelineRun`; the summary links them to the full audit.

A publication failure is saved as `resultPublicationError` without changing scientific
completion. Explicitly resuming a completed run retries publication. A relative external
locator is advisory: relocating a completed store preserves the original payload and can
record `resultLocatorStale`. Keep using the known external run directory when reopening or
rebinding. Older completed runs remain readable but are not automatically upgraded by opening
or reporting them.

## Provider contract

The model returns one structured choice or annotation batch. Registered options, evidence
identifiers, frozen inputs, and scientific gates are validated by Scarf. Default operational
limits are 30 observed requests per run, at most three per decision invocation, a 65,536-byte
serialized request, 4,096 output tokens, and a 120-second decision deadline. One semantic repair
and one transient transport retry are allowed within those limits. An explicit resume can
attempt unfinished decisions again while preserving the run-wide count and earlier failures.

Every decision and retry requests `thinking=False` with native reasoning effort `none` and
the controlled extra-body values below. Conflicting caller controls are overridden without
mutating the caller's model. Other settings are preserved.

```python
{
    "thinking": {"type": "disabled"},
    "reasoning_effort": "none",
    "chat_template_kwargs": {"thinking": False},
    "reasoning": {"enabled": False},
}
```

Provider support varies. Always-on models may ignore disable controls, and strict providers
may reject unsupported fields. Rejection is an operational failure; the agent does not retry
with reasoning enabled. Records retain supplied prompts, schemas, visible outputs, feedback,
timing, identity, and available usage, excluding credentials, hidden reasoning, and arbitrary
provider state. Missing usage stays unknown. These controls are not universal spending caps.

## Public interface

```{eval-rst}
.. autofunction:: scarf.agent.analyze_rna

.. autofunction:: scarf.agent.analyze_rna_async

.. autofunction:: scarf.agent.resume_rna

.. autofunction:: scarf.agent.resume_rna_async

.. autofunction:: scarf.agent.open_analysis

.. autoclass:: scarf.agent.AnalysisRun
   :members: status, pending_questions, pipeline_runs, candidates, exploration_coverage, decision_resolutions, pipeline, artifacts, compact_result, annotations, get_markers, plot_embedding, plot_markers, save_plots, report, export, replay_decisions
   :undoc-members:

.. autoclass:: scarf.agent.Study

.. autoclass:: scarf.agent.AnalysisConfig

.. autoclass:: scarf.agent.RuntimeConfig
```

## Prototype compatibility

The former `AutomatedWorkflowResult`, `AnalysisError`, and orchestrator interfaces are
unsupported. Prototype histories cannot be resumed or migrated; start a new run directory.
Existing numerical artifacts remain available through core Scarf. The current
{doc}`../../tutorials/agent_workflow` demonstrates the supported interface with an offline
scripted provider and real numerical execution.
