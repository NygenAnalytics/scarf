# Agent analysis API reference

The optional `scarf[agent]` dependency provides a bounded RNA workflow for prepared local
Scarf stores and Cytebase mounts. Numerical work uses the existing Scarf pipeline; agents
choose among registered options and provide provisional cluster identities.

## Start an analysis

Supply a configured Pydantic AI model, study context, and a new run directory outside the
numerical store. Downloading, mounting, and input conversion happen before this call.

```python
from scarf.agent import AnalysisConfig, Study, analyze_rna

run = analyze_rna(
    "study.zarr",
    run_dir="analysis/blood",
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

The default QC policy retains the supplied cohort and records outlier flags. Doublet scoring
is opt-in and never removes cells automatically. Harmony requires supported technical-batch
roles, evidence that protects biological differences, and the explicitly enabled doublet
diagnostics needed for correction validation. Missing replication does not block descriptive
population discovery.

Scientific settings in `AnalysisConfig` remain frozen for the run. `RuntimeConfig` controls
provider limits, threads, and memory. Configure credentials on the provider, not in saved
runtime settings. In a notebook with an active event loop, await `analyze_rna_async` instead.

## Inspect and resume

`AnalysisRun.status` is `running`, `needsInput`, `completed`, `failed`, or `interrupted`.
Inspect the status and `pending_questions` before using finalized numerical results. The
workflow records operational failures and unresolved questions; invalid API arguments and
incompatible resume requests can raise exceptions.

```python
from scarf.agent import open_analysis, resume_rna

run = open_analysis("analysis/blood")
print(run.pending_questions)
report_path = run.report()

# When input is requested, answer every exact questionId shown above.
run = resume_rna("analysis/blood", model=model, answers=answers)
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

## Public interface

```{eval-rst}
.. autofunction:: scarf.agent.analyze_rna

.. autofunction:: scarf.agent.analyze_rna_async

.. autofunction:: scarf.agent.resume_rna

.. autofunction:: scarf.agent.resume_rna_async

.. autofunction:: scarf.agent.open_analysis

.. autoclass:: scarf.agent.AnalysisRun
   :members: status, pending_questions, pipeline_runs, candidates, pipeline, artifacts, annotations, get_markers, plot_embedding, plot_markers, save_plots, report, export, replay_decisions
   :undoc-members:

.. autoclass:: scarf.agent.Study

.. autoclass:: scarf.agent.AnalysisConfig

.. autoclass:: scarf.agent.RuntimeConfig
```

## Prototype compatibility

The former `AutomatedWorkflowResult`, `AnalysisError`, and orchestrator interfaces are
unsupported. Prototype histories cannot be resumed or migrated; start a new run directory.
Existing numerical artifacts remain available through core Scarf. The older
{doc}`../../tutorials/agent_workflow` describes the prototype and requires migration to the
interface on this page before its examples can be executed.
