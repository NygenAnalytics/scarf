# Agent analysis API reference

The optional `scarf[agent]` dependency provides a small interface for selecting and explaining
RNA analysis settings. The root `scarf.agent` facade exports three objects:

```{eval-rst}
.. autofunction:: scarf.agent.analyze_rna

.. autoclass:: scarf.agent.AutomatedWorkflowResult
   :members: plot_embedding, get_markers, report

.. autoexception:: scarf.agent.AnalysisError
```

## Start an analysis

For model setup and a beginner walkthrough, see {doc}`../../tutorials/agent_workflow`.

```python
from scarf.agent import analyze_rna

result = analyze_rna(
    "study.zarr",
    model=model,
    study_context="Human blood from one healthy donor, with no treatment comparison.",
    study_objective="Identify stable major immune-cell populations.",
    score_doublets=False,
)
result.plot_embedding()
markers = result.get_markers()
report_path = result.report()
```

The four required inputs are the source, configured Pydantic AI model, study context, and study
objective. `assay` selects an RNA assay when several exist. `zarr_path` selects the destination
when converting a supported input file. Other modalities in an existing store are ignored.
Automated integration, HTO assignment, and biological significance or differential-expression
hypothesis execution are outside this workflow. Experimental Context still explores individual
and joint covariate patterns and possible explanations of the study design.

The beginner call returns only after completion. If it cannot complete, it raises `AnalysisError`,
whose `result` provides the status, notes, and exact saved address for investigation or advanced
resume. An identical repeated call reuses a completed analysis or resumes matching interrupted
work. Input and model identity checks prevent attaching changed analysis intent to saved work.

## Doublet scoring and correction

`analyze_rna` defaults to `score_doublets=False`. This disables optional advisory doublet
scoring when Harmony is unavailable or prohibited. When the assessed design permits Harmony,
the workflow still computes the doublet diagnostics required to compare native and corrected
representations. Set `score_doublets=True` to request advisory scoring as well. Neither option
automatically removes cells.

Correction requires an assessed design that protects relevant biology. When that design permits
correction, the workflow evaluates Harmony rather than treating uncertain benefit as a reason
to skip it. Demonstrated confounding can prohibit correction; improved mixing does not override
that constraint or the preservation checks.

## Use the result

`plot_embedding()` displays the exact final UMAP colored by its clusters and returns Scarf's
established `PlotResult`. It accepts `figsize`, `show`, `seed`, and `max_points` from 1 to 50,000.
Sampling affects only display; cluster counts and final marker statistics describe the complete
QC-retained population. The image provenance records the workspace, artifact references,
population counts, and display sampling. No live metadata is created or overwritten.

`get_markers()` returns the saved marker table using Scarf's standard `group_id`, `min_score`,
and `min_frac_exp` filters. `report()` returns the local `index.html` path and can regenerate the
single-page summary from saved evidence. It does not open a browser or rerun the analysis.

The result stores a small address and outcome, including `zarrPath`, `workspace`, and
`workflowRunId`. Requests, scientific reports, decision histories, and final artifacts remain
owned by the orchestration journal. They are not repeated as public result fields.

## Advanced control

```python
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
)

runner = AgentOrchestrator(model, config=AutomatedWorkflowConfig(
    inputPolicy="pause", scoreDoublets=False,
))
result = runner.run(AutomatedWorkflowRequest(
    sourcePath="study.zarr",
    workspace="analysis",
    studyContext="The observed study design and metadata roles.",
    studyObjective="The biological structure that must be preserved.",
))
```

The advanced interface exposes numerical limits, provider limits, existing-store workspaces, and
explicit pauses. Inspect its returned status and questions before continuing. The example sets
`scoreDoublets=False` to match the beginner call. The advanced configuration retains
`scoreDoublets=True` as its default, including for compatible older saved configurations.

### Screening and work limits

`screeningCells=None` selects automatic sampling: 10% of the QC-retained cohort, rounded up,
bounded to 10,000–100,000 cells, and capped by the retained population. An integer of at least 20 sets
a fixed-size override. A compatible saved integer keeps its exact meaning on resume; the new
default does not resize an existing screening population.

| Default allowance | Limit |
|---|---:|
| Initial screening population | Automatic sampling as above |
| Evidence-triggered enlargement | One, up to 100,000 cells |
| Candidate evaluations per screening population | 24 |
| Candidate evaluations across screening populations | 48 |
| Additional full-cohort validation graphs | 4 |
| Additional full-cohort validation partitions | 8 |
| Targeted full-cohort repair | 1, within those graph and partition allowances |

The selected settings are executed and validated on all retained cells. Screening includes
every retained cell in small datasets; those comparisons count in the screening allowance, and
exact artifacts can be reused for final validation. The additional-validation allowance is not
a cap on every graph built during all-cell screening. A recovery comparison and its required
controls must fit the remaining allowance before execution. Four corrected resolutions plus
four matched native controls, for example, use all eight additional partitions.

These limits count distinct admitted work, including failed attempts. Exact completed reuse
does not spend another slot. They do not bound wall time, retries, diagnostic suboperations,
or provider spend. QC, markers, stability, doublet diagnostics, I/O, and the final UMAP also
have costs.

### Failures and saved history

Beginner failures raise `AnalysisError`; its message includes the stage, reason, and available
resume location. Its `result` retains the structured outcome used by advanced callers. Keep the
store and repeat the same call, including model configuration and doublet setting, to reuse
matching work. An unresolved scientific requirement may need investigation before a repeat can
succeed. Changed data, metadata, study text, model configuration, or execution settings cannot
silently reinterpret an existing history.

The stage journal owns requests, committed evidence, validated decisions, rationales, and artifact
references. Failed model attempts retain available usage and validation feedback. History
separates attempted and completed operation calls from restored evidence and confirmed reuse.
Counts of saved artifacts do not establish how many computations ran: a core call may itself
reuse work, and older histories without operation records have unknown counts, not zero.

Compatible histories can append a context-evidence revision when a requested joint or
conditional question was unanswered. Previous records remain immutable; changed scientific
evidence must be reassessed.

The previous agent workflow records, result fields, candidate-budget aliases, and root imports
are unsupported. Histories without mandatory objective requirements and completed comparison
coverage cannot be resumed or used to regenerate reports under this contract. Start a new
workflow; existing historical HTML remains readable. Numerical artifacts remain accessible through
the ordinary Scarf artifact APIs; no saved records are silently migrated.

For a complete executable example, see {doc}`../../tutorials/agent_workflow`.
