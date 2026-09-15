---
description: Use Scarf safely in an autonomous or AI-assisted single-cell analysis.
---

(analysis_with_agents)=
# Analysis with AI agents

This page is a routing and reasoning guide for an AI agent that uses Scarf to analyse data.
It does not replace the workflow tutorials or define one correct analysis.
The study question, experimental design, and user instructions remain authoritative.
For an executable ingest-to-finalization example with persisted decisions and report generation,
see {doc}`tutorials/agent_workflow`.

## Scope and authority

An agent may inspect data, run documented public methods, create reversible analysis branches, and compare alternatives.
Report a supported provisional choice and label each statement as:

- a computational fact, such as which artifact or cells produced a result;
- statistical evidence, such as a mixing score or marker AUC;
- a biological interpretation, which may require study context and independent validation.

Do not convert a default, visual pattern, or single metric into biological ground truth.
State assumptions when study metadata are incomplete.
Stop rather than claim an identified treatment, disease, or batch effect when the relevant variables are perfectly confounded.

## How Scarf represents an analysis

### DataStore, selections, and artifacts

A `DataStore` contains count matrices, cell metadata, feature metadata, and persisted results.
A Boolean {term}`cell key` can define an initial cohort. Snapshot it before analysis so downstream
operations and agents consume an immutable cell-selection artifact. An immutable
{term}`feature selection` artifact selects assay features.
Analytical producers return exact {py:class}`~scarf.ArtifactRef` values and leave metadata
unchanged. Filtering returns a cell-selection artifact without changing live `I` or
deleting counts.

Persisted results are immutable {term}`artifacts <artifact>`.
Their provenance records the operation, scientific parameters, and input artifacts.
A durable {py:class}`~scarf.PipelineRun` records one complete recipe invocation and exposes frozen
views over its selections and result fields. Granular workflows retain returned artifact refs and
pass them explicitly.

### Branches and mounts

Alternative results can coexist.
Retain each granular method's returned {py:class}`~scarf.ArtifactRef` and pass that reference to the
next operation. No branch becomes a global implicit result.
See {doc}`tutorials/custom_analyses` for a complete example and {doc}`concepts/provenance` for the storage model.

Opening a mounted store with another workspace name does not create a separate analysis.
A mount records one source workspace, and reopening that target with a different workspace is rejected.
Create a separate mount target for each analysis that needs independent metadata and results.
Use artifact branches within one target when only parameters or downstream methods differ.
Mount behavior is documented in {doc}`tutorials/remote_stores`; general layout is documented in {doc}`tutorials/data_organization`.

## Start or continue safely

### Inspect the current store

Start with `snapshot = ds.summary()`.
It reports the workspace, default assay, resource budget, metadata column names, complete or
incomplete artifacts, pipeline-run counts, and completed run labels without exposing a store
location.
`active_cells` and each assay's `active_features` count the literal `I` columns.
They do not follow a historical artifact or pipeline-run selection.
Resolve and inspect feature-selection artifacts separately when evaluating a branch.
Use `snapshot.to_dict()` for a deterministic JSON-safe record.

Before choosing the next operation:

1. Identify the assays and the current default assay.
2. Inspect cell and feature metadata columns, including active selections.
3. Use `ds.pipeline.list_runs()` and run reports to identify durable recipe invocations.
4. Use `ds.list_artifacts(complete_only=True)` to find persisted alternatives.
5. Use `ds.inspect_artifact(ref)` before consuming an explicit result.
6. Use `ds.lineage(ref)` to verify upstream inputs.

`lineage.to_markdown()` and `lineage.to_mermaid()` produce compact provenance records for an analysis log.
Do not infer state by reading private Zarr paths.
Use public methods instead of mutating `ds.z`, `ds.zw`, or stored run records.

Inspect unfamiliar H5AD files with {py:func}`scarf.inspect_h5ad` before creating a reader.
This reports matrix candidates, encodings, dimensions, and suggested assays instead of relying on guessed keys.
Use the format-specific import guides for other inputs.

### Prospective execution record

Before the first mutating operation, make a short execution record containing:

- the scientific question and unit of inference;
- the cell-selection and feature-selection references, plus other input artifact references;
- the operations that will persist artifacts or export data;
- the alternatives and independent evidence that will be compared;
- the criteria for selecting a branch, preserving uncertainty, or stopping.

Update this record before changing the cohort, inputs, or decision criteria.
This prospective boundary makes unintended writes and retrospective justifications visible.
`AgentOrchestrator` keeps one authoritative stage history containing the immutable request,
effective configuration, evidence, decisions, checks, and final artifact references. The caller still owns the scientific question and
unit of inference.

### When to use the automated agent workflow

Use `analyze_rna` with a dataset, a configured model, study context, and a study objective.
The workflow analyzes one RNA assay, even when other modalities coexist in the store. Pass
`assay` explicitly when several RNA assays exist. Automated integration, HTO assignment, and
biological significance or differential-expression hypothesis execution are outside this
workflow; ordinary Scarf APIs remain available for them. Experimental Context still explores
individual and joint covariate patterns and possible explanations of the study design.

```python
from scarf.agent import analyze_rna

result = analyze_rna(
    "study.h5ad",
    model=model,
    study_context="One paragraph describing the study design and metadata roles.",
    study_objective="Discover stable populations relevant to the study.",
    zarr_path="study.zarr",
)
result.plot_embedding()
markers = result.get_markers()
report_path = result.report()
```

The beginner call returns a completed result or raises `AnalysisError`. Its result address is
available as `error.result` when work remains unresolved. There is no separate candidate-budget
argument on this interface. Advanced callers import the orchestrator and configuration from
`scarf.agent.orchestrator` for explicit workspaces, numerical limits, and resumable pauses.
See {doc}`reference/api/agent` for that boundary.

The model first interprets observed study metadata and proposes at most eight objective-led
design comparisons, with one follow-up round of at most four. Comparisons may use a single
explanatory variable, a joint categorical group, or conditioning within categorical strata.
Continuous conditioning bins and regression adjustments are not invented. Continuous biological
variables without a supported preservation measure remain explicitly unresolved.

Scarf starts from its RNA settings and four partitions of the same graph. The model reviews
quantitative diagnostics, marker and loading-gene evidence, and supplied images before accepting
or requesting one registered experiment. The model cannot generate executable analysis code or
arbitrary `DataStore` calls. When the design permits correction, a matched native/Harmony
evaluation is required even when correction initially appears unnecessary. Accepting correction
requires measured improvement while preserving protected biology, including supported joint
groups, and passing the existing doublet checks. Unsafe or unknown designs cannot license
correction.

New workflows use an immutable uniform screening cohort containing 10% of retained cells,
rounded up and bounded to 10,000–100,000 cells, never exceeding the retained population.
One larger nested screen may use up to 100,000 cells when the first is smaller. Existing
explicit integer screening sizes retain their exact meaning on resume. A compatible interrupted
run configured for 50,000 screening cells keeps that setting and its matching admitted work;
it does not switch to the new fractional default. Coverage and
rare-population concerns can require targeted full-cohort recovery of measured settings;
a missing screening comparison does not authorize an unbounded full-cohort search.
Screening selects settings; it does not replace the final QC-retained cohort. Selected settings
are executed and assessed on the full cohort. Each screening population must compare the
baseline against 2,000 and 4,000 variable genes, 10 and 30 PCA dimensions, and 21 and 41
neighbors, changing one setting at a time. The four baseline resolutions share one graph.
Supported batch-aware ranking and an evidence-nominated feature policy provide additional
comparisons. The agent interprets these results, proposes combined settings, and assesses their
actual execution and resolution alternatives before accepting them. A list of reviewed domains
or a general preference for defaults does not establish sufficient evidence.

The default limits allow 24 screening evaluations per population and 48 overall, with four
additional final-validation graphs, eight additional partitions, and one targeted repair.
For small cohorts, discovery uses every retained cell; these are full-sized comparisons counted
in the screening allowance. Exact completed artifacts are reused for final validation without
another admission. Report the analyzed population and diagnostic operations separately; the
candidate allowance does not bound doublet calculations, elapsed time, or provider cost.
A proposed recovery panel and its matched native controls must fit together before execution.
Four corrected resolutions and four native controls consume all eight additional partitions;
the one-repair limit does not reserve a ninth partition.

Advanced history records attempted, completed and failed operation calls separately from metric
cache hits, saved-evidence restores and confirmed artifact reuse, for each invocation. These
observed calls differ from counts of unique saved artifacts, and a core operation may itself
reuse earlier work. Older histories without operation records have unknown counts, not zero.

Experimental Context records objective evidence requirements before tuning. Explicit joint
or conditional covariate requests remain unresolved when only marginal comparisons were
nominated. Full study text is retained; compact model views deduplicate shared sources and
capture-design evidence while complete measurements remain in the stage journal. These
summaries retain adverse findings, missingness, protected-group loss and design constraints.
The agent can retrieve one exact saved policy/capture or design record when detailed thresholds
or donor examples are needed. This lookup performs no scientific recomputation. Completed
metadata inspection and design rounds are checkpointed before further model requests, so
interruption does not reset the eight-initial/four-follow-up allowance. Repeated donors and
incomplete pairing receive descriptive counts and support summaries without treating cells or
repeated samples as independent replicates. A method that cannot compute an association does
not establish that an effect is absent or unidentifiable. Essential unresolved evidence blocks
a consequential decision; measured confounding may prohibit correction while descriptive
population discovery remains possible.

RNA percentages use explicit gene-selection artifacts. Imported percentage columns remain
available for comparison but do not override a validated definition. In particular, a
mitochondrial symbol definition matches `MT-` rather than every gene beginning with `MT`.
Changing the metric definition requires new QC thresholds and dependent evidence.

All saved execution and decisions belong to the orchestration stage history. Identical calls
reuse completed work or resume matching interrupted work; changed inputs and identity checks
prevent silent reuse of stale evidence. The result's plot and marker methods use the exact saved
workspace and artifacts read-only. The cluster map displays at most 50,000 cells while retaining
full counts and provenance. `report()` returns or regenerates one local analysis summary from
saved evidence, with no new model calls or numerical analysis.

This release deliberately breaks the earlier agent imports and persistence contracts. The root
agent facade exports only `analyze_rna`, `AutomatedWorkflowResult`, and `AnalysisError`. Runs using removed configuration or incompatible saved contracts must be restarted; their
numerical artifacts remain readable through ordinary Scarf APIs.
There are no implicit migrations. Compatible histories with newly uncovered objective questions
receive an explicit context-evidence revision, preserving prior records and artifacts; essential
unanswered questions still prevent completion. Model attempt records include known provider
usage, output-validation feedback and failures, with unavailable usage labeled explicitly.
Standalone scientific agent APIs remain in their concrete
packages, such as `scarf.agent.biological_interpretation`.

### When to use the pipeline

`ds.pipeline.run()` is a persistent baseline workflow.
It writes immutable artifacts and a durable run ledger, but does not change live `I` or metadata
columns. It scores enabled Leiden resolutions in the same PCA or Harmony coordinates used to
build the graph, persists the decision as `run["cluster_selection"]`, and exposes the selected
Leiden candidate ref as `run["clusters"]`. Paris remains `run["paris"]` for diagnosis. Silhouette
supplies a reproducible baseline, not validation or ground truth. The agent orchestrator is a
separate multi-metric workflow and does not replace this pipeline selection.

Use `run.cells` and `run.features` for frozen inspection. Keep presentation and storage mutation on
the datastore: `ds.plots.embedding(run=run, ...)`, `ds.get_markers(marker=run["markers"], ...)`,
and `ds.to_anndata(run=run)`. Use an immutable run label when a completed run needs a
human-readable name. Use granular methods and explicit artifact refs for alternatives outside the
fixed recipe. For example,
`ds.plots.embedding(layout=embedding_ref, color_by=cluster_ref)` consumes granular outputs without
materializing either as metadata.

## Scientific decision loop

Use the following loop for each consequential choice:

1. **Frame the question.** Decide whether the target is population discovery, technical integration, reference mapping, abundance, within-population state, differential expression, or a continuous trajectory.
2. **Establish the unit of inference.** Identify subjects, biological replicates, conditions, batches, paired measurements, and repeated measures.
   Inspect their contingency before correction or statistical comparison.
3. **Keep a baseline.** Evaluate uncorrected data before batch correction and preserve broad, pre-subclustering labels before splitting populations.
4. **Create alternatives.** Change one consequential choice at a time where practical, use explicit artifact references, and keep seeds fixed.
5. **Compare independent evidence.** Combine graph, marker, metadata, replicate, mapping, and method-specific diagnostics.
   Do not optimize one score in isolation.
6. **Decide or stop.** Select a fit-for-purpose branch when evidence is coherent.
   Preserve alternatives and report uncertainty when evidence conflicts.
   Stop when the design cannot identify the requested effect.

## Route by task

Use {doc}`reference/api` for exact public signatures and result contracts.
The pages below explain when and how to evaluate those operations.

### Import, quality control, and feature selection

- Use {doc}`tutorials/import_and_export` for supported inputs and exports.
- Use {doc}`tutorials/quality_control` to inspect modality-specific distributions and derive dataset-specific, often sample-aware selections.
- Use {doc}`tutorials/feature_selection` to compare feature sets without treating partition agreement as proof of biological usefulness.

Report retained cells by sample or condition when those columns are available.
Do not copy thresholds from another tissue or protocol without inspecting the current distributions.

### Reduction, graph construction, and clustering

- Use {doc}`tutorials/graph_construction` for granular normalization, reduction, neighbour, and graph methods.
- Use {doc}`tutorials/dimensionality_reduction` to compare dimension counts and layouts.
- Use {doc}`tutorials/clustering` to compare Leiden resolutions, Paris cuts, graph connectivity, membership strength, and marker specificity.

There is no universally optimal partition.
A defensible partition should fit the question, avoid being driven solely by technical covariates, retain replicate support, and have interpretable positive and negative evidence.
A split population should rebuild feature selection and its graph inside the subset.
Preserve the parent labels so the split remains auditable.

### Integration, mapping, and paired modalities

Start with {doc}`tutorials/dataset_merging` when compatible datasets need one joint datastore, then use {doc}`tutorials/batch_correction` only when a defensible technical covariate should be reduced.

- Merge without correction when compatible datasets need joint inspection but no technical effect has been identified.
- Compare an uncorrected graph with partial PCA or Harmony when a defensible technical covariate should be reduced.
- Use fixed-reference mapping when queries must remain comparable to one reference over time.
- Use WNN by default for modalities measured in the same cells. Use SNN only when equal graph
  support is the intended comparison. Neither method integrates independent batches.

The batch-correction workflow compares source mixing and structural preservation.
Scarf's `metric_*` methods provide evidence, not an automatic winner.
Never correct a variable that is indistinguishable from the condition of interest and then claim the condition was preserved.

### Annotation, contrasts, and differential expression

Use {doc}`tutorials/annotation` to combine marker specificity, AUC, expression fraction, and known positive and negative markers.
Identify mixed populations explicitly and retain uncertain labels.

Marker search treats cells as observations.
For condition-level inference, use {doc}`tutorials/pseudobulk_and_differential_expression` to aggregate by biological replicate and export counts to a replicate-aware statistical method.
Descriptive pseudo-replicates made by splitting the same cells are not independent biological replicates.

Separate three questions that require different evidence:

- whether population abundance changes across samples;
- whether expression changes within a defined population;
- whether the population definition itself changes.

### Pseudotime and fate mapping

Use pseudotime only when the graph and biological question support a plausible continuous process.
Source and sink labels supervise the orientation; Scarf does not discover terminal states.

- Use {doc}`tutorials/pseudotime` for source and sink scoring.
- Use {doc}`tutorials/expression_dynamics` for expression dynamics.
- Use {doc}`tutorials/fate_mapping` for multiple terminal outcomes.
- Use {doc}`tutorials/trajectory_validation` to compare boundaries, graph components, marker tests, modules, and fate-probability validity.

Compare plausible source and sink definitions when the endpoints are uncertain.
Check validity keys, graph components, terminal probabilities, and expected marker trends.
A pseudotime or fate artifact is a model-based summary, not evidence of causal lineage.

### Methods outside Scarf

Use {doc}`tutorials/custom_analyses` for block streams, graph access, custom selections, external reductions, and supported export paths.
Export when the question needs RNA velocity, scVI, peak calling, FRiP/TSS, or another method outside Scarf.
Do not write arbitrary artifact groups directly.

## Troubleshooting

Classify the problem before retrying:

- **Input or schema:** inspect the source format and verify assay, feature, and cell identifiers.
- **Run or provenance:** inspect the pipeline report, stored cell selection, feature-selection refs,
  artifact status, and lineage.
  `ArtifactResolutionError.code` distinguishes missing, incomplete, or changed selection inputs.
  Do not consume incomplete artifacts.
- **Resource or I/O:** inspect the configured memory budget, worker count, storage profile, and remote latency.
  If an RNA store fails to open, treat it as a missing or outdated count layout and rebuild or `repack_zarr` rather than retrying the analysis stage.
  See {doc}`concepts/memory_and_execution`, {doc}`concepts/benchmarks`, and {doc}`tutorials/remote_stores`.
- **Numerical or graph:** check dimensions, graph components, neighbour count, convergence, validity masks, and method-specific diagnostics.
- **Scientific ambiguity:** preserve branches, seek another independent form of evidence, narrow the claim, or report that the available design does not resolve the alternatives.

In a granular workflow, retry the lowest failed stage. A failed pipeline run is not resumable;
start a new run, which can reuse matching complete artifacts from the earlier attempt. An automated
agent workflow validates its exact request and stage history before reusing completed work or
resuming an interruption. Explicit questions require grounded answers through the advanced
resume interface. A changed dataset, model identity, or configuration cannot be silently attached
to an older history.

## Progress and deterministic comparisons

`ds.pipeline.run(..., callback=...)` emits
{py:class}`~scarf.datastore.pipeline_accessor.PipelineEvent` values when stages start, complete,
fail, or are interrupted.
Use the callback to update an external progress record; callback failures do not control the pipeline.
The durable report records stage timing, sampled process-tree RSS, and exact created or reused
artifact plans. Granular analytical methods return artifact references.

When comparing branches, retain the same cells, features, neighbour settings, and seeds unless one is the variable being tested.
Record every deliberate difference.
A layout can vary in orientation or spacing without representing different biology.

## Analysis handoff

A useful handoff reports:

- the scientific question and unit of inference;
- the store workspace, exact cell selection, resolved feature selections, and relevant artifact refs;
- metadata roles and any confounding;
- alternatives considered and the evidence used to compare them;
- the selected result and why it is fit for the question;
- unresolved uncertainty, unsupported claims, and required external validation;
- exported files or external methods that continue the analysis.

Artifact provenance records how Scarf produced a result.
It does not replace this study-level reasoning record.
For an automated run, the result's plotting, markers, and report helpers resolve the exact final
artifacts from the authoritative stage history. The result is a small address, not a second saved
copy of requests, reports, decisions, or finalization state. The HTML report is a replaceable
presentation of that history.
See {doc}`index` for the implemented methods and current boundaries.
