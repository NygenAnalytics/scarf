---
description: Use Scarf safely in an autonomous or AI-assisted single-cell analysis.
---

(analysis_with_agents)=
# Analysis with AI agents

This page is a routing and reasoning guide for an AI agent that uses Scarf to analyse data.
It does not replace the workflow tutorials or define one correct analysis.
The study question, experimental design, and user instructions remain authoritative.
For an executable prepared-store-to-finalization example with saved decisions and reports,
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
`scarf.agent` saves the immutable request, scientific policy, measured evidence, structured
decisions, validation outcomes, and final references in an external run directory. A compact
completed result in the local store links to the selected core pipeline and that audit.
The caller still owns the scientific question and unit of inference.

### When to use the automated agent workflow

Use `analyze_rna` with a prepared local Scarf RNA store, a configured Pydantic AI model, and a
`Study`. Input conversion and mounting happen separately. The agent analyzes one selected RNA
assay, even when other modalities coexist. Automated multimodal integration, HTO assignment,
subclustering, differential-expression hypothesis testing, and causal inference are outside this
workflow; ordinary Scarf APIs remain available for separate analyses.

```python
from scarf.agent import AnalysisConfig, Study, analyze_rna

result = analyze_rna(
    "study.zarr",
    model=model,
    study=Study(
        context="One paragraph describing the study design and metadata roles.",
        objective="Describe populations and uncertainty relevant to the study.",
        excludedColumns=["author_annotation"],
    ),
    config=AnalysisConfig(assay="RNA"),
)
print(result.status)
report_path = result.report()
```

Use `await analyze_rna_async(...)` inside a notebook event loop. Unless `run_dir` is supplied,
the full audit and report go to `agent_runs/<runId>` under the current working directory.
`AnalysisRun.status` records `running`, `needsInput`, `completed`, `failed`, or `interrupted`.
An incomplete persisted outcome is inspectable rather than being presented as a completed
analysis. `open_analysis` reads saved results, and `resume_rna` explicitly continues compatible
unfinished work. See {doc}`reference/api/agent` and the executable
{doc}`tutorials/agent_workflow` for the complete interface.

During development, new analyses reject prior complete numerical artifacts, including inherited
mount artifacts. Imported labels and embeddings are permitted. Existing analysis output is not
deleted. Prepare a clean input separately when necessary; same-run execution and resume can
reuse the artifacts they created. The input identity, consumed metadata, and Scarf procedure
identity are checked before execution and resume.

The scientific procedure is fixed. It freezes the cohort and feature policy, then measures the
baseline and three independent native probes: HVGs, PCs, and neighbors. Each alternative changes
one setting from the same baseline. Baseline defaults are 1,000 HVGs, 21 PCs, 11 neighbors, and
four Leiden resolutions. The model chooses among feasible PC probes, shortlists at most two
measured partitions, compares their markers and diagnostics, and selects one finalist. An
eligible optional fifth representation can test Harmony against its exact native counterpart.
Finalization pins the selected recipe, adds UMAP, and reuses its completed markers.

All graphs, clusters, markers, and final coordinates use the full retained cohort. PC/covariate
and correction diagnostics read at most 10,000 cells; silhouette assessment uses at most 2,000.
Same-resolution parent comparisons use aligned full-cohort labels. Infeasible, identical-HVG,
or failed probes remain visible in `exploration_coverage`. The model cannot replace required
feasible native probes with an early declaration that defaults are adequate. Seven normal
pipeline invocations suffice without Harmony, or eight with it; recovery may add invocations.
These are search limits, not elapsed-time or provider-spending guarantees.

Default QC retains the supplied cohort and flags outliers. Additional global manual or gentle
MAD filtering requires caller configuration. Projected retention under alternative policies is
not executed filtering. The HVG blacklist is explicit and normally excludes mitochondrial
names matching `^mt-`, case-insensitively. Other biological gene families remain included unless
explicitly excluded. An organism value alone does not supply missing gene definitions.

Supplied sample, capture, technical-batch, and protected roles remain distinct. Inferred sample
or capture roles are diagnostic-only and cannot establish replication or permit correction.
Missing labels and masked values remain missing. Design cross-tabs, gene-family audits, bounded
PC associations, and finalist group/QC summaries are descriptive evidence, not causal tests.
Missing replication does not prevent descriptive population discovery.

Harmony requires complete declared batch labels, protected biological metadata, experimental
evidence separating batch from biology, and explicitly enabled doublet scoring. Unknown or
confounded designs retain native analysis. Corrected acceptance needs an exact native pair at
the same resolution and passes mixing, preservation, marker, and doublet gates with a `0.05`
tolerance. Better mixing alone cannot pass those gates. Doublet scoring is never enabled
implicitly and never removes cells automatically.

The default lenient interaction mode resolves only supported ambiguities through frozen,
recorded conservative policies. Optional unknown facts stay unknown. Model-declared acceptable
ties prefer native analysis, then frozen candidate/resolution order. This is not evidence of
biological superiority. Strict mode keeps those questions pending; essential missing input
blocks both modes. Provider failure, exhausted request budgets, and invalid structured output
after one bounded semantic repair remain failures. No unbounded scientific repair loop runs.

Every final cluster receives an annotation record in batches of at most eight. A named identity
requires at least two observed qualifying supporting markers; unassigned identities remain
valid outcomes. This validates the cited measurements, not the correctness of the proposed
identity or every narrative claim. Author labels and explicitly excluded columns are held out
from decision evidence. Marker support is the fraction of clusters with at least one qualifying
marker, not a biological-coherence score. Confidence is qualitative.

Complete audit records remain external. Numerical artifacts and pipeline ledgers remain in the
store. The compact `agent_results/<runId>` summary links the final configuration and selection
rationale to the external audit; `result.compact_result` reads it after source verification.
It is not a replacement for full evidence, annotations, or core provenance. The workflow does
not overwrite live `I`, feature metadata, or annotation columns. Report generation is offline
and supports every saved outcome; report errors do not downgrade scientific completion.

Prototype orchestrator, standalone-agent, `AutomatedWorkflowResult`, and `AnalysisError` APIs
are retired. Old histories remain untouched and are not automatically migrated. Current
resume requires unchanged scientific inputs and procedure/prompt identity, but operational
settings and the provider can change for unfinished decisions. Visible prompts, responses,
validation feedback, and available usage are recorded. Hidden reasoning and credentials are
excluded. Every provider request applies the agent's reasoning-off controls; always-on providers
cannot be guaranteed to honor them.

### When to use the pipeline

`ds.pipeline.run()` is a persistent baseline workflow.
It writes immutable artifacts and a durable run ledger, but does not change live `I` or metadata
columns. It scores enabled Leiden resolutions in the same PCA or Harmony coordinates used to
build the graph, persists the decision as `run["cluster_selection"]`, and exposes the selected
Leiden candidate ref as `run["clusters"]`. Paris remains `run["paris"]` for diagnosis. Silhouette
supplies a reproducible baseline, not validation or ground truth. The agent uses this pipeline
for numerical execution, compares its measured alternatives, and pins its final chosen resolution.

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
resuming an interruption. Explicit questions require grounded answers through `resume_rna`.
Changed scientific inputs or Scarf procedure identity require a new run. Provider and operational
settings may change for unfinished decisions, with the new invocation recorded.

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
