---
description: Use Scarf safely in an autonomous or AI-assisted single-cell analysis.
---

(analysis_with_agents)=
# Analysis with AI agents

This page is a routing and reasoning guide for an AI agent that uses Scarf to analyse data.
It does not replace the workflow tutorials or define one correct analysis.
The study question, experimental design, and user instructions remain authoritative.

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
Boolean {term}`cell key` and {term}`feat_key` columns define the exact rows used by an operation.
Filtering changes a selection rather than deleting counts.

Persisted results are immutable {term}`artifacts <artifact>`.
Their provenance records the operation, scientific parameters, and input artifacts.
The selected {py:class}`~scarf.AssayState` identifies the current normalization, reduction, neighbourhood graph, and related results for an assay.

### Branches and mounts

Alternative results can coexist.
Where a granular method supports it, pass `update_state=False`, retain its returned {py:class}`~scarf.ArtifactRef`, and pass that reference explicitly to the next operation.
Select a branch as current only after evaluating it.
See {doc}`tutorials/custom_analyses` for a complete example and {doc}`concepts/provenance` for the storage model.

Opening a mounted store with another workspace name does not create a separate analysis.
A mount records one source workspace, and reopening that target with a different workspace is rejected.
Create a separate mount target for each analysis that needs independent metadata and results.
Use artifact branches within one target when only parameters or downstream methods differ.
Mount behavior is documented in {doc}`tutorials/remote_stores`; general layout is documented in {doc}`tutorials/data_organization`.

## Start or resume safely

### Inspect the current store

Start with `snapshot = ds.summary()`.
It reports the workspace, default assay, resource budget, metadata column names, assay state, and complete or incomplete artifact inventory without exposing a store location.
`active_cells` and each assay's `active_features` count the literal `I` columns.
They do not follow another `cell_key` or `feat_key` selected in `AssayState`.
Inspect and count those columns separately when evaluating a branch.
Use `snapshot.to_dict()` for a deterministic JSON-safe record.

Before choosing the next operation:

1. Identify the assays and the current default assay.
2. Inspect cell and feature metadata columns, including active selections.
3. Read `ds.get_assay_state()` for the selected chain.
4. Use `ds.list_artifacts(complete_only=True)` to find persisted alternatives.
5. Use `ds.inspect_artifact(ref)` before consuming an explicit result.
6. Use `ds.lineage(ref)` to verify upstream inputs.

`lineage.to_markdown()` and `lineage.to_mermaid()` produce compact provenance records for an analysis log.
Do not infer state by reading private Zarr paths.
Use public methods instead of mutating `ds.z`, `ds.zw`, or assay-state attributes.

Inspect unfamiliar H5AD files with {py:func}`scarf.inspect_h5ad` before creating a reader.
This reports matrix candidates, encodings, dimensions, and suggested assays instead of relying on guessed keys.
Use the format-specific import guides for other inputs.

### Prospective execution record

Before the first mutating operation, make a short execution record containing:

- the scientific question and unit of inference;
- the selected cell and feature keys plus input artifact references;
- the operations that will publish state, metadata columns, or other persisted results;
- the alternatives and independent evidence that will be compared;
- the criteria for selecting a branch, preserving uncertainty, or stopping.

Update this record before changing the cohort, inputs, or decision criteria.
This prospective boundary makes unintended writes and retrospective justifications visible.

### When to use the pipeline

`ds.pipeline.run()` is a mutating baseline workflow.
It has no `update_state=False` mode, publishes its graph as current, and selects a clustering partition for downstream marker and doublet stages.
When several partitions are available, that selection uses the highest silhouette score.
The selected labels are also copied to `{assay}_clusters`.
Use granular methods and explicit artifact references for branch comparison.
Use the pipeline only when those state and metadata writes are intended, and treat its selected partition as provisional until independent evidence supports it.

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
- Use SNN or WNN for modalities measured in the same cells, not independent batches.

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
A pseudotime or fate field is a model-based summary, not evidence of causal lineage.

### Methods outside Scarf

Use {doc}`tutorials/custom_analyses` for block streams, graph access, custom selections, external reductions, and supported export paths.
Export when the question needs RNA velocity, scVI, peak calling, FRiP/TSS, or another method outside Scarf.
Do not write arbitrary artifact groups directly.

## Troubleshooting

Classify the problem before retrying:

- **Input or schema:** inspect the source format and verify assay, feature, and cell identifiers.
- **State or provenance:** inspect the selected `AssayState`, explicit artifact status, selection keys, and lineage.
  `ArtifactSelectionError.code` distinguishes missing, incomplete, or changed selection inputs.
  Do not consume incomplete artifacts.
- **Resource or I/O:** inspect the configured memory budget, worker count, storage profile, and remote latency.
  If an RNA store fails to open, treat it as a missing or outdated count layout and rebuild or `repack_zarr` rather than retrying the analysis stage.
  See {doc}`concepts/memory_and_execution`, {doc}`concepts/benchmarks`, and {doc}`tutorials/remote_stores`.
- **Numerical or graph:** check dimensions, graph components, neighbour count, convergence, validity masks, and method-specific diagnostics.
- **Scientific ambiguity:** preserve branches, seek another independent form of evidence, narrow the claim, or report that the available design does not resolve the alternatives.

Retry the lowest failed stage rather than restarting the complete workflow.
Identical valid requests reuse existing artifacts.

## Progress and deterministic comparisons

`ds.pipeline.run(..., callback=...)` emits {py:class}`~scarf.datastore.pipeline_accessor.PipelineEvent` values when stages start, complete, or fail.
Use the callback to update an external progress record; callback failures do not control the pipeline.
Granular methods continue to return their normal result records or artifact references.

When comparing branches, retain the same cells, features, neighbour settings, and seeds unless one is the variable being tested.
Record every deliberate difference.
A layout can vary in orientation or spacing without representing different biology.

## Analysis handoff

A useful handoff reports:

- the scientific question and unit of inference;
- the store workspace, active cell and feature keys, and relevant artifact refs;
- metadata roles and any confounding;
- alternatives considered and the evidence used to compare them;
- the selected result and why it is fit for the question;
- unresolved uncertainty, unsupported claims, and required external validation;
- exported files or external methods that continue the analysis.

Artifact provenance records how Scarf produced a result.
It does not replace this study-level reasoning record.
See {doc}`index` for the implemented methods and current boundaries.
