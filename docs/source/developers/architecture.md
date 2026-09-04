(architecture)=
# Architecture

Scarf uses concrete domain packages and a one-way module-load dependency structure.
The public API is exposed through package facades, while reusable computation and storage mechanics live in focused implementation modules.

## Dependency direction

Module-load dependencies point toward the earlier layers in this list:

1. Foundation: `storage`, `matrix`, and `utils`
2. Data model: `metadata`, `assay`, and `graph`
3. Domain algorithms: `neighbors`, `embeddings`, `clustering`, `trajectory`, `metrics`, `features`, `quality_control`, and `mapping`
4. Import and export: `readers`, `writers`, and `merge`
5. Orchestration: `datastore` and `datastore._operations`
6. Presentation: `plotting`

### Root facade

The root `scarf` package is a public import facade.
It does not form another runtime layer and must not eagerly import the implementation graph.
Only modules listed by that facade are available as lazy root attributes.
Presentation and lower-level algorithm packages such as `plotting`, `clustering`, `neighbors`, and `trajectory` require explicit imports.
This keeps the root surface small and avoids loading optional presentation code.

### Plotting and local imports

`plotting` does not import `datastore`.
Unified plotting consumes a narrow datastore adapter instead of resolving projection paths.
Existing heatmap functions still read Zarr-backed values from their duck-typed `store` and `assay` inputs.
Removing that remaining storage coupling is deferred.

Function-local imports may cross toward presentation when a public method explicitly requests a plot.
Examples include RNA feature selection and datastore quality-control helpers.
These calls do not create module-load cycles.

## Package responsibilities

### Foundation

- `storage/` owns stores, layouts, schemas, arrays, sharding, copying, resource budgets, storage
  profiles, materialization, ANN persistence, selection snapshots, run/stage records, Zarr runtime
  guards, and artifact lineage reports.
- `matrix/` owns the lazy blockwise matrix abstraction used over NumPy and Zarr arrays.
  Its arithmetic, indexing, and reduction behavior keeps it separate from low-level storage mechanics.
- `utils/` owns generic array, compute, logging, prefetch, process, and progress helpers.
  Zarr-specific helpers belong in `storage`, not `utils`.

Facade aliases do not change implementation ownership.
`scarf.utils.load_zarr` remains available for compatibility, but its implementation belongs to `storage.stores`.
Column prefetch uses `storage.parallel` because read-ahead limits and I/O concurrency are governed by the active storage budget.

### Data model

- `metadata/` owns Zarr-backed metadata tables, row streaming, and table queries.
  It is shared by datastore cell metadata and assay feature metadata, so neither `datastore`, `assay`, nor `storage` owns it.
  Shared value-selection contracts also live here so domain, orchestration, and presentation code can use one typed contract without reversing dependencies.
- `assay/` owns normalization, blockwise feature-summary computation, and the RNA, ATAC, and ADT assay types.
  `DataStore` owns planning and persistence of feature-summary artifacts; a bare `Assay.score_features` remains computation-only.
- `graph/` owns graph feature projection through named artifact inputs and rejects encoded-path inputs.
  Analysis execution follows explicit artifact references and must not resolve inputs by parsing
  encoded paths or choosing an implicit result.

Data-model modules may call domain algorithms from the method that needs them.
They must not import those packages at module load time.

### Domain algorithms

- `neighbors/` owns ANN construction, KNN queries, graph operations, diffusion, and weighted-neighbor integration.
  It does not own stored KNN graph arrays; persistence lives in `datastore` and `storage`.
- `embeddings/` owns PCA, LSI, Harmony correction, UMAP, SG-tSNE, embedding initialization, and
  the narrow storage adapter for imported coordinate artifacts.
- `clustering/` owns Leiden clustering and PARIS hierarchy operations.
- `trajectory/` owns pseudotime scoring, feature-profile aggregation, feature module clustering, and pseudotime result records.
- `metrics/` owns LISI, silhouette, graph, concordance, and integration scores.
- `features/` owns variability selection, LOWESS trend fitting, feature scoring, enrichment, rank and regression marker searches, GFF parsing, genomic intervals, and coordinate-based feature construction.
  It also owns presentation-independent feature resolution and normalized value fetching used by datastore workflows and plots.
- `quality_control/` owns filtering, HTO demultiplexing, doublet processing, cell-cycle assignment, and the default cell-cycle gene references.
- `mapping/` owns reference artifacts, feature alignment, confidence, Symphony-style correction, and mapping results.

Domain algorithm packages must not import `datastore`, `plotting`, or general import/export packages at module load time.
A domain that persists an artifact may use a narrow, named `storage` adapter.

### Import and export

- `cytebase/` lists, downloads, and opens public datasets.
- `readers/` parses supported input formats.
- `writers/` materializes Scarf stores and exports supported formats.
- `merge/` combines assays and datasets without importing `DataStore` during normal module loading.

Readers parse, writers materialize, and merge combines.
Format-specific code belongs in a module named for that format.

### Orchestration

`DataStore` remains the primary workflow API.
Its public class chain is kept for compatibility:

```text
BaseDataStore
  -> GraphDataStore
    -> MappingDatastore
      -> DataStore
```

Method implementations are grouped by responsibility under `datastore._operations`:

```text
graph
embeddings
clustering
trajectory
mapping
mapping_reference
quality_control
features
integration_metrics
presentation
```

Shared helpers under the same package include `enrichment_store` and `paris_persistence`.
Operation mixins have no runtime inheritance from datastore facades, no `__init__`, and no runtime imports of sibling operation mixins.
`TYPE_CHECKING` imports of siblings are allowed.
Reusable algorithms must be placed in their domain package before being exposed through a datastore method.

`datastore.pipeline_accessor` orchestrates the fixed basic RNA recipe. Focused internal modules own
recipe validation, run/stage ledger bookkeeping, filtering, frozen field assembly, and cluster
decision persistence. The reusable bounded silhouette comparison lives in
`metrics.cluster_selection`; its datastore adapter validates graph-coordinate lineage and persists
the immutable decision. `metrics.cluster_selection` is not part of the public `scarf.metrics`
facade. `datastore.pipeline_run` exposes the narrow durable `PipelineRun` handle and its frozen
cell and feature views. Pipeline execution creates immutable artifacts and a strict run/stage
ledger under `pipeline/runs`; it does not write live metadata. DataStore-owned plotting, marker
loading, and export consume narrow frozen-run views. Completed runs can be reopened by their
immutable label or exact run ID.

### Presentation

`plotting/` is the only plotting package.
It has no import dependency on `datastore`.
The removed `scarf.plots`, `scarf.plotting._legacy`, and `DataStore.plot_*` APIs must not be restored.
New plots should return the established plotting result types, accept documented data contracts, and use narrow adapters instead of adding storage-path knowledge.

`DataStore.plots` is a thin, store-bound accessor over the canonical store-first functions in `scarf.plotting`.
The accessor imports concrete plot implementations only when a method is called, so this convenience namespace does not reverse the dependency from plotting to datastore.

## Public facade policy

### Lazy and eager facades

The lazy facades in `scarf`, `features`, `readers`, `writers`, `merge`, `utils`, `neighbors`, `clustering`, `embeddings`, `trajectory`, and `plotting` are architectural boundaries, not temporary deprecation shims.
Their documented 1.x exports preserve stable import paths and defer optional or expensive implementations until an export is accessed.

The `assay`, `mapping`, `matrix`, `metadata`, `metrics`, and `quality_control` package initializers are eager domain facades.
API reference pages and public contract tests define which of their exports carry compatibility guarantees.

Reloading a lazy facade clears cached exports before resolving them again.
This keeps reload behavior deterministic for tests and interactive work.

Private facade exports used by repository tests are patch seams, not additions to the documented user API.
New production code should import its canonical implementation directly unless it intentionally needs a public patch seam.

### Breaking-release compatibility policy

This release intentionally has no compatibility bridge for the previous live-analysis contract.
The complete hard-break inventory is:

- `AssayState` and `IncompatibleAnalysisStateError` are removed. A store containing
  `{assay}/state` is rejected on open and must be rebuilt. Scarf never reads, migrates, or uses that
  group to choose a current result.
- `DataStore.pipeline.run()` accepts only the documented recipe options and returns a durable
  `PipelineRun`. Removed options and prior return values have no aliases or adapters.
- Feature selection, graph construction, embeddings, clusterings, scores, markers, mapping, and
  trajectory operations exchange exact `ArtifactRef` values. Consumers do not parse encoded
  metadata names, resolve an implicit latest result, or accept a live result column in place of an
  artifact.
- Analysis producers do not rewrite live `I` columns and do not insert clustering, UMAP, score, or
  marker columns. Callers use artifact loaders, frozen run views, and plotting adapters instead.
- Public result records use their current artifact-based constructors. Older positional layouts
  and field sets are unsupported.
- Pipeline run and stage records are strict, exact, and unversioned. Adding, removing, or renaming
  a persisted field in a later release is an accepted hard break. Unknown or incomplete document
  shapes fail closed.
- Mapping references and query projections use only their current exact-lineage contracts.
- Integration label metrics are split by input contract. `metric_clisi` and
  `metric_graph_connectivity` use the keyword `annotation_column` for imported cell metadata;
  `metric_label_concordance(first, second, metric=...)` compares exact clustering artifacts.
  Their former `label_colname` keywords and column- or array-based concordance inputs are
  unsupported.

Compatibility exists only where a current public facade or an explicit file-schema test says it
does. There are no silent migrations, implicit compatibility branches, or forwarding shims for
retired internal modules. Incompatible stores and artifacts fail with an actionable error.

## Placement rules

Use these rules when adding code:

1. Put Zarr mechanics in `storage`, blockwise matrix behavior in `matrix`, and generic operational helpers in `utils`.
2. Keep metadata and assay focused on table access, normalization, and persistence.
3. Put reusable computation in a concrete domain package.
4. Keep domain packages independent of datastore and plotting.
5. Use a named storage adapter when a domain persists an artifact.
6. Put parsing in readers, materialization in writers, and combination in merge.
7. Keep plotting free of datastore imports and use narrow adapters for new storage-backed inputs.
8. Add compatibility only at an existing public facade.
9. Use a concrete biological or computational package name.
   Do not introduce catch-all packages such as `core` or `analysis`.

Architecture boundaries are enforced in `tests/test_import_architecture.py`.
Public imports, result records, facade behavior, and wheel contents have separate contract tests.

## Accepted and deferred decisions

### Accepted

- The datastore class chain remains for public compatibility.
- Same-path package facades remain part of the public architecture.
- A small set of domain modules has narrow storage dependencies for persisted artifacts.
- Marker statistics and genomic feature construction live under `features`.
- Pseudotime-specific feature aggregation and module clustering live under `trajectory.feature_dynamics`.
- `metadata` remains a root data-model package because assays and datastore orchestration both depend on it.
- Unified plotting uses a datastore adapter instead of reading Zarr paths.
- Store-backed plotting is available through the lazy `DataStore.plots` accessor without moving implementation ownership out of `plotting`.
- Old flat compatibility modules remain deleted.

### Deferred

Deferred to a later structural phase:

- The graph and mapping operation modules remain large.
  Splitting them requires a separate behavioral and performance gate.
- `metadata` and `assay` retain established convenience methods, with their domain imports deferred to call time.
- Function-local cycles inside `assay` and `readers` remain accepted because there are no module-load cycles.
- Heatmap plotting still reads Zarr-backed values from duck-typed store and assay inputs.
  Replacing those reads requires a separate plotting adapter design.

### Rejected

- A vague `core` or `analysis` package.
- Restoring forwarding modules for retired private import paths.
- Restoring legacy plotting modules or datastore plotting methods.
- Moving storage-aware algorithms into datastore orchestration.

## Implementation references

Read {doc}`zarr_internals` for the on-disk implementation boundary and the public API reference for current contracts.
Use {doc}`contributing` for the test, documentation, and review workflow before making a change.
