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
ledger under `pipeline/runs`; it does not write live metadata. The ledger starts stages in recipe
order and can run one on a worker thread (`utils.background`) while later stages run; it writes
every record and callback from the calling thread. DataStore-owned plotting, marker
loading, and export consume narrow frozen-run views. Completed runs can be reopened by their
immutable label or exact run ID.

`agent/` owns the optional single-RNA workflow. Its lazy root facade exposes `analyze_rna`,
`AutomatedWorkflowResult`, and `AnalysisError`. Standalone scientific agent contracts and runners
remain in their concrete packages: `data_enrichment`, `experimental_context`, and
`biological_interpretation`. `parameter_tuning` has no standalone runner; it owns the candidate
contracts, execution, diagnostics, and selection checks used by the orchestrator's RNA tuning
stage. Internal modules import concrete owners rather than the root facade. `agent/tools/`
contains only infrastructure shared by more than one agent.

The orchestration stage history is the sole owner of the immutable request, scientific evidence,
choices, checks, work reservations, recovery, and final artifact references. Checkpoints belong
to their exact stage inputs; they do not create a second workflow lifecycle. The result is a small
address that resolves this history. `report/` renders one replaceable analysis page from saved
evidence. It does not call a provider or recompute scientific results.

### Presentation

`plotting/` owns the reusable plotting APIs.
It has no import dependency on `datastore`.
The removed `scarf.plots`, `scarf.plotting._legacy`, and `DataStore.plot_*` APIs must not be restored.
New plots should return the established plotting result types, accept documented data contracts, and use narrow adapters instead of adding storage-path knowledge.

The single-RNA agent keeps its bounded final-map display in `agent/_plots.py`. This limited report
view reads exact final artifact references, samples only displayed coordinates and labels, and
returns the existing public `PlotResult` and provenance types. It introduces no core plotting API
or module-load dependency from plotting to datastore.

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
- Derived assays are published atomically. `add_grouped_assay` and `add_melded_assay` stage the
  assay under a `scarf:pending_assay` marker and remove it on failure. Counts written by
  `create_zarr_count_assay` carry `complete=False` until they are finalized. A pending group left
  by a hard kill blocks its name, and repack refuses it, until
  `DataStore.discard_interrupted_assay` removes it.
- Public cell filters apply the pipeline filtering rules. `filter_cells` and `auto_filter_cells`
  exclude rows whose metric is missing and raise on missing sample labels among active cells,
  non-finite metrics used for automatic bounds, invalid bounds, duplicate attributes, and empty
  inputs or results. Selections over masked metadata columns record
  `missing_mask_fingerprints`; identities on unmasked stores are unchanged. Earlier selections
  over masked columns remain valid artifacts and must be recomputed.
- Cell-aligned artifact readers carry the linked missing mask. `select_cells`, groupings used by
  statistical testing and distribution plots, and integration metrics exclude or reject missing
  labels, and `run_doublet_detection` rejects clusterings with missing labels. These readers and
  pipeline filtering accept only the canonical `__scarf_missing__<name>` mask link.
- Query projections record the input `query_dataset_fingerprint`, which replaces
  `selected_expression_fingerprint`. A query cell with no counts in any shared reference feature
  is uninformative. The diagnostic `zeroNormCellCount` is renamed `uninformativeCellCount`, and
  `queryScaledDispersion` uses shared features only. Older projections fail to load with a
  request to rerun `run_mapping`. `array_hash` and `array_store_hash` use a length-prefixed
  encoding, so their values change.
- Mann-Whitney uses the exact permutation null when the two groups can be formed in at most
  100,000 ways. Its statistical-test artifacts record `p_value_policy`, and results carry
  `p_value_method`; identities of other tests are unchanged. Saved Mann-Whitney results without
  `p_value_method` fail to load with a request to recompute. A `StudyDesign`
  pairing column applies only to the paired Wilcoxon test.
- densMAP embeddings symmetrize neighbor distances and record `densmap_algorithm_version`, so
  earlier densMAP artifacts are not reused. Standard UMAP identities are unchanged.
- Clustering inputs are canonical and strict. `run_leiden_clustering` records resolutions as finite
  positive floats and requires a non-negative integer `random_seed`. Paris is never refitted
  silently: a reused hierarchy that cannot be read raises
  `ArtifactResolutionError(code="corrupt_payload")`, hierarchies missing required arrays or
  attributes and cuts whose diagnostics do not match the schema are not reused, and
  `load_paris_clustering` rejects such cuts with the same error. A
  fixed Paris cut needs `n_clusters` of 1 or at least the number of connected components; merges
  tied at the cut height are applied in hierarchy order, so exactly `n_clusters` clusters are
  returned. `run_topacedo_sampler` accepts `use_k` from 2 to the graph's `k`, and `use_k` equal to
  `k` shares the default identity. Clustering, sampling, and doublet entry points reject graph
  references that are not connectivity maps or integrated graphs.
- `get_markers` returns string `group_id` values in plot category order (numeric labels first in
  numeric order) and raises for an unknown group. `export_markers_to_csv` uses the same column
  order.
- `run_waggr` and `run_aucell` take `ambiguous_targets="drop"` and record
  `dropped_ambiguous_targets`. Enrichment artifacts written before this change load but are not
  reused.
- `scarf.metrics.silhouette_scoring` and `process_cluster` take no positional `ann_obj` and no
  `data_is_reduced` keyword. `silhouette_scoring` requires the keyword-only `distance_metric` and
  compares rows as given.
- `run_fate_mapping` treats `solver_tol` as an absolute bound on the largest Bellman residual of
  each solved sink column. Existing fate artifacts remain valid and are reused.
- Producers that cannot reuse a saved result raise `PermissionError` before computing on a
  read-only store. This covers statistical testing, enrichment, feature percentages, HTO
  demultiplexing, doublet detection, and `set_default_assay`.
- Public arguments are validated before any artifact is written: `load_graph(use_k=...)`,
  `run_lsi` `skip_first` and `rand_state`, `run_custom_reduction` loadings, `run_harmony`
  parameters, `integrate_assays(chunk_size=...)`, `select_hvgs` keywords, `run_umap` array
  initializations, and `make_bulk` column names.
- Library-size and CLR normalization promote integer counts to float64 before scaling or taking
  logarithms, so uint8 and int8 stores no longer raise and uint16 and int16 stores no longer wrap.
  Artifacts computed through `normed` from integer counts narrower than 32 bits record
  `count_arithmetic="float64"`, so earlier results on those stores are not reused. This covers
  `run_normalization` payloads for RNA with `renormalize_subset=False` and for CLR, marker tables
  on the fallback path, pseudotime markers and aggregations that call `normed`, statistical tests
  of assay-normalized features, and cell-cycle scores computed through `normed`. Identities on
  32-bit, 64-bit, and floating-point stores and every pipeline artifact identity are unchanged.
  Grouped ADT assays built from narrow counts must be rebuilt.
- RNA `normed` without subset renormalization maps a zero library total to 1, so zero-count cells
  normalize to 0 instead of NaN. Normalizations written earlier with `renormalize_subset=False`
  over selections that contain zero-count cells keep their identity and their NaN rows; PCA rejects
  them, and `invalidate_cache=True` recomputes them.
- Operations trust that prepared counts and artifacts do not change during a call. Writing to
  prepared data in place is outside the contract and is not detected.

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
