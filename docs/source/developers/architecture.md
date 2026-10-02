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
- `utils/` owns generic array, argument validation, compute, logging, process, and progress helpers.
  Zarr-specific helpers belong in `storage`, not `utils`.

Facade aliases do not change implementation ownership.
`scarf.load_zarr` is a facade alias for `storage.stores.load_zarr`.

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
- `features/` owns variability selection, LOWESS trend fitting, feature scoring, enrichment, rank and regression marker searches, GFF parsing, genomic intervals, coordinate-based feature construction, and the name-based gene-family registry.
  It also owns presentation-independent feature resolution and normalized value fetching used by datastore workflows and plots.
- `quality_control/` owns filtering, HTO demultiplexing, doublet processing, cell-cycle assignment, and the default cell-cycle gene references.
- `mapping/` owns reference artifacts, feature alignment, confidence, Symphony-style correction, label transfer, and mapping results.

Domain algorithm packages must not import `datastore`, `plotting`, or general import/export packages at module load time.
A domain that persists an artifact may use a narrow, named `storage` adapter.

### Import and export

- `cytebase/` discovers datasets and connects to their Scarf datastores. `DatasetEntry` owns
  catalog metadata and source-record descriptions. `Catalog.open_datastore` and
  `Catalog.mount_datastore` return `DataStore` objects; module-level embedding helpers resolve
  imported artifacts from those objects. Cell metadata and plotting use the existing `DataStore`
  APIs. Repository helpers list and download public example datasets.
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
ledger under `pipeline/runs`; it does not write live metadata. The ledger runs stages one at a
time in recipe order on the calling thread. DataStore-owned plotting, marker
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

The lazy facades in `scarf`, `agent`, `features` and its subpackages, `readers`, `writers`, `merge`, `utils`, `neighbors`, `clustering`, `embeddings`, `trajectory`, and `plotting` are architectural boundaries, not temporary deprecation shims.
Their documented 1.x exports preserve stable import paths and defer optional or expensive implementations until an export is accessed.
All of them use one private helper, `scarf._facade`: an export wins over a same-named submodule, resolving an export never replaces a bound value, and only `readers`, `writers`, `merge`, and `utils` present the objects they own under the facade's `__module__`.

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
- Pipeline stages run strictly in sequence. UMAP no longer runs on a worker thread beside the
  Leiden, Paris, cluster-selection, and membership-strength stages; outputs and artifact
  identities are unchanged. Durable stage records now attribute wall time and sampled memory to
  one stage at a time, and starting a stage while an earlier stage of its run is incomplete
  raises `ValueError`. Pipeline callbacks receive an enabled stage's `stage_started` and terminal
  event before the next stage starts; earlier, UMAP's `stage_completed` arrived after the events
  of those stages, with or without a worker thread. Earlier run records whose stage windows
  overlap still read. `scarf.utils.background` is removed.
- `DataStore.pipeline.run` takes a `params` mapping of per-stage settings. Every run records two
  more stages, `membership_strength` and `tsne`, skipped unless requested, and its configuration
  records `params`, `species`, `tsne`, `membershipStrength`, and the Leiden `selected`
  resolution. `pca_dims=0` skips PCA.
- Mapping references and query projections use only their current exact-lineage contracts.
- Label transfer is a saved artifact. `get_target_classes` and `get_target_label_evidence` are
  removed. `run_label_transfer` freezes the reference labels it reads, from a reference column or a
  reference cell-label artifact, into the query datastore as a `reference_labels` artifact and saves
  a `label_transfer` artifact; `get_label_transfer` loads it without the reference. An abstention is
  a missing label with an `abstentionReason` rather than an `na_val` string, `max_distance` applies
  to the saved labels, and `target_subset` is removed. Evidence renames `isUnknown` to `abstained`
  and adds `candidateLabel` and `nearestDistance`. It no longer repeats the projection's
  `featureCoverage` and `queryScaledDispersion` on every row; read them from
  `get_mapping_result(...).diagnostics`. Conformal sets come from
  `LabelTransferResult.prediction_sets` instead of a `predictionSet` column. `mapping_evidence`,
  `mapping_confusion`, and `mapping_calibration` take the transfer ref instead of a projection,
  reference, and threshold. `mapping_calibration` keeps the transfer's saved rules except the one
  on the swept metric, and without a `chosen_threshold` it marks that rule's own threshold.
  `mapping_confusion` names the abstention column with `abstention_label`. `mapping_score` takes `reference_labels` in place of
  `reference_class_group`. `ExternalArtifactRef` gains `anchor_assay` for an artifact of
  another datastore that is not scoped to the assay its dataset fingerprint describes.
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
- Mounted targets resolve their source's artifacts read only. From this release on, every mount,
  including mounts created by earlier release candidates, lists, loads, traces, and reuses the
  complete artifacts of its identity-checked source after its own, so a recipe that matches saved
  provenance reuses the source's results instead of recomputing them, and labels and embeddings
  imported into the source, such as the Cytebase `X_umap`, are visible on the mount. Nothing is
  migrated because no record changes: `ArtifactRef` stays a location-free name with a random ID,
  provenance and run records are unchanged, and `ExternalArtifactRef` stays the cross-dataset
  identity. Every write goes to the target, and a write inside a source artifact group raises
  `PermissionError`. Pipeline runs and their labels stay per store. The `matrixSource` top level is
  exact (`location`, `workspace`, `assays`), so a mount record with any other field fails to open
  and asks for a fresh target. `repack_zarr` reads a mount through the same namespace and writes a
  self-contained store. A read-only open of an unprepared store, such as the source that
  `mount_datastore` opens, asks for one writable open instead of a rebuild. `local_cache` follows
  the store that holds the normalized artifact rather than the datastore location, so a local mount
  stages a normalized artifact that it reuses from a remote source and never one that the target
  holds; staging on stores that are not mounts is unchanged.
- Public cell filters apply the pipeline filtering rules. `filter_cells` and `auto_filter_cells`
  exclude rows whose metric is missing and raise on missing sample labels among active cells,
  non-finite metrics used for automatic bounds, invalid bounds, duplicate attributes, and empty
  inputs or results. Selections over masked metadata columns record
  `missing_mask_fingerprints`; identities on unmasked stores are unchanged. Earlier selections
  over masked columns remain valid artifacts and must be recomputed.
- Cell-aligned artifact readers carry the linked missing mask. `select_cells`, groupings used by
  statistical testing and distribution plots, and integration metrics exclude or reject missing
  labels, and `run_doublet_detection` rejects clusterings with missing labels. These readers,
  trajectory cell-data inputs, and pipeline filtering accept only the canonical
  `__scarf_missing__<name>` mask link.
- Readers show rows that a linked missing mask flags as missing. `MetaData.to_pandas_dataframe`
  and `head`, `get_cell_vals`, live `to_anndata` and `to_h5ad`, and plots show them as missing:
  numeric values become float64 NaN, and in H5AD files booleans become nullable booleans and text
  gets a missing category. `MetaData.fetch` and `fetch_all` stay raw. `make_bulk` leaves masked
  cells out of every group, dot and matrix plots report `dropped_group_cells`, and
  `cluster_connectivity` rejects masked inputs.
- `run_marker_search`, `calc_membership_strength`, `smart_label`, `get_imputed` of a metadata
  column, and `silhouette_scoring` of a metadata column reject masked inputs before any reuse or
  write, so results from unmasked inputs keep their identities. `calc_membership_strength` and
  `smart_label` accept only categorical label kinds.
- `DataStore.snapshot_cluster_labels(labels, cell_selection=...)` freezes a cell metadata column,
  or a cell-label artifact read for a subset of its cell selection, into a datastore-scoped
  `cluster_labels` artifact with operation `snapshot_cluster_labels`. Floating-point labels that
  are whole numbers within the int64 range, such as the float64 ids that pandas writes for integer
  ids with missing values, are stored as int64; other floating-point labels raise `TypeError`, and
  missing, masked, and blank labels raise `ValueError`. Text is stored at the width of the selected
  labels, and integer and boolean labels keep their dtype. The identity records `source_column` as
  a parameter or `source_labels` as an input, the cell selection, and the `values_fingerprint` of
  the stored labels, and a snapshot is reused only while its payload holds exactly those labels.
  The missing-label errors of `run_marker_search`, `calc_membership_strength`, and `smart_label`
  name this method and the cell selection that each consumer needs. The group-name error of
  `run_marker_search` also names this method, and only `run_marker_search` applies the group-name
  rule.
- Query projections record the input `query_dataset_fingerprint`, which replaces
  `selected_expression_fingerprint`. A query cell with no counts in any shared reference feature
  is uninformative. The diagnostic `zeroNormCellCount` is renamed `uninformativeCellCount`, and
  `queryScaledDispersion` uses shared features only. Older projections fail to load with a
  request to rerun `run_mapping`.
- Mann-Whitney uses the exact permutation null when the two groups can be formed in at most
  100,000 ways. Its statistical-test artifacts record `p_value_policy`, and results carry
  `p_value_method`; identities of other tests are unchanged. Saved Mann-Whitney results without
  `p_value_method` fail to load with a request to recompute. A `StudyDesign`
  pairing column applies only to the paired Wilcoxon test.
- densMAP embeddings symmetrize neighbor distances and record `densmap_algorithm_version`, so
  earlier densMAP artifacts are not reused.
- `run_umap` records float parameters as floats and integer parameters as Python integers, so
  `min_dist=1` and `min_dist=1.0` identify the same embedding. The defaults `min_dist=1` and
  `negative_sample_rate=5` were recorded as integers, so every saved UMAP embedding recomputes on
  its next run. Numeric parameters reject booleans and non-finite values, `umap_dims`, `n_epochs`,
  and `random_seed` must be integers, and `parallel` and `use_density_map` must be booleans.
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
  order. Only plain decimal labels are numeric, so plots no longer read `"1_10"` as 110; other
  labels sort naturally, with digit runs before text and case-only ties broken by the exact text.
- `run_waggr` and `run_aucell` take `ambiguous_targets="drop"` and record
  `dropped_ambiguous_targets`. Enrichment artifacts without `layout` or
  `dropped_ambiguous_targets` fail to load.
- `scarf.metrics.silhouette_scoring` and `process_cluster` take no positional `ann_obj` and no
  `data_is_reduced` keyword. `silhouette_scoring` requires the keyword-only `distance_metric` and
  compares rows as given.
- `run_fate_mapping` treats `solver_tol` as an absolute bound on the largest Bellman residual of
  each solved sink column. Existing fate artifacts remain valid and are reused.
- `scarf.neighbors.diffusion_operator` is removed; it formed a powered operator with no memory
  bound. `bounded_diffusion_operator` is the only powered builder, and
  `neighbors.diffusion.transition_matrix` returns the graph-sized single step. The removal
  changes no doublet score and no diffusion or pipeline artifact identity.
- Assays are no longer `DataStore` instance attributes. `ds.<name>` resolves an assay only when no
  `DataStore` attribute has that name and the name does not start with an underscore;
  `get_assay(name)` returns any assay. Assays named like members, such as `cells` or
  `zarr_mode`, no longer break opening or replace the member.
- Producers that cannot reuse a saved result raise `PermissionError` on a read-only store before
  computing or writing. This covers every artifact producer, including marker search,
  `smart_label`, membership strength, cell filters and selections, graph, embedding, trajectory,
  and mapping producers, and derived assays. `run_mapping` and `build_mapping_reference` refuse a
  read-only store before planning.
- Public arguments are validated before any artifact is written: `load_graph(use_k=...)`,
  `run_lsi` `skip_first` and `rand_state`, `run_custom_reduction` loadings, `run_harmony`
  parameters, `integrate_assays(chunk_size=...)`, `select_hvgs` keywords, `run_umap` array
  initializations, and `make_bulk` column names.
- Normalization arithmetic no longer depends on the count storage dtype. Library-size, CLR, and
  TF-IDF normalization compute in float64 from the counts and float64 totals for bool, integer, and
  floating-point counts, so bool, uint8, and int8 stores no longer raise and uint16 and int16 stores
  no longer wrap. A persisted normalized value, and a value that marker search ranks under
  `norm_lib_size`, with or without `renormalize_subset`, is the single float32 rounding of its
  float64 value; marker search ranks the values of every other normalization unrounded. Library
  sizes, subset totals, TF-IDF term totals, and feature percentages accumulate in float64
  (`ChunkedArray.sum` takes NumPy's `dtype` keyword). `Assay.score_features` with
  `log_transform=True` takes the logarithms of assays without a normalization (CRISPR, ANTIGEN,
  CUSTOM, and `Assay`) in float64 for every count dtype, so their scores no longer depend on the
  storage dtype; NumPy took them in float16 for uint8 counts and in float32 for uint16 and float32
  counts, which could bin different control features and change the scores. Sums over cells, such as
  CLR log means and feature summaries, combine partial sums of the stored row blocks or `countsT`
  cell bands, so their last float64 bits follow the count layout, and dtypes of different widths or
  imports at different memory budgets can get different layouts. `make_bulk` sums of every assay and
  synthetic doublet counts add integer counts in an integer dtype and floating-point counts in
  float64, so integral counts sum exactly and `make_bulk` sums on float stores are float64; non-RNA
  sums of float32 counts previously added in float32, so their values also change for non-integral
  counts or sums above 2**24. RNA `normed` maps a zero library total to 1, so zero-count cells
  normalize to 0 instead of NaN. Grouped assays are counts rather than artifacts, so nothing
  recomputes them: grouped RNA assays that earlier releases built can hold NaN for cells without
  counts or wrapped values from 16-bit counts, and grouped ADT assays can hold means of CLR values
  computed in float32 or float16. Rebuild finite ones with `add_grouped_assay` under a new
  `assay_label`. A store whose grouped RNA assay holds NaN cannot be rebuilt with `--data-only`,
  because count writers reject NaN; re-import its source and build the grouped assay there. No
  operation records `count_arithmetic`, `checked_integer_sum`, or `zero_total_divisor`. Artifacts
  that earlier release candidates recorded with one of these fields, which includes every doublet
  score, have different identities and are recomputed on request. `load_pseudotime_markers`,
  `load_pseudotime_aggregation`, and loading or building a mapping reference reject such records and
  ask for a recompute. Artifacts that earlier release candidates computed from float32 counts keep
  their identities although current values differ: subset-renormalized library-size values by at
  most one float32 ulp (a few ulps for non-integral counts or subset totals above 2**24), CLR values
  by the error of their float32 log sums (typically 3e-5 relative at 8,000 cells and 8e-4 at
  200,000), whole-library library-size values only for non-integral counts or counts whose float32
  product with the size factor is inexact (above 134,217 at the default size factor of 1000),
  feature percentages and subset-renormalized TF-IDF values only for non-integral counts or totals
  above 2**24, and every result derived from them. Pseudotime markers and aggregations that stream
  library-size values without subset renormalization compute them in float64 instead of float32 on
  every store, so their results change at float32 resolution while their identities stay the same.
  Statistical tests compare the fingerprints of their values and recompute by themselves; recompute
  the other artifacts with `invalidate_cache=True`. Stores written by unreleased development builds
  are unsupported.
- Library-size marker search ranks raw counts of every storage dtype with one zero-aware kernel,
  with or without `renormalize_subset`, so float, signed, and unsigned stores no longer take
  different kernels, and subset-renormalized searches no longer rank unrounded float64 `normed`
  values with the dense kernel. The kernel checks its input while it reads it: a negative or
  non-finite normalized value of a tested feature, or a negative or non-finite total of a selected
  cell (its `<assay>_nCounts` value, or with `renormalize_subset=True` its sum over the tested
  features), raises `ValueError` naming the feature or the totals, and nothing is written;
  subset-renormalized searches used to accept negative counts and totals. Rank statistics use
  float64 group sizes instead of float32, so p-values and their adjusted values change in their low
  digits once a group's size times its complement's exceeds 2**24 (two groups of about 4,100 cells;
  about 1e-4 relative at a million cells), and the two groups of a two-group search now get equal
  p-values. Marker tables keep their identities, layout, and attributes, but library-size statistics
  now come from float64 values rounded once to float32 instead of float32 arithmetic, so values can
  move at the fifth decimal; recompute earlier tables with `invalidate_cache=True`.
  `scarf.features.find_markers_by_rank` returns a `RankMarkerResult` of the sorted group ids, their
  sizes, the ascending feature index, and the features-by-groups rank statistics instead of one
  DataFrame per group, and raises `IndexError` for a cell or feature index past the end of `countsT`
  instead of ranking an unwritten column as a cell. A `RankMarkerResult` requires at least two
  groups of at least two cells and at least one feature, as a saved marker table does, and its
  `table` method gives the ranked table of one group, with the string `group_id`, that a saved
  marker table reads back. Feature-stream plans charge what Zarr holds while it reads a band
  (`ArrayGeometry.readBytes`) and the streams' band indices, and size read-group destinations by the
  selected cells, so tight budgets admit fewer reads in flight and budgets that fitted only
  uncharged read buffers raise `MemoryError`. The marker search also reserves its result, the stored
  tables its writers finish, and kernel scratch. Read-group streams process one group at a time in
  order while the next is read, and the library-size kernel runs every thread of that compute worker
  over the features of a group, so marker memory no longer grows with the worker count. The band
  reads of the two groups in flight share the requested read width,
  `StorageIoPolicy(readWorkers=...)` or eight reads per worker, which `WorkShape.maxUnitsInFlight`
  lets the planner split into inner reads, and an execution report names the cause in
  `reductionReason` when memory or the band count leaves fewer reads in flight.
  `map_feature_read_groups` loses its `orderedCompute` and `extraItemsize` arguments.
- `find_markers_by_regression`, which `run_pseudotime_marker_search` calls, scales the regressor and
  each feature's values by a power of two below one in magnitude before it sums them. Pearson r does
  not depend on that exact scale, so ordinary inputs give bit-identical results, but a regressor or
  feature values whose squared deviations overflowed float64 (magnitudes above about 1e150 to 1e154,
  depending on the number of cells) now give their correlation instead of r = 0 with p = 1, or NaN,
  and a regressor whose squared deviations underflowed (below about 1e-160) is tested instead of
  reported as untested. Two-cell searches report r of exactly 1 or -1 instead of a least-squares
  value that could round below one. Pseudotime marker identities are unchanged.
- Row-block streams over a sharded array reserve what Zarr holds while it reads a block: a
  shard-level copy of the selection, the compressed bytes of every chunk the block touches in a
  shard, charged at their decoded size, and the one chunk decoded at a time, next to the block and
  the output of its first operation. Each later operation of a chain holds the previous output
  beside its own, and a block reserves the larger of these steps, so row blocks of log-normalized
  RNA, CLR, and TF-IDF values, which hold two float64 outputs at once, reserve 16 bytes per value of
  uint8 or uint16 counts instead of 10 or 12. Normalization writes, row-block streams, and
  reductions over such values therefore run fewer blocks at once and raise `MemoryError` where only
  the smaller reservation fitted; single operations, and chains over counts of 32 bits or more,
  reserve what they did. `ArrayGeometry.readBytes` takes the number of chunks decoded at once and
  charges an unsharded read only its result and decoded chunks, and `plan_feature_stream` sizes the
  feature blocks of a sharded array by the same read. Normalization, PCA, LSI, graph construction,
  quality control, export, subset, materialization, melding, and merge plan or size their row blocks
  with these bytes, so a budget admits fewer blocks at once and raises `MemoryError` where only the
  uncharged read buffers fitted; outputs and identities are unchanged. A row-block stream keeps no
  block after it yields it, feature streams drop their own references to a unit before they release
  its reservation, and the storage runner's pool workers keep neither a call nor its result once it
  returns. HVG feature statistics reserve the selected cells' inverse totals, the outputs, every
  band's partial statistics, and each compute worker's band scratch, and no longer keep the totals
  beside their inverses. `make_bulk` means that stream normalized row blocks reserve their group
  sums and divide them in place.
- Sparse count imports that write several assays from one source, such as Cell Ranger, MTX, and H5AD
  files with antibody or peak features, no longer raise `MemoryError` when a band of a wider assay
  arrives while bands of a narrower assay are pending and every band fits on its own. The writer
  writes the longest run of pending bands that fits, so the earlier bands are written first.
- Storage operations run outside an event loop no longer chain their errors to a `RuntimeError: no
  running event loop`, and a storage read that its own I/O loop cancels raises one `CancelledError`
  instead of a group of two. `scarf.storage.async_execution.ensure_zarr_host_ceiling` loses its
  `maxWorkers` argument, which nothing passed, and the `write_counts_t` metrics drop
  `sourceRepeatedDecodeCount` and `sourceRepeatedDecodeBytes`, which paired count layouts always
  left at 0 because every `countsT` shard holds whole `counts` chunks.
- ATAC `normed` no longer leaves its fitted TF-IDF state (`n_term_per_doc`, `n_docs`,
  `n_docs_per_term`) on the assay after the call, and RNA `normed` never changes `normMethod`.
  A `DataStore` is not designed for concurrent use from several threads.
- Count storage dtype: each assay's counts are stored unsigned, in the narrowest of uint8, uint16,
  uint32, and uint64 that holds the assay's largest value, exactly when every canonical
  (duplicate-summed) value of the assay is a non-negative integer. Other counts keep their source
  dtype: float32 or float64, or the signed integer dtype of a source with negative values. float16
  is not a count storage dtype; count writers reject it, and H5AD imports read float16 sources as
  float32. `scarf.storage.count_dtype` holds the one policy, and `scarf.utils.count_values` the
  value scans that readers and writers share; a scan takes the group of each feature and returns one
  range per group. Every import applies the policy to every encoding: H5AD, 10x HDF5 and Matrix
  Market (`CrToZarr` and `MtxToZarr`), CSV, `SparseToZarr`, and `SeuratToZarr`. Integral float,
  signed, and wider unsigned sources, unsorted or duplicate coordinates, CSC, and dense matrices
  therefore store the same dtype and content fingerprint at any memory budget, and 10x HDF5 and
  Matrix Market imports, which stored uint32, and Seurat imports, which kept float64, store the
  narrowest unsigned dtype of each assay. Imports that split one source matrix into assays (10x HDF5
  and Matrix Market feature types, H5AD `assay_split_key`) resolve each assay's dtype from its own
  features, so an assay's dtype and identity do not depend on the other assays of the source: an
  antibody, guide, or ATAC assay stores its own narrowest dtype beside RNA counts that need a wider
  one, and integral assays stay unsigned beside an assay with fractional or negative values. The
  count dtype arguments are removed: `CrToZarr` and `MtxToZarr` `dtype`, `MtxReader` `dtype` and its
  `consume` `dtype`, `CSVtoZarr` `dtype`, `SparseToZarr` `matrix_dtype`, and `DataStoreMerge`
  `dtype` (merge manifests no longer record it); `create_zarr_count_assay`,
  `create_empty_zarr_count_assay`, and `DerivedAssayTransaction.create_counts` require the dtype.
  `CrReader` subclasses implement `matrix_dtype` and `count_value_ranges(maxBytes,
  featureGroups=None)`, which returns the range of each group of features over the selected cells,
  and `H5adReader` has the same method. `MtxReader` scans the matrix once at construction for every
  orientation and filter mode, which replaces its coordinate-order probe, and keeps the largest
  count of each feature; it reads the file once more only when it dropped cells and an import splits
  its features into several assays. Its batches hold the counts of the kept cells in the narrowest
  unsigned dtype that holds them. `H5adReader` loses its `dtype` argument and its `matrixDtype`,
  `storageDtype`, and `infer_storage_dtype` members, and `H5adToZarr.storageDtype` becomes
  `storageDtypes`, the dtype of each imported assay; `consume` yields `sourceMatrixDtype` values for
  every encoding, except that the converted rows of a CSC integer source hold their duplicate-summed
  values in int64 or uint64 (`consumeDtype`), so duplicate sums past a narrow source dtype import as
  they do from CSR. Seurat count sources compressed over features, transposed, or stitched from
  Assay5 layers keep the duplicate coordinates of a cell in the source dtype, as sources compressed
  over cells do, and the writer sums them in 64 bits: such duplicates no longer raise OverflowError,
  and Assay5 duplicates whose sum exceeds a narrow source dtype, which were stored wrapped, store
  their sums. Subset and repack keep the source dtype, because they rebuild an existing dataset
  whose identity and copied artifacts must stay valid. Merges store the common type of the source
  count dtypes, widened when features summed by name could overflow it, instead of float64 for
  differing dtypes, and reject integer sources without a common integer dtype; an assay without
  features in any source stores uint8 instead of uint32. Derived assays (grouped and melded) keep
  their float64 values. Dense writers (CSV and dense Seurat counts) no longer cast batches before
  the checked cast, so a count that the stored dtype cannot hold raises OverflowError instead of
  wrapping. The Cytebase build no longer forces the source dtype, and its records drop
  `storageDtypePolicy`. Count matrices hold finite values: every count writer, including subset,
  merge, repack, and derived assays, rejects NaN and infinity, and the H5AD, 10x HDF5, Matrix
  Market, CSV, sparse, and Seurat imports read every value and reject them before they create the
  destination. Earlier stores whose counts hold them, including a store whose grouped RNA assay an
  earlier release built with NaN rows for cells without counts, can no longer be subset, merged, or
  repacked, even with `--data-only`; re-import their sources. Negative values stay storable.
- Operations trust that prepared counts and artifacts do not change during a call. Writing to
  prepared data in place is outside the contract and is not detected.
- Minimum versions rise to scipy 1.15, statsmodels 0.14.5 (earlier releases fail to import
  with scipy 1.16), threadpoolctl 3.5 (earlier releases cannot see the OpenBLAS in NumPy and
  SciPy wheels, so BLAS thread limits had no effect), huggingface-hub 2.0, and, for the `agent`
  and `test` extras, pydantic-ai-slim 2.51.
- Count layout: plans whose countsT chunks fell below half the chunk target (awkward cell counts
  such as primes) now use whole-target chunks, so stores written with those plans fail layout replay
  and must be re-imported. Count assays require Zarr format 3. Every count writer fits the layout to
  `mem_budget` before it creates its destination: `H5adToZarr`, `CrToZarr` and `MtxToZarr`,
  `CSVtoZarr`, `SparseToZarr`, `SeuratToZarr`, `SubsetZarr`, `subset_assay_zarr`, `repack_zarr`,
  `DataStoreMerge`, and `add_grouped_assay`, which wrote the default layout at every budget. Without
  a `policy`, a writer halves the default `unitBytes` and `chunkBytes` together until the counts
  write and the countsT transpose fit, and a write that does not fit with one-row count shards, or
  with its explicit `policy`, raises MemoryError before the destination exists. Sparse writers admit
  their sparse band writes and dense writers their dense row bands. Writers that choose their source
  batches (the sparse imports, the Seurat imports, and merge) admit batches of one destination row
  band, the batch their write starts from, so a fitted layout never starves the write to narrower
  batches; only one-row shards get one-row batches. `add_melded_assay` keeps sizing its shards to
  the melding band that fits `mem_budget`. The writers raise from their constructors, and
  `DataStoreMerge` from `plan`, instead of from `dump`; `SeuratToZarr` construction also prepares
  every selected source and reads its counts once, so source preparation errors surface there.
  `CrToZarr` and `MtxToZarr` take `lines_in_mem` in the constructor, and `dump` no longer accepts
  it, so the fit reserves the Matrix Market parse buffer that the write uses and a smaller buffer
  fits a smaller budget. An explicit `dump(batch_size=...)` reads at most one destination row band
  per batch, so it can no longer exceed what the fit admitted. Writers that write their assays one
  at a time (Seurat, subset, repack, and merge) fit each assay on its own. The fitted layout depends
  only on the budget, the data, the dtypes, and for Matrix Market imports `lines_in_mem`, never on
  the worker count. A resumed merge keeps the layout persisted with its completed counts, so a
  budget change between attempts cannot block it, and fits the layout of the counts it rewrites.
  Sparse imports admit the producer's buffering and the band writes as separate phases, so budgets
  that the summed plan refused now import. `write_counts_t`, `finalize_writer_counts_t`,
  `finalize_writer_counts_t_many`, and the merge writer's `write_assay_counts_t` lose their `policy`
  argument and replay the persisted layout. A `repack_zarr` that fails after it creates its
  destination removes it, so a failed copy, such as an unreadable label claim copied last, no longer
  leaves a store that opens without all its data. Identity does not depend on the layout: content,
  counts, and dataset fingerprints are unchanged, and only the layout fingerprint in
  `scarf:countMatrixLayout` differs. Existing stores are never rewritten and keep their dtype,
  layout, and identity. A re-import can store a different dtype and therefore get different counts
  and dataset fingerprints and new artifact ids; mounts and mapping references bound to a replaced
  store fail closed.
- `SubsetZarr` loses `overwrite_cell_data`. The parameter had no effect: the constructor always
  opens the destination empty, so a subset never found cell data to keep or replace. Passing it now
  raises TypeError. Explicit `cell_idx` values must be distinct. A repeated index previously wrote a
  store whose cell IDs repeat, and `DataStore` opened that store; it now raises ValueError before
  the destination is created or overwritten. Explicit `cell_idx` values must also be non-empty and
  non-negative: an empty index failed inside Python's `max()`, and a negative index wrapped around
  to a cell counted from the end, so `[0, -n]` repeated the first cell past the distinct-index
  check; both now raise ValueError before the destination is created or overwritten. A `cell_key`
  that selects no cells still writes a subset without cells. `subset_assay_zarr` applies the same
  rules to `cells_idx` and `feat_idx` (one-dimensional distinct in-range integers, raising
  IndexError, ValueError, or TypeError) and requires at least one feature, before it creates
  `out_grp`; it wrapped negative indices and repeated duplicates before. The docstrings of
  `H5adToZarr.dump`, `CrToZarr.dump` (and `MtxToZarr.dump`), `SparseToZarr`, and `CSVtoZarr.dump` no
  longer list an AssertionError for a row-count mismatch, because that check could not fail. The
  shard writers raise ValueError for a count stream with more or fewer rows than the destination, or
  for a batch of another width. A CSV file that changes after the reader's pass reports this
  ValueError, which `CSVtoZarr.dump` now documents.
- Stored contracts are strict. ANN indexes carry their complete metadata record including
  `byte_length`; `query_neighbors` requires recorded `ann_ef` and `parallel_threads`; mapping
  references require `ann_ef`; building and loading a Symphony mapping reference require recorded
  Harmony `batch_levels`, `batch_columns`, and `harmony_parameters`. HVG selections record
  `blacklist_fingerprint` and, for adaptive binning, `variance_estimator` and `variance_quantile`.
  Statistical-test artifacts missing any recorded attribute fail to load. Earlier artifacts
  without these records fail to load or are rebuilt.
- Metadata copies, column clearing, and run snapshots accept only the canonical
  `__scarf_missing__<name>` link and reject multi-dimensional columns. Copies and snapshots store
  text as unicode and text fingerprints decode bytes as UTF-8, so identities over byte-string
  columns change. `MetaData.columns` lists `I`, `ids`, `names`, then the other columns sorted.
  `MetaData.sift`, `multi_sift`, and covariate partitions treat masked rows as missing, and
  `insert` keeps an explicit boolean `fill_value`. `MetaData.get_index_by` matches values that
  are not text by their text and always returns int64 indices. `MetaData.insert` rejects names
  that are empty, `.` or `..`, contain `/` or `\`, because Zarr would nest them into groups, or
  start with the `__scarf_missing__` mask prefix; `reset_key` and `update_key` apply the same
  rule. A lookup of a name with a separator suggests the `_` spelling that imports store, and a
  lookup of an empty, `.` or `..` name raises `KeyError`. A cell or feature table that holds
  such a nested group from an earlier import raises an error that asks for the source to be
  re-imported on every read, write, and drop of that name; stores are not migrated.
- Storage operations raise a single task failure as itself and a cooperative shutdown as
  `ShutdownRequested`, not as an exception group. A pipeline stage whose exception group holds
  `KeyboardInterrupt` or `ShutdownRequested` is recorded as interrupted and raises that
  interruption. Scarf keeps no process-wide execution report history; collect reports with
  `execution_report_scope`.
- Recorded artifact inputs have one strict reader, `ArtifactStatus.input_ref(name)`. A missing or
  malformed input raises `ArtifactResolutionError(code="corrupt_payload")` naming the owning kind
  and input, in mapping references and projections, graph lineage, trajectory and
  cluster-selection inputs, plots, and pipeline run records. Query projections validate
  `cell_selection` with the stored-selection validator, and connectivity-map payloads follow the
  neighbor dimension rules, so `n_cells` above 2**32 - 1 is rejected.
- A failed or interrupted artifact write deletes its incomplete slot, so `list_artifacts` no
  longer shows orphaned incomplete artifacts after an error. `ProjectionWriter.abort` deletes the
  unfinished projection instead of leaving it incomplete. Starting any artifact on a read-only
  store raises `PermissionError` before writing, so `run_harmony` and
  `run_pseudotime_scoring(ss_vec=...)` no longer surface the Zarr read-only `ValueError`.
- Gene families: `scarf.features.gene_families` is the one registry of name-based families, and
  `ribosomal` always means RPS, RPL, MRPS, and MRPL. `DEFAULT_PERCENT_PATTERNS` moves there from
  `scarf.assay.classification`, and newly prepared stores record the ribosomal percentage pattern
  as `^RPS|^RPL|^MRPS|^MRPL`. Genes, percentage values, and the HVG blacklist are unchanged.
  Feature identity requires a digit after a registry ID prefix, as `prefix_species` does, so IDs
  of other Ensembl species such as chicken `ENSGALG` are no longer release-drift misses.
- `ChunkedArray` follows NumPy broadcasting: a one-dimensional operand aligns with columns and
  rows are scaled with an `(n_rows, 1)` operand. Scalar keys, other axes, ufunc keywords other
  than `dtype`, and arithmetic between two ChunkedArrays raise; zero-row reductions return NumPy
  shapes.
- Identities that change and are not reused: `run_pca` records `incremental_block_rows` for
  IncrementalPCA fits over several blocks; embedding initialization records
  `algorithm_version="minibatch_kmeans_v3"`, seeds its PCA, and no longer reassigns rarely used
  streamed k-means centroids; `smart_label` records `algorithm_version=3`; default cell-cycle
  genes use CENPU, PIMREG, and JPT1 (mouse Cenpu, Pimreg, Jpt1); parameter tuning's native
  doublet graphs use the candidate ANN seed.
- Results that change while identities stay the same, so earlier artifacts are reused and must be
  recomputed with `invalidate_cache=True`: Dunn tie corrections are exact for tie groups above
  about two million values; Welch reports `n1 + n2 - 2` degrees of freedom when neither group
  varies; fixed-strategy `fit_lowess` returns zero for genes without a positive finite mean and
  variance; graph artifacts built from NumPy boolean `symmetric_graph` flags may be
  unsymmetrized; `run_feature_percentage` gives NaN for a cell without counts.
- Integer arguments share one validator: NumPy integers are accepted, integer-like objects such
  as 0-d arrays are rejected, and messages read `<name> must be an integer` or
  `<name> must be at least N`.
- Paris fits raise each merge to at least its child heights, so hierarchies of tied graphs
  validate, and straight and fixed cuts number equal-size clusters by hierarchy node order.
  `load_paris_clustering` rejects cuts that do not name their hierarchy.
- Inputs are validated before any lookup or write. Graph flags accept only booleans (NumPy
  booleans included). `run_marker_search` rejects labels that are blank, `.`, `..`, or contain
  `/` or `\`. `smart_label` suffixes continue past `z` (`aa`, `ab`) and colliding names raise.
  `select_cells` raises when no cell is retained. `make_bulk` and integration metrics leave out or
  reject NaN, None, blank, and masked labels. Doublet, cell-cycle, prevalent-peak, membership,
  and statistical-testing arguments are checked first, and two-group tests reject `comparisons`
  that reverse the resolved group order. `run_mapping` rejects `query_batches` values that share
  text, such as `1` and `"1"`, and treats nested store locations as one store. `get_cell_vals`
  clipping covers every real numeric column. A stored `defaultAssay` must name an assay.
- Validation corrections that change only which error is raised, never a result or an artifact
  identity: `run_pseudotime_scoring` compares the graph's recorded cell count with its stored cell
  selection, as `run_diffusion_operator` does, and `scarf.trajectory.select_pseudotime_component`
  checks that the graph is square with one row per selected cell, so mismatches raise
  `ValueError` instead of `IndexError`. `run_pseudotime_marker_search` raises `ValueError` when a
  correlation is not finite (a raw pseudotime near the float64 limit) instead of writing an
  unloadable table. `select_cells` raises `ValueError` for an `include` integer beyond the
  float64 range on a floating artifact, instead of `OverflowError`. `integrate_assays` raises
  `ValueError` for sources over different cell selections, whatever their sizes, instead of
  reporting a healthy source as `corrupt_payload`. `query_neighbors` refuses exactly 2**32 cells,
  which graph payloads cannot hold. `SCARF_ZARR_PROFILE` accepts only `fast_local` and `cloud`;
  any other non-empty value, such as `Cloud`, raises `ValueError` instead of being ignored.
- Feature selections must select at least one feature, and every reader applies that one rule. For a
  stored selection that selects none, including the feature universe of an assay without features,
  `DataStore.resolve_features` and every operation that takes a feature selection raise
  `ArtifactResolutionError` (a `ValueError`) with code `corrupt_payload`: `Feature selection must
  select at least one feature`. Before, such a selection resolved and its index read back empty, and
  `run_waggr`, `run_aucell`, `run_marker_search`, `run_pseudotime_marker_search`,
  `run_pseudotime_aggregation`, `run_normalization`, and `run_feature_percentage` each raised their
  own error. No Scarf producer writes such a selection, so only records written or edited outside
  Scarf are affected. `select_hvgs` and `select_detected_features` no longer read a reused selection
  back to check it: a reused record that was emptied outside Scarf is returned, and its consumers
  refuse it. Results and artifact identities do not change.
- `run_waggr` raises the same `ValueError`s with clearer messages. A selected cell total that is
  negative or not finite now reads `<assay>_nCounts holds negative or non-finite totals of selected
  cells; WAGGR library-size normalization requires finite non-negative counts`; before, it said only
  that the normalization scalars must be finite. An RNA assay whose `sf` is `None` now reads `WAGGR
  requires a finite positive size factor` instead of claiming a non-default normalization. Results
  and artifact identities do not change.
- `run_lsi` with the streaming solver reduces its block so that both the fit and the coordinate
  write fit the memory budget; budgets that logged a reduction and then raised `MemoryError`
  finish with smaller blocks. The block size is an execution option, so identities do not change.
  UMAP's layout runs on `min(nthreads, NUMBA_NUM_THREADS)` Numba threads even when the calling
  thread had fewer, and `Assay.score_features` no longer warns for an empty CLR cell selection.
- `DataStoreMerge` refuses destinations that alias, contain, or lie inside a source or that
  already hold content, and creates destinations with mode "w-". Source names cannot contain
  `__`. `overwrite=True` refuses destinations with a prepared assay and clears `defaultAssay`.
  Manifests record `sourceCountFingerprints`, so merges interrupted before this release restart
  with `overwrite=True`. Differing unordered cell-column `levels` are unioned; other differing
  attributes are dropped with a warning.
- `DataStoreMerge` records the preset of each source assay's class: `RNA` for RNA-class assays,
  `ATAC` for ATAC, `ADT` for ADT-class assays such as HTO, and `Assay` otherwise. Merged ADT and
  ATAC assays therefore open as `ADTassay` and `ATACassay` and keep CLR and TF-IDF normalization;
  before, every non-RNA source assay was recorded as the generic `Assay`. `plan()` reports a
  destination whose prepared assay has incomplete counts as blocked (`canDump=False`, "A damaged
  prepared assay requires a fresh destination"), the refusal that `dump()` already raised, instead
  of planning a resume. A merged matrix group counts as complete only when its `complete` attribute
  is `true`, as every other merge component does.
- Imports: H5AD import stores missing categorical, nullable, and string values under linked
  masks and keeps nullable booleans as booleans. `inspect_h5ad` reads group-encoded AnnData
  indexes and prefers an ID column such as `gene_ids`, and reads a one-element `uns` text dataset
  as its element; `H5adReader` reads the index named by `_index`. H5AD sparse matrices need an encoding and a stored shape. H5AD, CSV, and Cell
  Ranger readers reject missing or repeated identifiers. CSV import types values over every row
  and rejects missing or negative counts. Matrix Market BED sidecars give `chrom:start-end`
  Peaks, and feature references are left-joined. Cell Ranger HDF5 reads `matrix` and rejects
  several genome groups. Import writers accept `assay_type`. RDS parsing rejects malformed
  character vectors and xz payloads that need more than 256 MiB of decoder memory.
- Reader edge corrections: `CSVReader` raises `ValueError` for a file without count columns, where
  every column is the ID column or is listed in `cell_data_cols` or `skip_cols`, instead of
  importing an assay without features that `DataStore` cannot open, and it copies `pandas_kwargs`
  instead of adding its `read_csv` settings, such as `chunksize`, to the caller's dictionary.
  `CrH5Reader.matrix_dtype` is the stored dtype in native byte order, with float16 read as float32
  as `H5adReader` reads it, and `consume` yields that dtype, so float16 and big-endian 10x HDF5
  counts import (integral counts unsigned, other counts float32 or the native float dtype) instead
  of failing in SciPy after the destination was created. `H5adReader.sourceMatrixDtype` is likewise
  the stored dtype in native byte order, with float16 read as float32, so an H5AD file whose `X` is
  big-endian (sparse `data` in CSR or CSC, or a dense matrix) imports with the count storage dtype
  instead of failing in SciPy after the destination was created. `CrReader._read_dataset(key)` takes
  a required key and returns a list: the base reader no longer accepts None from it, which no
  shipped reader returned, so the unreachable fallback from missing feature names to feature IDs and
  its warning are removed. A Matrix Market cell sidecar emptied after `inspect_mtx` raises the
  cell-count mismatch error (`Cell sidecar has 0 rows, expected N`) instead of `Cell sidecar must
  contain at least one column`.
- CSV rows have the header's field count: `CSVReader` counts the fields of every row at construction
  and raises ValueError naming the line (`CSV line 3 has 4 fields, but line 1 has 3`) for a row with
  more or fewer fields than the header, or than the first row of a file without one. pandas reads
  the file in chunks of `batch_size` rows and compares a row only with the row before it in the same
  chunk, so a row that started a chunk lost its extra fields, a short row was padded with missing
  values, and a header without a field for the row names, as R `write.table` writes it, made pandas
  take the first column as an implicit index and drop the cell names; each of these files used to
  import without an error. The check reads the text that `read_csv` opens, with the same
  decompression, encoding, and byte order mark handling, after `skip_rows` rows and without blank
  lines, and splits rows with `sep` and the `quotechar`, `quoting`, `doublequote`, `escapechar`, and
  `skipinitialspace` settings of `pandas_kwargs`. `sep` must be one character, so a
  regular-expression separator such as `\s+` raises ValueError, and `pandas_kwargs` cannot set
  `comment`, `dialect`, or `lineterminator`. The check reads the file once more at construction,
  about 16 percent of the reader's pandas pass. `CSVReader` raises KeyError for `skip_cols` names
  that are not CSV columns, as it does for `cell_data_cols`. `CSVtoZarr` no longer compares the
  reader's cell IDs with its row count, because a `CSVReader` returns one ID per row. `MtxReader`
  loses `cell_metadata_path`, which had no effect: the reader extracted the named file from a ZIP
  archive but never read it; passing it raises TypeError, and `MtxCandidate.cellMetadataPath` stays
  as the inspection report of a Parse candidate's cell metadata file. `CrH5Reader` raises ValueError
  (`filtering_cutoff cannot be negative`) for a negative `filtering_cutoff` before it opens the
  file, as `MtxReader` does, whether or not `is_filtered` is set; an unfiltered read used to keep
  every barcode with such a cutoff.
- H5AD geometry: `H5adReader` checks the matrix geometry when it is constructed, before a writer
  opens its destination. The matrix must be two-dimensional, a sparse group needs a stored shape
  even when `obs` and `var` are present, its `indptr` must have one entry more than the rows (CSR)
  or columns (CSC) of that shape, and the shape must equal the `obs` and `var` lengths; an absent
  table takes the matrix's length. Such files used to import with the rows past `obs` dropped or an
  empty feature for each extra `var` row, or failed after the destination existed. Dense `consume`
  rejects a `batch_size` below one, as sparse `consume` does. An `obs` or `var` stored as a dataset
  without fields reads as a table without columns, as `inspect_h5ad` reads it, and cell IDs that are
  not text, such as an integer `cell_ids_key` column, import with embeddings.
- Metadata column names: H5AD readers and inspection list dataframe columns from
  `column-order` and resolve each listed name as an HDF5 path, so columns that old AnnData
  versions nested under `/` are imported instead of skipped. Tables without `column-order`
  still list their direct children, and a nested index named by `_index` resolves either way.
  `H5adReader` rejects a `column-order` attribute that does not hold names when it is
  constructed, before a writer opens its destination. Every import writer stores a source cell or feature column
  whose name contains `/` or `\` under the name with `_` in their place, and logs the rename.
  Source names that are already valid keep their name; a renamed column whose name is taken
  gets the first free `_2`, `_3`, and so on, in source order. Reserved names are checked after
  renaming. Matrix Market feature-reference columns with separators are renamed instead of
  rejected. `inspect_h5ad` and the agent manifest see such columns, the manifest reports stored
  names, and `uns/batch_condition` columns map to stored names. Cytebase `obs_summary` uses
  stored names while `h5ad_keys` keeps source names. The original name is not recorded in the
  store, and `to_h5ad` exports the stored names. `DataStoreMerge` rejects a `source_column` or
  `prepend_text` with a separator or the mask prefix, at construction and again when planning,
  and run snapshots and stored selections treat `\` like `/` and name the `_` spelling. H5AD
  listing keeps a nested group visible when resolved names do not cover all of it, so the reader
  reports it, and leaves datasets with more than one dimension out of the planned names. A 10x
  feature-reference column that is empty for every feature is not planned.
- Seurat: `SeuratReader` and `inspect_seurat` resolve sidecars only inside `sidecar_root`
  (default: the `.rds` directory), and stream sources need it for sidecar-backed layers. Counts
  containing R `NA` raise `missing_count_value`. Dimnames and LogMap identifiers override names
  stored in sidecars. Factor metadata keeps empty levels. Transposed BPCells nodes, MergeFragments
  peak counts, and RegionSelect boundaries follow BPCells, so re-imported counts can differ.
  HDF5 sidecars in H5AD layout need `encoding-type` or `h5sparse_format` and take axis names only
  from the index named by `_index`; other HDF5 sparse groups ignore `sparse_layout` and `layout`
  attributes.
- Exports: `to_h5ad` writes AnnData 0.2.0 encodings and omits `_index` from `column-order`;
  `to_mtx(compress=True)` adds a feature-type column.
- Plotting: `dotplot` and `matrixplot` pool only features of one assay that share an explicit
  label and raise for other shared labels; `matrixplot` orders groups like `dotplot`.
  `marker_heatmap` clusters unclipped z-scores. Continuous color limits follow one policy across
  plots, category scales show only observed categories, `compose_results` keeps panel and raster
  colorbars and honors log scales, and plot functions close the figures they created when they
  raise. `distribution` reports a missing pair value with the `run_statistical_testing` message.
- Agents: workflow configs always save `inputPolicy` and `scoreDoublets`, so workflows paused
  before this release cannot resume, and saved decision, preprocessing, tuning, and context
  records with removed fields fail to load. `extraModelSettings` rejects credential and header
  keys. Resume answers are validated before any stage attempt, failed enrichment reports are not
  replayed, and parameter tuning is PCA only. Feature inventories, family diagnostics and
  policies, and marker tags use the registry families (`ribosomalProtein` and `mitoribosomal`
  merge into `ribosomal`; `hemoglobin`, `immuneReceptor`, `stress`, and `dissociation` are
  removed), and derived QC percentages match feature names only, so saved diagnostics are
  recomputed. `ParameterTuningReport` drops `assayReports` and `recommendedIntegrationId`, and
  `prepare_parameter_tuning_dependencies` drops `pair_harmony_candidates`. The final UMAP uses
  Scarf's categorical palette and natural cluster order. Cell QC execution rejects duplicate
  attributes and a sample artifact named like a metric, and colliding capture label keys raise
  instead of merging.
- Removed public API: `scarf.system_call`, `scarf.get_log_level`, `scarf.GffReader`,
  `scarf.coordinate_melding`, `scarf.utils.iter_column_blocks`, `scarf.utils.rss_peak_tracker`,
  `scarf.matrix.Block`, `ChunkedArray.blocks`, `map_blocks`, `dot`, `std`, `chunks`, and
  `nthreads`, `Assay.to_raw_sparse`, `Assay.mean_features`, `scarf.assay.rna_assay_type_names`,
  `MetaData.mount_location`, `unmount_location`, `remove_trend`, and `insert(location=)`,
  cytebase `Repository.list_files` and `open_zarr`, `DataStore.set_default_assay`,
  `last_execution_report`, `calibrate_label_transfer_threshold`, `metric_lisi`, and
  `load_metric_lisi` (use `scarf.metrics.compute_lisi`), `scarf.clustering.balanced_cut`,
  `BalancedCut`, and `paris_dendrogram`, `scarf.neighbors.wnn_integration`,
  `scarf.embeddings.run_harmony` (use `fit_harmony` or `DataStore.run_harmony`),
  `scarf.metrics.compute_simpson`, `knn_to_csr_matrix`, and `report_technical_nesting`,
  `scarf.mapping.array_hash`, `array_store_hash`, and `conformal_prediction_sets`,
  `MappingReference.fetch_layout`, `scarf.quality_control.write_doublet_target_zarr`,
  `scarf.readers.get_file_handle` and `read_file`, `H5adReader.open_clone`,
  `LoomReader` and `LoomToZarr` (Loom import, including agent ingest; convert Loom files to
  H5AD first), import-result `artifactRefs`, `scarf.writers.bed_to_sparse_array`,
  `create_cell_data`, `load_count_store`, and `load_zarr`, `scarf.utils.load_zarr` (use
  `scarf.load_zarr`), `storage.parallel.map_shards`, `scarf.plotting.collect_legends`,
  `FeatureSummary`, and `register_theme`, `recipes.run_plot_recipe`, `PlotOutput.step_name` and
  `written_path`, `PlotRecipeResult.results`, several `SeuratReader` members, the parameter
  tuning handoff, Pareto, WNN, and refinement exports, `AgentRunConfig.thinkingOffProfile`, and
  `AgentOrchestrator.initialize_request`.

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
