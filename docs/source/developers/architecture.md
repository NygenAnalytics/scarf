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
- `mapping/` owns reference artifacts, feature alignment, confidence, Symphony-style correction, and mapping results.

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
- `DataStore.pipeline.run` takes a `params` mapping of per-stage settings. Every run records two
  more stages, `membership_strength` and `tsne`, skipped unless requested, and its configuration
  records `params`, `species`, `tsne`, `membershipStrength`, and the Leiden `selected`
  resolution. `pca_dims=0` skips PCA.
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
  `neighbors.diffusion.transition_matrix` returns the graph-sized single step. Doublet scores and
  every diffusion and pipeline artifact identity are unchanged.
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
- Library-size and CLR normalization promote integer counts to float64 before scaling or taking
  logarithms, so uint8 and int8 stores no longer raise and uint16 and int16 stores no longer wrap.
  Artifacts whose values `normed` computes record `count_arithmetic="float64"`, so earlier
  results are not reused: library-size values from any integer counts, because the integer
  product could overflow at any width, and CLR values from integer counts narrower than 32 bits.
  This covers `run_normalization` payloads for RNA with `renormalize_subset=False` and for CLR,
  marker tables on the fallback path, pseudotime markers and aggregations that call `normed`,
  statistical tests of assay-normalized features, and cell-cycle scores computed through
  `normed`. Identities on floating-point stores, CLR identities on 32- and 64-bit counts, and
  every pipeline artifact identity are unchanged. Grouped ADT assays built from narrow counts
  must be rebuilt.
- RNA `normed` without subset renormalization maps a zero library total to 1, so zero-count cells
  normalize to 0 instead of NaN. `run_normalization` records `zero_total_divisor="one"` for RNA
  with `renormalize_subset=False` when the selection contains a zero-count cell, so normalizations
  written earlier with NaN rows are not reused. Other normalization identities and every pipeline
  artifact identity are unchanged. Grouped RNA assays built earlier hold NaN for zero-count cells
  and must be rebuilt.
- ATAC `normed` no longer leaves its fitted TF-IDF state (`n_term_per_doc`, `n_docs`,
  `n_docs_per_term`) on the assay after the call, and RNA `normed` never changes `normMethod`.
  A `DataStore` is not designed for concurrent use from several threads.
- float16 is not a count storage dtype. Count writers reject it, and H5AD imports read float16
  sources as float32.
- Operations trust that prepared counts and artifacts do not change during a call. Writing to
  prepared data in place is outside the contract and is not detected.
- Minimum versions rise to scipy 1.15, statsmodels 0.14.5 (earlier releases fail to import
  with scipy 1.16), threadpoolctl 3.5 (earlier releases cannot see the OpenBLAS in NumPy and
  SciPy wheels, so BLAS thread limits had no effect), huggingface-hub 2.0, and, for the `agent`
  and `test` extras, pydantic-ai-slim 2.51.
- Count layout: plans whose countsT chunks fell below half the chunk target (awkward cell counts
  such as primes) now use whole-target chunks, so stores written with those plans fail layout
  replay and must be re-imported. Count assays require Zarr format 3.
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
- Identities that change and are not reused: feature summaries computed through `normed` for a
  non-default `normMethod` on integer counts record `count_arithmetic="float64"`; `run_pca`
  records `incremental_block_rows` for IncrementalPCA fits over several blocks; embedding
  initialization records `algorithm_version="minibatch_kmeans_v3"`, seeds its PCA, and no longer
  reassigns rarely used streamed k-means centroids; `smart_label` records `algorithm_version=3`;
  default cell-cycle genes use CENPU, PIMREG, and JPT1 (mouse Cenpu, Pimreg, Jpt1); parameter
  tuning's native doublet graphs use the candidate ANN seed.
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
- `DataStoreMerge` refuses destinations that alias, contain, or lie inside a source or that
  already hold content, and creates destinations with mode "w-". Source names cannot contain
  `__`. `overwrite=True` refuses destinations with a prepared assay and clears `defaultAssay`.
  Manifests record `sourceCountFingerprints`, so merges interrupted before this release restart
  with `overwrite=True`. Differing unordered cell-column `levels` are unioned; other differing
  attributes are dropped with a warning.
- Imports: H5AD import stores missing categorical, nullable, and string values under linked
  masks and keeps nullable booleans as booleans. `inspect_h5ad` reads group-encoded AnnData
  indexes and prefers an ID column such as `gene_ids`, and reads a one-element `uns` text dataset
  as its element; `H5adReader` reads the index named by `_index`. H5AD sparse matrices need an encoding and a stored shape. H5AD, CSV, and Cell
  Ranger readers reject missing or repeated identifiers. CSV import types values over every row
  and rejects missing or negative counts. Matrix Market BED sidecars give `chrom:start-end`
  Peaks, and feature references are left-joined. Cell Ranger HDF5 reads `matrix` and rejects
  several genome groups. Import writers accept `assay_type`. RDS parsing rejects malformed
  character vectors and xz payloads that need more than 256 MiB of decoder memory.
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
