(migration_notes)=
# Migration notes

Notes for developers updating older Scarf workflows or documentation. See
{doc}`architecture` for current package responsibilities and placement rules.

## Plotting

The plotting API is a clean break from earlier releases. The old `scarf.plots`
module and flat `DataStore.plot_*` methods no longer exist. Store-backed plots
are available through `ds.plots`, while the same canonical functions remain
available from `scarf.plotting`.

```python
import scarf.plotting as splt

ds.plots.embedding(layout_key="RNA_UMAP", color_by="clusters")

# Equivalent standalone call
splt.embedding(ds, layout_key="RNA_UMAP", color_by="clusters")
```

Plot functions render by default with `show=True`. Pass `show=False` when the returned
`PlotResult` must remain available for figure access or saving.

`mark_hvgs(..., show_plot=True, **plot_kwargs)` now forwards keyword arguments to
`splt.highly_variable_features`. Rename older mean-variance plot kwargs as follows:
`ax_label_fs` to `label_size`, `fig_size` to `figsize`, `ss` to `point_sizes`, and
`cmaps` to `colormaps`.

## Merge APIs

Prefer `AssayMerge` and `DatasetMerge`. The former `ZarrMerge` alias has been
removed; construct `AssayMerge` directly.

## Cell keys

Filtering still marks cells inactive via boolean cell keys (default `I`) rather than
deleting rows. Custom keys remain the supported way to subset for reclustering and mapping.

## Graph construction

`DataStore.make_graph` has been removed. Existing datastores remain readable,
including stores written before artifact-backed graph stages.

Use `ds.pipeline.run(...)` for a standard workflow. For direct control, replace
one graph call with an explicit artifact chain:

```python
normalized = ds.run_normalization(feat_key="hvgs")
reduction = ds.run_pca(normalized, dims=15)
ann_index = ds.build_ann_index(reduction)
ds.build_embedding_initialization(reduction, n_centroids=100)
neighbors = ds.query_neighbors(ann_index, k=11)
ds.build_connectivity_map(neighbors)
```

Use `run_lsi` for ATAC or `run_custom_reduction` for custom loadings. The old
`dims=0` shortcut (neighbours on normalized features with no PCA) is now an
identity custom reduction: `run_custom_reduction(np.eye(n_features), normalized)`.
Replace Harmony flags with `run_harmony(batch_columns, reduction, ...)`, then
pass its result to `build_ann_index` and `query_neighbors`. Build Symphony
mapping references separately with `build_mapping_reference`.

The former graph call derived `ann_efc` and `ann_ef` from `k`, and `ann_m` from
`dims`. For equivalent settings, pass
`ann_efc=ann_ef=min(100, max(k * 3, 50))` and
`ann_m=min(max(48, int(dims * 1.5)), 64)` to `build_ann_index`.
`local_cache` remains available on reduction methods. It has been removed from
normalization, `run_harmony`, `build_ann_index`, and `query_neighbors` because
only reductions reread normalized expression.

Custom ANN fetcher and saver callbacks were removed because ANN payloads are
stored inside their artifacts. New ANN payloads include metric, dimension,
element-count, and digest checks. Metadata-free ANN payloads from existing
stores remain readable.

`build_connectivity_map` no longer accepts `batch_size`; connectivity is built
in memory and written once. New L2 neighbor artifacts store Euclidean
distances, rather than the squared values returned by HNSW. Inner-product ANN
is no longer accepted for graph construction because it can produce negative
values that are invalid for connectivity kernels. Existing graph artifacts
remain readable.

The neighbor artifact attribute formerly named `recall` measured only whether
the ANN query returned each cell itself. New artifacts store that diagnostic as
`self_hit_rate`.

## Paris clustering

Use `run_paris_clustering` for Paris clustering. Its default
`n_clusters="auto"` selects a branch-adaptive cut and writes
`RNA_paris_cluster` for the default RNA assay. The automatic cut combines
branch persistence with a configuration-null modularity guard over unweighted
graph topology, so a candidate split must add structure beyond the
degree-preserving null model. Its
`min_cluster_size` defaults to the graph's `k + 1`; pass it explicitly when a
workflow needs a stable lower bound. Pass an integer `n_clusters` to request a
fixed cut. The former `run_clustering` forwarding shim has been removed; call
`run_paris_clustering` directly. The DataStore balanced-cut mode has been
removed.

Paris now fits the additive graph `A + A.T`. The first Paris call on an older
store rebuilds its hierarchy and emits a warning. Fixed and adaptive labels can
therefore differ from labels created by earlier releases, even when an integer
cluster count is unchanged.

Adaptive Paris labels are cached under a content digest of the hierarchy
generation and `min_cluster_size`. Incomplete caches or caches missing the
required diagnostic arrays are ignored and recomputed on first use; fixed cuts
and the stored hierarchy are unaffected. `scikit-network` is no longer a runtime
dependency. It remains in Scarf's test dependencies for reference comparisons, so
applications that import it directly must declare it themselves.

## Mapping calls

The following compatibility paths remain accepted through Scarf 1.x and may be
removed in 2.0:

- Projection groups written before provenance schemas remain readable when their neighbor arrays are structurally valid. Scarf emits `DeprecationWarning`; rerun `run_mapping` to write full provenance.
- A writable legacy Harmony graph without a mapping artifact is rebuilt automatically the first time `run_mapping` needs it. For a read-only store, reopen it with `zarr_mode='r+'` and call `build_mapping_reference(..., batch_columns=[...])` once.
- `ref_mu=False` and `ref_sigma=False` no longer select query-derived statistics. They emit `DeprecationWarning` and use reference statistics. Remove these arguments.
- `exclude_missing=True` remains an alias for `missing_feature_policy='intersection'`.
- `run_coral=True` remains available with `DeprecationWarning`.

Recomputed results can differ from earlier Scarf releases. Recalibrate downstream thresholds after rebuilding.

## Metric names

Use `metric_label_concordance` for ARI or NMI label agreement.
Use `metric_proportional_batch_mixing` for Scarf's mean LISI summary adjusted
for global batch proportions, and `metric_ilisi` for the scIB median-scaled
batch metric. These two summaries are intentionally different.
Use `metric_graph_silhouette` for the sampled graph-guided silhouette score.

The former aliases `metric_integration`, `metric_batch_mixing`,
`metric_silhouette`, and `scarf.metrics.integration_score` have been removed.

## Internal import paths

The supported top-level imports and `scarf.writers` functions are unchanged. Code that
imports internal helpers directly should use these focused paths:

- `scarf.parallel` was removed. Use `scarf.storage.parallel`.
- `scarf._types` was removed. Use `scarf.storage.types`.
- `scarf.chunked.ChunkedArray` moved to `scarf.matrix.ChunkedArray`.
- `scarf.downloader`, `scarf.fetch_dataset`, and
  `scarf.show_available_datasets` were removed. Use
  `scarf.cytebase.connect("scarf_docs")` to list and download example datasets.
- `scarf.storage.zarr_store` was removed. Use the focused storage modules:
  - Profiles and location detection: `scarf.storage.profiles`
  - Store opening and root helpers: `scarf.storage.stores`
  - Chunk, shard, codec, and array specifications: `scarf.storage.layout`
  - Numeric and metadata array creation: `scarf.storage.arrays`
  - Dense, sparse, and transposed count writing: `scarf.storage.sharding`
  - Group copying and local staging: `scarf.storage.copy`
  - ANN index persistence: `scarf.storage.ann_index`
- Zarr array creation helpers live in `scarf.storage.arrays`.
- Assay schema creation and finalization helpers live in `scarf.storage.schema`.
- Chunked-to-Zarr materialization helpers live in `scarf.storage.materialize`.

The compatibility-only module paths were removed. Imports from the old paths
now fail, so import the focused modules directly.

### Metadata internals

The documented `scarf.metadata.MetaData` and `MetaDataRowBlock` paths are
unchanged. Table storage, row streaming, and query implementations now live in
focused modules under `scarf.metadata`.

### Assay internals

Documented and tested `scarf.assay` paths are unchanged. Implementations now
live in `normalization`, `persistence`, `base`, `rna`, `atac`, and `adt`.
Public classes retain `scarf.assay` metadata, while their source definitions
live in those focused modules. Other private implementation paths may change.

### Reader internals

Documented `scarf.readers` paths are unchanged. Implementations now live in
`_text`, `cellranger`, `h5ad`, `loom`, and `csv`. Private implementation paths
may change. Format modules load when their reader class is first requested.
Public reader classes retain `scarf.readers` metadata while their source
definitions live in the focused modules.

### Writer internals

Documented `scarf.writers` paths are unchanged. Implementations now live in
`_store`, `_materialize`, `cellranger`, `h5ad`, `loom`, `csv`, `sparse`,
`subset`, and `export`, and load when their first facade symbol is accessed.
Public classes and functions retain `scarf.writers` metadata while their
physical source lives in these focused modules. Tested facade patch seams
remain stable within 1.x. Other private implementation paths and incidental
module globals may change.

### Merge internals

Documented `scarf.merge` paths are unchanged. Implementations now live in
`assays` and `datasets` and load on first facade access. Public classes retain
`scarf.merge` metadata. Tested facade patch seams remain stable within 1.x.
Private implementation paths and other incidental module globals may change.

### Feature and quality-control internals

- `scarf.feat_utils.fit_lowess` moved to
  `scarf.features.variability.fit_lowess`.
- `scarf.feat_utils.binned_sampling` moved to `scarf.features.scoring.binned_sampling`.
- `scarf.feat_utils.hto_demux` moved to `scarf.quality_control.hto.hto_demux`.
- `scarf.doublet_utils` functions moved to `scarf.quality_control.doublets`.
- `scarf.bio_data` cell-cycle references moved to the public
  `scarf.quality_control` package.
- `scarf.meld_assay.GffReader` moved to
  `scarf.features.genomic.gff.GffReader`.
- BED and interval helpers from `scarf.meld_assay` moved to
  `scarf.features.genomic.intervals`.
- Count melding functions from `scarf.meld_assay` moved to
  `scarf.features.genomic.melding`.
- Rank and regression marker implementations moved to
  `scarf.features.markers`.
- `scarf.genomics` and `scarf.markers` were removed.

Top-level `scarf.GffReader` and `scarf.coordinate_melding` remain supported.

The public `scarf.utils` exports remain available. Repository code uses
focused implementation modules: `arrays`, `compute`, `logging`, `prefetch`,
`process`, and `progress`. Zarr opening and location types live in
`scarf.storage.stores`. These focused paths are internal and may change before
Scarf 2.0; supported user code should prefer top-level `scarf` exports.

### Mapping internals

- Private mapping orchestration now lives at `scarf.datastore._operations.mapping`.
- Unified plotting reads persisted layouts through
  `MappingDatastore._load_unified_layout_data`; plotting no longer resolves
  projection Zarr paths directly.
- `scarf.mapping_reference.MappingReference` moved to `scarf.mapping.reference.MappingReference`.
- `scarf.mapping_reference.MappingResult` moved to `scarf.mapping.models.MappingResult`.
- Symphony models moved from `scarf.symphony` to `scarf.mapping.models`; numerical functions moved to `scarf.mapping.symphony`.
- Helpers from `scarf.mapping_utils` moved to `scarf.mapping.hashing`, `scarf.mapping.confidence`, `scarf.mapping.coral`, and `scarf.mapping.features`.

### DataStore internals

- Private quality-control orchestration now lives at `scarf.datastore._operations.quality_control`.
- Private marker, HVG, feature, and assay orchestration now lives at
  `scarf.datastore._operations.features`.
- Private pseudotime marker and feature aggregation orchestration now lives at
  `scarf.datastore._operations.trajectory`.
- `_feature_column_chunk` and `_load_marker_cluster_frame` moved from `scarf.datastore.datastore` to `scarf.datastore._operations.features`.
- Private presentation and metric orchestration now lives at `scarf.datastore._operations.presentation`.
- Public methods remain available on `DataStore`, but method `__module__` and
  `__qualname__` values now identify their private operation mixins.

### Graph and trajectory internals

Graph, embedding, clustering, and trajectory implementations now use these
focused paths:

- `scarf.ann.AnnStream` moved to `scarf.neighbors.stream.AnnStream`.
- `scarf.ann.fix_knn_query` and `instantiate_knn_index` moved to
  `scarf.neighbors.index`.
- `EMBEDDING_CACHE_MAX_BYTES` was removed with the cross-stage embedding cache.
- Graph merging functions from `scarf.knn_utils` moved to
  `scarf.neighbors.graph`.
- `scarf.knn_utils._is_umap_version_new` moved to
  `scarf.neighbors.graph`.
- `scarf.knn_utils.wnn_integration` moved to
  `scarf.neighbors.integration.wnn_integration`.
- `scarf.knn_utils.run_sgtsne` and `export_knn_to_mtx` moved to
  `scarf.embeddings.sgtsne`.
- `self_query_knn` and `smoothen_dists` were removed. Connectivity construction
  now lives in `scarf.neighbors.graph`; `scarf.knn_utils` was removed.
- `scarf.umap` implementations moved to `scarf.embeddings.umap`.
- `scarf.dendrogram.BalancedCut` moved to
  `scarf.clustering.balanced_cut`. Cluster-tree helpers moved to
  `scarf.clustering.cluster_tree`.
- Pseudotime feature-profile clustering moved to
  `scarf.trajectory.feature_dynamics.knn_clustering`.
- `scarf.results` classes moved to `scarf.trajectory.results`.
- Private pseudotime helpers from `scarf.datastore.graph_datastore` and
  `scarf.datastore.datastore` moved to `scarf.trajectory.pseudotime` and
  `scarf.trajectory.feature_dynamics`.
- Harmony correction moved beside PCA and LSI under
  `scarf.embeddings.harmony`. The old `scarf.harmony` path was removed.

The old module paths and forwarding symbols were removed. Public `DataStore`
methods and the documented top-level `scarf` exports remain unchanged.

`AnnStream` no longer stores an `annPath` or decides whether an embedding fits
the in-memory cache budget. Code constructing it directly should track
persistence paths separately and pass `cache_embeddings` explicitly.
`DataStore` applies the existing 256 MiB limit automatically.

Tests that patch these internals must patch the name resolved by the consumer.
Patch the canonical module for imports performed inside a method, such as
`scarf.embeddings.sgtsne.run_sgtsne`. Graph orchestration aliases now resolve
under `scarf.datastore._operations.graph`, and pseudotime aliases resolve under
`scarf.datastore._operations.trajectory`. Do not patch the old facade module
because production code no longer resolves through it.

## Documentation execution

Executable pages live under `docs/source/` (especially `tutorials/`). Refresh the myst-nb
cache with:

```bash
cd docs && make execute-page PAGE=scrna_seq
# or
cd docs && make execute-docs JOBS=1
```

Timeout per page is 600 seconds (`nb_execution_timeout` in `conf.py`).
