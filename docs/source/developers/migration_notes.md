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

Embedding marker area now uses both the selected cell count and physical panel
area. Automatic categorical legends no longer disappear above 40 categories;
large legends wrap into columns. Continuous colorbars now carry the plotted
feature or metadata label. Exact numeric marker sizes, legend placement, edge
widths, and figure sizes remain available through explicit arguments.

Publication plotting additions are source compatible with existing calls:

- `DensityOverlay` and `Highlight` add contours and focused selections to
  `embedding`.
- `cluster_connectivity` summarizes the cell graph between groups at embedding
  centroids.
- Grouped `dotplot` calls render feature-set brackets, while
  `distribution(kind="stacked_violin")` provides aligned multi-feature rows.
- `compose_results` merges plots drawn into a Matplotlib mosaic and can render
  shared legends.
- `PlotRecipe`, `PlotStep`, and `run_recipe` support typed batch plotting plus
  strict JSON or TOML configuration. Serialized plot keyword names use lower
  camel case, such as `layoutKey` and `colorBy`.

Automatic dotplot marker areas now fit the physical grid cells of each panel.
Pass an explicit `SizeScale` to retain exact marker areas. Stacked violins use
independent value axes by default; pass `share_y=True` when direct magnitudes
must be compared on one scale. Embedding panel titles can be suppressed with
`show_titles=False` when a labeled colorbar or shared legend already identifies
the values.

`DensityOverlay(statistic="mean")` draws contours from a smoothed local mean for
continuous embedding panels. Regions below `min_support` effective support are
attenuated before contouring. Set `max_hotspots=1` to retain only the strongest
connected hotspot. `cluster_connectivity` accepts `cell_size`, `cell_alpha`, and
`cell_color` for its reference-cell background.

Distribution plots accept `sample_by` to aggregate biological samples before
drawing boxes or violins, and `split_by` to compare conditions within each
group. Composition summaries can display SD, SE, or 95% confidence intervals.
Matrix, marker, and pseudotime heatmaps now expose explicit ordering, clustering,
annotation scales, and caller-owned targets.

Mapping diagnostics are available as `mapping_score`, `mapping_evidence`,
`mapping_confusion`, `mapping_calibration`, and `mapping_projection`. Mapping
score is reference-side landing density.
`mapping_calibration` plots held-out accuracy against retained coverage over an
evidence threshold. It does not treat vote fraction as a calibrated
probability, and it fails with a clear error when `known_labels` only match the
transferred labels after string conversion. `mapping_projection` keeps the
reference layout fixed.

Legends and axis labels avoid repeating information. A continuous embedding
panel drops its title when the colorbar already carries the same label. Side
legends cap at four columns; when a categorical column has more entries than
fit, the most populated categories are retained in their declared order. The
legend title reports how many are shown and the omitted values are recorded in
`PlotResult.provenance.extras["omitted_legend_entries"]`. `compose_results`
merges legend blocks with scale-prefixed labels when separate right-side blocks
would overlap. Violin and box panels label the value axis `value` unless the
values were standardized or aggregated per sample. `matrixplot` rows follow the
requested feature order, including the order implied by feature-set mappings.

Requesting more colors than a palette provides no longer recycles them.
`palette_name="colorblind"` warns and switches to evenly spaced hues beyond
twelve categories.

Dark-theme scatter outlines now use a mid-gray rather than a near-white edge.
Multi-layout embeddings report one scale per unique color encoding and one
legend specification per color variable instead of duplicating that metadata
for every layout.

The dark theme exports to an opaque charcoal background unless
`transparent=True` is passed to `PlotResult.save`. Stacked composition bars now
have visible segment boundaries by default. Pass `segmentLinewidth=0` in a
serialized recipe, or `segment_linewidth=0` in Python, to remove them.

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

## DataStore summaries and assay names

`DataStore.summary()` reserves `summary` as an assay name. New writers reject that name before
opening or changing their destination, and `DataStore` rejects an existing `summary` assay before
changing store-level state.

This is a clean break. Scarf does not silently rename existing assays because the assay name can
also appear in `defaultAssay`, `assayTypes`, artifact paths, provenance, and dependent metadata.
Prefer recreating the datastore from its source with another assay name and recomputing dependent
artifacts. Any manual migration must update every such reference before the store is opened with
the current Scarf release.

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
in memory and written once. `run_normalization`, `Assay.save_normalized_data`,
`run_mapping`, and `run_doublet_detection` no longer accept `batch_size`
either. Normalized Zarr geometry now comes from the storage profile instead of
the caller, so a batch size can no longer produce undersized or oversized
chunks. `batch_size` also left PCA identity, so existing PCA artifacts are
recomputed once on the first run after upgrading. Feature-column operations
(`run_marker_search`, `run_pseudotime_marker_search`,
`run_pseudotime_aggregation`, `find_markers_by_regression`) default to the
stored feature chunk width capped by the operation memory budget when their
batch argument is left unset. New L2 neighbor artifacts store Euclidean
distances, rather than the squared values returned by HNSW. Inner-product ANN
is no longer accepted for graph construction because it can produce negative
values that are invalid for connectivity kernels. Existing graph artifacts
remain readable.

The neighbor artifact attribute formerly named `recall` measured only whether
the ANN query returned each cell itself. New artifacts store that diagnostic as
`self_hit_rate`.

## Graph consumers

`run_umap`, `run_tsne`, `run_leiden_clustering`, `run_topacedo_sampler`, and
`integrate_assays` now return the artifact reference they wrote, and
`run_paris_clustering` carries the same reference on the `ref` field of its
result. Cell metadata is still written on every call, so code that reads
columns such as `RNA_UMAP1` or `RNA_leiden_cluster` is unaffected.

These calls also accept a graph artifact as their first argument, following the
chain convention already used by `run_pca(normalized, ...)`:

```python
graph = ds.build_connectivity_map(neighbors)
umap = ds.run_umap(graph, n_epochs=300)
clusters = ds.run_leiden_clustering(graph, resolution=1.0)
```

Omitting the graph keeps the implicit behaviour: the current analysis chain of
the assay is used, or the integrated graph named by `integrated_graph`. Passing
both a graph artifact and `integrated_graph` raises. When a connectivity map is
passed, the assay, cell key, and feature key are read from the artifact.

`from_assay`, `cell_key`, and `feat_key` are keyword-only on these calls.
Positional calls such as `ds.run_umap("RNA", "I", "hvgs")` must become
`ds.run_umap(from_assay="RNA", cell_key="I", feat_key="hvgs")`.

`run_topacedo_sampler` no longer accepts `return_edges`. The Steiner tree edges
live in the sampling artifact:

```python
sampling = ds.run_topacedo_sampler(cluster_key="RNA_paris_cluster")
edges = ds.load_artifact(sampling)["edges"][:]
```

`run_tsne` raises `RuntimeError` on unsupported platforms and when the SG-tSNE
executable or `sgtsnepi` package is missing. It previously logged the failure
and returned `None`.

## Pipeline cluster selection

`ds.pipeline.run` now copies the labels of the partition it selects to
`{assay}_clusters`, for example `RNA_clusters`, and links that column to the
same artifact as the resolution-specific column. Selection runs whenever any
clustering ran, not only when doublet scoring or marker search need it, so the
column and the `selected_clusters` ref are always available afterwards. A single
partition is taken directly instead of being scored.

Doublet scoring and marker search now group by that column, so marker tables are
indexed under `RNA_clusters` rather than the winning resolution:

```python
ds.pipeline.run()
ds.plots.embedding(layout_key="RNA_UMAP", color_by="RNA_clusters")
ds.plots.marker_heatmap(group_key="RNA_clusters", topn=5)
```

Naming a partition explicitly keeps the old behaviour, and the marker table is
then indexed under that partition's own column:

```python
ds.pipeline.run(markers={"clusters": {"leiden": 1.0}})
ds.plots.marker_heatmap(group_key="RNA_leiden_1.0", topn=5)
```

## Marker statistics

Fresh marker tables use schema v2. They add AUC and
`p_value_adjusted`, calculated with Benjamini-Hochberg correction within each
one-versus-rest group over all tested features. The Mann-Whitney calculation is
two-sided and now applies continuity and large-tie correction consistently.
Recomputed raw and adjusted p-values can therefore differ from older tables.

Legacy marker tables remain readable. Rerun marker search when the new columns
or corrected statistics are required. Neither schema represents
replicate-aware differential expression.

## Logging and progress

Progress is no longer inferred from log severity. Code that expected a quiet
log level to hide progress bars should migrate to:

```python
scarf.configure_output(level="WARNING", progress=False)
```

`set_verbosity(level=..., filepath=...)` still selects a level and optional
file sink.

## H5AD import

H5AD import now decodes categorical and pandas-nullable metadata and imports
supported dense `obsm` arrays as numbered cell-metadata columns. Unsupported
group encodings, sparse `obsm` slots, and row-mismatched embeddings are skipped
with warnings. Review imported metadata names if downstream code previously
assumed those columns were unavailable.

## Import batching and Matrix Market

`CrToZarr.dump`, `H5adToZarr.dump`, and `SparseToZarr.dump` now choose source
batch rows automatically when `batch_size` is omitted. The selected rows follow
destination shard geometry when memory permits, but source reads remain
separate from physical shard-aligned writes. Existing explicit positive values
remain supported.

Use `inspect_mtx`, `MtxReader`, and `MtxToZarr` for generic Matrix Market
sources, Parse DGE directories, prefixed triplets, and direct MEX ZIP files.
Feature-major sources use temporary disk-backed CSR arrays. `CrDirReader`
retains its existing API as the Cell Ranger directory compatibility path.
CSV remains a small-data compatibility importer and Loom remains a legacy
compatibility importer.

New multimodal imports use stable assay names for CRISPR, multiplexing,
antigen, and custom features. BD `mRNA` and `AbSeq` map to RNA and ADT. These
names change the persisted assay layout for sources that previously produced
`ASSAYn`. 10x feature tags and validated feature-reference columns are now
retained as feature metadata. Guide-call tables are only reported by
inspection and are not imported.

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

Mapping now has a hard boundary between reference preparation and query
analysis. Build a scaled PCA chain in a writable reference datastore. A plain
PCA chain can feed the ANN index directly. A Symphony reference instead runs
`run_harmony` after PCA and builds its ANN index and neighbours from the
corrected coordinates. Package either chain with
`reference_ds.build_mapping_reference(neighbors)`.

After preparation, the reference may be reopened read-only. The query must be
a different physical store and must remain writable. To map a dataset against a
reference prepared from the same source counts, create a separate writable
analysis layer with `mount_datastore` and use that mounted datastore as the
query. Mapping currently accepts RNA queries only; ATAC query mapping remains
unsupported.

Run mapping from the writable query datastore:

```python
mapping = query_ds.run_mapping(
    reference,
    "atlas",
    query_assay="RNA",
)
```

The returned `MappingResult` and its `ArtifactRef` identify an immutable
projection stored in the query datastore. Mapping consumers accept a
`MappingResult`, projection `ArtifactRef`, or mapping name. Pass the
`MappingReference` explicitly when an `ArtifactRef` or name does not carry the
in-session reference handle:

```python
mapping = query_ds.get_mapping_result(
    "atlas",
    reference=reference,
    query_assay="RNA",
)
labels = query_ds.get_target_classes(
    mapping,
    reference_class_group="cell_type",
)
```

The only missing-feature policies are `reference_mean`, `zero`, and `error`.
`reference_mean` is the default. CORAL, `missing_feature_policy="intersection"`,
`exclude_missing`, `ref_mu`, `ref_sigma`, `MappingReference.map_query`,
`project_mapping_layout`, `load_unified_graph`, `run_unified_umap`,
`run_unified_tsne`, `unified_embedding`, and `mapping_correction` have been
removed.

Legacy projection groups are rejected by current loaders and consumers. Rerun
`run_mapping` against a prepared reference. References embedded in an older
graph or reduction are also rejected and must be rebuilt with
`build_mapping_reference(neighbors)`.

`SymphonyCorrectionModel` no longer carries `correction_ridge`, and Symphony
mapping references no longer store a `correction_ridge` attribute. The query
correction always applied the fixed unit penalty that upstream Symphony uses,
so the field described a setting that never changed the result. Code that
constructed the model directly should drop the argument. Symphony references
written before this change keep an unused attribute and are otherwise valid.

L2 projection distances are true Euclidean distances rather than squared HNSW
distances. Recalibrate thresholds based on distance-weighted mapping scores or
projection distances after rerunning a projection. Query cells whose projected
PCA vector has zero norm are marked uninformative. Label transfer abstains for
those cells, mapping scores exclude them, and their projected reference
coordinates are `NaN`.

`get_mapping_score` weights each neighbour by `1 / (log(distance + 1) + 1)`,
matching the published definition. Weights are absolute rather than normalised
per query cell, so a query population that sits far from the reference deposits
less total score than one that lands on it. Under the previous normalised
weights the total score of any query group was fixed at `multiplier / save_k`
regardless of how well that group matched. Absolute score values change, so
rescale any saved thresholds.

Label transfer, label evidence and embedding projection use weights normalised
per query cell, which is what a vote fraction and a coordinate average require.
This is not the same kernel as the released 0.x line, which weighted a neighbour
by `1 - distance / max(distance)` and therefore gave the furthest of the `save_k`
neighbours a weight of exactly zero. Votes are now spread across all `save_k`
neighbours instead of `save_k - 1`, so a single near neighbour no longer decides
a label on its own and near ties abstain more often. Recheck
`threshold_fraction` and `save_k` against a labelled holdout rather than reusing
0.x settings.

Call `query_ds.project_reference_embedding(...)` before
`mapping_projection`. It writes an immutable query embedding artifact and
links its two coordinate columns into query cell metadata. Reference
coordinates remain unchanged.

## HTO demultiplexing

`mark_hto_identities` now returns the cell metadata column name. Legacy HTO
identity artifacts are not reused because provenance records the corrected
normalization, clustering, background, cutoff, and singlet-assignment methods.

## Metric names

Use `metric_label_concordance` for ARI or NMI label agreement.
Use `metric_proportional_batch_mixing` for Scarf's mean LISI summary adjusted
for global batch proportions, and `metric_ilisi` for the scIB median-scaled
batch metric. These two summaries are intentionally different.
Use `metric_graph_silhouette` for the sampled graph-guided silhouette score.

`metric_lisi` now accepts `label_columns` and returns one per-cell array for
each requested column in a dictionary. It no longer accepts `label_colnames`,
`save_result`, or `return_lisi`, and it no longer writes `lisi__*` cell
metadata columns. Store a returned array explicitly when persistence is needed.

Use `metric_cluster_separability` to evaluate cluster labels in coordinates
from a completed `run_pca` artifact. The result keeps aggregate clustering
scores, per-cluster F1 scores, and confusion tables separate.

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
- `scarf.mapping_reference.MappingReference` moved to `scarf.mapping.reference.MappingReference`.
- `scarf.mapping_reference.MappingResult` moved to `scarf.mapping.models.MappingResult`.
- Symphony models moved from `scarf.symphony` to `scarf.mapping.models`; numerical functions moved to `scarf.mapping.symphony`.
- Helpers from `scarf.mapping_utils` moved to `scarf.mapping.hashing`,
  `scarf.mapping.confidence`, and `scarf.mapping.features`.

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
