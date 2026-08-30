(glossary)=
# Glossary

```{glossary}
:sorted:

artifact lineage
  Directed record of the exact selections and upstream artifacts that produced a stored result.
  Granular methods pass returned `ArtifactRef` values explicitly; no result is selected
  implicitly for an assay.

artifact
  A persisted analysis result such as a normalization, PCA, neighbourhood graph, or marker table.
  Scarf writes each one into the Zarr store next to a record of what produced it, so results outlive the session that computed them and can be inspected, compared, or reused later.

ArtifactRef
  Reference to an artifact, returned by the method that wrote it.
  It identifies a stored result without loading it, so it can be passed to the next step or held for later comparison.
  Read the data with `load_artifact`, and the parameters and status with `inspect_artifact`.

feature selection
  Immutable Boolean artifact aligned to the complete feature order of one assay.
  Producers such as `select_hvgs` return an `ArtifactRef` and leave feature metadata unchanged.
  Direct feature consumers require the exact ref through `features=`.

all features
  Assay-wide all-true feature-selection artifact returned by `DataStore.select_all_features` and
  used by complete-universe operations.
  It is distinct from the physical feature metadata column `I`.

provenance
  Record stored with every artifact naming the operation that produced it, the scientific parameters it used, and the artifacts it consumed.
  It is what lets Scarf recognise that a new request describes a result the store already holds.

reuse
  Returning an existing artifact instead of recomputing it, when the requested operation, parameters, and inputs match its provenance.
  Changing a parameter produces a new artifact, and the steps that depended on the previous one are recomputed rather than reused.

PipelineRun
  Durable handle returned by `ds.pipeline.run()` and reopened through `ds.pipeline.open(...)`.
  It maps stable output names to artifacts and exposes frozen cell and feature views plus reports.
  Plotting, marker loading, and export remain `DataStore` operations.

count matrix
  Sparse matrix of primary counts stored cell-major (`n_cells` × `n_features`), for features such as genes, peaks, or ADTs.
  Scarf stores that array as `counts` in the assay Zarr group.
  RNA assays also store `countsT`, the same values in gene-major order, so gene-wise stages can stream without scanning every cell.

highly variable genes
  Features selected with `select_hvgs` for neighbourhood-graph construction.

neighbourhood graph
  KNN graph of cells built by individual graph-construction methods or `ds.pipeline.run`.
  Embeddings, clustering, mapping, and multimodal integration reuse this graph.

cell key
  Boolean column in cell metadata that can be captured as an analytical input.
  Default is `I`.
  Filtering snapshots it into an immutable selection artifact, combines thresholds with that
  selection, and leaves the column unchanged.

DataStore
  Primary Scarf object that opens a Zarr store and exposes analysis methods.

assay
  Named modality inside a `DataStore` (for example `RNA`, `ADT`, `ATAC`).

Harmony
  Batch correction applied to PCA embeddings with `run_harmony` before ANN construction.

partial PCA
  PCA trained on an immutable cell subset via `pca_cell_selection`.
  A lightweight batch-correction option when one sample is the reference.

LISI
  Local Inverse Simpson Index.
  Per-cell measure of local label mixing in the KNN graph.
  Computed with `metric_lisi`.

iLISI
  Integration LISI.
  Median batch LISI scaled so higher values indicate better batch mixing.
  Computed with `metric_ilisi`.

cLISI
  Cell-type LISI.
  Median label LISI inverted and scaled so higher values indicate better biological-label conservation.
  Computed with `metric_clisi`.

proportion-aware batch mixing
  Scarf summary that rescales mean batch LISI against the observed global batch proportions.
  Computed with `metric_proportional_batch_mixing`.

graph connectivity
  Mean fraction of cells from each label retained in its largest connected component on Scarf's symmetrized graph.
  Computed with `metric_graph_connectivity`.

batch correction
  Adjusting embeddings or graphs so technical sample batches mix while biological structure is preserved.

mapping reference
  Immutable RNA mapping artifact built from a scaled PCA or Symphony neighbour chain with `build_mapping_reference(neighbors)`.
  A writable query datastore uses it to create query-owned projections without changing the reference.

Paris clustering
  Hierarchical graph clustering in Scarf (`run_paris_clustering`).
  Supports fixed cuts and branch-adaptive cuts guarded by configuration-null modularity, plus cluster-tree visualization.

Leiden clustering
  Graph community detection via `run_leiden_clustering`.
  A manual call returns a cluster-label artifact without adding metadata columns. The RNA pipeline
  runs default resolutions `0.5`, `0.75`, `1.0`, and `1.25`, plus Paris unless disabled. Its
  artifact-backed silhouette stage scores Leiden resolutions in the graph's coordinate space, and
  `run["clusters"]` is the selected Leiden candidate's exact ref. Paris remains a diagnostic run
  output. This automatic choice is a reproducible baseline, not biological validation.

SNN integration
  Shared-nearest-neighbor merge of explicit modality-specific connectivity-map refs via
  `integrate_assays(sources, method="snn")`.

WNN integration
  Hao-inspired weighted nearest-neighbor merge for two or more explicit modality neighbour refs
  via `integrate_assays(sources, method="wnn")`.
  Scarf scores the union of all existing, self-free KNN rows with affinity and the distance span from each modality's nearest to its `k`-th neighbour as bandwidth.
  During scoring it L2-normalizes modality coordinate rows; the affinities themselves are not L2-normalized.
  Unlike Seurat defaults, it does not build a wider 200-neighbour candidate pool or use SNN-far bandwidth.

TopACeDo
  Manifold-preserving cell subsampling using the KNN graph (`run_topacedo_sampler`).

densMAP
  Density-preserving UMAP variant enabled by passing explicit graph and initialization refs to
  `run_umap(..., use_density_map=True)`.

LSI
  Latent Semantic Indexing.
  Linear dimension reduction used for scATAC-seq graphs.

DataStoreMerge
  Canonical class for merging DataStores into one Zarr file.
```
