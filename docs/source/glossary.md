# Glossary

Short definitions of terms used across the Scarf documentation.

**Harmony**
: Batch correction method applied to PCA embeddings inside `make_graph(harmonize=True)`. Corrected coordinates are stored as `harmonizedData` in the Zarr hierarchy. A reusable mapping reference additionally preserves the converged state needed for Symphony-style query correction.

**Mapping reference**
: Content-addressed, write-once RNA/PCA artifact built with `build_mapping_reference`. It includes feature scaling, PCA loadings, Harmony cluster state, a reference-distance summary, and provenance required to map queries without moving reference cells.

**Symphony-style mapping**
: Query-to-reference mapping that projects a query into reference PCA space, assigns soft reference clusters, estimates query batch effects, and corrects only query coordinates before neighbor search. Scarf records the current fixed-assignment, scalar-ridge implementation as `symphonyStyleV1`; this name distinguishes it from a complete reimplementation of the Symphony R model.

**LISI (Local Inverse Simpson Index)**
: Metric of local label mixing in the KNN graph. Higher batch LISI after correction suggests better batch mixing. Computed with `metric_lisi`.

**Batch mixing score**
: Mean batch LISI rescaled to `[0, 1]` against the mixing that perfectly integrated data would reach given the batch sizes. Scores near 1 mean well-mixed batches. Computed with `metric_batch_mixing`.

**Silhouette score**
: Graph-based measure of how separated a cluster is from its nearest neighboring cluster. Ranges from -1 to 1, with values near 1 indicating well-separated clusters. Computed with `metric_silhouette`.

**Label concordance (ARI, NMI)**
: Agreement between two labelings of the same cells, such as clusters against reference annotations. ARI ranges from -1 to 1 and NMI from 0 to 1. It reflects label agreement, not batch mixing. Computed with `metric_label_concordance`.

**LSI (Latent Semantic Indexing)**
: Linear dimension reduction used for scATAC-seq graphs, analogous to PCA for RNA.

**Partial PCA**
: PCA trained on a subset of cells via `pca_cell_key`. A lightweight batch correction when one sample is the reference.

**Paris clustering**
: Hierarchical graph clustering in Scarf. Default method; supports `plot_cluster_tree`.

**Leiden clustering**
: Graph-based community detection. Often concordant with UMAP on smaller datasets.

**SNN integration**
: Shared-nearest-neighbor merge of modality-specific KNN graphs via `integrate_assays(method='snn')`.

**WNN integration**
: Weighted nearest neighbor merge for exactly two modalities via `integrate_assays(method='wnn')`.

**CORAL**
: Experimental feature-space domain adaptation used in `run_mapping(run_coral=True)`. It is deprecated in favor of Symphony-style mapping references.

**TopACeDo**
: Manifold-preserving cell subsampling using the KNN graph (`run_topacedo_sampler`).

**densMAP**
: Density-preserving UMAP variant enabled with `run_umap(use_density_map=True)`.

**Zarr profile**
: Storage layout preset (`fast_local` or `cloud`) set via `zarrProfile` or `SCARF_ZARR_PROFILE`.

**AssayMerge**
: Canonical class for merging assays from multiple DataStores into one Zarr file (replaces deprecated `ZarrMerge`).
