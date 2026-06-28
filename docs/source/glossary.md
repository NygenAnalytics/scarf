# Glossary

Short definitions of terms used across the Scarf documentation.

**Harmony**
: Batch correction method applied to PCA embeddings inside `make_graph(harmonize=True)`. Corrected coordinates are stored as `harmonizedData` in the Zarr hierarchy.

**LISI (Local Inverse Simpson Index)**
: Metric of local label mixing in the KNN graph. Higher batch LISI after correction suggests better batch mixing. Computed with `metric_lisi`.

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
: Domain adaptation used in `run_mapping(run_coral=True)` to align reference and target feature distributions.

**TopACeDo**
: Manifold-preserving cell subsampling using the KNN graph (`run_topacedo_sampler`).

**densMAP**
: Density-preserving UMAP variant enabled with `run_umap(use_density_map=True)`.

**Zarr profile**
: Storage layout preset (`fast_local` or `cloud`) set via `zarrProfile` or `SCARF_ZARR_PROFILE`.

**AssayMerge**
: Canonical class for merging assays from multiple DataStores into one Zarr file (replaces deprecated `ZarrMerge`).
