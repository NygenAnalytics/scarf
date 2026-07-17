(glossary)=
# Glossary

```{glossary}
count matrix
  Sparse matrix of features (genes, peaks, or ADTs) by cells. Scarf stores counts in a Zarr assay group.

highly variable genes
  Features selected with `mark_hvgs` for neighbourhood-graph construction.

neighbourhood graph
  KNN graph of cells built by `make_graph`. Embeddings, clustering, mapping, and multimodal integration reuse this graph.

cell key
  Boolean column in cell metadata selecting which cells participate in a step. Default is `I`. Filtering marks cells inactive rather than deleting them.

DataStore
  Primary Scarf object that opens a Zarr store and exposes analysis methods.

assay
  Named modality inside a `DataStore` (for example `RNA`, `ADT`, `ATAC`).

Harmony
  Batch correction applied to PCA embeddings inside `make_graph(harmonize=True)`.

partial PCA
  PCA trained on a subset of cells via `pca_cell_key`. A lightweight batch-correction option when one sample is the reference.

LISI
  Local Inverse Simpson Index. Metric of local label mixing in the KNN graph. Computed with `metric_lisi`.

batch correction
  Adjusting embeddings or graphs so technical sample batches mix while biological structure is preserved.

mapping reference
  Content-addressed RNA/PCA artifact from `build_mapping_reference` used for Symphony-style query mapping.

Paris clustering
  Hierarchical graph clustering in Scarf (`run_clustering`). Supports cluster trees.

Leiden clustering
  Graph community detection via `run_leiden_clustering`.

SNN integration
  Shared-nearest-neighbor merge of modality-specific KNN graphs via `integrate_assays(method='snn')`.

WNN integration
  Weighted nearest-neighbor merge for exactly two modalities via `integrate_assays(method='wnn')`.

TopACeDo
  Manifold-preserving cell subsampling using the KNN graph (`run_topacedo_sampler`).

densMAP
  Density-preserving UMAP variant enabled with `run_umap(use_density_map=True)`.

LSI
  Latent Semantic Indexing. Linear dimension reduction used for scATAC-seq graphs.

AssayMerge
  Canonical class for merging assays from multiple DataStores into one Zarr file.
```
