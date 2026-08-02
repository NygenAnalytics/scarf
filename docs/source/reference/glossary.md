(glossary)=
# Glossary

```{glossary}
:sorted:

analysis chain
  Record, kept per assay, of which results a workflow is currently building on:
  the active normalization, reduction, neighbour index, and connectivity graph.
  A method called without an explicit input takes its input from this chain,
  which is what lets `ds.run_umap()` know which graph to lay out. Exposed as
  `AssayState`.

artifact
  A persisted analysis result such as a normalization, PCA, neighbourhood graph,
  or marker table. Scarf writes each one into the Zarr store next to a record of
  what produced it, so results outlive the session that computed them and can be
  inspected, compared, or reused later.

ArtifactRef
  Reference to an artifact, returned by the method that wrote it. It identifies a
  stored result without loading it, so it can be passed to the next step or held
  for later comparison. Read the data with `load_artifact`, and the parameters
  and status with `inspect_artifact`.

feat_key
  Argument naming which feature selection a step should use. Selections are
  stored per cell key, so `mark_hvgs` under cell key `I` writes the column
  `I__hvgs` and later calls pass `feat_key='hvgs'`. Scarf supplies the prefix.

provenance
  Record stored with every artifact naming the operation that produced it, the
  scientific parameters it used, and the artifacts it consumed. It is what lets
  Scarf recognise that a new request describes a result the store already holds.

reuse
  Returning an existing artifact instead of recomputing it, when the requested
  operation, parameters, and inputs match its provenance. Changing a parameter
  produces a new artifact, and the steps that depended on the previous one are
  recomputed rather than reused.

update_state
  Keyword on graph-construction methods deciding whether their result joins the
  assay's analysis chain. Leave it at `True` for a linear workflow. Pass `False`
  to try a parameter without changing what later calls default to.

count matrix
  Sparse matrix of features (genes, peaks, or ADTs) by cells. Scarf stores counts in a Zarr assay group.

highly variable genes
  Features selected with `mark_hvgs` for neighbourhood-graph construction.

neighbourhood graph
  KNN graph of cells built by individual graph-construction methods or
  `ds.pipeline.run`. Embeddings, clustering, mapping, and multimodal integration
  reuse this graph.


cell key
  Boolean column in cell metadata selecting which cells participate in a step. Default is `I`. Filtering marks cells inactive rather than deleting them.

DataStore
  Primary Scarf object that opens a Zarr store and exposes analysis methods.

assay
  Named modality inside a `DataStore` (for example `RNA`, `ADT`, `ATAC`).

Harmony
  Batch correction applied to PCA embeddings with `run_harmony` before ANN construction.


partial PCA
  PCA trained on a subset of cells via `pca_cell_key`. A lightweight batch-correction option when one sample is the reference.

LISI
  Local Inverse Simpson Index. Per-cell measure of local label mixing in the KNN graph. Computed with `metric_lisi`.

iLISI
  Integration LISI. Median batch LISI scaled so higher values indicate better batch mixing. Computed with `metric_ilisi`.

cLISI
  Cell-type LISI. Median label LISI inverted and scaled so higher values indicate better biological-label conservation. Computed with `metric_clisi`.

proportion-aware batch mixing
  Scarf summary that rescales mean batch LISI against the observed global batch proportions. Computed with `metric_proportional_batch_mixing`.

graph connectivity
  Mean fraction of cells from each label retained in its largest connected component on Scarf's symmetrized graph. Computed with `metric_graph_connectivity`.

batch correction
  Adjusting embeddings or graphs so technical sample batches mix while biological structure is preserved.

mapping reference
  Immutable RNA mapping artifact built from a scaled PCA or Symphony neighbour
  chain with `build_mapping_reference(neighbors)`. A writable query datastore
  uses it to create query-owned projections without changing the reference.

Paris clustering
  Hierarchical graph clustering in Scarf (`run_paris_clustering`). Supports
  fixed cuts and branch-adaptive cuts guarded by configuration-null modularity,
  plus cluster-tree visualization.

Leiden clustering
  Graph community detection via `run_leiden_clustering`.

SNN integration
  Shared-nearest-neighbor merge of modality-specific KNN graphs via `integrate_assays(method='snn')`.

WNN integration
  Hao-inspired weighted nearest-neighbor merge for exactly two modalities via
  `integrate_assays(method='wnn')`. Scarf scores the union of the two existing,
  self-free KNN rows with L2-normalized affinity and a simple `k`-th-neighbour
  bandwidth. Unlike Seurat defaults, it does not build a wider 200-neighbour
  candidate pool or use SNN-far bandwidth, which keeps memory and query cost
  practical for millions of cells.

TopACeDo
  Manifold-preserving cell subsampling using the KNN graph (`run_topacedo_sampler`).

densMAP
  Density-preserving UMAP variant enabled with `run_umap(use_density_map=True)`.

LSI
  Latent Semantic Indexing. Linear dimension reduction used for scATAC-seq graphs.

AssayMerge
  Canonical class for merging assays from multiple DataStores into one Zarr file.
```
