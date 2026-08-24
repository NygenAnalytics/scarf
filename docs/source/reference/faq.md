(faq)=
# FAQ

## How to cite Scarf?

See {doc}`citation`.

## Who maintains Scarf? Is ScarfWeb required?

Scarf is open source and maintained by [Nygen](https://nygen.io).
[ScarfWeb](https://www.nygen.io/products/scarfweb) is Nygen's hosted, browser-based product built on Scarf.
You can install and run this library without ScarfWeb.
Bug reports and feature requests for the Python package belong on [GitHub issues](https://github.com/NygenAnalytics/scarf/issues).

## How does Scarf compare to Scanpy?

See {doc}`../scanpy_and_seurat` for a stage-by-stage mapping, round-trip notes, and a short Seurat subsection.

## How should an AI agent use Scarf?

Start with {doc}`../analysis_with_agents`.
It explains how to inspect and resume a datastore, create reversible branches, compare scientific evidence, route to the granular APIs, and report uncertainty without treating defaults or one metric as biological truth.

## How do I run Harmony batch correction in Scarf?

After PCA, call `ds.run_harmony(['batch_column'], pca)` then continue with `build_embedding_initialization`, `build_ann_index`, `query_neighbors`, and `build_connectivity_map`.
See {ref}`Harmony batch correction <harmony_batch_correction>` and the {ref}`dataset integration guide <integration_guide>`.

## Which integration method should I choose?

- Separate scRNA-seq batches: start with {doc}`../tutorials/dataset_merging`, then compare Harmony or partial PCA in {doc}`../tutorials/batch_correction`.
- Multiple assays in the same cells (CITE-seq): SNN or WNN ({ref}`recommended workflow <multimodal_integration>` and {doc}`../tutorials/multimodal_diagnostics` for diagnostics).
- Map onto an existing reference: {doc}`../tutorials/mapping_and_label_transfer`.

Scarf does not include Scanorama, BBKNN, scVI, ComBat, or other external integration packages.
Export with `to_anndata` or `to_h5ad` when you need those tools.

## How do I reduce log noise or disable progress?

Configure the two settings independently:

```python
scarf.configure_output(level="WARNING", progress=False)
```

For a batch log file, use `set_verbosity(..., filepath=...)`.
Enable timestamps with `configure_output(timestamps=True)`.
See {doc}`api/utilities` for all output settings and their defaults.
Tutorial pages show completed snapshots from their cached execution; live notebooks animate the same operations.

## What is the difference between SNN and WNN integration?

Both merge modality-specific KNN graphs with `integrate_assays`.
SNN (default) supports two or more assays and combines shared edge support.
WNN (`method='wnn'`) also supports two or more assays, learns one weight per assay and cell, and ranks candidates by the resulting blended affinity.

### Scarf WNN versus Seurat

Scarf WNN follows the weighting equations from Hao et al. but does not reproduce Seurat's default search exactly.
It considers the union of the existing KNN rows, uses the distance span from each assay's nearest to its `k`-th nonself neighbour as that assay's bandwidth, and L2-normalizes rows only during scoring.
For each cell and ordered pair of modalities, it compares within-modality and cross-modality prediction affinity.
It then sums the exponentiated directed scores for each target modality and normalizes those grouped strengths across all modalities.

Seurat normally searches a wider `knn.range=200` pool and uses SNN-far bandwidth.
Scarf instead reuses the stored assay graphs.
With two modalities, avoiding the wider search at ten million cells saves two additional index builds, 20 million queries, and 4 billion materialized candidate records.

Both differences are measured rather than assumed.
Given the same candidate pool and bandwidth, Scarf reproduces Seurat 5.5.1 to the float32 resolution of the stored graph in both two-modality and synthetic three-modality fixtures.
Against Seurat's shipped defaults on a two-modality CITE-seq subset, Scarf selects 89 percent of the same neighbours and its per-cell RNA weight correlates at 0.76.
The reference values ship with the test suite.

### Scaling notes

The per-cell prediction work grows quadratically with the number of modalities, while scoring blended graph edges is linear in the union candidate pool.
Stored neighbour rows, output edges, and modality weights remain linear in cell count.
See {ref}`WNN integration <wnn_integration>` for deviations and trade-offs.

## How do I compute LISI in Scarf?

Use `metric_lisi` for raw per-cell LISI values.
Use `metric_ilisi` for a single scIB-scaled batch-mixing score and `metric_clisi` for a scIB-scaled biological-label conservation score.
See {ref}`LISI metrics <lisi_metrics>`.

## Should I use tSNE or UMAP?

tSNE and UMAP are complementary visualization tools.
tSNE emphasizes local structure and can reveal fine-grained diversity.
UMAP preserves more global structure, which helps when cluster relationships matter.
We suggest tSNE for large (>50k cells) atlas-scale datasets because of its quick runtime.
UMAP runtime can span hours on atlas-scale datasets.
In Scarf, UMAP and tSNE use the same initial embedding by default and share the same input graph.

## What is densMAP?

Enable density-preserving UMAP with `run_umap(use_density_map=True)`.
Useful when preserving local density structure matters alongside cluster separation.

## Which clustering should we use, Paris or Leiden?

Leiden is faster than Paris, especially for large datasets.
On small datasets we have tested, Leiden results are often more concordant with UMAP clusters.
Paris provides a hierarchy that can show relationships between clusters.
Both methods have low computational requirements, so you can run both and view the Paris hierarchy with Leiden labels:

```python
ds.plots.cluster_tree(
    cluster_key="RNA_paris_cluster",
    fill_by_value="RNA_leiden_cluster",
)
```

## Why will my existing RNA Zarr store not open?

Current Scarf versions write RNA counts twice: cell-major `counts` and a gene-major `countsT` copy.
Opening an RNA assay fails if that copy is missing, incomplete, still Zarr v2, or does not match `counts`.
There is no silent rewrite on open.

Re-import the source, or write a new store with `python -m scarf.tools.repack_zarr`.
After a rewrite, recompute HVG, normalization, PCA, graph, and marker results.
Non-RNA assays do not use `countsT`.
See {doc}`../concepts/memory_and_execution`.

## How do I create a count matrix for my single-cell data?

Generating count matrices is the primary step of single-cell data analysis.
For scRNA-Seq you can use tools like [STARsolo], [alevin-fry] or [kallisto|bustools].
If your data was generated using 10x's commercial solution then you can use [Cell Ranger].
For single-cell ATAC-Seq data, [Cell Ranger ATAC] can be used if your data was generated using 10x's kit.
[Yan et al] reviews ATAC-seq analysis approaches; [Cusanovich et al] describes an scATAC-seq experimental approach.

[STARsolo]: https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md
[alevin-fry]: https://alevin-fry.readthedocs.io/en/stable/
[kallisto|bustools]: https://www.kallistobus.tools/
[Cell Ranger]: https://www.10xgenomics.com/support/software/cell-ranger/latest
[Yan et al]: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1929-3
[Cusanovich et al]: https://www.cell.com/cell/fulltext/S0092-8674(18)30855-9
[Cell Ranger ATAC]: https://www.10xgenomics.com/support/software/cell-ranger-atac/latest

## Can I use Scarf from R?

Not yet.
Please open a discussion on GitHub if an R API would be useful.

## What Python version does Scarf require?

Python 3.12 or newer (`requires-python >=3.12`).
See {ref}`installation <installation>`.
