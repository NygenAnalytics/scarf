(faq)=
# FAQ


## How to cite Scarf?

See {doc}`citation`.

## Who maintains Scarf? Is ScarfWeb required?

Scarf is open source and maintained by [Nygen](https://nygen.io).
[ScarfWeb](https://www.nygen.io/products/scarfweb) is Nygen's hosted, browser-based product
built on Scarf. You can install and run this library without ScarfWeb. Bug reports and
feature requests for the Python package belong on
[GitHub issues](https://github.com/parashardhapola/scarf/issues).

## How does Scarf compare to Scanpy?

See {doc}`../scarf_and_scanpy` for a stage-by-stage mapping, round-trip notes, and a short
Seurat subsection.

## How do I run Harmony batch correction in Scarf?

Pass `harmonize=True` and `batch_columns` to `make_graph` on a merged dataset. See {ref}`Harmony batch correction <harmony_batch_correction>` and the {ref}`integration methods guide <integration_guide>`.

## What is the difference between SNN and WNN integration?

Both merge modality-specific KNN graphs with `integrate_assays`. SNN (default) supports two or more assays. WNN (`method='wnn'`) requires exactly two assays and can weight modalities differently. See {ref}`WNN integration <wnn_integration>`.

## How do I compute LISI in Scarf?

Use `metric_lisi` on the latest KNN graph after integration. See {ref}`LISI metrics <lisi_metrics>`.

## Should I use tSNE or UMAP?

tSNE and UMAP are complementary visualization tools. tSNE emphasizes local structure and can reveal fine-grained diversity. UMAP preserves more global structure, which helps when cluster relationships matter. We suggest tSNE for large (>50k cells) atlas-scale datasets because of its quick runtime. UMAP runtime can span hours on atlas-scale datasets. In Scarf, UMAP and tSNE use the same initial embedding by default and share the same input graph.

## What is densMAP?

Enable density-preserving UMAP with `run_umap(use_density_map=True)`. Useful when preserving local density structure matters alongside cluster separation.

## Which clustering should we use, Paris or Leiden?

Leiden is faster than Paris, especially for large datasets. On small datasets we have tested,
Leiden results are often more concordant with UMAP clusters. Paris provides a hierarchy that
can show relationships between clusters. Both methods have low computational requirements, so
you can run both and view the Paris hierarchy with Leiden labels:

```python
import scarf.plotting as splt

splt.cluster_tree(
    ds,
    cluster_key="RNA_cluster",
    fill_by_value="RNA_leiden_cluster",
)
```

## How do I create a count matrix for my single-cell data?

Generating count matrices is the primary step of single-cell data analysis. For scRNA-Seq you can use
tools like [STARsolo], [alevin-fry] or [kallisto|bustools]. If your data was generated using
10x's commercial solution then you can use [Cell Ranger]. In the case of single-cell ATAC-Seq data,
try the protocol from [Yan et al] or [Cusanovich et al]. [Cell Ranger ATAC] can
be used if your data was generated using 10x's kit.

[STARsolo]: https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md
[alevin-fry]: https://alevin-fry.readthedocs.io/en/stable/
[kallisto|bustools]: https://www.kallistobus.tools/
[Cell Ranger]: https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger
[Yan et al]: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1929-3
[Cusanovich et al]: https://www.cell.com/cell/fulltext/S0092-8674(18)30855-9
[Cell Ranger ATAC]: https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/what-is-cell-ranger-atac

## Can I use Scarf from R?

Not yet. Please open a discussion on GitHub if an R API would be useful.

## What Python version does Scarf require?

Python 3.12 or newer (`requires-python >=3.12`). See {ref}`installation <installation>`.
