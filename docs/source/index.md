---
description: Memory-efficient single-cell RNA-seq, ATAC-seq, and CITE-seq analysis with neighbourhood graphs and Zarr-backed stores.
---

[![PyPI][pypi]][pypiLink] [![Docs][docs]][docsLink] [![Github Stars][stars]][github]

# About Scarf

{ref}`Installation <installation>` · {ref}`Quick start <quickstart>` · {doc}`scarf_and_scanpy` · [Source code on Github]

## Introduction

Scarf is a Python package for memory-efficient analysis of single-cell genomics data.
It stores count matrices and intermediate results in [Zarr]-backed stores and builds a
neighbourhood graph (KNN graph) of cells that downstream steps reuse. The same graph
feeds embeddings, clustering, mapping, and multimodal integration, which keeps those
steps concordant.

Scarf supports scRNA-seq, scATAC-seq, and CITE-seq workflows on local machines without
loading the full matrix into memory.

:::{image} _static/overview.svg
:width: 75%
:align: center
:alt: Overview of Scarf analysis stages from counts through neighbourhood graph to embeddings and clustering
:::

:::{admonition} Citation
The methods paper is published in [Nature Communications](https://doi.org/10.1038/s41467-022-32097-3).
:::

## What this documentation covers

- Installing Scarf and running a minimal scRNA-seq pipeline
- End-to-end tutorials for scRNA-seq, scATAC-seq, and CITE-seq
- Quality control, feature selection, neighbourhood graphs, embeddings, and clustering
- Marker genes, annotation, and subsetting with cell keys
- Merging batches, Harmony, partial PCA, and integration metrics
- Mapping query cells to a reference and transferring labels
- Multimodal SNN/WNN integration
- Large-scale analysis with downsampling and Zarr storage notes
- Plotting with `scarf.plotting` and the API reference

## What this documentation does not cover

- Raw FASTQ processing or alignment (use Cell Ranger, STARsolo, alevin-fry, or similar)
- Methods Scarf does not ship, including scVI, Scanorama, RNA velocity, and full DE
  pipelines with multiple-testing correction. Export a subset (for example with
  `to_anndata` / `to_h5ad`) and continue in Scanpy or another tool when you need those methods.

## Who should read this

Biologists and bioinformaticians analyzing single-cell count matrices. Prior experience
with Scanpy or Seurat helps but is not required. If you already use Scanpy and want a
lower-memory path for large data, start with {doc}`scarf_and_scanpy`.

## Structure of the documentation

1. **Get started**: installation, quick start, and Scarf compared with Scanpy
2. **Introductory tutorials**: canonical scRNA-seq, scATAC-seq, CITE-seq, import/export, plotting
3. **Data integration and mapping**: method choice, merge/Harmony, metrics, label transfer, reference atlases
4. **Other analyses**: QC depth, cell cycle, pseudotime, imputation, downsampling
5. **Reference**: API, glossary, FAQ, citation
6. **Developers**: contributing and internals

## Prerequisites

- Python 3.12 or newer
- Basic familiarity with NumPy and pandas
- A count matrix (for example Cell Ranger H5 or H5AD)

## Citation

Dhapola, P., Rodhe, J., Olofzon, R. et al. Scarf enables a highly memory-efficient
analysis of large-scale single-cell genomics data. Nat Commun 13, 4616 (2022).
https://doi.org/10.1038/s41467-022-32097-3

## Contributing and feedback

Report bugs and request features on [GitHub issues]. See {doc}`developers/contributing`
for how to build and execute documentation pages locally.

## Start here

1. {ref}`Install Scarf <installation>`
2. Run the {ref}`Quick start <quickstart>`
3. Read {doc}`scarf_and_scanpy` if you know Scanpy or Seurat
4. Work through {doc}`tutorials/scrna_seq`

Capability links:

- One scRNA-seq dataset: {doc}`tutorials/scrna_seq`
- Merge batches: {doc}`tutorials/data_integration`
- CITE-seq: {doc}`tutorials/cite_seq`
- Map to a reference: {doc}`tutorials/reference_atlas`
- Downsample large data: {doc}`tutorials/downsampling`

[pypi]: https://img.shields.io/pypi/v/scarf.svg
[pypiLink]: https://pypi.org/project/scarf
[docs]: https://readthedocs.org/projects/scarf/badge/?version=latest
[docsLink]: https://scarf.readthedocs.io
[stars]: https://img.shields.io/github/stars/parashardhapola/scarf?style=social
[github]: https://github.com/parashardhapola/scarf
[Source code on Github]: https://github.com/parashardhapola/scarf
[Zarr]: https://zarr.readthedocs.io
[GitHub issues]: https://github.com/parashardhapola/scarf/issues
