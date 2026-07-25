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

Scarf supports scRNA-seq, scATAC-seq, and CITE-seq workflows on a laptop or a single
server, reading data from local disk or object storage, without loading the full matrix
into memory.

:::{image} _static/overview.svg
:width: 75%
:align: center
:alt: Overview of Scarf analysis stages from counts through neighbourhood graph to embeddings and clustering
:::

:::{admonition} Citation
The methods paper is published in [Nature Communications](https://doi.org/10.1038/s41467-022-32097-3).
:::

## Why Scarf

Most single-cell tools load the whole count matrix into memory and hold every
intermediate result there too. That is fine until the dataset outgrows the machine.
Scarf takes a different route: counts and results live in a [Zarr] store on disk or
object storage, and the analysis is built around one neighbourhood graph that every
later step reuses. Four things follow from that design.

**Large datasets run on modest hardware.** The full pipeline (QC, HVGs, graph, UMAP,
Leiden, markers) has been measured end to end on feature-major `countsT` stores. Graph
construction sets most of the memory ceiling, so peak RAM grows slowly from hundreds of
thousands to tens of millions of cells.

| Cells | Peak RAM | End-to-end wall time |
|---|---|---|
| 100k | ~7 GiB | ~15 min |
| 500k | ~25 GiB | ~47 min |
| 5M | ~33 GiB | ~8.2 h |
| 10M | ~37 GiB | ~22.8 h |

Times are measured on 8 CPU cores with the countsT speed pack. These are measured anchors,
not projections. Actual numbers depend on hardware, storage, and parameters.

**It is remote-first.** A Zarr store can live on S3-compatible object storage, and Scarf
reads and writes it in place. You can analyze a dataset that never lands on your local
disk as a full copy, which is what makes shared-storage and cloud workflows practical.

**One graph keeps results concordant.** The standard workflow (`ds.pipeline`) or the
atomic graph steps (normalize, reduce, optional Harmony, ANN, neighbors, connectivity)
build a single KNN graph. Embeddings, clustering, mapping, and multimodal integration all
read from it, so your UMAP, clusters, and transferred labels share the same neighbourhood
structure rather than separately parameterized copies.

**Results persist as you work.** Every step writes back into the store, so you can stop,
inspect intermediate outputs, and resume later without recomputing. Filtering marks cells
inactive instead of deleting them, so quality-control choices stay reversible.

## What this documentation covers

- Installing Scarf and running a minimal scRNA-seq pipeline with `ds.pipeline.run`
- Concepts for provenance, graph state, and measured scale/memory behavior
- Atomic graph operations and end-to-end tutorials for scRNA-seq, scATAC-seq, and CITE-seq
- Quality control, feature selection, neighbourhood graphs, embeddings, and clustering
- Marker genes, gene-set enrichment, annotation, and subsetting with cell keys
- Merging batches, Harmony, partial PCA, and integration metrics
- Mapping query cells to a reference and transferring labels
- Multimodal SNN/WNN integration
- Zarr organization, remote stores, provenance reuse, and downsampling
- Plotting through `ds.plots` or `import scarf.plotting as splt`, with the API reference

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

1. **Get started**: installation, quick start, what is new in 1.0, and Scarf with Scanpy
2. **Concepts**: provenance, graph state, and scale/memory
3. **Core workflows**: atomic graph operations, then scRNA-seq, scATAC-seq, and CITE-seq
4. **Analysis essentials**: quality control, dimensionality reduction, clustering, annotation, and plotting
5. **Data management and scaling**: Zarr organization, import/export, remote stores, provenance reuse, and downsampling
6. **Integration and mapping**: merge/Harmony with metrics, label transfer, and reference atlases
7. **Specialized analyses**: gene-set enrichment, cell cycle, pseudobulk export, and trajectories

8. **Reference and support**: API, FAQ, glossary, citation, and community links
9. **Developers**: contributing and internals

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
3. Skim {doc}`whats_new_in_1_0` if you used Scarf 0.x
4. Read {doc}`scarf_and_scanpy` if you know Scanpy or Seurat
5. Work through {doc}`tutorials/scrna_seq` or {doc}`tutorials/atomic_graph_operations`

Capability links:

- Provenance and artifacts: {doc}`concepts/provenance`
- Scale and memory: {doc}`concepts/scale_and_memory`
- Atomic graph operations: {doc}`tutorials/atomic_graph_operations`
- Data organization: {doc}`tutorials/data_organization`
- Remote stores: {doc}`tutorials/remote_stores`
- One scRNA-seq dataset: {doc}`tutorials/scrna_seq`
- Merge batches: {doc}`tutorials/data_integration`
- CITE-seq: {doc}`tutorials/cite_seq`
- CITE-seq SNN/WNN integration: {doc}`tutorials/cite_seq_integration`
- Score gene sets: {doc}`tutorials/gene_set_enrichment`
- Map query cells to a reference: {doc}`tutorials/mapping_and_label_transfer`
- Build a reusable reference atlas: {doc}`tutorials/reference_atlas`
- Pseudotime modules: {doc}`tutorials/pseudotime_modules`
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
