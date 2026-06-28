---
description: Memory-efficient single-cell RNA-seq, ATAC-seq, and CITE-seq analysis with Harmony, WNN, and Zarr-backed workflows.
---

[![PyPI][pypi]][pypiLink] [![Docs][docs]][docsLink] [![Github Stars][stars]][github]

# Overview of Scarf

{ref}`Jump to installation <installation>` | {ref}`Quick start <quickstart>` | {ref}`Try live code <colab>` | [Source code on Github]

## What is Scarf

Scarf is a Python package that performs memory-efficient analysis of **single-cell genomics data**.
Using an efficient data chunking process (built on [Zarr]) Scarf manages to perform the core
steps of single-cell genomics data analysis with very low memory consumption. Scarf's core step
is to efficiently generate a neighbourhood graph (KNN graph) of cells. This graph forms the basis for
downstream steps of the analysis thus maximizing concordance between those steps.

:::{image} _static/overview.svg
:width: 75%
:align: center
:::

:::{admonition} Scarf is published!
The article describing Scarf is published in [Nature Communications](https://doi.org/10.1038/s41467-022-32097-3).
:::

## What does Scarf offer

- Analyze atlas scale scRNA-Seq datasets on your laptop (up to 4 million cells tested)
- Perform analysis of scATAC-Seq datasets (datasets with up to 700K cells and 1M peaks tested)
- Parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art TopACeDo method
- **Harmony** batch correction and **partial PCA** for merged datasets ({ref}`merging vignette <harmony_batch_correction>`)
- **WNN and SNN** multimodal integration for CITE-seq ({ref}`multimodal vignette <wnn_integration>`)
- **LISI** and related metrics to quantify integration quality ({ref}`LISI metrics <lisi_metrics>`)
- Project cells across datasets with label transfer and unified embeddings ({ref}`projection vignette <data_projection>`)
- Zarr v3 storage with cloud and local profiles ({ref}`data organization <data_organization>`)

See the {ref}`integration methods guide <integration_guide>` for help choosing between batch correction and integration approaches.

## Why use Scarf

The following flowchart can help one decide which tool to choose with respect to the size of the data.
This flowchart assumes that you do not have access or simply do not want to use a system with large RAM
capacity. Only a few popular tools have been mentioned here, this is not an exhaustive list.

:::{margin} **Useful references**
[Chen et. al.] benchmarked many scATAC-Seq tools for their scalability

[Luecken et. al.] benchmarked scalability of various data integration tools
:::

For large datasets on limited RAM, Scarf's chunked Zarr backend and graph-first workflow often remain practical when in-memory tools do not. See {ref}`FAQ <faq>` for comparisons with Scanpy.

### Example usage scenarios
- You have generated a knockdown model of some stem cells and want to see which lineages are
  affected as a result of this perturbation. So you perform scRNA-Seq of this perturbed stem cell
  derived populations. You download an atlas-scale data available for this tissue system but the
  data is too large and can't be analyzed on your laptop. Using Scarf you can quickly generate a
  UMAP of the atlas scale data and visualize all the author annotated clusters. Now you can project
  your perturbed population over this map to check how the heterogeneity has been affected due the
  perturbation.

- There is an atlas-scale (say more than 500K cells) dataset available for some embryonic
  tissue. You want to redo the cell trajectory analysis with a new and promising tool that just came
  out on Bioarxiv. However, the dataset is too large to be analyzed on your laptop. Using Scarf, you
  can perform clustering on the data on your laptop and run Scarf's cell downsampling algorithm to
  select the most representative cells from the clusters of your interest. You can then use this
  downsampled data and run the trajectory analysis on it.

## When not to use Scarf

Scarf prioritizes memory efficiency over breadth of third-party integrations. Methods such as Scanorama, scVI, or rich trajectory toolkits are not built in. For small datasets where speed matters most, Scanpy may be faster. Scarf works best as a complement for atlas-scale or multi-dataset workflows on modest hardware.

## Scarf development: the future

The core team is constantly working to improve Scarf's functionality, add new features and to
make the code more robust (aka testing). We aim to extend Scarf to more single-cell
methodologies by including normalization and dimension reduction strategies best suited to
those methods.

[pypi]: https://img.shields.io/pypi/v/scarf.svg
[pypiLink]: https://pypi.org/project/scarf
[docs]: https://readthedocs.org/projects/scarf/badge/?version=latest
[docsLink]: https://scarf.readthedocs.io
[stars]: https://img.shields.io/github/stars/parashardhapola/scarf?style=social
[github]: https://github.com/parashardhapola/scarf
[Source code on Github]: https://github.com/parashardhapola/scarf
[Zarr]: http://zarr.readthedocs.io
[Chen et. al.]: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1854-5
[Luecken et. al.]: https://www.biorxiv.org/content/10.1101/2020.05.22.111161v2.full
