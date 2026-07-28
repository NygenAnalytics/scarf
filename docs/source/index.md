---
description: Memory-efficient analysis of single-cell omics data on local disk or object storage, with provenance-backed results.
---

[![PyPI][pypi]][pypiLink] [![Docs][docs]][docsLink] [![Github Stars][stars]][github]

# About Scarf

Scarf stands for Single Cell Analysis on Remote File Systems.

{ref}`Installation <installation>` · {ref}`Quick start <quickstart>` · {doc}`scarf_and_scanpy` · [Source code on Github]

## Introduction

Scarf is a Python package for memory-efficient analysis of single-cell omics data. It
supports scRNA-seq, scATAC-seq, and CITE-seq workflows on a laptop or a single server,
reading data from local disk or object storage, without loading the full matrix into
memory.

One store carries a whole analysis: quality control and cell filtering, feature
selection, normalization, PCA, KNN graph, clustering, embeddings, and marker genes.
Beyond that core path, Scarf 1.0 covers:

- **Mapping and label transfer**: project query cells onto a reference, transfer
  labels and metadata, and build reusable reference atlases
- **Dataset integration**: merge batches or studies, correct with Harmony or partial
  PCA, combine modalities with SNN/WNN, and score the result with integration metrics
- **Pseudotime and expression dynamics**: order cells along a trajectory, group
  features into dynamic modules, and resolve competing fates against several sinks

Clustering, embeddings, mapping, and multimodal integration all read the same KNN
graph, so those results stay consistent with each other.

:::{image} _static/overview.svg
:width: 75%
:align: center
:alt: A count matrix split into compressed chunks read in parallel by incremental algorithms, producing a neighbourhood graph that feeds UMAP and tSNE, clustering, imputation, mapping, subsampling, and pseudotime ordering
:::

:::{admonition} Citation
Dhapola, P., Rodhe, J., Olofzon, R. et al. Scarf enables a highly memory-efficient
analysis of large-scale single-cell genomics data. Nat Commun 13, 4616 (2022).
<https://doi.org/10.1038/s41467-022-32097-3>
:::

## Why Scarf

**Large datasets on constrained resources.** Scarf is built for analyses that run into
millions and tens of millions of cells on hardware that cannot hold them in memory.
Counts and results live in a [Zarr] store on disk or object storage, and each step
streams what it needs under an explicit memory budget instead of loading the matrix.
The full analysis (cell filtering, feature selection, normalization, PCA, KNN graph,
UMAP, Leiden clustering, and marker genes) has been measured end to end. Graph
construction sets most of the memory ceiling, so peak memory grows slowly across two
orders of magnitude of cells.

| Cells | Peak memory | End-to-end wall time |
|---|---|---|
| 100k | ~7 GiB | ~15 min |
| 500k | ~25 GiB | ~47 min |
| 5M | ~33 GiB | ~8.2 h |
| 10M | ~36 GiB | ~22.8 h |

These runs used 8 CPU cores against a Zarr store on object storage. Your numbers will
depend on hardware, storage, and parameters.

**Provenance for analyses that branch.** At this scale one dataset fans out into many
branches: different filters, feature sets, clustering resolutions, and references.
Scarf records every step as an artifact carrying its operation, parameters, and inputs,
so you can trace how any result was produced. An identical request reuses the completed
artifact instead of recomputing it. That record is machine-readable and reuse is
deterministic, which is what lets an automated or agentic driver sweep parameters and
still account for everything that ran.

**Remote-first, without a local copy.** A Zarr store can live on S3-compatible object
storage, and Scarf reads and writes it in place. You can point at a shared or public
store, leave the counts where they are, and still compute locally: `mount_datastore`
gives you a writable store of your own that reads counts from the remote source, so
only your metadata and results land on local disk. Steps that make several passes over
the same data, such as PCA, stage that data into local scratch first so the repeat
passes do not go back over the network.

**Results persist as you work.** Every step writes back into the store, so you can
stop, inspect intermediate output, and resume later without recomputing. Filtering
marks cells inactive instead of deleting them, so quality-control choices stay
reversible.

**Methods that hold their memory ceiling.** Scarf brings SG-t-SNE and Paris clustering
into single-cell workflows alongside UMAP, densMAP, t-SNE, and Leiden. Paris is
implemented from scratch in Scarf, working on the graph in contraction rounds so
clustering millions of cells does not spike memory the way an in-memory hierarchy
would.

## What this documentation does not cover

- Raw FASTQ processing or alignment (use Cell Ranger, STARsolo, alevin-fry, or similar)
- Methods Scarf does not ship, including scVI, Scanorama, RNA velocity, and full DE
  pipelines with multiple-testing correction. Export a subset (for example with
  `to_anndata` / `to_h5ad`) and continue in Scanpy or another tool when you need those methods.

## Contributing and feedback

Report bugs and request features on [GitHub issues]. See {doc}`developers/contributing`
for how to build and execute documentation pages locally.

[pypi]: https://img.shields.io/pypi/v/scarf.svg
[pypiLink]: https://pypi.org/project/scarf
[docs]: https://readthedocs.org/projects/scarf/badge/?version=latest
[docsLink]: https://scarf.readthedocs.io
[stars]: https://img.shields.io/github/stars/parashardhapola/scarf?style=social
[github]: https://github.com/parashardhapola/scarf
[Source code on Github]: https://github.com/parashardhapola/scarf
[Zarr]: https://zarr.readthedocs.io
[GitHub issues]: https://github.com/parashardhapola/scarf/issues
