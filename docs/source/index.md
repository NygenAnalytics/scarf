---
description: What Scarf is designed for, how it works, the methods it implements, and its current boundaries.
---

[![PyPI][pypi]][pypiLink] [![Docs][docs]][docsLink] [![Github Stars][stars]][github]

# Scarf

Scarf stands for **S**ingle **C**ell **A**nalysis on **R**emote **F**ilesystems.
It is a Python package for memory-efficient analysis of single-cell RNA, ATAC, and multimodal data.
It is designed for analyses where the count matrix is too large, too remote, or too valuable to duplicate for every branch of an analysis.

Scarf stores counts, metadata, and persisted results in Zarr.
Operations stream bounded blocks instead of loading a complete matrix into memory.
The same `DataStore` can hold alternative cell selections, feature selections, {term}`analysis chains <analysis chain>`, clustering resolutions, and mappings while recording which inputs and parameters produced each result.

:::{image} _static/overview.svg
:width: 75%
:align: center
:alt: Compressed count-matrix chunks feed graph construction, embeddings, clustering, mapping, imputation, downsampling, and trajectory analysis
:::

## Start here

- {ref}`Install Scarf <installation>`
- Follow the {ref}`Quick start <quickstart>`
- Use {doc}`analysis_with_agents` to route an autonomous or AI-assisted analysis
- Read {doc}`scanpy_and_seurat` if you already use Scanpy or Seurat
- See {doc}`concepts/benchmarks` for measured end-to-end scale and stage timings

## Supported workflows

Scarf provides complete workflows for:

- scRNA-seq quality control, feature selection, normalization, graph construction, embedding, clustering, and marker discovery
- scATAC-seq peak filtering, LSI-based graph construction, clustering, and gene-score analysis
- CITE-seq RNA and ADT processing with shared-nearest-neighbour or weighted-nearest-neighbour integration
- pseudotime ordering, expression dynamics, modules, and multi-sink fate probabilities
- dataset merging, Harmony or partial-PCA correction with quantitative diagnostics, reference mapping, and label transfer
- cell-cycle scoring, gene-set activity, imputation, downsampling, and pseudobulk export

## Local, remote, and mounted data

Downloading and maintaining local copies of large collections can become a substantial part of a project.
Object storage is often used for distribution and archiving, with a full local copy required before analysis begins.

### Analyzing stores in place

A Scarf datastore can live on local disk, S3-compatible object storage, Google Cloud Storage, or Hugging Face, and be analysed in place.
When the source is read-only, `mount_datastore` creates a writable analysis layer: counts stay in the source while new cell metadata, feature metadata, and results are written to a target you control.
This is a logical mount rather than a filesystem symbolic link, and it does not copy the count matrices into each project.

### Scratch acceleration with local_cache

Operations that scan normalized expression repeatedly, such as PCA and LSI, can stage the blocks they need in local scratch space through `local_cache`.
Scratch is temporary I/O acceleration, not a second persistent datastore, and it does not download the complete source store.
Credentials are supplied when the store is opened and are not persisted inside it.

See {doc}`tutorials/remote_stores`.

## Planning memory, compute, and I/O

Scarf treats memory, CPU time, storage layout, and network access as one planning problem rather than optimising memory alone.
Counts are streamed in blocks.
RNA assays also store a gene-major `countsT` copy so gene-wise stages such as highly variable feature selection and marker search can stream by gene.
`mem_budget` shapes block size, write concurrency, and automatically sized feature batches.

`mem_budget` is a planning input, not a hard cap on process memory.
Graph structures, native libraries, Python objects, and allocator overhead consume memory in addition to the streamed blocks, so leave host headroom.
A smaller budget generally reduces peak memory and increases wall time.

Measured end-to-end timings and peak memory across input sizes are in {doc}`concepts/benchmarks`.
Resource controls and measurement caveats are explained in {doc}`concepts/memory_and_execution`.

## Provenance and reuse

Single-cell analysis is a chain of dependent operations.
Changing a cell filter can change feature selection, PCA, neighbours, clustering, and markers.
Scarf records each persisted result, or {term}`artifact`, with its operation, scientific parameters, and upstream inputs.
A clustering therefore names the graph it used, that graph names its neighbour query and reduction, and the reduction names its normalization and selected data.

Related branches can coexist in one datastore, and an identical request can {term}`reuse` a valid result rather than recomputing it.
Execution choices such as thread count or local scratch are recorded separately and do not change the scientific identity of a result.

This record is useful whenever the analysis is long-running, revisited after a gap, or executed through a pipeline or software agent, because the dependency chain can be inspected independently of the code or description that produced it.
See {doc}`analysis_with_agents` for the scientific decision and troubleshooting framework, {doc}`concepts/provenance` for the data model, and {doc}`tutorials/reuse_and_tracing` for an executable example.

## Selections and multi-scale analysis

Detailed work usually moves from a whole dataset to tissues, lineages, clusters, and smaller subpopulations.
Keeping a separate in-memory object for each subset makes it easy to lose track of which cells produced which result.

In Scarf, cell and feature selections are boolean masks stored in the datastore, and filtering marks cells inactive rather than deleting them.
Whole-dataset and subpopulation analyses can therefore share one object and one set of count matrices, while each stored result stays tied to the exact {term}`cell key` and {term}`feat_key` used to produce it.
Persisted outputs remain available between sessions, so an analysis can stop after an expensive stage and continue later.

## Implemented methods

Scarf implements a computational stage itself when an external implementation would require materializing the full matrix or would detach the result from its inputs.
It uses established libraries where they fit the streaming and provenance model.
The aim is one coherent execution path, not reimplementation for its own sake.

- **Reduction and graph construction:** streamed normalization, covariance (Gram-matrix) PCA with an incremental fallback, randomized streaming {term}`LSI`, approximate nearest-neighbour search, UMAP, {term}`densMAP`, and graph-based t-SNE.
- **Batch correction and mapping:** Scarf implementations of [Harmony](https://doi.org/10.1038/s41592-019-0619-0) and [Symphony-style](https://doi.org/10.1038/s41467-021-25957-x) fixed-reference mapping, with label transfer and mapping diagnostics.
- **Matched multi-omics:** {term}`SNN integration` and [Hao-inspired WNN](https://doi.org/10.1016/j.cell.2021.04.048) integration for two or more assays, with WNN reporting one per-cell weight per modality.
- **Clustering and sampling:** [Leiden](https://doi.org/10.1038/s41598-019-41695-z), a native implementation of [Paris hierarchical clustering](https://doi.org/10.48550/arXiv.1806.01664) with fixed and branch-adaptive cuts, and manifold-preserving {term}`TopACeDo` downsampling described in the [Scarf paper](https://doi.org/10.1038/s41467-022-32097-3).
- **Trajectory analysis:** memory-aware [Population Balance Analysis](https://doi.org/10.1073/pnas.1714723115), supervised terminal-state fate probabilities, pseudotime aggregation and modules, and [MAGIC-style graph diffusion](https://doi.org/10.1016/j.cell.2018.05.061) for imputation.
- **Statistics and diagnostics:** marker AUC, the [Mann-Whitney U test](https://doi.org/10.1214/aoms/1177730491) with [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) correction, doublet scores, cluster separability, and integration metrics informed by the [scIB benchmark](https://doi.org/10.1038/s41592-021-01336-8).
- **Gene sets, protein, and sample tags:** [AUCell](https://doi.org/10.1038/nmeth.4463) and [WAGGR](https://doi.org/10.1093/bioadv/vbac016) activity scores, CITE-seq processing, and [cell-hashing](https://doi.org/10.1186/s13059-018-1603-1) demultiplexing.

## Validation

Validation is method-specific.
Agreement measured for one method is not evidence of equivalence for the rest of the package, and none of these comparisons establish biological ground truth.

- HTO demultiplexing was compared with Seurat 5.5.1 on four GSE245108 matrices and matched 38,183 of 38,199 identities (99.9581%).
  All sixteen differences were k-means boundary cases at a single background cutoff.
- Symphony-style query correction is checked against a golden fixture generated with the R Symphony 0.1.3 `mapQuery` implementation, including a case with nonzero correction.
- Marker statistics are checked against SciPy, including continuity and large-tie corrections.
- Paris hierarchies are compared with the scikit-network reference implementation on tie-free graphs.
- WNN is checked against scalar reference calculations, a two-modality CITE-seq fixture, and a synthetic three-modality Seurat 5.5.1 fixture, plus modality-order symmetry, row-order invariance, and degenerate bandwidths.
  It follows the published weighting equations but is not bit-identical to Seurat defaults, as described in {ref}`the FAQ <faq>`.
- AUCell and WAGGR scores are checked against frozen decoupler 2.2 fixtures.
- Fate probabilities are checked against dense Dirichlet solutions on controlled graphs.

## Current boundaries

### Inputs and provenance

Scarf starts from count matrices; it does not process FASTQ files or perform alignment.
Use tools such as Cell Ranger, STARsolo, or alevin-fry first.

Provenance covers supported store-backed operations.
It does not capture arbitrary Python calculations, the full software environment, hardware, or study-level experimental records, and it describes computational relationships rather than scientific validity.

### Methods outside Scarf

Scarf does not ship every method in the single-cell ecosystem.
It does not provide scVI, Scanorama, RNA velocity, peak calling, FRiP/TSS APIs, or a complete replicate-aware differential-expression framework.
Marker searches report cell-level Mann-Whitney statistics, AUC, and within-group multiple-testing correction, which are not a substitute for inference across biological replicates.
{term}`WNN integration` requires two or more cell-aligned modalities, and Scarf does not align unpaired modalities.
Fate mapping needs user-supplied terminal states and does not infer them or use RNA velocity.
Imputation is intended for exploratory visualization rather than differential expression.
Export selected data to AnnData or another supported format when a downstream method is outside Scarf.

## Citation

If Scarf contributes to an analysis, cite:

Dhapola, P., Rodhe, J., Olofzon, R. et al. Scarf enables a highly memory-efficient analysis of large-scale single-cell genomics data.
*Nature Communications* 13, 4616 (2022).
<https://doi.org/10.1038/s41467-022-32097-3>

See {ref}`citation` for BibTeX and maintenance notes.

## Feedback and contributions

Use [GitHub issues](https://github.com/NygenAnalytics/scarf/issues) for bugs and feature requests.
See {doc}`developers/contributing` for development and documentation checks.

[pypi]: https://img.shields.io/pypi/v/scarf.svg
[pypiLink]: https://pypi.org/project/scarf
[docs]: https://readthedocs.org/projects/scarf/badge/?version=latest
[docsLink]: https://scarf.readthedocs.io
[stars]: https://img.shields.io/github/stars/NygenAnalytics/scarf?style=social
[github]: https://github.com/NygenAnalytics/scarf
