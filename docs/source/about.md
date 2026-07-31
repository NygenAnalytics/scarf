---
description: What Scarf is designed for, the workflows it supports, and its current boundaries.
---

# About Scarf

Scarf stands for Single Cell Analysis on Remote File Systems. It is designed for
single-cell analyses where the count matrix is too large, too remote, or too
valuable to duplicate for every branch of an analysis.

Scarf stores counts, metadata, and persisted results in Zarr. Operations stream
bounded blocks instead of loading a complete matrix into memory. The same
`DataStore` can hold alternative cell selections, feature selections, graph
chains, clustering resolutions, and mappings while recording which inputs and
parameters produced each result.

:::{image} _static/overview.svg
:width: 75%
:align: center
:alt: Compressed count-matrix chunks feed graph construction, embeddings, clustering, mapping, imputation, downsampling, and trajectory analysis
:::

## Supported workflows

Scarf provides complete workflows for:

- scRNA-seq quality control, feature selection, normalization, graph
  construction, embedding, clustering, and marker discovery
- scATAC-seq peak filtering, LSI-based graph construction, clustering, and
  gene-score analysis
- CITE-seq RNA and ADT processing with shared-nearest-neighbour or
  weighted-nearest-neighbour integration
- pseudotime ordering, expression dynamics, modules, and multi-sink fate
  probabilities
- dataset merging, Harmony or partial-PCA correction, integration metrics,
  reference mapping, and label transfer
- cell-cycle scoring, gene-set activity, imputation, downsampling, and
  pseudobulk export

## Why Scarf uses a persistent store

Single-cell analysis is a chain of dependent operations. Changing a cell filter
can change feature selection, PCA, neighbours, clustering, and markers. Scarf
records persisted results with their operation, scientific parameters, and
upstream inputs. Related branches can coexist in one datastore, and an identical
request can reuse a valid result rather than recomputing it.

Filtering is reversible: cells and features are marked by boolean selections
rather than deleted. Persisted outputs remain available between sessions, so an
analysis can stop after an expensive stage and continue later.

## Local, remote, and mounted data

A datastore can live on local disk or S3-compatible object storage. When a
remote source is read-only, `mount_datastore` creates a writable analysis layer
whose counts remain in the source while new metadata and results are stored in a
target you control. Multi-pass operations can use local scratch space without
turning that scratch cache into a second persistent datastore.

## Performance framing

Scarf is intended for laptops and single servers, including analyses with
millions of cells. Runtime and peak process memory depend on the selected
features, graph parameters, CPU count, compression, storage latency, and whether
feature-major `countsT` data is available. The public scaling guide reports
benchmarks with those resource envelopes rather than treating one result as a
universal expectation. See {doc}`concepts/scale_and_memory`.

## Current boundaries

Scarf starts from count matrices; it does not process FASTQ files or perform
alignment. Use tools such as Cell Ranger, STARsolo, or alevin-fry first.

Scarf does not ship every method in the single-cell ecosystem. It does not
provide scVI, Scanorama, RNA velocity, FRiP/TSS APIs, or a complete
replicate-aware differential-expression framework. Marker searches include
cell-level Mann-Whitney statistics, AUC, and within-group multiple-testing
correction, but those values are not a substitute for inference across
biological replicates. Export selected data to AnnData or another supported
format when a downstream method is outside Scarf.

## Citation

If Scarf contributes to an analysis, cite:

Dhapola, P., Rodhe, J., Olofzon, R. et al. Scarf enables a highly
memory-efficient analysis of large-scale single-cell genomics data. *Nature
Communications* 13, 4616 (2022).
<https://doi.org/10.1038/s41467-022-32097-3>

## Feedback and contributions

Use [GitHub issues](https://github.com/parashardhapola/scarf/issues) for bugs and
feature requests. See {doc}`developers/contributing` for development and
documentation checks.
