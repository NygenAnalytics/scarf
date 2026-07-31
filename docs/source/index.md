---
description: Start analysing single-cell data with Scarf.
---

[![PyPI][pypi]][pypiLink] [![Docs][docs]][docsLink] [![Github Stars][stars]][github]

# Scarf

Scarf is a Python package for memory-efficient analysis of single-cell RNA, ATAC,
and multimodal data. It streams Zarr-backed matrices from local or object storage
and records how persisted analysis results were produced.

## Start here

- {ref}`Install Scarf <installation>`
- Follow the {ref}`Quick start <quickstart>`
- Read {doc}`scarf_and_scanpy` if you already use Scanpy or Seurat
- Learn {doc}`about` for supported workflows, design choices, and limitations

## Core workflows

- {doc}`tutorials/scrna_seq`: move from counts through filtering, graph
  construction, clustering, and marker interpretation
- {doc}`tutorials/pseudotime`: order cells along a trajectory
- {doc}`tutorials/scatac_seq`: analyse chromatin accessibility
- {doc}`tutorials/cite_seq`: process RNA and protein measurements together

## Continue with a task

- {doc}`tutorials/data_integration`: merge datasets before correction
- {doc}`tutorials/mapping_and_label_transfer`: map query cells and transfer labels
- {doc}`tutorials/quality_control`: choose and inspect filtering decisions
- {doc}`tutorials/remote_stores`: work with data that does not fit on local disk
- {doc}`tutorials/plotting`: build and save interpretable figures
- {doc}`reference/api`: look up exact signatures and result contracts

[pypi]: https://img.shields.io/pypi/v/scarf.svg
[pypiLink]: https://pypi.org/project/scarf
[docs]: https://readthedocs.org/projects/scarf/badge/?version=latest
[docsLink]: https://scarf.readthedocs.io
[stars]: https://img.shields.io/github/stars/parashardhapola/scarf?style=social
[github]: https://github.com/parashardhapola/scarf
