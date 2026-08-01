# Scarf

<p align="left">
  <a href="https://pypi.org/project/scarf"><img src="https://img.shields.io/pypi/v/scarf.svg?color=4c72b0" alt="PyPI"></a>
  <a href="https://pypi.org/project/scarf"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-4c72b0.svg" alt="Python 3.12, 3.13, and 3.14"></a>
  <a href="https://scarf.readthedocs.io"><img src="https://readthedocs.org/projects/scarf/badge/?version=latest" alt="Docs"></a>
  <a href="https://github.com/parashardhapola/scarf/actions/workflows/pytest.yml"><img src="https://github.com/parashardhapola/scarf/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/parashardhapola/scarf"><img src="https://codecov.io/gh/parashardhapola/scarf/branch/master/graph/badge.svg?token=ZvJXuYq3pd" alt="Coverage"></a>
</p>

Scarf is a Python framework for analysing single-cell RNA, ATAC, protein, and multi-omic data, from a few thousand cells to tens of millions.

| Problem | How Scarf solves it | What you get |
| :-- | :-- | :-- |
| Your **dataset is larger than RAM** | Out-of-core algorithms , and neighbour search streams from cell-major and gene-major layouts, inside a memory budget you set | No subsampling, so rare populations survive, [benchmarked to 10M cells](profiling/BENCHMARKS.md) |
| The **data is stored remotely** and requires downloading | Fetches only the chunks an operation touches, and writes results to a store you own | Start analysing immediately, with one authoritative copy |
| A **single parameter change costs hours** of computation | Each step is fingerprinted by its settings and inputs, so reuse is by content, not by layer name | Only what changed recomputes, and the old version stays for comparison |
| Sub-population analysis leaves **scattered copies that nobody can trace back** | Subsets are masks in one file, and every result carries the cells and parameters behind it | A year later, a result still explains itself |

## Install

Python 3.12+.

```bash
uv venv --python 3.12
uv pip install --python .venv "scarf[extra]"
```

Detailed installation instructions [here](https://scarf.readthedocs.io/en/latest/installation.html)

## Quick start

```python
import scarf

reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
scarf.CrToZarr(reader, zarr_loc="data.zarr").dump()

ds = scarf.DataStore("data.zarr", nthreads=4)
ds.pipeline.run()

ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_clusters",
)
```

Read the [scRNA-seq tutorial](https://scarf.readthedocs.io/en/latest/tutorials/scrna_seq.html) for granular analysis workflow.

## Scarf's capabilities

<p align="center">
  <img src="https://raw.githubusercontent.com/parashardhapola/scarf/master/docs/source/_static/overview.png" alt="Compressed count chunks feed incremental algorithms and a neighbourhood graph that serves embedding, clustering, mapping, imputation, downsampling, and pseudotime" width="520">
</p>

| Area | Methods |
| :-- | :-- |
| Modalities | scRNA-seq, scATAC-seq, CITE-seq, matched multi-omics |
| Core workflow | Quality control, feature selection, normalization, PCA and LSI, KNN graph, UMAP, densMAP, t-SNE, Leiden, Paris, marker search |
| Integration | Harmony, partial PCA, shared and weighted nearest neighbours, integration metrics |
| Mapping | Symphony-style reference mapping, label transfer, projection diagnostics |
| Trajectory | Population Balance Analysis pseudotime, expression dynamics and modules, multi-sink fate probabilities |
| Also included | Cell-cycle scoring, gene-set activity, graph-diffusion imputation, doublet scores, HTO demultiplexing, TopACeDo downsampling, pseudobulk export |

## Documentation

Read workflow vignettes and API references on **[Read The Docs 📖](https://scarf.readthedocs.io/en/latest/)**

AI-assisted and autonomous workflows should start with **[Analysis with AI agents](https://scarf.readthedocs.io/en/latest/analysis_with_agents.html)**.

## Citation

Dhapola et al. Scarf enables a highly memory-efficient analysis of large-scale single-cell genomics data. [*Nature Communications* 13, 4616 (2022).](https://doi.org/10.1038/s41467-022-32097-3)

## Support

[GitHub issues](https://github.com/parashardhapola/scarf/issues)

Scarf is open source with [BSD 3-Clause License](LICENSE) and maintained by [Nygen](https://nygen.io).
