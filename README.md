# Scarf



Scarf is a Python framework for analysing single-cell RNA, ATAC, protein, and multi-omic data, from a few thousand cells to tens of millions.


| Problem                                                                        | How Scarf solves it                                                                                                          | What you get                                                                                     |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Your **dataset is larger than RAM**                                            | Out-of-core algorithms , and neighbour search streams from cell-major and gene-major layouts, inside a memory budget you set | No subsampling, so rare populations survive, [benchmarked to 10M cells](profiling/BENCHMARKS.md) |
| The **data is stored remotely** and requires downloading                       | Fetches only the chunks an operation touches, and writes results to a store you own                                          | Start analysing immediately, with one authoritative copy                                         |
| A **single parameter change costs hours** of computation                       | Each step is fingerprinted by its settings and inputs, so reuse is by content, not by layer name                             | Only what changed recomputes, and the old version stays for comparison                           |
| Sub-population analysis leaves **scattered copies that nobody can trace back** | Subsets are masks in one file, and every result carries the cells and parameters behind it                                   | A year later, a result still explains itself                                                     |




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

ds = scarf.DataStore(
    "s3://bucket/10M_cells.zarr",  # also gs://, hf://, or a local path
)
ds.pipeline.run()  # convenience: QC → HVGs → PCA → graph → UMAP → clustering → markers

ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_clusters",
)
```

Read the [scRNA-seq tutorial](https://scarf.readthedocs.io/en/latest/tutorials/scrna_seq.html) for a granular workflow, or [remote stores](https://scarf.readthedocs.io/en/latest/tutorials/remote_stores.html) for cloud setups.

## Documentation

Read workflow vignettes and API references on **[Read The Docs 📖](https://scarf.readthedocs.io/en/latest/)**

AI-assisted and autonomous workflows should start with **[Analysis with AI agents](https://scarf.readthedocs.io/en/latest/analysis_with_agents.html)**.

## Scarf's capabilities

| Area          | Methods                                                                                                                                         |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Modalities    | scRNA-seq, scATAC-seq, CITE-seq, matched multi-omics                                                                                            |
| Core workflow | Quality control, feature selection, normalization, PCA and LSI, KNN graph, UMAP, densMAP, t-SNE, Leiden, Paris, marker search                   |
| Integration   | Harmony, partial PCA, shared and weighted nearest neighbours, integration metrics                                                               |
| Mapping       | Symphony-style reference mapping, label transfer, projection diagnostics                                                                        |
| Trajectory    | Population Balance Analysis pseudotime, expression dynamics and modules, multi-sink fate probabilities                                          |
| Also included | Cell-cycle scoring, gene-set activity, graph-diffusion imputation, doublet scores, HTO demultiplexing, TopACeDo downsampling, pseudobulk export |




## Citation

Dhapola et al. Scarf enables a highly memory-efficient analysis of large-scale single-cell genomics data. *[Nature Communications* 13, 4616 (2022).](https://doi.org/10.1038/s41467-022-32097-3)

## Support

[GitHub issues](https://github.com/NygenAnalytics/scarf/issues)

Scarf is open source with [BSD 3-Clause License](LICENSE) and maintained by [Nygen](https://nygen.io).