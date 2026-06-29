# Scarf

Out-of-core, graph-first workflows for million-cell RNA-seq and CITE-seq.

[![PyPI](https://img.shields.io/pypi/v/scarf.svg)](https://pypi.org/project/scarf)
[![Docs](https://readthedocs.org/projects/scarf/badge/?version=latest)](https://scarf.readthedocs.io)
[![Tests](https://github.com/parashardhapola/scarf/actions/workflows/pytest.yml/badge.svg)](https://github.com/parashardhapola/scarf/actions/workflows/pytest.yml)
[![Coverage](https://codecov.io/gh/parashardhapola/scarf/branch/master/graph/badge.svg?token=ZvJXuYq3pd)](https://codecov.io/gh/parashardhapola/scarf)
[![Downloads](https://pepy.tech/badge/scarf)](https://pepy.tech/project/scarf)

## Installation

Requires Python 3.12+.

```bash
pip install scarf[extra]
```

## Features

- Analyze atlas-scale scRNA-seq on modest hardware (tested up to 4M cells)
- CITE-seq multimodal analysis with WNN and SNN integration
- Zarr-backed chunking for low memory use
- Parallel UMAP and SG-tSNE embeddings
- Harmony batch correction and partial PCA for merged datasets
- Cell projection and label transfer across datasets
- TopACeDo subsampling for representative cell selection
- Hierarchical clustering with interpretable dendrograms

## Quick start

Convert a Cell Ranger `filtered_feature_bc_matrix.h5` to Zarr, then run a minimal scRNA-seq workflow:

```python
import scarf

reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
scarf.CrToZarr(reader, zarr_loc="data.zarr").dump(batch_size=1000)

ds = scarf.DataStore("data.zarr", nthreads=4, min_features_per_cell=10)
ds.filter_cells(attrs=["RNA_nCounts", "RNA_nFeatures"], highs=[15000, 4000], lows=[1000, 500])

ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key="hvgs", k=11, dims=15, n_centroids=100)
ds.run_umap(n_epochs=250, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)

ds.plot_layout(layout_key="RNA_UMAP", color_by="RNA_leiden_cluster")
```

## Documentation

- [Installation guide](https://scarf.readthedocs.io/en/latest/install.html)
- [scRNA-seq tutorial](https://scarf.readthedocs.io/en/latest/vignettes/basic_tutorial_scRNAseq.html)
- [CITE-seq tutorial](https://scarf.readthedocs.io/en/latest/vignettes/multiple_modalities.html)
- [Integration guide](https://scarf.readthedocs.io/en/latest/integration_guide.html)
- [Full API](https://scarf.readthedocs.io/en/latest/api.html)

## Citation

If you use Scarf in your research, please cite [Dhapola et al., Nature Communications (2022)](https://doi.org/10.1038/s41467-022-32097-3).

## Support

Open an [issue on GitHub](https://github.com/parashardhapola/scarf/issues) if you run into problems.
