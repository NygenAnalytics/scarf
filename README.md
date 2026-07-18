# Scarf

Single-cell analysis that stays on disk.

Scarf runs scRNA-seq and CITE-seq neighbourhood-graph workflows on [Zarr](https://zarr.readthedocs.io)
stores, locally or on S3-compatible object storage. Counts, graphs, and embeddings persist
as you go, so atlas-scale analysis does not require loading the full matrix into memory.

If you already use [Scanpy](https://scanpy.readthedocs.io): Scarf covers the core path
(QC → HVGs → graph → UMAP/Leiden → markers → mapping) with a lower memory ceiling and
native remote stores. Export with `to_anndata` / `to_h5ad` when you need the wider
Scanpy ecosystem. Stage mapping: [Scarf and Scanpy](https://scarf.readthedocs.io/en/latest/scarf_and_scanpy.html).

[![PyPI](https://img.shields.io/pypi/v/scarf.svg)](https://pypi.org/project/scarf)
[![Docs](https://readthedocs.org/projects/scarf/badge/?version=latest)](https://scarf.readthedocs.io)
[![Tests](https://github.com/parashardhapola/scarf/actions/workflows/pytest.yml/badge.svg)](https://github.com/parashardhapola/scarf/actions/workflows/pytest.yml)
[![Coverage](https://codecov.io/gh/parashardhapola/scarf/branch/master/graph/badge.svg?token=ZvJXuYq3pd)](https://codecov.io/gh/parashardhapola/scarf)

## Install

Python 3.12+.

```bash
uv pip install "scarf[extra]"
```

## Quick start

```python
import scarf
import scarf.plotting as splt

reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
scarf.CrToZarr(reader, zarr_loc="data.zarr").dump(batch_size=1000)

ds = scarf.DataStore("data.zarr", nthreads=4, min_features_per_cell=10)
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures"],
    highs=[15000, 4000],
    lows=[1000, 500],
)
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key="hvgs", k=11, dims=15, n_centroids=100)
ds.run_umap(n_epochs=250, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)

splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="RNA_leiden_cluster",
)
```

Same path with more explanation: [docs quick start](https://scarf.readthedocs.io/en/latest/quickstart.html).

## What Scarf is good at

- **Remote-first Zarr**: analyze stores on S3 without a local full copy
- **Large matrices on modest RAM**: measured through 2.5M cells in 64 GiB (~4 h end-to-end on object storage); details in [`profiling/LEARNINGS.md`](profiling/LEARNINGS.md)
- **Graph-centric workflows**: UMAP, Leiden/Paris, Harmony, WNN/SNN, mapping, TopACeDo subsampling
- **Persistent results**: intermediates live in the store, not only in an in-memory object

## Docs

Start here: [Installation](https://scarf.readthedocs.io/en/latest/installation.html) ·
[Quick start](https://scarf.readthedocs.io/en/latest/quickstart.html) ·
[Scarf and Scanpy](https://scarf.readthedocs.io/en/latest/scarf_and_scanpy.html) ·
[scRNA-seq](https://scarf.readthedocs.io/en/latest/tutorials/scrna_seq.html) ·
[API](https://scarf.readthedocs.io/en/latest/reference/api.html)

Also: [CITE-seq](https://scarf.readthedocs.io/en/latest/tutorials/cite_seq.html),
[integration](https://scarf.readthedocs.io/en/latest/tutorials/choosing_integration_methods.html).

## Citation

[Dhapola et al., Nature Communications (2022)](https://doi.org/10.1038/s41467-022-32097-3)

## Support

[GitHub issues](https://github.com/parashardhapola/scarf/issues)

Scarf is open source and maintained by [Nygen](https://nygen.io). Nygen's hosted product,
[ScarfWeb](https://www.nygen.io/products/scarfweb), is built on Scarf; it is optional and not
required to use this library.
