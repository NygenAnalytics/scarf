# Scarf

Single-cell analysis that stays on disk.

Scarf runs scRNA-seq, scATAC-seq, and CITE-seq neighbourhood-graph workflows on
[Zarr](https://zarr.readthedocs.io) stores, locally or on S3-compatible object storage.
Counts, graphs, and embeddings persist as you go, so atlas-scale analysis does not require
loading the full matrix into memory.

If you already use [Scanpy](https://scanpy.readthedocs.io): Scarf covers the core path
(QC → HVGs → graph → UMAP/Leiden → markers → mapping) with a lower memory ceiling and
native remote stores. Export with `to_anndata` / `to_h5ad` when you need the wider
Scanpy ecosystem. See [Coming from Scanpy or Seurat](https://scarf.readthedocs.io/en/latest/scarf_and_scanpy.html)
for a stage-by-stage translation.

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

ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_leiden_cluster",
)
```

After `import scarf.plotting as splt`, the equivalent standalone call is
`splt.embedding(ds, layout_key="RNA_UMAP", color_by="RNA_leiden_cluster")`.

Same path with more explanation: [docs quick start](https://scarf.readthedocs.io/en/latest/quickstart.html).

## Why Scarf

- **Large datasets on modest RAM**: the full pipeline (QC, HVGs, graph, UMAP, Leiden, markers) fits inside a single 64 GiB machine from 100k to 10M cells. Peak memory is set by graph construction and barely grows with dataset size.
- **Remote-first Zarr**: keep the store on S3-compatible object storage and analyze it in place, without a local full copy.
- **One graph, reused everywhere**: `make_graph` builds a single KNN graph that embeddings, clustering, mapping, and WNN/SNN integration all read from, so those results stay concordant.
- **Persistent results**: every step writes back into the store, so you can stop, inspect, and resume without recomputing. Filtering marks cells inactive rather than deleting them.

| Cells | Peak RAM | End-to-end |
|---|---|---|
| 100k | ~7 GiB | ~15 min |
| 500k | ~25 GiB | ~45 min |
| 1M | ~28 GiB | ~1.5 h |
| 2.5M | ~33 GiB | ~2.5 h |
| 5M | ~33 GiB | ~4.5 h |
| 10M | ~37 GiB | ~7 h |

On 8 CPU cores. Measured through 2.5M cells; 5M and 10M projected from the same curve. Full method and per-stage details in [`profiling/LEARNINGS.md`](profiling/LEARNINGS.md).

Graph-centric methods available: UMAP, densMAP, tSNE, Leiden and Paris clustering, Harmony, WNN/SNN multimodal integration, reference mapping and label transfer, and TopACeDo subsampling.

## Docs

Start here: [Installation](https://scarf.readthedocs.io/en/latest/installation.html) ·
[Quick start](https://scarf.readthedocs.io/en/latest/quickstart.html) ·
[Coming from Scanpy or Seurat](https://scarf.readthedocs.io/en/latest/scarf_and_scanpy.html) ·
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
