---
description: Minimal scRNA-seq workflow in Scarf from count matrix to UMAP and clustering.
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(quickstart)=

# Quick start

This page runs a minimal scRNA-seq pipeline on a 5K PBMC dataset. For the full walkthrough,
see {doc}`tutorials/scrna_seq`. If you know Scanpy, skim {doc}`scarf_and_scanpy` first.

## Load counts into Zarr

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

scarf.fetch_dataset('tenx_5K_pbmc_rnaseq', save_path='scarf_datasets')
reader = scarf.CrH5Reader('scarf_datasets/tenx_5K_pbmc_rnaseq/data.h5')
scarf.CrToZarr(reader, zarr_loc='scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr').dump(batch_size=1000)
```

## Open the store and filter cells

`filter_cells` marks cells inactive (cell key `I`) rather than deleting them from the store.

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr', nthreads=4, min_features_per_cell=10)
ds.filter_cells(attrs=['RNA_nCounts', 'RNA_nFeatures'], highs=[15000, 4000], lows=[1000, 500])
```

## Neighbourhood graph, UMAP, and clustering

`mark_hvgs` selects highly variable genes. `make_graph` normalizes, runs PCA, and builds the
KNN graph that UMAP and Leiden reuse.

```{code-cell} ipython3
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
ds.run_umap(n_epochs=250, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)
```

## Plot the embedding

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

## What was saved

Typical columns and keys written by this pipeline:

- Cell QC: `RNA_nCounts`, `RNA_nFeatures` (and mito/ribo fractions when patterns match)
- Active cells: boolean key `I`
- Embedding: `RNA_UMAP1`, `RNA_UMAP2`
- Clusters: `RNA_leiden_cluster`

## Next steps

- Full scRNA-seq chapter: {doc}`tutorials/scrna_seq`
- Publication plotting: {ref}`plotting with scarf.plotting <plotting_showcase>`
- Batch correction: {ref}`integration methods guide <integration_guide>`
