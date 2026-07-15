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

This page runs the minimal scRNA-seq pipeline. For a full walkthrough of data handling, see {ref}`scRNA-Seq workflow <scrna_seq_workflow>`.

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.fetch_dataset('tenx_5K_pbmc_rnaseq', save_path='scarf_datasets')
reader = scarf.CrH5Reader('scarf_datasets/tenx_5K_pbmc_rnaseq/data.h5')
scarf.CrToZarr(reader, zarr_loc='scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr').dump(batch_size=1000)
```

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr', nthreads=4, min_features_per_cell=10)
ds.filter_cells(attrs=['RNA_nCounts', 'RNA_nFeatures'], highs=[15000, 4000], lows=[1000, 500])
```

```{code-cell} ipython3
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
ds.run_umap(n_epochs=250, spread=5, min_dist=1, parallel=True)
ds.run_leiden_clustering(resolution=0.5)
```

```{code-cell} ipython3
result = splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
    show=False,
)
result.figure
```

For publication-oriented figures (shared color scales, dotplots, composition, export),
see {ref}`plotting with scarf.plotting <plotting_showcase>`.

For batch correction on merged datasets, see {ref}`integration methods guide <integration_guide>`.
