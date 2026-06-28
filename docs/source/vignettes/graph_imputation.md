---
description: Graph-based feature imputation with get_imputed (MAGIC-style diffusion on the KNN graph).
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

## Graph-based imputation

Scarf can impute feature values by diffusing expression along the KNN graph, similar to MAGIC. Use `get_imputed` after building a graph.

```{code-cell} ipython3
%load_ext autotime

import scarf

scarf.fetch_dataset(
    'tenx_5K_pbmc_rnaseq',
    save_path='scarf_datasets',
    as_zarr=True
)

ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10
)
```

If the store is not preprocessed, run a minimal graph workflow:

```{code-cell} ipython3
ds.filter_cells(attrs=['RNA_nCounts'], highs=[15000], lows=[1000])
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15)
ds.run_umap(n_epochs=200, parallel=True)
```

Impute a gene and compare raw versus imputed expression on UMAP:

```{code-cell} ipython3
imputed_cd4 = ds.get_imputed(feature_name='CD4', t=2)

ds.cells.insert('CD4_imputed', imputed_cd4, overwrite=True)

ds.plot_layout(layout_key='RNA_UMAP', color_by='CD4', from_assay='RNA')
ds.plot_layout(layout_key='RNA_UMAP', color_by='CD4_imputed')
```

The `t` parameter controls diffusion depth (higher values smooth more). The diffusion operator is cached under the graph location as `magic_<t>` in the Zarr store.

---
That is all for this vignette.
