---
description: Graph-based feature imputation with get_imputed.
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

(imputation)=

# Imputation

Scarf can impute feature values by diffusing expression along the KNN graph (MAGIC-style).
Use `get_imputed` after building a neighbourhood graph.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- A graph built from a suitable cell and feature subset

## What you will learn

- Build a graph for imputation
- Diffuse expression for a selected gene
- Store and compare the imputed values

## Dataset

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

scarf.fetch_dataset(
    'tenx_5K_pbmc_rnaseq',
    save_path='scarf_datasets',
    as_zarr=True,
)
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
```

## Guided steps

### 1. Build a graph

```{code-cell} ipython3
ds.filter_cells(attrs=['RNA_nCounts'], highs=[15000], lows=[1000], reset_previous=True)
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15)
ds.run_umap(n_epochs=150, parallel=True)
```

### 2. Impute a gene

```{code-cell} ipython3
imputed_cd4 = ds.get_imputed(feature_name='CD4', t=2)
ds.cells.insert('CD4_imputed', imputed_cd4, overwrite=True)
splt.embedding(ds, layout_key='RNA_UMAP', color_by='CD4', show=False).figure;
```

```{code-cell} ipython3
splt.embedding(ds, layout_key='RNA_UMAP', color_by='CD4_imputed', show=False).figure;
```

The `t` parameter controls diffusion depth. Higher values smooth more. The diffusion operator
is cached under the graph location in the Zarr store.

Use imputation when you need a smoother view of a sparse marker for visualization or for
ranking cells along a continuum. Do not use imputed values as input counts for differential
expression or for claiming that a gene was detected in a cell. A quick check is to compare
the raw and imputed UMAPs above: the imputed panel should retain the same high-expression
regions while filling gaps inside those neighborhoods.

## Common mistakes

- Imputing values before building the matching graph
- Treating imputed expression as a replacement for observed counts
- Using a large `t` value without checking how much it smooths local variation

## Saved results

The diffusion operator is cached with the graph. The `CD4_imputed` column is saved in cell
metadata after `insert`.

## Next steps

- {doc}`scrna_seq`
- {doc}`plotting`
