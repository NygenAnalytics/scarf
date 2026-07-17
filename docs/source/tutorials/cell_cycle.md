---
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

(cell_cycle)=

# Cell cycle

Score S-phase and G2M-phase gene sets to assign a cell-cycle phase to each cell.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a cell graph or embedding for visualization

## What you will learn

- Run cell-cycle scoring with Scarf's built-in gene sets
- Inspect phase labels and phase-specific scores
- Compare scores with values imported from another workflow

## Dataset

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')
scarf.__version__
```

## Guided steps

### 1. Fetch pre-analyzed data

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors. 

We have stored this data on Scarf's online repository for quick access. We processed the data to identify the highly variable genes (top 2000) and create a neighbourhood graph of cells. A UMAP embedding was calculated for the cells. 

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    save_path='./scarf_datasets',
    as_zarr=True,
)
```

```{code-cell} ipython3
ds = scarf.DataStore(
    f"scarf_datasets/bastidas-ponce_4K_pancreas-d15_rnaseq/data.zarr",
    nthreads=4, 
    default_assay='RNA'
)
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='clusters',
).figure;
```

---
### 2. Run cell-cycle scoring

The cell cycle scoring function in Scarf is highly inspired by the [equivalent function](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes_cell_cycle.html) in Scanpy. The cell cycle phase of each individual cell is identified following steps below:
- A list of S and G2M phase is provided to the function (Scarf already has a generic list of genes that works both for human and mouse data)
- Average expression of all the genes (separately for S and G2M lists) in across `cell_key` cells is calculated
- The log average expression is divided in `n_bins` bins
- A control set of genes is identified by sampling genes from same expression bins where phase's genes are
present.
- The average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
- A phase score is calculated as: Ep-Ec
Cell cycle phase is assigned as follows (cells default to S, then rules override):
- G2M phase: G2M score > S score
- G1 phase: S score < 0 and G2M score < 0
- otherwise the cell stays S

```{code-cell} ipython3
ds.run_cell_cycle_scoring()
```

---
### 3. Visualize cell-cycle phases

By default the cell cycle phase information in stored under cell attribute table under column/key `RNA_cell_cycle_phase`.
We can color the UMAP plot based on these values.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cell_cycle_phase',
).figure;
```

We can clearly see that cycling group of cells in the 'ductal' group. You can provide your own custom color mappings like below:

```{code-cell} ipython3
color_key = {
    'G1': 'grey',
    'S': 'salmon',
    'G2M': 'green',
}

splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cell_cycle_phase',
    categorical_scale=splt.CategoricalScale(palette=color_key),
).figure;
```

---
### 4. Visualize phase-specific scores

The individual and S and G2M scores for each cell are stored under columns `RNA_S_score` and `RNA_G2M_score`. We can visualize the distribution of these scores on the UMAP plots

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=['RNA_S_score', 'RNA_G2M_score'],
).figure;
```

---
### 5. Compare scores calculated with Scanpy

The dataset we downloaded, already had cell cycle scores calculated using Scanpy. For example, the S phase scores are stored under the column `S_score`. We can plot these scores on the UMAP.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='S_score',
).figure;
```

Unsurprisingly, these scores look very similar to those obtained through Scarf. Let's quantify the concordance below

```{code-cell} ipython3
import matplotlib.pyplot as plt
from scipy.stats import linregress

fig, axis  = plt.subplots(1, 2, figsize=(6,3))
for n,i in enumerate(['S_score', 'G2M_score']):
    x = ds.cells.fetch(f"RNA_{i}")
    y = ds.cells.fetch(i)
    res = linregress(x, y)
    
    ax = axis[n]
    ax.scatter(x, y, color=color_key[i.split('_')[0]])
    ax.plot(x, res.intercept + res.slope*x, label='fitted line', c='k')
    ax.set_xlabel(f"{i} (Scarf)")
    ax.set_ylabel(f"{i} (Scanpy)")
    ax.set_title(f"Corr. coef.: {round(res.rvalue, 2)} (pvalue: {res.pvalue})")

plt.tight_layout()
plt.show()
```

High correlation coefficients indicate a large degree of concordance between the scores obtained using Scanpy and Scarf

+++

## Common mistakes

- Applying a human or mouse gene set to data with incompatible feature names
- Interpreting a phase score as evidence of cell proliferation without checking the underlying genes
- Comparing scores across workflows with different gene sets or normalization

## Saved results

Scarf stores `RNA_cell_cycle_phase`, `RNA_S_score`, and `RNA_G2M_score` in cell metadata.

## Next steps

- {doc}`plotting`
- {doc}`annotation`
- {doc}`data_organization`
