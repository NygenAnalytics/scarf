---
description: Score S and G2M gene programs and assign cell-cycle phases.
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

scarf.configure_output(level='WARNING', progress=True)
```

## 1. Fetch pre-analyzed data

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors.

The prepared Zarr store is available from the `scarf_docs` Cytebase catalog.
It already includes the top 2000 highly variable genes, a neighbourhood graph, and a UMAP embedding.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)

ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    default_assay='RNA'
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='clusters',
)
```

## 2. Run cell-cycle scoring

The cell cycle scoring function in Scarf is highly inspired by the [equivalent function](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes_cell_cycle.html) in Scanpy.
The cell cycle phase of each individual cell is identified following steps below:
- A list of S and G2M phase is provided to the function (Scarf already has a generic list of genes that works both for human and mouse data)
- Per-gene averages across `cell_key` cells are calculated genome-wide from the assay's current normalized values (default RNA: library-size, not log)
- Those averages are divided into `n_bins` bins
- A control set of genes is identified by sampling genes from same expression bins where phase's genes are present.
- S and G2M are then scored separately (two `score_features` calls): for each phase, the average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
- A phase score is calculated as: Ep-Ec
Cell cycle phase is assigned as follows (cells default to S, then rules override):
- G2M phase: G2M score > S score
- G1 phase: S score < 0 and G2M score < 0
- otherwise the cell stays S

```{code-cell} ipython3
ds.run_cell_cycle_scoring()
```

The bundled list contains one marker that is absent from this assay.
The warning about one unmatched name is expected, and Scarf scores the cells with the remaining markers.

## 3. Visualize cell-cycle phases

By default, the cell-cycle phase is stored in the cell metadata column `RNA_cell_cycle_phase`.
Explicit colors keep the phase encoding consistent with the composition plot below:

```{code-cell} ipython3
color_key = {
    'G1': 'grey',
    'S': 'salmon',
    'G2M': 'green',
}

ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_cell_cycle_phase',
    categorical_scale=splt.CategoricalScale(palette=color_key),
)
```

Cycling cells should be concentrated in the ductal region rather than spread uniformly across the embedding.

Phase composition per cluster shows which groups are enriched for S or G2M relative to G1:

```{code-cell} ipython3
ds.plots.composition(
    category_by='RNA_cell_cycle_phase',
    sample_by='clusters',
    kind='stacked',
    categorical_scale=splt.CategoricalScale(
        palette=color_key,
        order=['G1', 'S', 'G2M'],
    ),
)
```

Stacked bars are cluster-wise phase fractions among active cells; ductal-associated clusters should show a higher S/G2M share if the embedding pattern above holds.

## 4. Visualize phase-specific scores

The individual and S and G2M scores for each cell are stored under columns `RNA_S_score` and `RNA_G2M_score`.
We can visualize the distribution of these scores on the UMAP plots.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['RNA_S_score', 'RNA_G2M_score'],
)
```

## 5. Compare scores calculated with Scanpy

The dataset we downloaded, already had cell cycle scores calculated using Scanpy.
For example, the S phase scores are stored under the column `S_score`.
We can plot these scores on the UMAP.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='S_score',
)
```

The Scanpy scores look similar to Scarf's.
Quantify the concordance:

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

High correlation coefficients indicate a large degree of concordance between the scores obtained using Scanpy and Scarf.

## Common mistakes and limitations

- Applying a human or mouse gene set to data with incompatible feature names
- Interpreting a phase score as evidence of cell proliferation without checking the underlying genes
- Comparing scores across workflows with different gene sets or normalization

`run_cell_cycle_scoring` stores `RNA_cell_cycle_phase`, `RNA_S_score`, and `RNA_G2M_score` beside the scoring step so they can be reused by plots and downstream metadata queries.
