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

## Setup

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)
```

## 1. Open the pre-analyzed store

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors.

The store contains an analysis saved as `docs_default`. We use its selected cells and UMAP so
we can focus on cell-cycle scoring.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
analysis_run = ds.pipeline.open(label="docs_default")
```

```{code-cell} ipython3
ds.plots.embedding(
    run=analysis_run,
    color_by="clusters",
)
```

## 2. Run cell-cycle scoring

Scarf's scorer follows the same general strategy as
[Scanpy's cell-cycle scorer](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes_cell_cycle.html):

- Match the supplied S and G2M markers, using Scarf's human-and-mouse lists by default.
- Group genes into bins with similar mean log-normalized expression across the selected cells.
- Sample control genes from the same expression bins as each phase's markers.
- Subtract mean control expression from mean marker expression for each cell and phase.

Cells with two negative scores are assigned G1. Otherwise, G2M wins when its score exceeds the S
score, and the remaining cells are assigned S.

```{code-cell} ipython3
cell_cycle_ref = ds.run_cell_cycle_scoring(analysis_run["analysis_cell_selection"])
cell_cycle_values = ds.load_artifact(cell_cycle_ref)
s_score = np.asarray(cell_cycle_values["s_score"][:])
g2m_score = np.asarray(cell_cycle_values["g2m_score"][:])
phase = np.asarray(cell_cycle_values["phase"][:]).astype(str)
```

Two markers in the bundled G2M list are absent from this assay.
The warning is expected, and Scarf scores the cells with the remaining markers.

The returned reference identifies the saved phases and scores. Keep it to load the same result
later. Scoring requires a writable datastore.

## 3. Visualize cell-cycle phases

Pass the result directly to the embedding plot to color cells by phase:

```{code-cell} ipython3
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by=cell_cycle_ref,
)
```

Look for populations enriched for S or G2M. Cycling cells may be concentrated in one population
or spread across several, depending on the tissue and experimental conditions.

Phase composition for the pipeline's selected clustering shows which groups are enriched for S or
G2M relative to G1:

```{code-cell} ipython3
pd.crosstab(analysis_run.cells.fetch("clusters"), phase, normalize="index")
```

Rows are cluster-wise phase fractions among the cells captured by the run.

## 4. Visualize phase-specific scores

The S and G2M score arrays are stored in the same artifact.

```{code-cell} ipython3
umap = analysis_run.cells.to_pandas_dataframe(["umap_1", "umap_2"])
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, values, title in (
    (axes[0], s_score, "S score"),
    (axes[1], g2m_score, "G2M score"),
):
    points = axis.scatter(umap["umap_1"], umap["umap_2"], c=values, s=3)
    axis.set_title(title)
    figure.colorbar(points, ax=axis)
figure.tight_layout()
figure
```

## Optional: compare with Scanpy scores

The rebuilt dataset retains cell-cycle scores calculated with Scanpy in the `S_score` and
`G2M_score` metadata columns. Plot both on the pipeline run's exact UMAP.

```{code-cell} ipython3
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by=["S_score", "G2M_score"],
    n_columns=2,
)
```

The Scanpy scores look similar to Scarf's.
Quantify the concordance:

```{code-cell} ipython3
pd.Series(
    {
        "S": np.corrcoef(s_score, ds.cells.fetch("S_score"))[0, 1],
        "G2M": np.corrcoef(g2m_score, ds.cells.fetch("G2M_score"))[0, 1],
    },
    name="Pearson r",
)
```

High correlation coefficients indicate a large degree of concordance between the scores obtained using Scanpy and Scarf.

## Common mistakes and limitations

- Applying a human or mouse gene set to data with incompatible feature names
- Interpreting a phase score as evidence of cell proliferation without checking the underlying genes
- Comparing scores across workflows with different gene sets or normalization

`run_cell_cycle_scoring` stores phase and both scores in one immutable artifact. Retain its exact ref
for loading and downstream analysis.
