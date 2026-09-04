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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)
```

## 1. Open the pre-analyzed store

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors.

The rebuilt Zarr store is available from the `scarf_docs` Cytebase catalog. It contains a completed
pipeline run named `docs_default`. This page opens that current store directly and writes only the
cell-cycle artifact taught below.

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
- Bin genome-wide mean normalized expression across the selected cells.
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

The bundled list contains one marker that is absent from this assay.
The warning about one unmatched name is expected, and Scarf scores the cells with the remaining markers.

`DataStore.run_cell_cycle_scoring` owns persistence.
Its cell-cycle artifact has exactly the `feature_summary` and `cell_selection` artifact inputs; resolved S and G2M feature indexes plus `control_size`, `n_bins`, and `rand_seed` are parameters.
It requires a writable datastore and fails before planning with `Cell-cycle scoring requires a DataStore opened with zarr_mode='r+'` when opened read-only.
In contrast, a direct `Assay.score_features(...)` call computes blockwise in memory and does not create summaries, artifacts, or metadata columns.

## 3. Visualize cell-cycle phases

Cell-cycle phase remains in the returned artifact. Explicit colors keep the phase encoding
consistent with the composition summary below:

```{code-cell} ipython3
phase_colors = {
    "G1": "grey",
    "S": "salmon",
    "G2M": "green",
}

umap = analysis_run.cells.to_pandas_dataframe(["umap_1", "umap_2"])
plt.scatter(
    umap["umap_1"],
    umap["umap_2"],
    c=[phase_colors[value] for value in phase],
    s=3,
)
```

Cycling cells should form localized regions rather than be spread uniformly across the embedding.

Phase composition for the pipeline's selected clustering shows which groups are enriched for S or
G2M relative to G1:

```{code-cell} ipython3
pd.crosstab(analysis_run.cells.fetch("clusters"), phase, normalize="index")
```

Rows are cluster-wise phase fractions among the cells captured by the run.

## 4. Visualize phase-specific scores

The S and G2M score arrays are stored in the same artifact.

```{code-cell} ipython3
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

## 5. Compare with Scanpy scores

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
