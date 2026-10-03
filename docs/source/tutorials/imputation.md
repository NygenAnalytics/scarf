---
description: Smooth sparse expression over a neighborhood graph and compare it with observed values.
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

# Smooth sparse expression with graph diffusion

A zero RNA count can mean that a gene was not expressed or that its transcripts were
not captured. Graph diffusion borrows signal from neighboring cells to make regional
patterns easier to see. It creates a weighted average, not a new measurement.

Here we smooth CD4 expression in a prepared PBMC analysis. CD4 is useful for this
comparison because its RNA signal can be sparse, and it occurs in both T cells and
monocytes. Use other markers to interpret its location, as described in {doc}`annotation`.

## Open the prepared graph

```{code-cell}
# Open count stores and run Scarf analyses.
import scarf
# Select explicit fields and display options for plots.
from scarf.plotting import CellField, ColorScale, FeatureRef

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Open the saved analysis and its exact results.
run = ds.pipeline.open(label="docs_default")
# Keep the graph from the saved analysis.
graph = run["connectivity_map"]
# Inspect the opened store's cells and features.
ds
```

The graph and UMAP come from the same saved analysis. First view its clusters so we
have population boundaries to compare with the expression maps.

```{code-cell}
# Locate the PBMC clusters before comparing CD4 expression.
ds.plots.embedding(run=run, layout="umap", color_by="clusters", legend_loc="on_data")
```

## Start with the default smoothing

`run_diffusion_operator` defaults to two diffusion steps (`t=2`). Each step spreads
signal across graph neighbors. `get_imputed` applies that operator to one feature.

```{code-cell}
# Build the diffusion operator with the default two steps.
diffusion = ds.run_diffusion_operator(graph)
# Apply diffusion to normalized CD4 expression.
smoothed = ds.get_imputed(feature_name="CD4", diffusion=diffusion)
# Save the calculated values in the cell table.
ds.cells.insert("CD4_imputed_t2", smoothed, key="I", overwrite=True)
# Check the number of smoothed cells and their expression range.
{
    "cells": len(smoothed),
    "minimum": round(float(smoothed.min()), 4),
    "maximum": round(float(smoothed.max()), 4),
}
```

The inserted column lets us plot the smoothed values beside observed, normalized CD4
expression. Both panels share a color scale. In this prepared store, the active cells
(`I`) match the saved graph's cell selection.

```{code-cell}
# Compare observed CD4 with two-step smoothing on a shared color scale.
ds.plots.embedding(
    layout=run["umap"],
    color_by=[
        FeatureRef("CD4", label="Observed CD4"),
        CellField("CD4_imputed_t2", kind="continuous", label="Diffusion t=2"),
    ],
    n_columns=2,
    color_scale=ColorScale(scope="shared"),
    sort_values=True,
)
```

Look for smoother signal within regions that already express CD4. Then check whether
signal extends into unrelated populations. A filled gap does not mean that CD4 was
detected in that cell, and a smoother map is not automatically more accurate.

## Explore the effect of diffusion depth

A larger `t` lets signal travel through more graph steps. To judge whether the default
smooths too little or too much, compare it with one and three steps:

```{code-cell}
# Compute the two additional diffusion depths.
for t in (1, 3):
    # Build the operator for this diffusion depth.
    operator = ds.run_diffusion_operator(graph, t=t)
    # Smooth CD4 expression with this operator.
    values = ds.get_imputed(feature_name="CD4", diffusion=operator)
    # Save the calculated values in the cell table.
    ds.cells.insert(f"CD4_imputed_t{t}", values, key="I", overwrite=True)

# Confirm which smoothing depths are ready to compare.
{"diffusion steps": [1, 2, 3], "cells per column": len(values)}
```

Place the observed values beside all three smoothing depths:

```{code-cell} ipython3
# Name the three smoothed columns for the comparison plot.
diffusion_fields = [
    CellField(f"CD4_imputed_t{t}", kind="continuous", label=f"Diffusion t={t}")
    for t in (1, 2, 3)
]
# Compare observed CD4 with all three diffusion depths.
ds.plots.embedding(
    layout=run["umap"],
    color_by=[FeatureRef("CD4", label="Observed CD4"), *diffusion_fields],
    n_columns=4,
    color_scale=ColorScale(scope="shared"),
    sort_values=True,
)
```

The larger depths fill more of the patchy CD4 pattern in this example. Use the observed
panel and cluster map to judge where that added signal goes. Diffusion can blur a real
boundary if the graph connects cells from different populations. Revisit the graph in
{doc}`graph_construction` when none of these depths preserves the intended structure.

## Limits of smoothing

- Use observed counts, with an appropriate sample-level model, for differential
  expression. Smoothed values share information between cells and can give misleading
  statistical certainty.
- Smoothing can strengthen apparent correlations between genes. It does not establish
  a regulatory relationship.
- Doublets or unsuitable graph edges can spread signal between unrelated populations.
  Check these possibilities before interpreting a new gradient.
