---
description: Smooth sparse expression over a neighbourhood graph and compare it with observed values.
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

# Imputation by graph diffusion

Dropout makes observed single-cell expression sparse.
Scarf can diffuse a feature over the neighbourhood graph to reveal coherent regional patterns.
Imputation is a visualization and exploratory-analysis aid.
It does not create new molecular observations and should not replace counts in differential expression.

## 1. Standalone setup

Diffusion needs a neighbourhood graph, and the published PBMC store already carries one along with its UMAP.
Opening it is the whole setup.
Building the same {term}`analysis chain` from counts is covered in {doc}`scrna_seq`.

```{code-cell} ipython3
import pandas as pd
import scarf
import scarf.plotting as splt

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
```

## 2. Diffuse one feature

`t` controls diffusion depth.
A larger value mixes information over more graph steps and can erase real boundaries.

```{code-cell} ipython3
for t in (1, 2, 4):
    ds.cells.insert(
        f"CD4_imputed_t{t}",
        ds.get_imputed(feature_name="CD4", t=t),
        overwrite=True,
    )
```

```{code-cell} ipython3
observed = ds.get_cell_vals(from_assay="RNA", cell_key="I", k="CD4")
cd4_series = {
    "Observed CD4": observed,
    "Diffusion t=1": ds.cells.fetch("CD4_imputed_t1", key="I"),
    "Diffusion t=2": ds.cells.fetch("CD4_imputed_t2", key="I"),
    "Diffusion t=4": ds.cells.fetch("CD4_imputed_t4", key="I"),
}
cd4_summary = pd.DataFrame(
    {
        label: {
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "zero_fraction": float((vals == 0).mean()),
            "filled_zeros": int(((observed == 0) & (vals > 0)).sum()),
        }
        for label, vals in cd4_series.items()
    }
).T
cd4_summary
```

## 3. Compare observed and diffused values

Mean stays near the observed level while max falls with `t` as diffusion spreads peak signal across neighbours.
`zero_fraction` also falls with `t`.
`filled_zeros` counts active cells that were zero for observed CD4 and became nonzero after diffusion.
That count is the size of the nonzero-as-detection mistake for this feature.

Paris clusters on the published UMAP give the population context for the CD4 panels below.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_paris_cluster",
)
```

```{code-cell} ipython3
imputation_comparison = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[
        "CD4",
        "CD4_imputed_t1",
        "CD4_imputed_t2",
        "CD4_imputed_t4",
    ],
    n_columns=4,
    color_scale=splt.ColorScale(scope="shared"),
    sort_values=True,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    imputation_comparison.axes.values(),
    ("Observed CD4", "Diffusion t=1", "Diffusion t=2", "Diffusion t=4"),
    strict=True,
):
    axis.set_title(title)
imputation_comparison.figure
```

The imputed panels should fill gaps inside the same high-expression neighbourhoods visible in the observed panel.
Use the cluster map to check that high CD4 stays inside T-cell-like partitions rather than spreading into unrelated populations.
Signal across unrelated clusters indicates excessive diffusion or a graph that does not represent the intended biology.

## 4. Caveats

The result depends on the active cell selection, feature selection, and graph.
By default, each diffusion operator remains cached in memory for reuse across features.
Set `cache_operator=False` when memory matters more than repeated feature speed.
Inserted columns such as `CD4_imputed_t2` are explicit cell metadata.
Do not interpret a nonzero imputed value as detection in that cell, use it for marker significance, or feed it to replicate-aware differential expression.
The `filled_zeros` column above is the concrete count of that mismatch for CD4 at each `t`.
