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

# Graph Imputation Primer

Single-cell RNA sequencing measures mRNA counts cell by cell, but the resulting capture of mRNA molecules is sparse (anywhere from 10-40% on average). Many genes in many cells record zero counts even when the gene is expressed, because its transcripts were simply not captured and sequenced. This dropout means observed zeros mix true absence with missed molecules, and biologically coherent marker signals can look patchy and unreliable on a UMAP.

Graph imputation addresses dropout by borrowing information across similar cells. The general concept of graph imputation is this, in which it helps to address noisy transcriptomic measurements by propagating expression signals across a network of transcriptomically similar cells. SCARF does this through building a k-nearest-neighbors graph connecting cells with similar profiles, converts it into a transition matrix where each cell distributes weight across its neighbors, and diffuses each feature over that graph. The imputed value for a cell is therefore a weighted average of its graph neighborhood rather than its own counts alone, reflecting the overall transcriptional similarity of the neighborhood.

It is important to note that graph imputation it is strictly a visualization and exploratory aid, as it creates no new molecular observations and must not be used for any downstream differential expression or marker significance testing.

# Smoothing sparse CD4 expression with graph diffusion

In this tutorial, we utilize a preconducted PBMC analysis to smooth the expression of the T-cell marker CD4 and check whether diffusion fills missing gaps inside the expected populations of where we would see CD4. CD4 is a good teaching example because it is dropout-prone and biologically tricky, as it serves one of the key examples of how mRNA expression can differ heavily from protein expression.

During this tutorial, its important to note that since human monocytes also express CD4, multiple genes are required to properly annotate a cluster, a point further discussed in in {doc}`annotation`. The core {doc}`scrna_seq` workflow shows the broad PBMC map this page reuses, and {doc}`graph_construction` covers how the underlying neighborhood graph is chosen.

## 1. Open the prepared baseline

Diffusion needs a the completed k-nearest-neighbourhood graph, thus we pull that information alongside its 2 dimensional embedding (the UMAP).


```{code-cell}
import pandas as pd

import scarf
import scarf.plotting as splt

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
graph = run["connectivity_map"]
```

## 2. Diffuse one feature

For graph imputation in SCARF, the key hyperparameter is `t`, which controls diffusion depth. 



 the number of graph steps each cell's signal mixes across.
`t=1` is a light touch over immediate neighbors, the default `t=2` smooths a little further,
and `t=4` spreads signal widely and can erase real boundaries between populations.
A larger operator is also denser in memory, so if a large `t` raises a memory error,
retry with a smaller `t` or a larger datastore memory budget.

Read the summary table in the next cell as a depth check. Mean should stay near the observed
level while max falls with `t` as peak signal spreads across neighbors. `zero_fraction`
falls and `filled_zeros` rises with `t` by construction, so more filling is not by itself
better. Prefer the smallest `t` that fills gaps inside the same high-expression
neighborhoods without pushing signal into unrelated clusters.

```{code-cell}
diffusion_operators = {
    t: ds.run_diffusion_operator(graph, t=t)
    for t in (1, 2, 4)
}
diffused_by_t = {
    t: ds.get_imputed(feature_name="CD4", diffusion=diffusion)
    for t, diffusion in diffusion_operators.items()
}
for t, values in diffused_by_t.items():
    ds.cells.insert(f"CD4_imputed_t{t}", values, key="I", overwrite=True)
```

```{code-cell}
observed = ds.get_cell_vals(from_assay="RNA", cell_key="I", k="CD4")
cd4_series = {
    "Observed CD4": observed,
    **{f"Diffusion t={t}": values for t, values in diffused_by_t.items()},
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

The frozen selected clustering gives population context for the CD4 panels below.

```{code-cell}
ds.plots.embedding(
    run=run,
    layout="umap",
    color_by="clusters",
)
```

The comparison uses the exact UMAP artifact from the same run while coloring by the live columns
created above.

```{code-cell}
imputation_comparison = ds.plots.embedding(
    layout=run["umap"],
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

The result depends on the immutable cell selection and graph captured by each diffusion artifact.
Repeating `run_diffusion_operator` with the same graph and `t` reuses the complete stored artifact.
Pass that exact ref to `get_imputed`, or use `load_diffusion_operator` when direct sparse-matrix work
is needed.
Inserted columns such as `CD4_imputed_t2` are explicit cell metadata.
Do not interpret a nonzero imputed value as detection in that cell, use it for marker significance, or feed it to replicate-aware differential expression.
The `filled_zeros` column above is the concrete count of that mismatch for CD4 at each `t`.
