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

# Graph Imputation Primer

Single-cell RNA sequencing measures mRNA counts cell by cell, but the resulting capture of mRNA molecules is sparse (anywhere from 10-40% on average). Many genes in many cells record zero counts even when the gene is expressed, because its transcripts were simply not captured and sequenced. This dropout means observed zeros mix true absence with missed molecules, and biologically coherent marker signals can look patchy and unreliable on a UMAP.

Graph imputation addresses dropout by borrowing information across similar cells. The general concept of graph imputation is this, in which it helps to address noisy transcriptomic measurements by propagating expression signals across a network of transcriptomically similar cells. SCARF does this through building a k-nearest-neighbors graph connecting cells with similar profiles, converts it into a transition matrix where each cell distributes weight across its neighbors, and diffuses each feature over that graph. The imputed value for a cell is therefore a weighted average of its graph neighborhood rather than its own counts alone. This allows for one to observe the overall transcriptional similarity of the neighborhood.

It is important to note that graph imputation is strictly a visualization and exploratory aid, as it creates no new molecular observations and must not be used for any downstream differential expression or marker significance testing.

# Smoothing sparse CD4 expression with graph diffusion

In this tutorial, we utilize a prepared PBMC analysis to smooth the expression of the T-cell marker CD4 and check whether diffusion fills missing gaps inside the expected populations of where we would see CD4. CD4 is a good teaching example because it is dropout-prone and biologically tricky, as it serves one of the key examples of how mRNA expression can differ heavily from protein expression.

During this tutorial, it's important to note that this is for exploratory analysis, and that this information can't be solely used to assign cell types. For example, since human monocytes also express CD4, the signal we observe could be from T-cells or monocytes; Thus, multiple genes are required to properly annotate a cluster, a point further discussed in {doc}`annotation`. The core {doc}`scrna_seq` workflow shows the PBMC map this page uses, and {doc}`graph_construction` covers how the underlying neighborhood graph is chosen and built.

## Open the prepared result

Diffusion needs the completed k-nearest-neighborhood graph, thus we pull that information alongside its 2-dimensional embedding (the UMAP).

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

## Conduct graph diffusion on CD4 & compare observed and diffused values

For graph imputation in SCARF, the key hyperparameter is `t`, which controls diffusion depth. We can conceptually think about our values of `t` as the number of diffusion steps. During the diffusion progress, when `t` = 1, each cell is averaging only its directly adjacent cells. When `t` = 2, each step has each  2-hops, thus concurrently averages a neighbor of a neighbor and so on and so forth for greater `t`, resulting in the smoothing over a wider neighborhood, rather than the adjacent cells. As we approach `t` = 3, each cell now averages 3-hops, resulting in a broader smoothing of the global signal and potential borrowing from unrelated populations. 

SCARF's default is `t` = 2

```{code-cell}
diffusion_operators = {
    t: ds.run_diffusion_operator(graph, t=t)
    for t in (1, 2, 3)
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

The summary table above can be read from top to bottom, with the first column representing the observed data. We see that generally, when `t` = 1, the mean should look about the same as the observed mean, while the max drops a little as peak signal spreads to directly adjacent cells. When `t` = 2, the max drops further and more zeros fill in, since each cell now pulls from a wider neighborhood, thus the max drops further as more cells are added to the average. As we approach `t` = 3, even more zeros fill in, but this filling happens by design as each cell reaches further, so more filling alone does not mean a better result. Ideally, you want to find the balance between the t that fills the gaps inside your neighborhood that already has expression, without pulling the signal from other clusters.

Higher `t` values increase diffusion, which can blend distinct phenotypes, creating smooth "gradients" between cell types that are biologically distinct; comparing these results against our observed or "true" data is critical. This means that as `t` approaches infinity, the resulting graph collapses into a global average where every cell in the connected graph now has the nearly the exact same global average expression.

**With this understanding, we can now see our smoothing results in the form of UMAPs below.**

To compare our observed vs. diffusion values throughout our clusters of interest, we can first start by viewing our clustering results to gain a population context for the CD4 panels below.

```{code-cell}
ds.plots.embedding(
    run=run,
    layout="umap",
    color_by="clusters",
    legend_loc="on_data"
)
```

Now, using this UMAP as a starting point, we can compare our observed vs. diffused values of CD4 visually.

```{code-cell}
imputation_comparison = ds.plots.embedding(
    layout=run["umap"],
    color_by=[
        "CD4",
        "CD4_imputed_t1",
        "CD4_imputed_t2",
        "CD4_imputed_t3",
    ],
    n_columns=4,
    color_scale=splt.ColorScale(scope="shared"),
    sort_values=True,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    imputation_comparison.axes.values(),
    ("Observed CD4", "Diffusion t=1", "Diffusion t=2", "Diffusion t=3"),
    strict=True,
):
    axis.set_title(title)
imputation_comparison.figure
```

The imputed panels should fill gaps inside the same high-expression neighborhoods visible in the observed panel. Use the cluster map to check that high CD4 stays inside clusters of interest rather than spreading into unrelated populations. Signal across unrelated clusters indicates excessive diffusion or a graph that does not represent the intended biology.

## Important caveats to consider regarding imputation

- **The Differential Expression Trap (Loss of Independence):** Never run differential expression or marker gene testing on imputed values. Imputation replaces an independent observation with a weighted linear combination of neighboring cells. This destroys the statistical independence ({math}`N`) assumed by tests (Wilcoxon, DESeq2, MAST), artificially collapses within-group variance, and deflates p-values into astronomical, meaningless false positives ({math}`p < 10^{-200}`).
- **Hallucinated Co-Expression (Spurious Correlations):** If two completely unrelated genes are expressed in overlapping subsets of cells in the same neighborhood, diffusing both genes over the same graph will mathematically force their expression vectors to align. This can manufacture artificial regulatory networks and false gene-gene correlations.
- **Doublet Bridges and Edge Leakage:** In single-cell graphs, unresolved doublets (e.g., a T-cell/B-cell doublet) act as physical bridges between distinct graph clusters. Graph diffusion treats these bridges as legitimate highways, allowing CD4 signal to "bleed" directly across the bridge into B-cell clusters.
