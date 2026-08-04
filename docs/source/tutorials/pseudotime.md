---
description: Order cells along a supervised trajectory and identify genes that change with progression.
jupytext:
  formats: ipynb,md:myst
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

# Pseudotime ordering

Pseudotime represents progress through a continuous biological process. This
tutorial uses annotated start and terminal populations to orient one reliable
trajectory, then identifies genes that change with that ordering. It does not
infer terminal states or prove lineage relationships.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and UMAP embedding

## What you will learn

- Score supervised pseudotime with source and sink clusters
- Find features correlated with the ordering
- Interpret one expression checkpoint on the embedding

## Dataset

```{code-cell} ipython3
import scarf

scarf.configure_output(level='WARNING', progress=True)
```

---
## 1) Fetch pre-analyzed data

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors.

The prepared Zarr store is available from the `scarf_docs` Cytebase catalog. It already
includes the top 2000 highly variable genes, a neighbourhood graph, and a UMAP embedding. 

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

The store includes cell-type annotations in the `clusters` column:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ['clusters'],
    key='I'
)['clusters'].value_counts()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='clusters',
    legend_loc='on_data',
)
```

---
## 2) Estimate pseudotime ordering

Scarf uses a memory-efficient implementation of the [PBA algorithm](https://github.com/AllonKleinLab/PBA) ([Weinreb et al. 2018, PNAS](https://doi.org/10.1073/pnas.1714723115)) to estimate a pseudotime ordering. `run_pseudotime_scoring` works with any assay that has a neighborhood graph.

The method is supervised: the source groups define the beginning and the sink
groups define the end. Those choices orient the result. Here, ductal cells
represent the progenitor pool and the hormone-expressing states represent the
endpoints.

```{code-cell} ipython3
pseudotime = ds.run_pseudotime_scoring(
    source_sink_key='clusters',
    sources=['Ductal'],
    sinks=['Alpha', 'Beta', 'Delta'],
)
```

Any cell metadata column with group labels works for `source_sink_key`, including a Scarf
clustering such as `RNA_leiden_cluster`. Without provided annotations, read the cluster
labels off the embedding and pick the ones at each end of the trajectory. Every group named
in `sources` or `sinks` has to be present among the scored cells, otherwise the call fails
rather than guessing.

The scores are stored under the generated column named by
`pseudotime.pseudotime_key` (by default `RNA_pseudotime`). The companion key
`pseudotime.validity_key` identifies cells that were scored.

```{code-cell} ipython3
{
    "pseudotime_key": pseudotime.pseudotime_key,
    "validity_key": pseudotime.validity_key,
    "valid cells": int(ds.cells.fetch_all(pseudotime.validity_key).sum()),
}
```

Values should progress from the ductal region toward the endocrine endpoints.
A disconnected or internally reversed pattern is a reason to revisit the graph
and endpoint choices. Restrict the embedding to scored cells with
`subset_by=pseudotime.validity_key`.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=pseudotime.pseudotime_key,
    subset_by=pseudotime.validity_key,
)
```

Compare the score distribution across annotated populations. Early source
clusters should sit lower than the sink populations.

```{code-cell} ipython3
ds.plots.distribution(
    keys=pseudotime.pseudotime_key,
    group_by='clusters',
    subset_by=pseudotime.validity_key,
    kind='violin',
)
```

---
## 3) Identify pseudotime correlated features

`run_pseudotime_marker_search` calculates a correlation coefficient and p-value
for each selected feature against the ordering. Features that fail the
minimum-cell or variance checks remain untested with `NaN` p-values.
Benjamini-Hochberg adjustment runs only over tested features.

```{code-cell} ipython3
markers = ds.run_pseudotime_marker_search(
    cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
)
```

The correlations, raw p-values, and adjusted p-values are saved in feature
metadata. Their generated column names are available as
`markers.correlation_key`, `markers.p_value_key`, and
`markers.p_value_adjusted_key`. The returned `markers.table` contains the same
values with feature names.

Count how many features were tested versus left as `NaN`:

```{code-cell} ipython3
markers.table[["p_value", "p_value_adjusted"]].isna().sum()
```

```{code-cell} ipython3
markers.table[["p_value", "p_value_adjusted"]].notna().sum()
```

Rank the strongest positive associations (increasing with pseudotime) and the
strongest negative associations (decreasing with pseudotime) separately:

```{code-cell} ipython3
tested = markers.table.loc[
    markers.table["p_value_adjusted"].notna(),
    ["feature_name", "r_value", "p_value_adjusted"],
]
(
    tested.loc[tested["r_value"] > 0]
    .sort_values("r_value", ascending=False)
    .head(10)
)
```

```{code-cell} ipython3
(
    tested.loc[tested["r_value"] < 0]
    .sort_values("r_value", ascending=True)
    .head(10)
)
```

---
## 4) Visualize pseudotime correlated features

Use one decreasing ductal-associated gene and one increasing
endocrine-associated gene as a biological checkpoint. The table confirms that
their correlations point in opposite directions and that both pass the
adjusted significance threshold.

```{code-cell} ipython3
checkpoint_genes = ["Spp1", "Cpe"]
checkpoint = (
    markers.table.set_index("feature_name")
    .loc[checkpoint_genes, ["r_value", "p_value_adjusted"]]
)
checkpoint
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=checkpoint_genes,
    n_columns=2,
    sort_values=True,
)
```

`Spp1` should be strongest near the ductal source, while `Cpe` should increase
toward endocrine states. This agreement is a useful checkpoint, not proof of a
causal lineage or evidence that every dynamic gene changes monotonically.

As an optional check, scatter each checkpoint gene against the ordering among
scored cells:

```{code-cell} ipython3
import matplotlib.pyplot as plt

cell_key = pseudotime.validity_key
ptime = ds.get_cell_vals(
    from_assay="RNA",
    cell_key=cell_key,
    k=pseudotime.pseudotime_key,
)
figure, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
for axis, gene in zip(axes, checkpoint_genes, strict=True):
    expression = ds.get_cell_vals(
        from_assay="RNA",
        cell_key=cell_key,
        k=gene,
    )
    axis.scatter(ptime, expression, s=4, alpha=0.35)
    axis.set(xlabel="Pseudotime", ylabel=gene)
figure.tight_layout()
plt.show()
```

## Common mistakes and limitations

- Choosing source or sink clusters that do not sit at the intended ends of the trajectory
- Interpreting linear correlation as the only form of expression dynamics along pseudotime
- Ignoring `RNA_pseudotime__valid` when the graph has more than one connected component

Use {doc}`expression_dynamics` to group nonlinear expression patterns and
{doc}`trajectory_validation` for component policy, marker-testing assumptions,
module diagnostics, and fate-probability validation.
