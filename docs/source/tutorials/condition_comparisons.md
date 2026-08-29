---
description: Compare control and IFN-beta-stimulated PBMCs with Welch's t-test, Mann-Whitney, and Kruskal-Wallis, then read the persisted variants back and overlay brackets on violin plots.
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

# Comparing biological conditions with statistical testing

Scarf can compare value distributions of genes or cell metadata across an experimental design and keep every result as a retrievable artifact.
This page uses the classic interferon response design: peripheral blood mononuclear cells (PBMCs) left untreated versus stimulated with IFN-beta.
Today we will compare control and IFN-beta-stimulated PBMCs with Welch's t-test, Mann-Whitney, and Kruskal-Wallis, then read the persisted variants back and overlay brackets on violin plots.
You will run Welch's t-tests between the two conditions, cross-check them against the rank-based defaults, test across many cell types with Kruskal-Wallis plus Dunn's post-hoc, and overlay significance brackets directly onto violin plots.

Everything on this page is descriptive distribution testing on single cells.
It tells you which genes shift between conditions and how strongly, but it is not replicate-aware differential expression; see the note at the end and {doc}`pseudobulk_and_differential_expression` for that distinction.

## Prerequisites

- Scarf installed (see {doc}`/quickstart`)
- Network access to download the dataset on first run

## What you will learn

- Run one-sided and two-sided Welch's t-tests between two conditions
- Read persisted results back with `get_statistical_tests` by repeating variant parameters
- Test many genes in one call with pooled Benjamini-Hochberg correction
- Cross-check parametric results against Mann-Whitney and Kruskal-Wallis with Dunn's post-hoc
- Overlay significance brackets onto violin, stacked violin, and box plots

## Dataset

`kang_29K_ctrl-ifnb_pbmc_rnaseq` is the published merge of two Kang-style PBMC stores: untreated control cells and cells stimulated with IFN-beta.
Each cell carries a `sample_id` metadata column that records its treatment (`ctrl` or `stim`), so the column doubles as the condition label.
Interferon-stimulated genes such as `ISG15`, `IFIT1`, and `MX1` are strongly induced in the stimulated cells, while housekeeping or lineage genes like `CD3D` and `LYZ` shift much less, which makes the dataset convenient for seeing the difference between strong, subtle, and absent effects.

## 1. Open a writable copy of the store

Download the published store, repack its counts, and mount a writable analysis copy.
Statistical results are persisted as artifacts, so the store must be writable.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

repository = scarf.cytebase.connect("scarf_docs")
merged_path = repository.download_dataset(
    name="kang_29K_ctrl-ifnb_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = str(Path(analysis_directory.name) / "counts.zarr")
repack_store(
    f"{merged_path}/data.zarr",
    repacked_counts,
    nthreads=2,
)
ds = scarf.mount_datastore(
    repacked_counts,
    at=str(Path(analysis_directory.name) / "condition_analysis.zarr"),
    default_assay="RNA",
    nthreads=4,
)
```

## 2. Inspect the condition column

`sample_id` records the treatment of every cell.
Counting its values confirms the two-group design and gives the per-condition cell numbers.

```{code-cell} ipython3
import numpy as np

sample_id = np.asarray(ds.cells.fetch_all("sample_id")).astype(str)
values, counts = np.unique(sample_id, return_counts=True)
print(dict(zip(values.tolist(), counts.tolist())))
```

With exactly two conditions, the natural parametric choice is Welch's t-test (`test="welch"`, aliased by `"t_test"`).
It runs on raw normalized cell values, always uses unequal variances, and accepts a one-sided `alternative`.
The rank-based `mann_whitney` covers the same two-group design without distributional assumptions, and `auto` always picks `mann_whitney` for two groups; parametric tests are opt-in only.

## 3. Welch's t-test between conditions

Run the test for the strongest interferon-stimulated gene.
The returned table reports the group means, `mean_difference` (mean of `group_1` minus mean of `group_2`), the Welch statistic with its degrees of freedom, and the p-value.
Because the store is writable, the result is persisted as an artifact under a slot that encodes every variant parameter.

```{code-cell} ipython3
isg15 = ds.run_statistical_testing(
    "ISG15",
    group_by="sample_id",
    test="welch",
)
isg15.tables["ISG15"]
```

`group_1` is the first category encountered in the data (`stim` here) and `group_2` the second (`ctrl`), so a positive `mean_difference` means the gene is higher in stimulated cells.
Pass `groups=["ctrl", "stim"]` to reverse the contrast direction; the means swap sides and the statistic flips sign while the p-value stays equal.

Reading the result back requires repeating the same variant parameters.
Nothing here changed from the defaults, so the defaults are enough.

```{code-cell} ipython3
loaded = ds.get_statistical_tests(
    group_key="sample_id",
    method="welch",
    keys="ISG15",
)
loaded.tables["ISG15"]
```

A one-sided alternative asks whether `group_1` is greater than `group_2`.
The alternative is part of the variant identity, so the one-sided result is stored under its own slot and both remain retrievable side by side.

```{code-cell} ipython3
isg15_greater = ds.run_statistical_testing(
    "ISG15",
    group_by="sample_id",
    test="welch",
    alternative="greater",
)
loaded_greater = ds.get_statistical_tests(
    group_key="sample_id",
    method="welch",
    keys="ISG15",
    alternative="greater",
)
print(
    "two-sided p:",
    float(isg15.tables["ISG15"]["p_value"].iloc[0]),
)
print(
    "greater    p:",
    float(loaded_greater.tables["ISG15"]["p_value"].iloc[0]),
)
```

## 4. Test many genes in one call

Passing a list of genes tests each one and pools all their p-values into a single Benjamini-Hochberg correction, recorded as `p_value_adjusted`.
The adjusted column controls the false discovery rate across the family of tested genes, which matters once several results are inspected together.
With a single gene the adjustment is a no-op.

```{code-cell} ipython3
import pandas as pd

panel = ["ISG15", "IFIT1", "MX1", "CD3D", "LYZ"]
panel_result = ds.run_statistical_testing(
    panel,
    group_by="sample_id",
    test="welch",
)
pd.concat(
    {gene: panel_result.tables[gene] for gene in panel},
    names=["gene"],
).reset_index(level="gene")[
    ["gene", "mean_1", "mean_2", "mean_difference", "p_value", "p_value_adjusted"]
]
```

The induced genes move by tens of normalized counts while `CD3D` shifts by a fraction of a count; both effects are visible in one table because the pooled correction keeps the family of claims controlled.

## 5. Overlay brackets on the distributions

`distribution` accepts the result object and draws significance brackets over the matching panels with pure matplotlib.
Each pairwise test row becomes one bracket between the two group positions; the bracket label prefers `p_value_adjusted` when present.
Extremely strong effects can underflow the p-value to exactly `0.0`, so an extremely large panel may legitimately read `p=0`.

```{code-cell} ipython3
ds.plots.distribution(
    "ISG15",
    group_by="sample_id",
    kind="violin",
    stats_results=isg15,
)
```

Dock a whole panel of tests onto one stacked violin by passing the multi-gene result.
Every gene row gets its own bracket, and `share_y` keeps the value axes comparable across rows.

```{code-cell} ipython3
ds.plots.distribution(
    panel,
    group_by="sample_id",
    kind="stacked_violin",
    share_y=True,
    stats_results=panel_result,
)
```

Pass `stats_show_p=False` to render `***`/`**`/`*`/`ns` thresholds instead of numeric p-values, or `orientation="horizontal"` with `kind="box"` to mirror the bracket onto the value axis.
If the result does not describe the plotted selection (different grouping, cell key, or cell count), the plot warns and skips the bracket instead of drawing something wrong.

## 6. Cross-check with the rank-based tests

Single-cell values are zero-inflated and non-normal, so the rank-based tests are the appropriate defaults.
`mann_whitney` answers the same two-group question without assuming anything about the distribution shape.

```{code-cell} ipython3
isg15_mw = ds.run_statistical_testing(
    "ISG15",
    group_by="sample_id",
    test="mann_whitney",
)
isg15_mw.tables["ISG15"]
```

For three or more groups the design changes shape.
Grouping by the Leiden clusters (approximating cell types) and running Kruskal-Wallis with Dunn's post-hoc yields one omnibus row plus every pairwise contrast between clusters, each with its own p-value and pooled adjustment.

```{code-cell} ipython3
cluster_result = ds.run_statistical_testing(
    ["ISG15"],
    group_by="RNA_clusters",
    test="kruskal_wallis",
    posthoc="dunn",
)
print("omnibus:", cluster_result.tables["ISG15"].to_dict("records"))
print("pairwise rows:", len(cluster_result.posthoc_tables["ISG15"]))
cluster_result.posthoc_tables["ISG15"].head()
```

The omnibus p-value answers only "does ISG15 differ anywhere across clusters".
The post-hoc table is where you read which pairs of clusters differ; the same pattern applies to Welch's omnibus sibling, one-way ANOVA (`test="one_way_anova"`), whose single row spans all groups when annotated on a plot.

## Common mistakes and limitations

- Cell-level results are descriptive. Treating thousands of cells as independent observations makes tiny effect sizes significant; the emitted warning is part of the contract, not noise. For condition-level claims, aggregate to biological samples (`sample_by`) or export counts for DESeq2 or edgeR as shown in {doc}`pseudobulk_and_differential_expression`.
- `get_statistical_tests` raises unless every variant parameter matches the stored run: keys, groups, comparisons, adjustment, `alternative`, and normalization.
- Welch requires exactly two surviving groups and one-way ANOVA at least two; neither accepts sample aggregation, and `posthoc="dunn"` belongs to Kruskal-Wallis only.
- `test="auto"` never selects a parametric method; request `welch` or `one_way_anova` explicitly when a parametric summary is wanted beside the rank-based defaults.
- Brackets are drawn from the persisted table, never recomputed by the plot. A result computed on a different grouping or cell selection warns and skips instead of annotating the wrong panel.
