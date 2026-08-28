---
description: Aggregate cells with make_bulk and export for external differential expression.
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

(pseudobulk_and_differential_expression)=

# Pseudobulk and differential expression

Scarf stops at aggregation and export.
Use `make_bulk` to build bulk-like count profiles, then export those counts (with sample-level metadata) for condition-level differential expression in an external tool such as edgeR or DESeq2.
`run_marker_search` is separate: it returns Mann-Whitney `p_value` columns plus within-group `p_value_adjusted` (Benjamini-Hochberg) for cluster interpretation.
Those adjusted values are still cell-level marker statistics, not replicate-aware FDR-controlled DE.

## Prerequisites

- A clustered RNA `DataStore`
- Optional: multi-sample metadata for true pseudobulk designs

## What you will learn

- Export aggregate counts from `make_bulk` for external DE
- Keep marker statistics separate from condition-level DE
- Build optional pseudo-replicates within groups

## Dataset

This page uses the Kang control PBMC store and Leiden clusters as groups.
For multi-sample designs, merge samples first ({doc}`dataset_merging`) and pass sample and cell-type columns to `make_bulk`.
The catalog snapshot is structurally repacked in a temporary directory and mounted into a clean writable analysis target, preserving literal cluster and UMAP metadata without reusing legacy analysis state.

```{code-cell} ipython3
from tempfile import TemporaryDirectory

import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'kang_15K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = f'{analysis_directory.name}/counts.zarr'
repack_store(
    f'{dataset}/data.zarr',
    repacked_counts,
    nthreads=2,
)
ds = scarf.mount_datastore(
    repacked_counts,
    at=f'{analysis_directory.name}/analysis.zarr',
    nthreads=4,
    default_assay='RNA',
)
```

The published Kang store carries a UMAP and a Leiden partition under `RNA_clusters`.
Those groups are the aggregation key below, and they may differ from the author-provided `cluster_labels` column.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_clusters',
)
```

Leiden clusters on the published UMAP; these labels are the grouping key for `make_bulk` below.

## 1. Inspect marker ranks separately from DE

```{code-cell} ipython3
ds.set_feature_selection(
    feature_indexes=range(ds.RNA.feats.N),
    label='all_for_markers',
)
all_features = ds.resolve_features('RNA', 'all_features')
marker_ref = ds.run_marker_search(
    group_key='RNA_clusters',
    cell_key='I',
    features=all_features,
)
markers = ds.get_markers(
    marker=marker_ref,
    cell_key='I',
    group_key='RNA_clusters',
    group_id='1',
    min_score=-1,
    min_frac_exp=-1,
)
markers[
    [
        'feature_name',
        'score',
        'frac_exp',
        'auc',
        'p_value',
        'p_value_adjusted',
    ]
].head()
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    marker=marker_ref,
    group_key='RNA_clusters',
    cell_key='I',
    topn=2,
    figsize=(8, 8),
)
```

Rows are top markers per cluster for interpretation only, not exported DE results.

```{note}
Use marker tables for cluster interpretation.
Do not treat within-group `p_value_adjusted` columns as replicate-aware differential expression results.
```

## 1b. Descriptive group comparisons with run_statistical_testing

`run_statistical_testing` compares one feature at a time across groups and persists each variant under its own slot.
The rank-based defaults suit single-cell data: `mann_whitney` for two groups, `kruskal_wallis` (optionally with Dunn's post-hoc) for three or more, and paired Wilcoxon on aggregated samples.

Explicitly requested parametric tests (`welch`, aliased by `t_test`, and `one_way_anova`) also run on raw cell-level values.
They are descriptive only: they treat cells as independent, cannot be combined with sample aggregation, and make assumptions that zero-inflated single-cell values often violate.
Welch honours the one-sided `alternative`; automatic method selection never chooses a parametric test.

```{code-cell} ipython3
# Compare two clusters that are present in the Kang dataset.
result = ds.run_statistical_testing(
    ['MALAT1'],
    group_by='RNA_clusters',
    groups=[1, 2],
    test='welch',
    alternative='greater',
)

# Pin retrieval to the exact immutable result returned by the run.
loaded = ds.get_statistical_tests(artifact=result.artifact)
loaded.tables['MALAT1']
```

```{code-cell} ipython3
# The plotted selection and test design must match the stored result.
figure = ds.plots.distribution(
    ['MALAT1'],
    group_by='RNA_clusters',
    groups=[1, 2],
    kind='violin',
    stats_results=result,
    show=False,
)
```

Brackets are drawn directly with matplotlib over the seaborn violins and prefer the pooled `p_value_adjusted` column when present.

## 2. Aggregate with make_bulk

Cell counts per Leiden group set the scale for each bulk column:

```{code-cell} ipython3
group_sizes = (
    ds.cells.to_pandas_dataframe(
        columns=['RNA_clusters'],
        key='I',
    )['RNA_clusters']
    .astype(str)
    .value_counts()
    .sort_index()
)
group_sizes
```

`make_bulk` sums raw counts per group.
`return_fraction=True` adds the fraction of cells with non-zero counts in the same pass:

```{code-cell} ipython3
bulk, fracs = ds.make_bulk(
    group_key='RNA_clusters',
    aggr_type='sum',
    feature_label='name',
    return_fraction=True,
)
totals = bulk.sum().rename('total_counts')
totals.index = totals.index.astype(str)
pd.concat(
    [group_sizes.rename('n_cells'), totals],
    axis=1,
)
```

Top expressed genes across groups (summed counts), with detection fractions alongside:

```{code-cell} ipython3
top_genes = bulk.sum(axis=1).sort_values(ascending=False).head(8).index
bulk.loc[top_genes]
```

```{code-cell} ipython3
fracs.loc[top_genes]
```

Optional pseudo-replicates within each group:

```{note}
`pseudo_reps > 1` randomly splits cells within each group.
These are descriptive resamples of the same cells, not independent biological replicates.
Do not treat them as replicates in edgeR, DESeq2, or PyDESeq2.
For replicate-aware differential expression, aggregate with a sample-aware `group_key` (for example sample nested with cell type) and keep true biological replicates in the exported metadata.
```

```{code-cell} ipython3
bulk_reps = ds.make_bulk(
    group_key='RNA_clusters',
    aggr_type='sum',
    feature_label='name',
    pseudo_reps=2,
)
list(bulk_reps.columns)
```

Column names carry the group label plus `_Rep1` / `_Rep2`.
Shape doubles the group count because each cluster is split once:

```{code-cell} ipython3
bulk_reps.shape
```

## 3. Export counts for external DE

```{code-cell} ipython3
export_path = 'scarf_datasets/kang_pseudobulk_counts.csv'
meta_path = 'scarf_datasets/kang_pseudobulk_sample_meta.csv'
bulk.to_csv(export_path)
sample_meta = pd.concat(
    [group_sizes.rename('n_cells'), totals],
    axis=1,
)
sample_meta.index.name = 'group'
sample_meta.to_csv(meta_path)
print(f'Wrote {bulk.shape[0]} features x {bulk.shape[1]} groups to {export_path}')
print(f'Wrote sample metadata for {sample_meta.shape[0]} groups to {meta_path}')
```

Peek the written count matrix and the group-level metadata that pairs with its columns:

```{code-cell} ipython3
pd.read_csv(export_path, index_col=0, nrows=5).iloc[:, :5]
```

```{code-cell} ipython3
pd.read_csv(meta_path, index_col=0)
```

Use the exported count matrix and sample-level metadata with a method appropriate to the study design, such as edgeR or DESeq2.
Scarf stops at aggregation and export; it does not run those models.
For a true multi-sample design, merge donors or conditions first ({doc}`dataset_merging`), aggregate with a sample-aware `group_key` (for example sample nested with cell type), and keep biological replicates in the exported metadata.

## Common mistakes

- Reporting Scarf's within-group marker adjustment as replicate-aware DE
- Building pseudobulk without a sample (donor) covariate when the biological question is condition-level
- Treating `pseudo_reps` splits as independent biological replicates
- Expecting Scarf to run DESeq2/edgeR-style models in-process
