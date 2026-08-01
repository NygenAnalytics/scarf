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

Scarf stops at aggregation and export. Use `make_bulk` to build bulk-like count profiles, then
export those counts (with sample-level metadata) for condition-level differential expression in
an external tool such as edgeR or DESeq2. `run_marker_search` is separate: it returns
Mann-Whitney `p_value` columns plus within-group `p_value_adjusted` (Benjamini-Hochberg) for
cluster interpretation. Those adjusted values are still cell-level marker statistics, not
replicate-aware FDR-controlled DE.

## Prerequisites

- A clustered RNA `DataStore`
- Optional: multi-sample metadata for true pseudobulk designs

## What you will learn

- Export aggregate counts from `make_bulk` for external DE
- Keep marker statistics separate from condition-level DE
- Build optional pseudo-replicates within groups

## Dataset

This page uses the Kang control PBMC store and Leiden clusters as groups. For multi-sample
designs, merge samples first ({doc}`data_integration`) and pass sample and cell-type columns
to `make_bulk`.

```{code-cell} ipython3
import scarf

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'kang_15K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
)
```

The published Kang store carries a UMAP and a Leiden partition under
`RNA_clusters`. Those groups are the aggregation key below, and they may differ
from the author-provided `cluster_labels` column.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_clusters',
)
```

Leiden clusters on the published UMAP; these labels are the grouping key for `make_bulk` below.

## Guided steps

### 1. Inspect marker ranks separately from DE

```{code-cell} ipython3
ds.run_marker_search(group_key='RNA_clusters')
markers = ds.get_markers(
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
    group_key='RNA_clusters',
    topn=2,
    figsize=(8, 8),
)
```

Rows are top markers per cluster for interpretation only, not exported DE results.

```{note}
Use marker tables for cluster interpretation. Do not treat within-group `p_value_adjusted`
columns as replicate-aware differential expression results.
```

### 2. Aggregate with make_bulk

```{code-cell} ipython3
bulk = ds.make_bulk(
    group_key='RNA_clusters',
    aggr_type='sum',
    feature_label='name',
)
bulk.iloc[:5, :5]
```

Optional pseudo-replicates within each group:

```{note}
`pseudo_reps > 1` randomly splits cells within each group. These are descriptive
resamples of the same cells, not independent biological replicates. Do not treat
them as replicates in edgeR, DESeq2, or PyDESeq2. For replicate-aware differential
expression, aggregate with a sample-aware `group_key` (for example sample nested
with cell type) and keep true biological replicates in the exported metadata.
```

```{code-cell} ipython3
bulk_reps = ds.make_bulk(
    group_key='RNA_clusters',
    aggr_type='sum',
    feature_label='name',
    pseudo_reps=2,
)
bulk_reps.shape
```

### 3. Export counts for external DE

```{code-cell} ipython3
export_path = 'scarf_datasets/kang_pseudobulk_counts.csv'
bulk.to_csv(export_path)
print(f'Wrote {bulk.shape[0]} features x {bulk.shape[1]} groups to {export_path}')
```

Use the exported count matrix and sample-level metadata with a method appropriate to the study
design, such as edgeR or DESeq2. Scarf stops at aggregation and export; it does not run those
models. For a true multi-sample design, merge donors or conditions first
({doc}`data_integration`), aggregate with a sample-aware `group_key` (for example sample
nested with cell type), and keep biological replicates in the exported metadata.

## Common mistakes

- Reporting Scarf's within-group marker adjustment as replicate-aware DE
- Building pseudobulk without a sample (donor) covariate when the biological question is condition-level
- Treating `pseudo_reps` splits as independent biological replicates
- Expecting Scarf to run DESeq2/edgeR-style models in-process
