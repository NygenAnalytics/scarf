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

`make_bulk` aggregates cells into bulk-like profiles. It is not a full differential-expression
suite. `run_marker_search` returns Mann-Whitney `p_value` columns without multiple-testing
correction. Export aggregate counts for condition-level differential expression in an external
tool.

## Prerequisites

- A clustered RNA `DataStore`
- Optional: multi-sample metadata for true pseudobulk designs

## What you will learn

- Aggregate raw counts with `make_bulk`
- Inspect marker statistics separately from differential expression
- Export pseudobulk counts for an external analysis

## Dataset

This page uses the Kang control PBMC store and Leiden clusters as groups. For multi-sample
designs, merge samples first ({doc}`data_integration`) and pass sample and cell-type columns
to `make_bulk`.

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

scarf.fetch_dataset(
    'kang_15K_pbmc_rnaseq',
    save_path='scarf_datasets',
    as_zarr=True,
)
ds = scarf.DataStore(
    'scarf_datasets/kang_15K_pbmc_rnaseq/data.zarr',
    nthreads=4,
)
ds
```

```{code-cell} ipython3
# Ensure a clustering column exists for aggregation demos.
if 'RNA_leiden_cluster' not in ds.cells.columns:
    ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
    ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
    ds.run_leiden_clustering(resolution=0.5)
```

The catalog Kang store already includes a UMAP. Leiden groups used for `make_bulk` below may
differ from the published `cluster_labels` column.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

## Guided steps

### 1. Inspect marker ranks separately from DE

```{code-cell} ipython3
ds.run_marker_search(group_key='RNA_leiden_cluster', gene_batch_size=100)
markers = ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id='1',
    min_score=-1,
    min_frac_exp=-1,
)
markers.head()
```

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key='RNA_leiden_cluster',
    topn=3,
    figsize=(5, 7),
)
```

```{note}
Use marker tables for cluster interpretation. Do not treat uncorrected `p_value` columns as
FDR-controlled differential expression results.
```

### 2. Aggregate with make_bulk

```{code-cell} ipython3
bulk = ds.make_bulk(
    group_key='RNA_leiden_cluster',
    aggr_type='sum',
    feature_label='name',
)
bulk.iloc[:5, :5]
```

Optional pseudo-replicates within each group:

```{code-cell} ipython3
bulk_reps = ds.make_bulk(
    group_key='RNA_leiden_cluster',
    aggr_type='sum',
    feature_label='name',
    pseudo_reps=2,
)
bulk_reps.shape
```

### 3. Export counts for external DE

```{code-cell} ipython3
bulk.to_csv('scarf_datasets/kang_pseudobulk_counts.csv')
```

Use the exported count matrix and sample-level metadata with a method appropriate to the study
design, such as edgeR or DESeq2. For a true multi-sample design, merge donors or conditions
first ({doc}`data_integration`), aggregate with a sample-aware `group_key` (for example
sample nested with cell type), and keep biological replicates in the exported metadata.

## Common mistakes

- Reporting Scarf marker p-values as FDR-corrected DE
- Building pseudobulk without a sample (donor) covariate when the biological question is condition-level
- Expecting Scarf to run DESeq2/edgeR-style models in-process

## Saved results

`make_bulk` returns a pandas DataFrame. The example exports it to
`scarf_datasets/kang_pseudobulk_counts.csv`. Marker-search results are stored under the assay's
`markers` group in the Zarr store.

## Next steps

- {doc}`annotation`
- {doc}`import_and_export`
- {doc}`../scarf_and_scanpy`
