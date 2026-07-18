---
description: Cell quality control, filtering thresholds, auto_filter_cells, and doublet scores in Scarf.
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

(quality_control)=

# Quality control

This chapter covers cell QC distributions, choosing filter thresholds, `auto_filter_cells`,
and doublet scoring. The recommended one-path QC in {doc}`scrna_seq` stays short; use this
page when you need more control.

## Prerequisites

- {doc}`scrna_seq` or {ref}`quickstart <quickstart>`
- A Zarr store for an RNA assay

## What you will learn

- Inspect per-cell QC columns
- Set manual thresholds with `filter_cells`
- Use `auto_filter_cells` for percentile-based bounds
- Compute doublet scores after an initial clustering

## Dataset

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

scarf.fetch_dataset(
    'tenx_5K_pbmc_rnaseq',
    save_path='scarf_datasets',
    as_zarr=True,
)
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds
```

## 1) Inspect QC distributions

On open, Scarf computes columns such as `RNA_nCounts`, `RNA_nFeatures`, and mito/ribo
fractions when gene names match the configured patterns.

```{code-cell} ipython3
qc_cols = [
    c for c in (
        'RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito', 'RNA_percentRibo'
    )
    if c in ds.cells.columns
]
splt.distribution(
    ds,
    keys=qc_cols,
    kind='violin',
    max_points=2000,
)
```

## 2) Manual thresholds

Thresholds are dataset-specific. The values below match the PBMC example in
{doc}`scrna_seq`. Filtered cells are marked inactive in cell key `I`.

```{code-cell} ipython3
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
splt.distribution(
    ds,
    keys=qc_cols,
    kind='violin',
    max_points=2000,
    color='coral',
)
```

## 3) Automatic thresholds

`auto_filter_cells` models each QC column as a normal distribution from its median and
standard deviation, then takes density points at `min_p` and `max_p` (defaults 0.01 and
0.99). Default columns are nCounts, nFeatures, percentMito, and percentRibo when present.

```{code-cell} ipython3
ds_auto = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds_auto.auto_filter_cells(show_qc_plots=True)
```

## 4) Doublet scores

`run_doublet_detection` simulates doublets, maps them onto the existing neighbourhood
graph, and writes a per-cell score (default base label `doublet_score`). It does not
remove cells automatically. It requires an existing cluster column.

```{code-cell} ipython3
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
ds.run_leiden_clustering(resolution=0.5)
score_col = ds.run_doublet_detection(cluster_key='RNA_leiden_cluster')
ds.run_umap(n_epochs=100, spread=5, min_dist=1, parallel=True)
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=score_col,
)
```

Filter on the score yourself when you choose a cutoff for your data:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(columns=[score_col], key='I').describe()
```

## Common mistakes and limitations

- Copying thresholds from another dataset without checking distributions
- Expecting `run_doublet_detection` to drop cells (it only scores)
- Running doublet detection before `make_graph` and clustering

## Summary of saved results

| Kind | Keys |
|---|---|
| QC | `RNA_nCounts`, `RNA_nFeatures`, `RNA_percentMito`, … |
| Active cells | `I` |
| Doublets | `*doublet_score` (assay-prefixed) |

## Next steps

- {doc}`scrna_seq`
- {doc}`annotation`
- {doc}`data_organization`
