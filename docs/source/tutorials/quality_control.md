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

scarf.set_verbosity('WARNING')

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds
```

## 1) Inspect QC distributions

On open, Scarf streams the count matrix once to compute initialization statistics:
columns such as `RNA_nCounts`, `RNA_nFeatures`, and mito/ribo fractions when gene
names match the configured patterns, plus per-feature cell counts for feature filtering.

```{code-cell} ipython3
qc_cols = [
    c for c in (
        'RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito', 'RNA_percentRibo'
    )
    if c in ds.cells.columns
]
ds.plots.distribution(
    keys=qc_cols,
    kind='violin',
    max_points=2000,
)
```

Each violin shows the distribution of one QC metric before filtering; use the tails to set
dataset-specific cutoffs.

## 2) Manual thresholds

Thresholds are dataset-specific. The values below match the PBMC example in
{doc}`scrna_seq`. Filtered cells are marked inactive in cell key `I`.

```{code-cell} ipython3
# Prepared stores may already have an `I` filter. Reset so before/after counts are meaningful.
ds.cells.reset_key(key='I')
n_before = int(ds.cells.fetch_all('I').sum())
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
n_after = int(ds.cells.fetch_all('I').sum())
print(f'Active cells before filter: {n_before}')
print(f'Active cells after filter: {n_after}')
ds.plots.distribution(
    keys=qc_cols,
    kind='violin',
    max_points=2000,
    color='coral',
)
```

After filtering, the same metrics are restricted to active cells (`I=True`). If
`n_before` and `n_after` are equal, the thresholds did not remove additional cells on
this store; inspect the violins and adjust the cutoffs.

## 3) Custom percent-feature columns

`add_percent_feature` stores the fraction of counts from features matching a regular
expression. Built-in mito and ribo columns use this mechanism. The example below adds a
hemoglobin-gene fraction when those gene names are present.

```{code-cell} ipython3
ds.RNA.add_percent_feature(feat_pattern='^HB[AB]', name='RNA_percentHB')
if 'RNA_percentHB' in ds.cells.columns:
    print(
        ds.cells.to_pandas_dataframe(['RNA_percentHB'], key='I')['RNA_percentHB']
        .describe()
    )
else:
    print('No features matched ^HB[AB] in this store')
```

## 4) Automatic thresholds

`auto_filter_cells` models each QC column as a normal distribution from its median and
standard deviation, then takes density points at `min_p` and `max_p` (defaults 0.01 and
0.99). Default columns are nCounts, nFeatures, percentMito, and percentRibo when present.
Like repeated `filter_cells` calls, it intersects its result with the current `I` column and
does not reactivate cells. The call below intentionally refines the manual filter from the
previous section. Use a fresh store or reset `I` before the call when comparing it as an
alternative filtering strategy.

```{code-cell} ipython3
ds.auto_filter_cells(show_qc_plots=True)
```

The plots show fitted density bounds on each QC column used for the automatic cutoffs.

## 5) Doublet scores

`run_doublet_detection` simulates doublets, maps them onto the existing neighbourhood
graph, and writes a per-cell score (default base label `doublet_score`). It does not
remove cells automatically. It requires an existing cluster column.

```{code-cell} ipython3
ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
ds.run_normalization(feat_key='hvgs')
ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=11)
ds.build_connectivity_map()
ds.run_leiden_clustering(resolution=0.5)
score_col = ds.run_doublet_detection(cluster_key='RNA_leiden_cluster')
ds.run_umap(n_epochs=100, spread=5, min_dist=1, parallel=True)
```


```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=score_col,
)
```

Higher doublet scores mark cells that map near simulated doublets. Inspect the score
distribution before applying a cutoff:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(columns=[score_col], key='I').describe()
```

## Common mistakes and limitations

- Copying thresholds from another dataset without checking distributions
- Expecting `run_doublet_detection` to drop cells (it only scores)
- Running doublet detection before building the neighbourhood graph and clustering


## Summary of saved results

| Kind | Keys |
|---|---|
| QC | `RNA_nCounts`, `RNA_nFeatures`, `RNA_percentMito`, … |
| Active cells | `I` |
| Doublets | `*doublet_score` (assay-prefixed) |

## Further reading

- [Single-cell best practices: quality control](https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html)
- [Scanpy clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)

## Next steps

- {doc}`dimensionality_reduction_and_clustering`
- {doc}`annotation`
- {doc}`data_organization`
