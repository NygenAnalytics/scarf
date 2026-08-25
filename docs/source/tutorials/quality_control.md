---
description: Inspect and filter cell quality across RNA, ATAC, and multimodal assays.
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

# Quality control across assays

Quality control defines the cells and features that downstream analyses can use.
Scarf stores selections as boolean metadata rather than deleting data, so thresholds can be inspected, reset, and replaced.
This guide covers manual, global automatic, and sample-aware filtering, followed by assay-specific checks.

## Prerequisites

- {doc}`scrna_seq` or {ref}`quickstart <quickstart>`
- A Zarr store for an RNA assay

## What you will learn

- Inspect per-cell QC columns
- Set manual thresholds with `filter_cells`
- Compare global Gaussian and per-sample MAD bounds
- Compute doublet scores after an initial clustering
- Recognize current RNA, ATAC, and ADT support boundaries

## Standalone setup

Quality control has to see the population it is judging, so this page builds its store from raw counts rather than opening one that has already been filtered.

```{code-cell} ipython3
import numpy as np

import scarf

scarf.configure_output(level='WARNING', progress=True)

counts = scarf.cytebase.connect("scarf_docs").download(
    'tenx_5K_pbmc_rnaseq/data.h5',
    destination='scarf_datasets',
)[0]

store = counts.with_name('data.zarr')
reader = scarf.CrH5Reader(str(counts))
scarf.CrToZarr(
    reader,
    zarr_loc=str(store),
).dump()

ds = scarf.DataStore(
    str(store),
    nthreads=4,
    min_features_per_cell=10,
)
```

The `I` {term}`cell key` is the active selection used by most methods.
Filtering updates that key and leaves all rows in the datastore.
Feature selections work the same way within each assay.

## 1. Inspect QC distributions

On open, Scarf streams the count matrix once to compute initialization statistics: columns such as `RNA_nCounts`, `RNA_nFeatures`, and mito/ribo fractions when gene names match the configured patterns, plus per-feature detection statistics used by explicit selection producers.

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

Each violin uses the active cell key `I`, so cells already below `min_features_per_cell` from open are excluded.
Use the tails to set further cutoffs.

## 2. Manual thresholds

Thresholds are dataset-specific.
The values below match the PBMC example in {doc}`scrna_seq`.
Filtered cells are marked inactive in cell key `I`.

```{code-cell} ipython3
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

After filtering, the same metrics are restricted to active cells (`I=True`), so the coral violins are visibly tighter than the ones above: the long low-count tail and the high-mito shoulder are both gone.
Roughly a fifth of the barcodes drop out here, which is typical for this dataset and is the number worth sanity-checking against your own expectations before continuing.

## 3. Custom percent-feature columns

`add_percent_feature` stores the fraction of counts from features matching a regular expression.
Built-in mito and ribo columns use this mechanism.
The example below adds a hemoglobin-gene fraction when those gene names are present.

```{code-cell} ipython3
ds.RNA.add_percent_feature(feat_pattern='^HB[AB]', name='RNA_percentHB')
if 'RNA_percentHB' in ds.cells.columns:
    print(
        ds.cells.to_pandas_dataframe(['RNA_percentHB'], key='I')['RNA_percentHB']
        .describe()
    )
    ds.plots.distribution(
        keys=['RNA_percentHB'],
        kind='violin',
        max_points=2000,
    )
else:
    print('No features matched ^HB[AB] in this store')
```

Most cells sit near zero hemoglobin fraction; the long upper tail is the set worth inspecting before you set an upper cutoff for erythrocyte contamination.

## 4. Automatic thresholds

`auto_filter_cells` fits a normal distribution (`loc=median`, `scale=std`) to each QC column, then takes quantiles at `min_p` and `max_p` via `scipy.stats.norm.ppf` (defaults 0.01 and 0.99).
Default columns are nCounts, nFeatures, percentMito, and percentRibo when present.
Like repeated `filter_cells` calls, it intersects its result with the current `I` column and does not reactivate cells.
The call below intentionally refines the manual filter from the previous section.
Use a fresh store or reset `I` before the call when comparing it as an alternative filtering strategy.

```{code-cell} ipython3
ds.auto_filter_cells(show_qc_plots=True)
```

With `show_qc_plots=True`, Scarf shows pre- and post-filtering distributions via `distribution(...)` for each QC column used.

## 5. Per-sample MAD filtering

Global bounds can penalize a sample whose count-depth distribution differs from the pooled distribution.
With `sample_column`, Scarf calculates robust bounds within each sample using the median absolute deviation (MAD).

The PBMC teaching dataset has no biological sample column.
The balanced `qc_sample` labels below make the API executable but must not be interpreted as a real sample-aware result.

```{code-cell} ipython3
ds.cells.insert(
    "qc_sample",
    np.asarray(
        [f"sample_{index % 4}" for index in range(ds.cells.N)]
    ),
    overwrite=True,
)
ds.cells.reset_key(key="I")
n_before = int(ds.cells.fetch_all('I').sum())
ds.auto_filter_cells(
    attrs=qc_cols,
    sample_column="qc_sample",
    n_mads=3.0,
    min_cells_per_sample=20,
    show_qc_plots=True,
)
n_after = int(ds.cells.fetch_all('I').sum())
print(f'Active cells before MAD filter: {n_before}')
print(f'Active cells after MAD filter: {n_after}')
```

The pre/post plots and active-cell counts show the sample-aware path.
The `qc_sample` labels are synthetic and balanced, so treat the retained counts as a mechanics demo, not a biological sample comparison.
Use a real sample column when comparing depth distributions across donors or batches.

Count-like metrics such as `nCounts` and `nFeatures` use log1p values and two-sided bounds.
Percentage metrics such as `percentMito` and `percentRibo` use their original scale and an upper bound only.
Samples with fewer than `min_cells_per_sample` active cells are retained with a warning because stable within-sample bounds cannot be estimated.
`min_p` and `max_p` apply only to the global Gaussian path and must remain at their defaults when `sample_column` is used.

The same options can be forwarded through the standard pipeline:

```python
ds.pipeline.run(
    filtering={
        "method": "auto",
        "sample_column": "sample_id",
        "n_mads": 3.0,
        "min_cells_per_sample": 20,
    },
    clustering_concurrency=1,
)
```

## 6. RNA percentages and feature exclusions

`add_percent_feature` measures the fraction of each cell's counts matching a gene-name pattern.
High mitochondrial, ribosomal, or hemoglobin fractions can indicate damaged cells or study-specific biology.
The mito/ribo violins in section 1 and the hemoglobin violin in section 3 are the inspection step before you apply upper thresholds.

Gene families excluded from the graph are a separate feature-selection decision.
See {doc}`feature_selection` for the default HVG blacklist and supported overrides.

## 7. Doublet scores

`run_doublet_detection` simulates doublets, maps them onto the existing neighbourhood graph, and writes a per-cell score (default base label `doublet_score`).
It does not remove cells automatically.
It requires an existing cluster column.

```{code-cell} ipython3
hvg_ref = ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
ds.run_normalization(features=hvg_ref)
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

Higher doublet scores mark cells that map near simulated doublets.
Inspect the score distribution before applying a cutoff:

```{code-cell} ipython3
ds.plots.distribution(
    keys=score_col,
    kind="ecdf",
)
```

The score distribution and embedding should be reviewed together.
A threshold is study-dependent, and `run_doublet_detection` does not remove cells.
After choosing an upper bound from the ECDF shoulder, apply it as an additional filter.
The teaching cutoff below keeps the upper 5% of scores on this PBMC run; replace it with a study-specific value when the ECDF shape differs.

```{code-cell} ipython3
scores = ds.cells.to_pandas_dataframe([score_col], key='I')[score_col]
print(scores.describe())
doublet_threshold = float(scores.quantile(0.95))
print(f'Doublet threshold (95th percentile): {doublet_threshold:.4f}')
n_before = int(ds.cells.fetch_all('I').sum())
ds.filter_cells(
    attrs=[score_col],
    lows=[None],
    highs=[doublet_threshold],
    reset_previous=False,
)
n_after = int(ds.cells.fetch_all('I').sum())
print(f'Active cells before doublet filter: {n_before}')
print(f'Active cells after doublet filter: {n_after}')
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=score_col,
)
```

With `reset_previous=False`, this call intersects the threshold with the current selection and does not reactivate cells removed by earlier QC.
The post-filter embedding should lose the highest-scoring hotspot cells from the map above.

## 8. ATAC quality control

Scarf initializes per-cell ATAC fragment or cut-site counts and accessible-peak counts, and it records per-peak detection statistics for explicit prevalent-peak selection.
`mark_prevalent_peaks` selects peaks for LSI and graph construction.
Scarf does not currently calculate FRiP or TSS enrichment, so those metrics must be imported as metadata or computed with an external tool rather than implied by the available columns.

## 9. ADT and multimodal quality control

ADT panels often include control antibodies that should be marked inactive in feature metadata after their names are inspected.
RNA, ADT, and HTO assays share one cell table, so changing `I` applies the same cell selection across modalities.
Check whether an RNA-driven filter is appropriate for the protein question before reusing it automatically.
Hashtag demultiplexing is covered separately in {doc}`hto_demultiplexing`.

## Common mistakes and limitations

- Copying thresholds from another dataset without checking distributions
- Pooling samples with different depth distributions and then applying one global bound
- Passing `min_p` or `max_p` to the sample-aware MAD path
- Expecting `run_doublet_detection` to drop cells (it only scores)
- Running doublet detection before building the neighbourhood graph and clustering
- Claiming FRiP or TSS enrichment from the ATAC metrics Scarf currently provides
