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
Scarf keeps the source matrix intact and returns immutable selection artifacts from analytical
filters. This guide covers manual, global automatic, and sample-aware filtering, followed by
assay-specific checks.

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
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

counts = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]

store = counts.with_name("data.zarr")
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

The `I` {term}`cell key` is the initial live selection. Filtering snapshots that input and returns a
new immutable selection artifact. It leaves all rows and the live key unchanged. Feature
selections are immutable artifacts aligned to one assay.

## 1. Inspect QC distributions

On open, Scarf streams the count matrix once to compute initialization statistics: columns such as
`RNA_nCounts`, `RNA_nFeatures`, and mito/ribo percentages when gene names match the configured
patterns, plus per-feature detection statistics used by explicit selection producers.

```{code-cell} ipython3
qc_cols = [
    c
    for c in ("RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo")
    if c in ds.cells.columns
]
qc_cell_selection = ds.snapshot_cell_selection("I")
ds.plots.distribution(
    keys=qc_cols,
    cell_selection=qc_cell_selection,
    kind="violin",
    max_points=2000,
)
```

Each violin uses an immutable snapshot of `I`, so cells already below `min_features_per_cell` from
open are excluded. Use the tails to set further cutoffs.

## 2. Manual thresholds

Thresholds are dataset-specific.
The values below match the PBMC example in {doc}`scrna_seq`.
Filtering returns a new immutable selection and leaves cell key `I` unchanged.
The named QC columns are read from current cell metadata when the method is called. If any
explicitly named column is absent, filtering raises an error instead of silently omitting it.

```{code-cell} ipython3
n_before = int(ds.cells.fetch_all("I").sum())
manual_filter = {
    "attrs": ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    "highs": [15000, 4000, 15],
    "lows": [1000, 500, 0],
}
manual_selection = ds.filter_cells(**manual_filter)
manual_mask = np.asarray(ds.load_artifact(manual_selection)["values"][:], dtype=bool)
print(f"Cells in input selection: {n_before}")
print(f"Cells in filtered selection: {int(manual_mask.sum())}")
pd.DataFrame({key: ds.cells.fetch_all(key)[manual_mask] for key in qc_cols}).describe()
```

The filtered summary should lose the long low-count tail and high-mito shoulder. Roughly a fifth of
the barcodes drop out here, which is typical for this dataset and is the number worth
sanity-checking against your own expectations before continuing.

## 3. Automatic thresholds

`auto_filter_cells` fits a normal distribution (`loc=median`, `scale=std`) to each QC column, then takes quantiles at `min_p` and `max_p` via `scipy.stats.norm.ppf` (defaults 0.01 and 0.99).
Default columns are nCounts, nFeatures, percentMito, and percentRibo when present.
Pass the previous selection explicitly to compose filters.

```{code-cell} ipython3
automatic_selection = ds.auto_filter_cells(cell_selection=manual_selection)
automatic_mask = np.asarray(
    ds.load_artifact(automatic_selection)["values"][:],
    dtype=bool,
)
print(f"Cells after automatic refinement: {int(automatic_mask.sum())}")
```

Inspect the selected values through the returned mask before accepting the thresholds.

## 4. Per-sample MAD filtering

Global bounds can penalize a sample whose count-depth distribution differs from the pooled distribution.
With `sample_column`, Scarf calculates robust bounds within each sample using the median absolute deviation (MAD).

The PBMC teaching dataset has no biological sample column.
Use a real sample column when the dataset contains multiple donors or batches. The call has the
same immutable-selection contract as the global filter:

```python
sample_selection = ds.auto_filter_cells(
    attrs=qc_cols,
    cell_selection=manual_selection,
    sample_column="sample_id",
    n_mads=3.0,
    min_cells_per_sample=20,
)
```

The PBMC page does not execute this call because inventing sample assignments would produce a
mechanics demo with no biological interpretation.

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
)
```

## 5. RNA percentages and feature exclusions

Ingestion-owned mitochondrial and ribosomal percentage columns measure the fraction of each cell's
counts matching configured gene-name patterns. High values can indicate damaged cells or
study-specific biology. Inspect their distributions before applying upper thresholds.

For another gene set, create an explicit feature selection and calculate its percentage over an
explicit cell selection. The datastore method returns a `quality_metric` artifact and does not add
a cell column:

```{code-cell} ipython3
feature_names = ds.RNA.feats.fetch_all("names").astype(str)
stress_features = ds.set_feature_selection(
    from_assay="RNA",
    mask=np.char.startswith(feature_names, "HSP"),
)
stress_percentage = ds.run_feature_percentage(
    manual_selection,
    stress_features,
)
stress_values = np.asarray(ds.load_artifact(stress_percentage)["values"][:])
pd.Series(stress_values, name="percent stress features").describe()
```

Gene families excluded from the graph are a separate feature-selection decision.
See {doc}`feature_selection` for the default HVG blacklist and supported overrides.

## 6. Doublet scores

`run_doublet_detection` is the atomic score producer. The standard pipeline supplies its exact
clustering and graph refs, and returns the same per-cell doublet artifact. It does not remove cells
automatically. Reusing the manual-filter mapping keeps this example's QC bounds in one place.

```{code-cell} ipython3
doublet_run = ds.pipeline.run(
    filtering={"method": "manual", **manual_filter},
    hvg_count=500,
    pca_dims=15,
    leiden={"partitions": [0.5]},
    cell_cycle=False,
    paris=False,
    doublets=True,
    markers=False,
)
doublets = doublet_run["doublets"]
scores = np.asarray(doublet_run.cells.fetch("doublet_score"))
ds.plots.embedding(
    run=doublet_run,
    color_by="doublet_score",
    sort_values=True,
)
```

Higher doublet scores mark cells that map near simulated doublets.
Inspect the score distribution before applying a cutoff:

```{code-cell} ipython3
pd.Series(scores, name="doublet_score").plot(kind="hist", bins=40)
```

The score distribution and embedding should be reviewed together.
A threshold is study-dependent, and `run_doublet_detection` does not remove cells.
After choosing an upper bound from the score distribution, apply it as an additional filter.
The teaching cutoff below identifies the upper 5% of scores on this PBMC run; replace it with a
study-specific value when the upper-tail shape differs.

```{code-cell} ipython3
scores_series = pd.Series(scores, name="doublet_score")
print(scores_series.describe())
doublet_threshold = float(scores_series.quantile(0.95))
print(f"Doublet threshold (95th percentile): {doublet_threshold:.4f}")
print(f"Cells above threshold: {int((scores > doublet_threshold).sum())}")
```

Compose the upper bound with the score artifact's stored input selection. This retains scores at
the bound and leaves live metadata unchanged:

```{code-cell} ipython3
doublet_filtered = ds.select_cells(
    doublets,
    high=doublet_threshold,
    keep_bounds=True,
)
int(np.asarray(ds.load_artifact(doublet_filtered)["values"][:]).sum())
```

`select_cells` accepts any numeric cell artifact with a one-dimensional `values` payload. `low`
and `high` define the retained range. By default it composes with the source artifact's selection;
`cell_selection=` can narrow that input further but cannot add cells absent from the source.

## 7. ATAC quality control

Scarf initializes per-cell ATAC fragment or cut-site counts and accessible-peak counts, and it records per-peak detection statistics for explicit prevalent-peak selection.
`select_prevalent_peaks` returns the peak selection used for LSI and graph construction.
Scarf does not currently calculate FRiP or TSS enrichment, so those metrics must be imported as metadata or computed with an external tool rather than implied by the available columns.

## 8. ADT and multimodal quality control

ADT panels often include control antibodies that should be excluded with an explicit feature
selection after their names are inspected. RNA, ADT, and HTO assays share one cell table, so one
cell-selection artifact can be passed to each compatible assay operation. Check whether an
RNA-driven filter is appropriate for the protein question before reusing it automatically.
Hashtag demultiplexing is covered separately in {doc}`hto_demultiplexing`.

## Common mistakes and limitations

- Copying thresholds from another dataset without checking distributions
- Pooling samples with different depth distributions and then applying one global bound
- Passing `min_p` or `max_p` to the sample-aware MAD path
- Expecting `run_doublet_detection` to drop cells (it only scores)
- Running doublet detection before building the neighbourhood graph and clustering
- Claiming FRiP or TSS enrichment from the ATAC metrics Scarf currently provides
