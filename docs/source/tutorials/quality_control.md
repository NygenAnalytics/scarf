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
Scarf keeps the count matrix intact. Each filter returns a saved selection that you can pass to
the next analysis step. This guide starts with inspecting the cells and applying the default
automatic filter, then shows when to choose your own thresholds.

## Prerequisites

- {doc}`scrna_seq` or {ref}`quickstart <quickstart>`
- A Zarr store for an RNA assay

## What you will learn

- Inspect per-cell QC columns
- Apply the default automatic filter and inspect its result
- Set manual thresholds or filter each sample separately
- Compute doublet scores after an initial clustering
- Recognize current RNA, ATAC, and ADT support boundaries

## Standalone setup

Quality control needs the population before filtering, so this page imports raw counts into a
separate `quality_control.zarr` store. Running the setup again replaces that tutorial store.
We set `min_features_per_cell=10` to retain low-feature cells for inspection before choosing
stricter thresholds.

```{code-cell} ipython3
# Arrange and save Matplotlib figures.
import matplotlib.pyplot as plt
# Work with numeric arrays and cell masks.
import numpy as np
# Summarize cells and results in tables.
import pandas as pd

# Open count stores and run Scarf analyses.
import scarf

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the raw 10x count file for quality control.
counts = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]

# Choose a separate store for this quality-control example.
store = counts.with_name("quality_control.zarr")
# Open a reader for the selected source format.
reader = scarf.CrH5Reader(str(counts))
# Write the converted count store.
scarf.CrToZarr(reader, zarr_loc=str(store)).dump()

# Open the count store for this analysis.
ds = scarf.DataStore(str(store), nthreads=4, min_features_per_cell=10)
# Inspect the opened store's cells and features.
ds
```

The `I` {term}`cell key` marks the cells retained when the store is opened. Filtering saves a new
selection and leaves `I` and the counts unchanged.

## 1. Inspect QC distributions

Scarf prepares QC columns when a new store is first opened. These include total counts, detected
features, and mitochondrial and ribosomal percentages when the gene names match their patterns.

```{code-cell} ipython3
# Keep the quality measurements available in this store.
qc_cols = [
    c
    for c in ("RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo")
    if c in ds.cells.columns
]
# Freeze the input cells before comparing filters.
qc_cell_selection = ds.snapshot_cell_selection("I")
# Inspect the input distributions of the available quality measurements.
ds.plots.distribution(keys=qc_cols, cell_selection=qc_cell_selection)
```

Each violin uses an immutable snapshot of `I`, so cells already below `min_features_per_cell` from
open are excluded. Use the tails to set further cutoffs.

## 2. Start with automatic thresholds

`auto_filter_cells` uses the median absolute deviation (MAD), a measure of spread that is less
affected by extreme values than the standard deviation. Its default bounds are three scaled
MADs from the median. Counts and detected features use log1p values and two-sided bounds;
mitochondrial and ribosomal percentages use upper bounds only.

```{code-cell} ipython3
# Filter cells with the default robust thresholds.
automatic_selection = ds.auto_filter_cells(cell_selection=qc_cell_selection)
# Load the cells retained by automatic filtering.
automatic_mask = np.asarray(ds.load_artifact(automatic_selection)["values"][:], dtype=bool)
# Count cells retained by automatic filtering.
print(f"Cells after automatic filtering: {int(automatic_mask.sum())}")
# Inspect the same quality measurements after automatic filtering.
ds.plots.distribution(keys=qc_cols, cell_selection=automatic_selection)
```

Compare these distributions with the input before accepting the result. Automatic bounds are a
starting point; a rare population may have a different count-depth distribution.

## 3. Choose manual thresholds when needed

Thresholds are dataset-specific.
The values below match the PBMC example in {doc}`scrna_seq`.
Filtering returns a new immutable selection and leaves cell key `I` unchanged.
The named QC columns are read from current cell metadata when the method is called. If any
explicitly named column is absent, filtering raises an error instead of silently omitting it.

```{code-cell} ipython3
# Count cells before applying manual thresholds.
n_before = int(ds.cells.fetch_all("I").sum())
# Define the manual count, feature, and mitochondrial thresholds.
manual_filter = {
    "attrs": ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    "highs": [15000, 4000, 15],
    "lows": [1000, 500, 0],
}
# Apply the manual thresholds to the input cells.
manual_selection = ds.filter_cells(**manual_filter)
# Load the cells retained by manual filtering.
manual_mask = np.asarray(ds.load_artifact(manual_selection)["values"][:], dtype=bool)
# Report the input cell count.
print(f"Cells in input selection: {n_before}")
# Report the retained cell count.
print(f"Cells in filtered selection: {int(manual_mask.sum())}")
# Summarize the selected values and their spread.
pd.DataFrame({key: ds.cells.fetch_all(key)[manual_mask] for key in qc_cols}).describe()
```

The filtered summary should lose the long low-count tail and high-mito shoulder. Roughly a fifth of
the barcodes drop out here, which is typical for this dataset and is the number worth
sanity-checking against your own expectations before continuing.

The automatic and manual selections are alternatives here. The doublet example below uses the
manual thresholds so it follows the same cells as the core PBMC workflow. To combine filters in
your own analysis, pass the previous result as `cell_selection=`.

## 4. Per-sample MAD filtering

Global bounds can penalize a sample whose count-depth distribution differs from the pooled distribution.
With `sample_column`, Scarf calculates robust bounds within each sample using the median absolute deviation (MAD).

The PBMC teaching dataset has no biological sample column.
Use a real sample column when the dataset contains multiple donors or batches. The call has the
same immutable-selection contract as the global filter:

```python
# Estimate quality thresholds separately for each sample.
sample_selection = ds.auto_filter_cells(
    cell_selection=qc_cell_selection,
    sample_column="sample_id",
)
```

This example needs a real `sample_id` column, which the PBMC dataset does not have.
Samples with fewer than 20 selected cells are retained with a warning because their bounds
cannot be estimated reliably. The same rule applies to a pooled selection with fewer than 20 cells.

The same options can be forwarded through the standard pipeline:

```python
# Pass the sample-specific filtering choice through the pipeline.
ds.pipeline.run(
    filtering={"sample_column": "sample_id"},
)
```

## 5. RNA percentages and feature exclusions

Ingestion-owned mitochondrial and ribosomal percentage columns measure the fraction of each cell's
counts matching configured gene-name patterns. High values can indicate damaged cells or
study-specific biology. Inspect their distributions before applying upper thresholds.

The default mitochondrial pattern is case-insensitive `^MT-`.
It matches names such as `MT-CO1` and `mt-Co1` without including `MTOR` or metallothioneins.
Percentage columns are fixed when a store is first prepared. That first open for writing discards
any existing column with a percentage name, logs a warning, and computes the percentage from the
counts. Later opens keep the stored values when `mito_pattern` and `ribo_pattern` are omitted. An
explicit pattern that differs from the recorded one raises an error instead of replacing prepared
data. To apply a different pattern, import the data into a fresh store and pass the pattern on
its first open, or compute a separate `quality_metric` artifact as shown below.

For another gene set, create an explicit feature selection and calculate its percentage over an
explicit cell selection. The datastore method returns a `quality_metric` artifact and does not add
a cell column:

```{code-cell} ipython3
# Read the RNA feature names.
feature_names = ds.RNA.feats.fetch_all("names").astype(str)
# Select genes whose names start with HSP.
stress_features = ds.set_feature_selection(
    from_assay="RNA",
    mask=np.char.startswith(feature_names, "HSP"),
)
# Calculate the selected genes' count percentage in each retained cell.
stress_percentage = ds.run_feature_percentage(manual_selection, stress_features)
# Load the calculated percentages.
stress_values = np.asarray(ds.load_artifact(stress_percentage)["values"][:])
# Summarize the selected values and their spread.
pd.Series(stress_values, name="percent stress features").describe()
```

Gene families excluded from the graph are a separate feature-selection decision.
See {doc}`feature_selection` for the default HVG blacklist and supported overrides.

## 6. Doublet scores

The pipeline builds a graph and clusters before calculating doublet scores. It does not remove
cells automatically. Here we reuse the manual QC thresholds and the prepared PBMC example's
500 genes, 15 PCs, and Leiden resolution 0.5. These are teaching-dataset settings, not the API
defaults. We skip cell-cycle scoring, Paris, and markers because they are not needed for this check.

```{code-cell} ipython3
# Run the graph and clustering steps needed for doublet scoring.
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
# Keep the doublet-score artifact for further filtering.
doublets = doublet_run["doublets"]
# Read the doublet scores in the analysis cell order.
scores = np.asarray(doublet_run.cells.fetch("doublet_score"))
# Name the scores for summaries and plots.
scores_series = pd.Series(scores, name="doublet_score")
# Summarize the selected values and their spread.
scores_series.describe()
```

Locate high doublet scores on the embedding:

```{code-cell} ipython3
# Locate cells with high doublet scores on the UMAP.
ds.plots.embedding(run=doublet_run, color_by="doublet_score", sort_values=True)
```

Higher doublet scores mark cells that map near simulated doublets.
Inspect the score distribution before applying a cutoff:

```{code-cell} ipython3
# Plot the doublet-score distribution.
scores_series.plot(kind="hist", bins=40, xlabel="Doublet score")
# Display the completed figure.
plt.show()
```

The score distribution and embedding should be reviewed together.
A threshold is study-dependent, and `run_doublet_detection` does not remove cells.
After choosing an upper bound from the score distribution, apply it as an additional filter.
The teaching cutoff below identifies the upper 5% of scores on this PBMC run; replace it with a
study-specific value when the upper-tail shape differs.

```{code-cell} ipython3
# Use the 95th percentile as this example's upper cutoff.
doublet_threshold = float(scores_series.quantile(0.95))
# Report the selected doublet-score cutoff.
print(f"Doublet threshold (95th percentile): {doublet_threshold:.4f}")
# Count cells excluded by the cutoff.
print(f"Cells above threshold: {int((scores > doublet_threshold).sum())}")
```

Compose the upper bound with the score artifact's stored input selection. This retains scores at
the bound and leaves live metadata unchanged:

```{code-cell} ipython3
# Retain cells whose score is at or below the chosen cutoff.
doublet_filtered = ds.select_cells(doublets, high=doublet_threshold, keep_bounds=True)
# Count the cells retained after applying the cutoff.
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
- Expecting `run_doublet_detection` to drop cells (it only scores)
- Running doublet detection before building the neighbourhood graph and clustering
- Claiming FRiP or TSS enrichment from the ATAC metrics Scarf currently provides
