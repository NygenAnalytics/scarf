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

Quality control (QC) defines the cells and features that downstream analyses can use. QC across the assays is arguably one of the most critical parts of the data analysis due to the great influence it can have on downstream tools. Scarf allows for a greater deal of flexibility during QC, as it keeps the count matrix intact. This means that after applying a QC threshold, if you realize it doesn't appropriately model error in your data, you can simply reset to the original store instead of permanently losing those cells like in other methods. Each filter returns a saved selection that you can pass to the next analysis step. This guide starts with inspecting the cells and applying the default automatic filter, then shows how to choose your own thresholds.

## Initial setup

Quality control needs the raw population before filtering, thus this page imports the raw counts of the dataset we analyze. Our first piece of QC that we can immediately apply as we import the dataset is `min_features_per_cell=10`; running this simply hides all cells with less than 10 genes present, removing them from our active cell selection and analysis. We set `min_features_per_cell=10` to retain low-feature cells for inspection before choosing stricter thresholds.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

counts = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]

store = counts.with_name("quality_control.zarr")
reader = scarf.CrH5Reader(str(counts))
scarf.CrToZarr(reader, zarr_loc=str(store)).dump()

ds = scarf.DataStore(str(store), nthreads=4, min_features_per_cell=10)
ds
```

The `I` {term}`cell key` marks the cells retained when the store is opened. Filtering saves a new
selection and leaves `I` and the counts unchanged. If you need to revert to the entire counts store refer to {doc}`reuse_and_tracing`.

## Inspect QC distributions

Scarf prepares QC columns when a new store is first opened, with these including total counts, detected features, and mitochondrial and ribosomal percentages when the gene names match their patterns.

```{code-cell} ipython3
qc_cols = [
    c
    for c in ("RNA_nCounts", "RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo")
    if c in ds.cells.columns
]
qc_cell_selection = ds.snapshot_cell_selection("I")
ds.plots.distribution(keys=qc_cols, cell_selection=qc_cell_selection)
```

Each violin uses the active snapshot of `I`, so cells already below `min_features_per_cell` are excluded at open. You can then, from these violin plots, use the tails of the distributions to set bounds that remove outlying cells.

## Using automatic thresholds

If you want to automatically perform the QC, you can use the `auto_filter_cells` function, which uses the median absolute deviation (MAD), a measure of spread that is less affected by extreme values than the standard deviation to filter cells. Its default bounds are three scaled MADs from the median. Counts and detected features use log1p values and two-sided bounds;
mitochondrial and ribosomal percentages use upper bounds only and not the lower bounds.

```{code-cell} ipython3
automatic_selection = ds.auto_filter_cells(cell_selection=qc_cell_selection)
automatic_mask = np.asarray(ds.load_artifact(automatic_selection)["values"][:], dtype=bool)
print(f"Cells after automatic filtering: {int(automatic_mask.sum())}")
ds.plots.distribution(keys=qc_cols, cell_selection=automatic_selection)
```

Compare these distributions with the input before accepting the result. Automatic bounds are a
starting point; a rare population may have a different count-depth distribution.

## Choosing manual thresholds

Thresholds are dataset-specific, thus you usually have to manually define your thresholds.
The values below are simply based off the visualized distributions of the data Scarf automatically calculates (as discussed above). Filtering returns a new selection and leaves cell key `I` unchanged. The named QC columns are read from current cell metadata when the method is called. If any explicitly named column is absent, filtering raises an error; a common reason for an error is the mito or ribosomal data not being calculated as the gene names are Ensembl IDs.

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

The filtered summary should lose the long low-count tail and high-mito shoulder. Roughly a fifth of the cells drop out here, which is typical for this dataset and is the number worth
sanity-checking against your own expectations before continuing.

The automatic and manual selections are alternatives here. The doublet example below uses the
manual thresholds so it follows the same cells as the core PBMC workflow. To combine filters in
your own analysis, pass the previous result as `cell_selection=`.

## Per-sample MAD filtering

Global bounds can penalize a sample whose count-depth distribution differs from the pooled distribution, thus performing the MAD filtering separately for each specific sample may solve the issue. With `sample_column`, Scarf calculates robust bounds within each sample using the median absolute deviation (MAD) instead of for the global bound.

```python
sample_selection = ds.auto_filter_cells(
    cell_selection=qc_cell_selection,
    sample_column="sample_id",
)
```

This example needs a real `sample_id` column, which the PBMC dataset does not have.
Samples with fewer than 20 selected cells are retained with a warning because their bounds
cannot be estimated reliably. The same rule applies to a pooled selection with fewer than 20 cells (although this is very rare nowadays).

The same options can be forwarded through the standard pipeline:

```python
ds.pipeline.run(
    filtering={"sample_column": "sample_id"},
)
```

## RNA percentages and feature exclusions

The mitochondrial and ribosomal percentage columns measure the fraction of each cell's counts matching configured gene-name patterns. These gene-name patterns are simply searching for the genes that can represent information about the mitochondria or the ribosome, which both have their own RNA. High values of both mitochondrial and/or ribosomal counts can indicate damaged cells or study-specific biology. Inspect their distributions before applying upper thresholds if you are applying thresholds manually.

The default mitochondrial pattern is case-insensitive to genes that start with `^MT-`.
It matches names such as `MT-CO1` and `mt-Co1` without including `MTOR` for example.
Percentage columns are created when the store is first created, and is thus one of the reasons the first load of the store may take longer than expected. That first open for writing discards
any existing column with a percentage name, logs a warning, and computes the percentage from the counts on its own. Later opens keep the stored values when `mito_pattern` and `ribo_pattern` are omitted. To apply a different pattern for detecting the mitochondrial and ribosomal statistics, import the data into a fresh store and pass the pattern on its first open, or compute a separate `quality_metric` artifact as shown below.

For another gene set, create an explicit feature selection and calculate its percentage over an
explicit cell selection. The datastore method returns a `quality_metric` artifact and does not add
a cell column:

```{code-cell} ipython3
feature_names = ds.RNA.feats.fetch_all("names").astype(str)
stress_features = ds.set_feature_selection(
    from_assay="RNA",
    mask=np.char.startswith(feature_names, "HSP"),
)
stress_percentage = ds.run_feature_percentage(manual_selection, stress_features)
stress_values = np.asarray(ds.load_artifact(stress_percentage)["values"][:])
pd.Series(stress_values, name="percent stress features").describe()
```

Gene families excluded from the graph are a separate feature-selection decision.
See {doc}`feature_selection` for the default HVG blacklist and supported overrides.

## Removing doublets

Sometimes, during the wet-lab process behind the sequencing of the cells, you may end up with doublets or multiplets: these are droplets that end up with multiple cells present inside of them, thus resulting in 2 cells being sequenced and posing as one. Doublets can often be a source of false biological signals in your dataset and a confounder, thus removing them during the QC process is highly recommended.

The doublet removal pipeline builds a graph and clusters before calculating doublet scores. It does not remove cells automatically. Here we reuse the manual QC thresholds and the following parameters for removal, which is 500 highly variable genes, 15 PCs, and a Leiden resolution of 0.5. These are teaching-dataset settings, not the API defaults. We skip cell-cycle scoring, Paris, and markers because they are not needed for this check.

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
scores_series = pd.Series(scores, name="doublet_score")
scores_series.describe()
```

Locate high doublet scores on the embedding:

```{code-cell} ipython3
ds.plots.embedding(run=doublet_run, color_by="doublet_score", sort_values=True)
```

Higher doublet scores mark cells that map near simulated doublets. Inspect the score distribution before applying a cutoff:

```{code-cell} ipython3
scores_series.plot(kind="hist", bins=40, xlabel="Doublet score")
plt.show()
```

The score distribution and embedding should be reviewed together.
A threshold is study-dependent, and `run_doublet_detection` does not remove cells.
After choosing an upper bound from the score distribution, apply it as an additional filter.
The teaching cutoff below identifies the upper 5% of scores on this PBMC run; replace it with a
study-specific value when the upper-tail shape differs.

```{code-cell} ipython3
doublet_threshold = float(scores_series.quantile(0.95))
print(f"Doublet threshold (95th percentile): {doublet_threshold:.4f}")
print(f"Cells above threshold: {int((scores > doublet_threshold).sum())}")
```

Compose the upper bound with the score artifact's stored input selection. This retains scores at
the bound and leaves live metadata unchanged:

```{code-cell} ipython3
doublet_filtered = ds.select_cells(doublets, high=doublet_threshold, keep_bounds=True)
int(np.asarray(ds.load_artifact(doublet_filtered)["values"][:]).sum())
```

## ATAC quality control

Scarf initializes per-cell ATAC fragment or cut-site counts and accessible-peak counts, and it records per-peak detection statistics for explicit prevalent-peak selection.
`select_prevalent_peaks` returns the peak selection used for LSI and graph construction.
Scarf does not currently calculate FRiP or TSS enrichment, so those metrics must be imported as metadata or computed with an external tool rather than implied by the available columns for QC.

## ADT and multimodal quality control (CITE-seq)

ADT panels often include control antibodies that should be excluded with an explicit feature
selection after their names are inspected. RNA, ADT, and HTO assays share one cell table, so one cell-selection artifact can be passed to each compatible assay operation. Check whether an
RNA-driven filter is appropriate for the protein question before reusing it automatically.
Hashtag demultiplexing is covered separately in {doc}`hto_demultiplexing`.

## Common mistakes

- **Pooling samples with different depth distributions and then applying one global bound:** A deeply sequenced sample will dominate the pooled bounds, so pass `sample_column` to `auto_filter_cells` and let each sample's bounds come from its own distribution.
- **Running doublet detection before building the neighborhood graph and clustering**: Scores are computed against simulated doublets placed on that graph, so build and cluster first and review the score embedding together with the distribution before cutting.
- **Reusing the teaching 95th-percentile cutoff as a default:** It marks the upper 5% of this PBMC run only, so replace it with a study-specific value when the upper-tail shape differs.
