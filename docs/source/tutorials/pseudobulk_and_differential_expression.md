---
description: Aggregate counts with make_bulk and export them for external differential expression.
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

Scarf aggregates counts and exports them. Use a replicate-aware external method such as edgeR or
DESeq2 for condition-level differential expression. Scarf marker tables answer a different,
cell-level question and must not be reported as replicate-aware differential expression.

## 1. Open the frozen baseline

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "kang_15K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
run = ds.pipeline.open(label="docs_default")
ds.plots.embedding(run=run, layout="umap", color_by="clusters")
```

The rebuilt teaching store records its clustering and marker table as immutable outputs of this
exact run.

## 2. Keep marker interpretation separate

```{code-cell} ipython3
marker_ref = run["markers"]
cluster_values = np.asarray(run.cells.fetch("clusters"))
group_id = pd.Series(cluster_values).value_counts().index[0]
markers = ds.get_markers(
    marker=marker_ref,
    group_id=group_id,
    min_score=-1,
    min_frac_exp=-1,
)
markers[
    [
        "feature_name",
        "score",
        "frac_exp",
        "auc",
        "p_value",
        "p_value_adjusted",
    ]
].head()
```

The adjusted values use the marker-table correction scope documented in
{doc}`../reference/api/datastore`. They are not biological-replicate FDR values.

```{note}
Use marker tables for cluster interpretation.
Do not treat within-group `p_value_adjusted` columns as replicate-aware differential expression results.
```

## 3. Run descriptive group comparisons

`run_statistical_testing` compares one feature at a time across an exact categorical artifact and
persists each variant as an immutable result artifact. The rank-based defaults suit single-cell
data: `mann_whitney` for two groups, `kruskal_wallis` with optional Dunn post-hoc comparisons for
three or more, and paired Wilcoxon on aggregated samples.

Explicit parametric tests also run on raw cell-level values. They are descriptive only: they treat
cells as independent, cannot be combined with sample aggregation, and make assumptions that
zero-inflated single-cell values often violate. Automatic method selection never chooses a
parametric test.

```{code-cell} ipython3
# Compare two clusters that are present in the Kang dataset.
result = ds.run_statistical_testing(
    ["MALAT1"],
    run["clusters"],
    groups=[1, 2],
    test="welch",
    alternative="greater",
)

# Pin retrieval to the exact immutable result returned by statistical testing.
loaded = ds.get_statistical_tests(result.artifact)
loaded.tables["MALAT1"]
```

```{code-cell} ipython3
# The plotted selection and test design must match the stored result.
ds.plots.distribution(
    ["MALAT1"],
    grouping=run["clusters"],
    groups=[1, 2],
    kind="violin",
    stats_results=result.artifact,
)
```

Brackets prefer the pooled `p_value_adjusted` column when present.

## 4. Aggregate raw counts

`make_bulk` accepts the same exact categorical artifact. Its lineage supplies the cell selection,
so no cluster column needs to be created. Cell counts per Leiden group set the scale for each bulk
column:

```{code-cell} ipython3
group_sizes = (
    pd.Series(cluster_values, name="group")
    .astype(str)
    .value_counts()
    .sort_index()
)
```

```{code-cell} ipython3
bulk, fractions = ds.make_bulk(
    run["clusters"],
    from_assay="RNA",
    aggr_type="sum",
    feature_label="name",
    return_fraction=True,
)
totals = bulk.sum().rename("total_counts")
totals.index = totals.index.astype(str)
sample_metadata = pd.concat(
    [group_sizes.rename("n_cells"), totals],
    axis=1,
)
sample_metadata
```

`bulk` contains summed raw counts. `fractions` contains the fraction of cells with a nonzero count
for the same feature and group.

Pass a user-owned metadata column name as `groups` when its categories were authored outside an
analysis producer. Use `cell_selection=` to restrict either input path explicitly; for an artifact,
the requested selection must be a subset of its stored selection. For nested aggregation,
`secondary_groups=` accepts another compatible categorical artifact or metadata column.

```{code-cell} ipython3
top_features = bulk.sum(axis=1).sort_values(ascending=False).head(8).index
bulk.loc[top_features], fractions.loc[top_features]
```

For a real condition comparison, the grouping must retain biological sample identity, usually as
`secondary_groups="sample_id"` alongside the cell-type artifact. Merge samples first as described
in {doc}`dataset_merging`.

## 5. Export counts and sample metadata

```{code-cell} ipython3
export_directory = TemporaryDirectory()
counts_csv = Path(export_directory.name) / "pseudobulk_counts.csv"
metadata_csv = Path(export_directory.name) / "pseudobulk_metadata.csv"

bulk.to_csv(counts_csv)
sample_metadata.index.name = "group"
sample_metadata.to_csv(metadata_csv)

pd.read_csv(counts_csv, index_col=0, nrows=5).iloc[:, :5]
```

Use the count matrix and design metadata with a method appropriate to the study. Scarf does not fit
edgeR or DESeq2 models.

## About pseudo-replicates

`make_bulk(..., pseudo_reps=2)` randomly divides cells within each group. These splits can be useful
for descriptive stability checks, but they are not independent biological replicates and must not
be used as such in differential expression.

## Common mistakes

- Reporting marker-table adjusted p-values as replicate-aware differential expression
- Aggregating by cell type while discarding sample or donor identity
- Treating random cell splits as biological replicates
- Expecting Scarf to fit an external count model
