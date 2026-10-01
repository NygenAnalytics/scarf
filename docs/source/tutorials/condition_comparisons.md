---
description: Compare rheumatoid arthritis and control gamma-delta T cells with a paired, donor-level Wilcoxon workflow on the Binvignat PBMC dataset.
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
# Comparing biological conditions with statistical testing

Comparing biological conditions with statistical testing can be used for exploratory analysis to see differences in potential gene expression across conditions. In this tutorial, we ask a simple and narrow question, do matched rheumatoid arthritis (RA) and control donors differ in gamma-delta T-cell expression of the 6 genes of our choice? For the general analysis here, we average cells within donors, and then conduct a paired Wilcoxon signed-rank test across the 18 matched RA-control donor pairs.

This avoids treating 1,386 cells as independent biological replicates, avoiding the issue of **pseudoreplication**.

The result is a sample-level distribution test on normalized expression, not a raw-count pseudobulk differential expression model.

## Dataset and prerequisites

The data for this tutorial comes from the [CELLxGENE collection](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b) from the [Binvignat et al. paper](https://doi.org/10.1172/jci.insight.178499).
This page takes the extracted CELLxGENE H5AD. Because of this, to replicate this tutorial, ensure you have enough local disk space for both the H5AD and its converted Zarr store, plus network access on the first run.

The dataset URL below is used only to download a file; SCARF **does not** download the model for you.

## Download, inspect, and mount the count source

Download the H5AD if it is absent, then inspect it before conversion.
The assertions pin the matrix choice and dimensions used for this analysis.
`raw/X` contains integer-like counts, while `X` is also present as another matrix candidate.

```{code-cell}
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

import scarf
from scarf.plotting import CellField, StudyDesign

scarf.configure_output(level="WARNING", progress=False)

DATA_URL = (
    "https://datasets.cellxgene.cziscience.com/"
    "3b751975-34bb-409a-a9b7-98380f0450ea.h5ad"
)
dataset_directory = Path(environ.get("SCARF_DOCS_DATA_DIR", "scarf_datasets"))
dataset_directory.mkdir(parents=True, exist_ok=True)
h5ad_path = dataset_directory / "binvignat_ra_pbmc.h5ad"
source_store = dataset_directory / "binvignat_ra_pbmc.zarr"

if not h5ad_path.exists():
    partial_path = h5ad_path.with_suffix(".h5ad.part")
    urlretrieve(DATA_URL, partial_path)
    partial_path.replace(h5ad_path)

inspection = scarf.inspect_h5ad(str(h5ad_path))
assert inspection.matrixKey == "raw/X"
assert {"raw/X", "X"}.issubset(inspection.matrixCandidates)
assert inspection.integerLike is True
assert (inspection.nCells, inspection.nFeatures) == (108_717, 21_648)
{
    "matrix": inspection.matrixKey,
    "encoding": inspection.matrixEncoding,
    "integer-like": inspection.integerLike,
    "shape": (inspection.nCells, inspection.nFeatures),
}
```

Convert only when the reusable local source store is absent.
The reader is built from the inspected keys rather than from assumptions about the H5AD layout.

```{code-cell}
if not source_store.exists():
    with TemporaryDirectory(dir=dataset_directory) as conversion_directory:
        staged_store = Path(conversion_directory) / source_store.name
        reader = scarf.H5adReader.from_inspect(inspection)
        try:
            scarf.H5adToZarr(
                reader,
                zarr_loc=str(staged_store),
                assay_name="RNA",
                nthreads=4,
            ).dump()
        finally:
            reader.h5.close()
        staged_store.replace(source_store)
```

Initialize the count source with no feature-count filter, then mount it into a temporary writable analysis store.
The source continues to own the count matrix.
The mounted target owns the cell selection and statistical artifacts created below.

```{code-cell}
source = scarf.DataStore(
    str(source_store),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
assert source.cells.N == 108_717

analysis_directory = TemporaryDirectory()
ds = scarf.mount_datastore(
    str(source_store),
    at=str(Path(analysis_directory.name) / "condition_analysis.zarr"),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
```

## Freeze the matched gamma-delta T-cell cohort

Here, all we do is select our cells of interest that publication has already labeled, `yd T cells` .

```{code-cell}
GROUPS = ["normal", "rheumatoid arthritis"]

fine_annotation = np.asarray(ds.cells.fetch_all("fine_annot"), dtype=object)
pair_index = np.asarray(
    ds.cells.fetch_all("pair_index_CW"),
    dtype=np.float64,
)
matched_gamma_delta = np.isfinite(pair_index) & (
    fine_annotation == "yd T cells"
)

ds.cells.insert(
    "matched_yd_t_cells",
    matched_gamma_delta,
    overwrite=True,
)
cells = ds.snapshot_cell_selection("matched_yd_t_cells")
```

Before you run any tests, ensure that each donor must map to one disease group and one pair, and every pair must contain one donor from each group.

```{code-cell}
donor_id = np.asarray(ds.cells.fetch_all("donor_id"), dtype=object)
disease = np.asarray(ds.cells.fetch_all("disease"), dtype=object)

donor_design = pd.DataFrame(
    {
        "donor_id": donor_id[matched_gamma_delta],
        "disease": disease[matched_gamma_delta],
        "pair_index_CW": pair_index[matched_gamma_delta],
    }
).drop_duplicates()

assert int(matched_gamma_delta.sum()) == 1_386
assert len(donor_design) == 36
assert donor_design["donor_id"].nunique() == 36
assert donor_design["pair_index_CW"].nunique() == 18
assert set(donor_design["disease"]) == set(GROUPS)

pair_balance = donor_design.groupby(
    "pair_index_CW",
    observed=True,
).agg(
    donors=("donor_id", "nunique"),
    conditions=("disease", "nunique"),
)
assert pair_balance["donors"].eq(2).all()
assert pair_balance["conditions"].eq(2).all()

print(
    {
        "cells": int(matched_gamma_delta.sum()),
        "donors": donor_design["donor_id"].nunique(),
        "matched_pairs": donor_design["pair_index_CW"].nunique(),
    }
)
```

This resulting check now leads too 1,386 cells from 36 donors in 18 matched donor pairs.

## Run the paired donor-level test

For each gene, Scarf first averages normalized expression within each donor. Then, SCARF uses `pair_by` to aligns the RA donor and control donor carrying the same `pair_index_CW` value before the signed-rank test.
The explicit group order fixes the labels as normal and RA, with the test being two-sided.

```{code-cell}
panel = ["IFNG", "IFIT2", "TNF", "GZMA", "ISG15", "S100A4"]
condition = CellField("disease")

paired_result = ds.run_statistical_testing(
    panel,
    condition,
    cell_selection=cells,
    groups=GROUPS,
    test="wilcoxon",
    sample_by="donor_id",
    pair_by="pair_index_CW",
    sample_stat="mean",
    adjustment="fdr_bh",
)

panel_table = pd.concat(
    {gene: paired_result.tables[gene] for gene in panel},
    names=["gene"],
).reset_index(level="gene")
panel_table = panel_table[
    [
        "gene",
        "group_1",
        "group_2",
        "n_pairs",
        "statistic",
        "p_value",
        "p_value_adjusted",
    ]
]

assert panel_table["n_pairs"].eq(18).all()
assert panel_table["p_value_adjusted"].notna().all()
assert not panel_table["p_value_adjusted"].le(0.05).any()
panel_table
```

We also perform a Benjamini-Hochberg correction for multiple hypothesis testing. In a true setting, we'd want our genes to have passed the `p_value_adjusted <= 0.05` threshold.

## Plot the donor-level expression distributions

`distribution` supports the same sample and pairing identity through `StudyDesign`.
The plot below contains donor means, not cell-level observations, and reuses the persisted adjusted p-values for its brackets.
It does not recompute the tests.

```{code-cell}
plot_design = StudyDesign(
    sample_by="donor_id",
    condition_by="disease",
    pair_by="pair_index_CW",
)

ds.plots.distribution(
    panel,
    grouping=condition,
    cell_selection=cells,
    groups=GROUPS,
    study_design=plot_design,
    sample_stat="mean",
    kind="stacked_violin",
    share_y=False,
    max_points=100,
    point_size=2.0,
    point_alpha=0.55,
    stats_results=paired_result,
    stats_show_p=False,
    figsize=(7.0, 12.0),
    title="Matched donor means in gamma-delta T cells",
)
```

Each row runs on its own different scale, so compare normal versus rheumatoid arthritis within a row, not the violin heights between rows. Each point is one donor mean, with 18 donors in each disease group. The `ns` brackets summarize the paired Wilcoxon tests with pooled false-discovery-rate correction: nothing here reaches significance; to understand why, we discuss these further in the caveats section. Note that pairing went into the test only; the plot itself does not join matched donors with lines.

## Important caveats to consider regarding condition comparisons

- **Statistical model disconnect (donor means vs. count models):** Unweighted cell averaging flattens measurement uncertainty, treating a donor with 5 cells identically to one with 120. Unlike parametric pseudobulk GLMs (such as DESeq2 or edgeR) that explicitly model library sizes and negative binomial count dispersion, a Wilcoxon signed-rank test discards magnitude in favor of relative ranks, sacrificing statistical power.
- **Unmodeled confounding and covariates:** While donor pairing controls for matched baseline variables, a paired univariate test cannot model multi-factor covariates (such as batch effects or processing date). Any technical batch divergence across pairs or subtle sub-lineage shifts (e.g., {math}`V\delta1` versus {math}`V\delta2` composition changes) leaks directly into the donor differences.
- **Power constraints vs. biological truth:** Failing to reach adjusted significance ({math}`q \le 0.05`) across six candidate genes at {math}`N = 18` matched pairs reflects the conservative penalty of non-parametric ranking and FDR pooling. A null result on a targeted exploratory panel is an absence of statistical power for small effect sizes, not proof that {math}`\gamma\delta` T cells are transcriptionally identical in vivo.
