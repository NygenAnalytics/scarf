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

This workflow asks one narrow question: do matched rheumatoid arthritis (RA) and control donors differ in gamma-delta T-cell expression of six selected genes?
The donor is the unit of inference.
Cells are averaged within donors, and a paired Wilcoxon signed-rank test compares the 18 matched RA-control pairs.
This avoids treating 1,386 cells as independent biological replicates.

The result is a sample-level distribution test on normalized expression.
It is not a raw-count pseudobulk differential expression model.

## Dataset and prerequisites

The data come from the [CELLxGENE collection](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b) for the [Binvignat et al. paper](https://doi.org/10.1172/jci.insight.178499).
This page pins the versioned CELLxGENE H5AD rather than depending on a mutable collection download.
You need enough local disk space for both the H5AD and its converted Zarr store, plus network access on the first run.

The dataset URL below is used only to download a file.
Scarf does not compute against the URL: inspection reads the local H5AD, and every analysis step reads a local Zarr count source.

## 1. Download, inspect, and mount the count source

Download the H5AD if it is absent, then inspect it before conversion.
The assertions pin the matrix choice and dimensions used for this analysis.
`raw/X` contains integer-like counts, while `X` is also present as another matrix candidate.

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3
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

## 2. Freeze the matched gamma-delta T-cell cohort

The H5AD records the paper's fine annotation as `fine_annot` and its matched case-control index as `pair_index_CW`.
Keep only cells labeled exactly `yd T cells` with a finite pair index, then freeze that mask as an immutable selection.

```{code-cell} ipython3
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

Check the inferential structure, not just the cell count.
Each donor must map to one disease group and one pair, and every pair must contain one donor from each group.

```{code-cell} ipython3
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

This selection contains 1,386 cells from 36 donors in 18 matched pairs.

## 3. Run the paired donor-level test

For each gene, Scarf first averages normalized expression within each donor.
`pair_by` then aligns the RA donor and control donor carrying the same `pair_index_CW` value before the signed-rank test.
The explicit group order fixes the labels as normal and RA; the test remains two-sided.

```{code-cell} ipython3
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

The Benjamini-Hochberg correction is pooled across all six tests.
In the validated run, none of these genes passed the `p_value_adjusted <= 0.05` threshold.

## 4. Plot the donor-level distributions

`distribution` supports the same sample and pairing identity through `StudyDesign`.
The plot below contains donor means, not cell-level observations, and reuses the persisted adjusted p-values for its brackets.
It does not recompute the tests.

```{code-cell} ipython3
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
    figsize=(7.0, 9.0),
    title="Matched donor means in gamma-delta T cells",
)
```

Each row has its own value scale, so compare the two disease distributions within a gene, not violin heights across genes.
Each point is one donor mean, with 18 donors in each disease group.
The `ns` brackets reflect the paired tests and their pooled false-discovery-rate correction.
Pairing affects the test, but this plot does not connect matched donors with lines.

## Interpretation and limits

This simple paired Wilcoxon panel did not survive pooled false-discovery-rate correction.
It also did not reproduce the paper's count-model pseudobulk result for gamma-delta T cells.
That is not a contradiction: this workflow tests six donor-mean normalized-expression distributions with a signed-rank test, while a count model uses raw sample-level counts and models their mean-variance relationship.

Matching does not remove every processing effect. One selected pair spans batches 2 and 3, and
this paired Wilcoxon test has no batch term, so a batch contribution cannot be separated here.

Do not generalize this null six-gene panel to the full transcriptome, and do not treat it as a reanalysis of every paper contrast or covariate.
For a count-model analysis, export raw counts aggregated by biological sample and carry the design into DESeq2, edgeR, or another suitable method; see {doc}`pseudobulk_and_differential_expression`.
