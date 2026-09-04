---
description: Aggregate raw counts by donor and export a matched rheumatoid arthritis design for external differential expression.
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

The inferential unit in condition-level differential expression is the biological donor, not the
cell. This tutorial sums raw counts from γδ T cells into one column per donor, keeps the matched
study design, and exports the result for a replicate-aware method such as edgeR or DESeq2.

Scarf performs the aggregation and export. It does not fit the differential expression model on
this page.

## Dataset and study design

The data come from the CZ CELLxGENE collection
[Single-cell RNA-Seq analysis reveals cell subsets and gene signatures associated with rheumatoid
arthritis disease activity](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b),
published with [Binvignat et al.](https://doi.org/10.1172/jci.insight.178499). The study contains
PBMCs from 18 people with rheumatoid arthritis (RA) and 18 matched controls processed across three
batches.

This page downloads the
[versioned H5AD file](https://datasets.cellxgene.cziscience.com/3b751975-34bb-409a-a9b7-98380f0450ea.h5ad),
converts it to a local Zarr store, and mounts that local store into a temporary local analysis
target. It is a download workflow, not remote analysis.

## 1. Download and inspect the raw-count matrix

```{code-cell} ipython3
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset_url = (
    "https://datasets.cellxgene.cziscience.com/"
    "3b751975-34bb-409a-a9b7-98380f0450ea.h5ad"
)
dataset_directory = Path(environ.get("SCARF_DOCS_DATA_DIR", "scarf_datasets"))
dataset_directory.mkdir(parents=True, exist_ok=True)
h5ad_path = dataset_directory / "binvignat_ra_pbmc.h5ad"

if not h5ad_path.exists():
    partial_path = h5ad_path.with_suffix(".h5ad.part")
    urlretrieve(dataset_url, partial_path)
    partial_path.replace(h5ad_path)

inspection = scarf.inspect_h5ad(str(h5ad_path))
assert inspection.matrixKey == "raw/X"
assert inspection.integerLike is True
assert (inspection.nCells, inspection.nFeatures) == (108_717, 21_648)
{
    "matrix": inspection.matrixKey,
    "encoding": inspection.matrixEncoding,
    "integer-like": inspection.integerLike,
    "shape": (inspection.nCells, inspection.nFeatures),
}
```

The assertions guard the two properties required for pseudobulk: the selected matrix is the raw
count matrix, and the pinned file has the expected 108,717 cells by 21,648 features. A normalized
`X` matrix would not be an interchangeable input to a count model.

## 2. Import once and mount a writable analysis store

Convert the H5AD only when its reusable source store is absent. Initializing with
`min_features_per_cell=0` retains every cell already curated in the published file. The temporary
mount receives metadata and new selection artifacts while the count matrices remain in the local
source store.

```{code-cell} ipython3
source_store = dataset_directory / "binvignat_ra_pbmc.zarr"
if not source_store.exists():
    with TemporaryDirectory(dir=dataset_directory) as conversion_directory:
        staged_store = Path(conversion_directory) / source_store.name
        reader = scarf.H5adReader.from_inspect(inspection)
        try:
            scarf.H5adToZarr(
                reader,
                zarr_loc=str(staged_store),
                nthreads=4,
            ).dump()
        finally:
            reader.h5.close()
        staged_store.replace(source_store)

source = scarf.DataStore(
    str(source_store),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
assert int(np.asarray(source.cells.fetch_all("I"), dtype=bool).sum()) == 108_717

analysis_directory = TemporaryDirectory()
ds = scarf.mount_datastore(
    str(source_store),
    at=str(Path(analysis_directory.name) / "ra_pseudobulk.zarr"),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
```

## 3. Freeze the paired γδ T-cell selection

The CELLxGENE metadata names this subset `yd T cells`. Keep only cells with a finite matched-pair
identifier so every selected cell can be assigned to the donor design used below.

```{code-cell} ipython3
cell_metadata = pd.DataFrame(
    {
        column: ds.cells.fetch_all(column)
        for column in (
            "donor_id",
            "disease",
            "batch",
            "pair_index_CW",
            "fine_annot",
        )
    }
)
cell_metadata["pair_index_CW"] = pd.to_numeric(
    cell_metadata["pair_index_CW"],
    errors="coerce",
)

selection_mask = pd.Series(
    np.isfinite(cell_metadata["pair_index_CW"].to_numpy(dtype=float)),
    index=cell_metadata.index,
) & cell_metadata["fine_annot"].eq("yd T cells")
ds.cells.insert(
    "paired_yd_t_cells",
    selection_mask.to_numpy(),
    overwrite=True,
)
selection = ds.snapshot_cell_selection("paired_yd_t_cells")

pd.Series(
    {
        "selected cells": int(selection_mask.sum()),
        "represented donors": int(
            cell_metadata.loc[selection_mask, "donor_id"].nunique()
        ),
    }
)
```

The selection is frozen before aggregation so the count columns and donor metadata share one exact
cell population. For the control donor sequenced in multiple batches, only its batch-1 cells carry
the finite matched-pair identifier and enter this selection. Its other technical repeats are not
silently reassigned to that pair.

## 4. Sum raw counts by biological donor

`aggr_type="sum"` streams raw assay counts and produces one column per `donor_id`. Empty features
are removed by the default `remove_empty_features=True` behavior.

```{code-cell} ipython3
bulk = ds.make_bulk(
    "donor_id",
    cell_selection=selection,
    aggr_type="sum",
    feature_label="name",
)
assert bulk.shape == (13_547, 36)
bulk.iloc[:5, :6]
```

The validated local run produced 13,547 expressed features by 36 donors and took about 147 seconds.
That duration is a local observation, not a hardware-independent benchmark.

## 5. Build and verify the donor design

Each donor must have exactly one disease, matched-pair value, and batch within this exact selected
population. The design is then aligned to the count-matrix columns before export.

```{code-cell} ipython3
selected_metadata = cell_metadata.loc[
    selection_mask,
    ["donor_id", "disease", "batch", "pair_index_CW"],
].copy()

within_donor_levels = selected_metadata.groupby("donor_id", sort=False)[
    ["disease", "pair_index_CW", "batch"]
].nunique(dropna=False)
assert within_donor_levels.eq(1).all().all()

donor_metadata = (
    selected_metadata[["donor_id", "disease", "pair_index_CW", "batch"]]
    .drop_duplicates()
    .set_index("donor_id")
)
donor_metadata = donor_metadata.reindex(bulk.columns)
assert donor_metadata.index.is_unique
assert donor_metadata.notna().all().all()

disease_counts = donor_metadata["disease"].value_counts()
assert disease_counts.to_dict() == {
    "normal": 18,
    "rheumatoid arthritis": 18,
}

pair_sizes = donor_metadata.groupby("pair_index_CW").size()
pair_conditions = donor_metadata.groupby("pair_index_CW")["disease"].nunique()
assert len(pair_sizes) == 18
assert pair_sizes.eq(2).all()
assert pair_conditions.eq(2).all()

donor_metadata.groupby(["batch", "disease"]).size().unstack(fill_value=0)
```

The 36 columns are 36 biological replicates, arranged as 18 RA-control pairs. `batch` describes
the cells that actually contributed to each donor column. It does not reattach excluded technical
repeats from elsewhere in the source H5AD.

## 6. Explore a reported γδ T-cell panel

Library-normalized values are useful for a compact descriptive view before modeling. The figure
below converts the donor pseudobulks to log2 counts per million (CPM) only for visualization. Each
grey segment connects one matched pair.

```{code-cell} ipython3
panel_genes = ["IFNG", "IFIT2", "TNF", "GZMA", "ISG15", "S100A4"]
missing_genes = sorted(set(panel_genes).difference(bulk.index))
assert not missing_genes, f"Missing panel genes: {missing_genes}"

library_sizes = bulk.sum(axis=0)
assert library_sizes.gt(0).all()
log2_cpm = np.log2(bulk.div(library_sizes, axis=1).mul(1_000_000) + 1)
panel = log2_cpm.loc[panel_genes].T.join(donor_metadata)

condition_order = ("normal", "rheumatoid arthritis")
condition_colors = {
    "normal": "#4C78A8",
    "rheumatoid arthritis": "#E45756",
}
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex=True)
for gene, ax in zip(panel_genes, axes.flat, strict=True):
    for _, paired in panel.groupby("pair_index_CW", sort=True):
        ordered = paired.set_index("disease").reindex(condition_order)
        ax.plot(
            (0, 1),
            ordered[gene],
            color="0.78",
            linewidth=0.8,
            zorder=1,
        )
    for position, condition in enumerate(condition_order):
        values = panel.loc[panel["disease"].eq(condition), gene]
        ax.scatter(
            np.full(len(values), position),
            values,
            color=condition_colors[condition],
            s=20,
            zorder=2,
        )
    ax.set_title(gene)
    ax.set_xticks((0, 1), ("Control", "RA"))
    ax.set_ylabel("log2(CPM + 1)")

fig.suptitle("Matched-donor γδ T-cell pseudobulk expression", y=1.02)
fig.tight_layout()
fig
```

This panel shows donor heterogeneity and paired direction, but it does not estimate dispersion,
adjust for batch, fit the matched design, or test a hypothesis. It must not be reported as a
differential expression result.

The paper applied pseudobulk modeling across 18 PBMC subsets and reported 168 differentially
expressed genes in total. Its γδ T-cell result included downregulation of IFNG, IFIT2, TNF, GZMA,
ISG15, and S100A4 in RA. This page displays those genes for orientation but does not reproduce the
paper's replicate-aware model or its significance claims.

## 7. Export raw counts and design metadata

```{code-cell} ipython3
export_directory = TemporaryDirectory()
counts_csv = Path(export_directory.name) / "yd_t_cell_raw_counts.csv"
metadata_csv = Path(export_directory.name) / "yd_t_cell_donor_design.csv"

bulk.to_csv(counts_csv)
donor_metadata.index.name = "donor_id"
donor_metadata.to_csv(metadata_csv)

pd.read_csv(metadata_csv, index_col="donor_id").head()
```

Use `bulk` as the raw feature-by-donor count matrix. Do not give the log2 CPM panel to edgeR or
DESeq2 as count input. The external model must use donor-level replication, estimate the disease
contrast, and use a model whose terms are identifiable for the chosen design. Both the matched-pair
identifier and the observed processing batch are exported, but they should not be added blindly as
fixed effects: the full intercept-plus-disease-plus-pair-plus-batch design is rank deficient for
this selected cohort. A paired contrast and the paper's batch-adjusted model answer related but
distinct questions. The Binvignat paper used DESeq2 with a likelihood-ratio test corrected for
batch; reproducing that analysis requires its exact sample definition, aggregation, model,
filtering, and multiple-testing choices.

## Pseudo-replicates are not biological replicates

`make_bulk(..., pseudo_reps=2)` randomly divides cells within a donor. Those partitions can support
descriptive stability checks, but they come from the same person and do not increase the biological
sample size. This tutorial leaves `pseudo_reps` at its default of one.

## Common mistakes

- Aggregating all RA cells and all control cells into only two columns
- Treating cells or random within-donor splits as independent biological replicates
- Fitting a count model to the library-normalized plotting values
- Ignoring the matched-pair or processing-batch metadata
- Reporting the exploratory panel as a Scarf differential expression result
