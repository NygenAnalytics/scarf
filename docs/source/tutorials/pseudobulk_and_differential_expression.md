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

# Pseudobulk and differential expression (DE) primer

To compare gene expression changes across conditions, we need independent biological samples in each group. Pseudobulking works for single-cell RNA sequencing data because summing raw counts within each donor produces donor-level count data that mirrors bulk RNA-seq count data. Testing for differential expression tells us which genes change between conditions, pointing to the programs driving disease or response. Without it, we can describe what genes are present and potentially driving an effect, but not what is different about them across the conditions.

## Dataset and study design

The data come from the CZ CELLxGENE collection [Single-cell RNA-Seq analysis reveals cell subsets and gene signatures associated with rheumatoid arthritis disease activity](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b), published with [Binvignat et al.](https://doi.org/10.1172/jci.insight.178499). The study contains PBMCs from 18 people with rheumatoid arthritis (RA) and 18 matched controls processed across three batches. If you remember correctly, this is the same dataset used in the {doc}`condition_comparisons`

Here each sample is a donor. This tutorial simply goes through the motions of preparing our data for true DE analysis, byt summing the raw counts from γδ T cells into one column per donor, keeping the matched study design, and thus exporting the result for a method such as edgeR or DESeq2. Scarf performs the aggregation and export. It does not fit the differential expression model on this page.

## Prepare the study data

To begin, simply download the dataset and inspect it to derive the existing information.

```{code-cell}
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

After inspection, we see the expected 108,717 cells by 21,648 features (genes). After inspection, we can simply proceed by converting the .h5ad file to its corresponding zarr format.

```{code-cell}
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
                # The default count layout of these 108,717 cells plans for about 5 GB.
                mem_budget="6G",
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

## Select the γδ T-cells

When selecting the cells of interest for pseudobulking, and then downstream DE analysis,  you generally want to keep only cells with a finite matched-pair dentifier so every selected cell can be assigned to the donor design used below.

```{code-cell}
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

This selection simply freezes out cells to ensure that the when we conduct the pseudobulking, the count data and the donor metadata is described on the same cells.

## "Pseudobulk" (sum raw counts by biological donor)

Pseudobulking is simply the process of summing the raw counts by biological donor, thus, we use the `aggr_type="sum"` function to take the raw assay counts and produce one column of counts per `donor_id`.

```{code-cell}
bulk = ds.make_bulk(
    "donor_id",
    cell_selection=selection,
    aggr_type="sum",
    feature_label="name",
)
assert bulk.shape == (13_547, 36)
bulk.iloc[:5, :6]
```

The resulting pseudobulk leaves us with 13,547 expressed features and 36 donor columns. Our features dropped from > 20,000 to ~13000 as this step automatically removes features with zero counts across the selected γδ T cells; the same applies for whatever your selection of cells is.

## Build and verify the donor design

Each donor must have exactly one disease, matched-pair value, and batch within this exact selected population. The design is then aligned to the count-matrix columns before  we export

```{code-cell}
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

The 36 columns are 36 biological replicates, arranged as 18 RA-control pairs. The `batch` describes the cells that actually contributed to each donor column.

## Export raw counts and design metadata

With our counts and our metadata table, simply export the 2 tables for DE analysis offline. These will remain available after the notebook closes. The donor metadata has already been aligned to the count columns.

```{code-cell}
export_directory = Path("pseudobulk_exports")
export_directory.mkdir(exist_ok=True)
counts_csv = export_directory / "yd_t_cell_raw_counts.csv"
metadata_csv = export_directory / "yd_t_cell_donor_design.csv"

bulk.to_csv(counts_csv)
donor_metadata.index.name = "donor_id"
donor_metadata.to_csv(metadata_csv)
print(counts_csv, metadata_csv, sep="\n")
```

Use `bulk` as the raw feature-by-donor count matrix. The external model must use donor-level replication and account for the study design. The linked publication at the top of this notebook allows you find the exact parameters the authors used, and replicate the results yourself.

## Optional: explore a reported γδ T-cell panel

Library-normalized values are useful for a compact descriptive view before modeling. These values show each donor's expression on a common per-million scale, so differences between conditions can be eyeballed before any model is fit. This also adjusts for differences in the amount of cells between groups, as the common scale used is counts per million. Doing this only describes the data, and doesn't test it for any changes.  The figure below converts the donor pseudobulks to log2 counts per million (CPM) only for visualization.

Start with one gene, IFNG, so each line can show one matched pair.

```{code-cell}
panel_genes = ["IFNG", "IFIT2", "TNF", "GZMA", "ISG15", "S100A4"]
missing_genes = sorted(set(panel_genes).difference(bulk.index))
assert not missing_genes, f"Missing panel genes: {missing_genes}"

library_sizes = bulk.sum(axis=0)
assert library_sizes.gt(0).all()
log2_cpm = np.log2(bulk.div(library_sizes, axis=1).mul(1_000_000) + 1)
panel = log2_cpm.loc[panel_genes].T.join(donor_metadata)

paired_values = panel.pivot(index="pair_index_CW", columns="disease", values="IFNG")
paired_values = paired_values[["normal", "rheumatoid arthritis"]]
axis = paired_values.T.plot(marker="o", legend=False, color="0.6", alpha=0.6)
axis.set_xticks([0, 1], ["Control", "RA"])
axis.set_ylabel("log2(CPM + 1)")
axis.set_title("IFNG in matched γδ T-cell pseudobulks")
plt.show()
```

This panel shows donor heterogeneity and paired direction, but it does not estimate dispersion, adjust for batch, fit the matched design, or test a hypothesis. It must not be reported as a differential expression result.

## Pseudo-replicates are not biological replicates

`make_bulk(..., pseudo_reps=2)` randomly divides cells within a donor. Those partitions can support descriptive stability checks, but they come from the same person and do not increase the biological sample size. This tutorial leaves `pseudo_reps` at its default of one. Using `pseudo_reps` is usually a last resort effort when you don't have enough biological donors, thus limit its use.

## Important caveats to consider regarding pseudobulk and differential expression

- **Conflating pseudo-replicates with biological replicates:** Subsetting cells or splitting a donor into random partitions ( with pseudo_reps) does not increase the true biological sample size. Treating non-independent cells or partitions as distinct replicates artificially inflates degrees of freedom, leading to massive false-positive rates in downstream models.
- **Feeding normalized values into count-based models:** Exploratory log2(CPM) values are strictly descriptive and intended for visualization. Differential expression frameworks like DESeq2 and edgeR require raw, unnormalized integer counts to accurately model negative binomial dispersion and compute internal library size factors. *NEVER* feed normalized counts into these models.
- **Omitting matched-pair and batch covariates:** Aggregating donors into two monolithic condition pools or excluding pair_index_CW and batch from the downstream design matrix throws away the statistical power of a matched study and risks confounding disease signatures with technical batch variation.
