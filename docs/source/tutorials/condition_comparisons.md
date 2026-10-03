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
# Compare biological conditions across donors

Do matched rheumatoid arthritis (RA) and control donors differ in gamma-delta T-cell
expression of six selected genes? We will average expression within each donor and
compare the 18 matched donor pairs with a Wilcoxon signed-rank test.

The donor is the biological replicate. Treating all 1,386 cells as independent replicates
would give a misleading sample size, a problem called **pseudoreplication**. This example
compares donor means of normalized expression. For a raw-count pseudobulk model, see
{doc}`pseudobulk_and_differential_expression`.

## Prepare the source data

The dataset comes from [Binvignat et al.](https://doi.org/10.1172/jci.insight.178499) and its
[CELLxGENE collection](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b).
Unlike the prepared PBMC tutorials, this page starts from the full H5AD file. The first
run needs network access, several GiB of memory, and space for both the download and a
converted Zarr store. Later runs reuse those files.

The preparation below handles the large source once. The statistical workflow starts
at **Select the matched cells**. For background on conversion, see {doc}`import_and_export`.

### Download and inspect

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

dataset_directory = Path(environ.get("SCARF_DOCS_DATA_DIR", "scarf_datasets"))
dataset_directory.mkdir(parents=True, exist_ok=True)
h5ad_path = dataset_directory / "binvignat_ra_pbmc.h5ad"
source_store = dataset_directory / "binvignat_ra_pbmc.zarr"
```

Download to a temporary filename so an interrupted download is not mistaken for a
complete H5AD file.

```{code-cell}
if not h5ad_path.exists():
    partial_path = h5ad_path.with_suffix(".h5ad.part")
    urlretrieve(
        "https://datasets.cellxgene.cziscience.com/3b751975-34bb-409a-a9b7-98380f0450ea.h5ad",
        partial_path,
    )
    partial_path.replace(h5ad_path)

inspection = scarf.inspect_h5ad(str(h5ad_path))
assert inspection.matrixKey == "raw/X"
assert inspection.integerLike is True
assert (inspection.nCells, inspection.nFeatures) == (108_717, 21_648)
inspection.matrixKey, inspection.nCells, inspection.nFeatures
```

This file stores counts in `raw/X`. Inspection identifies that matrix before conversion,
so we do not accidentally analyze the other expression matrix in `X`.

### Convert and open a working store

Conversion uses a 6 GiB budget because the planned count layout needs about 5 GB. The
finished store is moved into place only after conversion succeeds.

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
                mem_budget="6G",
            ).dump()
        finally:
            reader.h5.close()
        staged_store.replace(source_store)
```

Keep the published cohort without an extra feature-count filter. Mounting puts the new
selection and test results in a temporary working store while leaving the count matrix
in its reusable source store.

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

## Select the matched cells

The publication calls gamma-delta T cells `yd T cells` in `fine_annot`.
`pair_index_CW` identifies matched RA-control pairs. Keep cells with that annotation
and a recorded pair, then save the selection for both the test and plot.

```{code-cell}
GROUPS = ["normal", "rheumatoid arthritis"]
fine_annotation = np.asarray(ds.cells.fetch_all("fine_annot"), dtype=object)
pair_index = np.asarray(ds.cells.fetch_all("pair_index_CW"), dtype=float)
matched_gamma_delta = np.isfinite(pair_index) & (fine_annotation == "yd T cells")
ds.cells.insert("matched_yd_t_cells", matched_gamma_delta, overwrite=True)
cells = ds.snapshot_cell_selection("matched_yd_t_cells")
```

Before testing, check that each donor belongs to one condition and one pair. Each pair
must contain one donor from each condition.

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
assert len(donor_design) == donor_design["donor_id"].nunique() == 36
assert donor_design["pair_index_CW"].nunique() == 18
assert set(donor_design["disease"]) == set(GROUPS)
```

```{code-cell}
pair_balance = donor_design.groupby("pair_index_CW", observed=True).agg(
    donors=("donor_id", "nunique"),
    conditions=("disease", "nunique"),
)
assert pair_balance["donors"].eq(2).all()
assert pair_balance["conditions"].eq(2).all()
donor_design.groupby("disease", observed=True).size().rename("donors")
```

We have 1,386 cells from 36 donors in 18 matched pairs. Those 18 pairs, not the number
of cells, determine the replication for the test.

## Test a small gene panel

`StudyDesign` records which column identifies donors, conditions, and matched pairs.
We use the same design again for the plot. Scarf averages normalized expression within
donors by default, then aligns the pairs for a two-sided signed-rank test.

```{code-cell}
panel = ["IFNG", "IFIT2", "TNF", "GZMA", "ISG15", "S100A4"]
condition = CellField("disease")
design = StudyDesign(
    sample_by="donor_id",
    condition_by="disease",
    pair_by="pair_index_CW",
)
paired_result = ds.run_statistical_testing(
    panel,
    condition,
    cell_selection=cells,
    groups=GROUPS,
    study_design=design,
    test="wilcoxon",
)
```

The default Benjamini-Hochberg correction covers all six tests together. Read the
adjusted p-values alongside the donor distributions, rather than looking only for a
threshold crossing.

```{code-cell}
panel_table = pd.concat(
    {gene: paired_result.tables[gene] for gene in panel},
    names=["gene"],
).reset_index(level="gene")
panel_table[["gene", "n_pairs", "statistic", "p_value", "p_value_adjusted"]]
```

## View the donor distributions

Each plotted point is a donor mean. The plot reads the saved test results for its
brackets, so the displayed statistics come from the same comparison. Give the six rows
enough space and enlarge the points so individual donors remain visible.

```{code-cell}
ds.plots.distribution(
    panel,
    grouping=condition,
    cell_selection=cells,
    groups=GROUPS,
    study_design=design,
    kind="stacked_violin",
    figsize=(7, 12),
    point_size=2.0,
    point_alpha=0.55,
    share_y=False,
    stats_results=paired_result,
    stats_show_p=False,
    title="Matched donor means in gamma-delta T cells",
)
```

Compare RA and control **within each gene's row** because the rows have separate value
scales. The plot does not join matched donors with lines, although pairing is used in
the test. None of these six comparisons passes an adjusted p-value threshold of 0.05
in this example.

## Interpret the scope of the result

This donor-mean test does not reproduce the paper's count-model pseudobulk analysis.
The approaches summarize expression differently and make different statistical
assumptions. A nonsignificant result here does not establish that the conditions are
identical, nor does it tell us by itself why a difference was not detected.

- Donors with few selected cells may have less precise means. This simple test does
  not model that uncertainty separately.
- Matching does not remove all confounding. One selected pair spans batches 2 and 3,
  and this test has no batch term to separate that contribution.
- This is a six-gene exploratory panel. Do not generalize its result to every gene or
  gamma-delta T-cell state.

For count-model analysis with a sample design, continue to
{doc}`pseudobulk_and_differential_expression`.
