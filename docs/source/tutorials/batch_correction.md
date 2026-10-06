---
description: Compare an uncorrected rheumatoid arthritis PBMC graph with Harmony batch correction.
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

(harmony_batch_correction)=

# Correcting batch effects with Harmony

This tutorial compares one uncorrected RNA analysis with the same analysis after Harmony correction.
It uses the Binvignat rheumatoid arthritis PBMC dataset from the
[CELLxGENE collection](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b)
associated with the [study in JCI Insight](https://insight.jci.org/articles/view/178499).
The tutorial constructs equal cell counts for each sequencing-batch and disease combination, so
technical mixing can be assessed without making disease identical to batch. This sampling is not
a donor-balanced biological design.

We will first choose comparable cells, then run the same pipeline twice, adding
`harmony_batch_columns=["batch"]` to the second run.

The code downloads the
[versioned H5AD file](https://datasets.cellxgene.cziscience.com/3b751975-34bb-409a-a9b7-98380f0450ea.h5ad)
to local disk, converts it to a local Zarr store, and runs every analysis locally. Passing the
download URL to `urlretrieve` transfers the file only. It does not make Scarf compute against a
remote dataset.

## 1. Prepare the local source store

Download the H5AD file only when it is absent. Inspecting it before conversion makes the matrix
choice and dimensions explicit. This tutorial requires the integer-like count matrix at `raw/X`
with 108,717 cells and 21,648 features.

```{code-cell} ipython3
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

import scarf

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="ERROR", progress=False)
```

Choose local paths for the reusable download and converted counts.

```{code-cell} ipython3
# Choose a local folder for reusable input files.
dataset_directory = Path(environ.get("SCARF_DOCS_DATA_DIR", "scarf_datasets"))
# Create the local folder if it does not already exist.
dataset_directory.mkdir(parents=True, exist_ok=True)

# Use the versioned H5AD file from the published collection.
download_url = (
    "https://datasets.cellxgene.cziscience.com/"
    "3b751975-34bb-409a-a9b7-98380f0450ea.h5ad"
)
# Name the local H5AD download.
h5ad_path = dataset_directory / "binvignat_ra_pbmc.h5ad"
# Name the reusable converted count store.
source_store = dataset_directory / "binvignat_ra_pbmc.zarr"

# Reuse completed local work when it is already present.
if not h5ad_path.exists():
    # Use a temporary filename until the download finishes.
    partial_path = h5ad_path.with_suffix(".h5ad.part")
    # Download the complete input file to the temporary filename.
    urlretrieve(download_url, partial_path)
    # Move the completed file or store to its reusable path.
    partial_path.replace(h5ad_path)
```

Check that the file contains the expected counts before converting it:

```{code-cell} ipython3
# Inspect the available count matrix before conversion.
inspection = scarf.inspect_h5ad(str(h5ad_path))
# Verify that the count matrix is stored at raw/X.
assert inspection.matrixKey == "raw/X"
# Verify that the matrix contains integer-like counts.
assert inspection.integerLike is True
# Verify the published cell and gene counts.
assert (inspection.nCells, inspection.nFeatures) == (108_717, 21_648)
# Show the count matrix and dimensions needed for conversion.
{
    "count matrix": inspection.matrixKey,
    "cells": inspection.nCells,
    "genes": inspection.nFeatures,
    "integer counts": inspection.integerLike,
}
```

Convert once and keep the source store for later sessions. This dataset needs a 6 GB import
budget for the default count layout. The temporary directory keeps an interrupted conversion
separate from the finished store.

```{code-cell} ipython3
# Reuse completed local work when it is already present.
if not source_store.exists():
    # Stage conversion in a temporary folder before publishing the completed store.
    with TemporaryDirectory(dir=dataset_directory) as conversion_directory:
        # Convert into a temporary store until every write succeeds.
        staged_store = Path(conversion_directory) / source_store.name
        # Read the count matrix and its cell and feature identifiers.
        reader = scarf.H5adReader.from_inspect(inspection)
        # Ensure the input reader is closed even if conversion fails.
        try:
            # Write the prepared counts and metadata to the new store.
            scarf.H5adToZarr(
                reader,
                zarr_loc=str(staged_store),
                nthreads=4,
                # The default count layout of these 108,717 cells plans for about 5 GB.
                mem_budget="6G",
            ).dump()
        finally:
            # Close the H5AD reader after conversion.
            reader.h5.close()
        # Move the completed file or store to its reusable path.
        staged_store.replace(source_store)
```

Initialize the source with `min_features_per_cell=0`, which drops only cells without detected
features (none here), so the imported cell axis stays intact while the tutorial defines its own
exact selection.

```{code-cell} ipython3
# Open the source without dropping cells before selecting the cohort.
source = scarf.DataStore(
    str(source_store), default_assay="RNA", min_features_per_cell=0, nthreads=4
)
# Check the initialized count matrix dimensions.
{"cells": source.cells.N, "genes": source.RNA.feats.N}
```

Mount the source into a temporary writable analysis store. Count matrices remain in the local
source store, while the selection and pipeline artifacts are written below the temporary
directory. Keep `analysis_directory` bound for as long as the mounted datastore is in use.

```{code-cell} ipython3
# Keep new analysis results in a temporary working folder.
analysis_directory = TemporaryDirectory()
# Choose the path for the writable working store.
analysis_store = Path(analysis_directory.name) / "ra_batch_demo.zarr"

# Open the datastore for the following analysis.
ds = scarf.mount_datastore(
    str(source_store),
    at=str(analysis_store),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
# Check that mounting preserved the source dimensions.
{
    "same cell count": ds.cells.N == source.cells.N,
    "same gene count": ds.RNA.feats.N == source.RNA.feats.N,
}
```

## 2. Create a balanced 9,000-cell analysis

There are three batches and two disease groups. Select 1,500 cells from every batch and disease
combination with one seeded generator. The resulting `docs_ra_batch_demo` column contains exactly
9,000 cells and is reproducible from the same source file.

```{code-cell} ipython3
# Read the sequencing-batch label for each cell.
batch = ds.cells.fetch_all("batch").astype(str)
# Read each cell's biological condition.
disease = ds.cells.fetch_all("disease").astype(str)
# List the distinct sequencing batches.
batch_values = np.unique(batch)
# List the distinct biological conditions.
disease_values = np.unique(disease)

# Require the expected three sequencing batches.
assert batch_values.size == 3
# Require the expected two biological conditions.
assert disease_values.size == 2

# Compare the cell counts or fractions across the selected groups.
pd.crosstab(batch, disease, rownames=["batch"], colnames=["condition"])
```

Draw the same number of cells from each batch and disease group.

```{code-cell} ipython3
# Seed the generator so this example is reproducible.
rng = np.random.default_rng(42)
# Start with no cells selected.
selected = np.zeros(ds.cells.N, dtype=bool)

# Sample every sequencing batch with the same disease balance.
for batch_value in batch_values:
    # Choose an equal-sized sample from this disease group.
    for disease_value in disease_values:
        # Find cells in this batch and disease combination.
        candidates = np.flatnonzero((batch == batch_value) & (disease == disease_value))
        # Require enough cells to draw 1,500 without replacement.
        assert candidates.size >= 1_500
        # Draw the same number of cells from this group.
        chosen = rng.choice(candidates, size=1_500, replace=False)
        # Add the sampled cells to the shared selection.
        selected[chosen] = True

# Check that balanced sampling retained exactly 9,000 cells.
assert selected.sum() == 9_000
# Save the values in cell metadata using the stated selection.
ds.cells.insert(column_name="docs_ra_batch_demo", values=selected, overwrite=True)

# Compare the cell counts or fractions across the selected groups.
pd.crosstab(
    batch[selected], disease[selected], rownames=["batch"], colnames=["condition"]
)
```

The constructed subset has equal cell counts for every batch and disease combination, but donor
representation is not balanced. Disease is a biological condition rather than a correction
covariate, so only `batch` is supplied to Harmony below.

```{raw} html
<span id="harmony"></span>
<span id="harmony-batch-correction"></span>
```

## 3. Run matched uncorrected and Harmony pipelines

Both runs use the same cells and the default feature count, PCA dimensions, and neighbour count.
The only analysis difference is `harmony_batch_columns=["batch"]` in the second run.
We turn off filtering because we already chose the cells, and skip clustering, cell-cycle scores,
doublet detection, and markers because this comparison only needs the graph and UMAP.

```{code-cell} ipython3
# Keep the cells and analysis settings identical in both runs.
pipeline_options = {
    "cell_key": "docs_ra_batch_demo",
    "filtering": False,
    "leiden": False,
    "cell_cycle": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    "snapshot_columns": ("batch", "disease", "rough_annot"),
}
```

Run the uncorrected analysis with these shared settings.

```{code-cell} ipython3
# Run the baseline without batch correction.
uncorrected = ds.pipeline.run(label="ra_uncorrected", **pipeline_options)
# Check the cell count used by the uncorrected run.
{"analysis cells": int(uncorrected.cells.fetch_all("I").sum())}
```

Repeat the analysis with batch correction.

```{code-cell} ipython3
# Run the same analysis with sequencing batch supplied to Harmony.
harmony = ds.pipeline.run(
    label="ra_harmony", harmony_batch_columns=["batch"], **pipeline_options
)
# Check that the corrected run uses the same cell count.
{"analysis cells": int(harmony.cells.fetch_all("I").sum())}
```

Harmony adjusts PCA coordinates before the neighbour graph is built. It does not change the count
matrix. The two frozen runs keep the exact selections, metadata, layouts, and graph artifacts used
for the comparison.

## 4. Compare the layouts

Plot each run by sequencing batch and by the imported broad annotation. Look for better mixing
of batches while the broad cell types remain distinct.

```{code-cell} ipython3
# Inspect batch separation before correction.
ds.plots.embedding(run=uncorrected, color_by="batch")
# Inspect cell-type structure before correction.
ds.plots.embedding(run=uncorrected, color_by="rough_annot")
```

The Harmony result:

```{code-cell} ipython3
# Inspect batch mixing after Harmony correction.
ds.plots.embedding(run=harmony, color_by="batch")
# Check whether broad cell-type structure remains after correction.
ds.plots.embedding(run=harmony, color_by="rough_annot")
```

(lisi_metrics)=
(integration_metrics)=

## 5. Quantify mixing and structural preservation

iLISI measures local mixing of `batch`. cLISI measures local separation of `rough_annot`, and graph
connectivity measures whether cells sharing that annotation remain connected. Scarf scales all
three metrics so higher values are better. Use the exact neighbour and connectivity artifacts from
each run. The LISI metrics choose their neighbourhood scale from the neighbour count by default.

```{code-cell} ipython3
# Measure batch mixing and annotation preservation for one run.
def integration_diagnostics(run):
    return {
        "iLISI (batch)": ds.metric_ilisi(
            batch_colname="batch", neighbors=run["neighbors"]
        ),
        "cLISI (rough_annot)": ds.metric_clisi(
            annotation_column="rough_annot", neighbors=run["neighbors"]
        ),
        "graph connectivity (rough_annot)": ds.metric_graph_connectivity(
            annotation_column="rough_annot", graph=run["connectivity_map"]
        ),
    }
```

Calculate the same diagnostics for both completed runs.

```{code-cell} ipython3
# Collect the mixing and preservation metrics for both runs.
score_frame = pd.DataFrame(
    {
        "Uncorrected": integration_diagnostics(uncorrected),
        "Harmony": integration_diagnostics(harmony),
    }
).T
# Compare mixing and preservation metrics for the two runs.
score_frame.round(3)
```

Read these scores together. Higher iLISI after Harmony, with similar cLISI and graph connectivity,
would support improved technical mixing without obvious loss of broad cell-type structure. These metrics
are diagnostics, not proof that correction is biologically valid or that every disease-associated
signal was preserved. The subset equalizes cell counts, not biological replicates, and disease was
not used as a correction covariate. Biological conclusions still require a donor-aware design and
targeted downstream checks.

See {doc}`../reference/api/graph_construction` for the PCA and Harmony contracts, and
{doc}`../reference/api/integration` for the metric definitions.
