---
description: Run a default RNA analysis or explore a prepared PBMC result with Scarf.
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

(quickstart)=

# Quick start

Start with the standard RNA pipeline, then explore its clusters and markers. If you do not have
a count file ready, the prepared example below lets you try the plots without running an analysis.

Complete the {ref}`installation <installation>` with the `extra` dependencies first.

## Run an RNA analysis

For a filtered Cell Ranger H5 file, convert the counts to a Scarf store and run the pipeline:

```python
import scarf

# Read the count matrix and its cell and feature identifiers.
reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
# Write the prepared counts and metadata to the new store.
scarf.CrToZarr(reader, zarr_loc="analysis.zarr").dump()

# Open the datastore for the following analysis.
ds = scarf.DataStore("analysis.zarr")
# Run the default RNA analysis and retain its results.
run = ds.pipeline.run(label="baseline")
# Color the new RNA embedding by its selected clusters.
ds.plots.embedding(run=run, color_by="clusters")
```

Use a new or empty output path for conversion. A writer raises `FileExistsError` for a path that
already holds data. Pass `overwrite=True` to replace an earlier conversion that no `DataStore` has
opened; once opened, a store is prepared and is never replaced, so delete it yourself or choose
another path. The pipeline uses Scarf's default settings for
filtering, feature selection, PCA, neighbours, UMAP, clustering, and marker search. It also scores
cell cycle and doublets. These settings are a starting point;
review {doc}`tutorials/quality_control` and the marker evidence before interpreting a new dataset.

`run` holds the results of this analysis. Keep it to make plots and read marker tables without
repeating the computation. Each named run needs a new label; reopen a completed one with
`ds.pipeline.open(label="baseline")`.

## Try a prepared example

Download an already analyzed dataset of about 5,000 blood cells:

```{code-cell} ipython3
import scarf

# Download the prepared example, including its saved analysis.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
```

Open the downloaded store and inspect its saved analysis.

```{code-cell} ipython3
# Open the datastore for the following analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr")
# Open the saved analysis and retain its exact results.
run = ds.pipeline.open(label="docs_default")
# Inspect the store's assays and dimensions.
ds
```

`pipeline.open` reads saved results; `pipeline.run` computes an analysis. Here `docs_default` is
the name of the prepared example. It used dataset-specific filtering, 500 variable genes, and
15 PCs, so its labels and plots need not match a new run with the current defaults.

```{code-cell} ipython3
# Color the prepared PBMC embedding by its saved clusters.
ds.plots.embedding(run=run, color_by="clusters")
```

Each colour marks a cluster of cells with similar RNA profiles. A cluster number is not a cell
type. Continue with {doc}`tutorials/scrna_seq` to compare markers and give the groups biological
names. For another input format, see {doc}`tutorials/import_and_export`; for a familiar workflow
translation, see {doc}`scanpy` or {doc}`seurat`.
