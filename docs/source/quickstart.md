---
description: Minimal scRNA-seq workflow in Scarf from count matrix to UMAP and clustering.
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

Go from a Cell Ranger count matrix to a clustered UMAP with Scarf's default RNA
pipeline. This example uses a public 5K PBMC dataset and writes the analysis to
`scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr`.

Install Scarf with the plotting dependencies before you begin:

```bash
uv pip install "scarf[extra]"
```

## Download and convert the counts

```{code-cell} ipython3
import scarf

scarf.configure_output(level="ERROR", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
)

reader = scarf.CrH5Reader(f"{dataset}/data.h5")
scarf.CrToZarr(
    reader,
    zarr_loc=f"{dataset}/data.zarr",
).dump(batch_size=1000)
```

The same reader and writer work with a Cell Ranger H5 file from your own dataset.
Scarf converts the counts to Zarr so later steps can stream data from disk.
Scarf normally uses INFO logging with progress enabled. This page suppresses both
to keep its cached output compact. Log level and progress are independent; batch
runs can use `scarf.configure_output(progress=False, timestamps=True)`. See the
{doc}`reference/api/utilities` for file logging and the full output contract.

```{code-cell} ipython3
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
```

## Run the RNA pipeline

The default pipeline filters cells, scores cell cycle, selects highly variable
genes, normalizes counts, runs PCA, builds a neighbourhood graph, and calculates
UMAP. It also runs Leiden at resolutions 0.5, 0.75, 1.0, and 1.25, plus Paris
clustering. The partition with the highest PCA silhouette is used for doublet
scoring and marker search unless you choose one explicitly.

```{code-cell} ipython3
artifacts = ds.pipeline.run()
```

The return value maps result names to `ArtifactRef` objects, which identify the
persisted outputs and their provenance:

```{code-cell} ipython3
sorted(artifacts)
```

Most optional stages accept `False`. For example, cell-cycle scoring, UMAP,
Paris, doublet scoring, and marker search can be disabled separately; pass an
empty `leiden` mapping to skip Leiden. Highly variable feature selection remains
required. Use {doc}`tutorials/custom_graph_construction` for stage-by-stage
control and {doc}`tutorials/clustering` for choosing a partition.

## Plot the result

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_leiden_1.0",
)
```

Each colour is a Leiden cluster. The Zarr store now contains the UMAP
coordinates, cluster labels, marker genes, and intermediate results. Several
broad PBMC populations should separate without every group becoming an isolated
island. A tiny group dominated by low-count cells is a reason to revisit quality
control before assigning a cell type.

Continue with the complete {doc}`tutorials/scrna_seq` workflow or translate an
existing workflow with {doc}`scarf_and_scanpy`. The
{doc}`reference/api/pipeline` documents every option, returned artifact, and the
callback contract for advanced automation.
