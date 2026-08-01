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

Complete the {ref}`installation <installation>` with the `extra` dependencies before you begin.
Run this notebook from the same environment so its kernel imports that Scarf installation.

## Download and convert the counts

```{code-cell} ipython3
import scarf

scarf.configure_output(level="ERROR", progress=True)

counts = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]

store = counts.with_name("data.zarr")
reader = scarf.CrH5Reader(str(counts))
scarf.CrToZarr(
    reader,
    zarr_loc=str(store),
).dump()
```

The same reader and writer work with a Cell Ranger H5 file from your own dataset.
Scarf converts the counts to Zarr so later steps can stream data from disk.
Scarf normally uses INFO logging with progress enabled. Read the completed bars
in this cached page as a record of the work performed during execution. Log level
and progress are independent; batch runs can use
`scarf.configure_output(progress=False, timestamps=True)`. See the
{doc}`reference/api/utilities` for file logging and the full output contract.

```{code-cell} ipython3
ds = scarf.DataStore(
    str(store),
    nthreads=4,
)
```

## Run the RNA pipeline

The default pipeline filters cells, scores cell cycle, selects highly variable
genes, normalizes counts, runs PCA, builds a neighbourhood graph, and calculates
UMAP. It also runs Leiden at resolutions 0.5, 0.75, 1.0, and 1.25, plus Paris
clustering. The partition with the highest PCA silhouette is copied to
`RNA_clusters` and used for doublet scoring and marker search unless you choose
one explicitly.

```{code-cell} ipython3
artifacts = ds.pipeline.run()
```

The return value maps each result name to an {term}`ArtifactRef`: a handle on a
stored result that names it without loading it. Every result the pipeline wrote
is an {term}`artifact` in the Zarr store, saved together with the
{term}`provenance` record of what produced it.

```{code-cell} ipython3
sorted(artifacts)
```

Most optional stages accept `False`. For example, cell-cycle scoring, UMAP,
Paris, doublet scoring, and marker search can be disabled separately; pass an
empty `leiden` mapping to skip Leiden. Highly variable feature selection remains
required. Use {doc}`tutorials/custom_graph_construction` for stage-by-stage
control and {doc}`tutorials/clustering` for choosing a partition.

## Plot the result

Colour the embedding by `RNA_clusters` to see the partition the pipeline chose.
The resolution-specific columns, such as `RNA_leiden_0.5`, stay available for
comparison.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_clusters",
)
```

Several broad PBMC populations should separate without every group becoming an
isolated island. A tiny group dominated by low-count cells is a reason to
revisit quality control before assigning a cell type.

Marker search ran on the same partition, so its table is already in the store:

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key="RNA_clusters",
    topn=5,
    figsize=(5, 9),
)
```

Each column is a cluster and each row one of its top-scoring genes. The clean
block structure is the signal that the partition tracks real populations, and
recognisable PBMC markers name most of them: `CD14` and `FPR1` for monocytes,
`CD8A` and `GZMK` for cytotoxic T cells, `KLRF1` and `FGFBP2` for NK cells,
`TCL1A` and `IGHD` for naive B cells, `IGHG1` and `IGHA1` for plasma cells,
`PPBP` and `GNG11` for platelets, `LILRA4` and `IL3RA` for plasmacytoid
dendritic cells. Read one cluster's full table with
`ds.get_markers(group_key="RNA_clusters", group_id=...)`.

The Zarr store now holds the UMAP coordinates, cluster labels, marker tables,
and every intermediate result.

Continue with the complete {doc}`tutorials/scrna_seq` workflow or translate an
existing workflow with {doc}`scarf_and_scanpy`. The
{doc}`reference/api/pipeline` documents every option, returned artifact, and the
callback contract for advanced automation.
