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

Go from a Cell Ranger count matrix to a clustered UMAP with Scarf's default RNA pipeline.
This example uses a public 5K PBMC dataset and writes the analysis to
`scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr`.

Complete the {ref}`installation <installation>` with the `extra` dependencies before you begin.
Run this notebook from the same environment so its kernel imports that Scarf installation.

## Download and convert the counts

```{code-cell} ipython3
import scarf

counts = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]

store = counts.with_name("data.zarr")
reader = scarf.CrH5Reader(str(counts))
```

```{code-cell} ipython3
reader.nCells, reader.nFeatures
```

```{code-cell} ipython3
scarf.CrToZarr(reader, zarr_loc=str(store)).dump()
```

The same reader and writer work with a Cell Ranger H5 file from your own dataset.
Scarf converts the counts to Zarr so later steps can stream data from disk.

## Open the datastore

```{code-cell} ipython3
ds = scarf.DataStore(str(store), nthreads=4)
```

## Run the RNA pipeline

The default pipeline filters cells, scores cell cycle, selects highly variable genes, normalizes
counts, runs PCA, builds a neighbourhood graph, and calculates UMAP. It also runs Leiden at
resolutions 0.5, 0.75, 1.0, and 1.25, plus Paris clustering, doublet scoring, and marker search.
It scores Leiden resolutions in the graph's PCA or Harmony coordinates with a deterministic
silhouette sample, then exposes the selected Leiden candidate as `clusters`. Paris remains
available as `run["paris"]` for diagnosis. This automatic choice is a reproducible baseline, not
biological validation.

```{code-cell} ipython3
run = ds.pipeline.run(label="baseline")
run
```

The return value is a durable {py:class}`~scarf.PipelineRun`. It maps stable result names to exact
immutable {term}`ArtifactRef` values and keeps frozen cell and feature views. The pipeline itself
does not change live `I` or write analytical outputs to metadata.

Inspect the selected labels without copying them into live metadata:

```{code-cell} ipython3
run.cells.to_pandas_dataframe(["clusters"])["clusters"].value_counts().sort_index()
```

Use {doc}`tutorials/graph_construction` when you need stage-by-stage control and explicit refs.

## Consume the frozen result

Plotting stays on `DataStore` and reads only the run's frozen fields:

```{code-cell} ipython3
ds.plots.embedding(run=run, layout="umap", color_by="clusters")
```

Several broad PBMC populations should separate without every group becoming an isolated island.

Marker search used the selected partition. Read its immutable table through the exact ref:

```{code-cell} ipython3
cluster_id = run.cells.fetch("clusters")[0]
ds.get_markers(marker=run["markers"], group_id=cluster_id).head(10)
```

## Reopen a named run

A completed run can be reopened by its immutable label or exact run ID:

```{code-cell} ipython3
assert ds.pipeline.open(label="baseline").run_id == run.run_id
```

Labels are bound to one successful run and cannot be moved to another run. An unlabeled run can be
opened with its `run_id`.

Continue with the complete {doc}`tutorials/scrna_seq` workflow or translate an existing workflow
with {doc}`scanpy_and_seurat`. The {doc}`reference/api/pipeline` documents every pipeline option,
frozen views, and failure reports.
