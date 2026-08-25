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
This example uses a public 5K PBMC dataset and writes the analysis to `scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr`.

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
```

```{code-cell} ipython3
reader.nCells, reader.nFeatures
```

```{code-cell} ipython3
scarf.CrToZarr(
    reader,
    zarr_loc=str(store),
).dump()
```

The same reader and writer work with a Cell Ranger H5 file from your own dataset.
Scarf converts the counts to Zarr so later steps can stream data from disk.
Scarf normally uses INFO logging with progress enabled.
Read the completed bars in this cached page as a record of the work performed during execution.
Log level and progress are independent; batch runs can use `scarf.configure_output(progress=False, timestamps=True)`.
See the {doc}`reference/api/utilities` for file logging and the full output contract.

## Open the datastore

```{code-cell} ipython3
ds = scarf.DataStore(
    str(store),
    nthreads=4,
)
```

```{code-cell} ipython3
print(f"Active cells: {int(ds.cells.fetch_all('I').sum())} / {ds.cells.N}")
ds.cells.to_pandas_dataframe(
    columns=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
).describe().loc[["min", "50%", "max"]]
```

## Run the RNA pipeline

The default pipeline filters cells, scores cell cycle, selects highly variable genes, normalizes counts, runs PCA, builds a neighbourhood graph, and calculates UMAP.
It also runs Leiden at resolutions 0.5, 0.75, 1.0, and 1.25, plus Paris clustering.
The partition with the highest PCA silhouette is copied to `RNA_clusters` and used for doublet scoring and marker search unless you choose one explicitly.

```{code-cell} ipython3
n_before = int(ds.cells.fetch_all("I").sum())
artifacts = ds.pipeline.run()
n_after = int(ds.cells.fetch_all("I").sum())
print(f"Active cells before pipeline: {n_before}")
print(f"Active cells after pipeline: {n_after}")

selected = next(
    key
    for key, ref in artifacts.items()
    if key != "selected_clusters" and ref == artifacts["selected_clusters"]
)
print(f"RNA_clusters selected from: {selected}")
```

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=["RNA_clusters"],
    key="I",
)["RNA_clusters"].value_counts().sort_index()
```

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=["RNA_leiden_0.5"],
    key="I",
)["RNA_leiden_0.5"].value_counts().sort_index()
```

The return value maps each result name to an {term}`ArtifactRef`: a handle on a stored result that names it without loading it.
Every result the pipeline wrote is an {term}`artifact` in the Zarr store, saved together with the {term}`provenance` record of what produced it.
`artifacts["highly_variable_features"]` is the exact immutable feature selection passed to normalization; the plain `hvgs` label is its published convenience name.
`RNA_leiden_0.5` is one of the Leiden partitions kept alongside the selected `RNA_clusters` labels.

```{code-cell} ipython3
sorted(artifacts)
```

Most optional stages accept `False`.
For example, cell-cycle scoring, UMAP, Paris, doublet scoring, and marker search can be disabled separately; pass an empty `leiden` mapping to skip Leiden.
Highly variable feature selection remains required.
Use {doc}`tutorials/graph_construction` for stage-by-stage control and {doc}`tutorials/clustering` for choosing a partition.

## Plot the result

Colour the embedding by `RNA_clusters` to see the partition the pipeline chose.
The resolution-specific columns, such as `RNA_leiden_0.5`, stay available for comparison.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_clusters",
)
```

Several broad PBMC populations should separate without every group becoming an isolated island.
Per-cluster library size flags any tiny low-count group that should be revisited in quality control before assigning a cell type:

```{code-cell} ipython3
(
    ds.cells.to_pandas_dataframe(
        columns=["RNA_clusters", "RNA_nCounts"],
        key="I",
    )
    .groupby("RNA_clusters")["RNA_nCounts"]
    .agg(n_cells="size", median_nCounts="median")
    .sort_values("median_nCounts")
)
```

Marker search ran on the same partition, so its table is already in the store:

```{code-cell} ipython3
ds.plots.marker_heatmap(
    marker=artifacts["markers"],
    group_key="RNA_clusters",
    topn=5,
    figsize=(5, 9),
)
```

Each column is a cluster and each row one of its top-scoring genes.
The clean block structure is the signal that the partition tracks real populations.
Inspect one cluster's ranked markers directly:

```{code-cell} ipython3
cluster_id = (
    ds.cells.to_pandas_dataframe(columns=["RNA_clusters"], key="I")["RNA_clusters"]
    .value_counts()
    .index[0]
)
print(f"Markers for cluster {cluster_id}")
ds.get_markers(
    marker=artifacts["markers"],
    group_key="RNA_clusters",
    group_id=cluster_id,
).head(10)
```

The Zarr store now holds the UMAP coordinates, cluster labels, marker tables, and every intermediate result.

Continue with the complete {doc}`tutorials/scrna_seq` workflow or translate an existing workflow with {doc}`scanpy_and_seurat`.
The {doc}`reference/api/pipeline` documents every option, returned artifact, and the callback contract for advanced automation.
