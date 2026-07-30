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

```{code-cell} ipython3
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
```

## Run the RNA pipeline

The default pipeline filters cells, selects highly variable genes, normalizes
counts, builds a neighbourhood graph, and calculates embeddings, clusters,
doublet scores, and marker genes.

```{code-cell} ipython3
artifacts = ds.pipeline.run()
```

## Plot the result

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_leiden_1.0",
)
```

Each colour is a Leiden cluster. The Zarr store now contains the UMAP
coordinates, cluster labels, marker genes, and intermediate results.

## Next steps

- Inspect quality control and tune the analysis for your data:
  {doc}`tutorials/scrna_seq`
- See pipeline options and returned results: {doc}`reference/api/pipeline`
- Translate a Scanpy or Seurat workflow: {doc}`scarf_and_scanpy`
