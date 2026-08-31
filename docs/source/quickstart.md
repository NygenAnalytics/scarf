---
description: Open a prepared scRNA-seq result and reach a clustered PBMC map with Scarf.
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

Open a prepared 5K PBMC analysis and reach its pipeline-selected Leiden map. This is the shortest
route to a familiar single-cell result; the RNA workflow explains the biological evidence behind
the populations.

Complete the {ref}`installation <installation>` with the `extra` dependencies first.

```{code-cell} ipython3
import scarf

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
```

The named run binds the filtered cells, UMAP, selected Leiden partition, and marker result from one
completed workflow. Plotting reads those frozen outputs directly.

```{code-cell} ipython3
ds.plots.embedding(
    run=run,
    color_by="clusters",
    legend_loc="on_data",
)
```

The map separates several broad PBMC populations. Continue with {doc}`tutorials/scrna_seq` to name
them from marker evidence rather than from UMAP position alone.

## Use your own Cell Ranger counts

The prepared result removes setup time from this first encounter. With your own filtered Cell
Ranger H5 file, the corresponding path is:

```python
reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
scarf.CrToZarr(reader, zarr_loc="analysis.zarr").dump()

ds = scarf.DataStore("analysis.zarr", nthreads=4)
run = ds.pipeline.run(label="baseline")
ds.plots.embedding(run=run, color_by="clusters")
```

The stages match a familiar Scanpy or Seurat workflow: filtering, feature selection, normalization,
PCA, neighbours, UMAP, and Leiden clustering. Use the focused {doc}`scanpy` or {doc}`seurat` guide
when translating an existing analysis. For measured scale evidence rather than a teaching dataset,
see {doc}`concepts/benchmarks`.
