---
description: Translate a Seurat workflow to Scarf and import a saved Seurat object.
---

(seurat-users)=
# Scarf for Seurat users

This guide maps familiar Seurat stages to Scarf's store-backed API. Read
{doc}`scanpy_and_seurat` first for the shared execution and artifact model, or start with the
{ref}`Quick start <quickstart>` to inspect a familiar prepared result.

## Workflow map

The methods below are approximate counterparts rather than identical implementations. After
importing an `.rds` file, the Scarf column describes analysis on the resulting Zarr store.

| Goal | Seurat | Scarf |
|---|---|---|
| Hold the analysis | `SeuratObject` | `DataStore` |
| Load a saved project | `readRDS()` | `inspect_seurat`, `SeuratReader`, `SeuratToZarr`, then `DataStore` |
| Work with modalities | Assays such as `RNA` and `ADT` | Assays in the same Zarr store |
| Select and normalize features | `NormalizeData`, `FindVariableFeatures` | `ds.select_hvgs`, then `ds.run_normalization`, using exact refs |
| Scale, reduce, and find neighbours | `ScaleData`, `RunPCA`, `FindNeighbors` | `ds.run_pca` standardizes features by default, followed by Scarf's neighbour-graph methods |
| Embed and cluster | `RunUMAP`, `FindClusters` | `ds.run_umap`, `ds.run_leiden_clustering` |
| Correct batches with Harmony | Harmony integration after PCA | `ds.run_harmony` after PCA, followed by graph construction |
| Integrate modalities with WNN | `FindMultiModalNeighbors` | Build neighbours per assay, then call `ds.integrate_assays(...)`; WNN is the default |
| Select representative cells | Sketching | TopACeDo through {doc}`tutorials/downsampling` |

Scarf WNN follows the published weighting equations but uses the supplied neighbour candidates
rather than Seurat's wider default search. See {doc}`tutorials/multimodal_diagnostics` before
interpreting differences between integration methods. SNN remains available with `method="snn"`.

## Import a saved Seurat object

Scarf imports a saved Seurat object from an `.rds` file. It reads the on-disk RDS document and
does not attach to a live R session. It does not read `.h5seurat`.

Inspect the RDS file, select importable assays and reductions, then write a Zarr store:

```python
import scarf

inspection = scarf.inspect_seurat("pbmc.rds")
with scarf.SeuratReader(
    "pbmc.rds",
    assays=["RNA"],
    reductions=["pca"],
) as reader:
    imported = scarf.SeuratToZarr(reader, zarr_loc="pbmc.zarr").dump()
ds = scarf.DataStore("pbmc.zarr")
imported.activeIdentity, imported.reductionArtifacts["pca"]
```

The importer brings across supported count layers and literal cell metadata. It returns exact
artifact refs for `active.ident` and selected reductions. Neighbour graphs, images, commands, and
most tool slots stay behind. Graphs, clusterings, marker searches, and integrated analyses are
rebuilt in Scarf rather than imported from the RDS object. Scarf does not write `.rds` or
`.h5seurat`.

Typical next steps are `ds.pipeline.run()`, {doc}`tutorials/scrna_seq`, or
{doc}`tutorials/graph_construction`. For multimodal data, build a neighbour artifact per assay
before calling `integrate_assays`. When you only need raw matrices, original 10x HDF5 or Matrix
Market counts are still preferable to an RDS export.

To return to Seurat, write H5AD or Matrix Market from Scarf and convert or import it with the tools
used by your R workflow. See {doc}`tutorials/import_and_export` for the full Seurat import contract
and the other format paths.
