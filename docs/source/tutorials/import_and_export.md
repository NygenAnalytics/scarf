---
description: Inspect and convert common count formats, then export complete or selected data.
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

# Import and export

Scarf reads common single-cell count formats, writes a Zarr store for analysis, and exports counts or metadata to interoperable formats.
Most imports follow the same pattern: inspect when the source layout can vary, open a reader, then call a matching `*ToZarr` writer.

| Source | Inspect | Reader | Writer |
|---|---|---|---|
| 10x HDF5 | (inferred by reader) | `CrH5Reader` | `CrToZarr` |
| Matrix Market / MEX | `inspect_mtx` | `MtxReader` | `MtxToZarr` |
| AnnData H5AD | `inspect_h5ad` | `H5adReader` | `H5adToZarr` |
| Seurat RDS | `inspect_seurat` | `SeuratReader` | `SeuratToZarr` |
| Dense CSV | | `CSVReader` | `CSVtoZarr` |
| SciPy CSR | | | `SparseToZarr` |

Export paths write Matrix Market or H5AD.
Scarf does not write Seurat `.rds` or `.h5seurat` files.
See {doc}`../scanpy` or {doc}`../seurat` for ecosystem-specific workflow mapping.

Start with the Matrix Market example below. The later sections cover other input formats and
show how to keep supplied analysis results or handle a larger import.

## Prerequisites

- Scarf installed with the optional dependencies required by the source format
- A source count matrix in a supported format

## What you will learn

- Download datasets from the `scarf_docs` Cytebase catalog
- Convert 10x HDF5, MTX, H5AD, Seurat RDS, CSV, and sparse inputs to Zarr
- Export an assay to MTX or H5AD

## 1. Download example datasets

Scarf hosts example datasets in the public [Cytebase bucket](https://huggingface.co/buckets/Nygen/cytebase) in formats such as MTX, 10x HDF5, and H5AD.
Connect to the `scarf_docs` repository to download them:

```{code-cell} ipython3
# Manage local file and directory paths.
from pathlib import Path
# Create temporary directories for this example.
from tempfile import TemporaryDirectory

# Work with numeric arrays and cell masks.
import numpy as np
# Summarize inspected files and matrices in tables.
import pandas as pd
# Open count stores and run Scarf analyses.
import scarf

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="ERROR", progress=False)
# Connect to the public example-data repository.
datasets = scarf.cytebase.connect("scarf_docs")
# Keep this tutorial's temporary files together.
outputs = TemporaryDirectory()
# Use the temporary directory for downloads and conversions.
output_dir = Path(outputs.name)
# Show the local path used by this step.
output_dir
```

**Naming format**: `<author>_<number of cells>_<cell/tissue type or species>_<single-cell method>`

Each download returns the directory it wrote, which the readers below use as their input path.

```{code-cell} ipython3
# Download the Matrix Market example.
mtx_dir = datasets.download_dataset(
    name="xin_1K_pancreas_rnaseq",
    destination=output_dir,
)
# Show the local path used by this step.
mtx_dir
```

This tutorial writes downloads and converted stores below one temporary directory. Replace
`output_dir` with a persistent project directory in your own workflow.

## 2. Import Matrix Market

Inspect a Matrix Market source before selecting a triplet.
A source can be an `.mtx` or `.mtx.gz` file, a directory, or a direct MEX ZIP.
Inspection recognizes canonical 10x names, common prefixed triplets, and Parse DGE directories.
It returns every complete candidate instead of choosing between alternatives such as raw and filtered matrices.

```{code-cell} ipython3
# Find complete matrix, feature, and cell-file triplets.
candidates = scarf.inspect_mtx(str(mtx_dir))
# Compare the candidate files and dimensions without repeating the download path.
pd.DataFrame(
    {
        index: {
            "matrix file": str(Path(candidate.matrixPath).relative_to(mtx_dir)),
            "feature file": str(Path(candidate.featurePath).relative_to(mtx_dir)),
            "cell file": str(Path(candidate.cellPath).relative_to(mtx_dir)),
            "orientation": candidate.matrixOrientation,
            "cells": candidate.nCells,
            "features": candidate.nFeatures,
            "stored entries": candidate.nEntries,
        }
        for index, candidate in enumerate(candidates)
    }
).rename_axis(columns="candidate")
```

This directory contains one complete matrix. Open its reader, choose an output path, and call
`dump()` to write the Scarf store. If your inspection finds several candidates, choose the one
whose cells and counts you want to analyse:

```{code-cell} ipython3
# Choose where to write the converted Matrix Market store.
mtx_store = output_dir / "xin_1K.zarr"
# Open a reader for the selected source format.
reader = scarf.MtxReader(candidates[0])
# Write the converted count store.
scarf.MtxToZarr(reader, zarr_loc=str(mtx_store)).dump()
# Show the path of the converted Matrix Market store.
mtx_store
```

Open the store and inspect its cells and features:

```{code-cell} ipython3
# Open the converted Matrix Market store.
ds_mtx = scarf.DataStore(str(mtx_store))
# Inspect the opened store's cells and features.
ds_mtx
```

The store is ready for analysis. Continue with {doc}`scrna_seq` for an RNA workflow, or read on
if you need another input format.

### 2.1 Parse DGE directories

Parse DGE matrices use cells by genes orientation and require `cell_metadata.csv`.
Scarf imports its non-ID columns.
It recognizes `bc_wells` and `bc_index`; pass `cell_id_key` when both are present:

```python
# Select the Matrix Market triplet in the Parse directory.
candidate = scarf.inspect_mtx("/path/to/parse_dge")[0]
# Open a reader for the selected source format.
reader = scarf.MtxReader(candidate, cell_id_key="bc_index")
# Write the converted count store.
scarf.MtxToZarr(reader, zarr_loc="parse.zarr").dump()
```

## 3. Import H5AD

H5AD files vary in where they store counts, feature names, metadata, and layers.
Inspect the file before conversion rather than assuming `X`, `obs`, and `var` contain the intended values.

```{code-cell} ipython3
# Download the pancreas H5AD example.
h5ad_dir = datasets.download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination=output_dir,
)
# Locate the H5AD file to inspect.
h5ad_path = str(h5ad_dir / "data.h5ad")
# Inspect the available count matrices and metadata.
inspection = scarf.inspect_h5ad(h5ad_path)
# Review the source matrices, dimensions, and metadata.
pd.Series(
    {
        "selected matrix": inspection.matrixKey,
        "available matrices": ", ".join(inspection.matrixCandidates),
        "encoding": inspection.matrixEncoding,
        "integer-like counts": inspection.integerLike,
        "cells": inspection.nCells,
        "features": inspection.nFeatures,
        "cell metadata": inspection.cellAttrsKey,
        "feature metadata": inspection.featureAttrsKey,
        "feature names": inspection.featureNameKey,
    },
    name="value",
).to_frame()
```

`H5adReader.from_inspect` uses the discovered matrix and metadata keys.
Override the inspection only after confirming that another layer contains the raw count matrix required by the analysis.
This file also contains a UMAP and cell-type labels. Here we ask the reader to keep them with
`embedding_roles` and `cluster_keys`, so we can plot and export them later. Omit these two
arguments for a counts-and-metadata import.

```{code-cell} ipython3
# Open a reader for the selected source format.
reader = scarf.H5adReader.from_inspect(
    inspection,
    embedding_roles={"X_umap": "umap"},
    cluster_keys=("clusters",),
)

# Choose the output path for the pancreas store.
pancreas_store = output_dir / "differentiating_pancreatic_cells.zarr"
# Import counts and the explicitly selected analysis results.
h5ad_import = scarf.H5adToZarr(
    reader,
    zarr_loc=str(pancreas_store),
    analysis_assay="RNA",
).dump()
# Inspect the imported UMAP and cluster references.
{
    "embeddings": dict(h5ad_import.embeddingArtifacts),
    "clusters": dict(h5ad_import.clusterArtifacts),
}
```

`embedding_roles` and `cluster_keys` select analytical values for artifact import.
The result maps their source names to exact refs in `embeddingArtifacts` and `clusterArtifacts`.
These values are not flattened into live cell metadata. Load them through the datastore or pass
their refs directly to consumers. Sparse, non-numeric, or row-mismatched selected embeddings are
rejected.

For a file with several assay types, pass `assay_split_key` to the writer to split them;
`from_inspect` alone writes one assay. The {doc}`../reference/api/import_export` reference covers
multi-assay imports and metadata handling.

## 4. Import a larger 10x HDF5 dataset

Scarf stores data as dense, compressed chunks in Zarr.
`CrH5Reader` and `CrToZarr` convert Cell Ranger HDF5 into that layout.
Assay type is inferred from the H5 feature types (RNA, ATAC, or multimodal).
`mem_budget` bounds the memory the conversion plans for.
The default count shards of this 89,796-peak assay do not fit `mem_budget="8G"`, so the import
passes the smaller `policy` that its `MemoryError` names. Non-RNA assays write no `countsT`
copy, so the smaller shards do not slow later `countsT` reads.

```{code-cell} ipython3
# Choose count shards that fit the import memory budget.
from scarf.storage.count_matrix import CountMatrixPolicy

# Download the 10x ATAC count file.
tenx_h5 = datasets.download_dataset(
    name="tenx_10K_pbmc-v1_atacseq",
    destination=output_dir,
)
# Choose the output path for the ATAC store.
atac_store = output_dir / "pbmc_atac.zarr"
# Open a reader for the selected source format.
reader = scarf.CrH5Reader(str(tenx_h5 / "data.h5"))
# Write the converted count store.
scarf.CrToZarr(
    reader,
    zarr_loc=str(atac_store),
    mem_budget="8G",
    policy=CountMatrixPolicy(unitBytes=500_000_000, chunkBytes=50_000_000),
).dump()
# Show the path of the converted ATAC store.
atac_store
```

Open the written store.
The summary lists an ATAC assay, which confirms that inference from the H5 feature types survived the dump:

```{code-cell} ipython3
# Open the converted ATAC store.
ds_atac = scarf.DataStore(str(atac_store))
# Inspect the opened store's cells and features.
ds_atac
```

## 5. Import Seurat RDS

Scarf can import a serialized Seurat object from an `.rds` file through `inspect_seurat`, `SeuratReader`, and `SeuratToZarr`.
This path reads the on-disk RDS document.
It does not attach to a live R session, and it does not read `.h5seurat`.

Inspect first.
The result reports which assays and reductions are importable, their dimensions, and any blocking diagnostics or notices:

```python
# Open count stores and run Scarf analyses.
import scarf

# Inspect the available count matrices and metadata.
inspection = scarf.inspect_seurat("pbmc.rds")
# Compare the active assay with the importable assays and reductions.
{
    "active assay": inspection.activeAssay,
    "importable assays": [assay.name for assay in inspection.assays if assay.importable],
    "importable reductions": [r.name for r in inspection.reductions if r.importable],
}
```

Open a reader for the assays and reductions you want, then write the Zarr store.
Omitting `reductions` selects every available reduction; `SeuratToZarr` raises if any selected reduction is not importable.
Pass an empty sequence to skip reductions, or pass only importable names to import a subset.
`SeuratToZarr` prepares and reads each selected count layer once when it is created, so the
integral counts that Seurat holds as R doubles are stored unsigned, and each assay gets its own
count dtype and layout:

```python
# Open the requested Seurat assays and reductions.
with scarf.SeuratReader(
    "pbmc.rds",
    assays=["RNA"],
    reductions=["pca"],
) as reader:
    # Import the selected Seurat assays and reductions.
    result = scarf.SeuratToZarr(reader, zarr_loc="pbmc_from_seurat.zarr").dump()

# Open the count store for this analysis.
ds = scarf.DataStore("pbmc_from_seurat.zarr")
# Inspect the imported assays, identities, and PCA result.
{
    "assays": result.assayNames,
    "default assay": result.defaultAssay,
    "notices": result.notices,
    "identities": result.activeIdentity,
    "PCA": result.reductionArtifacts["pca"],
}
```

`activeIdentity` is an exact cluster-label artifact. `reductionArtifacts` maps each requested
Seurat reduction name to its exact imported-coordinate ref. Neither result is installed as a live
analysis column. Pass the reduction ref to graph construction or load either payload explicitly.

What this import covers:

- Legacy `Assay`, `Assay5`, and `ChromatinAssay` count layers when their matrix layout is supported
- Literal cell metadata, plus artifact refs for `active.ident` and selected reductions such as PCA
  or LSI
- Partial Assay5 cell membership as per-assay boolean columns (`<assay>_I`) when the selected count
  layers hold only some cells

What it does not import as analysis artifacts:

- Neighbour graphs, Seurat `neighbors` objects, images, commands, and most `tools` slots
- Normalized layers when the selected count layer is used for the Scarf assay
- Transposed Assay5 storage (`Assay5T`)
- A return path to `.rds` or `.h5seurat` (export H5AD or MTX instead)

Pass `assay_layers` when an assay stores several count layers and you need a non-default choice.
A cell that only an unselected layer holds is recorded as not measured by that assay.
Pass `sidecar_path_remaps` when a `SaveSeuratRds` sidecar cache points at moved on-disk matrices.
Prefer original 10x HDF5 or Matrix Market counts when they are available and you only need raw matrices.

`SeuratReader` and `inspect_seurat` hold the stitching indexes of each `Assay5` in memory and
charge them to `maxMetadataBytes` of their `matrix_limits`, 256 MiB by default. The indexes cover
the feature and cell positions of every selected count layer and the assay's cell membership,
which adds about 36 budget bytes per cell that earlier releases did not charge. An assay over the
limit is reported as not importable with the blocking diagnostic `metadata_index_limit`, and
`SeuratToZarr` raises that `SeuratImportError`, so an import that fit close to the limit before can
now fail. Pass `matrix_limits` with a larger `maxMetadataBytes`, or select fewer count layers with
`assay_layers`.

## 6. Export to Matrix Market

Open the H5AD-derived store written in section 3 and load the selected analytical artifacts:

```{code-cell} ipython3
# Open the count store for this analysis.
ds = scarf.DataStore(str(pancreas_store))

# Load the UMAP coordinates kept during H5AD import.
imported_umap = np.asarray(
    ds.load_artifact(h5ad_import.embeddingArtifacts["X_umap"])["values"][:]
)
# Load the cluster labels kept during H5AD import.
imported_clusters = np.asarray(
    ds.load_artifact(h5ad_import.clusterArtifacts["clusters"])["values"][:]
)
# Check the imported coordinate and label dimensions.
imported_umap.shape, imported_clusters.shape
```

Plot the imported layout colored by the imported cluster labels:

```{code-cell} ipython3
# Plot the imported UMAP using the cluster labels kept from H5AD.
ds.plots.embedding(
    layout=h5ad_import.embeddingArtifacts["X_umap"],
    color_by=h5ad_import.clusterArtifacts["clusters"],
)
```

```{code-cell} ipython3
# Choose a directory for the exported Matrix Market files.
mtx_export = output_dir / "diff_pancreas"
# Export the assay counts to Matrix Market files.
scarf.writers.to_mtx(assay=ds.RNA, mtx_directory=str(mtx_export))
# List the exported files with their sizes in bytes.
[(path.name, path.stat().st_size) for path in sorted(mtx_export.iterdir())]
```

## 7. Export to H5AD and AnnData

`DataStore.to_anndata` returns an in-memory AnnData object with counts, cell and feature metadata, and optional assay layers.
Artifact results are attached explicitly when another tool expects a particular AnnData slot.

For a completed pipeline, prefer `ds.to_anndata(run=run)`. That form exports the run's frozen
assay, cell selection, feature selection, and metadata. It rejects live `cell_key`,
`feature_indexes`, and `feature_names` overrides so the exported axes cannot drift from the run.

### 7.1 Attach exact imported artifacts

```{code-cell} ipython3
# Build an AnnData object from the assay.
adata = ds.to_anndata(from_assay="RNA")
# Attach the imported UMAP in AnnData's embedding slot.
adata.obsm["X_umap"] = imported_umap
# Attach the imported cluster labels to AnnData's cell table.
adata.obs["clusters"] = imported_clusters
# Choose the exported H5AD filename.
h5ad_export = output_dir / "diff_pancreas.h5ad"
# Write counts, metadata, and attached results to H5AD.
adata.write_h5ad(h5ad_export)
# Confirm the written filename and its size in bytes.
{"file": h5ad_export.name, "bytes": h5ad_export.stat().st_size}
```

Reload the H5AD and confirm that the explicitly attached layout is in `obsm`:

```{code-cell} ipython3
# Reload the exported H5AD with AnnData.
import anndata as ad

# Reload the exported H5AD to verify its contents.
adata = ad.read_h5ad(h5ad_export)
# Verify the exported embedding name and dimensions.
sorted(adata.obsm.keys()), adata.obsm["X_umap"].shape
```

### 7.2 Export a feature panel with `to_anndata`

Full-assay export can require enough memory and disk for the selected cell by feature matrix.
When only a marker panel is needed, select features before materializing AnnData.
Resolve the requested display names against the store once, then export only those columns:

```{code-cell} ipython3
# Read the feature names available in the store.
all_names = ds.RNA.feats.fetch_all("names").astype(str)
# Match requested genes without depending on letter case.
name_lookup = {name.upper(): name for name in all_names}
# Resolve the small marker panel against the stored gene names.
panel = [name_lookup[gene] for gene in ("GCG", "SST", "KRT19")]
# Export only the requested features into memory.
selected = ds.to_anndata(from_assay="RNA", matrix="raw", feature_names=panel)
# Check the exported dimensions and gene order.
{
    "shape": selected.shape,
    "genes": selected.var["names"].tolist(),
}
```

Use `feature_indexes` instead when stable feature rows are already available.
`feature_names` and `feature_indexes` are mutually exclusive, preserve the requested order, and reject duplicate or unknown selections.

The panel export contains live metadata only. Attach an exact artifact payload in memory when the
receiving tool needs that result.

## 8. Import CSV

`CSVReader` and `CSVtoZarr` provide small-data compatibility for dense CSV count matrices.
The toy matrix below is synthesized in-notebook so the conversion does not depend on a catalog file.
Rows are cells and columns are features; `cell_data_cols` moves selected columns into cell metadata.

```{code-cell} ipython3
# Choose a filename for the small CSV example.
csv_path = output_dir / "toy_counts.csv"
# Write the small example count table.
csv_path.write_text(
    "quality,geneA,geneB,geneC\n"
    "10,1,0,2\n"
    "20,0,3,0\n"
    "30,4,5,6\n"
    "40,7,0,8\n"
    "50,9,10,0\n",
    encoding="utf-8",
)

# Inspect the example CSV before conversion.
print(csv_path.read_text(encoding="utf-8"))
```

Convert the CSV, keeping `quality` as cell metadata:

```{code-cell} ipython3
# Choose where to write the converted CSV store.
csv_zarr = output_dir / "toy_csv.zarr"
# Open a reader for the selected source format.
reader = scarf.CSVReader(
    str(csv_path),
    cell_data_cols=["quality"],
)
# Write the converted count store.
scarf.CSVtoZarr(reader, zarr_loc=str(csv_zarr), assay_name="RNA").dump()
# Open the converted CSV store.
ds_csv = scarf.DataStore(str(csv_zarr))
# Inspect the first few rows of the result.
ds_csv.cells.head()
```

`quality` is cell metadata rather than a count column, which is what `cell_data_cols` is for.

## 9. Import sparse matrices

`SparseToZarr` accepts a SciPy CSR matrix with matching cell and feature IDs.

```{code-cell} ipython3
# Construct a small sparse count matrix.
from scipy.sparse import csr_matrix

# Construct a small sparse cell-by-feature count matrix.
mat = csr_matrix(
    (
        [1, 10, 15, 10, 20, 2, 3, 1, 5],
        ([0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 3, 8, 2, 3, 1, 2, 8, 9]),
    ),
    shape=(3, 10),
)
# Display the three-cell toy matrix before conversion.
mat.toarray()
```

Write this small matrix with matching cell and feature identifiers:

```{code-cell} ipython3
# Choose where to write the sparse-matrix example.
sparse_zarr = output_dir / "toy_sparse.zarr"
# Write the converted count store.
scarf.SparseToZarr(
    mat,
    zarr_loc=str(sparse_zarr),
    cell_ids=[f"cell_{i}" for i in range(mat.shape[0])],
    feature_ids=[f"feat_{i}" for i in range(mat.shape[1])],
    assay_name="RNA",
).dump()
# Open the converted sparse-matrix store.
ds_sparse = scarf.DataStore(str(sparse_zarr))
# Inspect the opened store's cells and features.
ds_sparse
```

For a complete `DataStoreMerge` example, continue to {doc}`dataset_merging`.

## 10. Other import paths

### 10.1 Chunked arrays

`chunked_to_zarr` writes from a Scarf `ChunkedArray` when lazy out-of-core conversion is needed.

### 10.2 Remote Zarr destinations

Writers also accept remote Zarr locations.
Choose the `cloud` profile for an object-store destination and pass credentials through the environment or runtime configuration:

```python
# Read object-store credentials from the environment.
import os

# Inspect the available count matrices and metadata.
inspection = scarf.inspect_h5ad("counts.h5ad")
# Open a reader for the selected source format.
reader = scarf.H5adReader.from_inspect(inspection)
# Configure a writer for the remote destination.
writer = scarf.H5adToZarr(
    reader,
    zarr_loc="s3://my-bucket/project/data.zarr",
    storage_options={
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    },
    profile="cloud",
)
# Write the converted count store.
writer.dump()
```

### 10.3 Count storage and memory

Writers choose a count dtype from the values without changing them. Integral non-negative
counts use an unsigned integer dtype that fits; fractional or negative values keep their source
dtype. NaN and infinite counts are rejected before the store is created.

RNA writers store both cell-major `counts` and gene-major `countsT` arrays. The latter supports
feature selection and marker search; non-RNA assays store `counts` only. New stores start with
all feature rows available, and analytical feature selections are saved separately.

Start with the default storage layout. If it cannot fit the import's `mem_budget`, the error
names a smaller `CountMatrixPolicy` that fits, as used in the ATAC example above. Prefer a larger
memory budget when your machine allows it: smaller RNA shards slow later gene-major reads.
See {doc}`../concepts/memory_and_execution` for memory planning and
{doc}`../reference/api/import_export` for reader buffering and storage options.

## Common mistakes and limitations

- Fetching a prepared Zarr store when the aim is to demonstrate source-format conversion
- Reusing an existing Zarr output path: a writer raises `FileExistsError` for a path that holds
  data. `overwrite=True` replaces only an earlier conversion that no `DataStore` has opened; a
  prepared store, one that a `DataStore` has opened, and other files are never replaced, so delete
  them yourself or choose another path
- Writing a store inside another store, such as `data.zarr/RNA` or `data.zarr/new.zarr`: writers
  raise `ValueError`, so give every store its own directory outside other stores
- Passing a count dtype to a reader or writer; the stored dtype follows the values
- Exporting normalized values when a downstream method requires raw counts
- Expecting an older RNA Zarr store without `countsT` to open in the current Scarf version
- Assuming an H5AD file uses `X` for raw counts without inspecting its layers
- Omitting `embedding_roles` or `cluster_keys` when analytical H5AD values should become artifacts
- Selecting sparse, non-numeric, or row-mismatched `obsm` arrays as embeddings
- Materializing a full AnnData object when a feature panel would answer the export question
- Treating Seurat neighbour graphs, images, or normalized layers as imported Scarf artifacts
- Expecting Scarf to read `.h5seurat` or write Seurat `.rds` files

Conversion writes the requested Zarr target, and export commands write MTX or H5AD at the supplied destination.
