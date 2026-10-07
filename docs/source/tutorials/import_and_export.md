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

Scarf reads common single-cell count formats, writes a Zarr store for analysis, and exports counts or metadata to interoperable formats that you can run the analysis on.
Most imports follow the same general pattern.

Importing data from other tools or modalities follows the general pattern and required function calls discussed below.


| Source              | Inspect              | Reader         | Writer         |
| --------------------- | ---------------------- | ---------------- | ---------------- |
| 10x HDF5            | (inferred by reader) | `CrH5Reader`   | `CrToZarr`     |
| Matrix Market / MEX | `inspect_mtx`        | `MtxReader`    | `MtxToZarr`    |
| AnnData H5AD        | `inspect_h5ad`       | `H5adReader`   | `H5adToZarr`   |
| Seurat RDS          | `inspect_seurat`     | `SeuratReader` | `SeuratToZarr` |
| Dense CSV           |                      | `CSVReader`    | `CSVtoZarr`    |
| SciPy CSR           |                      |                | `SparseToZarr` |

Export paths write matrix or H5AD file; Scarf does not write Seurat `.rds` or `.h5seurat` files. If you want to learn more about the ecosystem-specific workflow mapping for Scanpy or Seurat, see {doc}`../scanpy` or {doc}`../seurat` .

The tutorial begins with how to import data from the matrix format, and then later covers other input formats, alongside how to handle a larger import of data.

## Handling Common Import Formats

## Download the example datasets

Scarf hosts example datasets in the public [Cytebase bucket](https://huggingface.co/buckets/Nygen/cytebase) in formats such as MTX, 10x HDF5, and H5AD for learning purposes; connect to the `scarf_docs` repository to download them:

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scarf

scarf.configure_output(level="ERROR", progress=False)
datasets = scarf.cytebase.connect("scarf_docs")
outputs = TemporaryDirectory()
output_dir = Path(outputs.name)
output_dir
```

**Naming format**: `<author>_<number of cells>_<cell/tissue type or species>_<single-cell method>`

```{code-cell} ipython3
mtx_dir = datasets.download_dataset(
    name="xin_1K_pancreas_rnaseq",
    destination=output_dir,
)
mtx_dir
```

This tutorial writes downloads and converted stores below one temporary directory. Replace
`output_dir` with a solid project directory in your own workflow that won't be meddled with.

## Import Matrix Market files

Inspect a Matrix Market file before selecting it for its input. A source can be an `.mtx` or `.mtx.gz` file, a directory, or a direct MEX ZIP. Inspection recognizes canonical 10x names, common prefixed triplets, and Parse DGE directories. It returns every complete candidate instead of choosing between alternatives such as raw and filtered matrices, so when a source ships both you see each triplet with its cell and feature counts and pick the one to convert yourself.

```{code-cell} ipython3
candidates = scarf.inspect_mtx(str(mtx_dir))
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
`dump()` to write the Scarf store. If your inspection finds several candidates, choose the one whose cells and counts you want to analyze:

```{code-cell} ipython3
mtx_store = output_dir / "xin_1K.zarr"
reader = scarf.MtxReader(candidates[0])
scarf.MtxToZarr(reader, zarr_loc=str(mtx_store)).dump()
mtx_store
```

Open the store and inspect its cells and features:

```{code-cell} ipython3
ds_mtx = scarf.DataStore(str(mtx_store))
ds_mtx
```

The store is ready for analysis. Continue with {doc}`scrna_seq` for an RNA workflow, or read on
if you need another input format.

### Parse DGE directories

Parse DGE matrices use cells by genes orientation and require `cell_metadata.csv` that matches with the matrix. Scarf imports its non-ID columns, and it recognizes `bc_wells` and `bc_index`. Pass `cell_id_key` when both columns are present:

```python
candidate = scarf.inspect_mtx("/path/to/parse_dge")[0]
reader = scarf.MtxReader(candidate, cell_id_key="bc_index")
scarf.MtxToZarr(reader, zarr_loc="parse.zarr").dump()
```

## Import H5AD

H5AD files vary in where they store counts, feature names, metadata, and layers. They are also more common for downloadable analysis; due to this variability as to how the data can be stored, make sure to inspect the file before conversion rather than assuming `X`, `obs`, and `var` contain the intended values:

```{code-cell} ipython3
h5ad_dir = datasets.download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination=output_dir,
)
h5ad_path = str(h5ad_dir / "data.h5ad")
inspection = scarf.inspect_h5ad(h5ad_path)
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
Override the inspection only after confirming that another layer contains the raw count matrix that you need for the analysis. This example file also contains a UMAP and cell-type labels, which not all datasets will have. Here, we ask the reader to keep them with `embedding_roles` and `cluster_keys`, so we can plot and export them later. Omit these two arguments if you just want to input the count information and the metadata.

```{code-cell} ipython3
reader = scarf.H5adReader.from_inspect(
    inspection,
    embedding_roles={"X_umap": "umap"},
    cluster_keys=("clusters",),
)

pancreas_store = output_dir / "differentiating_pancreatic_cells.zarr"
h5ad_import = scarf.H5adToZarr(
    reader,
    zarr_loc=str(pancreas_store),
    analysis_assay="RNA",
).dump()
{
    "embeddings": dict(h5ad_import.embeddingArtifacts),
    "clusters": dict(h5ad_import.clusterArtifacts),
}
```

`embedding_roles` and `cluster_keys` pick which analytical values travel with the import, so the UMAP and cluster labels you name are carried over as artifacts. The result maps each source name to its exact reference in `embeddingArtifacts` and `clusterArtifacts`, which you can load through the datastore or hand directly to downstream consumers. These values are never flattened into live cell metadata, thus they stay separate from the active columns and cannot drift with later edits. Anything that does not fit an embedding slot is rejected up front, so sparse, non-numeric, or row-mismatched selections never become artifacts.

For a file with several assay types, pass `assay_split_key` to the writer to split them up during conversion; `from_inspect` alone writes one assay. The {doc}`../reference/api/import_export` reference covers multi-assay imports and metadata handling.

## Import a larger 10x HDF5 dataset

`CrH5Reader` and `CrToZarr` convert Cell Ranger HDF5 files into the Zarr format required for Scarf.
Assay type is inferred from the H5 feature types (RNA, ATAC, or multimodal), thus we don't need to select. If you need to set a memory budget, `mem_budget` bounds the memory used during the conversion. The default count shards of this 89,796-peak assay do not fit `mem_budget="8G"`, so the import passes the smaller `policy` that its `MemoryError` names.

```{code-cell} ipython3
from scarf.storage.count_matrix import CountMatrixPolicy

tenx_h5 = datasets.download_dataset(
    name="tenx_10K_pbmc-v1_atacseq",
    destination=output_dir,
)
atac_store = output_dir / "pbmc_atac.zarr"
reader = scarf.CrH5Reader(str(tenx_h5 / "data.h5"))
scarf.CrToZarr(
    reader,
    zarr_loc=str(atac_store),
    mem_budget="8G",
    policy=CountMatrixPolicy(unitBytes=500_000_000, chunkBytes=50_000_000),
).dump()
atac_store
```

When we open the written store, the summary lists an ATAC assay, which confirms that inference from the H5 feature types survived the dump:

```{code-cell} ipython3
ds_atac = scarf.DataStore(str(atac_store))
ds_atac
```

## Import a Seurat RDS

Scarf can import a serialized Seurat object from an `.rds` file through `inspect_seurat`, `SeuratReader`, and `SeuratToZarr`. This path reads the on-disk RDS document, does not attach to a live R session, and it does not read `.h5seurat`. Always inspect the dataset before importing.
the import result reports which assays and reductions are importable, their dimensions, and any blocking diagnostics or notices:

```python
import scarf

inspection = scarf.inspect_seurat("pbmc.rds")
{
    "active assay": inspection.activeAssay,
    "importable assays": [assay.name for assay in inspection.assays if assay.importable],
    "importable reductions": [r.name for r in inspection.reductions if r.importable],
}
```

Open a reader for the assays and reductions you want, then write the Zarr store.
Omitting `reductions` (these are the dimensionality reductions) selects every available reduction; Pass an empty sequence to skip reductions, or pass only importable names to import a subset of the data. `SeuratToZarr` prepares and reads each selected count layer once when it is created, so the integral counts that Seurat holds as R doubles (data format that R uses for counts) are stored unsigned, and each assay gets its own count dtype and layout:

```python
with scarf.SeuratReader(
    "pbmc.rds",
    assays=["RNA"],
    reductions=["pca"],
) as reader:
    result = scarf.SeuratToZarr(reader, zarr_loc="pbmc_from_seurat.zarr").dump()

ds = scarf.DataStore("pbmc_from_seurat.zarr")
{
    "assays": result.assayNames,
    "default assay": result.defaultAssay,
    "notices": result.notices,
    "identities": result.activeIdentity,
    "PCA": result.reductionArtifacts["pca"],
}
```

The `activeIdentity` is an exact cluster-label artifact. `reductionArtifacts` maps each requested
Seurat reduction name to its exact imported-coordinate ref. Neither result is installed as a live
analysis column. Pass the reduction ref to graph construction or load either column explicitly.

The imports and conversion from Seurat files to Zarr do not cover, meaning Scarf cannot do this:

- Neighbor graphs, Seurat `neighbors` objects, images, commands, and most `tools` slots
- Normalized layers when the selected count layer is used for the Scarf assay
- Transposed Assay5 storage (`Assay5T`)
- A return path to `.rds` or `.h5seurat` (export H5AD or MTX instead)

Scarf prefers original 10x HDF5 or Matrix Market counts when they are available and you only need raw matrices due to limitations with importing Seurat files.

## Handling Exports

## Export to Matrix Market

Open the H5AD-derived store that we did when we discussed above and load the selected analytical artifacts:

```{code-cell} ipython3
ds = scarf.DataStore(str(pancreas_store))

imported_umap = np.asarray(
    ds.load_artifact(h5ad_import.embeddingArtifacts["X_umap"])["values"][:]
)
imported_clusters = np.asarray(
    ds.load_artifact(h5ad_import.clusterArtifacts["clusters"])["values"][:]
)
imported_umap.shape, imported_clusters.shape
```

Plot the imported layout colored by the imported cluster labels:

```{code-cell} ipython3
ds.plots.embedding(
    layout=h5ad_import.embeddingArtifacts["X_umap"],
    color_by=h5ad_import.clusterArtifacts["clusters"],
)
```

Then to export to the matrix market format, use the code below:

```{code-cell} ipython3
mtx_export = output_dir / "diff_pancreas"
scarf.writers.to_mtx(assay=ds.RNA, mtx_directory=str(mtx_export))
[(path.name, path.stat().st_size) for path in sorted(mtx_export.iterdir())]
```

## Export to H5AD and AnnData

Say you want to also attach specific imported artifacts, such as where the imported UMAP coordinates live. You can materialize the assay into memory as an AnnData object with `to_anndata`, insert the artifact values into its slots, and then export the result as H5AD as the tutorial does below. This never modifies the Zarr store; only the written H5AD carries the attached values.

### Attach exact imported artifacts and then export

Say you want to also attach specific import artifacts, such as the data as to where the UMAP is located, you can load the Zarr file into memory by converting it to an AnnData object, inserting the information, and then either converting back to a Zarr, or exporting as the tutorial does below.

```{code-cell} ipython3
adata = ds.to_anndata(from_assay="RNA")
adata.obsm["X_umap"] = imported_umap
adata.obs["clusters"] = imported_clusters
h5ad_export = output_dir / "diff_pancreas.h5ad"
adata.write_h5ad(h5ad_export)
{"file": h5ad_export.name, "bytes": h5ad_export.stat().st_size}
```

Reload the H5AD and confirm that the explicitly attached layout that we want to export is in `obsm`:

```{code-cell} ipython3
import anndata as ad

adata = ad.read_h5ad(h5ad_export)
sorted(adata.obsm.keys()), adata.obsm["X_umap"].shape
```

### Export a feature panel with `to_anndata`

Full-assay export can require a large amount of memory and disk to export all of the information. If you only need to export part of the dataset, such as a marker panel of genes, you can select features before materializing AnnData. If you want to export everything, simply leave out the `feature_names` section.

```{code-cell} ipython3
all_names = ds.RNA.feats.fetch_all("names").astype(str)
name_lookup = {name.upper(): name for name in all_names}
panel = [name_lookup[gene] for gene in ("GCG", "SST", "KRT19")]
selected = ds.to_anndata(from_assay="RNA", matrix="raw", feature_names=panel)
{
    "shape": selected.shape,
    "genes": selected.var["names"].tolist(),
}
```

The panel export contains live metadata only, meaning any cells that were filtered out and marked off from the analysis will not be included.

## Handling Unique Imports

## Import CSV files

`CSVReader` and `CSVtoZarr` provide small-data compatibility for dense CSV count matrices.
The toy matrix below is synthesized in-notebook simply for example purposes, as dense CSV count matrices are less common. Rows are cells and columns are features; `cell_data_cols` moves selected columns into cell metadata.

```{code-cell} ipython3
csv_path = output_dir / "toy_counts.csv"
csv_path.write_text(
    "quality,geneA,geneB,geneC\n"
    "10,1,0,2\n"
    "20,0,3,0\n"
    "30,4,5,6\n"
    "40,7,0,8\n"
    "50,9,10,0\n",
    encoding="utf-8",
)

print(csv_path.read_text(encoding="utf-8"))
```

Convert the CSV, keeping `quality` as cell metadata:

```{code-cell} ipython3
csv_zarr = output_dir / "toy_csv.zarr"
reader = scarf.CSVReader(
    str(csv_path),
    cell_data_cols=["quality"],
)
scarf.CSVtoZarr(reader, zarr_loc=str(csv_zarr), assay_name="RNA").dump()
ds_csv = scarf.DataStore(str(csv_zarr))
ds_csv.cells.head()
```

`quality` is cell metadata rather than a count column, which is what `cell_data_cols` is for.

## Import sparse matrices

`SparseToZarr` accepts a SciPy CSR matrix with matching cell and feature IDs. Here, we use a synthetic dataset as an example, but the general concept remains the same for real data.

```{code-cell} ipython3
from scipy.sparse import csr_matrix

mat = csr_matrix(
    (
        [1, 10, 15, 10, 20, 2, 3, 1, 5],
        ([0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 3, 8, 2, 3, 1, 2, 8, 9]),
    ),
    shape=(3, 10),
)
mat.toarray()
```

Write this small matrix with matching cell and feature identifiers:

```{code-cell} ipython3
sparse_zarr = output_dir / "toy_sparse.zarr"
scarf.SparseToZarr(
    mat,
    zarr_loc=str(sparse_zarr),
    cell_ids=[f"cell_{i}" for i in range(mat.shape[0])],
    feature_ids=[f"feat_{i}" for i in range(mat.shape[1])],
    assay_name="RNA",
).dump()
ds_sparse = scarf.DataStore(str(sparse_zarr))
ds_sparse
```

For a complete `DataStoreMerge` example, continue to {doc}`dataset_merging`.

## Other import paths

### Chunked arrays, Remote Zarr destinations, and Count storage & memory

`chunked_to_zarr` writes from a Scarf `ChunkedArray` when lazy out-of-core conversion is needed, which is simply a more compute efficient manner of converting data.

 For importing from remote filesystems, like an AWS S3 bucket, Scarf is able to natively handle this

Choose the `cloud` profile for an object-store destination and pass credentials through the environment or runtime configuration to download the dataset.

```python
import os

inspection = scarf.inspect_h5ad("counts.h5ad")
reader = scarf.H5adReader.from_inspect(inspection)
writer = scarf.H5adToZarr(
    reader,
    zarr_loc="s3://my-bucket/project/data.zarr",
    storage_options={
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    },
    profile="cloud",
)
writer.dump()
```

Scarf can also perform the import in a more memory efficient manner if required.

Start with the default import layout, and if the import does not fit its `mem_budget`, the error names a smaller `CountMatrixPolicy` that fits, as used in the ATAC example above. This imports the data in smaller, more manageable chunks at the cost of time, so prefer a larger memory budget when your machine allows it: smaller RNA shards also slow later gene-major reads.
See {doc}`../concepts/memory_and_execution` for memory planning and
{doc}`../reference/api/import_export` for reader buffering and storage options.

## Common mistakes & their solutions

- Exporting normalized values when a downstream method requires raw counts. Keep the default raw matrix unless the receiving tool explicitly asks for normalized values.
- Assuming an H5AD file uses `X` for raw counts without inspecting its layers. Run `inspect_h5ad` first and convert from the layer that actually holds the raw counts.
- Omitting `embedding_roles` or `cluster_keys` when analytical H5AD values should become artifacts. Pass them to the reader up front, since values left out of the import never enter the artifact record.
- Materializing a full AnnData object when a panel of features would answer the same question. Resolve the panel with `feature_names` or `feature_indexes` first and export only those columns.
- Treating Seurat neighbor graphs, images, or normalized layers as imported Scarf artifacts. These slots are always ignored, so plan the analysis around the imported counts, metadata, reductions, and identities instead.
