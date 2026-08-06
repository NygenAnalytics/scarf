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
| Loom | | `LoomReader` | `LoomToZarr` |

Export paths write Matrix Market or H5AD.
Scarf does not write Seurat `.rds` or `.h5seurat` files.
See {doc}`../scanpy_and_seurat` for Scanpy and Seurat workflow mapping.

## Prerequisites

- Scarf installed with the optional dependencies required by the source format
- A source count matrix in a supported format

## What you will learn

- Download datasets from the `scarf_docs` Cytebase catalog
- Convert 10x HDF5, MTX, H5AD, Seurat RDS, CSV, and sparse inputs to Zarr
- Export an assay to MTX or H5AD
- Merge full DataStores with `DataStoreMerge`

## 1. Download example datasets

Scarf hosts example datasets in the public [Cytebase bucket](https://huggingface.co/buckets/Nygen/cytebase) in formats such as MTX, 10x HDF5, and H5AD.
Connect to the `scarf_docs` repository to list or download them:

```{code-cell} ipython3
import scarf

scarf.configure_output(level='ERROR', progress=True)
```

```{code-cell} ipython3
datasets = scarf.cytebase.connect("scarf_docs")
datasets.list_datasets()
```

**Naming format**: `<author>_<number of cells>_<cell/tissue type or species>_<single-cell method>`

Each download returns the directory it wrote, which the readers below use as their input path.

```{code-cell} ipython3
# This dataset is in Cellranger (10x) HDF5 format.
tenx_h5 = datasets.download_dataset(
    name='tenx_10K_pbmc-v1_atacseq',
    destination='scarf_datasets'
)

# This dataset is in MTX format along with barcodes and features TSV files.
mtx_dir = datasets.download_dataset(
    name='xin_1K_pancreas_rnaseq',
    destination='scarf_datasets'
)

# This dataset is in H5ad (anndata) format.
h5ad_dir = datasets.download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets'
)

tenx_h5, mtx_dir, h5ad_dir
```

The downloads land under `scarf_datasets` in the current working directory unless `destination` is changed.
The cell prints each returned path.

## 2. Import 10x HDF5

Scarf stores data as dense, compressed chunks in Zarr.
`CrH5Reader` and `CrToZarr` convert Cell Ranger HDF5 into that layout.
Assay type is inferred from the H5 feature types (RNA, ATAC, or multimodal).

```{code-cell} ipython3
# Assay type is inferred from the H5 contents (RNA, ATAC, or multimodal).
reader = scarf.CrH5Reader(f'{tenx_h5}/data.h5')

# change value of `zarr_loc` to your choice of filename and path
writer = scarf.CrToZarr(
    reader,
    zarr_loc='scarf_datasets/pbmc_atac.zarr'  
)  
writer.dump()
```

Open the written store.
The summary lists an ATAC assay, which confirms that inference from the H5 feature types survived the dump:

```{code-cell} ipython3
ds_atac = scarf.DataStore('scarf_datasets/pbmc_atac.zarr')
ds_atac
```

## 3. Import Matrix Market

Inspect a Matrix Market source before selecting a triplet.
A source can be an `.mtx` or `.mtx.gz` file, a directory, or a direct MEX ZIP.
Inspection recognizes canonical 10x names, common prefixed triplets, and Parse DGE directories.
It returns every complete candidate instead of choosing between alternatives such as raw and filtered matrices.

```{code-cell} ipython3
candidates = scarf.inspect_mtx(str(mtx_dir))
candidates
```

Select one candidate explicitly when inspection reports more than one.
This example contains a count above the `uint32` range, so both the reader and writer use `uint64`:

```{code-cell} ipython3
reader = scarf.MtxReader(candidates[0], dtype='uint64')

# change value of `zarr_loc` to your choice of filename and path
writer = scarf.MtxToZarr(
    reader,
    zarr_loc='scarf_datasets/xin_1K.zarr',
    dtype='uint64'
)
writer.dump()
```

Reopen the store and check that the count matrix kept the requested width and the candidate dimensions:

```{code-cell} ipython3
ds_mtx = scarf.DataStore('scarf_datasets/xin_1K.zarr')
ds_mtx.RNA.rawData.dtype, ds_mtx.RNA.rawData.shape
```

Cell-major coordinates stream directly.
Feature-major coordinates, including BD Rhapsody MEX output, are converted to a temporary disk-backed CSR matrix.
The reader checks available capacity and reports the exact required bytes before creating those files.
Pass `temp_dir` to `MtxReader` when the system temporary directory is too small.

### 3.1 Parse DGE directories

Parse DGE matrices use cells by genes orientation and require `cell_metadata.csv`.
Scarf imports its non-ID columns.
It recognizes `bc_wells` and `bc_index`; pass `cell_id_key` when both are present:

```python
candidate = scarf.inspect_mtx("/path/to/parse_dge")[0]
reader = scarf.MtxReader(candidate, cell_id_key="bc_index")
scarf.MtxToZarr(reader, zarr_loc="parse.zarr").dump()
```

## 4. Import H5AD

H5AD files vary in where they store counts, feature names, metadata, and layers.
Inspect the file before conversion rather than assuming `X`, `obs`, and `var` contain the intended values.

```{code-cell} ipython3
h5ad_path = f'{h5ad_dir}/data.h5ad'
inspection = scarf.inspect_h5ad(h5ad_path)
inspection
```

`H5adReader.from_inspect` uses the discovered matrix and metadata keys.
Override the inspection only after confirming that another layer contains the raw count matrix required by the analysis.

```{code-cell} ipython3
reader = scarf.H5adReader.from_inspect(inspection)

# change value of `zarr_loc` to your choice of filename and path
writer = scarf.H5adToZarr(
    reader,
    zarr_loc='scarf_datasets/differentiating_pancreatic_cells.zarr'
)
writer.dump()
```

Categorical columns are decoded from category codes.
Missing categorical or object values become `None`; missing numeric nullable values become `NaN`.
Unsupported group-encoded columns are skipped with a warning rather than treated as valid metadata.

Supported dense `obsm` arrays with one row per cell are flattened into numbered cell columns.
For example, a two-column `X_umap` array becomes `X_umap1` and `X_umap2`.
Sparse or row-mismatched `obsm` entries are warned about and skipped.

Source read batches are selected automatically from destination shard geometry and the conversion memory budget.
Physical writes remain shard-aligned even when the selected source batch is smaller.
An explicit positive `batch_size` remains available for controlled profiling and expert workflows.

10x feature types are retained in feature metadata when present.
Stable multi-assay names (CRISPR guide, multiplexing, antigen, custom, RNA, antibody, and similar) require `assay_split_key` on `H5adToZarr` (for example `feature_types`).
A plain `from_inspect` path without `assay_split_key` writes everything into one assay (default RNA).
Inspection may set `assaySplitKey` and `suggestedAssays`, but `to_reader_kwargs` does not pass `assaySplitKey` through.
Pass `assay_split_key` and optional `assay_name_map` on the writer to split.
`reclassify_features` is a `CrReader` API (10x HDF5 / MEX), not `H5adReader`.

## 5. Import Seurat RDS

Scarf can import a serialized Seurat object from an `.rds` file through `inspect_seurat`, `SeuratReader`, and `SeuratToZarr`.
This path reads the on-disk RDS document.
It does not attach to a live R session, and it does not read `.h5seurat`.

Inspect first.
The result reports which assays and reductions are importable, their dimensions, and any blocking diagnostics or notices:

```python
import scarf

inspection = scarf.inspect_seurat("pbmc.rds")
inspection.activeAssay
[assay.name for assay in inspection.assays if assay.importable]
[
    reduction.name
    for reduction in inspection.reductions
    if reduction.importable
]
```

Open a reader for the assays and reductions you want, then write the Zarr store.
Omitting `reductions` selects every available reduction; `SeuratToZarr` raises if any selected reduction is not importable.
Pass an empty sequence to skip reductions, or pass only importable names to import a subset:

```python
with scarf.SeuratReader(
    "pbmc.rds",
    assays=["RNA"],
    reductions=["pca"],
) as reader:
    result = scarf.SeuratToZarr(
        reader,
        zarr_loc="pbmc_from_seurat.zarr",
    ).dump()

ds = scarf.DataStore("pbmc_from_seurat.zarr")
result.assayNames, result.defaultAssay, result.notices
```

What this import covers:

- Legacy `Assay`, `Assay5`, and `ChromatinAssay` count layers when their matrix layout is supported
- Cell metadata, `active.ident`, and selected reductions such as PCA or LSI
- Partial Assay5 cell membership as per-assay boolean columns when needed

What it does not import as analysis artifacts:

- Neighbour graphs, Seurat `neighbors` objects, images, commands, and most `tools` slots
- Normalized layers when the selected count layer is used for the Scarf assay
- Transposed Assay5 storage (`Assay5T`)
- A return path to `.rds` or `.h5seurat` (export H5AD or MTX instead)

Pass `assay_layers` when an assay stores several count layers and you need a non-default choice.
Pass `sidecar_path_remaps` when a `SaveSeuratRds` sidecar cache points at moved on-disk matrices.
Prefer original 10x HDF5 or Matrix Market counts when they are available and you only need raw matrices.

## 6. Export to Matrix Market

Open the H5AD-derived store written in section 4 and inspect transferred columns before exporting:

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/differentiating_pancreatic_cells.zarr')

ds
```

```{code-cell} ipython3
{
    "cellColumns": ds.cells.columns[:12],
    "embeddingColumns": [
        name
        for name in ds.cells.columns
        if str(name).startswith(("X_umap", "X_tsne"))
    ],
}
```

Plot the imported layout colored by the imported cluster labels:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='X_umap',
    color_by='clusters',
)
```

```{code-cell} ipython3
scarf.writers.to_mtx(
    assay=ds.RNA,
    mtx_directory='scarf_datasets/diff_pancreas'
)
```

## 7. Export to H5AD and AnnData

`to_h5ad` exports the count matrix and metadata, and promotes recognized `{assay}_UMAP` / `{assay}_tSNE` column pairs into AnnData `obsm`.
Imported layouts arrive as `X_umap1` / `X_umap2`, so copy them to the Scarf export names before writing.
`DataStore.to_anndata` returns an in-memory AnnData object with counts, cell and feature metadata, and optional assay layers.
It currently leaves layout coordinates as ordinary `obs` columns rather than populating `obsm` (see {doc}`downsampling`).

### 7.1 Promote layouts for `to_h5ad`

```{code-cell} ipython3
ds.cells.insert(
    column_name='RNA_UMAP1',
    values=ds.cells.fetch_all('X_umap1'),
    overwrite=True,
)
ds.cells.insert(
    column_name='RNA_UMAP2',
    values=ds.cells.fetch_all('X_umap2'),
    overwrite=True,
)
scarf.writers.to_h5ad(
    assay=ds.RNA,
    h5ad_filename='scarf_datasets/diff_pancreas.h5ad'
)
```

Reload the H5AD and confirm that the promoted layout is in `obsm`:

```{code-cell} ipython3
import anndata as ad

adata = ad.read_h5ad('scarf_datasets/diff_pancreas.h5ad')
sorted(adata.obsm.keys()), adata.obsm['X_umap'].shape
```

### 7.2 Export a feature panel with `to_anndata`

Full-assay export can require enough memory and disk for the selected cell by feature matrix.
When only a marker panel is needed, select features before materializing AnnData.
Genes absent from the feature table are omitted from the panel; here `INS` is not present, so it is dropped:

```{code-cell} ipython3
all_names = ds.RNA.feats.fetch_all("names").astype(str)
name_lookup = {name.upper(): name for name in all_names}
requested = ["GCG", "INS", "SST", "KRT19"]
panel = [
    name_lookup[gene]
    for gene in requested
    if gene in name_lookup
]
dropped = [gene for gene in requested if gene not in name_lookup]
if not panel:
    panel = all_names[:4].tolist()
selected = ds.to_anndata(
    from_assay="RNA",
    cell_key="I",
    matrix="raw",
    feature_names=panel,
)
{
    "shape": selected.shape,
    "genes": selected.var_names.tolist(),
    "dropped": dropped,
    "umapInObs": [
        name
        for name in selected.obs.columns
        if str(name).startswith("X_umap")
    ],
    "obsmKeys": list(selected.obsm.keys()),
}
```

Use `feature_indexes` instead when stable feature rows are already available.
`feature_names` and `feature_indexes` are mutually exclusive, preserve the requested order, and reject duplicate or unknown selections.

The panel export keeps layout coordinates in `obs` and leaves `obsm` empty.
That is the `to_anndata` side of the distinction above: use `to_h5ad` when another tool expects `obsm["X_umap"]`.

## 8. Import CSV

`CSVReader` and `CSVtoZarr` provide small-data compatibility for dense CSV count matrices.
The toy matrix below is synthesized in-notebook so the conversion does not depend on a catalog file.
Rows are cells and columns are features; `cell_data_cols` moves selected columns into cell metadata.

```{code-cell} ipython3
from pathlib import Path
import shutil

import numpy as np
from scipy.sparse import csr_matrix

csv_dir = Path('scarf_datasets')
csv_dir.mkdir(parents=True, exist_ok=True)
csv_path = csv_dir / 'toy_counts.csv'
csv_path.write_text(
    'quality,geneA,geneB,geneC\n'
    '10,1,0,2\n'
    '20,0,3,0\n'
    '30,4,5,6\n'
    '40,7,0,8\n'
    '50,9,10,0\n',
    encoding='utf-8',
)

csv_zarr = csv_dir / 'toy_csv.zarr'
if csv_zarr.exists():
    shutil.rmtree(csv_zarr)

reader = scarf.CSVReader(
    str(csv_path),
    cell_data_cols=['quality'],
)
writer = scarf.CSVtoZarr(
    reader,
    zarr_loc=str(csv_zarr),
    assay_name='RNA',
    dtype=np.dtype('uint16'),
)
writer.dump()
ds_csv = scarf.DataStore(str(csv_zarr))
ds_csv.cells.head()
```

`quality` is cell metadata rather than a count column, which is what `cell_data_cols` is for.

## 9. Import sparse matrices

`SparseToZarr` accepts a SciPy CSR matrix with matching cell and feature IDs.

```{code-cell} ipython3
mat = csr_matrix(
    (
        [1, 10, 15, 10, 20, 2, 3, 1, 5],
        ([0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 3, 8, 2, 3, 1, 2, 8, 9]),
    ),
    shape=(3, 10),
)
sparse_zarr = Path('scarf_datasets/toy_sparse.zarr')
if sparse_zarr.exists():
    shutil.rmtree(sparse_zarr)
sparse_writer = scarf.SparseToZarr(
    mat,
    zarr_loc=str(sparse_zarr),
    cell_ids=[f'cell_{i}' for i in range(mat.shape[0])],
    feature_ids=[f'feat_{i}' for i in range(mat.shape[1])],
    assay_name='RNA',
)
sparse_writer.dump()
ds_sparse = scarf.DataStore(str(sparse_zarr))
ds_sparse
```

## 10. Merge DataStores

`DataStoreMerge` merges multiple full DataStores (all assays per dataset) into one Zarr file.
The example below merges two tiny stores created with `SparseToZarr`.
For single-assay merges pass `assays=["RNA"]` (or the assay name you need).

```{code-cell} ipython3
for name, values in [
    ('toy_merge_a.zarr', [1, 2, 3, 4, 5, 6]),
    ('toy_merge_b.zarr', [7, 8, 9, 10, 11, 12]),
]:
    path = Path('scarf_datasets') / name
    if path.exists():
        shutil.rmtree(path)
    m = csr_matrix(np.asarray(values, dtype=np.uint16).reshape(3, 2))
    scarf.SparseToZarr(
        m,
        zarr_loc=str(path),
        cell_ids=[f'{path.stem}_{i}' for i in range(3)],
        feature_ids=['g1', 'g2'],
        assay_name='RNA',
    ).dump()

ds_a = scarf.DataStore('scarf_datasets/toy_merge_a.zarr')
ds_b = scarf.DataStore('scarf_datasets/toy_merge_b.zarr')
merger = scarf.DataStoreMerge(
    datasets=[ds_a, ds_b],
    zarr_path='scarf_datasets/toy_merged.zarr',
    names=['a', 'b'],
    source_column='sample_id',
    overwrite=True,
)
merger.dump()
ds_merged = scarf.DataStore('scarf_datasets/toy_merged.zarr')
ds_merged.cells.head()
```

## 11. Other import paths

### 11.1 Loom

Loom import remains available through `LoomReader` and `LoomToZarr` with the same dump pattern as the readers above.
This page does not execute a Loom example.

### 11.2 Dask arrays

`dask_to_zarr` writes from a Dask array when lazy out-of-core conversion is needed.

### 11.3 Remote Zarr destinations

Writers also accept remote Zarr locations.
Choose the `cloud` profile for an object-store destination and pass credentials through the environment or runtime configuration:

```python
import os

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

## Common mistakes and limitations

- Fetching a prepared Zarr store when the aim is to demonstrate source-format conversion
- Reusing an existing Zarr output path without confirming that it can be overwritten
- Exporting normalized values when a downstream method requires raw counts
- Using `DataStoreMerge` without `assays=` when you only need one modality from multi-assay stores
- Assuming an H5AD file uses `X` for raw counts without inspecting its layers
- Expecting sparse or malformed `obsm` arrays to be imported as embeddings
- Materializing a full AnnData object when a feature panel would answer the export question
- Treating Seurat neighbour graphs, images, or normalized layers as imported Scarf artifacts
- Expecting Scarf to read `.h5seurat` or write Seurat `.rds` files

Conversion writes the requested Zarr target.
Export commands write MTX or H5AD at the supplied destination, and `DataStoreMerge` writes its merged store at `zarr_path`.
