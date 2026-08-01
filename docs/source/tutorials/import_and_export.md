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

Scarf reads common single-cell count formats, writes a Zarr store for analysis, and exports
counts or metadata to interoperable formats.

## Prerequisites

- Scarf installed with the optional dependencies required by the source format
- A source count matrix in a supported format

## What you will learn

- Download datasets from the `scarf_docs` Cytebase catalog
- Convert 10x HDF5, MTX, H5AD, CSV, and sparse inputs to Zarr
- Merge full DataStores with `DatasetMerge`
- Export an assay to MTX or H5AD

## Guided steps

```{code-cell} ipython3
import scarf

scarf.configure_output(level='ERROR', progress=True)
```

### 1. Download datasets from Cytebase

Scarf hosts example datasets in the public [Cytebase bucket](https://huggingface.co/buckets/Nygen/cytebase) in formats such as MTX, 10x HDF5, and H5AD. Connect to the `scarf_docs` repository to list or download them:

```{code-cell} ipython3
datasets = scarf.cytebase.connect("scarf_docs")
datasets.list_datasets()
```

**Naming format**: `<author>_<number of cells>_<cell/tissue type or species>_<single-cell method>`

Examples used below:

Each download returns the directory it wrote, which the readers below use as their input
path.

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
```

The downloads land under `scarf_datasets` in the current working directory unless
`destination` is changed.

### 2. Convert data to a Scarf Zarr store

Scarf stores data as dense, compressed chunks in Zarr. `scarf.readers` and `scarf.writers`
provide complementary classes that convert common count formats into that layout.

#### From 10x's HDF5 file format

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

#### From Matrix Market count files

Inspect a Matrix Market source before selecting a triplet. A source can be an
`.mtx` or `.mtx.gz` file, a directory, or a direct MEX ZIP. Inspection recognizes
canonical 10x names, common prefixed triplets, and Parse DGE directories. It
returns every complete candidate instead of choosing between alternatives such
as raw and filtered matrices.

```{code-cell} ipython3
candidates = scarf.inspect_mtx(str(mtx_dir))
candidates
```

Select one candidate explicitly when inspection reports more than one. This
example contains a count above the `uint32` range, so both the reader and writer
use `uint64`:

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

Cell-major coordinates stream directly. Feature-major coordinates, including
BD Rhapsody MEX output, are converted to a temporary disk-backed CSR matrix.
The reader checks available capacity and reports the exact required bytes
before creating those files. Pass `temp_dir` to `MtxReader` when the system
temporary directory is too small.

Parse DGE matrices use cells by genes orientation and require
`cell_metadata.csv`. Scarf imports its non-ID columns. It recognizes
`bc_wells` and `bc_index`; pass `cell_id_key` when both are present:

```python
candidate = scarf.inspect_mtx("/path/to/parse_dge")[0]
reader = scarf.MtxReader(candidate, cell_id_key="bc_index")
scarf.MtxToZarr(reader, zarr_loc="parse.zarr").dump()
```

#### From AnnData H5AD file format

H5AD files vary in where they store counts, feature names, metadata, and
layers. Inspect the file before conversion rather than assuming `X`, `obs`, and
`var` contain the intended values.

```{code-cell} ipython3
h5ad_path = f'{h5ad_dir}/data.h5ad'
inspection = scarf.inspect_h5ad(h5ad_path)
inspection
```

`H5adReader.from_inspect` uses the discovered matrix and metadata keys. Override
the inspection only after confirming that another layer contains the raw count
matrix required by the analysis.

```{code-cell} ipython3
reader = scarf.H5adReader.from_inspect(inspection)

# change value of `zarr_loc` to your choice of filename and path
writer = scarf.H5adToZarr(
    reader,
    zarr_loc='scarf_datasets/differentiating_pancreatic_cells.zarr'
)
writer.dump()
```

Categorical columns are decoded from category codes. Missing categorical or
object values become `None`; missing numeric nullable values become `NaN`.
Unsupported group-encoded columns are skipped with a warning rather than
treated as valid metadata.

Supported dense `obsm` arrays with one row per cell are flattened into numbered
cell columns. For example, a two-column `X_umap` array becomes `X_umap1` and
`X_umap2`. Sparse or row-mismatched `obsm` entries are warned about and skipped.

Source read batches are selected automatically from destination shard geometry
and the conversion memory budget. Physical writes remain shard-aligned even
when the selected source batch is smaller. An explicit positive `batch_size`
remains available for controlled profiling and expert workflows.

10x feature types are retained in feature metadata. CRISPR guide, multiplexing,
antigen, custom, RNA, and antibody labels receive stable assay names. Scarf
does not infer BD guide features from names. Reclassify known feature indexes
before constructing the writer when a vendor labels guides as gene expression.

Loom import remains available through `LoomReader` and `LoomToZarr` for legacy
compatibility.

### 3. Export data from a Zarr store

#### To Cellranger (10x) MTX format

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/differentiating_pancreatic_cells.zarr')

ds
```

Check the imported cell columns before relying on transferred labels or
embeddings:

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

```{code-cell} ipython3
scarf.writers.to_mtx(
    assay=ds.RNA,
    mtx_directory='scarf_datasets/diff_pancreas'
)
```

#### To H5ad format

`to_h5ad` exports the count matrix and metadata, and writes UMAP or tSNE coordinate pairs to
AnnData `obsm` by default. `DataStore.to_anndata` returns an in-memory AnnData object with
counts, cell and feature metadata, and optional assay layers. It currently leaves layout
coordinates as ordinary `obs` columns rather than populating `obsm` (see {doc}`downsampling`).

```{code-cell} ipython3
scarf.writers.to_h5ad(
    assay=ds.RNA,
    h5ad_filename='scarf_datasets/diff_pancreas.h5ad'
)
```

Full-assay export can require enough memory and disk for the selected cell by
feature matrix. When only a marker panel is needed, select features before
materializing AnnData:

```{code-cell} ipython3
all_names = ds.RNA.feats.fetch_all("names").astype(str)
name_lookup = {name.upper(): name for name in all_names}
panel = [
    name_lookup[gene]
    for gene in ["GCG", "INS", "SST", "KRT19"]
    if gene in name_lookup
]
if not panel:
    panel = all_names[:4].tolist()
selected = ds.to_anndata(
    from_assay="RNA",
    cell_key="I",
    matrix="raw",
    feature_names=panel,
)
selected.shape, selected.var_names.tolist()
```

Use `feature_indexes` instead when stable feature rows are already available.
`feature_names` and `feature_indexes` are mutually exclusive, preserve the
requested order, and reject duplicate or unknown selections.

`to_h5ad` writes recognized UMAP and t-SNE coordinate pairs to `obsm`.
`DataStore.to_anndata` currently leaves layout coordinates as ordinary `obs`
columns. This distinction matters when another tool expects `obsm["X_umap"]`.

Writers also accept remote Zarr locations. Choose the `cloud` profile for an
object-store destination and pass credentials through the environment or
runtime configuration:

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

### 4. Convert a CSV matrix to Zarr

`CSVReader` and `CSVtoZarr` provide small-data compatibility for dense CSV count matrices. The toy matrix below is
synthesized in-notebook so the conversion does not depend on a catalog file. Rows are cells
and columns are features; `cell_data_cols` moves selected columns into cell metadata.

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
ds_csv
```

### 5. Convert a sparse matrix to Zarr

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

### 6. Merge full DataStores with DatasetMerge

`DatasetMerge` merges multiple full DataStores (all assays per dataset) into one Zarr file.
The example below merges two tiny stores created with `SparseToZarr`. For single-assay merges
of larger studies see `AssayMerge` in {doc}`data_integration`.

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
merger = scarf.DatasetMerge(
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

`dask_to_zarr` writes from a Dask array when lazy out-of-core conversion is needed. Loom
import uses `LoomReader` / `LoomToZarr` with the same dump pattern as the readers above; this
page does not execute a Loom example.

## Common mistakes and limitations

- Fetching a prepared Zarr store when the aim is to demonstrate source-format conversion
- Reusing an existing Zarr output path without confirming that it can be overwritten
- Exporting normalized values when a downstream method requires raw counts
- Using `DatasetMerge` when you only need one assay from each store (`AssayMerge` is enough)
- Assuming an H5AD file uses `X` for raw counts without inspecting its layers
- Expecting sparse or malformed `obsm` arrays to be imported as embeddings
- Materializing a full AnnData object when a feature panel would answer the export question

Conversion writes the requested Zarr target. Export commands write MTX or H5AD
at the supplied destination, and `DatasetMerge` writes its merged store at
`zarr_path`.
