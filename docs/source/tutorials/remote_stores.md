---
description: Open or mount remote Scarf DataStores, tune cloud storage, and stage local scratch.
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

(remote_stores)=

# Working with remote stores

Scarf can open a Zarr store on object storage and run analysis without first copying the full matrix to local disk.
Counts stream in tiles sized from your memory budget.
Published remote object-store funnel timings are in {doc}`../concepts/benchmarks`; resource controls are in {doc}`../concepts/memory_and_execution`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Credentials for your bucket (except for anonymous public reads)
- Enough local disk for optional `local_cache` scratch during reduction

## What you will learn

- Open a release-matched store on object storage without downloading it
- Open `DataStore` on `s3://` or `gs://` with `storage_options`
- Mount shared count matrices into a separate writable analysis store
- Reopen a durable pipeline run from its mounted target
- Select the `cloud` storage profile
- Stage normalized data locally for PCA with `local_cache`

## 1. Open a release-matched remote store

An unpacked store built by the same Scarf release can be opened directly instead of downloading
an archive. Replace the URI and storage options below with those for your object store. The
executable sections use the rebuilt documentation archive so they do not depend on a separate
publication step.

```python
import scarf

ds = scarf.DataStore(
    "s3://my-bucket/current-scarf-store.zarr",
    zarr_mode="r",
    storage_options={"anon": True},
    nthreads=4,
)
print(ds)
```

Opening a store over object storage costs many small metadata requests, so expect this step to take a few minutes on a home connection.
Nothing but metadata is read until you touch the counts.
The printed summary lists active cells, assays, and the cell and feature columns already present in the remote store.

Counts and metadata in a matching published store remain readable. Durable analysis is reopened
through its pipeline label and exact artifacts. The next section mounts the rebuilt count matrices
and records a new pipeline run in a separate target.

```python
run = ds.pipeline.open(label="docs_default")
ds.plots.embedding(
    run=run,
    color_by="clusters",
)
```

## 2. Mount read-only counts

Use `mount_datastore` when count matrices must remain in a shared source store, but each analysis needs its own writable store.
Scarf copies cell and feature metadata into the target.
Mount validates and reads primary `counts` for matrix identity.
For RNA sources, the matching gene-major `countsT` copy must already be present on Zarr v3.
It is mounted with `counts` and is not rewritten into the target.
Non-RNA assays have no `countsT`.
New metadata and analysis artifacts are written only to the target.

The public snapshot was rebuilt from raw counts with the current paired RNA layout. Download it
once, then mount those count arrays into a separate writable target.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scarf
import zarr

scarf.configure_output(level="ERROR", progress=False)
repository = scarf.cytebase.connect("scarf_docs")
mount_directory = TemporaryDirectory()
staged_dataset = repository.download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination=mount_directory.name,
    zarr=True,
)
source_path = staged_dataset / 'data.zarr'
target_path = Path(mount_directory.name) / 'analysis.zarr'
```

The target path must not already exist:

```{code-cell} ipython3
mounted = scarf.mount_datastore(
    str(source_path),
    at=str(target_path),
    default_assay='RNA',
    nthreads=4,
)

matrix_source = zarr.open_group(str(target_path), mode='r').attrs['matrixSource']
print('Target path:', target_path)
print('Target exists:', target_path.exists())
print('Downloaded dataset:', staged_dataset)
print('Mounted count source:', matrix_source['location'])
print('Mounted assays:', sorted(matrix_source['assays']))
```

Counts and RNA `countsT` stay in the downloaded count source.
Cell and feature metadata are copied once, while new analysis artifacts are written to the target.
The printed `matrixSource` record is what later reopen uses to resolve those counts.

```{mermaid}
flowchart LR
    source["Read-only remote dataset"]
    staged["Current local count source<br/>counts, RNA countsT, metadata"]
    target["Writable mounted target<br/>metadata and new artifacts"]
    source -->|download once| staged
    staged -->|mount count blocks| target
```

The target assay has no physical count array, while `rawData` exposes the complete mounted count matrix:

```{code-cell} ipython3
target_root = zarr.open_group(str(target_path), mode='r')

print('Counts stored in target:', 'counts' in target_root['RNA'])
print('Mounted shape:', mounted.RNA.rawData.shape)
```

Run the standard RNA pipeline through the mount. Count blocks are read from the current source,
while the run record and its normalized data, reductions, graph, UMAP, and
clusters are written only to the local target. Because that target is a local path, `local_cache`
staging is skipped here; Section 4 makes the remote-only policy explicit.

```{code-cell} ipython3
mounted_run = mounted.pipeline.run(
    filtering=False,
    hvg_count=500,
    pca_dims=15,
    leiden={"partitions": [0.5]},
    cell_cycle=False,
    paris=False,
    doublets=False,
    markers=False,
)
normalized = mounted_run["normalized"]
pca = mounted_run["pca"]
```

```{code-cell} ipython3
mounted.plots.embedding(run=mounted_run, color_by="clusters")
```

The populated embedding demonstrates that mounted counts behave like a normal datastore input.
The target remains much smaller than a dense local copy of the source matrix:

```{code-cell} ipython3
target_bytes = sum(
    path.stat().st_size
    for path in target_path.rglob("*")
    if path.is_file()
)
logical_count_bytes = int(
    np.prod(mounted.RNA.rawData.shape)
    * mounted.RNA.rawData.dtype.itemsize
)
print("Writable target bytes:", target_bytes)
print("Dense logical count bytes:", logical_count_bytes)
```

Opening the target later resolves the source automatically.
The source must remain accessible at the recorded path or URI:

```{code-cell} ipython3
reopened = scarf.DataStore(
    str(target_path),
    nthreads=4,
)
reopened_run = reopened.pipeline.open(run_id=mounted_run.run_id)
same_counts = np.array_equal(
    reopened.RNA.rawData[:20, :20].compute(),
    mounted.RNA.rawData[:20, :20].compute(),
)
print('Counts still resolve:', same_counts)
print('Reopened pipeline status:', reopened_run.status)
print(
    'Normalization complete:',
    reopened.inspect_artifact(reopened_run['normalized']).complete,
)
```

For a current S3 or GCS source with a complete RNA `countsT`, pass the URI directly and keep the target local:

```python
mounted = scarf.mount_datastore(
    's3://shared-bucket/atlas.zarr',
    at='my-analysis.zarr',
    storage_options={'skip_signature': True},
    zarrProfile='fast_local',
)
```

The mount records matrix shape, dtype, and source identity.
Reopening fails if the source no longer matches that identity.
Metadata is copied at mount time, so later source metadata changes are not synchronized into the target.

## 3. Open your own remote store

Pass the URI as `zarr_loc` and any fsspec/obstore options as `storage_options`.
Use `zarrProfile="cloud"` when writing new arrays so they use the cloud compression profile.
Existing arrays keep the physical layout chosen when they were created.

Anonymous read-only example against your own public bucket:

```python
import scarf

ds = scarf.DataStore(
    "s3://example-bucket/path/to/data.zarr",
    zarr_mode="r",
    zarrProfile="cloud",
    storage_options={"skip_signature": True},
    mem_budget="16G",
    nthreads=8,
)
```

Credentialed read-write template (do not embed secrets in notebooks):

```python
import os
import scarf

remote_writable = scarf.DataStore(
    "s3://my-bucket/project/data.zarr",
    zarr_mode="r+",
    zarrProfile="cloud",
    storage_options={
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        # "endpoint": "https://...",  # S3-compatible endpoints
    },
    mem_budget="32G",
    nthreads=8,
)
```

Google Cloud Storage uses a `gs://` URI.
Pass the provider options your environment already uses for obstore/fsspec (for example application-default credentials on the VM, or an explicit token in `storage_options`).

After open, call the same analysis APIs as on a local store. Use
`remote_writable.pipeline.run(...)` for the standard RNA workflow, and atomic producers for a
deliberate branch or execution option.

## 4. Local scratch for reductions

PCA fitting and score projection make multiple passes over normalized expression.
`local_cache` stages that normalized artifact to local disk when the *DataStore location* is remote (object-storage URI or non-local backend).
It does not key off whether counts alone are remote.
A mounted local target already holds normalized artifacts locally, so staging is skipped there even when `counts` stream from a remote source.
Harmony, ANN, and neighbor queries read persisted reduced coordinates and do not use normalized-expression scratch.

| Value | Behavior |
|---|---|
| `"auto"` (default) | Stage for remote stores; skip for local stores |
| `True` | Temporary scratch directory, deleted when the stage ends |
| `False` | No staging; every pass reads the store URI |
| `"/path/to/scratch"` | Persistent scratch keyed by artifact ID |

The mounted target in this page is local, so normalized artifacts already live on local disk and Scarf skips staging even when a scratch path is supplied.
This executable checkpoint makes that distinction explicit:

```{code-cell} ipython3
scratch_directory = TemporaryDirectory()
scratch_dir = Path(scratch_directory.name) / 'pca_scratch'
pca_without_staging = mounted.run_pca(
    normalized,
    dims=15,
    local_cache=str(scratch_dir),
)
staged_bytes = sum(
    path.stat().st_size for path in scratch_dir.rglob('*') if path.is_file()
)
print('Local cache path:', scratch_dir)
print('Staged normalized bytes:', staged_bytes)
print('Local cache present after PCA:', scratch_dir.exists())
print('Reduction reused:', pca_without_staging == pca)
```

On your own writable remote store, the same path-string policy stages normalized blocks and keeps
the cache for inspection or reuse. Only the stages needed to demonstrate the PCA option are shown:

```python
cell_selection = remote_writable.snapshot_cell_selection(cell_key="I")
features = remote_writable.select_hvgs(
    cell_selection,
    min_cells=20,
    top_n=2000,
    show_plot=False,
)
normalized = remote_writable.run_normalization(cell_selection, features)
reduction = remote_writable.run_pca(
    normalized,
    dims=15,
    local_cache="/tmp/scarf_pca_scratch",
)
```

`local_cache` is an execution option.
It does not change artifact identity, so a completed remote-normalized artifact can be reused with a different scratch policy later.
Temporary scratch (`True` or `"auto"` on a remote store) is deleted when the stage ends, on both success and failure.
A path-string cache is kept for reuse or inspection.

Plan local disk for float32 dense blocks roughly as `n_cells × n_features × 4` bytes (about 8 GiB for 1M cells × 2000 HVGs).

## 5. Honest performance expectations

Gene-wise stages and small metadata opens feel remote latency most.
Remote-first analysis is still useful for shared stores; download-then-analyze remains available when you need local disk performance.
Published remote object-store funnel timings and caveats are in {doc}`../concepts/benchmarks`.
Resource planning controls are in {doc}`../concepts/memory_and_execution`.

For custom statistics over mounted graphs or count blocks, followed by a supported selective export, continue with {doc}`custom_analyses`.
