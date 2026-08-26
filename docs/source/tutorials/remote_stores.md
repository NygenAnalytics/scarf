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

- Open a public demo store on object storage without downloading it
- Open `DataStore` on `s3://` or `gs://` with `storage_options`
- Mount shared count matrices into a separate writable analysis store
- Select the `cloud` storage profile
- Stage normalized data locally for PCA with `local_cache`
- Repack an older store with `scarf.tools.repack_zarr`

## 1. Open the public demo store

The `scarf_docs` Cytebase repository publishes one analyzed store unpacked, so you can open it directly instead of downloading an archive.
`Repository.open_zarr` returns a read-only Zarr group; pass its store to `DataStore`.

```{code-cell} ipython3
import scarf

scarf.configure_output(level="ERROR", progress=True)

repository = scarf.cytebase.connect('scarf_docs')
remote_group = repository.open_zarr('tenx_5K_pbmc_rnaseq/data.zarr')

ds = scarf.DataStore(remote_group.store, zarr_mode='r', nthreads=4)
print(ds)
```

Opening a store over object storage costs many small metadata requests, so expect this step to take a few minutes on a home connection.
Nothing but metadata is read until you touch the counts.
The printed summary lists active cells, assays, and the cell and feature columns already present in the remote store.

The published store predates the current artifact contract.
Counts and literal metadata remain readable, including its UMAP columns and cluster partition, while legacy analysis state fails closed instead of being migrated silently.
The next section mounts those count matrices and builds a current analysis chain in a separate target.

```{code-cell} ipython3
try:
    ds.get_assay_state('RNA')
except scarf.IncompatibleAnalysisStateError as error:
    print('Legacy state rejected:', error.code)

ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_clusters',
)
```

## 2. Prepare and mount read-only counts

Use `mount_datastore` when count matrices must remain in a shared source store, but each analysis needs its own writable store.
Scarf copies cell and feature metadata into the target.
Mount validates and reads primary `counts` for matrix identity.
For RNA sources, the matching gene-major `countsT` copy must already be present on Zarr v3.
It is mounted with `counts` and is not rewritten into the target.
Non-RNA assays have no `countsT`.
New metadata and analysis artifacts are written only to the target.

The public snapshot used here predates the required RNA `countsT` layout, so first repack it structurally into a temporary local count source.
The repack streams from the public URI, creates the transpose, and leaves the remote source unchanged.
Mounting that repacked source into a separate target then prevents copied legacy analysis state from becoming active.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zarr

from scarf.tools.repack_zarr import repack_store

source_uri = (
    "hf://buckets/Nygen/cytebase/scarf_docs/"
    "tenx_5K_pbmc_rnaseq/data.zarr"
)
mount_directory = TemporaryDirectory()
repacked_path = Path(mount_directory.name) / 'counts.zarr'
target_path = Path(mount_directory.name) / 'analysis.zarr'

repack_store(
    source_uri,
    str(repacked_path),
    storage_options={"token": False},
    nthreads=2,
)
```

The target path must not already exist:

```{code-cell} ipython3
mounted = scarf.mount_datastore(
    str(repacked_path),
    at=str(target_path),
    default_assay='RNA',
    nthreads=4,
)

matrix_source = zarr.open_group(str(target_path), mode='r').attrs['matrixSource']
print('Target path:', target_path)
print('Target exists:', target_path.exists())
print('Original source URI:', source_uri)
print('Mounted count source:', matrix_source['location'])
print('Mounted assays:', sorted(matrix_source['assays']))
```

Counts and RNA `countsT` stay in the repacked count source.
Cell and feature metadata are copied once, while new analysis artifacts are written to the target.
The printed `matrixSource` record is what later reopen uses to resolve the repacked counts.

```{mermaid}
flowchart LR
    source["Read-only remote source<br/>legacy counts"]
    repack["Temporary structural repack<br/>counts and RNA countsT"]
    mount["Mounted DataStore"]
    target["Writable target<br/>metadata and new artifacts"]
    source -->|stream once| repack
    repack -->|stream count blocks| mount
    mount -->|write analysis results| target
```

The target assay has no physical count array, while `rawData` exposes the complete remote matrix:

```{code-cell} ipython3
target_root = zarr.open_group(str(target_path), mode='r')

print('Counts stored in target:', 'counts' in target_root['RNA'])
print('Mounted shape:', mounted.RNA.rawData.shape)
```

Run a complete graph and plotting checkpoint through the mount.
Count blocks are read from the structurally repacked source.
Normalized data, reductions, neighbours, graph, UMAP, and clusters are written only to the local target.
Because that target is a local path, `local_cache` staging is skipped here; Section 4 makes the remote-only policy explicit and gives a writable-remote template.

```{code-cell} ipython3
mounted_features = mounted.mark_hvgs(
    min_cells=20,
    top_n=500,
    show_plot=False,
    label='mounted_hvgs',
)
normalized = mounted.run_normalization(features=mounted_features)
pca = mounted.run_pca(normalized, dims=15)

mounted.build_embedding_initialization(pca)
ann_index = mounted.build_ann_index(pca)
neighbors = mounted.query_neighbors(ann_index, k=11)
graph = mounted.build_connectivity_map(neighbors)
mounted.run_umap(
    graph=graph,
    n_epochs=100,
    spread=5,
    min_dist=1,
    parallel=True,
    label="mounted_UMAP",
)
mounted.run_leiden_clustering(
    graph=graph,
    resolution=0.5,
    label="mounted_clusters",
)
```

```{code-cell} ipython3
mounted.plots.embedding(
    layout_key="RNA_mounted_UMAP",
    color_by="RNA_mounted_clusters",
)
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
same_counts = np.array_equal(
    reopened.RNA.rawData[:20, :20].compute(),
    mounted.RNA.rawData[:20, :20].compute(),
)
print('Counts still resolve:', same_counts)
print('Normalization complete:', reopened.inspect_artifact(normalized).complete)
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

ds = scarf.DataStore(
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

After open, call the same analysis APIs as on a local store: `ds.pipeline.run(...)` or the individual graph-construction methods.

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
from pathlib import Path
from tempfile import TemporaryDirectory

scratch_directory = TemporaryDirectory()
scratch_dir = Path(scratch_directory.name) / 'pca_scratch'
pca_without_staging = mounted.run_pca(
    normalized,
    dims=15,
    local_cache=str(scratch_dir),
    update_state=False,
)
staged_bytes = sum(
    path.stat().st_size for path in scratch_dir.rglob('*') if path.is_file()
)
print('Local cache path:', scratch_dir)
print('Staged normalized bytes:', staged_bytes)
print('Local cache present after PCA:', scratch_dir.exists())
print('Reduction reused:', pca_without_staging == pca)
```

On your own writable remote store, the same path-string policy stages normalized blocks and keeps the cache for inspection or reuse:

```python
features = remote_writable.mark_hvgs(
    min_cells=20,
    top_n=2000,
    show_plot=False,
)
normalized = remote_writable.run_normalization(features=features)
reduction = remote_writable.run_pca(
    normalized,
    dims=15,
    local_cache="/tmp/scarf_pca_scratch",
)
remote_writable.build_embedding_initialization(reduction)
ann_index = remote_writable.build_ann_index(reduction)
neighbors = remote_writable.query_neighbors(ann_index, k=11)
remote_writable.build_connectivity_map(neighbors)
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

## 6. Repack older stores

New Scarf writers emit Zarr v3.
An RNA store that predates the paired `counts` / `countsT` layout, or that is still Zarr v2, will not open as an RNA assay.
Repack writes a new store with the current layout and storage profile, locally or on object storage:

```bash
uv run python -m scarf.tools.repack_zarr \
  s3://bucket/input.zarr s3://bucket/output.zarr \
  --profile cloud \
  --mem-budget 8G \
  --storage-options '{"skip_signature": true}'
```

Paths stay as URIs (do not pass them through `pathlib.Path`).
Use `--storage-options` for backend credentials or public reads (`skip_signature` for anonymous S3/GCS).
Point `DataStore` at the output URI afterward.
Repacking rewrites physical layout; it is not an analysis step.
After a rewrite, recompute HVG, normalization, PCA, graph, and marker results rather than resuming them from the input store.

For custom statistics over mounted graphs or count blocks, followed by a supported selective export, continue with {doc}`custom_analyses`.
