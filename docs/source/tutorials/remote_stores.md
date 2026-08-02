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

Scarf can open a Zarr store on object storage and run analysis without first
copying the full matrix to local disk. Counts stream in tiles sized from your
memory budget. Read {doc}`../concepts/scale_and_memory` for measured local versus
remote overhead.

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

## Open the public demo store

The `scarf_docs` Cytebase repository publishes one analyzed store unpacked, so you
can open it directly instead of downloading an archive. `Repository.open_zarr`
returns a read-only Zarr group; pass its store to `DataStore`.

```{code-cell} ipython3
import scarf

scarf.configure_output(level="ERROR", progress=True)

repository = scarf.cytebase.connect('scarf_docs')
remote_group = repository.open_zarr('tenx_5K_pbmc_rnaseq/data.zarr')

ds = scarf.DataStore(remote_group.store, zarr_mode='r', nthreads=4)
```

Opening a store over object storage costs many small metadata requests, so expect
this step to take a few minutes on a home connection. Nothing but metadata is read
until you touch the counts.

The store already carries a full analysis, so its {term}`artifacts <artifact>`
and current {term}`analysis chain` are readable straight away:

```{code-cell} ipython3
state = ds.get_assay_state('RNA')
print('Reduction available:', state.reduction is not None)
print('Graph available:', state.connectivity_map is not None)
```

## Mount read-only counts into a writable store

Use `mount_datastore` when count matrices must remain in a shared source store,
but each analysis needs its own writable store. Scarf copies cell and feature
metadata into the target. It reads `counts` and `countsT` from the source and
writes new metadata and analysis artifacts only to the target.

The mounted target below lives in a temporary local directory, but its count
source is the public remote URI. No count archive is downloaded first.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zarr

source_uri = (
    "hf://buckets/Nygen/cytebase/scarf_docs/"
    "tenx_5K_pbmc_rnaseq/data.zarr"
)
mount_directory = TemporaryDirectory()
target_path = Path(mount_directory.name) / 'analysis.zarr'
```

The target path must not already exist:

```{code-cell} ipython3
mounted = scarf.mount_datastore(
    source_uri,
    at=str(target_path),
    storage_options={"token": False},
    nthreads=4,
)
```

Counts and optional `countsT` stay in the source. Cell and feature metadata are
copied once, while new analysis artifacts are written to the target.

```{mermaid}
flowchart LR
    source["Read-only remote source<br/>counts and countsT"]
    mount["Mounted DataStore"]
    target["Writable target<br/>metadata and new artifacts"]
    source -->|stream count blocks| mount
    mount -->|write analysis results| target
```

The target assay has no physical count array, while `rawData` exposes the
complete remote matrix:

```{code-cell} ipython3
target_root = zarr.open_group(str(target_path), mode='r')

print('Counts stored in target:', 'counts' in target_root['RNA'])
print('Mounted shape:', mounted.RNA.rawData.shape)
```

Run a complete graph and plotting checkpoint through the mount. Count blocks are
read from the remote source; reductions, neighbours, graph, UMAP, and clusters
are stored locally.

```{code-cell} ipython3
normalized = mounted.run_normalization(feat_key='hvgs')
pca = mounted.run_pca(normalized, dims=15, local_cache=True)
mounted.build_embedding_initialization(pca)
mounted.build_ann_index(pca)
mounted.query_neighbors(k=11)
mounted.build_connectivity_map()
mounted.run_umap(
    n_epochs=100,
    spread=5,
    min_dist=1,
    parallel=True,
    label="mounted_UMAP",
)
mounted.run_leiden_clustering(
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

The populated embedding demonstrates that mounted counts behave like a normal
datastore input. The target remains much smaller than a dense local copy of the
source matrix:

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

Opening the target later resolves the source automatically. The source must
remain accessible at the recorded path or URI:

```{code-cell} ipython3
reopened = scarf.DataStore(
    str(target_path),
    nthreads=4,
    storage_options={"token": False},
)
same_counts = np.array_equal(
    reopened.RNA.rawData[:20, :20].compute(),
    mounted.RNA.rawData[:20, :20].compute(),
)
print('Counts still resolve:', same_counts)
print('Normalization complete:', reopened.inspect_artifact(normalized).complete)
```

For an S3 or GCS source, pass the URI directly. The target can remain local:

```python
mounted = scarf.mount_datastore(
    's3://shared-bucket/atlas.zarr',
    at='my-analysis.zarr',
    storage_options={'skip_signature': True},
    zarrProfile='fast_local',
)
```

The mount records matrix shape, dtype, and source identity. Reopening fails if
the source no longer matches that identity. Metadata is copied at mount time,
so later source metadata changes are not synchronized into the target.

## Open your own remote store

Pass the URI as `zarr_loc` and any fsspec/obstore options as `storage_options`.
Use `zarrProfile="cloud"` when writing new arrays so they use the cloud
compression profile. Existing arrays keep the physical layout chosen when they
were created.

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

Google Cloud Storage uses a `gs://` URI. Pass the provider options your
environment already uses for obstore/fsspec (for example application-default
credentials on the VM, or an explicit token in `storage_options`).

After open, call the same analysis APIs as on a local store:
`ds.pipeline.run(...)` or the individual graph-construction methods.

## Local scratch for reductions

PCA fitting and score projection make multiple passes over normalized
expression. Against object storage, set `local_cache` on the reduction so those
passes hit local disk. Harmony, ANN, and neighbor queries read persisted
reduced coordinates and do not use normalized-expression scratch.

| Value | Behavior |
|---|---|
| `"auto"` (default) | Stage for remote stores; skip for local stores |
| `True` | Temporary scratch directory, removed after success |
| `False` | No staging; every pass reads the store URI |
| `"/path/to/scratch"` | Persistent scratch keyed by artifact ID |

```python
normalized = ds.run_normalization(feat_key="hvgs")
reduction = ds.run_pca(normalized, dims=15, local_cache=True)
ds.build_embedding_initialization(reduction)
ann_index = ds.build_ann_index(reduction)
neighbors = ds.query_neighbors(ann_index, k=11)
ds.build_connectivity_map(neighbors)
```

`local_cache` is an execution option. It does not change artifact identity, so
a completed remote-normalized artifact can be reused with a different scratch
policy later. Temporary scratch is deleted after a successful stage; failed
runs may leave a directory behind for debugging.

Plan local disk for float32 dense blocks roughly as
`n_cells × n_features × 4` bytes (about 8 GiB for 1M cells × 2000 HVGs).

## Honest performance expectations

At 100k cells, a matched countsT funnel was about **1.75× slower** on remote
object storage than on ephemeral local disk (~735 s vs ~421 s). Gene-wise
stages and small metadata opens feel remote latency most. Remote-first analysis
is still useful for shared stores; download-then-analyze remains available when
you need the local ceiling. See {doc}`../concepts/scale_and_memory` for the
resource envelope and benchmark caveats.

## Repack older stores

New Scarf writers emit Zarr v3 with profile-specific sharding. Repack a local
or remote store when you want cloud-oriented layout without re-importing
counts:

```bash
uv run python -m scarf.tools.repack_zarr \
  s3://bucket/input.zarr s3://bucket/output.zarr \
  --profile cloud \
  --mem-budget 8G \
  --storage-options '{"skip_signature": true}'
```

Paths stay as URIs (do not pass them through `pathlib.Path`). Use
`--storage-options` for backend credentials or public reads (`skip_signature`
for anonymous S3/GCS). Point `DataStore` at the output URI afterward. Repacking
rewrites layout for a storage profile; it is not an analysis step.

For custom statistics over mounted graphs or count blocks, followed by a
supported selective export, continue with {doc}`custom_analyses`.
