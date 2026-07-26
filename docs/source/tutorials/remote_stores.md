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

# Remote Zarr stores

Scarf can open a Zarr store on object storage and run analysis without first
copying the full matrix to local disk. Counts stream in tiles sized from your
memory budget. Read {doc}`../concepts/scale_and_memory` for measured local versus
remote overhead.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Credentials for your bucket (except for anonymous public reads)
- Enough local disk for optional `local_cache` scratch on multi-pass graph steps

## What you will learn

- Open a public demo store on object storage without downloading it
- Open `DataStore` on `s3://` or `gs://` with `storage_options`
- Mount shared count matrices into a separate writable analysis store
- Select the `cloud` storage profile
- Stage normalized data locally with `local_cache`
- Repack an older store with `scarf.tools.repack_zarr`

## Open the public demo store

The `scarf_docs` Cytebase repository publishes one analyzed store unpacked, so you
can open it directly instead of downloading an archive. `Repository.open_zarr`
returns a read-only Zarr group; pass its store to `DataStore`.

```{code-cell} ipython3
import scarf

repository = scarf.cytebase.connect('scarf_docs')
remote_group = repository.open_zarr('tenx_5K_pbmc_rnaseq/data.zarr')

ds = scarf.DataStore(remote_group.store, zarr_mode='r', nthreads=4)
ds
```

Opening a store over object storage costs many small metadata requests, so expect
this step to take a few minutes on a home connection. Nothing but metadata is read
until you touch the counts.

The store already carries a full analysis, so its artifacts and published state are
readable straight away:

```{code-cell} ipython3
state = ds.get_assay_state('RNA')
print('Reduction published:', state.reduction is not None)
print('Graph published:', state.connectivity_map is not None)
```

## Mount read-only counts into a writable store

Use `mount_datastore` when count matrices must remain in a shared source store,
but each analysis needs its own writable store. Scarf copies cell and feature
metadata into the target. It reads `counts` and `countsT` from the source and
writes new metadata and analysis artifacts only to the target.

This executable example downloads the prepared demo store so it has a source
path accepted by `mount_datastore`. The mounted target lives in a temporary
directory for the lifetime of the notebook.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zarr

mounted_dataset = repository.download_dataset(
    name='tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
source_path = Path(mounted_dataset) / 'data.zarr'
mount_directory = TemporaryDirectory()
target_path = Path(mount_directory.name) / 'analysis.zarr'
```

The target path must not already exist:

```{code-cell} ipython3
mounted = scarf.mount_datastore(
    str(source_path),
    at=str(target_path),
    nthreads=4,
)
mounted
```

The target assay has no count array. `rawData` still exposes the complete source
matrix:

```{code-cell} ipython3
source_root = zarr.open_group(str(source_path), mode='r')
source_counts = source_root['RNA/counts']

print('Counts stored in target:', 'counts' in mounted.z['RNA'])
print('Source shape:', source_counts.shape)
print('Mounted shape:', mounted.RNA.rawData.shape)
```

Metadata changes and analysis methods persist their output in the target. This
example adds a cell column and normalizes the existing `hvgs` feature selection:

```{code-cell} ipython3
mounted.cells.insert(
    'analysisCopy',
    np.ones(
        len(mounted.cells.active_index('I')),
        dtype=bool,
    ),
)

normalization = mounted.run_normalization(feat_key='hvgs')
mounted.inspect_artifact(normalization).complete
```

The new cell column is present only in the target:

```{code-cell} ipython3
print(
    'Cell column in source:',
    'analysisCopy' in source_root['cellData'],
)
print(
    'Cell column in target:',
    'analysisCopy' in mounted.z['cellData'],
)
```

Opening the target later resolves the source automatically. The source must
remain accessible at the recorded path or URI:

```{code-cell} ipython3
reopened = scarf.DataStore(str(target_path), nthreads=4)
same_counts = np.array_equal(
    reopened.RNA.rawData[:20, :20].compute(),
    source_counts[:20, :20],
)
print('Counts still match:', same_counts)
print('Normalization complete:', reopened.inspect_artifact(normalization).complete)
```

For an S3 or GCS source, pass the URI directly. The target can remain local:

```python
mounted = scarf.mount_datastore(
    's3://shared-bucket/atlas.zarr',
    at='my-analysis.zarr',
    storage_options={'anon': True},
    zarrProfile='fast_local',
)
```

The mount records matrix shape, dtype, and source identity. Reopening fails if
the source no longer matches that identity. Metadata is copied at mount time,
so later source metadata changes are not synchronized into the target.

## Open your own remote store

Pass the URI as `zarr_loc` and any fsspec/obstore options as `storage_options`.
Use `zarrProfile="cloud"` so count chunks match object-store friendly sizes.

Anonymous read-only example against your own public bucket:

```python
import scarf

ds = scarf.DataStore(
    "s3://example-bucket/path/to/data.zarr",
    zarr_mode="r",
    zarrProfile="cloud",
    storage_options={"anon": True},
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
`ds.pipeline.run(...)` or the atomic graph methods.

## Local scratch for graph stages

Multi-pass PCA, ANN, and neighbor queries re-read the normalized matrix.
Against object storage, set `local_cache` so those passes hit local disk:

| Value | Behavior |
|---|---|
| `"auto"` (default) | Stage for remote stores; skip for local stores |
| `True` | Temporary scratch directory, removed after success |
| `False` | No staging; every pass reads the store URI |
| `"/path/to/scratch"` | Persistent scratch keyed by artifact ID |

```python
ds.run_normalization(feat_key="hvgs", local_cache=True)
ds.run_pca(dims=15, local_cache=True)
ds.build_embedding_initialization()
ds.build_ann_index(local_cache=True)
ds.query_neighbors(k=11, local_cache=True)
ds.build_connectivity_map()
```

`local_cache` is an execution option. It does not change artifact identity, so
a completed remote-normalized artifact can be reused with a different scratch
policy later. Temporary scratch is deleted after a successful stage; failed
runs may leave a directory behind for debugging.

Plan local disk for float32 dense blocks roughly as
`n_cells × n_features × 4` bytes (about 8 GiB for 1M cells × 2000 HVGs).

## Honest performance expectations

At 100k cells, a matched countsT funnel was about **1.75× slower** on remote
object storage than on ephemeral local disk in Scarf's profiling campaign
(~735 s vs ~421 s). Gene-wise stages and small metadata opens feel remote
latency most. Remote-first analysis is still the product path for shared
stores; download-then-analyze remains available when you want the local ceiling.

## Repack older stores

New Scarf writers emit Zarr v3 with profile-specific sharding. Repack a local
or remote store when you want cloud-oriented layout without re-importing
counts:

```bash
uv run python -m scarf.tools.repack_zarr \
  input.zarr output.zarr --profile cloud
```

Point `DataStore` at the output URI afterward. Repacking is a layout migration
tool, not an analysis step.

## Next steps

- {doc}`../concepts/scale_and_memory`
- {doc}`data_organization`
- {doc}`atomic_graph_operations`
- {doc}`../installation`
