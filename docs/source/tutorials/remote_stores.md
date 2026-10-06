---
description: Distinguish direct object-store access from mounted analysis targets and local scratch.
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

# Remote stores and mounted analysis targets

A mounted datastore lets you save an analysis separately from its counts. This is useful when
several people share a source or when counts live in object storage.

We will first try a mount with a downloaded dataset, then show how the same call accepts a remote
source. Only the local example is executed on this page. For a worked example that reads a public
remote store, see {doc}`cytebase`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Credentials for your bucket when adapting the non-executed templates
- Enough local disk for optional `local_cache` scratch during reduction

## What you will learn

- Distinguish direct object-store access from the local mechanics executed on this page
- Mount a count source into a separate writable analysis target
- Reopen a durable pipeline run from its mounted target
- Adapt non-executed `s3://` and `gs://` templates for your environment
- Stage normalized data locally for PCA with `local_cache`

## 1. Executed example: download, then mount locally

The defining property of a mounted datastore is the separation between its count source and its
writable analysis target. The count source can be a local path or an object-store URI. The target
stores copied metadata plus new artifacts, while count blocks continue to resolve from the source.
Mounting does not by itself mean that either location is remote.

Use `mount_datastore` when count matrices must remain in a shared source store, but each analysis
needs its own writable target.
Scarf copies cell and feature metadata into the target.
Mount validates and reads primary `counts` for matrix identity.
For RNA sources, the matching gene-major `countsT` copy must already be present on Zarr v3.
It is mounted with `counts` and is not rewritten into the target.
Non-RNA assays have no `countsT`.
New metadata and analysis artifacts are written only to the target.

Download the example to a temporary directory, then create a separate target for the analysis.
For your own work, use persistent paths and keep the count source available.

```{code-cell} ipython3
# Manage local file and directory paths.
from pathlib import Path
# Create temporary directories for this example.
from tempfile import TemporaryDirectory

# Open count stores and run Scarf analyses.
import scarf

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="ERROR", progress=False)
# Connect to the public example-data repository.
repository = scarf.cytebase.connect("scarf_docs")
# Keep the local source and mounted target in one temporary directory.
mount_directory = TemporaryDirectory()
# Download a local count source for the mounting example.
staged_dataset = repository.download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination=mount_directory.name,
    zarr=True,
)
# Locate the downloaded count store.
source_path = staged_dataset / 'data.zarr'
# Choose a separate path for writable analysis results.
target_path = Path(mount_directory.name) / 'analysis.zarr'
# Check that the count source and analysis target have distinct paths.
{"count source": str(source_path), "analysis target": str(target_path)}
```

The target path must be absent or an empty directory, and must not be, contain, or lie inside the
source, or lie inside any other store:

```{code-cell} ipython3
# Mount the count source into a separate writable target.
mounted = scarf.mount_datastore(
    str(source_path),
    at=str(target_path),
    default_assay='RNA',
    nthreads=4,
)
# Inspect the opened store's cells and features.
mounted
```

Counts and RNA `countsT` stay in the downloaded count source.
Cell and feature metadata are copied once, while new analysis artifacts are written to the target.

```{mermaid}
flowchart LR
    source["Dataset archive"]
    staged["Current local count source<br/>counts, RNA countsT, metadata"]
    target["Separate local analysis target<br/>metadata and new artifacts"]
    source -->|download once| staged
    staged -->|mount count blocks| target
```

Run the RNA pipeline with its defaults through the local mount. Count blocks are read from the separate
local source, while the run record and its normalized data, reductions, graph, UMAP, and
clusters are written only to the local target. Because the target and its source are both local,
`local_cache` staging is skipped here; Section 3 makes that policy explicit.

```{code-cell} ipython3
# Run the RNA analysis through the mounted store.
mounted_run = mounted.pipeline.run(label="mounted_analysis")
# Confirm that the saved pipeline run completed.
mounted_run.status
```

```{code-cell} ipython3
# Show the clusters produced by the mounted analysis.
mounted.plots.embedding(run=mounted_run, color_by="clusters")
```

Opening the target later resolves the source automatically.
The source must remain accessible at the recorded path or URI:

```{code-cell} ipython3
# Reopen the mounted target from its saved path.
reopened = scarf.DataStore(str(target_path), nthreads=4)
# Open the exact run saved in the mounted target.
reopened_run = reopened.pipeline.open(run_id=mounted_run.run_id)
# Confirm that the saved pipeline run completed.
reopened_run.status
```

The mount records matrix shape, dtype, and source identity. Reopening fails if the source no
longer matches that identity. Metadata is copied at mount time, so later source metadata changes
are not synchronized into the target.

The target also resolves the source's artifacts read only, after its own. Results already saved in
the source, including labels and embeddings imported with it, can be listed, loaded, traced, and
used as inputs on the mount, and a step whose provenance matches a saved result reuses it instead
of writing a copy. New artifacts are still written only to the target, and pipeline runs and their
labels stay with the store that holds them. Results that reuse source artifacts need the source,
as counts do; `python -m scarf.tools.repack_zarr` copies a mount into a self-contained store.

## 2. Non-executed object-store templates

These templates are not executed. Replace the example locations and storage options with those
for your own bucket.

### Mount an object-store count source

A mounted analysis can keep shared counts at an object-store URI while writing metadata and
artifacts to a local target. This is the remote form of the source/target separation demonstrated
locally in Section 1:

```python
# Mount the count source into a separate writable target.
mounted = scarf.mount_datastore(
    's3://shared-bucket/atlas.zarr',
    at='my-analysis.zarr',
    storage_options={'skip_signature': True},
    zarrProfile='fast_local',
)
```

The source must remain available at the recorded URI whenever the target is opened.

### Open a datastore directly

Pass an object-store URI as `zarr_loc` and provider options as `storage_options`. This anonymous
S3 shape is a template, not a tested public dataset:

```python
# Open count stores and run Scarf analyses.
import scarf

# Open the count store for this analysis.
ds = scarf.DataStore(
    "s3://bucket/path/to/data.zarr",
    zarr_mode="r",
    storage_options={"skip_signature": True},
)
```

For a writable remote store, use the `cloud` profile for newly written arrays. Existing arrays
retain the layout chosen when they were created. Read credentials from the environment rather
than embedding secrets in notebooks:

```python
# Read object-store credentials from the environment.
import os
# Open count stores and run Scarf analyses.
import scarf

# Open a writable remote store with credentials from the environment.
remote_writable = scarf.DataStore(
    "s3://my-bucket/project/data.zarr",
    zarr_mode="r+",
    zarrProfile="cloud",
    storage_options={
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        # "endpoint": "https://...",  # S3-compatible endpoints
    },
)
```

Google Cloud Storage uses a `gs://` URI.
Pass the provider options your environment already uses for obstore or fsspec, such as
application-default credentials on the VM or an explicit token in `storage_options`.

After opening a writable store, use the same analysis calls as for local data.

## 3. Local scratch for reductions

PCA fitting and score projection make multiple passes over normalized expression.
`local_cache` stages that normalized artifact to local disk when the *store that holds it* is remote (object-storage URI or non-local backend).
It does not key off whether counts alone are remote.
On a mounted target, the target holds the normalized artifacts it writes and the source holds the ones the target reuses from it.
A local mount therefore stages a normalized artifact that it reuses from a remote source, and skips staging for one that it wrote, even when `counts` stream from a remote source.
Harmony, ANN, and neighbor queries read persisted reduced coordinates and do not use normalized-expression scratch.

| Value | Behavior |
|---|---|
| `"auto"` (default) | Stage for remote stores; skip for local stores |
| `True` | Stage a remote artifact in temporary scratch, deleted when the stage ends |
| `False` | No staging; every pass reads the store URI |
| `"/path/to/scratch"` | Stage a remote artifact in persistent scratch keyed by artifact ID |

The mounted target and its source in this page are both local, so their normalized artifacts
need no extra staging. A repeated PCA call can also reuse an existing result before reading
normalized blocks at all.

For a writable remote store opened from the non-executed template above, the same path-string
policy stages normalized blocks and keeps the cache for inspection or reuse. The following is
also a non-executed template:

```python
# Freeze the current active cells before selecting genes.
cell_selection = remote_writable.snapshot_cell_selection(cell_key="I")
# Select variable genes for the remote reduction.
features = remote_writable.select_hvgs(
    cell_selection,
    min_cells=20,
    top_n=2000,
    show_plot=False,
)
# Normalize counts over the selected features.
normalized = remote_writable.run_normalization(cell_selection, features)
# Fit PCA with persistent local scratch for remote normalized data.
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

## 4. Performance evidence and expectations

Do not infer object-store performance from this page's executable local mount. A separate fixed
workflow measured Scarf against S3-compatible object storage in a recorded cloud environment; its
timings, memory observations, and limits are in {doc}`../concepts/benchmarks`. Those measurements
do not compare remote with local storage.

Object-store latency, request costs, credentials, and provider behavior remain environment
specific. Downloading first is still available when a local workflow better fits those constraints.
Resource planning controls are in {doc}`../concepts/memory_and_execution`.

For custom statistics over mounted graphs or count blocks, followed by a supported selective export, continue with {doc}`custom_analyses`.
