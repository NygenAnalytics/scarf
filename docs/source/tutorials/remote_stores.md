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

SCARF has a special feature, which is to analyze data on remote stores. Additionally, in SCARF, a mounted datastore lets you save an analysis separately from its counts; This means that if several people are sharing a common upstream file, or counts live in a common stroage, everyone can access it and take the analysis onto their own respective branches.

To begin, we will first mount a downloaded dataset, then show how the same feature accepts a remote source. Only the local example is executed on this page (local, downloaded dataset). For a worked example that reads a public remote store, see {doc}`cytebase`; Working on a public remote store could entail downloading data from an atlas, regardless, refer to the referenced document.

## Local example: download, then mount locally

The defining property of a mounted datastore is the separation between its count source and its
writable analysis target. The count source can be a local path or an object-store URI. The target
stores copied metadata plus new artifacts (modifications we make), while count blocks continue to resolve from the source. This way, the original count blocks do not get modified.  Mounting does not by itself mean that either location is remote.

To mount a datastore, we use`mount_datastore`. With this, SCARF copies cell and feature metadata into the target. Mount validates and reads the primary stored `counts`for matrix identity. For RNA based assays, the matching gene matrix of counts must allready be transposed into the format of genes x cells, as if the matrix is present in cells x genes, SCARF will not automatically transpose for it. Thus, the matching source (downloaded dataset) must already have a copy of `countsT` for the analysis. Non-RNA assays don't require this transposed copy of the counts, thus this requirement is only specific to the RNA modality of any sequencing assay here.

Download the example to a temporary directory, then create a separate target for the analysis, that way our initial dataset doesn't get modified. For your own work, use persistent paths and keep the count source available.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import scarf

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
{"count source": str(source_path), "analysis target": str(target_path)}

mounted = scarf.mount_datastore(
    str(source_path),
    at=str(target_path),
    default_assay='RNA',
    nthreads=4,
)
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
clusters are written only to the local target.

```{code-cell} ipython3
mounted_run = mounted.pipeline.run(label="mounted_analysis")
mounted_run.status
```

```{code-cell} ipython3
mounted.plots.embedding(run=mounted_run, color_by="clusters")
```

Opening the target later resolves the source automatically, and the source must remain accessible at the recorded path or URI:

```{code-cell} ipython3
reopened = scarf.DataStore(str(target_path), nthreads=4)
reopened_run = reopened.pipeline.open(run_id=mounted_run.run_id)
reopened_run.status
```

The mount writes down the matrix shape, dtype, and source identity (as to to where the file came from). Reopen later and it checks that record: if the source (initial file) changed underneath, reopening fails instead of analyzing the wrong data. Metadata is copied once at mount time, so later edits to the source's metadata never flow directly into the target

The target also sees the source's (zarr folder) saved results, after its own: labels and embeddings that came with the source can be listed, loaded, traced, and fed into new steps, and a step matching saved provenance reuses the result instead of writing a copy. Everything new still lands in the target, and each pipeline run stays with the store that holds them. Results built on source artifacts need the source around, just like counts do; `python -m scarf.tools.repack_zarr` folds a mount into one self-contained store when you need to hand it off.

## Non-executed object-store templates

These templates are not executed. Replace the example locations and storage options with those for your own bucket where the data is mounted remotely.

### Mount an object-store count source

A mounted analysis can keep shared counts at an object-store URI while writing metadata and artifacts to a local target.

```python
mounted = scarf.mount_datastore(
    's3://shared-bucket/atlas.zarr',
    at='my-analysis.zarr',
    storage_options={'skip_signature': True},
    zarrProfile='fast_local',
)
```

The source must remain available at the recorded URI whenever the target is opened.

### Open a datastore directly

Pass an object-store URI as `zarr_loc` and provider options as `storage_options`. The S3 shape is a example here, and is not a tested public dataset:

```python
import scarf

ds = scarf.DataStore(
    "s3://bucket/path/to/data.zarr",
    zarr_mode="r",
    storage_options={"skip_signature": True},
)
```

For a writable remote store, use the `cloud`descriptor for newly written arrays. Existing arrays retain the layout chosen when they were created. Read credentials from the environment rather than embedding secrets in notebooks:

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
    },
)
```

Google Cloud Storage uses a `gs://` URI; Always pass the provider specific options to your environment based on what your group is using for the analysis, such as application-default credentials on the VM or an explicit token in `storage_options`.

After opening a writable store, use the same analysis calls as for local data.

## Local scratch for reductions

PCA fitting and score projection make multiple passes over normalized expression; The `local_cache` stages those normalized artifacts (changes) to local disk when the *store (folder/initial dataset)that holds it* is remote (object-storage URI or non-local backend). In simple terms, `local_cache` simply determines where the normalized changes itself live, and if that is on a remote store, then stage a copy locally; if the normalized changes already exist locally, then skip resaving an entirely new store.

On a mounted target, the target folder holds the normalized artifacts it writes, while the source folder holds the ones the target reuses. So a local mount stages a normalized artifact it pulls from a remote source, and skips staging for one it wrote itself, even when the counts stream in remotely. Harmony, ANN, and neighbor queries only read already-reduced coordinates, so they never need this scratch space.


| Value                | Behavior                                                                  |
| ---------------------- | --------------------------------------------------------------------------- |
| `"auto"` (default)   | Stage for remote stores; skip for local stores                            |
| `True`               | Stage a remote artifact in temporary scratch, deleted when the stage ends |
| `False`              | No staging; every pass reads the store URI                                |
| `"/path/to/scratch"` | Stage a remote artifact in persistent scratch keyed by artifact ID        |

On this page the mounted target and its source are both local, so their normalized artifacts
need no extra staging. A repeated PCA call can reuse an existing result without rereading
normalized blocks at all.

For a writable remote store opened from the non-executed template above, the same path-string
policy stages normalized blocks and keeps the cache around for inspection or reuse. What follows is
also a non-executed template:

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

`local_cache` only controls where scratch lives during a run, never what gets computed, so an artifact finished under one scratch policy can be reused later under another. Temporary scratch (`True`, or `"auto"` on a remote store) is wiped when the stage ends, success or failure; a path-string cache is kept under the artifact's ID for reuse or inspection. Plan local disk as roughly `n_cells × n_features × 4` bytes of float32 data (about 8 GiB for 1M cells × 2000 HVGs).

## Performance evidence and expectations

Do not judge object-store speed from this page: everything executable here runs locally. A separate fixed workflow measured Scarf against S3-compatible storage in a recorded cloud environment, with timings, memory notes, and limits in {doc}`../concepts/benchmarks`. Those numbers do not compare remote against local storage.

Latency, request costs, credentials, and provider behavior all depend on your environment. When a local workflow fits those constraints better, downloading first is still on the table. Resource planning controls live in {doc}`../concepts/memory_and_execution`.

For custom statistics over mounted graphs or count blocks, followed by a supported selective export, continue with {doc}`custom_analyses`.
