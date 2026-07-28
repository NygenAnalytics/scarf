---
description: Build the neighbourhood graph with atomic Scarf operations and migrate from make_graph.
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

(atomic_graph_operations)=

# Atomic graph operations

{doc}`scrna_seq` runs the graph steps one after another and lets each step pick up
the previous result from the store. That is enough for a single linear analysis.
This page covers the same steps while capturing the reference each one returns,
which is what you need to branch the chain, recompute part of it, or insert
Harmony. Concepts live in {doc}`../concepts/graph_and_state`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Familiarity with cell key `I` and HVG feature keys
- The linear graph walkthrough in {doc}`scrna_seq`

## What you will learn

- Capture the `ArtifactRef` each graph step returns and pass it to the next step
- Publish `AssayState` for downstream UMAP and clustering
- Map former `make_graph` arguments onto atomic methods

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures'],
    highs=[15000, 4000],
    lows=[1000, 500],
    reset_previous=True,
)
if 'I__hvgs' not in ds.RNA.feats.columns:
    ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
```

## Atomic chain

Each call returns an `ArtifactRef` that names the artifact it wrote. Passing that
reference to the next call states the input explicitly instead of relying on
whatever the store currently has selected. Defaults still publish into `AssayState`.

```{code-cell} ipython3
normalized = ds.run_normalization(feat_key='hvgs')
pca = ds.run_pca(normalized, dims=15)
init = ds.build_embedding_initialization(pca, n_centroids=100)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
graph = ds.build_connectivity_map(neighbors)

state = ds.get_assay_state('RNA')
(
    state.normalized,
    state.reduction,
    state.embedding_initialization,
    state.connectivity_map,
)
```

Optional Harmony sits between reduction and ANN:

```python
# coordinates = ds.run_harmony(['batch'], pca)
# ann = ds.build_ann_index(coordinates)
```

Downstream steps then use the published graph and embedding initialization:

```{code-cell} ipython3
ds.run_umap(n_epochs=100, parallel=True)
ds.run_leiden_clustering(resolution=0.5)
```

## Migration from `make_graph`

`DataStore.make_graph` has been removed. Use the table below to migrate an older
workflow. Existing datastores and their stored graphs remain readable.

| Former `make_graph` concern | Atomic / pipeline replacement |
|---|---|
| Whole RNA workflow with defaults | `ds.pipeline.run(...)` |
| `feat_key`, `cell_key`, `log_transform`, `renormalize_subset` | `run_normalization(...)` |
| `dims`, `feat_scaling`, `pca_cell_key`, `custom_loadings`, `show_elbow_plot` | `run_pca(...)` (or `run_lsi` for ATAC) |
| `dims=0` (use normalized features directly) | `run_custom_reduction(np.eye(n_features), normalized, ...)` |
| `harmonize=True`, `batch_columns`, `harmony_params` | `run_harmony(batch_columns, reduction, ...)` |
| `ann_metric`, `ann_efc`, `ann_ef`, `ann_m`, `ann_parallel` | `build_ann_index(...)` |
| `k` | `query_neighbors(..., k=...)` |
| `local_connectivity`, `bandwidth` | `build_connectivity_map(...)` |
| `n_centroids`, `rand_state` (k-means init) | `build_embedding_initialization(...)` (required before UMAP unless you pass `ini_embed`) |
| `local_cache` | Use on `run_pca`, `run_lsi`, or `run_custom_reduction` |
| `update_keys=True` | `update_state=True` on atomic methods |
| `return_ann_object=True` | Load ANN from the `ann_index` artifact when needed |

```{warning}
The removed method derived ANN search breadth from `k` as
`min(100, max(k * 3, 50))` and `ann_m` from `dims` as
`min(max(48, int(dims * 1.5)), 64)`. Direct `build_ann_index` uses fixed
defaults of 50, 50, and 48. Pass all three derived values explicitly when you
need to reproduce an older graph exactly.
```

Artifact identity is operation-specific. `dims` and `k` are identity
parameters, while `local_cache` is a reduction execution option. `batch_size`
is execution-only everywhere except `build_embedding_initialization`, where
mini-batch k-means genuinely depends on it. Leave `batch_size` unset to follow
the stored chunk layout. See {doc}`../concepts/provenance` and
{doc}`provenance_and_reuse`.

For Harmony, pass the `run_harmony` result to `build_ann_index` and
`query_neighbors`. Call `build_mapping_reference` separately when query mapping
needs a Symphony reference. `run_harmony`, `build_ann_index`, and
`query_neighbors` no longer accept `local_cache` because they read persisted
coordinate artifacts.

## Next steps

- {doc}`provenance_and_reuse`
- {doc}`../concepts/graph_and_state`
- {doc}`scrna_seq`
