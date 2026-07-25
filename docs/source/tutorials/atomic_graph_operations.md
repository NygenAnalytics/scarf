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

Use the atomic chain when you need explicit refs, branching (for example
Harmony), or partial recomputation. For a default RNA analysis, prefer
{ref}`Quick start <quickstart>` and `ds.pipeline.run`. Concepts live in
{doc}`../concepts/graph_and_state`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Familiarity with cell key `I` and HVG feature keys

## What you will learn

- Run normalization → PCA → ANN → neighbors → connectivity as separate calls
- Publish `AssayState` for downstream UMAP and clustering
- Map former `make_graph` arguments onto atomic methods

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
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

Each call returns an `ArtifactRef`. Defaults publish into `AssayState`.

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

`make_graph` is deprecated. It still runs the same chain underneath. Prefer the
table below in new code.

| Former `make_graph` concern | Atomic / pipeline replacement |
|---|---|
| Whole RNA workflow with defaults | `ds.pipeline.run(...)` |
| `feat_key`, `cell_key`, `log_transform`, `renormalize_subset` | `run_normalization(...)` |
| `dims`, `feat_scaling`, `pca_cell_key`, `custom_loadings`, `show_elbow_plot` | `run_pca(...)` (or `run_lsi` for ATAC) |
| `harmonize=True`, `batch_columns`, `harmony_params` | `run_harmony(batch_columns, reduction, ...)` |
| `ann_metric`, `ann_efc`, `ann_ef`, `ann_m`, `ann_parallel` | `build_ann_index(...)` |
| `k` | `query_neighbors(..., k=...)` |
| `local_connectivity`, `bandwidth` | `build_connectivity_map(...)` |
| `n_centroids`, `rand_state` (k-means init) | `build_embedding_initialization(...)` (required before UMAP unless you pass `ini_embed`) |
| `local_cache` | Same name on atomic methods (`"auto"`, `True`, `False`, or a path) |
| `update_keys=True` | `update_state=True` on atomic methods |
| `return_ann_object=True` | Load ANN from the `ann_index` artifact when needed |

Identity parameters (for example `dims`, `k`) decide artifact reuse.
Execution-only options (for example `local_cache`, `batch_size`) do not. See
{doc}`../concepts/provenance` and {doc}`provenance_and_reuse`.

## Next steps

- {doc}`provenance_and_reuse`
- {doc}`../concepts/graph_and_state`
- {doc}`scrna_seq`
