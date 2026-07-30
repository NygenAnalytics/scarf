---
description: Reuse upstream artifacts, invalidate downstream steps, and walk provenance inputs.
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

(provenance_and_reuse)=

# Provenance and reuse

This tutorial shows when Scarf reuses complete artifacts and when it builds new
ones. Read {doc}`../concepts/provenance` for the identity rules.

## Prerequisites

- Scarf installed with the `extra` optional dependencies

## What you will learn

- Reuse normalization, PCA, and ANN when only neighbor `k` changes
- Rebuild reduction and everything downstream when `dims` changes
- Force a new artifact with `invalidate_cache=True`
- Walk `inputs` from a connectivity artifact back to selections

## Dataset

```{code-cell} ipython3
import scarf

scarf.configure_output(level='WARNING', progress=False)

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

## Build a baseline chain

Keep `update_state=False` so side comparisons do not move the published state
until you choose a winner.

```{code-cell} ipython3
normalized = ds.run_normalization(
    feat_key='hvgs',
    update_state=False,
)
pca = ds.run_pca(normalized, dims=15, update_state=False)
ann = ds.build_ann_index(pca, update_state=False)
neighbors_k11 = ds.query_neighbors(ann, k=11, update_state=False)
graph_k11 = ds.build_connectivity_map(neighbors_k11, update_state=False)
```

## Vary `k`: reuse upstream

A new neighbor count changes only neighbors and connectivity provenance.
Normalization, PCA, and ANN refs stay the same object.

```{code-cell} ipython3
neighbors_k15 = ds.query_neighbors(ann, k=15, update_state=False)
graph_k15 = ds.build_connectivity_map(neighbors_k15, update_state=False)

print('normalization reused:', ds.run_normalization(feat_key='hvgs', update_state=False) == normalized)
print('PCA reused:', ds.run_pca(normalized, dims=15, update_state=False) == pca)
print('ANN index reused:', ds.build_ann_index(pca, update_state=False) == ann)
print('neighbors recomputed:', neighbors_k15 != neighbors_k11)
print('graph recomputed:', graph_k15 != graph_k11)
```

## Vary `dims`: invalidate downstream

A new PCA dimensionality creates a new reduction. ANN, neighbors, and
connectivity that depend on the old reduction are not reused for the new chain.

```{code-cell} ipython3
pca_dims20 = ds.run_pca(normalized, dims=20, update_state=False)
ann_dims20 = ds.build_ann_index(pca_dims20, update_state=False)
neighbors_dims20 = ds.query_neighbors(ann_dims20, k=11, update_state=False)
graph_dims20 = ds.build_connectivity_map(neighbors_dims20, update_state=False)

print('PCA recomputed:', pca_dims20 != pca)
print('ANN index recomputed:', ann_dims20 != ann)
print('neighbors recomputed:', neighbors_dims20 != neighbors_k11)
print('graph recomputed:', graph_dims20 != graph_k11)
print('normalization reused:', ds.run_normalization(feat_key='hvgs', update_state=False) == normalized)
```

## Force recompute

`invalidate_cache=True` skips provenance reuse even when parameters match.
Previous complete artifacts remain on disk.

```{code-cell} ipython3
forced = ds.run_normalization(
    feat_key='hvgs',
    update_state=False,
    invalidate_cache=True,
)
status = ds.inspect_artifact(forced)
print('new artifact:', forced != normalized)
print('complete:', status.complete)
print('operation:', status.operation)
```

## Walk inputs

There is no package lineage helper yet. Recurse through `status.inputs`:

```{code-cell} ipython3
def walk_inputs(store, ref, prefix=''):
    status = store.inspect_artifact(ref)
    print(f'{prefix}{ref.kind} ({status.operation})')
    for name, value in (status.inputs or {}).items():
        if isinstance(value, dict) and value.get('type') == 'artifact':
            walk_inputs(
                store,
                scarf.ArtifactRef.from_dict(value),
                prefix=prefix + '  ',
            )
        else:
            print(f'{prefix}  {name}: {value}')

walk_inputs(ds, graph_k11)
```

## Next steps

- {doc}`../concepts/provenance`
- {doc}`atomic_graph_operations`
- {doc}`data_organization`
