---
description: Reuse upstream artifacts, branch parameters, and inspect lineage reports.
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

(reuse_and_tracing)=

# Provenance and reuse

This tutorial shows when Scarf will {term}`reuse` a completed {term}`artifact` and when it builds a new one.
Read {doc}`../concepts/provenance` for the rules that decide which of the two happens.

## Prerequisites

- Scarf installed with the `extra` optional dependencies

## What you will learn

- Reuse normalization, PCA, and ANN when only neighbor `k` changes
- Rebuild reduction and everything downstream when `dims` changes
- Force a new artifact with `invalidate_cache=True`
- Compare upstream lineage for neighbour-count and dimensionality forks

## Dataset

```{code-cell} ipython3
import scarf

scarf.configure_output(level='WARNING', progress=True)

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
# Remake HVGs after this tutorial's cell filter so lineage does not pull in a
# feature selection that was computed under an earlier cell mask on the store.
ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
```

## 1. Build a baseline chain

Keep `update_state=False` so side comparisons do not replace the current {term}`analysis chain` until you choose a branch.

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

## 2. Vary `k`: reuse upstream

A new neighbor count changes only the neighbors and connectivity {term}`provenance`.
The normalization, PCA, and ANN references are unchanged.

```{code-cell} ipython3
neighbors_k15 = ds.query_neighbors(ann, k=15, update_state=False)
graph_k15 = ds.build_connectivity_map(neighbors_k15, update_state=False)

print('normalization reused:', ds.run_normalization(feat_key='hvgs', update_state=False) == normalized)
print('PCA reused:', ds.run_pca(normalized, dims=15, update_state=False) == pca)
print('ANN index reused:', ds.build_ann_index(pca, update_state=False) == ann)
print('neighbors recomputed:', neighbors_k15 != neighbors_k11)
print('graph recomputed:', graph_k15 != graph_k11)
```

Degree and edge-weight distributions shift with `k` even though the upstream artifacts are identical:

```{code-cell} ipython3
import scarf.plotting as splt

matrix_k11 = ds.load_graph(graph_loc=ds.inspect_artifact(graph_k11).path)
matrix_k15 = ds.load_graph(graph_loc=ds.inspect_artifact(graph_k15).path)
print('edges (nnz):', {'k11': matrix_k11.nnz, 'k15': matrix_k15.nnz})
splt.graph_qc(matrix_k11)
splt.graph_qc(matrix_k15)
```

## 3. Vary `dims`: invalidate downstream

A new PCA dimensionality creates a new reduction.
ANN, neighbors, and connectivity that depend on the old reduction are not reused for the new chain.

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

## 4. Force recompute

`invalidate_cache=True` skips {term}`reuse` even when the parameters match.
Previously completed artifacts remain on disk.
The new reference has a different id and path.
The operation and parameters stay the same.

For normalization, `invalidate_cache` also writes fresh cell and feature selection snapshots and records those new selection artifacts as inputs.
The input roles stay the same (`cell_selection`, `feature_selection`), but the selection artifact ids differ:

```{code-cell} ipython3
forced = ds.run_normalization(
    feat_key='hvgs',
    update_state=False,
    invalidate_cache=True,
)
status = ds.inspect_artifact(forced)
baseline = ds.inspect_artifact(normalized)
baseline_inputs = baseline.inputs or {}
forced_inputs = status.inputs or {}

print('new artifact:', forced != normalized)
print('complete:', status.complete)
print('operation:', status.operation)
print('path:', status.path)
print('baseline path:', baseline.path)
print('same parameters:', status.parameters == baseline.parameters)
print('same input roles:', set(baseline_inputs) == set(forced_inputs))
print('same inputs:', forced_inputs == baseline_inputs)
for name in sorted(set(baseline_inputs) | set(forced_inputs)):
    left = baseline_inputs.get(name)
    right = forced_inputs.get(name)
    if left == right:
        print(f'{name}: same')
        continue
    print(f'{name}: different artifact id')
    print('  baseline:', left)
    print('  forced:', right)
```

## 5. Compare lineage

Build one read-only report from both neighbour-count branches and the `dims=20` fork.
Shared upstream nodes appear once; the forks show where each branch diverged.
Because HVGs were remade after this page's cell filter, the graph should not include an older mito filter that lived on the shared store:

```{code-cell} ipython3
lineage = ds.lineage(
    {
        'k11 graph': graph_k11,
        'k15 graph': graph_k15,
        'dims20 graph': graph_dims20,
    }
)
lineage
```

Notebook display renders the Mermaid dependency graph and the artifact details beneath it.
The `k` branches should diverge after the ANN index.
The `dims=20` branch should fork earlier, at PCA, then carry its own ANN, neighbours, and graph.

Export the same report when it needs to travel with an analysis.
`to_markdown()` is what notebook display uses; showing it here makes that export explicit:

```{code-cell} ipython3
from IPython.display import Markdown

Markdown(lineage.to_markdown())
```

`lineage.to_mermaid()` returns only the diagram source when a tooling pipeline needs that form alone.
