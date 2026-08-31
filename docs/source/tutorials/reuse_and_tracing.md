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

This is the focused guide to artifact mechanics. It shows how to reopen one exact result, branch a
parameter, identify reused work, force recomputation, and compare lineage. Read
{doc}`../concepts/provenance` first for the short mental model.

## Prerequisites

- Scarf installed with the `extra` optional dependencies

## What you will learn

- Reuse normalization, PCA, and ANN when only neighbor `k` changes
- Rebuild reduction and everything downstream when `dims` changes
- Force a new artifact with `invalidate_cache=True`
- Compare upstream lineage for neighbour-count and dimensionality forks

## Dataset

The rebuilt PBMC store carries a completed pipeline run labelled `docs_default`. Its exact
selection, normalization, PCA, neighbour, and graph refs provide the baseline. This page creates
only the parameter forks needed to demonstrate reuse.

```{code-cell} ipython3
import scarf
scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
baseline_run = ds.pipeline.open(label="docs_default")
cell_selection = baseline_run["analysis_cell_selection"]
hvg_ref = baseline_run["highly_variable_features"]
```

## 1. Open and inspect the baseline chain

The completed run retains every immutable reference needed to keep side comparisons separate.

```{code-cell} ipython3
normalized = baseline_run["normalized"]
pca = baseline_run["pca"]
ann = baseline_run["ann_index"]
neighbors_k11 = baseline_run["neighbors"]
graph_k11 = baseline_run["connectivity_map"]
```

The catalog can also find results by exact provenance predicates. It returns every match and never
chooses a latest result, so one-item destructuring is an explicit cardinality check:

```{code-cell} ipython3
[reopened_graph] = ds.list_artifacts(
    from_assay="RNA",
    kind="connectivity_map",
    operation="build_connectivity_map",
    inputs={"neighbors": neighbors_k11},
    complete_only=True,
)
assert reopened_graph == graph_k11

status = ds.inspect_artifact(reopened_graph)
{
    "operation": status.operation,
    "parameters": status.parameters,
    "inputs": status.inputs,
    "complete": status.complete,
}
```

## 2. Vary `k`: reuse upstream

A new neighbor count changes only the neighbors and connectivity {term}`provenance`.
The normalization, PCA, and ANN references are unchanged.

```{code-cell} ipython3
neighbors_k15 = ds.query_neighbors(ann, k=15)
graph_k15 = ds.build_connectivity_map(neighbors_k15)

{
    "normalization reused": ds.run_normalization(cell_selection, hvg_ref) == normalized,
    "PCA reused": ds.run_pca(normalized, dims=15) == pca,
    "ANN index reused": ds.build_ann_index(pca) == ann,
    "neighbors recomputed": neighbors_k15 != neighbors_k11,
    "graph recomputed": graph_k15 != graph_k11,
}
```

The graph-construction guide owns the scientific effect of changing `k`. Here the important result
is where the immutable branch begins.

## 3. Vary `dims`: invalidate downstream

A new PCA dimensionality creates a new reduction.
ANN, neighbors, and connectivity that depend on the old reduction are not reused for the new chain.

```{code-cell} ipython3
pca_dims20 = ds.run_pca(normalized, dims=20)
ann_dims20 = ds.build_ann_index(pca_dims20)
neighbors_dims20 = ds.query_neighbors(ann_dims20, k=11)
graph_dims20 = ds.build_connectivity_map(neighbors_dims20)

{
    "PCA recomputed": pca_dims20 != pca,
    "ANN index recomputed": ann_dims20 != ann,
    "neighbors recomputed": neighbors_dims20 != neighbors_k11,
    "graph recomputed": graph_dims20 != graph_k11,
}
```

## 4. Force recompute

`invalidate_cache=True` skips {term}`reuse` even when the parameters match.
Previously completed artifacts remain on disk.
The new reference has a different id and path.
The operation and parameters stay the same.

For normalization, the call below would write a fresh normalized artifact while retaining the
exact immutable `cell_selection` and `feature_selection` inputs. It is not executed here because a
throwaway duplicate adds no evidence to the lineage figure.

```python
forced = ds.run_normalization(
    cell_selection,
    hvg_ref,
    invalidate_cache=True,
)
status = ds.inspect_artifact(forced)
baseline_status = ds.inspect_artifact(normalized)
baseline_inputs = baseline_status.inputs or {}
forced_inputs = status.inputs or {}

{
    "new artifact": forced != normalized,
    "complete": status.complete,
    "operation": status.operation,
    "path": status.path,
    "baseline path": baseline_status.path,
    "same parameters": status.parameters == baseline_status.parameters,
    "same inputs": forced_inputs == baseline_inputs,
}
```

## 5. Compare lineage

Build one read-only report from both neighbour-count branches and the `dims=20` fork.
Shared upstream nodes appear once; the forks show where each branch diverged.

```{code-cell} ipython3
lineage = ds.lineage(
    {
        "k11 graph": graph_k11,
        "k15 graph": graph_k15,
        "dims20 graph": graph_dims20,
    }
)
lineage
```

Notebook display renders the Mermaid dependency graph and the artifact details beneath it.
The `k` branches should diverge after the ANN index.
The `dims=20` branch should fork earlier, at PCA, then carry its own ANN, neighbours, and graph.

Export the same report when it needs to travel with an analysis.
`to_markdown()` is what notebook display uses. Inspect a short preview before writing or sending
the complete string elsewhere:

```{code-cell} ipython3
lineage_markdown = lineage.to_markdown()
lineage_markdown.splitlines()[:12]
```

`lineage.to_mermaid()` returns only the diagram source when a tooling pipeline needs that form alone.
