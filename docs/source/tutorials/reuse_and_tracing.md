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

Scarf saves analysis results so you can reuse earlier work and compare new settings.
This guide shows how to reopen a result, change one parameter, and see which steps ran again. Read
{doc}`../concepts/provenance` first for the short mental model.

## Prerequisites

- Scarf installed with the `extra` optional dependencies

## What you will learn

- Reuse normalization, PCA, and ANN when only neighbor `k` changes
- Rebuild reduction and everything downstream when `dims` changes
- Force a new artifact with `invalidate_cache=True`
- Compare upstream lineage for neighbour-count and dimensionality forks

## Dataset

The prepared PBMC store carries a completed example run labelled `docs_default`. Its exact
selection, normalization, PCA, neighbour, and graph refs provide the baseline. An earlier release
built that graph, so the lineage report below marks it stale. This page creates only the parameter
forks needed to demonstrate reuse.

This example was prepared with 15 principal components and 11 neighbours. Those are the settings
we match below to reuse its results; `docs_default` is a saved label, not a promise that every
setting is the API default.

```{code-cell} ipython3
# Open count stores and run Scarf analyses.
import scarf
# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Open the saved analysis used to check reuse.
baseline_run = ds.pipeline.open(label="docs_default")
# Keep the cells used by the saved analysis.
cell_selection = baseline_run["analysis_cell_selection"]
# Keep the variable genes used by the saved analysis.
hvg_ref = baseline_run["highly_variable_features"]
# Inspect the opened store's cells and features.
ds
```

## 1. Open and inspect the baseline chain

The completed run retains every immutable reference needed to keep side comparisons separate.

```{code-cell} ipython3
# Reopen the saved normalized expression.
normalized = baseline_run["normalized"]
# Reopen the saved PCA result.
pca = baseline_run["pca"]
# Keep the saved nearest-neighbor index.
ann = baseline_run["ann_index"]
# Keep the saved 11-neighbor result.
neighbors_k11 = baseline_run["neighbors"]
# Keep the baseline connectivity graph.
graph_k11 = baseline_run["connectivity_map"]
# Inspect the saved chain from normalization to connectivity.
{
    "normalization": normalized,
    "PCA": pca,
    "neighbor index": ann,
    "neighbors": neighbors_k11,
    "graph": graph_k11,
}
```

The catalog can also find results by exact provenance predicates. It returns every match and never
chooses a latest result, so one-item destructuring is an explicit cardinality check:

```{code-cell} ipython3
# Find the exact saved graph from its neighbor input.
[reopened_graph] = ds.list_artifacts(
    from_assay="RNA",
    kind="connectivity_map",
    operation="build_connectivity_map",
    inputs={"neighbors": neighbors_k11},
    complete_only=True,
)
# Check that the catalog returned the baseline graph.
assert reopened_graph == graph_k11

# Inspect the saved graph's operation, parameters, and inputs.
status = ds.inspect_artifact(reopened_graph)
# Inspect the graph operation and the exact inputs it records.
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
# Query the existing index with 15 neighbors per cell.
neighbors_k15 = ds.query_neighbors(ann, k=15)
# Build connectivity for the new neighbor count.
graph_k15 = ds.build_connectivity_map(neighbors_k15)

# Check which results were reused and which changed.
{
    "normalization reused": ds.run_normalization(cell_selection, hvg_ref) == normalized,
    "PCA reused": ds.run_pca(normalized, dims=15) == pca,
    "ANN index reused": ds.build_ann_index(pca) == ann,
    "neighbors recomputed": neighbors_k15 != neighbors_k11,
    "graph recomputed": graph_k15 != graph_k11,
}
```

See {doc}`graph_construction` for how changing `k` affects an analysis. Here, notice that only
neighbours and the graph needed new results.

## 3. Vary `dims`: build a separate branch

A new PCA dimensionality creates a new reduction.
ANN, neighbors, and connectivity that depend on the old reduction are not reused for the new chain.
The earlier results remain available.

```{code-cell} ipython3
# Fit a separate reduction with 20 principal components.
pca_dims20 = ds.run_pca(normalized, dims=20)
# Build a search index from the 20-component reduction.
ann_dims20 = ds.build_ann_index(pca_dims20)
# Find 11 neighbors using the new reduction.
neighbors_dims20 = ds.query_neighbors(ann_dims20, k=11)
# Build the connectivity graph for this reduction.
graph_dims20 = ds.build_connectivity_map(neighbors_dims20)

# Check which results were reused and which changed.
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

Scarf also stops reusing a result on its own when a release changes what the operation computes.
Such a release gives the operation a new revision, and an earlier result of that operation is
computed again, with one log line that names the result it replaced and what changed. The earlier
result remains on disk, and lineage reports mark it as stale.
See {doc}`../developers/operation_revisions`.

For normalization, the call below would write a fresh normalized artifact while retaining the
exact immutable `cell_selection` and `feature_selection` inputs. It is not executed here because a
throwaway duplicate adds no evidence to the lineage figure.

```python
# Force normalization to create a new result with the same inputs.
forced = ds.run_normalization(cell_selection, hvg_ref, invalidate_cache=True)
# Check that forcing recomputation produced a distinct result.
forced != normalized
```

## 5. Compare lineage

Build one read-only report from both neighbour-count branches and the `dims=20` fork.
Shared upstream nodes appear once; the forks show where each branch diverged.

```{code-cell} ipython3
# Trace the shared inputs and differences between all three graphs.
lineage = ds.lineage(
    {
        "k11 graph": graph_k11,
        "k15 graph": graph_k15,
        "dims20 graph": graph_dims20,
    }
)
# Display the shared inputs and diverging analysis branches.
lineage
```

Notebook display renders the Mermaid dependency graph and the artifact details beneath it.
The `k` branches should diverge after the ANN index.
The `dims=20` branch should fork earlier, at PCA, then carry its own ANN, neighbours, and graph.
The prepared graph was built by an earlier release, so the report marks it stale; a graph built
from the same neighbours is a new artifact.

Export the same report when it needs to travel with an analysis.
`to_markdown()` is what notebook display uses. Inspect a short preview before writing or sending
the complete string elsewhere:

```{code-cell} ipython3
# Export the displayed lineage report as Markdown.
lineage_markdown = lineage.to_markdown()
# Preview the opening lines of the exported report.
lineage_markdown.splitlines()[:12]
```

`lineage.to_mermaid()` returns only the diagram source when a tooling pipeline needs that form alone.
