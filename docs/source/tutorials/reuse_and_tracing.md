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

Scarf saves analysis results so you can reuse earlier work and compare new settings even if you return to the analysis months later. This guide shows how to reopen a result, change one parameter, and see which steps ran again.

## Dataset

The prepared PBMC store carries a completed example run labelled `docs_default`. Its exact selection, normalization, PCA, neighbour, and graph refs provide the baseline for what we analyze and compare against today. This page creates only the parameter forks needed to demonstrate reuse.

This example was prepared with 15 principal components and 11 neighbours. Those are the settings
we match below to reuse its results.

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
ds
```

## Open and inspect the baseline chain

The completed run retains every reference needed to keep side comparisons separate.

```{code-cell} ipython3
normalized = baseline_run["normalized"]
pca = baseline_run["pca"]
ann = baseline_run["ann_index"]
neighbors_k11 = baseline_run["neighbors"]
graph_k11 = baseline_run["connectivity_map"]
{
    "normalization": normalized,
    "PCA": pca,
    "neighbor index": ann,
    "neighbors": neighbors_k11,
    "graph": graph_k11,
}
```

The catalog finds results by exact inputs and returns every match instead of guessing a latest one, so unpacking into `[reopened_graph]` asserts the search found exactly one graph:

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

## Vary `k`: reuse upstream

A new neighbor count changes only the neighbors and connectivity {term}`provenance`. The rest of the parameters, like the normalization, PCA, and ANN references reamin unchanged.

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

See {doc}`graph_construction` for how changing `k` affects an analysis. Here, notice that only neighbours and the graph needed new results and we don't to recompute everything before that as well.

## Vary `dims`: build a separate branch

A new PCA dimensionality creates a new branch of analysis to explore; Since the ANN, neighbors, and the connectivity maps rely on the old PCA reduction, those results are simply pushed to side and not reused for this new analysis chain. If we want to go back and access those previous results (pca = 15 dims), that remains possible with SCARF.

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

For reference, these are the previous 15-PC results everything above is compared against — not the new 20-dim objects:

```{code-cell} ipython3
{
    "baseline PCA": pca,
    "baseline neighbors": neighbors_k11,
    "baseline graph": graph_k11,
}
```

## Force recompute

`invalidate_cache=True` tells Scarf to skip {term}`reuse` even when the parameters match. Previously completed artifacts stay on disk untouched. The new reference gets a different id and path, making it a truly new branch.

For normalization, the call below would write a fresh normalized artifact while keeping the exact immutable `cell_selection` and `feature_selection` inputs. It is not executed here because a throwaway duplicate adds no evidence to the lineage figure.

```python
forced = ds.run_normalization(cell_selection, hvg_ref, invalidate_cache=True)
forced != normalized
```

## Compare lineage

All three graphs go into one read-only report: both neighbour-count branches and the `dims=20` fork. Anything they share shows up once, so you can see exactly where each branch splits off.

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

You get the Mermaid dependency graph with the artifact details underneath. The two `k` branches should part ways after the ANN index, while the `dims=20` branch leaves earlier, at PCA, taking its own ANN, neighbours, and graph with it.

If the report needs to travel with an analysis, you can export it; `to_markdown()` gives you exactly what the notebook shows; skim the first lines before sending the whole thing on its way.

```{code-cell} ipython3
lineage_markdown = lineage.to_markdown()
lineage_markdown.splitlines()[:12]
```

And `lineage.to_mermaid()` hands you just the diagram source, for pipelines that only want the picture.
