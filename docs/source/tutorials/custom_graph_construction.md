---
description: Build an assay graph stage by stage, branch parameters, and control the current analysis chain.
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

(custom_graph_construction)=
(graph_and_state)=

# Building neighbourhood graphs step by step

Embeddings, clustering, imputation, and trajectories consume a cell graph.
Scarf builds that graph from a selected cell population and feature set through
separate persisted stages. Calling the stages directly is useful when you need
to branch one parameter, insert batch correction, or {term}`reuse` an expensive
reduction.

Feature selection is covered in {doc}`feature_selection`. This guide begins once
the feature key exists.

## The standard graph workflow

For RNA, the graph stages are:

1. Normalize the selected genes.
2. Reduce them with PCA.
3. Optionally correct the reduced coordinates with Harmony.
4. Build an approximate-neighbour index.
5. Query `k` neighbours per cell.
6. Convert neighbour distances into weighted connectivity.
7. Build a separate initialization for UMAP.

ATAC follows the same shape but uses TF-IDF normalization and LSI. The
initialization does not define graph edges; it only supplies starting
coordinates for a layout.

`ds.pipeline.run()` orchestrates the standard RNA path and can continue through
UMAP, clustering, doublet scoring, and markers. Use it when the defaults match
the analysis. The stage methods below expose the same persisted results with
more control.

## Build an RNA graph explicitly

```{code-cell} ipython3
import pandas as pd

import scarf
import scarf.plotting as splt

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures"],
    highs=[15000, 4000],
    lows=[1000, 500],
    reset_previous=True,
)
if "I__hvgs" not in ds.RNA.feats.columns:
    ds.mark_hvgs(
        min_cells=20,
        top_n=500,
        show_plot=False,
    )
```

Each method returns an {term}`ArtifactRef`. Passing it to the next method makes
the dependency explicit.

```{code-cell} ipython3
normalized = ds.run_normalization(feat_key="hvgs")
pca = ds.run_pca(normalized, dims=15)
initialization = ds.build_embedding_initialization(
    pca,
    n_centroids=100,
)
ann_index = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann_index, k=11)
graph = ds.build_connectivity_map(neighbors)
graph
```

`load_graph` returns the sparse cell-by-cell connectivity matrix for supported
custom graph analyses.

```{code-cell} ipython3
loaded_graph = ds.load_graph()
loaded_graph.shape, loaded_graph.nnz
```

```{code-cell} ipython3
splt.graph_qc(loaded_graph)
```

The graph should include every active cell and have finite nonzero
connectivities. A disconnected graph, many isolated cells, or degree structure
driven by a QC metric warrants revisiting features, PCA dimensions, or `k`.

## Understand the current analysis chain

Successful stages normally update a small pointer set for the assay. In plain
language, this is the {term}`analysis chain` that downstream calls should use
when no explicit input is supplied. The public class representing it is
`AssayState`.

```{code-cell} ipython3
state = ds.get_assay_state("RNA")
{
    "cell selection": state.cell_key,
    "feature selection": state.feat_key,
    "normalization": state.normalized,
    "reduction": state.reduction,
    "neighbours": state.neighbors,
    "connectivity": state.connectivity_map,
}
```

`run_umap`, Leiden, and Paris resolve the current connectivity and
initialization from this chain. They write their cell-metadata columns as usual
and return the artifact they wrote, so a result can be inspected or reused
without looking the column up first.

```{code-cell} ipython3
umap = ds.run_umap(
    n_epochs=100,
    parallel=True,
)
clusters = ds.run_leiden_clustering(resolution=0.5)
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_leiden_cluster",
)
```

```{code-cell} ipython3
ds.inspect_artifact(clusters).parameters
```

## Branch without changing the current chain

Suppose the PCA and ANN index are expensive but two neighbour counts need to be
compared. Reuse the same index and set `update_state=False` on the side branch.

```{code-cell} ipython3
neighbors_k21 = ds.query_neighbors(
    ann_index,
    k=21,
    update_state=False,
)
graph_k21 = ds.build_connectivity_map(
    neighbors_k21,
    update_state=False,
)

current_state = ds.get_assay_state("RNA")
current_state.connectivity_map == graph, graph_k21 != graph
```

The side branch remains a complete, addressable artifact, while downstream
calls without explicit inputs still use the `k=11` graph. This prevents a
parameter experiment from silently replacing the selected chain.

To analyse the side branch, pass its graph as the first argument. The current
chain is untouched, so both partitions stay available for comparison.

```{code-cell} ipython3
ds.run_leiden_clustering(
    graph_k21,
    resolution=0.5,
    label="leiden_k21",
)
pd.crosstab(
    pd.Series(ds.cells.fetch("RNA_leiden_cluster"), name="k=11"),
    pd.Series(ds.cells.fetch("RNA_leiden_k21"), name="k=21"),
)
```

Most cells keep their group. The off-diagonal mass shows which splits depend on
`k`: the smallest `k=11` clusters are absorbed once every cell sees more
neighbours, so treat those boundaries as provisional.

## Recompute only what changed

Artifact identity includes the operation, scientific parameters, and upstream
inputs. Calling an identical stage reuses its completed result. Changing `k`
reuses normalization, PCA, and the ANN index but creates new neighbour and
connectivity artifacts. Changing the cell or feature selection invalidates all
dependent stages.

Harmony fits between PCA and the ANN index:

```python
corrected = ds.run_harmony(["technical_batch"], pca)
corrected_index = ds.build_ann_index(corrected)
corrected_neighbors = ds.query_neighbors(corrected_index, k=21)
corrected_graph = ds.build_connectivity_map(corrected_neighbors)
```

Use {doc}`../concepts/provenance` to inspect complete lineage and
{doc}`provenance_and_reuse` for reuse and invalidation patterns.
