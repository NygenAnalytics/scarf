# Graph construction API reference

These `DataStore` methods build the neighbourhood graph as separate, provenance-backed steps.
Stage builders return an {py:class}`~scarf.ArtifactRef` and, by default, select it in {py:class}`~scarf.AssayState` as part of the current analysis chain (`update_state=True`).
{py:meth}`~scarf.DataStore.load_graph` returns a `csr_matrix` and has no `update_state` parameter.
Prefer `ds.pipeline.run` for a default RNA recipe; use individual methods when you need branching, custom parameters, or partial recomputation.

See {doc}`../../tutorials/graph_construction`.

## Chain order

1. {py:meth}`~scarf.DataStore.run_normalization`
2. {py:meth}`~scarf.DataStore.run_pca`, {py:meth}`~scarf.DataStore.run_lsi`, or {py:meth}`~scarf.DataStore.run_custom_reduction`
3. Optional {py:meth}`~scarf.DataStore.run_harmony` (batch correction before ANN)
4. {py:meth}`~scarf.DataStore.build_ann_index`
5. {py:meth}`~scarf.DataStore.query_neighbors`
6. {py:meth}`~scarf.DataStore.build_connectivity_map`

{py:meth}`~scarf.DataStore.build_embedding_initialization` depends only on the reduction.
Call it when you need K-means initialization for UMAP, unless you pass ``ini_embed`` yourself.
It is not required to build the neighbourhood graph.

## Downstream methods

Downstream UMAP, t-SNE, clustering, and sampling read the current `AssayState.connectivity_map` when `graph=None`.
Each also accepts an exact connectivity-map or integrated-graph reference through `graph=`, so a side branch can be analysed without changing the selected chain.
They do not accept an independent feature selection: named lineage edges project the normalized feature selections used to construct the graph.
Leiden and UMAP return the {py:class}`~scarf.ArtifactRef` they wrote; Paris clustering returns a {py:class}`~scarf.clustering.ParisClusteringResult` whose ``ref`` field holds that artifact:

```python
graph_k21 = ds.build_connectivity_map(neighbors_k21, update_state=False)
ds.run_leiden_clustering(graph=graph_k21, resolution=0.5, label="leiden_k21")
```

## Methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.run_normalization
   scarf.DataStore.run_pca
   scarf.DataStore.run_lsi
   scarf.DataStore.run_custom_reduction
   scarf.DataStore.run_harmony
   scarf.DataStore.build_ann_index
   scarf.DataStore.query_neighbors
   scarf.DataStore.build_connectivity_map
   scarf.DataStore.build_embedding_initialization
   scarf.DataStore.load_graph
```

```{eval-rst}
.. automethod:: scarf.DataStore.run_normalization
.. automethod:: scarf.DataStore.run_pca
.. automethod:: scarf.DataStore.run_lsi
.. automethod:: scarf.DataStore.run_custom_reduction
.. automethod:: scarf.DataStore.run_harmony
.. automethod:: scarf.DataStore.build_ann_index
.. automethod:: scarf.DataStore.query_neighbors
.. automethod:: scarf.DataStore.build_connectivity_map
.. automethod:: scarf.DataStore.build_embedding_initialization
.. automethod:: scarf.DataStore.load_graph
```
