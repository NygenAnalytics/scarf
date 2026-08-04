# Graph construction API reference

These `DataStore` methods build the neighbourhood graph as separate, provenance-backed
steps. Each call returns an {py:class}`~scarf.ArtifactRef` and, by default,
selects it in {py:class}`~scarf.AssayState` as part of the current analysis
chain (`update_state=True`). Prefer `ds.pipeline.run` for a default RNA recipe;
use individual methods when you need branching, custom parameters, or partial
recomputation.

See {doc}`../../tutorials/graph_construction`.

## Chain order

1. {py:meth}`~scarf.DataStore.run_normalization`
2. {py:meth}`~scarf.DataStore.run_pca`, {py:meth}`~scarf.DataStore.run_lsi`, or
   {py:meth}`~scarf.DataStore.run_custom_reduction`
3. Optional {py:meth}`~scarf.DataStore.run_harmony` (batch correction before ANN)
4. {py:meth}`~scarf.DataStore.build_embedding_initialization` (needed before UMAP unless
   you pass ``ini_embed``)
5. {py:meth}`~scarf.DataStore.build_ann_index`
6. {py:meth}`~scarf.DataStore.query_neighbors`
7. {py:meth}`~scarf.DataStore.build_connectivity_map`

Downstream UMAP, t-SNE, clustering, and sampling read the current analysis
chain when they are called without a graph. Each also accepts a connectivity
map or integrated graph as its first argument and returns the artifact it
wrote, so a side branch can be analysed without changing the selected chain:

```python
graph_k21 = ds.build_connectivity_map(neighbors_k21, update_state=False)
ds.run_leiden_clustering(graph_k21, resolution=0.5, label="leiden_k21")
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
   scarf.DataStore.build_embedding_initialization
   scarf.DataStore.build_ann_index
   scarf.DataStore.query_neighbors
   scarf.DataStore.build_connectivity_map
   scarf.DataStore.load_graph
```

```{eval-rst}
.. automethod:: scarf.DataStore.run_normalization
.. automethod:: scarf.DataStore.run_pca
.. automethod:: scarf.DataStore.run_lsi
.. automethod:: scarf.DataStore.run_custom_reduction
.. automethod:: scarf.DataStore.run_harmony
.. automethod:: scarf.DataStore.build_embedding_initialization
.. automethod:: scarf.DataStore.build_ann_index
.. automethod:: scarf.DataStore.query_neighbors
.. automethod:: scarf.DataStore.build_connectivity_map
.. automethod:: scarf.DataStore.load_graph
```

