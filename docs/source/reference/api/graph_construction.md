# Graph construction API reference

These `DataStore` methods build the neighbourhood graph as separate, provenance-backed
steps. Each call returns an {py:class}`~scarf.ArtifactRef` and, by default,
selects it in {py:class}`~scarf.AssayState` as part of the current analysis
chain (`update_state=True`). Prefer `ds.pipeline.run` for a default RNA recipe;
use individual methods when you need branching, custom parameters, or partial
recomputation.

See {doc}`../../tutorials/custom_graph_construction`.

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


## Migration from `make_graph`

`DataStore.make_graph` has been removed. Use `ds.pipeline.run` or the methods
above. Existing datastores remain readable without rebuilding their graphs.


| Former `make_graph` concern | Replacement |
|---|---|
| Whole RNA workflow with defaults | `ds.pipeline.run(...)` |
| `feat_key`, `cell_key`, `log_transform`, `renormalize_subset` | `run_normalization(...)` |
| `dims`, `feat_scaling`, `pca_cell_key`, `custom_loadings`, `show_elbow_plot` | `run_pca(...)` (or `run_lsi` for ATAC) |
| `dims=0` (use normalized features directly) | `run_custom_reduction(np.eye(n_features), normalized, ...)` |
| `harmonize=True`, `batch_columns`, `harmony_params` | `run_harmony(batch_columns, reduction, ...)` |
| `ann_metric`, `ann_efc`, `ann_ef`, `ann_m`, `ann_parallel` | `build_ann_index(...)` |
| `k` | `query_neighbors(..., k=...)` |
| `local_connectivity`, `bandwidth` | `build_connectivity_map(...)` |
| `n_centroids`, `rand_state` (k-means init) | `build_embedding_initialization(...)` |
| `local_cache` | `run_pca`, `run_lsi`, or `run_custom_reduction` |
| `update_keys=True` | `update_state=True` on individual graph-construction methods |
| `return_ann_object=True` | Reload the completed ANN and neighbors artifacts when an `AnnStream` is required |

The removed method derived some ANN construction defaults from `k` and `dims`. Direct
`build_ann_index` uses fixed defaults (`ann_efc=50`, `ann_ef=50`, `ann_m=48`) unless you
override them. For parity when `k > 16`, pass
`ann_efc=ann_ef=min(100, max(k * 3, 50))`. For dimension parity, pass
`ann_m=min(max(48, int(dims * 1.5)), 64)`.

Harmony workflows should call `run_harmony`, build ANN and neighbors from its
returned coordinates, then call `build_mapping_reference` when Symphony mapping
support is required. `local_cache` is no longer accepted by `run_harmony`,
`build_ann_index`, or `query_neighbors` because those stages read persisted
coordinates instead of normalized expression.

ANN graph construction supports L2 and cosine metrics. Persisted L2 neighbor
distances are Euclidean distances; Scarf converts the squared values returned
by HNSW before connectivity is calculated.

