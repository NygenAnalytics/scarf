# Atomic graph operations

These `DataStore` methods build the neighbourhood graph as separate, provenance-backed
steps. Each call returns an {py:class}`~scarf.ArtifactRef` and, by default, publishes into
{py:class}`~scarf.AssayState` (`update_state=True`). Prefer `ds.pipeline.run` for a default
RNA recipe; use this chain when you need branching (for example Harmony), custom parameters,
or partial recomputation.

See {doc}`../../concepts/graph_and_state` and
{doc}`../../tutorials/atomic_graph_operations`.

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

Downstream UMAP and clustering read the published state.

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
```


## Migration from `make_graph`

`DataStore.make_graph` is deprecated. It remains a facade over the same atomic chain and
emits `DeprecationWarning`. Prefer `ds.pipeline.run` or the methods above.


| Former `make_graph` concern | Replacement |
|---|---|
| Whole RNA workflow with defaults | `ds.pipeline.run(...)` |
| `feat_key`, `cell_key`, `log_transform`, `renormalize_subset` | `run_normalization(...)` |
| `dims`, `feat_scaling`, `pca_cell_key`, `custom_loadings`, `show_elbow_plot` | `run_pca(...)` (or `run_lsi` for ATAC) |
| `harmonize=True`, `batch_columns`, `harmony_params` | `run_harmony(batch_columns, reduction, ...)` |
| `ann_metric`, `ann_efc`, `ann_ef`, `ann_m`, `ann_parallel` | `build_ann_index(...)` |
| `k` | `query_neighbors(..., k=...)` |
| `local_connectivity`, `bandwidth` | `build_connectivity_map(...)` |
| `n_centroids`, `rand_state` (k-means init) | `build_embedding_initialization(...)` |
| `local_cache` | Same name on atomic methods (`"auto"`, `True`, `False`, or a path) |
| `update_keys=True` | `update_state=True` on atomic methods |

`make_graph` derives some ANN construction defaults from `k` and `dims`. Direct
`build_ann_index` uses fixed defaults (`ann_efc=50`, `ann_ef=50`, `ann_m=48`) unless you
override them. Pass explicit ANN parameters when you need parity with an old `make_graph`
call. The full deprecated signature remains under {doc}`datastore`.

