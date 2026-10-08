# Graph construction API reference

These `DataStore` methods build a neighbourhood graph as explicit, provenance-backed stages. Each
builder returns an {py:class}`~scarf.ArtifactRef`; pass that exact ref to the next stage.

Prefer `ds.pipeline.run()` for the fixed rich RNA recipe. Use the individual methods for a partial
workflow, an alternative branch, or parameters intentionally kept out of the pipeline surface.
See {doc}`../../tutorials/graph_construction` for an executable walkthrough.

## Explicit chain

Capture the live Boolean cell column once at the workflow boundary. The returned selection is
immutable; later live changes do not alter it.

```python
cells = ds.snapshot_cell_selection("I")
features = ds.select_hvgs(cells, top_n=1000, show_plot=False)

normalized = ds.run_normalization(cells, features)
pca = ds.run_pca(normalized, dims=21)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
graph = ds.build_connectivity_map(neighbors)
initialization = ds.build_embedding_initialization(pca)
```

The full stage order is:

1. {py:meth}`~scarf.DataStore.snapshot_cell_selection` and an exact feature-selection ref
2. {py:meth}`~scarf.DataStore.run_normalization`
3. {py:meth}`~scarf.DataStore.run_pca`, {py:meth}`~scarf.DataStore.run_lsi`, or
   {py:meth}`~scarf.DataStore.run_custom_reduction`, or none
4. Optional {py:meth}`~scarf.DataStore.run_harmony`
5. {py:meth}`~scarf.DataStore.build_ann_index`
6. {py:meth}`~scarf.DataStore.query_neighbors`
7. {py:meth}`~scarf.DataStore.build_connectivity_map`

Without a reduction, `build_ann_index(normalized)` builds the graph on the normalized artifact
itself: its float32 values of the selected features are the coordinates, one per feature, as in a
pipeline run with `pca_dims=0`. Every later stage, including the embedding initialization and
an SNN integration, accepts such a graph. Harmony corrects reduced coordinates only, and WNN
integration, mapping references, and doublet scoring need them too, so each rejects normalized
coordinates with an error that names a reduction such as `run_pca`. Because normalized values can
be coordinates, `run_normalization` writes them with a checked writer: a value that is NaN or
infinite once rounded to float32, such as one from a custom normalizer or the `log1p` of a
negative count, raises `NonFiniteArtifactError`, a `ValueError` that names `run_normalization`,
and no complete artifact is saved.

The hnswlib index holds every cell in memory, about `4 * dims + 8 * ann_m + 130` bytes per cell on
x86-64 Linux (`scarf.neighbors.index.ann_index_peak_bytes`), where `dims` is the number of
coordinate dimensions: the PCA dimensions, or the selected features of normalized coordinates.
`build_ann_index` holds it while it adds the coordinate blocks and while it saves it, and
`query_neighbors` while it loads and queries it, beside the indices and distances of every cell's
neighbours, 8 bytes per neighbour. Each compares what it holds with the datastore's memory budget
before it creates or loads the index and raises `MemoryError` over it, naming the bytes, the cells,
the dimensions, and the limit. The index moves between the store and memory through a temporary
file of about `4 * dims + 8 * ann_m + 20` bytes per cell
(`scarf.neighbors.index.ann_index_file_bytes`) in the system's temporary directory, which needs
that much free space.

{py:meth}`~scarf.DataStore.build_embedding_initialization` depends on explicit reduction, Harmony,
imported, or normalized coordinates. It is needed for UMAP unless an initialization array is
supplied, but is not part of connectivity construction.

`query_neighbors` reads the coordinate ref named by the ANN artifact. Its optional `coordinates=`
argument is only an equality check; it cannot redirect an index to different coordinates.

## Downstream consumers

UMAP, t-SNE, clustering, sampling, diffusion, trajectories, graph metrics, and graph-backed plots
require an exact graph or neighbour artifact. Analytical producers return artifacts and do not add
live metadata fields:

```python
umap = ds.run_umap(
    graph,
    initialization,
)
leiden = ds.run_leiden_clustering(
    graph,
    resolution=0.5,
)
ds.plots.embedding(layout=umap, color_by=leiden)
```

The scientific cell selection comes from graph lineage. Load a payload from its exact ref, or pass
the ref to another consumer.

Graph consumers do not accept an independent feature selection. Named lineage edges identify the
normalized feature selections used to construct a native graph. Imported-coordinate graphs have
no such selection; integrated graphs preserve the ordered projections of their explicit sources.

## Methods

```{eval-rst}
.. autosummary::
   :nosignatures:

   scarf.DataStore.snapshot_cell_selection
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
.. automethod:: scarf.DataStore.snapshot_cell_selection
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
