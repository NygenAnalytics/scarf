(graph_and_state)=
# Graph construction and published state

Scarf builds one neighbourhood graph per selected cell and feature set, then
reuses that graph for embeddings, clustering, mapping, and multimodal
integration. You can construct that graph at three levels of control. They are
not competing APIs: all three publish the same `AssayState` that downstream
methods read.

## Three levels, one system

### 1. Beginner path: `ds.pipeline.run`

`DataStore.pipeline.run` runs the standard RNA recipe
(`pipeline_id="basic_rna_analysis"`). It filters cells, selects HVGs, runs the
atomic graph chain, and optionally continues to UMAP, Leiden, Paris, doublets,
and markers. Step arguments accept `None` (defaults), `False` (skip), or a
dict forwarded to the underlying method.

Use this when you want a complete, provenance-backed analysis with minimal
orchestration. Capture the returned `dict[str, ArtifactRef]` if you need the
refs later.

### 2. Controlled path: atomic operations

Call the graph steps yourself when you need branching, custom parameters, or
partial recomputation:

1. `run_normalization`
2. `run_pca` or `run_lsi`
3. optional `run_harmony`
4. `build_embedding_initialization` (required before UMAP unless you pass `ini_embed`)
5. `build_ann_index`
6. `query_neighbors`
7. `build_connectivity_map`

By default, successful steps publish into `AssayState` for the assay
(`update_state=True`), and each step reads the published stage it needs. A plain chain of
calls with no arguments passed between them is therefore enough for a linear analysis.

Each call also returns an `ArtifactRef`. Capture those refs and pass them into later steps
when you want a branch, for example two values of `k` from one ANN index. Combine that with
`update_state=False` for side branches you do not want as the current published chain.

See {doc}`../tutorials/atomic_graph_operations` for a short executable chain and
a `make_graph` migration table.

### 3. Compatibility path: deprecated `make_graph`

`make_graph` remains available as a facade over the same atomic chain. It emits
`DeprecationWarning` and will be removed in a future major release. Prefer
`pipeline.run` or the atomic methods in new code. Existing tutorials that still
call `make_graph` produce the same published state once the call finishes.

## AssayState

`AssayState` is the assay-level pointer set for the current graph chain. Read it
with:

```python
state = ds.get_assay_state("RNA")
```

When present, it records the selected `cell_key` and `feat_key` plus refs for
the published stages: normalized matrix, feature scaling, reduction, optional
batch correction, ANN index, embedding initialization, neighbors, and
connectivity map. Named results (for example embeddings or cluster labels) can
appear under `named_results`.

Downstream methods such as `run_umap`, `run_leiden_clustering`, and
`run_paris_clustering` resolve the graph through this published state (or an
explicit graph location when you pass one). Construction path does not matter:
pipeline, atomic calls, and `make_graph` all leave the same kind of state for
consumers.

If `get_assay_state` returns `None`, the assay has no current published artifact
chain. Released archives that predate artifacts often look like this even when
encoded `normed__...` groups exist on disk. Running any publishing atomic step
(or the pipeline) creates current-format state. Details and inspection patterns
are in {doc}`../tutorials/data_organization`.

## Choosing a level

| Goal | Use |
|---|---|
| Standard RNA analysis with sensible defaults | `ds.pipeline.run` |
| Change one stage, branch Harmony, or control reuse | Atomic methods |
| Keep an old script running briefly | `make_graph` (deprecated) |

Stay on one level per workflow unless you have a reason to drop into atomic
calls. Mixing is supported because state is shared, but reading
{doc}`provenance` first makes reuse and invalidation predictable.
