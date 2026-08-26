(provenance)=
# Provenance and artifacts

Single-cell analysis is a chain of dependent choices.
A marker table depends on a clustering, the clustering depends on a graph, and the graph depends on a particular cell set, feature set, normalization, and reduction.
When several parameter branches live in one datastore, filenames and cluster labels alone do not explain which choices produced a result.

Scarf persists each substantial result as an {term}`artifact` and records what produced it.
This lets a user inspect an inherited datastore, compare branches, and {term}`reuse` upstream work without maintaining a separate datastore for every parameter choice.

```{mermaid}
flowchart LR
    counts["Counts"]
    selection["Cell and feature selections"]
    norm["Normalization"]
    pca15["PCA, 15 dimensions"]
    pca30["PCA, 30 dimensions"]
    graph15["Neighbour graph"]
    graph30["Neighbour graph"]
    clusters["Clusters"]
    markers["Marker table"]
    counts --> selection --> norm
    norm --> pca15 --> graph15 --> clusters --> markers
    norm --> pca30 --> graph30
```

The two PCA branches can share counts, selections, and normalization.
Their downstream graphs remain distinct because their inputs differ.

## What Scarf records

An artifact has a stable reference, its stored payload, and a {term}`provenance` record.
Provenance is only:

- the operation that produced it, such as `run_pca`
- scientific parameters that can change the result
- input selections and upstream artifact references

The artifact also stores sibling attributes that are not part of provenance: execution options (for example local scratch policy) and whether the write completed successfully.
Reuse matches on provenance only.

Feature selections follow the same model.
`mark_hvgs`, `mark_prevalent_peaks`, and manual selection return immutable references and publish plain labels for convenience.
Pass a returned reference between stages when you want to pin a branch, or resolve one exact label with `ds.resolve_features(assay, label)`.
There is no cell-key prefix, composite feature expression, or implicit latest feature selection.
Use the reserved `all_features` label for the assay-wide universe.

### Analysis chain and reuse

Downstream methods receive these references directly or resolve them from the assay's current {term}`analysis chain`.
A completed result with the same operation, parameters, and inputs can be reused.
Changing PCA dimensions creates a new reduction and new dependent results, while the matching normalization can still be reused.

The current analysis chain is a convenience for a linear workflow, not the only history in the store.
Side branches can be created without selecting them as current.
See {doc}`../tutorials/graph_construction` for that relationship.

Graph-derived methods accept `graph=`.
When it is omitted, Scarf resolves the assay's current `connectivity_map` from `AssayState`.
The graph's named lineage edges identify the normalized artifact and its feature selection, so graph consumers do not accept a second feature-selection argument.
Imported-coordinate graphs have no normalized feature selection; integrated graphs can project zero, one, or several selections from their named sources.

## Inspect a result

Use the public datastore methods rather than reading Zarr attributes directly:

```python
refs = ds.list_artifacts(kind="reduction", complete_only=True)
status = ds.inspect_artifact(refs[0])

status.operation
status.parameters
status.inputs
status.execution_options
```

`list_artifacts` uses the default assay unless another assay is supplied.
Store-level outputs can be listed with `scope="datastore"`.
`load_artifact(ref)` opens the payload only after Scarf confirms that the artifact exists and is complete.

## View upstream lineage

`DataStore.lineage` follows artifact inputs upstream and returns an `ArtifactLineage` report:

```python
lineage = ds.lineage(graph_ref)
lineage
```

Notebook display renders a Mermaid dependency graph followed by operation, parameter, execution-option, and external-input details.
The same report can be exported explicitly:

```python
mermaid_source = lineage.to_mermaid()
markdown_report = lineage.to_markdown()
```

Pass a named mapping to compare several outputs in one report:

```python
lineage = ds.lineage(
    {
        "baselineGraph": baseline_graph,
        "alternativeGraph": alternative_graph,
    }
)
```

This answers questions such as which PCA fed a graph, which clustering produced a marker table, and where two analysis branches diverged.

## Reuse, replacement, and limits

`invalidate_cache=True` asks a producing method to write a new artifact even when a completed match exists.
It does not delete the older artifact.
A failed or interrupted writer leaves an incomplete result, which downstream readers reject.

Provenance does not prove that an analysis choice was scientifically suitable, delete superseded branches, or replace study records.
It records the computational relationships needed to inspect and reproduce store-backed results.

For an executable walkthrough, see {doc}`../tutorials/reuse_and_tracing`.
