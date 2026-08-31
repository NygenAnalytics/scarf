(provenance)=

# Why Scarf records provenance

A single-cell result is a chain of choices. A marker table depends on a clustering, the clustering
depends on a graph, and the graph depends on exact cells, features, normalization, and reduction.
When several branches share one datastore, a filename or cluster column cannot identify that chain.

Scarf therefore persists substantial results as immutable {term}`artifacts <artifact>`. Each artifact records
the producing operation, scientific parameters, and exact upstream inputs. Matching provenance can
reuse completed work; changing an input creates a distinct downstream branch.

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

The branches share counts, selections, and normalization. Their reductions and downstream graphs
remain distinct because those inputs differ. No branch becomes an implicit current result.

A {py:class}`~scarf.PipelineRun` adds one durable record for a complete workflow invocation. It
binds named outputs and frozen cell and feature views without changing the artifact identity rules.

Provenance establishes computational relationships. It does not prove that a parameter was
scientifically appropriate, decide which branch is best, or replace study records.

Use {doc}`../tutorials/reuse_and_tracing` for the single executable guide to listing, inspecting,
branching, invalidating, and comparing artifact lineage. Use
{doc}`../reference/api/artifacts` for the complete API contract.
