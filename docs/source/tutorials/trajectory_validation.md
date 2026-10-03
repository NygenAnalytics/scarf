---
description: Check trajectory assumptions, changing genes, and candidate fate probabilities.
---
(trajectory_validation)=

# Check a trajectory interpretation

A plausible UMAP is only a starting point. Use these checks after {doc}`pseudotime`,
{doc}`expression_dynamics`, or {doc}`fate_mapping` to decide which conclusions the result
supports. The snippets below continue those examples; this page does not run a separate
analysis.

## Check the biological orientation

Ask whether known early markers occur toward the start and terminal markers toward the
end. Intermediate populations should be allowed intermediate scores. Also check whether
batch, cell quality, or another process could explain the ordering.

Try plausible alternative endpoints and modest graph changes. If they give different
orderings, describe that uncertainty instead of treating one smooth map as a unique
answer. The custom source/sink weighting in the pancreas examples is part of their
assumptions; changing it can change the scores.

## Count the cells that were scored

```python
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
float(pseudotime.valid.mean())
```

By default, pseudotime uses the largest connected graph component. Other cells receive
undefined values and `valid=False`. A separate component may reflect biology, filtering,
or graph settings. Report how many cells were excluded and investigate where they lie.
Use the validity mask for summaries you write yourself; Scarf applies it internally
when searching for pseudotime markers or aggregating expression.

## Inspect changing genes and modules

```python
markers = ds.load_pseudotime_markers(marker_ref)
modules = ds.load_pseudotime_aggregation(modules_ref)
```

Correlations can miss transient or branch-specific changes. Inspect the profiles and
several genes in a module, not just its name or size. Compare smoothing widths and
feature choices. Module numbers are identifiers, not developmental stages.

Untested marker features have missing p-values, and correction covers tested features
only. The aggregation result contains features that passed its expression and variance
checks. Neither result establishes that a gene causes the process.

## Separate probability checks from biological evidence

```python
fate = ds.load_fate_mapping(fate_ref)
valid_probabilities = fate.values[fate.valid]
row_sum_error = abs(valid_probabilities.sum(axis=1) - 1.0).max()
row_sum_error
```

Valid rows should be finite, non-negative, and sum to one. Probabilities at selected
sinks are fixed by the model, so high values there check the calculation rather than
validate the biology. Ignore invalid rows when interpreting probabilities.

For biological support, use independent endpoint evidence and compare plausible sink
sets. Include the possibility that a real outcome is missing. A smooth gradient can
reflect graph geometry even when the underlying biological decision is abrupt.

## Keep the inputs with the results

Record which graph, cells, features, and endpoints produced the result. Saved references
let you check this directly:

```python
ds.inspect_artifact(fate_ref)
ds.lineage(fate_ref)
```

The trajectory results should refer to the intended graph and cell selection. Marker
and module results should use the intended feature selection and pseudotime result.
Keep alternatives separate when comparing graph or endpoint changes; see
{doc}`reuse_and_tracing` for following saved analyses.

When reporting a trajectory, include the endpoint evidence, graph and smoothing
choices, valid-cell counts, sensitivity checks, and remaining uncertainty alongside
the saved result references.
