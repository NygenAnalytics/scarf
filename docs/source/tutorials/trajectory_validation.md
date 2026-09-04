---
description: Validate pseudotime, dynamic-feature, and fate artifacts through exact references.
---

(trajectory_validation)=

# Validate trajectory results

Trajectory outputs are model summaries over one explicit graph. A plausible embedding is not enough
to validate orientation, dynamic features, or terminal probabilities. Keep the exact graph,
pseudotime, marker, aggregation, sink-label, and fate refs together.

## 1. Check lineage and axes

Start from the returned refs:

```python
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
markers = ds.load_pseudotime_markers(marker_ref)
modules = ds.load_pseudotime_aggregation(module_ref)
fate = ds.load_fate_mapping(fate_ref)
```

Verify that:

- `pseudotime.graph` is the graph intended for the biological question;
- every downstream result names the same `pseudotime_ref` and cell selection;
- marker and module feature selections are the intended exact refs;
- sink labels align to the same graph rows;
- ordered cell and feature identities still validate.

Use `ds.inspect_artifact(ref)` for status and inputs and `ds.lineage(ref)` for the full upstream
chain. Do not reconstruct choices from private Zarr paths.

## 2. Check graph components and validity

`run_pseudotime_scoring(..., component_policy="largest")` scores the largest connected component and
stores `NaN` plus `valid=False` elsewhere. A disconnected graph may represent separate biology,
over-filtering, or an unsuitable neighbourhood count. Report both the valid fraction and component
decision.

```python
valid_fraction = float(pseudotime.valid.mean())
```

Do not silently treat invalid cells as early or late. Rebuilding a graph is a new branch, not a
repair of the old pseudotime artifact.

## 3. Challenge source and sink choices

Run plausible alternative orientations with explicit source/sink label artifacts or custom
zero-sum vectors. Compare whether:

- source populations sit at low pseudotime;
- terminal populations sit at high pseudotime;
- known early and late markers change in the expected directions;
- results are stable to modest graph and endpoint changes;
- the chosen labels are not confounded with batch or technical quality.

If alternatives disagree, preserve them and narrow the biological claim.

## 4. Validate dynamic features and modules

`load_pseudotime_markers` returns the tested table. Untested features have `NaN` p-values, and
multiple-testing adjustment covers tested features only. Correlation can miss transient or
branch-specific patterns.

`load_pseudotime_aggregation` returns valid feature indexes, cluster labels, and lazy binned
profiles. Check module size, representative genes, within-module coherence, and sensitivity to
window size and feature selection. Module numbers are identifiers, not developmental stages.

## 5. Validate fate probabilities

For `fate = ds.load_fate_mapping(fate_ref)`, require:

- finite, non-negative values for `fate.valid` rows;
- row sums close to one;
- high probability near the corresponding sink labels;
- no interpretation for invalid rows;
- sensitivity checks for plausible sink definitions and solver parameters.

```python
valid_probabilities = fate.values[fate.valid]
row_sum_error = abs(valid_probabilities.sum(axis=1) - 1.0).max()
```

A smooth probability gradient can still reflect graph geometry or endpoint supervision rather than
cell fate.

## 6. Handoff checklist

Report:

- all exact refs and their lineage;
- source, sink, and component policies;
- valid-cell counts;
- marker-testing and module settings;
- probability validity checks;
- alternative branches considered;
- biological assumptions and unresolved uncertainty.

See {doc}`pseudotime`, {doc}`expression_dynamics`, and {doc}`fate_mapping` for the producer and loader
journeys.
