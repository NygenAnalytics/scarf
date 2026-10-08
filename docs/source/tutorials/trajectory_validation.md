---
description: Check trajectory assumptions, changing genes, and candidate fate probabilities.
---
(trajectory_validation)=

# Trajectory validation checklist

A plausible UMAP is only a starting point in the journey of pseudotime and trajectory inference. In this tutorial, we can learn about the general checks to perform after {doc}`pseudotime`, {doc}`expression_dynamics`, or {doc}`fate_mapping`, to then decide which conclusions each result supports. We can see if early [source] markers sit towards the start of the inference, and the terminal towards the ends; all the while we allow for the intermediate populations to stay intermediate by ruling out cell quality, batch effects, or some other process driving the issues.

Furthermore, because the custom source/sink weighting in the pancreas examples is itself an assumption, try plausible alternative endpoints and modest graph changes to see if they give different orderings. Visualizing these uncertainties can allow you to rule out other plausible branches that may be invalid. This tutorial simply serves as a checker

## Count the cells that were scored

```Python
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
float(pseudotime.valid.mean())
```

By default, pseudotime uses the largest connected graph component, with other cells receiving
undefined values and `valid=False`. A separate component may reflect biology, filtering,
or graph settings. Report how many cells were excluded and investigate where they lie.
The validity mask simply tells us what cells the pseudotime results actually speak fors.

## Inspect changing genes and modules

```python
markers = ds.load_pseudotime_markers(marker_ref)
markers.table.head()
```

To inspect the changes within theexpression profiles with the module plots in {doc}`expression_dynamics`, you can run the below to load the aggregration results when you need its values for further checks:

```python
modules = ds.load_pseudotime_aggregation(modules_ref)
modules
```

Correlations can miss transient or branch-specific changes as previously discussed, thus inspect the profiles and genes in a module. Compare smoothing widths and feature choices to find what describes your data best. 

Remember, untested marker features have missing p-values, and correction covers tested features
only.

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
let you check this directly, and then then inspect how those inputs connect:

```python
ds.inspect_artifact(fate_ref)
ds.lineage(fate_ref)
```

The trajectory results should refer to the intended graph and cell selection. Marker and module results should use the intended feature selection and pseudotime result. Keep alternatives (other trajectories) separate when comparing graph or endpoint changes to avoid confusion, and refer to {doc}`reuse_and_tracing` for following saved analyses.

## Important caveats to consider regarding trajectory validation

- **Mistaking technical covariates or circular endpoints for true progression:** A smooth trajectory across a low-dimensional embedding often tracks technical confounders, such as sequencing depth, mitochondrial read percentage, or uncorrected batch variation, rather than developmental time. Furthermore, validating an axis using only the manually chosen source and sink markers is circular; the scoring algorithm mathematically guarantees those populations sit at the extremes. Validating the trajectory requires testing alternative endpoint definitions, inspecting intermediate transition markers, and ensuring the axis does not merely correlate with library quality metrics, which is why thorough quality control is required.
- **Ignoring graph fragmentation and the valid-cell fraction (pseudotime.valid):** SCARF calculates trajectory metrics exclusively across the largest connected graph component, assigning valid=False and undefined values to disconnected cells. Omitting the validity check (pseudotime.valid.mean()) risks silently discarding substantial cell populations or entire uncharacterized lineages. Custom summaries and external exports that fail to filter by pseudotime.valid will propagate NaN values, distort statistical distributions, and misrepresent dataset representation.
- **Conflating mathematical solver convergence with biological commitment:** Verifying that fate probabilities sum to one across rows confirms the numerical calculation worked, not that the biological model is complete or correct. Graph diffusion models inherently generate smooth, continuous gradients across continuous manifolds, which can obscure abrupt, threshold-driven transcriptional switches (such as bistable transcription factor cross-repression). Treating a continuous mathematical probability gradient as proof of gradual, reversible biological plasticity risks misinterpreting discrete commitment checkpoints.
