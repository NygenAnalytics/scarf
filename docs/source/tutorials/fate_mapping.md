---
description: Compute and validate immutable multi-sink fate-probability artifacts.
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Fate Mapping Primer

During embryonic development, the pancreas builds its hormone-producing endocrine cells from a pool of progenitor cells. Around embryonic day 15.5 in the mouse, Ductal-like progenitors differentiate into the three major endocrine fates: Alpha cells (glucagon), Beta cells (insulin), and Delta cells (somatostatin). Single-cell RNA sequencing captures cells at discrete points along this transition, but each measurement is a snapshot. It shows cell states, not the direction of travel between them; fate mapping speculates about the end result of the cell state.

Fate mapping models branching as an **absorbing Markov chain** over the cell graph, with random walks directed forward along pseudotime:

1. **Directed flow:** the graph's transition probabilities are biased along pseudotime, so random walks flow primarily from progenitors toward differentiated endpoints.
2. **Absorbing sinks:** biological terminal states are defined as absorbing boundaries that trap random walkers.
3. **Hitting probabilities:** for each transient cell, the algorithm estimates the probability that a walk starting at that cell is absorbed by each terminal sink.

We can think of pseudotime analysis and fate mapping as answering two fundamentally different questions on a branching path:

- Pseudotime measures how far a cell has traveled along differentiation (like a single progress bar), but the issue is that, a single number cannot represent a fork in the road, or a cell differentiating into a different state. Pseudotime can indicate that a cell is differentiating, but not down what path.
- Fate mapping estimates which branch that cell is likely to take, being a step further than pseudotime. For each cell we have, fate mapping will output a probability for every candidate destination.

These probabilities are an exploratory mathematical summary, and do not provide biological proof that a cell is pursuing this lineage. Fate mapping simply reflects how closely a cell is connected to your chosen endpoints across this specific graph in terms of the mathematics. They do not track living cells over time, and they cannot rescue bad biological assumptions: if you pick the wrong terminal endpoints, the algorithm will still produce clean, confident probabilities toward the wrong destinations, thus maintaining an accurate biological context is key before performing pseudotime mapping.


# Estimate terminal-outcome probabilities with fate mapping

Here, we use a pre-run analysis of the developing pancreas to orient a graph from Ductal progenitor cells we discussed toward final fates of Alpha, Beta, or Delta cells. Following this, we use this information to estimate how terminal probability distributes across all three candidate outcomes.

## Reuse the prepared graph and sink labels

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
analysis_run = ds.pipeline.open(label="docs_default")
graph = analysis_run["connectivity_map"]

annotations = ds.cells.fetch("clusters", key="I")
source = annotations == "Ductal"
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any():
    raise ValueError("Source and sink annotations must both be present")
source_sink_vector = np.zeros(len(annotations), dtype=float)
source_sink_vector[source] = -1.0 / source.sum()
source_sink_vector[sink] = 1.0 / sink.sum()

pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
sink_labels_ref = analysis_run["clusters"]
sink_labels = analysis_run.cells.fetch("clusters")
```

Since single-cell neighbor graphs are undirected, a connection between two cells means they look similar, not that one comes before the other (one cell is a precursor of another). To give the graph an arrow of time, we define a potential gradient with the source/sink vector. We can conceptually think of the source/sink vector as (bear with me) a pressure difference across a plumbing network; by pumping water in at the progenitor cells (the start of the graph) and opening the drains at the mature cell types (the terminal of the graph), we can create a continuous downhill slope across the graph that guides flow through the intermediate states.

In this plumbing analogy, the total volume pumped in must exactly equal the volume draining out. The negative source mass ({math}`-1.0`) is divided evenly across all Ductal cells, marking the "start here" point (where the water flows in), and the positive sink mass ({math}`+1.0`) is shared across the mature Alpha, Beta, and Delta cells, marking the "end here" ends (where the water flows out). With this, the total sum is at zero.

This per-cell weighting is deliberate feature here, as dividing mass by group size prevents abundant populations from exerting an unfair gravitational pull over rarer cell types simply due to having a larger cell count. The resulting pressure drop turns undirected neighbor links into a directed downstream flow; Once again, our undirected neighbors are simply are similar cells in a neighborhood. Scoring the graph against it yields pseudotime, which is the continuous coordinate that tracks how far each cell has drifted from the initial progenitor pool (THE START).


To truly start the fate-mapping, we need to define the clusters we want to study. These clusters are defined via unsupervised clustering, which assigns cells to arbitrary numbers like cluster 3 or cluster 11. The issue is that developmental biology is defined by functional marker expression, not by clusters: Alpha cells produce glucagon (Gcg), Beta cells produce insulin (Ins1/Ins2), and Delta cells produce somatostatin (Sst). Rather than taking a circular shortcut, such as crowning whichever clusters score the highest pseudotime, we define terminal sinks directly from these biological priors, because only marker-grounded endpoints can support a trustworthy fate map.



```{code-cell}
candidate_fates = ["Alpha", "Beta", "Delta"]

# Map biological names to the run's integer sink labels. Both fetches cover
# the same cells in the same order, so positional grouping is exact.
cluster_names = np.asarray(ds.cells.fetch("clusters", key="I"))
assert len(cluster_names) == len(sink_labels)
name_by_id = (
    pd.Series(cluster_names).groupby(sink_labels).agg(lambda s: s.mode().iat[0])
)
share_by_id = (
    pd.Series(cluster_names)
    .groupby(sink_labels)
    .agg(lambda s: float((s == s.mode().iat[0]).mean()))
)
id_by_name = {}
for i, name in name_by_id.items():
    id_by_name.setdefault(name, []).append(int(i))
missing_fates = [fate for fate in candidate_fates if fate not in id_by_name]
if missing_fates:
    raise ValueError(f"Target sinks missing from dataset: {missing_fates}")
ambiguous_fates = {
    fate: id_by_name[fate] for fate in candidate_fates if len(id_by_name[fate]) != 1
}
if ambiguous_fates:
    raise ValueError(f"Ambiguous sink mapping, refine the selection: {ambiguous_fates}")
terminal_labels = [id_by_name[fate][0] for fate in candidate_fates]
print(f"Tracking differentiation potential into {len(terminal_labels)} fates: {candidate_fates}")
sink_names = {int(label): str(name_by_id.loc[label]) for label in terminal_labels}
pd.DataFrame(
    {
        "sink label": terminal_labels,
        "majority annotation": [sink_names[int(label)] for label in terminal_labels],
        "majority share": [
            round(float(share_by_id.loc[label]), 3) for label in terminal_labels
        ],
    }
)
```

The three labels above come from lineage markers, not from ranking pseudotime, and the table reports each pick's majority share so the mapping stays auditable. That choice breaks a loop worth naming: had the sinks been picked by greatest mean pseudotime, annotations would have oriented the graph, the orientation would have produced pseudotime, and pseudotime would then have picked the sinks, rewarding sequencing-depth artifacts, outlier populations, or technical dead ends with terminus status. Never copy a highest-pseudotime rule into a real pipeline as endpoint discovery. In a real analysis, endpoints must come from study-specific evidence (for example, marker-supported terminal states reviewed against negative controls, as in {doc}`annotation`), because every downstream probability inherits the choice. A defensible sink is one whose probability mass concentrates near its own label on the map below; a sink whose probability spreads evenly or peaks elsewhere is a rejected hypothesis, not a discovery.

## Compute the fate probabilities

```{code-cell}
# Execute fate mapping across all three lineages.
fate_ref = ds.run_fate_mapping(
    pseudotime_ref,
    sink_labels_ref,
    sinks=terminal_labels,
)
fate = ds.load_fate_mapping(fate_ref)
{
    "artifact": fate.ref,
    "pseudotime": fate.pseudotime,
    "sink labels": fate.sink_labels,
    "valid cells": int(fate.valid.sum()),
}
```

The producer writes one artifact containing all probability columns and validity, leaving cell
metadata unchanged. The loaded `fate` object carries the probability matrix (`values`, one row per selected cell), the validity mask (`valid`), the exact producing refs (`ref`, `pseudotime`, `sink_labels`), and the sink label order matching the matrix columns. Keep all of these together in the analysis record: a probability matrix without its pseudotime and sink-label refs cannot be reinterpreted later.

```{code-cell}
valid_probabilities = fate.values[fate.valid]
probability_summary = pd.DataFrame(
    valid_probabilities,
    columns=[str(label) for label in fate.sink_labels],
)
# Theoretical check: absorbing probabilities must sum to 1.0 per cell.
probability_summary["row-sum error"] = np.abs(
    probability_summary.sum(axis=1) - 1.0
)
probability_summary.agg(["min", "median", "max"])
```

The summary above is the depth check for this page. Each row is one valid cell, each probability column is one sink, and `row-sum error` measures how far that cell's probabilities deviate from summing to one. Require finite, non-negative values with row-sum error near zero. Cells excluded by the validity mask do not appear here at all: a clean table over few valid cells can hide a large excluded population, so always compare the valid count against the cluster map before trusting the picture. Rows sum to one because the final sink column is the remainder after the others, so a tidy row sum validates arithmetic, not biology. Cells in graph components containing no sink cells are excluded as invalid, because a walk starting there can never reach a boundary.

The probability panels below are drawn directly with matplotlib: one panel per sink, coloring the shared UMAP by that sink's probability column for valid cells only. Read names, not numbers: each title pairs the sink label with its majority annotation from the table above. Expect smooth color gradients from the progenitor pool into each sink. That smoothness is guaranteed output shape: the solver returns the smoothest interpolation consistent with the pinned boundaries, so a gradient cannot prove cells commit gradually in vivo, where circuits such as Pax4/Arx cross-repression can flip abruptly.

```{code-cell}
umap = np.asarray(ds.load_artifact(analysis_run["umap"])["values"][:])
num_sinks = len(fate.sink_labels)
figure, axes = plt.subplots(1, num_sinks, figsize=(4 * num_sinks, 3.5))
for index, label in enumerate(fate.sink_labels):
    axis = axes[index]
    points = axis.scatter(
        umap[fate.valid, 0],
        umap[fate.valid, 1],
        c=fate.values[fate.valid, index],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=3,
    )
    # Biological lineage name paired with the integer sink label.
    axis.set_title(f"Fate Probability: {label} ({sink_names[int(label)]})")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(points, ax=axis, fraction=0.046, pad=0.04)
figure.tight_layout()
figure
```

## Important caveats to consider regarding fate mapping

- **Probabilities are model summaries, not lineage proof:** terminal probabilities describe where cells sit relative to supervised endpoints on one explicit graph. They cannot establish that a cell becomes a given type, and they inherit every assumption in the source, sink, and component choices.
- **Endpoints must come from markers, not heuristics:** this page resolves Alpha, Beta, and Delta names to integer labels through majority annotation, failing loudly on missing or ambiguous fates. A highest-pseudotime rule would be circular here, and forcing fewer sinks than the tissue's lineages manufactures an artificial tug-of-war. A real claim needs endpoints from independent evidence, plus sensitivity checks across plausible alternatives.
- **Smooth gradients vs. discrete switches:** the absorbing random-walk model generates a mathematically continuous probability gradient across the graph. This does **not** mean in vivo commitment is gradual. If a lineage decision is governed by an abrupt transcriptional switch (e.g., mutual inhibition between *Arx* and *Pax4*), the algorithm still outputs intermediate values (e.g., {math}`P = 0.5`) for cells near the decision boundary simply due to graph neighborhood averaging.
- **Absorbing assumption:** fate mapping assumes every cell eventually reaches one of the defined sinks, so rows always sum to one. Omit an authentic endpoint and its probability mass is forced into the remaining fates: on this store the rare Epsilon lineage is absent from the sinks, and its cells must land in Alpha, Beta, or Delta columns. Audit the sink set against the tissue's known lineages before interpreting shares.
- **Component disconnections:** random walks cannot traverse disconnected graph components. Cells with invalid probability masks (`fate.valid == False`) are typically disconnected from the root or the sinks. Never evaluate fate distributions without first inspecting the proportion of unmapped cells.
- **Keep the ref triple together:** pseudotime, sink-label, and fate refs form one lineage chain. Rebuilding the graph is a new branch, not a repair of the old probabilities.

See {doc}`pseudotime` for the ordering, {doc}`expression_dynamics` for feature modules along the same axis, and {doc}`trajectory_validation` for broader diagnostics.
