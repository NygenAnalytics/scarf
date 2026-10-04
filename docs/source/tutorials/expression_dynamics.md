---
description: Follow gene expression along pseudotime and inspect groups with similar profiles.
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
# Expression dynamics primer

Pseudotime analysis orders cells in comparison to a starting and end point, but ordering alone does not say what actual changes we observe along the way. For example, some genes climb or fall steadily from progenitor to terminus; others switch on briefly in the middle and off again. A correlation score catches the steady ones and misses the transient ones, because a rise-and-fall pattern has no overall trend to correlate with.

**Expression dynamics** fills that gap by smoothing each gene's expression along the pseudotime ordering, then grouping genes with similar smoothed profiles into modules. Each module is one shared trajectory shape: early genes fading out, late genes turning on, intermediate genes peaking mid-path. Here, we reuse the pancreas ordering to build those modules.

# Follow gene expression along pseudotime

Some genes increase or decrease steadily along a process. Others rise briefly and fall
again, which a correlation score can miss. Here we smooth expression along the pancreas
pseudotime ordering and group genes with similar profiles into **modules**.

## Pull the completed analysis

```{code-cell}
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
```

Open the downloaded store and its saved analysis.

```{code-cell}
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
analysis_run = ds.pipeline.open(label="docs_default")
graph = analysis_run["connectivity_map"]
all_features = analysis_run["feature_universe"]
ds

annotations = ds.cells.fetch("clusters", key="I")
source = annotations == "Ductal"
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any():
    raise ValueError("Source and sink annotations must both be present")
source_sink_vector = np.zeros(len(annotations), dtype=float)
source_sink_vector[source] = -1.0 / source.sum()
source_sink_vector[sink] = 1.0 / sink.sum()
pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
pd.Series({"source cells": int(source.sum()), "sink cells": int(sink.sum())})
```

We can utilize the existing annotations to orient our graph to study the expression dynamics. Similarly as in the pseudotime tutorial, the source and pooled sinks each receive a total mass of one, with opposite signs.

## Group changing expression profiles

SCARF orders the valid cells, those with pseudotime scores inside the graph, then smooths each retained gene over a 200-cell rolling window successively across them. When we say smooths over the trajectories, we mean replacing each cell's own [noisy/sparse] measurement with the average of its 200-cell neighborhood along the ordering, so shared trends emerge while cell-to-cell jitter cancels out. The smoothed trajectories are summarized into 50 ordered slots from early to late (relative based on the pseudotime), and the genes' smoothed profiles are then clustered into 10 modules, which we visualize with the heatmap below.

```{code-cell}
modules_ref = ds.run_pseudotime_aggregation(pseudotime_ref, features=all_features)
ds.plots.pseudotime_heatmap(aggregation=modules_ref)
```

The heatmap can be interpreted with the far left suggesting early on the pseudotime and the further right being later among the pseudotime. By default, each gene is scaled relative to its own variation, thus red indicates higher expression and blue lower expression for that specific gene; the colors do not show which gene has the greatest absolute expression!

## Inspect a module's genes

Load the saved result to see how many genes each module contains:

```{code-cell}
modules = ds.load_pseudotime_aggregation(modules_ref)
module_genes = pd.DataFrame({
    "gene": modules.feature_names,
    "module": modules.feature_clusters,
})
module_genes.groupby("module").size().rename("genes")
```

Choose a module from the heatmap, then list its genes. The example below selects the
first module label only to show the lookup; the returned genes are not ranked markers.

```{code-cell}
module_id = module_genes["module"].min()
module_genes.loc[module_genes["module"] == module_id, "gene"].head(20)
```

## Adjust smoothing only when needed

A wide smoothing window (# of cells) can hide a brief expression peak, whereas a narrower window can retain more noise. To modify the smoothing window for your dataset, simply run the below:

```python
modules_ref = ds.run_pseudotime_aggregation(
    pseudotime_ref, features=all_features, window_size=100
)
```

To modify the number of modules, add the hyperparameter `n_clusters`, which controls the requested number of modules. The default for `n_clusters` is 10. To control the number of pseudotime bins, pass in `chunk_size`, in which the default is 50. Change one choice at a time and compare the profiles before interpreting a split or merged module.

## Common mistakes and limitations

- **Lineage dilution from pooling branched endpoints:** Pooling distinct terminal populations (e.g., Alpha, Beta, and Delta cells) into a single trajectory collapses multiple diverging paths into one final state. Averaging expression across mutually exclusive fates blurs branch-specific dynamics, causing lineage-restricted drivers (e.g., Arx vs. Pax4) to appear artificially muted, diluted, or conflicting along the visualized shared axis.
- **Smoothing window artifacts and hyperparameter sensitivity:** The sliding window (window_size) creates a strict trade-off between technical noise reduction and temporal resolution. A window that is too wide oversmooths sharp, transient regulatory pulses (such as fleeting transcription factor spikes), whereas a window that is too narrow fits to stochastic dropout noise. Sample multiple different parameters to identify what best works for your question.
- **Conflating relative kinetic shapes with expression magnitude and co-regulation:** The pseudotime heatmap standardizes each gene relative to its own variance, making low-abundance, noisy transcripts appear as visually pronounced as major lineage-defining effectors. Furthermore, sharing a kinetic expression curve along pseudotime reflects temporal correlation, not shared upstream regulation; genes within the same module do not necessarily share transcription factor motifs or common regulatory network.

See {doc}`trajectory_validation` for broader checks on orientation choices, validity masks, and endpoint sensitivity, and {doc}`fate_mapping` for splitting each cell's outcome across multiple terminal fates instead of a single axis position.
