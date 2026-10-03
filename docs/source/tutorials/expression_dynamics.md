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
# Follow gene expression along pseudotime

Some genes increase or decrease steadily along a process. Others rise briefly and fall
again, which a correlation score can miss. Here we smooth expression along the pancreas
pseudotime ordering and group genes with similar profiles into **modules**.

Start with {doc}`pseudotime` for the endpoint choices and scoring method. This page uses
the same ordering and focuses on how to read the expression heatmap.

## Recreate the ordering

```{code-cell}
import numpy as np
import pandas as pd

import scarf

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example, including its saved analysis.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
```

Open the downloaded store and its saved analysis.

```{code-cell} ipython3
# Open the datastore for the following analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Reuse the saved run and its frozen cell selection.
analysis_run = ds.pipeline.open(label="docs_default")
# Keep the graph used by the saved analysis.
graph = analysis_run["connectivity_map"]
# Use the complete feature universe for this assay.
all_features = analysis_run["feature_universe"]
# Inspect the opened assays and their dimensions.
ds
```

Use the published cell-type annotations to orient the graph. As in the pseudotime
example, the source and pooled sinks each receive a total mass of one, with opposite signs.

```{code-cell}
# Read the published cell-type labels for the selected cells.
annotations = ds.cells.fetch("clusters", key="I")
# Mark ductal cells as the source population.
source = annotations == "Ductal"
# Mark the pooled terminal populations.
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
# Require at least one annotated source cell and one sink cell.
if not source.any() or not sink.any():
    raise ValueError("Source and sink annotations must both be present")
# Start every graph cell with zero source or sink mass.
source_sink_vector = np.zeros(len(annotations), dtype=float)
# Share total mass -1 among the source cells.
source_sink_vector[source] = -1.0 / source.sum()
# Share total mass +1 among the pooled sink cells.
source_sink_vector[sink] = 1.0 / sink.sum()
# Orient the graph using the chosen source and sinks.
pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
# Check how many cells anchor the two ends of the ordering.
pd.Series({"source cells": int(source.sum()), "sink cells": int(sink.sum())})
```

## Group changing expression profiles

Start with the default aggregation settings. Scarf orders valid cells by pseudotime,
smooths each retained gene over a 200-cell window, summarizes it in 50 bins, and groups
similar profiles into 10 modules.

```{code-cell}
# Group genes with similar expression profiles along pseudotime.
modules_ref = ds.run_pseudotime_aggregation(pseudotime_ref, features=all_features)
# Display the ordered expression profiles and their modules.
ds.plots.pseudotime_heatmap(aggregation=modules_ref)
```

Read from early to late pseudotime across the heatmap. Look for groups that peak early,
late, or in the middle. By default, each gene is scaled relative to its own variation.
Red indicates higher expression and blue lower expression for that gene; the colours do
not show which gene has the greatest absolute expression.

Module numbers are labels, not developmental stages. Ten modules is a starting choice,
not a claim that the process has ten biological programs.

## Inspect a module's genes

Load the saved result to see how many genes each module contains:

```{code-cell}
# Load the saved gene-module assignments.
modules = ds.load_pseudotime_aggregation(modules_ref)
# Pair each retained gene with its module label.
module_genes = pd.DataFrame({
    "gene": modules.feature_names,
    "module": modules.feature_clusters,
})
# Count the genes assigned to each expression module.
module_genes.groupby("module").size().rename("genes")
```

Choose a module from the heatmap, then list its genes. The example below selects the
first module label only to show the lookup; the returned genes are not ranked markers.

```{code-cell}
# Choose the first module to demonstrate gene lookup.
module_id = module_genes["module"].min()
# Inspect the first twenty genes in the chosen module.
module_genes.loc[module_genes["module"] == module_id, "gene"].head(20)
```

Check whether several genes support a shared process before naming the module. Follow
up candidate genes with marker maps or other independent evidence.

## Adjust smoothing only when needed

A wide window can hide a brief expression peak; a narrow one can retain more noise.
To explore a shorter window, repeat the call with one changed setting:

```python
# Group genes with similar expression profiles along pseudotime.
modules_ref = ds.run_pseudotime_aggregation(
    pseudotime_ref, features=all_features, window_size=100
)
```

`n_clusters` controls the requested number of modules. `chunk_size` controls the number
of displayed pseudotime bins, not a memory batch size. Change one choice at a time and
compare the profiles before interpreting a split or merged module.

## Limits of these modules

- Expression or variance checks exclude some features; the loaded result contains the
  retained features only.
- Genes with similar profiles are not necessarily regulated by the same mechanism.
- A single pooled ordering can obscure changes specific to one branch.
- Compare plausible endpoint and smoothing choices before treating a module as stable.

See {doc}`trajectory_validation` for broader checks and {doc}`fate_mapping` for multiple
terminal outcomes.
