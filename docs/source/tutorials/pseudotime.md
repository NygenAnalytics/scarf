---
description: Order pancreas cell states and find genes associated with pseudotime.
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
# Order cell states with pseudotime

A single-cell experiment measures cells at one time point, but it may capture several
stages of a process. **Pseudotime** places those cells along a graph-based coordinate.
In Scarf, chosen starting and terminal populations orient that coordinate. It does not
measure elapsed time or establish which cells give rise to others.

We use a developing-pancreas dataset to order cells from Ductal-like progenitors toward
Alpha, Beta, and Delta populations, then find genes associated with that ordering.

## Open the prepared analysis

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
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
analysis_run = ds.pipeline.open(label="docs_default")
graph = analysis_run["connectivity_map"]
all_features = analysis_run["feature_universe"]
```

The saved analysis supplies the graph, UMAP, and feature selection. The store's live
`clusters` column contains the published cell-type annotations; these differ from the
numbered clusters calculated by the pipeline.

## Choose a start and endpoints

We use the published Ductal annotation as the source and pool Alpha, Beta, and Delta
as sinks. This is a biological assumption to check with marker evidence, not something
pseudotime discovers for us.

The vector below assigns a total of -1 to the source and +1 to the pooled sinks.
All other cells receive zero. Sharing each total across its cells keeps source and sink
mass balanced despite their different sizes.

```{code-cell}
labels = ds.cells.fetch("clusters", key="I")
source = labels == "Ductal"
sink = np.isin(labels, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any():
    raise ValueError("Source and sink labels must both be present")
source_sink_vector = np.zeros(len(labels), dtype=float)
source_sink_vector[source] = -1.0 / source.sum()
source_sink_vector[sink] = 1.0 / sink.sum()
```

## Score and view pseudotime

```{code-cell}
pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
pseudotime = ds.load_pseudotime_scoring(pseudotime_ref)
ds.cells.insert("pseudotime", pseudotime.values, key="I", overwrite=True)
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by="pseudotime",
    sort_values=True,
)
```

Low values should lie toward the Ductal region and high values toward the endocrine
endpoints. The default rescales the score from zero to one. Intermediate values are
expected for intermediate populations; the spacing is not a clock.

Check the score within each annotation, using only cells marked valid:

```{code-cell}
pd.DataFrame(
    {
        "cell type": labels[pseudotime.valid],
        "pseudotime": pseudotime.values[pseudotime.valid],
    }
).groupby("cell type")["pseudotime"].describe()
```

In this example, Ductal cells have low scores, endocrine endpoints have high scores,
and Ngn3-high progenitors occupy the middle. A pattern that disagrees with known markers
is a reason to revisit the graph or endpoint choice.

## Find genes associated with the ordering

```{code-cell}
marker_ref = ds.run_pseudotime_marker_search(pseudotime_ref, features=all_features)
markers = ds.load_pseudotime_markers(marker_ref)
tested = markers.table.loc[
    markers.table["p_value_adjusted"].notna(),
    ["feature_name", "r_value", "p_value_adjusted"],
]
increasing = tested.loc[tested["r_value"] > 0].nlargest(10, "r_value")
decreasing = tested.loc[tested["r_value"] < 0].nsmallest(10, "r_value")
pd.concat({"increasing": increasing, "decreasing": decreasing})
```

Positive correlations identify genes whose expression tends to increase along the
chosen axis; negative correlations identify decreasing genes. These are candidates to
inspect, not evidence that they drive development. Untested features have missing
p-values, and multiple-testing correction covers only the tested features.

## Check the limits

- Source and sink choices supervise the ordering. Try plausible alternatives before
  making a developmental claim.
- By default, Scarf scores the largest connected graph component. Other cells have
  `valid=False` and undefined values. Use the validity mask for your own summaries;
  Scarf's marker and aggregation methods apply it automatically.
- Correlation can miss genes that rise and fall along the path. Use
  {doc}`expression_dynamics` to inspect such profiles.

Continue to {doc}`fate_mapping` for several candidate terminal outcomes, or
{doc}`trajectory_validation` for checks across graph and endpoint choices.
