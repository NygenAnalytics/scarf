---
description: Aggregate expression along an immutable pseudotime artifact and inspect feature modules.
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

# Expression dynamics along pseudotime

Pseudotime correlation captures monotonic change. Aggregation adds smoothed feature profiles and
clusters them into early, intermediate, and late modules. Both operations persist immutable
artifacts and leave feature metadata unchanged.

## 1. Open the prepared graph and orient it

```{code-cell} ipython3
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
all_features = analysis_run["feature_universe"]

annotations = ds.cells.fetch("clusters", key="I")
source = annotations == "Ductal"
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
if not source.any() or not sink.any():
    raise ValueError("Source and sink annotations must both be present")
source_sink_vector = np.zeros(len(annotations), dtype=float)
source_sink_vector[source] = -1.0 / source.sum()
source_sink_vector[sink] = 1.0 / sink.sum()
pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)
```

The rebuilt catalog store contains the completed `docs_default` pipeline run. This page reuses its
exact graph and feature universe instead of rebuilding preprocessing. The literal `clusters`
column contains the published cell-type annotations used only to orient the trajectory. The new
pseudotime and module results remain exact artifacts.

## 2. Aggregate and cluster feature profiles

```{code-cell} ipython3
modules_ref = ds.run_pseudotime_aggregation(
    pseudotime_ref,
    features=all_features,
    n_clusters=15,
    window_size=200,
    chunk_size=100,
)
modules = ds.load_pseudotime_aggregation(modules_ref)
{
    "artifact": modules.ref,
    "pseudotime": modules.pseudotime,
    "feature selection": modules.feature_selection,
    "assigned features": len(modules.feature_indices),
}
```

The loader returns the valid feature rows only, their physical assay indexes, module labels, frozen
feature names and IDs, and a lazy binned matrix. Use `modules.feature_names` or
`modules.feature_ids` so later edits to live feature metadata cannot relabel the saved result.

```{code-cell} ipython3
feature_names = modules.feature_names
module_frame = pd.DataFrame(
    {
        "feature_name": feature_names,
        "module": modules.feature_clusters,
    }
)
module_frame["module"].value_counts().sort_index()
```

```{code-cell} ipython3
examples = (
    module_frame.groupby("module", sort=True)["feature_name"]
    .first()
    .rename("example gene")
)
examples
```

The first feature is a compact example for each module, not a ranked representative.

## 3. Inspect the ordered profiles

```{code-cell} ipython3
ds.plots.pseudotime_heatmap(
    aggregation=modules_ref,
    figsize=(8, 8),
)
```

A useful result contains coherent early, intermediate, and late patterns rather than one block of
uniformly expressed genes. Module numbers are labels, not developmental stages. Inspect example
genes and module stability before assigning biological meaning.

## 4. Build a grouped assay when needed

The aggregation ref can be consumed directly when a module-level assay is useful downstream:

```{code-cell} ipython3
ds.add_grouped_assay(
    modules_ref,
    assay_label="TrajectoryModules",
)
ds.TrajectoryModules
```

This is an explicit new-assay construction step. It does not write the module labels into the
RNA feature table. A feature metadata column can be passed instead of an artifact when the groups
were deliberately authored as metadata.

## Common mistakes and limitations

- Treating module order as a causal sequence
- Comparing modules built from different feature or pseudotime refs as if inputs matched
- Ignoring invalid features removed by expression or variance checks
- Using one trajectory orientation when plausible source and sink alternatives remain

See {doc}`pseudotime` for scoring and {doc}`trajectory_validation` for broader checks.
