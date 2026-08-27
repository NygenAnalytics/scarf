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

## 1. Build an oriented graph

```{code-cell} ipython3
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = f"{analysis_directory.name}/counts.zarr"
repack_store(f"{dataset}/data.zarr", repacked_counts, nthreads=2)
ds = scarf.mount_datastore(
    repacked_counts,
    at=f"{analysis_directory.name}/analysis.zarr",
    nthreads=4,
    default_assay="RNA",
)

cell_selection = ds.snapshot_cell_selection("I")
hvg = ds.select_hvgs(cell_selection, top_n=2000, show_plot=False)
normalized = ds.run_normalization(cell_selection, hvg)
pca = ds.run_pca(normalized, dims=15)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
graph = ds.build_connectivity_map(neighbors)

annotations = np.asarray(ds.cells.fetch("clusters", key="I"))
source = annotations == "Ductal"
sink = np.isin(annotations, ["Alpha", "Beta", "Delta"])
source_sink_vector = np.zeros(len(annotations), dtype=float)
source_sink_vector[source] = 1.0 / int(source.sum())
source_sink_vector[sink] = -1.0 / int(sink.sum())
pseudotime_ref = ds.run_pseudotime_scoring(graph, ss_vec=source_sink_vector)

all_features = ds.set_feature_selection(
    from_assay="RNA",
    feature_indexes=range(ds.RNA.feats.N),
)
```

The `clusters` labels used to orient this example are prepared catalog metadata copied by the mount.
The new pseudotime and module results remain exact artifacts.

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

The loader returns the valid feature rows only, their physical assay indexes, module labels, and a
lazy binned matrix.

```{code-cell} ipython3
feature_names = np.asarray(ds.RNA.feats.fetch_all("names"))[modules.feature_indices]
module_frame = pd.DataFrame(
    {
        "feature_name": feature_names,
        "module": modules.feature_clusters,
    }
)
module_frame["module"].value_counts().sort_index()
```

```{code-cell} ipython3
representatives = (
    module_frame.groupby("module", sort=True)["feature_name"]
    .first()
    .rename("representative gene")
)
representatives
```

## 3. Inspect the ordered profiles

```{code-cell} ipython3
ds.plots.pseudotime_heatmap(
    aggregation=modules_ref,
    figsize=(8, 8),
)
```

A useful result contains coherent early, intermediate, and late patterns rather than one block of
uniformly expressed genes. Module numbers are labels, not developmental stages. Verify
representative genes and stability before assigning biological meaning.

## 4. Build a grouped assay when needed

The aggregation ref can be consumed directly when a module-level assay is useful downstream:

```{code-cell} ipython3
ds.add_grouped_assay(
    modules_ref,
    assay_label="TrajectoryModules",
)
ds.TrajectoryModules
```

This is an explicit new-assay construction step. It does not publish the module labels into the
RNA feature table. A feature metadata column can be passed instead of an artifact when the groups
were deliberately authored as metadata.

## Common mistakes and limitations

- Treating module order as a causal sequence
- Comparing modules built from different feature or pseudotime refs as if inputs matched
- Ignoring invalid features removed by expression or variance checks
- Using one trajectory orientation when plausible source and sink alternatives remain

See {doc}`pseudotime` for scoring and {doc}`trajectory_validation` for broader checks.
