---
description: Select representative cells with TopACeDo and export a deliberate subset.
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
# Cell downsampling (primer)

A large dataset is often easier to explore, or easier to pass to another tool, when you work with
a smaller set of representative cells. Random subsampling is a poor way to do this: it can drop
rare populations and thin out the structure you wanted to keep. SCARF's TopACeDo instead uses the neighbourhood graph, built as in {doc}`graph_construction`, together with a Paris clustering to
choose cells that cover that structure and thus effectively downsample. It selects seed cells within each cluster and then adds the cells that connect those seeds through the graph. The selection is saved as an artifact, so the source data is never changed.

This page walks the full path: open the graph and the Paris cut the sampler requires, run the
sampler, inspect what it selected, and export the chosen cells into a separate store. The sample
is chosen to preserve topology, not to be statistically interchangeable; the caveats at the end
state what it can and cannot support.

A smaller set of representative cells can make a large dataset easier to explore or pass to
another tool. TopACeDo uses the neighbourhood graph and a Paris clustering to choose cells that
cover its structure. It saves the selected cells as an artifact; it does not change the source data.

## 1. Open the required artifacts

TopACeDo requires a Paris cut from the same graph. The rebuilt PBMC store contains a completed
example run labeled `docs_default` and a 15-cluster Paris cut built from its graph. Calling
`run_paris_clustering` with that graph and cut size reuses the exact stored result. A Leiden
partition or a cut from another graph is rejected.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
```

Open the downloaded store and its saved analysis.

```{code-cell} ipython3
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
graph = run["connectivity_map"]
umap = run["umap"]
paris = ds.run_paris_clustering(graph, n_clusters=15)
ds
```

`paris` is an exact `cluster_cut` ref. Inspect its labels through the dedicated loader:

```{code-cell} ipython3
paris_result = ds.load_paris_clustering(paris)
cluster_counts = pd.Series(paris_result.labels, name="cluster").value_counts()
cluster_counts.sort_index().to_frame("cells")
```

## 2. Choose representative cells

```{code-cell} ipython3
sampling = ds.run_topacedo_sampler(graph, paris)
sampling_data = ds.load_artifact(sampling)
sampled = np.asarray(sampling_data["sampled"][:], dtype=bool)
{"cells": int(sampled.size), "selected": int(sampled.sum())}
```

The default 5% rate controls seed selection within each cluster, with a minimum number of
seeds per cluster. The sampler then adds cells that connect those seeds through the graph.
The final sample can therefore exceed 5%. Check the selected count above; this is not a
request for an exact sample size.

```{code-cell} ipython3
coordinates = np.asarray(ds.load_artifact(umap)["values"][:])
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(coordinates[:, 0], coordinates[:, 1], s=3)
axes[0].set_title("All selected cells")
axes[1].scatter(coordinates[sampled, 0], coordinates[sampled, 1], s=3)
axes[1].set_title("TopACeDo sample")
figure.tight_layout()
```

Downsampling preserves graph coverage. It does not make the sample a statistically interchangeable
replacement for the complete dataset.

## 3. Export the selected cells

The sampler mask follows the graph's compact cell-selection order. Map it back to physical row
indices, then pass those rows directly to `SubsetZarr`. No temporary metadata column is needed.

```{code-cell} ipython3
graph_cell_indices = np.flatnonzero(run.cells.fetch_all("I"))
sampled_cell_indices = graph_cell_indices[sampled]

{"cells to export": len(sampled_cell_indices)}
```

Write those physical rows into a separate store.

```{code-cell} ipython3
export_directory = TemporaryDirectory()
subset_path = Path(export_directory.name) / "subset.zarr"
writer = scarf.SubsetZarr(
    zarr_loc=str(subset_path),
    assays=[ds.RNA],
    cell_idx=sampled_cell_indices,
    reset_cell_filter=False,
    overwrite_existing_file=True,
)
writer.dump()

subset = scarf.DataStore(str(subset_path))
{"exported cells": subset.cells.N, "retained genes": subset.RNA.feats.N}
```

`SubsetZarr` retains every feature in the listed assays. Use `to_anndata` when you need an in-memory
handoff with both axes constrained.

## 4. Inspect coverage and adjust the sample

Check how many cells were retained from each Paris cluster before choosing a different rate:

```{code-cell} ipython3
summary = pd.DataFrame({"cluster": paris_result.labels, "sampled": sampled})
summary.groupby("cluster")["sampled"].agg(cells="size", selected="sum")
```

If you need a larger sample, pass `max_sampling_rate=0.1` to raise the maximum seed-selection
rate to 10% per cluster. The minimum-per-cluster rule and added connecting cells still affect
the final size. Compare coverage on the original layout after changing the rate.

The artifact also contains seed cells, density estimates, shared-neighbour summaries, and the
selected graph edges. These are available through `load_artifact` when you need to inspect how
the sampler made its choices.

## Important caveats to consider regarding downsampling

- **Conflating topological coverage with quantitative frequency representation:** TopACeDo samples cluster seeds and graph-connecting paths to preserve global and local manifold geometry, which deliberately inflates the representation of rare cell states and underrepresents dense populations. Treating this topology-preserving subset as a statistically representative subsample will severely distort cell-type proportions, differential abundance testing, and donor-level cell frequency comparisons.
- **Treating max_sampling_rate as a fixed target sample size:** The sampling rate parameter (default 5%) only controls initial seed selection within each Paris cluster. Because TopACeDo enforces a minimum seed count per cluster and subsequently pulls in intermediate graph neighbors to preserve structural connectivity, the final exported sample size frequently and unpredictably exceeds the nominal percentage, especially in datasets with numerous clusters or fragmented graphs.
- **Graph-cluster artifact decoupling and selection-order indexing errors:** TopACeDo requires a dedicated Paris hierarchy cut (cluster_cut) derived from the exact same graph artifact; supplying Leiden partitions or cuts from an alternate graph will fail. Furthermore, the resulting boolean mask aligns with the graph's compact cell selection rather than the raw datastore rows, so exporting directly without mapping back to physical store indices (run.cells.fetch_all("I")) corrupts cell identities in the exported subset store.
