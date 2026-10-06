---
description: Compare immutable Leiden and Paris clustering artifacts and inspect cluster evidence.
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

# Clustering and cluster evidence

Clustering groups cells with similar neighbours. A cluster may represent a cell type, a cell
state, or technical variation, so it needs biological interpretation. We first inspect a saved
clustering, then change the resolution while keeping the graph fixed.

## 1. Open one graph

```{code-cell} ipython3
from dataclasses import asdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example, including its saved analysis.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq", destination="scarf_datasets", zarr=True
)
```

Open the downloaded store and its saved analysis.

```{code-cell} ipython3
# Open the datastore for the following analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Open the saved clustering analysis.
clustering_run = ds.pipeline.open(label="docs_default")
# Keep the graph used by the saved analysis.
graph = clustering_run["connectivity_map"]
# Reuse the saved UMAP coordinates.
umap = clustering_run["umap"]
# Inspect the opened assays and their dimensions.
ds
```

The store carries an analysis saved as `docs_default`. Inspect its selected clusters first:

```{code-cell} ipython3
# Inspect the prepared baseline clustering.
ds.plots.embedding(run=clustering_run, color_by="clusters")
```

This prepared example used Leiden resolution 0.5. The standalone method defaults to 1.0, and the
standard pipeline can compare several resolutions. Cluster numbers are labels, not cell types.
We use the same graph and UMAP below so only the clustering changes.

## 2. Sweep Leiden resolution

```{code-cell} ipython3
# Vary the resolution while keeping the graph unchanged.
leiden_refs = {
    0.3: ds.run_leiden_clustering(graph, resolution=0.3),
    0.5: clustering_run["leiden_0.5"],
    0.8: ds.run_leiden_clustering(graph, resolution=0.8),
}
# Load the labels for each resolution.
leiden_values = {
    resolution: np.asarray(ds.load_artifact(ref)["values"][:])
    for resolution, ref in leiden_refs.items()
}

# Compare cluster sizes across the three Leiden resolutions.
pd.DataFrame(
    {
        resolution: pd.Series(values).value_counts()
        for resolution, values in leiden_values.items()
    }
).fillna(0).astype(int)
```

```{code-cell} ipython3
# Create one plotting axis for each comparison panel.
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
# Draw each comparison on its own labeled axis.
for axis, resolution in zip(axes, leiden_values, strict=True):
    # Place this resolution on the common UMAP for comparison.
    ds.plots.embedding(
        layout=umap,
        color_by=leiden_refs[resolution],
        target=axis,
        show_titles=False,
        show=False,
    )
    # Label the panel with the quantity being compared.
    axis.set_title(f"Leiden {resolution}")
# Adjust spacing so panel labels remain readable.
figure.tight_layout()
# Display the completed figure.
figure
```

Higher resolution usually produces more and smaller groups. Reject a split when it is driven by a
technical covariate, has weak marker evidence, or disappears under a modest parameter change.

The adjusted Rand index (ARI) compares partitions without requiring the cluster numbers to
match. A value of one means the partitions agree. It does not tell us which partition is better:

```{code-cell} ipython3
# Compare partition agreement for every resolution pair.
pd.Series(
    {
        f"{first} vs {second}": ds.metric_label_concordance(
            leiden_refs[first], leiden_refs[second]
        )
        for first, second in combinations(leiden_refs, 2)
    },
    name="ARI",
)
```

## 3. Inspect membership strength

Membership strength measures how strongly a cell connects to its assigned cluster: it is the
fraction of the cell's graph neighbours that carry the cell's own cluster label. A value of 1 means
that every neighbour shares the cell's cluster. A cell whose neighbours sit mostly in other
clusters scores low, even when those neighbours all share one other label.
We use resolution 0.5 for this walkthrough.

```{code-cell} ipython3
# Inspect the Leiden partition at resolution 0.5.
chosen = leiden_refs[0.5]
# Keep the labels for the partition being reviewed.
chosen_values = leiden_values[0.5]
# Measure each cell's connection to its assigned cluster.
membership = ds.calc_membership_strength(chosen, graph)
# Locate cells with weak or strong cluster membership.
ds.plots.embedding(layout=umap, color_by=membership)
```

```{code-cell} ipython3
# Show how the chosen clusters connect on this graph.
ds.plots.cluster_connectivity(graph=graph, groups=chosen, layout=umap)
```

Low values throughout one cluster suggest a weak boundary: many of its cells have most of their
neighbours in other clusters. A narrow band of low values between otherwise coherent groups may
represent continuous biology.

## 4. Review marker evidence

Marker search requires exact cluster and feature-selection refs and returns one immutable marker
table artifact.

```{code-cell} ipython3
# Keep the marker result for the selected clustering.
markers = ds.run_marker_search(chosen, features=clustering_run["feature_universe"])
# Count the cells in each selected cluster.
sizes = pd.Series(chosen_values).value_counts()
# Identify the most populated cluster.
largest = sizes.index[0]
# Identify the least populated cluster.
smallest = sizes.index[-1]

# Load markers for the largest cluster.
largest_markers = ds.get_markers(marker=markers, group_id=largest)
# Load markers for the smallest cluster.
smallest_markers = ds.get_markers(marker=markers, group_id=smallest)
# Identify the largest and smallest clusters being compared below.
pd.Series({"largest cluster": largest, "smallest cluster": smallest})
```

Markers for the largest cluster:

```{code-cell} ipython3
# Inspect the leading markers for the largest cluster.
marker_columns = ["feature_name", "score", "auc", "p_value", "p_value_adjusted"]
# Preview these statistics for the ten leading markers.
largest_markers[marker_columns].head(10)
```

Markers for the smallest cluster:

```{code-cell} ipython3
# Inspect the leading markers for the smallest cluster.
smallest_markers[marker_columns].head(10)
```

The p-values are cell-level one-versus-rest marker tests with within-group adjustment. They are not
replicate-aware differential expression. A defensible partition combines marker evidence, graph
support, technical covariates, replicate coverage, and the study question.

## 5. Pipeline cluster selection

When a pipeline run includes multiple Leiden candidates, it scores them with one deterministic
shared sample of at most 10,000 cells in the graph's PCA or Harmony coordinates or, with
`pca_dims=0`, its normalized values. Paris can still run as `clustering_run["paris"]`, but it is
not an automatic winner. The `cluster_selection` artifact persists the scores, sampling policy,
invalid-candidate reasons, tie order, and selected key:

```python
# Keep the saved cluster-selection diagnostics.
decision_ref = clustering_run["cluster_selection"]
# Keep the clustering chosen by the pipeline.
selected_cluster_ref = clustering_run["clusters"]
```

This automatic choice is a reproducible baseline, not proof that the selected resolution is best
for every biological question. Retain alternative refs when the decision needs domain-specific
evidence.

## Optional: compare Paris cuts

`run_paris_clustering` returns a `cluster_cut` ref. Load the domain result explicitly when
hierarchy diagnostics are needed.

```{code-cell} ipython3
# Choose an adaptive cut of the Paris hierarchy.
paris_auto = ds.run_paris_clustering(graph)
# Load the Paris labels and hierarchy diagnostics.
paris_result = ds.load_paris_clustering(paris_auto)
# Inspect the size and persistence of each selected Paris group.
pd.DataFrame([asdict(item) for item in paris_result.diagnostics])[
    ["label", "size", "persistence", "decision_margin", "forced"]
]
```

```{code-cell} ipython3
# Show the hierarchy supporting the Paris partition.
ds.plots.cluster_tree(graph=graph, clusters=paris_auto)
```

Persistence measures how long a selected branch survives in the hierarchy. The decision margin
measures the preference for retaining it. A forced group satisfies a structural constraint and is
not, by itself, strong biological evidence.

```{code-cell} ipython3
# Cut the same hierarchy at the adaptive result's cluster count.
paris_fixed = ds.run_paris_clustering(graph, n_clusters=paris_result.n_clusters)
# Compare the adaptive Paris cut with fixed Paris and Leiden partitions.
pd.Series(
    {
        "auto vs fixed ARI": ds.metric_label_concordance(paris_auto, paris_fixed),
        "Leiden vs Paris ARI": ds.metric_label_concordance(chosen, paris_auto),
    }
)
```
