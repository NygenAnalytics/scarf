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
# Clustering primer

Cells that share neighbors on the graph get grouped into clusters. These neighbors that the cells share all contain similar transcriptomic profiles, thus why they end up nearby. However, a cluster is a mathematical partition, not a biological verdict. Clustering may mark a cell type, a transient state, or a technical artifact; only marker evidence and stability checks ensure that it is a truly biologically similar group. The Leiden clustering method draws boundaries on the neighbor graph by optimizing modularity at a chosen resolution, so turning the resolution knob merges rare populations at coarse settings and fractures homogeneous ones at fine settings, all on the same graph. Higher values for the Leiden resolution mean more communities, whereas smaller values indicate fewer, broader communities.

The `Paris` clustering method instead builds a hierarchy of nested merges, letting you cut adaptively or at a fixed cluster count.

Here, we inspect a saved Leiden clustering on the prepared PBMC analysis, change the resolution while holding the graph fixed, and compare against the Paris hierarchy, judging every neighborhood by marker coherence and agreement rather than by count alone.

# Clustering and cluster evidence

## Open the existing graph

```{code-cell}
from dataclasses import asdict
from itertools import combinations

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

```{code-cell}
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
clustering_run = ds.pipeline.open(label="docs_default")
graph = clustering_run["connectivity_map"]
umap = clustering_run["umap"]
ds
```

The store carries an analysis saved as `docs_default`. Inspect its selected clusters first:

```{code-cell}
ds.plots.embedding(run=clustering_run, color_by="clusters")
```

This prepared example used Leiden resolution 0.5, with the default of SCARF being 1.0. During standard analysis, it is critical that you sweep across several resolutions.
We use the same graph and UMAP below so only the clustering changes.

## Sweep Leiden resolution

```{code-cell}
leiden_refs = {
    0.3: ds.run_leiden_clustering(graph, resolution=0.3),
    0.5: clustering_run["leiden_0.5"],
    0.8: ds.run_leiden_clustering(graph, resolution=0.8),
}
leiden_values = {
    resolution: np.asarray(ds.load_artifact(ref)["values"][:])
    for resolution, ref in leiden_refs.items()
}

pd.DataFrame(
    {
        resolution: pd.Series(values).value_counts()
        for resolution, values in leiden_values.items()
    }
).fillna(0).astype(int)
```

```{code-cell}
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, resolution in zip(axes, leiden_values, strict=True):
    ds.plots.embedding(
        layout=umap,
        color_by=leiden_refs[resolution],
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(f"Leiden {resolution}")
figure.tight_layout()
figure
```

Higher resolution usually produces more and smaller groups. Reject a split when it is driven by a technical covariate, has weak marker evidence, or disappears under a modest parameter change.

A metric to validate clustering is the adjusted Rand index (ARI), which compares partitions without requiring the cluster numbers to match. A value of one means the partitions agree, but it does not tell us which partition is better:

```{code-cell}
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

## Inspect membership strength

Membership strength measures how strongly a cell connects to its assigned cluster.
We use resolution 0.5 for this walkthrough.

```{code-cell}
chosen = leiden_refs[0.5]
chosen_values = leiden_values[0.5]
membership = ds.calc_membership_strength(chosen, graph)
ds.plots.embedding(layout=umap, color_by=membership)
```

```{code-cell}
ds.plots.cluster_connectivity(graph=graph, groups=chosen, layout=umap)
```

Low values throughout one cluster suggest a weak boundary. A narrow band of low values between otherwise coherent groups may represent continuous biology.

## Review marker evidence

Marker search requires exact cluster and feature-selection refs and returns a final table to interpret.

```{code-cell}
markers = ds.run_marker_search(chosen, features=clustering_run["feature_universe"])
sizes = pd.Series(chosen_values).value_counts()
largest = sizes.index[0]
smallest = sizes.index[-1]

largest_markers = ds.get_markers(marker=markers, group_id=largest)
smallest_markers = ds.get_markers(marker=markers, group_id=smallest)
pd.Series({"largest cluster": largest, "smallest cluster": smallest})
```

Markers for the largest cluster:

```{code-cell}
marker_columns = ["feature_name", "score", "auc", "p_value", "p_value_adjusted"]
largest_markers[marker_columns].head(10)
```

Markers for the smallest cluster:

```{code-cell}
smallest_markers[marker_columns].head(10)
```

The p-values are cell-level one-versus-rest marker tests with within-group adjustment. They are not replicate-aware differential expression: you can't report the p-values as direct changes in gene expression.

## Pipeline cluster selection

Sometimes you run the full pipeline instead of picking a resolution yourself, and it produces several Leiden candidates at once. Scarf does not just grab one silently, it scores every candidate on the same shared sample of at most 10,000 cells in PCA or Harmony coordinates, so the contest is fair and reruns give the same winner. Paris can enter the equation through `clustering_run["paris"]`. Everything about the decision is saved in the `cluster_selection` artifact, with the scores, the sampling policy, why candidate resolutions failed, where they tied, and the winning key resolution.

```python
decision_ref = clustering_run["cluster_selection"]
selected_cluster_ref = clustering_run["clusters"]
```

The winner is simply a reproducible starting point.

## Optional: compare Paris clusters

Leiden is not the only way to divide up the graph, as previously discussed at the beginning, Paris clustering is also a method. `run_paris_clustering` builds a hierarchy of nested merges and returns a `cluster_cut` ref; load the result explicitly when you need its diagnostics, and not just its labels.

```{code-cell}
paris_auto = ds.run_paris_clustering(graph)
paris_result = ds.load_paris_clustering(paris_auto)
pd.DataFrame([asdict(item) for item in paris_result.diagnostics])[
    ["label", "size", "persistence", "decision_margin", "forced"]
]
```

```{code-cell}
ds.plots.cluster_tree(graph=graph, clusters=paris_auto)
```

Persistence measures how long a selected branch survives in the hierarchy, with long-lived branches as the sturdier claims. The decision margin measures how strongly the cut prefers keeping it. A forced group exists because a structural constraint demanded it, not because the data supported it, so it carries no biological weight on its own.

```{code-cell}
paris_fixed = ds.run_paris_clustering(graph, n_clusters=paris_result.n_clusters)
pd.Series(
    {
        "auto vs fixed ARI": ds.metric_label_concordance(paris_auto, paris_fixed),
        "Leiden vs Paris ARI": ds.metric_label_concordance(chosen, paris_auto),
    }
)
```

## Important caveats to consider regarding clustering

- **Conflating cell-level marker tests with replicate-aware differential expression:** Scarf's marker search performs one-versus-rest statistical tests treating individual cells as independent observations. These cell-level {math}`p`-values suffer from massive artificial inflation due to pseudo-replication and ignore donor-level variance; an astronomically low {math}`p`-value confirms cluster separation on the graph, not reproducible biological differential expression; Therefore, don't interpret these values as changes in gene expression.
- **Discretizing continuous biological gradients (ignoring membership strength):** Graph partitioning algorithms like Leiden and Paris enforce discrete boundaries even across continuous developmental trajectories or cell-cycle axes. Failing to evaluate membership strength and cluster connectivity risks misinterpreting transitional intermediates with weak boundaries as distinct, terminal cell types.
- **Over-relying on automated resolution selection and technical splits:** Automated heuristics (such as coordinate-based sampling scores) and high Leiden resolutions optimize mathematical graph modularity, not biological reality. Sweeping to higher resolutions frequently fractures populations along technical confounders, such as sequencing depth, mitochondrial fraction, or batch effects, or produces "forced" hierarchical cuts in Paris that lack stable marker persistence.
