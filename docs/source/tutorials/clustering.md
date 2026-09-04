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

Clustering is a model of graph structure, not a cell-type verdict. This page keeps one graph fixed,
compares several Leiden resolutions with a Paris hierarchy, and reads every result through its
exact immutable artifact ref.

## 1. Open one graph

```{code-cell} ipython3
from dataclasses import asdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
clustering_run = ds.pipeline.open(label="docs_default")
graph = clustering_run["connectivity_map"]
umap = clustering_run["umap"]
umap_values = np.asarray(ds.load_artifact(umap)["values"][:])
```

The rebuilt store carries one completed pipeline run under the immutable label `docs_default`.
Its exact graph and UMAP refs are the baseline below. Only the additional clustering choices are
created on this page, so feature, PCA, graph, and layout effects stay out of the comparison.

## 2. Sweep Leiden resolution

```{code-cell} ipython3
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

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, resolution in zip(axes, leiden_values, strict=True):
    axis.scatter(
        umap_values[:, 0],
        umap_values[:, 1],
        c=leiden_values[resolution],
        s=3,
        cmap="tab20",
    )
    axis.set_title(f"Leiden {resolution}")
figure.tight_layout()
figure
```

Higher resolution usually produces more and smaller groups. Reject a split when it is driven by a
technical covariate, has weak marker evidence, or disappears under a modest parameter change.

ARI and NMI quantify agreement without selecting a winner:

```{code-cell} ipython3
agreement = []
for first, second in combinations(leiden_values, 2):
    agreement.append(
        {
            "comparison": f"{first} vs {second}",
            "ARI": adjusted_rand_score(
                leiden_values[first],
                leiden_values[second],
            ),
            "NMI": normalized_mutual_info_score(
                leiden_values[first],
                leiden_values[second],
            ),
        }
    )
pd.DataFrame(agreement)
```

## 3. Inspect membership strength

Use the chosen cluster ref directly. The result is another axis-aligned artifact, not a metadata
column. Resolution `0.5` is fixed here for the diagnostic walkthrough.

```{code-cell} ipython3
chosen = leiden_refs[0.5]
chosen_values = leiden_values[0.5]
membership = ds.calc_membership_strength(chosen, graph)
membership_values = np.asarray(ds.load_artifact(membership)["values"][:])

figure, axis = plt.subplots(figsize=(5, 4))
points = axis.scatter(
    umap_values[:, 0],
    umap_values[:, 1],
    c=membership_values,
    s=3,
)
figure.colorbar(points, ax=axis, label="membership strength")
figure.tight_layout()
figure
```

```{code-cell} ipython3
ds.plots.cluster_connectivity(
    graph=graph,
    groups=chosen,
    layout=umap,
)
```

Low values throughout one cluster suggest a weak boundary. A narrow band of low values between
otherwise coherent groups may represent continuous biology.

## 4. Compare Paris cuts

`run_paris_clustering` returns a `cluster_cut` ref. Load the domain result explicitly when
hierarchy diagnostics are needed.

```{code-cell} ipython3
paris_auto = ds.run_paris_clustering(graph)
paris_result = ds.load_paris_clustering(paris_auto)
pd.DataFrame([asdict(item) for item in paris_result.diagnostics])[
    ["label", "size", "persistence", "decision_margin", "forced"]
]
```

```{code-cell} ipython3
ds.plots.cluster_tree(graph=graph, clusters=paris_auto)
```

Persistence measures how long a selected branch survives in the hierarchy. The decision margin
measures the preference for retaining it. A forced group satisfies a structural constraint and is
not, by itself, strong biological evidence.

```{code-cell} ipython3
paris_fixed = ds.run_paris_clustering(
    graph,
    n_clusters=paris_result.n_clusters,
)
paris_auto_values = np.asarray(ds.load_artifact(paris_auto)["labels"][:])
paris_fixed_values = np.asarray(ds.load_artifact(paris_fixed)["labels"][:])
pd.Series(
    {
        "auto vs fixed ARI": adjusted_rand_score(
            paris_auto_values,
            paris_fixed_values,
        ),
        "Leiden vs Paris ARI": adjusted_rand_score(
            chosen_values,
            paris_auto_values,
        ),
    }
)
```

## 5. Review marker evidence

Marker search requires exact cluster and feature-selection refs and returns one immutable marker
table artifact.

```{code-cell} ipython3
markers = ds.run_marker_search(
    chosen,
    features=clustering_run["feature_universe"],
)
sizes = pd.Series(chosen_values).value_counts()
largest = sizes.index[0]
smallest = sizes.index[-1]

largest_markers = ds.get_markers(marker=markers, group_id=largest)
smallest_markers = ds.get_markers(marker=markers, group_id=smallest)
```

```{code-cell} ipython3
largest_markers[
    ["feature_name", "score", "auc", "p_value", "p_value_adjusted"]
].head(10)
```

```{code-cell} ipython3
smallest_markers[
    ["feature_name", "score", "auc", "p_value", "p_value_adjusted"]
].head(10)
```

The p-values are cell-level one-versus-rest marker tests with within-group adjustment. They are not
replicate-aware differential expression. A defensible partition combines marker evidence, graph
support, technical covariates, replicate coverage, and the study question.

## 6. Pipeline cluster selection

When a pipeline run includes multiple Leiden candidates, it scores them with one deterministic
shared sample of at most 10,000 cells in the graph's PCA or Harmony coordinates. Paris can still
run as `clustering_run["paris"]`, but it is not an automatic winner. The `cluster_selection`
artifact persists the scores, sampling policy, invalid-candidate reasons, tie order, and selected
key:

```python
decision_ref = clustering_run["cluster_selection"]
selected_cluster_ref = clustering_run["clusters"]
```

This automatic choice is a reproducible baseline, not proof that the selected resolution is best
for every biological question. Retain alternative refs when the decision needs domain-specific
evidence.
