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

## 1. Build one graph

```{code-cell} ipython3
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = str(Path(analysis_directory.name) / "counts.zarr")
repack_store(f"{dataset}/data.zarr", repacked_counts, nthreads=2)
ds = scarf.mount_datastore(
    repacked_counts,
    at=str(Path(analysis_directory.name) / "clustering_analysis.zarr"),
    default_assay="RNA",
    nthreads=4,
    min_features_per_cell=10,
)
```

With the source mounted, build the normalized representation and the graph used by every
comparison below.

```{code-cell} ipython3
cell_selection = ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
)
features = ds.select_hvgs(
    cell_selection,
    min_cells=20,
    top_n=500,
    show_plot=False,
)
normalized = ds.run_normalization(cell_selection, features)
pca = ds.run_pca(normalized, dims=15)
initialization = ds.build_embedding_initialization(pca)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=11)
graph = ds.build_connectivity_map(neighbors)
umap = ds.run_umap(
    graph,
    initialization,
    n_epochs=150,
    spread=5,
    min_dist=1,
    parallel=True,
)
umap_values = np.asarray(ds.load_artifact(umap)["values"][:])
```

All candidates below consume this exact graph. Changing only the clustering choice keeps graph and
feature effects out of the comparison.

## 2. Sweep Leiden resolution

```{code-cell} ipython3
leiden_refs = {
    resolution: ds.run_leiden_clustering(graph, resolution=resolution)
    for resolution in (0.3, 0.5, 0.8)
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
column.

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

`run_paris_clustering` returns a `cluster_cut` ref. Load the domain result explicitly when hierarchy
diagnostics are needed.

```{code-cell} ipython3
paris_auto = ds.run_paris_clustering(graph, n_clusters="auto")
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
all_features = ds.set_feature_selection(
    from_assay="RNA",
    feature_indexes=range(ds.RNA.feats.N),
)
markers = ds.run_marker_search(chosen, features=all_features)
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

The standard RNA pipeline runs Leiden at `0.5`, `0.75`, `1.0`, and `1.25` plus Paris. Its
`cluster_selection` stage evaluates every valid candidate with one deterministic sample of at most
10,000 cells in PCA space. It persists the scores, invalid-candidate reasons, tie order, and selected
key:

```python
run = ds.pipeline.run()
decision_ref = run["cluster_selection"]
selected_cluster_ref = run["clusters"]
```

This automatic choice is a reproducible baseline, not proof that the selected resolution is best
for every biological question. Retain alternative refs when the decision needs domain-specific
evidence.
