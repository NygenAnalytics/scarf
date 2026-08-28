---
description: Choose informative RNA genes, understand Scarf's default exclusions, and compare feature-set sizes.
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

# Choosing informative features

Feature selection decides which measured genes define PCA and the neighbourhood graph.
`select_hvgs` models the relationship between mean expression and variance, then selects genes
whose corrected variance is high relative to genes with similar abundance.

This is distinct from cell quality control.
A gene can be measured correctly and still be excluded because it contributes broad technical or confounding variation to the graph.

For a simple detection threshold, `select_detected_features(cell_selection, min_cells=...)` returns
a selection from the same immutable feature summary.
The threshold is inclusive and the method returns the selection reference.

## 1. Fit the mean-variance model

The rebuilt PBMC store carries a completed `docs_default` run. Repeating its 500-gene selection
with `show_plot=True` reuses the exact stored artifact and its diagnostics. Later sections create
only the alternative feature selections they compare.

```{code-cell} ipython3
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
baseline = ds.pipeline.open(label="docs_default")
cell_selection = baseline["analysis_cell_selection"]
hvg_500 = ds.select_hvgs(
    cell_selection,
    top_n=500,
    show_plot=True,
)
hvg_500_values = np.asarray(ds.load_artifact(hvg_500)["values"][:])
{
    "matches docs_default": hvg_500 == baseline["highly_variable_features"],
    "selected genes": int(hvg_500_values.sum()),
}
```

The plot should retain genes above the fitted mean-variance trend across a useful expression range.
A selection concentrated only among the most abundant genes can make library size or housekeeping programs dominate the graph.

## 2. Understand the default exclusions

Scarf applies the following case-insensitive regular expression to gene names:

```text
^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|
^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$
```

Matching starts at the beginning of the name.
Count how many genes in this dataset fall into each family:

```{code-cell} ipython3
default_blacklist = (
    "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|"
    "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
)
exclusion_families = {
    "mitochondrial (MT-)": r"^MT-",
    "ribosomal protein (RPS/RPL)": r"^RPS|^RPL",
    "mitoribosomal (MRPS/MRPL)": r"^MRPS|^MRPL",
    "cell cycle (CCN)": r"^CCN",
    "HLA": r"^HLA-",
    "H2": r"^H2-",
    "histone (HIST)": r"^HIST",
    "sex-linked": (
        r"^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$"
        r"|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
    ),
}
family_counts = pd.Series(
    {
        name: len(ds.RNA.feats.grep(pattern))
        for name, pattern in exclusion_families.items()
    },
    name="genes matching pattern",
)
print(
    "Default blacklist matches:",
    len(ds.RNA.feats.grep(default_blacklist)),
)
family_counts
```

These families can dominate broad variation without representing the cell identities sought in a typical heterogeneity workflow.
They can be biologically relevant in another study, so the default is a starting point rather than a claim that those genes are unimportant.

By default, `max_cells` is `n_selected - 20`.
Genes detected in at least that many selected cells are excluded as nearly ubiquitous.

Clearing the blacklist keeps every gene name while retaining other HVG filters.
Compare the same `top_n` with and without the default pattern:

```{code-cell} ipython3
hvg_no_blacklist = ds.select_hvgs(
    cell_selection,
    top_n=500,
    blacklist="",
    show_plot=False,
)
selection_values = {
    "hvgs_default": hvg_500_values,
    "hvgs_no_blacklist": np.asarray(
        ds.load_artifact(hvg_no_blacklist)["values"][:]
    ),
}
pd.Series(
    {
        key: int(values.sum())
        for key, values in selection_values.items()
    },
    name="selected genes",
)
```

```{code-cell} ipython3
feature_names = ds.RNA.feats.fetch_all("names")
default_values = selection_values["hvgs_default"]
unblocked_values = selection_values["hvgs_no_blacklist"]
only_without_blacklist = feature_names[unblocked_values & ~default_values]
print("Genes selected only when blacklist is cleared:", len(only_without_blacklist))
pd.Series(only_without_blacklist).head(15)
```

Other overrides stay available for study-specific work:

```python
# Replace the default with a study-specific, case-insensitive regex.
ds.select_hvgs(cell_selection, blacklist=r"^MT-|^RPS|^RPL", top_n=2000)

# Disable the nearly ubiquitous gene filter.
ds.select_hvgs(cell_selection, max_cells=np.inf, top_n=2000)
```

HVG reuse is based on the resolved algorithm inputs: `min_cells`, effective `max_cells`, `top_n`, variance and mean bounds, `n_bins`, `lowess_frac`, the resolved blacklist, `keep_bounds`, and `bin_strategy`.
When `max_cells` is omitted, Scarf applies `n_selected - 20` unless that would be at or below `min_cells`, in which case the upper limit is infinite.
The effective value is stored, so an omitted value and an explicitly equivalent value reuse the same artifact.
Plotting options, threads, and `invalidate_cache` are not part of scientific identity.
On reuse, an HVG plot reads stored corrected-variance diagnostics rather than recomputing the selection.

## 3. Compare feature-set size

The number of selected genes changes the PCA basis and can change neighbourhood structure.
This comparison keeps all other graph choices fixed. The 500-gene branch comes directly from
`docs_default`; only the 1,000-gene branch is new.

```{code-cell} ipython3
feature_1000 = ds.select_hvgs(
    cell_selection,
    top_n=1000,
    show_plot=False,
)
normalized_1000 = ds.run_normalization(cell_selection, feature_1000)
pca_1000 = ds.run_pca(normalized_1000, dims=15)
initialization_1000 = ds.build_embedding_initialization(pca_1000)
ann_1000 = ds.build_ann_index(pca_1000)
neighbors_1000 = ds.query_neighbors(ann_1000, k=11)
graph_1000 = ds.build_connectivity_map(neighbors_1000)
feature_branches = {
    500: (baseline["umap"], baseline["leiden_0.5"]),
    1000: (
        ds.run_umap(graph_1000, initialization_1000),
        ds.run_leiden_clustering(graph_1000, resolution=0.5),
    ),
}
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
cluster_values = {}
for axis, top_n in zip(axes, feature_branches, strict=True):
    umap_ref, cluster_ref = feature_branches[top_n]
    coordinates = np.asarray(ds.load_artifact(umap_ref)["values"][:])
    labels = np.asarray(ds.load_artifact(cluster_ref)["values"][:])
    cluster_values[top_n] = labels
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        s=3,
        cmap="tab20",
    )
    axis.set_title(f"{top_n:,} selected genes")
figure.tight_layout()
figure
```

A cross-tabulation shows how partitions rematch when the feature set grows. The margins report
cluster sizes; off-diagonal mass marks groups that split or merge.

```{code-cell} ipython3
cluster_500 = cluster_values[500]
cluster_1000 = cluster_values[1000]
pd.crosstab(
    pd.Series(cluster_500, name="500 genes"),
    pd.Series(cluster_1000, name="1,000 genes"),
    margins=True,
)
```

```{code-cell} ipython3
pd.Series(
    {
        "adjusted Rand index": adjusted_rand_score(cluster_500, cluster_1000),
        "normalized mutual information": normalized_mutual_info_score(
            cluster_500,
            cluster_1000,
        ),
    },
    name="500 vs 1,000 selected genes",
)
```

A larger set can recover weaker populations, but it can also restore unwanted programs.
ARI and NMI quantify partition agreement but do not identify which feature set is more biologically useful.
Compare marker specificity and known biology instead of choosing the layout that appears most separated.

## 4. Install an externally chosen feature set

`set_feature_selection` accepts either a boolean mask aligned to the complete feature metadata order or physical feature indexes.
It records an immutable selection artifact, so downstream {term}`artifacts <artifact>` can trace
which genes were used from the artifact itself.
Exactly one input form is required; duplicate or out-of-range indexes, misaligned masks, and empty selections are rejected.
Verify the mask length and selected count before building a graph:

```{code-cell} ipython3
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
manual_mask = np.isin(
    feature_names.astype(str),
    panel_genes,
)
print("mask length:", len(manual_mask), "selected:", int(manual_mask.sum()))
custom_features = ds.set_feature_selection(mask=manual_mask)
custom_features
```

Construct selections with Scarf's metadata helpers when possible: `sift` and `multi_sift` return boolean masks for `mask=`; `get_index_by` returns integer feature-table indexes for `feature_indexes=`.

Both producer calls return an {term}`ArtifactRef`.
Pass that reference directly when continuing a branch:

```python
normalized = ds.run_normalization(cell_selection, custom_features)
```

Retain or persist exact refs in the analysis record. To request the complete feature universe,
use the canonical all-features producer:

```python
all_features = ds.select_all_features(from_assay="RNA")
```

`all_features` is an immutable all-true artifact for this exact assay axis.

The standard pipeline exposes the common feature-count choice directly:

```python
ds.pipeline.run(hvg_count=2000)
```

Use `select_hvgs` plus the explicit stage methods when you need blacklist or mean-variance tuning.

For scATAC-seq, prevalent peak selection is the analogous step.
`select_prevalent_peaks` returns a feature-selection artifact whose scientific identity contains
only its `feature_summary` input and `top_n` parameter.
See {doc}`scatac_seq`; peak prevalence, not an RNA mean-variance model, defines the candidate accessibility features.
