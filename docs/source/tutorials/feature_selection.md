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
`mark_hvgs` models the relationship between mean expression and variance, then selects genes whose corrected variance is high relative to genes with similar abundance.

This is distinct from cell quality control.
A gene can be measured correctly and still be excluded because it contributes broad technical or confounding variation to the graph.

For a simple detection threshold, `select_detected_features(min_cells=...)` publishes a selection from the same immutable feature summary.
The threshold is inclusive and the method returns the selection reference.

## 1. Fit the mean-variance model

The published PBMC store contains literal metadata from an earlier analysis.
Structurally repack it into a temporary source with the current RNA count layout, then mount those count matrices into a fresh page-local store.
The published source remains unchanged and the selections below are current immutable artifacts.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts_path = Path(analysis_directory.name) / "counts.zarr"
analysis_path = Path(analysis_directory.name) / "feature_selection.zarr"
repack_store(
    f"{dataset}/data.zarr",
    str(repacked_counts_path),
    nthreads=2,
)
ds = scarf.mount_datastore(
    str(repacked_counts_path),
    at=str(analysis_path),
    default_assay="RNA",
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
hvg_500 = ds.mark_hvgs(
    min_cells=20,
    top_n=500,
    show_plot=True,
    label="hvgs_500",
)
print(
    "Selected genes:",
    int(ds.RNA.feats.fetch_all("hvgs_500").sum()),
)
hvg_500
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
for key, kwargs in (
    ("hvgs_default", {}),
    ("hvgs_no_blacklist", {"blacklist": ""}),
):
    ds.mark_hvgs(
        min_cells=20,
        top_n=500,
        show_plot=False,
        label=key,
        **kwargs,
    )

pd.Series(
    {
        key: int(ds.RNA.feats.fetch_all(key).sum())
        for key in ("hvgs_default", "hvgs_no_blacklist")
    },
    name="selected genes",
)
```

```{code-cell} ipython3
feature_names = ds.RNA.feats.fetch_all("names")
only_without_blacklist = feature_names[
    ds.RNA.feats.fetch_all("hvgs_no_blacklist")
    & ~ds.RNA.feats.fetch_all("hvgs_default")
]
print("Genes selected only when blacklist is cleared:", len(only_without_blacklist))
pd.Series(only_without_blacklist).head(15)
```

Other overrides stay available for study-specific work:

```python
# Replace the default with a study-specific, case-insensitive regex.
ds.mark_hvgs(blacklist=r"^MT-|^RPS|^RPL", top_n=2000)

# Disable the nearly ubiquitous gene filter.
ds.mark_hvgs(max_cells=np.inf, top_n=2000)
```

HVG reuse is based on the resolved algorithm inputs: `min_cells`, effective `max_cells`, `top_n`, variance and mean bounds, `n_bins`, `lowess_frac`, the resolved blacklist, `keep_bounds`, and `bin_strategy`.
When `max_cells` is omitted, Scarf applies `n_selected - 20` unless that would be at or below `min_cells`, in which case the upper limit is infinite.
The effective value is stored, so an omitted value and an explicitly equivalent value reuse the same artifact.
The publication label, plotting options, threads, and `invalidate_cache` are not part of scientific identity.
On reuse, an HVG plot reads stored corrected-variance diagnostics rather than recomputing the selection.

## 3. Compare feature-set size

The number of selected genes changes the PCA basis and can change neighbourhood structure.
This small comparison keeps all other graph choices fixed.

```{code-cell} ipython3
for top_n, key in ((300, "hvgs_300"), (1000, "hvgs_1000")):
    feature_ref = ds.mark_hvgs(
        min_cells=20,
        top_n=top_n,
        show_plot=False,
        label=key,
    )
    normalized_ref = ds.run_normalization(features=feature_ref)
    pca_ref = ds.run_pca(normalized_ref, dims=15)
    ds.build_embedding_initialization(pca_ref)
    ann_ref = ds.build_ann_index(pca_ref)
    neighbors_ref = ds.query_neighbors(ann_ref, k=11)
    graph_ref = ds.build_connectivity_map(neighbors_ref)
    ds.run_umap(
        graph=graph_ref,
        n_epochs=150,
        spread=5,
        min_dist=1,
        parallel=True,
        label=f"{key}_UMAP",
    )
    ds.run_leiden_clustering(
        graph=graph_ref,
        resolution=0.5,
        label=f"{key}_clusters",
    )
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, top_n in zip(axes, (300, 1000), strict=True):
    ds.plots.embedding(
        layout_key=f"RNA_hvgs_{top_n}_UMAP",
        color_by=f"RNA_hvgs_{top_n}_clusters",
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(f"{top_n:,} selected genes")
figure.tight_layout()
figure
```

```{code-cell} ipython3
pd.Series(
    {
        key: int(ds.RNA.feats.fetch_all(key).sum())
        for key in ("hvgs_300", "hvgs_1000")
    },
    name="selected genes",
)
```

Cluster sizes and the cross-tabulation show how partitions rematch when the feature set grows.
Off-diagonal mass marks groups that split or merge.

```{code-cell} ipython3
pd.DataFrame(
    {
        "300 genes": pd.Series(
            ds.cells.fetch("RNA_hvgs_300_clusters")
        ).value_counts().sort_index(),
        "1,000 genes": pd.Series(
            ds.cells.fetch("RNA_hvgs_1000_clusters")
        ).value_counts().sort_index(),
    }
)
```

```{code-cell} ipython3
pd.crosstab(
    pd.Series(ds.cells.fetch("RNA_hvgs_300_clusters"), name="300 genes"),
    pd.Series(ds.cells.fetch("RNA_hvgs_1000_clusters"), name="1,000 genes"),
)
```

```{code-cell} ipython3
feature_set_partitions = [
    "RNA_hvgs_300_clusters",
    "RNA_hvgs_1000_clusters",
]
pd.Series(
    {
        "adjusted Rand index": ds.metric_label_concordance(
            feature_set_partitions,
            metric="ari",
        ),
        "normalized mutual information": ds.metric_label_concordance(
            feature_set_partitions,
            metric="nmi",
        ),
    },
    name="300 vs 1,000 selected genes",
)
```

A larger set can recover weaker populations, but it can also restore unwanted programs.
ARI and NMI quantify partition agreement but do not identify which feature set is more biologically useful.
Compare marker specificity and known biology instead of choosing the layout that appears most separated.

## 4. Install an externally chosen feature set

`set_feature_selection` accepts either a boolean mask aligned to the complete feature metadata order or physical feature indexes.
It records an immutable selection artifact and publishes its values under one plain label, so downstream {term}`artifacts <artifact>` can trace which genes were used.
Exactly one input form is required; duplicate or out-of-range indexes, misaligned masks, and empty selections are rejected.
Verify the mask length and selected count before building a graph:

```{code-cell} ipython3
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
manual_mask = np.isin(
    ds.RNA.feats.fetch_all("names").astype(str),
    panel_genes,
)
print("mask length:", len(manual_mask), "selected:", int(manual_mask.sum()))
custom_features = ds.set_feature_selection(
    mask=manual_mask,
    label="custom_features",
)
custom_features, int(ds.RNA.feats.fetch_all("custom_features").sum())
```

Construct selections with Scarf's metadata helpers when possible: `sift` and `multi_sift` return boolean masks for `mask=`; `get_index_by` returns integer feature-table indexes for `feature_indexes=`.

Both producer calls return an {term}`ArtifactRef`.
Pass that reference directly when continuing a branch:

```python
normalized = ds.run_normalization(features=custom_features)
```

An exact published label is a convenience for resuming later:

```python
custom_features = ds.resolve_features("RNA", "custom_features")
all_features = ds.resolve_features("RNA", "all_features")
```

`all_features` is the immutable all-true universe for the assay.
Labels use lowercase snake case and are not prefixed with a cell key.

The standard pipeline forwards the same choices:

```python
ds.pipeline.run(
    highly_variable_features={
        "top_n": 2000,
        "blacklist": "",
        "max_cells": np.inf,
    },
    clustering_concurrency=1,
)
```

For scATAC-seq, prevalent peak selection is the analogous step.
`mark_prevalent_peaks` returns and publishes a feature-selection artifact whose scientific identity contains only its `feature_summary` input and `top_n` parameter.
See {doc}`scatac_seq`; peak prevalence, not an RNA mean-variance model, defines the candidate accessibility features.
