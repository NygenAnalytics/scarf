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

## 1. Fit the mean-variance model

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
    min_features_per_cell=10,
)
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
ds.mark_hvgs(
    min_cells=20,
    top_n=500,
    show_plot=True,
    hvg_key_name="hvgs_500",
)
print(
    "Selected genes:",
    int(ds.RNA.feats.fetch_all("I__hvgs_500").sum()),
)
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
        hvg_key_name=key,
        **kwargs,
    )

pd.Series(
    {
        key: int(ds.RNA.feats.fetch_all(f"I__{key}").sum())
        for key in ("hvgs_default", "hvgs_no_blacklist")
    },
    name="selected genes",
)
```

```{code-cell} ipython3
feature_names = ds.RNA.feats.fetch_all("names")
only_without_blacklist = feature_names[
    ds.RNA.feats.fetch_all("I__hvgs_no_blacklist")
    & ~ds.RNA.feats.fetch_all("I__hvgs_default")
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

## 3. Compare feature-set size

The number of selected genes changes the PCA basis and can change neighbourhood structure.
This small comparison keeps all other graph choices fixed.

```{code-cell} ipython3
for top_n, key in ((300, "hvgs_300"), (1000, "hvgs_1000")):
    ds.mark_hvgs(
        min_cells=20,
        top_n=top_n,
        show_plot=False,
        hvg_key_name=key,
    )
    ds.run_normalization(feat_key=key)
    ds.run_pca(dims=15)
    ds.build_embedding_initialization()
    ds.build_ann_index()
    ds.query_neighbors(k=11)
    ds.build_connectivity_map()
    ds.run_umap(
        n_epochs=150,
        spread=5,
        min_dist=1,
        parallel=True,
        label=f"{key}_UMAP",
    )
    ds.run_leiden_clustering(
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
        key: int(ds.RNA.feats.fetch_all(f"I__{key}").sum())
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

`set_hvgs` accepts either a boolean mask aligned to feature metadata or physical feature indexes.
It records the supplied selection so downstream {term}`artifacts <artifact>` can trace which genes were used.
Verify the mask length and selected count before building a graph:

```{code-cell} ipython3
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
manual_mask = np.isin(
    ds.RNA.feats.fetch_all("names").astype(str),
    panel_genes,
)
print("mask length:", len(manual_mask), "selected:", int(manual_mask.sum()))
custom_feature_key = ds.set_hvgs(
    cell_key="I",
    mask=manual_mask,
    hvg_key_name="custom_features",
    blacklist="",
)
custom_feature_key, int(ds.RNA.feats.fetch_all(custom_feature_key).sum())
```

Construct selections with Scarf's metadata helpers when possible: `sift` and `multi_sift` return boolean masks for `mask=`; `get_index_by` returns integer feature-table indexes for `feature_indexes=`.

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
See {doc}`scatac_seq`; peak prevalence, not an RNA mean-variance model, defines the candidate accessibility features.
