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
# Feature selection: choosing informative features

Feature selection decides which measured genes define the downstream dimensionality reduction steps and the neighborhood graph. Feature selection is often performed to find the most highly variable genes (HVGs) across the entire dataset, as these are the genes with the most variance in terms of their expression. Reducing our dataset to a small subset of genes not only reduces the computational cost, but makes the analysis more interpretable by filtering out noise. More specifically, in Scarf, `select_hvgs` models the relationship between mean expression and variance, then selects genes
whose corrected variance is high relative to genes with similar abundance.

## Fit the mean-variance model

We start with a pre-analyzed dataset in which we can pull out the number of HVGs that we want to fit for the model. The current API default for the number of HVGs is 1,000 genes.

```{code-cell} ipython3
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
baseline = ds.pipeline.open(label="docs_default")
cell_selection = baseline["analysis_cell_selection"]
ds
```

Fit and inspect the 500-gene selection for this dataset.

```{code-cell} ipython3
hvg_500 = ds.select_hvgs(cell_selection, top_n=500)
hvg_500_values = np.asarray(ds.load_artifact(hvg_500)["values"][:])
{
    "matches docs_default": hvg_500 == baseline["highly_variable_features"],
    "selected genes": int(hvg_500_values.sum()),
}
```

The plot should retain genes above the fitted mean-variance trend across a useful expression range to represent the diversity of the dataset. A selection concentrated only among the most abundant genes can make library size or housekeeping programs dominate the graph, and thus not be an accurate summary of the data.

## Understand the default exclusions

Scarf applies the following case-insensitive exclusion to gene names:

```text
^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|
^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$
```

Each family is defined once in `scarf.features.gene_families`; `ribosomal` covers RPS, RPL, MRPS, and MRPL. We also have `mitochondrial` genes and sex-linked genes. These are left out explicitly.

For our purposes in this tutorial, we can count how many genes in this dataset fall into each family:

```{code-cell} ipython3
from scarf.features.gene_families import GENE_FAMILY_PATTERNS
from scarf.features.variability import DEFAULT_HVG_BLACKLIST

family_counts = pd.Series(
    {
        name: len(ds.RNA.feats.grep(pattern))
        for name, pattern in GENE_FAMILY_PATTERNS.items()
    },
    name="genes matching pattern",
)
print("Default blacklist matches:", len(ds.RNA.feats.grep(DEFAULT_HVG_BLACKLIST)))
family_counts
```

These families can dominate broad variation without representing the cell identities sought in a typical heterogeneity workflow; this could be a result of the cell quality, the cell state, whether it's alive or dead, or even the sequencing depth. They can be biologically relevant in another study, so the default is a starting point rather than a claim that those genes are unimportant.

By default, `max_cells` is `n_selected - 20`. This means that genes detected in at least that many selected cells are excluded as we can interpret them as nearly ubiquitous. However, this bar is high, so the model is mostly just fit to find genes that have high variance in comparison to similarly expressed genes, regardless of how many cells express them.

Say you wanted to clear the blacklist of genes not included for the HVG search, then clearing the blacklist keeps every gene name while retaining other HVG filters.
Compare the same `top_n` with and without the default pattern:

```{code-cell} ipython3
hvg_no_blacklist = ds.select_hvgs(
    cell_selection, top_n=500, blacklist="", show_plot=False
)
unblocked_values = np.asarray(ds.load_artifact(hvg_no_blacklist)["values"][:])
selection_values = {
    "hvgs_default": hvg_500_values,
    "hvgs_no_blacklist": unblocked_values,
}
pd.Series(
    {key: int(values.sum()) for key, values in selection_values.items()},
    name="selected genes",
)
```

```{code-cell} ipython3
feature_names = ds.RNA.feats.fetch_all("names")
default_values = selection_values["hvgs_default"]
only_without_blacklist = feature_names[unblocked_values & ~default_values]
print(
    "Genes selected only when blacklist is cleared:", len(only_without_blacklist)
)
pd.Series(only_without_blacklist).head(15)
```

Other overrides stay available for study-specific work:

```python
ds.select_hvgs(cell_selection, blacklist=r"^MT-|^RPS|^RPL", top_n=2000)

ds.select_hvgs(cell_selection, max_cells=np.inf, top_n=2000)
```

Repeating a selection with the same cells and settings reuses the saved result to avoid recomputation, including the diagnostics used for its plot.

## Compare feature-set size

The number of selected genes changes the basis of the dimensionality reduction with PCA, and can thus change neighborhood structure. This comparison keeps all other graph choices fixed. The 500-gene branch comes directly from above, and only the 1,000-gene branch is new.

```{code-cell} ipython3
feature_1000 = ds.select_hvgs(cell_selection, top_n=1000, show_plot=False)
feature_1000_values = np.asarray(ds.load_artifact(feature_1000)["values"][:])
{"selected genes": int(feature_1000_values.sum())}
```

Normalize the selected genes and construct the new graph.

```{code-cell} ipython3
normalized_1000 = ds.run_normalization(cell_selection, feature_1000)
pca_1000 = ds.run_pca(normalized_1000, dims=15)
initialization_1000 = ds.build_embedding_initialization(pca_1000)
ann_1000 = ds.build_ann_index(pca_1000)
neighbors_1000 = ds.query_neighbors(ann_1000, k=11)
graph_1000 = ds.build_connectivity_map(neighbors_1000)
pca_shape = ds.load_artifact(pca_1000)["data"].shape
{"cells": pca_shape[0], "PCs": pca_shape[1]}
```

Build the layout and clustering for the new graph.

```{code-cell} ipython3
feature_branches = {
    500: (baseline["umap"], baseline["leiden_0.5"]),
    1000: (
        ds.run_umap(graph_1000, initialization_1000),
        ds.run_leiden_clustering(graph_1000, resolution=0.5),
    ),
}
layout_shapes = {
    top_n: ds.load_artifact(layout_ref)["values"].shape
    for top_n, (layout_ref, _) in feature_branches.items()
}
pd.DataFrame(
    layout_shapes, index=["cells", "embedding dimensions"]
).T.rename_axis("selected genes")
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
cluster_values = {}
for axis, top_n in zip(axes, feature_branches, strict=True):
    umap_ref, cluster_ref = feature_branches[top_n]
    labels = np.asarray(ds.load_artifact(cluster_ref)["values"][:])
    cluster_values[top_n] = labels
    ds.plots.embedding(
        layout=umap_ref,
        color_by=cluster_ref,
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(f"{top_n:,} selected genes")
figure.tight_layout()
figure
```

The cross-tabulation simply compares the two cluster divisions cell by cell, so you can see what happened to each group when the feature set grew. Cluster numbers are arbitrary labels, so a number changing does not mean the group changed; what matters is whether the same cells stayed together. A row spreading across several columns means one group got split into pieces, while a column collecting several rows means previously separate groups collapsed into one.

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
        "adjusted Rand index": ds.metric_label_concordance(
            feature_branches[500][1], feature_branches[1000][1]
        )
    },
    name="500 vs 1,000 selected genes",
)
```

A larger set can recover weaker populations, but it can also restore unwanted programs.
You could use the adjusted Rand index (ARI) to measure the agreement between partitions without requiring their cluster numbers to match. It does not identify which feature set is more biologically useful, thus for interpreting the identity of clusters, compare marker specificity and known biology instead of choosing the layout that appears most separated.

## Install an externally chosen feature set

Say there is a specific set of genes that you want to use for your data that will likely describe the variance of the dataset, then you can import your externally chosen feature set. `set_feature_selection` accepts either a boolean mask aligned to the complete gene metadata order (so certain genes are included or excluded) or the physical feature indexes of what row the genes are located in. Exactly one input form is required; duplicate or out-of-range indexes, misaligned masks, and empty selections are rejected.

```{code-cell} ipython3
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
manual_mask = np.isin(feature_names.astype(str), panel_genes)
print("mask length:", len(manual_mask), "selected:", int(manual_mask.sum()))
custom_features = ds.set_feature_selection(mask=manual_mask)
custom_features
```

Construct selections with Scarf's metadata helpers when possible; `sift` and `multi_sift` return boolean masks for `mask=`, and `get_index_by` returns integer feature-table indexes for `feature_indexes=`. This gives you the flexibility to simply use Scarf's inbuilt helpers vs. writing other code to locate the information.

Once the custom features are selected, pass the reference that contains them directly when continuing a branch:

```python
normalized = ds.run_normalization(cell_selection, custom_features)
```

Retain or persist exact references in the analysis record; if you want to request the complete feature universe, use the canonical all-features producer:

```python
all_features = ds.select_all_features(from_assay="RNA")
```

In your own workflow, the standard pipeline uses 1,000 HVGs by default. Change the count when there is a reason to include more or fewer genes:

```python
ds.pipeline.run(hvg_count=2000)
```

Other HVG settings are available through the pipeline's `params` mapping. For example, if a study needs genes that the default blacklist removes:

```python
ds.pipeline.run(params={"hvg": {"blacklist": ""}})
```

If you want to compare feature sets on the same frozen cells, as we did above, use the explicit stage methods rather than the pipeline. If you don't need a full HVG model and simply want to drop genes that are barely detected, `select_detected_features(cell_selection, min_cells=...)` is a plain detection threshold that does exactly that.

The analogous step for scATAC-seq is prevalent peak selection. Here the candidate features are defined by how prevalent each peak is across cells, not by an RNA mean-variance model, and `select_prevalent_peaks` returns a feature-selection artifact whose scientific identity contains only its `feature_summary` input and `top_n` parameter. See {doc}`scatac_seq` for the overview of the workflow.
