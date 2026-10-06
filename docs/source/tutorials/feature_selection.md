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

## 1. Fit the mean-variance model

The PBMC store carries an analysis saved as `docs_default`. It used 500 genes and 15 PCs to keep
this teaching example small. The current API defaults are 1,000 genes and 21 PCs.
We first repeat the saved 500-gene selection to inspect how those genes were chosen.

```{code-cell} ipython3
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
# Open the prepared baseline for the comparisons below.
baseline = ds.pipeline.open(label="docs_default")
# Reuse the run's frozen analysis cells.
cell_selection = baseline["analysis_cell_selection"]
# Inspect the opened assays and their dimensions.
ds
```

Fit and inspect the 500-gene selection used by the saved example.

```{code-cell} ipython3
# Select the same 500 genes used by the prepared baseline.
hvg_500 = ds.select_hvgs(cell_selection, top_n=500)
# Load the selected-gene mask on the full feature axis.
hvg_500_values = np.asarray(ds.load_artifact(hvg_500)["values"][:])
# Check agreement with the saved selection and its gene count.
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
Each family is defined once in `scarf.features.gene_families`; `ribosomal` covers RPS, RPL, MRPS, and MRPL.
Count how many genes in this dataset fall into each family:

```{code-cell} ipython3
from scarf.features.gene_families import GENE_FAMILY_PATTERNS
from scarf.features.variability import DEFAULT_HVG_BLACKLIST

# Count genes matching each default exclusion family.
family_counts = pd.Series(
    {
        name: len(ds.RNA.feats.grep(pattern))
        for name, pattern in GENE_FAMILY_PATTERNS.items()
    },
    name="genes matching pattern",
)
# Count all genes matched by the combined default blacklist.
print("Default blacklist matches:", len(ds.RNA.feats.grep(DEFAULT_HVG_BLACKLIST)))
# Inspect which excluded gene families are represented in the assay.
family_counts
```

These families can dominate broad variation without representing the cell identities sought in a typical heterogeneity workflow.
They can be biologically relevant in another study, so the default is a starting point rather than a claim that those genes are unimportant.

By default, `max_cells` is `n_selected - 20`.
Genes detected in at least that many selected cells are excluded as nearly ubiquitous.

Clearing the blacklist keeps every gene name while retaining other HVG filters.
Compare the same `top_n` with and without the default pattern:

```{code-cell} ipython3
# Repeat selection with the same count and no name blacklist.
hvg_no_blacklist = ds.select_hvgs(
    cell_selection, top_n=500, blacklist="", show_plot=False
)
# Load the alternative selection on the same feature axis.
unblocked_values = np.asarray(ds.load_artifact(hvg_no_blacklist)["values"][:])
# Keep both feature masks for a direct comparison.
selection_values = {
    "hvgs_default": hvg_500_values,
    "hvgs_no_blacklist": unblocked_values,
}
# Compare selected-gene counts with and without name exclusions.
pd.Series(
    {key: int(values.sum()) for key, values in selection_values.items()},
    name="selected genes",
)
```

```{code-cell} ipython3
# Read gene names on the full feature axis.
feature_names = ds.RNA.feats.fetch_all("names")
# Keep the default selection mask.
default_values = selection_values["hvgs_default"]
# Find genes added only when the blacklist is cleared.
only_without_blacklist = feature_names[unblocked_values & ~default_values]
# Count genes added only when the blacklist is cleared.
print(
    "Genes selected only when blacklist is cleared:", len(only_without_blacklist)
)
# Inspect the newly included gene names.
pd.Series(only_without_blacklist).head(15)
```

Other overrides stay available for study-specific work:

```python
# Replace the default with a study-specific, case-insensitive regex.
ds.select_hvgs(cell_selection, blacklist=r"^MT-|^RPS|^RPL", top_n=2000)

# Disable the nearly ubiquitous gene filter.
ds.select_hvgs(cell_selection, max_cells=np.inf, top_n=2000)
```

Repeating a selection with the same cells and settings reuses the saved result, including the
diagnostics used for its plot. See {doc}`reuse_and_tracing` for details.

## 3. Compare feature-set size

The number of selected genes changes the PCA basis and can change neighbourhood structure.
This comparison keeps all other graph choices fixed. The 500-gene branch comes directly from
`docs_default`; only the 1,000-gene branch is new. An earlier release built the baseline graph,
before revision 2 of `build_connectivity_map` corrected the edge weights that the new branch uses,
so the two branches also differ in those weights.

```{code-cell} ipython3
# Select 1,000 genes on the same frozen cell population.
feature_1000 = ds.select_hvgs(cell_selection, top_n=1000, show_plot=False)
# Read the selected-gene mask for the new branch.
feature_1000_values = np.asarray(ds.load_artifact(feature_1000)["values"][:])
# Check the number of genes retained for the new branch.
{"selected genes": int(feature_1000_values.sum())}
```

Normalize the selected genes and construct the new graph.

```{code-cell} ipython3
# Normalize the new feature selection.
normalized_1000 = ds.run_normalization(cell_selection, feature_1000)
# Keep 15 PCA dimensions for this feature-count comparison.
pca_1000 = ds.run_pca(normalized_1000, dims=15)
# Prepare the embedding initialization from the new PCA.
initialization_1000 = ds.build_embedding_initialization(pca_1000)
# Build a neighbor-search index for the new PCA.
ann_1000 = ds.build_ann_index(pca_1000)
# Keep the baseline neighbor count of 11.
neighbors_1000 = ds.query_neighbors(ann_1000, k=11)
# Build connectivity from the new neighbors.
graph_1000 = ds.build_connectivity_map(neighbors_1000)
# Read the PCA dimensions supplied to this graph without loading its values.
pca_shape = ds.load_artifact(pca_1000)["data"].shape
# Inspect the cell and principal-component counts.
{"cells": pca_shape[0], "PCs": pca_shape[1]}
```

Build the layout and clustering for the new graph.

```{code-cell} ipython3
# Pair the layout and clustering for each feature count.
feature_branches = {
    500: (baseline["umap"], baseline["leiden_0.5"]),
    1000: (
        ds.run_umap(graph_1000, initialization_1000),
        ds.run_leiden_clustering(graph_1000, resolution=0.5),
    ),
}
# Read the dimensions of the saved layout for each feature count.
layout_shapes = {
    top_n: ds.load_artifact(layout_ref)["values"].shape
    for top_n, (layout_ref, _) in feature_branches.items()
}
# Check that both layouts contain the same cells and two embedding coordinates.
pd.DataFrame(
    layout_shapes, index=["cells", "embedding dimensions"]
).T.rename_axis("selected genes")
```

```{code-cell} ipython3
# Create one plotting axis for each comparison panel.
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
# Retain the cluster labels from each plotted feature branch.
cluster_values = {}
# Draw each comparison on its own labeled axis.
for axis, top_n in zip(axes, feature_branches, strict=True):
    # Select this branch's layout and cluster labels.
    umap_ref, cluster_ref = feature_branches[top_n]
    # Read the cluster labels for this branch.
    labels = np.asarray(ds.load_artifact(cluster_ref)["values"][:])
    # Retain those labels for the cross-tabulation below.
    cluster_values[top_n] = labels
    # Plot this feature branch with its corresponding clustering.
    ds.plots.embedding(
        layout=umap_ref,
        color_by=cluster_ref,
        target=axis,
        show_titles=False,
        show=False,
    )
    # Label the panel with the quantity being compared.
    axis.set_title(f"{top_n:,} selected genes")
# Adjust spacing so panel labels remain readable.
figure.tight_layout()
# Display the completed figure.
figure
```

A cross-tabulation shows how partitions match when the feature set grows. Cluster numbers can
change without changing the groups. A row spread across several columns suggests a split; a
column collecting several rows suggests a merge. The margins report cluster sizes.

```{code-cell} ipython3
# Read the baseline 500-gene cluster labels.
cluster_500 = cluster_values[500]
# Read the 1,000-gene cluster labels.
cluster_1000 = cluster_values[1000]
# Compare the cell counts or fractions across the selected groups.
pd.crosstab(
    pd.Series(cluster_500, name="500 genes"),
    pd.Series(cluster_1000, name="1,000 genes"),
    margins=True,
)
```

```{code-cell} ipython3
# Quantify partition agreement between 500 and 1,000 genes.
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
The adjusted Rand index (ARI) measures agreement between partitions without requiring their
cluster numbers to match. It does not identify which feature set is more biologically useful.
Compare marker specificity and known biology instead of choosing the layout that appears most separated.

## 4. Install an externally chosen feature set

`set_feature_selection` accepts either a boolean mask aligned to the complete feature metadata order or physical feature indexes.
It records an immutable selection artifact, so downstream {term}`artifacts <artifact>` can trace
which genes were used from the artifact itself.
Exactly one input form is required; duplicate or out-of-range indexes, misaligned masks, and empty selections are rejected.
Verify the mask length and selected count before building a graph:

```{code-cell} ipython3
# Choose a small marker panel for the comparison.
panel_genes = ["CD3D", "MS4A1", "CD14", "LYZ", "NKG7", "GNLY"]
# Match the chosen genes to the complete feature axis.
manual_mask = np.isin(feature_names.astype(str), panel_genes)
# Check that the manual mask matches the assay axis and is nonempty.
print("mask length:", len(manual_mask), "selected:", int(manual_mask.sum()))
# Save the externally chosen feature selection.
custom_features = ds.set_feature_selection(mask=manual_mask)
# Inspect the reference for the saved manual gene selection.
custom_features
```

Construct selections with Scarf's metadata helpers when possible: `sift` and `multi_sift` return boolean masks for `mask=`; `get_index_by` returns integer feature-table indexes for `feature_indexes=`.

Both producer calls return an {term}`ArtifactRef`.
Pass that reference directly when continuing a branch:

```python
# Keep the normalization fixed while comparing reductions.
normalized = ds.run_normalization(cell_selection, custom_features)
```

Retain or persist exact refs in the analysis record. To request the complete feature universe,
use the canonical all-features producer:

```python
# Use the complete feature universe for this assay.
all_features = ds.select_all_features(from_assay="RNA")
```

`all_features` is an immutable all-true artifact for this exact assay axis.

In your own workflow, the standard pipeline uses 1,000 HVGs by default. Change the count when
there is a reason to include more or fewer genes:

```python
# Run the pipeline with the explicitly changed feature setting.
ds.pipeline.run(hvg_count=2000)
```

Other HVG settings are available through the pipeline's `params` mapping. For example, if a
study needs genes that the default blacklist removes:

```python
# Run the pipeline with the explicitly changed feature setting.
ds.pipeline.run(params={"hvg": {"blacklist": ""}})
```

Use the explicit stage methods when you want to compare feature sets on the same frozen cells,
as we did above. For a simple detection threshold instead of an HVG model, use
`select_detected_features(cell_selection, min_cells=...)`.

For scATAC-seq, prevalent peak selection is the analogous step.
`select_prevalent_peaks` returns a feature-selection artifact whose scientific identity contains
only its `feature_summary` input and `top_n` parameter.
See {doc}`scatac_seq`; peak prevalence, not an RNA mean-variance model, defines the candidate accessibility features.
