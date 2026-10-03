---
description: Review marker evidence for cell-type annotations.
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
(annotation)=

# Annotate cell types and states

Clustering groups cells with similar expression profiles. Annotation asks what those groups
might be. We will start with broad PBMC labels, check several supporting and negative markers,
and then refine the labels where the evidence allows it.

The tissue matters: a plausible blood-cell identity may make little sense in another sample.
Use the marker patterns together with that context, and keep uncertain labels provisional.
The {doc}`scrna_seq` tutorial introduces the broad populations used here.

## Open the prepared analysis

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr")
run = ds.pipeline.open(label="docs_default")
clusters = run["clusters"]
markers = run["markers"]
cluster_values = run.cells.fetch("clusters").astype(str)
```

This is the same prepared result as the RNA tutorial. Cluster numbers and labels below belong
to this result; a new analysis may produce different clusters.

## 1. Read the marker evidence

Start with the default marker filters and inspect one cluster:

```{code-cell} ipython3
group_markers = ds.get_markers(marker=markers, group_id="1")
group_markers[
    ["feature_name", "score", "frac_exp", "fold_change", "auc", "p_value_adjusted"]
].head(10)
```

Each row compares expression in this cluster with the rest of the analyzed cells.

| Column | How to read it |
| --- | --- |
| `frac_exp` | Fraction of cells with detected expression, from 0 to 1. A value of 0.8 means 80% of the cluster. |
| `fold_change` | Mean expression in the cluster relative to the other cells. Check detection too: a few cells can drive a large difference. |
| `auc` | Values above 0.5 favour higher expression in this cluster; below 0.5 favour the other cells. A value near 0.5 gives little separation. |
| `score` | A relative specificity score based on expression ranks across clusters. It is not a detection fraction or a probability of cell identity. |
| `p_value_adjusted` | A two-sided Mann-Whitney p-value with Benjamini-Hochberg correction within the cluster's tested genes. It does not account for biological replication. |

A low specificity score does not establish that a gene is absent. Read `frac_exp` for detection
and compare it with `frac_exp_rest` when evaluating negative evidence.

## 2. Assign initial cell types

Plot a few familiar markers alongside the clusters. Drawing high values last makes sparse
expression easier to see in these panels.

```{code-cell} ipython3
panel_genes = ["CD14", "CD19", "CD8A", "CD4", "NCAM1", "IL3RA"]
ds.plots.embedding(
    layout=run["umap"],
    color_by=[*panel_genes, clusters],
    n_columns=3,
    sort_values=True,
)
```

CD14 suggests a monocyte population, CD19 highlights candidate B cells, and NCAM1 (CD56)
highlights an NK-like population. CD8A and CD4 help locate candidate T-cell groups, but neither
is sufficient alone. In this example, CD4's highest specificity score is in a monocyte cluster.
IL3RA suggests a pDC-like population that we will check with other markers.

The default marker filters hide weak and negative evidence. For the remaining comparisons,
load all marker rows once by relaxing both filters. Keep this table for the later panels.

```{code-cell} ipython3
all_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
panel_stats = all_markers[all_markers["feature_name"].isin(panel_genes)]
panel_best = (
    panel_stats.sort_values("score", ascending=False)
    .groupby("feature_name", sort=False)
    .head(1)
    .set_index("feature_name")
    .reindex(panel_genes)
)
panel_best[["group_id", "score", "frac_exp", "auc"]].round(2)
```

These are starting hypotheses. Several clusters receive the same broad name:

```{code-cell} ipython3
proposed_labels = {
    "1": "CD14 monocytes",
    "2": "monocytes",
    "3": "B cells",
    "4": "T cells",
    "5": "T cells",
    "6": "NK cells",
    "7": "T cells",
    "8": "T cells",
    "9": "B cells",
    "10": "pDC-like cells",
}
pd.Series(proposed_labels, name="proposed_cell_type")
```

## 3. Look for several supporting markers

A heatmap helps us check whether several genes support each proposed identity. Show the three
leading markers per cluster, with enough width to read the cluster labels:

```{code-cell} ipython3
ds.plots.marker_heatmap(marker=markers, topn=3, figsize=(6, 6))
```

Read the patterns together:

- **Monocytes:** cluster 1 has S100A12, VCAN, and CD14. Cluster 2 has a distinct CTSL, TCF7L2,
  and SMIM25 pattern. Keep its broader monocyte label while investigating its state.
- **B cells:** IGHD, FCER2, and TCL1A support a naive B-cell interpretation for cluster 9.
  IGHA1, TNFRSF13B, and IGHG1 support a different, memory-like B-cell population in cluster 3.
- **NK and T cells:** KLRF1, FGFBP2, and ADGRG1 support the NK interpretation for cluster 6.
  Clusters 7 and 8 need comparison of CD8 and T-cell markers with this cytotoxic program.
- **Other T-cell groups:** cluster 4 retains a broad T-cell label. Cluster 5's TNFRSF4 and PI16,
  together with CD4 and IL7R, support a CD4 T-cell interpretation.
- **pDC-like cells:** LILRA4 and SERPINF1 provide support alongside IL3RA in cluster 10.

For finer T-cell labels, inspect a small panel rather than a long list of every possible marker:

```{code-cell} ipython3
t_cell_genes = ["CD3D", "CD8A", "CD8B", "CCR7", "IL7R", "CD27", "GZMK"]
t_cell_evidence = all_markers[
    all_markers["group_id"].isin(["7", "8"])
    & all_markers["feature_name"].isin(t_cell_genes)
]
t_cell_evidence[["group_id", "feature_name", "frac_exp", "score"]].round(2)
```

Cluster 7 has a GZMK-associated cytotoxic T-cell pattern. Cluster 8 combines CD8 expression with
CCR7, IL7R, and CD27, supporting a naive-like CD8 interpretation. These descriptions remain
hypotheses about state, not proof of a separate lineage.

## 4. Check competing identities

Supporting markers are only part of the evidence. For an NK-cell call, inspect T-cell markers
such as CD3D and CD3E. For a CD8 T-cell call, compare CD4 with CD8A and CD8B.

```{code-cell} ipython3
comparison_genes = ["CD3D", "CD3E", "CD4", "NKG7", "GNLY", "CD8A", "CD8B"]
comparison = all_markers[
    all_markers["group_id"].isin(["6", "8"])
    & all_markers["feature_name"].isin(comparison_genes)
]
comparison[
    ["group_id", "feature_name", "frac_exp", "frac_exp_rest", "auc"]
].round(2)
```

In cluster 6, CD3D and CD3E are depleted compared with the other cells, while NKG7 and GNLY are
widely detected. Together, these observations support the NK label. The T-cell markers are not
literally absent, so this table alone cannot rule out mixed cells or background RNA.

In cluster 8, CD4 detection is low while CD8A and CD8B are common. This supports a CD8 T-cell
label. Check other competing lineages in the same way; low detection of one marker does not
establish that a population is pure.

## 5. Save and compare the reviewed labels

Keep the broad labels where we have not established a finer identity. Update the groups with
additional supporting evidence:

```{code-cell} ipython3
final_labels = proposed_labels | {
    "3": "memory B cells",
    "5": "CD4+ T cells",
    "7": "CD8+ T cells",
    "8": "naive CD8 T cells",
    "9": "naive B cells",
}
```

The cell table includes cells outside this run. Align the labels to the run's selection before
saving them. Rerunning this cell replaces the two annotation columns, leaving the computed
clusters unchanged.

```{code-cell} ipython3
analysis_cells = run.cells.fetch_all("I").astype(bool)
initial_cell_type = np.full(ds.cells.N, "Not analyzed", dtype=object)
reviewed_cell_type = np.full(ds.cells.N, "Not analyzed", dtype=object)
initial_cell_type[analysis_cells] = [proposed_labels[value] for value in cluster_values]
reviewed_cell_type[analysis_cells] = [final_labels[value] for value in cluster_values]
ds.cells.insert("proposed_cell_type", initial_cell_type, overwrite=True)
ds.cells.insert("reviewed_cell_type", reviewed_cell_type, overwrite=True)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout=run["umap"],
    color_by=["proposed_cell_type", "reviewed_cell_type"],
)
```

The saved `naive CD8 T cells` and `memory B cells` labels summarize the interpretations above.
They are provisional teaching labels for this dataset. Keep the marker evidence and the run
identity with them when recording an analysis.

## What still needs review?

Clusters can separate continuous states within one lineage. Compare marker evidence across
partitions with {doc}`clustering` before treating each cluster as a distinct cell type.

Mixed lineage markers may reflect doublets, background RNA, or uncertain identity. Check
per-cell evidence and {doc}`quality_control`; a cluster-average score cannot settle the question.
Rare populations deserve this review as much as large ones.

These marker statistics compare cells. Use {doc}`pseudobulk_and_differential_expression` for
questions about differences between biological conditions. Use {doc}`mapping_and_label_transfer`
when transferring labels from a fixed reference.

## Find marker references for your tissue

Use resources as evidence to compare with your measured markers, rather than a list of names
to assign automatically:

- Marker collections: [CellMarker](http://bio-bigdata.hrbmu.edu.cn/CellMarker/),
  [PanglaoDB](https://panglaodb.se/), and [Human Protein Atlas](https://www.proteinatlas.org/).
- Reference atlases and annotation tools: [Azimuth](https://azimuth.hubmapconsortium.org/),
  [ScType](https://github.com/IanevskiAleksandr/sc-type), [CellTypist](https://www.celltypist.org/),
  [SingleR](https://bioconductor.org/packages/release/bioc/html/SingleR.html), and
  [CyteType](https://www.nygen.io/products/cytetype).
- Gene-list interpretation: [Enrichr](https://maayanlab.cloud/Enrichr/) and
  [ToppGene](https://toppgene.cchmc.org/).
- Protein-marker panels: [BioLegend](https://www.biolegend.com/en-us/cell-markers),
  [BD Biosciences](https://www.bdbiosciences.com/), and
  [OMIPs in Cytometry Part A](https://onlinelibrary.wiley.com/journal/15524930).
