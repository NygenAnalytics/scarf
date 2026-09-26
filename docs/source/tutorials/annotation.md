
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

# Cell-Type & State Annotation Primer

After the clustering process, you are left with numbers separating groups of potentially distinct cell types or cell states. To proceed with the analysis process, annotating the cell types that may exist inside your dataset is critical. Approaching this problem requires biological context, in terms of the tissue or sample where you obtained your sample. For example, if you did sequencing on PBMCs versus the brain, you would expect vastly different cell types, and by having this context, you enable yourself to have some for a ground truth of what you cell types you can expect versus what you can not. Annotation generally requires marker genes for each cluster, in which marker genes are genes that have significantly differing expression across cell types. 

The tutorial here today simply guides you through the basic annotation process for PBMCs by determining the marker genes for each cluster, and using the corresponding metrics to assign cell types. Marker gene determination genneraly functions through Mann-Whitney U testing gene expression differences across all clusters, after which it is followed by a Benjamini-Hochberg correction for multiple hypothesis correction. SCARF takes on the liberty to calculate other metrics that can be utilized for identifying marker genes for each cluster.

Further resources on where to find markers for your unique samples can be found at the bottom of this document.

# Review cell-type markers and assign cell types

To begin, you must generally reach a point where clustering and marker search has been performed. Here, we grab the results of clustering and marker search for a pre-completed analysis.

```{code-cell}
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
clusters = run["clusters"]
markers = run["markers"]
cluster_values = np.asarray(run.cells.fetch("clusters"))
```

We first begin by inspecting the calculated markers, and gaining a strong conceptual understanding of the metrics available.

## Identify and interpret the markers

```{code-cell}
group_id = pd.Series(cluster_values).value_counts().index[0]
group_markers = ds.get_markers(
    marker=markers,
    group_id=group_id,
    min_score=-1,
    min_frac_exp=-1,
)
group_markers[
    [
        "feature_name",
        "score",
        "frac_exp",
        "fold_change",
        "auc",
        "p_value",
        "p_value_adjusted",
    ]
].head(12)
```

SCARF automatically calculates other metrics for like marker identification, and understanding their function and pitfalls is crucial when weighing the difference between cell type x and y on the same cluster n. 

- `fold_change` compares the average expression in a target cluster vs. all other cells; However, this can be skewed by high-expression noise or by a few extreme outliers.
- `frac_exp` helps to report the percentage of cells in a cluster with non-zero counts for a gene. This helps as say a gene has a 10 fold change, but has a small expressed fraction, it may not be the most useful marker to identify a cell type.
- `auc`measures how well a single gene's expression level predicts whether a cell belonds to a cluster, with values near 0.5 meaning this gene can be a marker for any clusters, and 1 meaning that this gene can be perfectly identified with a specific cluster. This approach is insensitive to outliers and can be a more impartial metric to utilize
- `score` is SCARF's unique specificity rank, which measures how uniquely a gene's expression is confined to this cluster on a 0–1 scale (summing to 1 across all clusters for each gene). A score near 1 flags a highly cluster-exclusive marker; a score near ‭$1 / n_{\text{clusters}}$‬‭‬ indicates a ubiquitous housekeeping gene that is useless for naming; and a score near 0 indicates absence.
- `p_value_adjusted` is simply the Benjamini-Hochberg p value correction we discussed earlier to correct for the multiple hypothesis testing the Mann-Whitney U test employs. 

## Assign initial cell types 

After we have the marker table, assignment runs in three moves: read the numbered cluster map against the marker panels, pull each panel gene's statistics from the marker table, and record one provisional name per cluster. 

Naming comes before validation: propose a cell type for each cluster from where its markers are
expressed, then test that proposal in the next step. Color the cluster map by canonical lineage
genes and read which clusters light up together.

```{code-cell}
ds.plots.embedding(
    layout=run["umap"],
    color_by=["CD3D", "CD4", "CD8A", "MS4A1", "CD14", "NKG7"],
    n_columns=3,
    sort_values=True,
)
```

Read this numbered map against the marker panels above: the IDs sitting on each lit-up region are
the clusters that panel gene nominates.

```{code-cell}
ds.plots.embedding(
    layout=run["umap"],
    color_by=clusters,
    legend_loc="on_data",
)
```

CD3D lights up one block of clusters (the T-cell candidates); within it, CD4 and CD8A separate
helper-leaning from cytotoxic-leaning regions, which is how similar T clusters are told apart.
MS4A1 marks a separate block (the B-cell candidates), CD14 marks the monocyte block, and NKG7
marks the NK-like block. Clusters sharing one program take one provisional name: T cells, B
cells, CD14 monocytes, FCGR3A monocytes (CD14-low in the heatmap below), NK cells, and the small
pDC-like group. Similar clusters are therefore split or merged by expression distribution, not by
UMAP distance alone.

Confirm the visual read against the stored statistics: each row is one panel gene, each column
one cluster, and each entry its specificity score.

```{code-cell}
panel_genes = ["CD3D", "CD4", "CD8A", "MS4A1", "CD14", "NKG7"]
panel_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
panel_stats = panel_markers[panel_markers["feature_name"].isin(panel_genes)]
panel_stats.pivot(index="feature_name", columns="group_id", values="score").reindex(
    panel_genes
)
```

Record one provisional name per cluster from the panels and the table above. This proposal is
data, not yet annotation: step 3 tests it, and step 4 writes it.

```{code-cell}
proposed_labels = {
    "1": "CD14 monocytes",
    "2": "FCGR3A monocytes",
    "3": "B cells",
    "4": "T cells",
    "5": "NK cells",
    "6": "T cells",
    "7": "T cells",
    "8": "B cells",
    "9": "T cells",
    "10": "pDC-like cells",
}
pd.Series(proposed_labels, name="proposed_cell_type")
```

## 3. Do several markers support each cluster interpretation?

One gene never carries an annotation. Check that multiple independent markers agree with each
provisional name before writing it down.

```{code-cell}
ds.plots.marker_heatmap(
    marker=markers,
    topn=3,
    figsize=(6, 8),
)
```

Look for coherent programs rather than a single winning gene. In a real study, also inspect
expected negative markers, cluster size, technical covariates, donor coverage, and doublet scores.

## 4. Write the reviewed mapping

This example records the broad teaching labels supported in {doc}`scrna_seq`. Multiple Leiden
clusters intentionally map to the same lineage. The mapping is tied to this run and should not be
copied to another graph or dataset.

```{code-cell}
# Proposal from step 2, confirmed against the heatmap in step 3.
label_map = dict(proposed_labels)
observed = {str(value) for value in np.unique(cluster_values)}
assert observed == set(label_map)

analysis_cells = np.asarray(run.cells.fetch_all("I"), dtype=bool)
cell_type = np.full(len(analysis_cells), "Not analyzed", dtype=object)
cell_type[analysis_cells] = [label_map[str(value)] for value in cluster_values]
ds.cells.insert("reviewed_cell_type", cell_type, overwrite=True)
pd.Series(cell_type[analysis_cells]).value_counts()
```

The insertion is an explicit user metadata edit. It does not alter the immutable clustering or
marker artifacts. To update cell types later, edit `label_map` and re-run the cell above:
`overwrite=True` replaces the `reviewed_cell_type` column in place, so revision is one edit plus
one re-execution, with the clustering and marker evidence untouched.

### Question: does the reviewed annotation remain spatially coherent?

```{code-cell}
ds.plots.embedding(
    layout=run["umap"],
    color_by="reviewed_cell_type",
    legend_loc="right",
)
```

Keep the label map, marker ref, run ID, and review rationale in the study record. Cluster IDs can
change when the graph or partition changes. Overlap-based `smart_label` helps compare two
partitions, but it is not ontology annotation; that partition-comparison role belongs in
{doc}`clustering`.

For scATAC-seq, {doc}`scatac_seq` uses GeneScores to display marker accessibility. GeneScores are
accessibility summaries, not measured RNA expression. Use {doc}`mapping_and_label_transfer` when a
query should inherit labels from a fixed reference rather than be annotated de novo.

## Important caveats to consider regarding annotation

- **[Placeholder]:** [Placeholder]
- **[Placeholder]:** [Placeholder]
- **[Placeholder]:** [Placeholder]

## Resources for marker-based annotation
