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

After we have the marker table, we can begin to assign our initial cell types. We can can begin by first using our existing information of marker genes to see their spatial orientation in the UMAP embeddings. Comparing their orientation on the UMAP vs the clusters allows us to then target our search through the marker gene dataset, and thus we can identify if the metrics provided in the marker database support the marker genes we are using. If the metrics do support, then we can assign our initial cell identities, and dig into the data further to see if multiple markers support our annotations.

```{code-cell}
ds.plots.embedding(
    layout=run["umap"],
    color_by=["CD3D", "CD4", "CD8A", "MS4A1", "CD14", "NKG7", "KLF4"],
    n_columns=3,
    sort_values=True,
)

ds.plots.embedding(
    layout=run["umap"],
    color_by=clusters,
    legend_loc="on_data",
)
```

Here, we can see the UMAP of our select marker genes for our predicted cell types alongside the clusters they may be present inside off.

CD3D lights up clusters that may our candidate T-cells. MS4A1 marks a separate block of potential B-cells, CD14 marks the monocyte likely block, and NKG7 marks the NK-like block. Newer literature indicates there may be a circulating subset of plasmacytoid dendritic cells (pDCs), which are represented here by KLF4.

To confirm the visual readings on the UMAP, we can now utilize the marker table

```{code-cell}
panel_genes = ["CD3D", "CD4", "CD8A", "MS4A1", "CD14", "NKG7", "KLF4"]
panel_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
panel_stats = panel_markers[panel_markers["feature_name"].isin(panel_genes)]
panel_stats.pivot(index="feature_name", columns="group_id", values="score").reindex(
    panel_genes
)
```

With the information in the markers table supporting our intepretation, we can move forward.

Clusters with these localized gene expression take an initial name for now as described below:

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

## Using several markers to support cluster interpretation and identify cell states

One gene can never be used to annotate a cell, thus we dig further by using alternative markers to validate our initial interpretations.

```{code-cell}
ds.plots.marker_heatmap(
    marker=markers,
    topn=3,
    figsize=(6, 8),
)
```

Look for coherent programs rather than a single gene by comparing against the literature or existing databases (resources can be found at the end of this document). By using multiple markers, we can also begin to bridge towards not only identifying the identity of a cluster, but the state it may be in.

In the heatmap, we can see that Cluster 1 displays a clear triplet of CD14, VCAN, and S100A12, which confirms our CD14-monocyte signature, while Cluster 2 expresses a distinct CTSL and TCF7L2 program with only residual CD14, indicating the prescense of a second monocyte state (potentially non-classical monocytes). 

Cluster 3 shows an FCER2, IGHD, and TNFRSF13B trio, which is a classic naive B-cell program. Looking at cytotoxicity linked genes, we can see the signal split across clusters 6 and 7: Cluster 6 is topped by KLRF1, FGFBP2, and ADGRG1, whereas Cluster 7 is led by TRGC2, GZMK, KLRG1, and CD8B, meaning the decision between an NK cell and a cytotoxic T-cell identity rests on which exclusive markers in the markers table, or through the use of negative controls.

The (large) Cluster 4 is dominated by ADTRP, ANKRD55, and FHIT, a program it partly shares with Cluster 5 (LMNA and TNFRSF4); while these gene names may seem less familiar than canonical markers, their exclusivity to the T-cell annotation helps rule out alternative B-cell, monocyte, or NK identities. 

Lastly, clusters with unique marker combinations, like Cluster 9's TCL1A and FCER2 signals, or Cluster 8's NELL2-dominated program, provide a key example where we our existing UMAPS can help in visualizing the spatial orientation of the expression of these genes.

## Validate annotations against negative controls

Going a step further, as you would in a real study, negative controls validate annotations by adding a layer of cell-type exclusivity: confirming that a cluster not only turns on the right genes, but also properly silences the genes belonging to competing or mutually execlusive lineages. In the several markers section above, we hint at the idea of negative markers as a way to differeniate between different cell states, and even see it in use for cluster 4, in how we validate that cluster 4 & 5 are likely T cells vs. B-cells, monocytes, or even NK cells. 

**Negative controls can be verified by simply searching for them in our marker tables and analyzing their metrics:**

- `frac_exp` for low expression prevalence; our negative controls should have low detection within the target cluster (`frac_exp` $\approx 0$‬).
- `score` for low exclusivity; a `score` near 0 means almost none of the gene's expression rank exists in the cluster you are studying

For our examples, the negative controls we use are... []**FILL THIS IN!]**


 **Cluster 6 is topped by KLRF1, FGFBP2, and ADGRG1, whereas Cluster 7 is led by TRGC2, GZMK, KLRG1, and CD8B, meaning the decision between an NK cell and a cytotoxic T-cell identity rests on which specific exclusive markers lead each column, or through the use of negative controls**

USE THE NEGATIVE CONTROLS AS AN EXAMPLE FOR THIS AS IT WORKS SUPER NICELY SUPER SUPER NICELY

## Write the reviewed mapping

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

### Do the final annotations remain spatially coherent?

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

# Alternative annotation steps / depth [create title]

ehh maybe required maybe not
