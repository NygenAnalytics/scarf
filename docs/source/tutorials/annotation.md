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

# Cell-Type and State Annotation Primer

After the clustering process, you are left with numbers separating groups of potentially distinct cell types or cell states. To proceed with the analysis process, annotating the cell types that may exist inside your dataset is critical. Approaching this problem requires biological context, in terms of the tissue or sample where you obtained your sample. For example, if you did sequencing on PBMCs versus the brain, you would expect vastly different cell types, and by having this context, you enable yourself to have some form of ground truth of what cell types you can expect versus what you cannot. Annotation generally requires marker genes for each cluster, in which marker genes are genes that have significantly differing expression across cell types.

The tutorial here guides you through the basic annotation process for PBMCs by determining the marker genes for each cluster, and using the corresponding metrics to assign cell types. Marker gene determination generally functions through a two-sided Mann-Whitney U test, which tests changes in gene expression from one cluster versus the rest, with a Benjamini-Hochberg correction within each cluster.

SCARF also calculates other metrics that can be utilized for identifying marker genes for each cluster. The core {doc}`scrna_seq` workflow shows the corresponding broad PBMC dotplot, and {doc}`clustering` covers how the partition is chosen and scored.

Further resources on where to find markers for your unique samples can be found at the bottom of this document.

## Review cell-type markers and assign cell types

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

SCARF automatically calculates other metrics for marker identification, and understanding their function and pitfalls is crucial when weighing the difference between cell type x and y on the same cluster.

- `fold_change` compares the average expression in a target cluster vs. all other cells; however, this can be skewed by high-expression noise or by a few extreme outliers.
- `frac_exp` helps to report the percentage of cells in a cluster with non-zero counts for a gene. This helps as say a gene has a 10 fold change, but has a small expressed fraction, it may not be the most useful marker to identify a cell type.
- `auc` measures how well a single gene's expression level predicts whether a cell belongs to a cluster, with values near 0.5 meaning no separation of clusters based on this gene, and 1 meaning that this gene can be perfectly identified with a specific cluster. This approach is insensitive to outliers and can be a more impartial metric to utilize.
- `score` is SCARF's unique specificity rank, which measures how uniquely a gene's expression is confined to this cluster on a 0-1 scale (summing to 1 across all clusters for each gene). A score near 1 flags a highly cluster-exclusive marker; a score near {math}`1 / n_{\text{clusters}}` indicates a ubiquitous housekeeping gene that is useless for naming; and a score near 0 indicates absence.
- `p_value_adjusted` is simply the Benjamini-Hochberg p value correction we discussed earlier to correct for the multiple hypothesis testing the two-sided Mann-Whitney U test employs.

## Assign initial cell types

After we have the marker table, we can begin to assign our initial cell types. We can begin by using our existing information of marker genes to see their spatial orientation in the UMAP embeddings. Comparing their orientation on the UMAP vs the clusters allows us to then target our search through the marker gene dataset, and thus we can identify if the metrics provided in the marker database support the marker genes we are using. If the metrics do support, then we can assign our initial cell identities; later, we can dig into the data further to see if multiple markers support our annotations.

```{code-cell}
ds.plots.embedding(
    layout=run["umap"],
    color_by=["CD14", "CD19", "CD8A", "CD4", "NCAM1", "IL3RA", clusters],
    n_columns=3,
    sort_values=True,
    legend_loc="on_data"
)
```

Here, we can see the UMAP of our select marker genes for our predicted cell types alongside the clusters they may be present inside of.

CD8A/CD4 lights up regions that may hold our candidate T cells. CD19 marks two separate blocks of potential B cells; CD14 marks the likely monocyte block; and NCAM1 (CD56) marks the NK-like block. IL3RA signal indicates plasmacytoid dendritic cells (pDCs) are present. The table below names the top-scoring cluster for each panel gene, and those winners motivate the initial names. Note that CD4's top-scoring cluster is the CD14 monocyte cluster, because human monocytes also express CD4; this is one reason a single gene is never enough to name a cluster.

To confirm the visual readings on the UMAP, we can now utilize the marker table.

```{code-cell}
panel_genes = ["CD14", "CD19", "CD8A", "CD4", "NCAM1", "IL3RA"]
panel_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
panel_stats = panel_markers[panel_markers["feature_name"].isin(panel_genes)]
panel_best = (
    panel_stats.sort_values("score", ascending=False)
    .groupby("feature_name", sort=False)
    .head(1)
    .set_index("feature_name")
    .reindex(panel_genes)[["group_id", "score", "frac_exp", "auc"]]
    .round(2)
    .reset_index()
)
panel_best
```

With the information in the markers table supporting our interpretation, we can move forward.

Clusters with these localized gene expression take an initial name for now as described below:

```{code-cell}
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
assert set(proposed_labels) == {str(value) for value in np.unique(cluster_values)}
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

Look for coherent programs rather than a single gene by comparing against the literature or existing databases (resources can be found at the end of this document). By using multiple markers, we can also begin to bridge towards not only identifying the identity of a cluster, but the state it may be in. The heatmap figure and the marker table printed beneath it name the leading genes for each cluster; the interpretations below read directly off those printed outputs rather than fixed cluster numbers.

In the heatmap, we can see that Cluster 1 displays a clear triplet of S100A12, VCAN, and CD14, which confirms our CD14-monocyte signature, while Cluster 2 expresses a distinct CTSL, TCF7L2, and SMIM25 program with only residual CD14 and FCGR3A (CD16) detected in every cell, indicating the presence of a second monocyte state (potentially non-classical monocytes).

Cluster 9 shows an IGHD, FCER2, and TCL1A trio, which is a classic naive B-cell program, while Cluster 3 expresses a distinct IGHA1, TNFRSF13B, and IGHG1 program, indicating class-switched immunoglobulin heavy chains and a memory B-cell state rather than a second naive pool. Looking at cytotoxic and CD8 programs, the signal spans clusters 6, 7, and 8 with different leading genes; cluster 6 is dominated by KLRF1, FGFBP2, and ADGRG1 (NK program), cluster 7 is led by GZMK, TRGC2, and KLRG1 with CD8A and CD8B detected in about 40% of its cells, and cluster 8 by LINC02446 and CD8B with CD8A close behind. Cluster 8 also carries CCR7, IL7R, and CD27 signal that gives it a naive-like cast within the CD8 T-cell programs. SELL (CD62L) is detected in most of its cells, but SELL is broad across this dataset (`score` 0.13), so the call stays naive-like rather than textbook naive. The decision between an NK-cell and a cytotoxic T-cell identity therefore rests on which exclusive markers lead each column, confirmed through the negative controls below.

The (large) Cluster 4 is dominated by ADTRP, ANKRD55, and FHIT; while these gene names may seem less familiar than canonical markers, cluster 4 can be identified as T cells by the process of elimination of B-cell, monocyte, and NK programs based on the genes above, supported by its CCR7 and IL7R signal. On the other hand, cluster 5 is led by TNFRSF4, LMNA, and PI16 with IL7R and CD4 detection, indicating a CD4+ T-cell state that refines our initial broad T-cell call.

Lastly, clusters with unique marker combinations, like Cluster 10's SERPINF1 and LILRA4 signals alongside IL3RA, confirm the pDC-like program. Cluster 8's NELL2 signal alongside CD8B provides a key example where our existing UMAPs can help in visualizing the spatial orientation of the expression of these genes.

```{code-cell}
heatmap_genes = [
    "CD14", "VCAN", "S100A12", "CTSL", "TCF7L2", "SMIM25", "FCGR3A", "FCER2",
    "IGHD", "IGHA1", "IGHG1", "TNFRSF13B", "KLRF1", "FGFBP2", "GNLY", "XCL1",
    "XCL2", "KLRC1", "GZMK", "TRGC2", "KLRG1", "CD8A", "LINC02446",
    "CD8B", "CD3D", "CD3E", "ADTRP", "ANKRD55", "FHIT", "TSHZ2", "TNFRSF4",
    "LMNA", "NPDC1", "PI16", "NKG7", "TCL1A", "NELL2", "LILRA4", "IL3RA",
    "SERPINF1", "ADGRG1", "AKR1C3", "CCR7", "IL7R", "CD27",
]
heatmap_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
heatmap_hits = heatmap_markers[heatmap_markers["feature_name"].isin(heatmap_genes)]
heatmap_table = (
    heatmap_hits.sort_values("score", ascending=False)
    .groupby("feature_name", sort=False)
    .head(1)
    .sort_values("score", ascending=False)
)
heatmap_table["proposed_identity"] = (
    heatmap_table["group_id"].astype(str).map(proposed_labels)
)
heatmap_table[["feature_name", "group_id", "score", "frac_exp", "proposed_identity"]]
```

## Validate annotations against negative controls

Going a step further, as you would in a real study, negative controls validate annotations by adding a layer of cell-type exclusivity, as confirming that a cluster not only has the right genes, but also properly silences the genes belonging to competing or mutually exclusive lineages. In the several markers section above, we hint at the idea of negative markers as a way to differentiate between different cell states, and even see it in use for cluster 4, in how we validate that cluster 4 is likely T cells, and cluster 6 NK cells, versus B-cell or monocyte alternatives.

**Negative controls can be verified by simply searching for them in our marker tables and analyzing their metrics:**

- `frac_exp` for low expression prevalence; our negative controls should have low detection within the target cluster (`frac_exp` {math}`\approx 0`).
- `score` for low exclusivity; a `score` near 0 means almost none of the gene's expression rank exists in the cluster you are studying.
- `auc` for exclusivity; AUC can be used as another metric for exclusivity of marker expression across multiple clusters.

For our examples, the negative controls we use are CD3D/CD3E in cluster 6 (NK cells), CD4 in cluster 8 (naive CD8 T cells), and MS4A1/CD14 in clusters 7 and 8. This is because true NK cells lack T-cell receptors, thus we can use CD3D/CD3E for our negative controls. For the naive CD8 T cells we attempt to identify, CD4 works as a negative control because it is dominantly expressed in helper T cells, not cytotoxic CD8 T cells. And finally, our MS4A1 (CD20) and CD14 across both clusters.

```{code-cell}
neg_genes = ["CD3D", "CD3E", "CD4", "MS4A1", "CD14"]
neg_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
neg_stats = neg_markers[neg_markers["feature_name"].isin(neg_genes)]
neg_stats[["group_id", "feature_name", "score", "frac_exp", "auc"]].sort_values(
    ["feature_name", "group_id"]
)
```

The table above shows the results of CD3D/CD3E across all ten clusters: both genes hit ceiling detection almost everywhere T cells truly live (`frac_exp` 0.97-0.99). In cluster 6, detection drops to 0.31/0.43 against a rest-of-data background of 0.65/0.66, with low exclusivity (`score` 0.11/0.11, `auc` 0.39/0.40): depleted, not absent, so on its own this disfavors rather than rules out T identity. Set against other marker genes like NKG7, GNLY, KLRF1, and FGFBP2 (which can be seen below) at detection of 0.92-1.00 against backgrounds of 0.04-0.21 in the same cluster, the combined evidence supports NK cells for cluster 6.

The same pattern exists with our other genes, with CD4 in cluster 8 sitting at `frac_exp` = 0.02 against a background of 0.37 (roughly 18-fold depleted) with score near zero, confirming the helper program is silenced while CD8A (`score` 0.51, `frac_exp` = 0.87) and CD8B (0.70, 0.95) support the naive CD8 identity (as shown below).

MS4A1 and CD14 sit near zero in clusters 7 and 8: MS4A1 at `frac_exp` 0.08/0.02 against backgrounds of 0.16/0.16, and CD14 at 0.01/0.03 against 0.21/0.19, with the `score` peaking at 0.01, confirming neither cluster carries B-cell or monocyte contamination.

```{code-cell}
anchor_genes = ["NKG7", "GNLY", "KLRF1", "FGFBP2", "CD8A", "CD8B"]
anchor_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
anchor_stats = anchor_markers[anchor_markers["feature_name"].isin(anchor_genes)]
anchor_stats[["group_id", "feature_name", "score", "frac_exp", "auc"]].sort_values(
    ["feature_name", "group_id"]
)
```

The cell below pins the winning cluster of every panel and anchor gene with asserts, so the identities claimed above stay guarded if the partition ever changes.

```{code-cell}
guard_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
best_cluster = (
    guard_markers[guard_markers["feature_name"].isin(
        ["NKG7", "GNLY", "KLRF1", "FGFBP2", "CD8A", "CD8B", "IL3RA",
         "LILRA4", "IGHD", "IGHA1", "CD14", "CDKN1C", "TCF7L2", "MS4A1",
         "CD27", "CCR7", "GZMK", "PRF1"]
    )]
    .sort_values("score", ascending=False)
    .groupby("feature_name", sort=False)
    .head(1)
    .set_index("feature_name")["group_id"]
    .astype(str)
    .to_dict()
)
assert best_cluster["NKG7"] == "6"
assert best_cluster["GNLY"] == "6"
assert best_cluster["KLRF1"] == "6"
assert best_cluster["FGFBP2"] == "6"
assert best_cluster["CD8A"] == "8"
assert best_cluster["CD8B"] == "8"
assert best_cluster["CD27"] == "8"
assert best_cluster["CCR7"] in {"4", "8"}
assert best_cluster["GZMK"] == "7"
assert best_cluster["PRF1"] == "6"
assert best_cluster["IL3RA"] == "10"
assert best_cluster["LILRA4"] == "10"
assert best_cluster["IGHD"] == "9"
assert best_cluster["IGHA1"] == "3"
assert best_cluster["CD14"] == "1"
assert best_cluster["CDKN1C"] == "2"
assert best_cluster["TCF7L2"] == "2"
assert best_cluster["MS4A1"] in {"3", "9"}
best_cluster
```

## Write & visualize the final reviewed mapping

With all of the analysis we performed above, we can now finalize our annotations as follows. This example records the broad teaching labels supported in {doc}`scrna_seq`.

```{code-cell}
final_labels = {
    "1": "CD14 monocytes",
    "2": "monocytes",
    "3": "memory B cells",
    "4": "T cells",
    "5": "CD4+ T cells",
    "6": "NK cells",
    "7": "CD8+ T cells",
    "8": "naive CD8 T cells",
    "9": "naive B cells",
    "10": "pDC-like cells",
}
observed = {str(value) for value in np.unique(cluster_values)}
assert observed == set(final_labels)

analysis_cells = np.asarray(run.cells.fetch_all("I"), dtype=bool)
cell_type = np.full(len(analysis_cells), "Not analyzed", dtype=object)
cell_type[analysis_cells] = [final_labels[str(value)] for value in cluster_values]
ds.cells.insert("reviewed_cell_type", cell_type, overwrite=True)
pd.Series(cell_type[analysis_cells]).value_counts()
```

With all our updated annotations, we can now visualize them to see the difference between our initial cell types versus our final ones.

```{code-cell}
initial_cell_type = np.full(len(analysis_cells), "Not analyzed", dtype=object)
initial_cell_type[analysis_cells] = [proposed_labels[str(value)] for value in cluster_values]
ds.cells.insert("proposed_cell_type", initial_cell_type, overwrite=True)
ds.plots.embedding(
    layout=run["umap"],
    color_by="proposed_cell_type",
    legend_loc="right",
)

ds.plots.embedding(
    layout=run["umap"],
    color_by="reviewed_cell_type",
    legend_loc="right",
)
```

## Important caveats to consider regarding annotation

**Clustering algorithms discretize continuous biological spectrums:** Graph clustering (such as Leiden) forces cells into rigid, separate categories. In reality, biological processes, such as T-cell activation, monocyte differentiation, and exhausted states, exist along continuous transcriptional trajectories. Neighboring clusters often represent transitional points along a gradient rather than isolated, distinct cell types.

- **Heterotypic doublets may mimic "novel" transitional populations:** Droplets that capture two different cells (e.g., a T cell and a B cell) generate hybrid transcriptomes. Because they express moderate levels of conflicting marker programs, they frequently group into small, intermediate clusters. Always evaluate doublet scores and negative markers before proceeding, thus why quality control is so critical. In this run, the tiny clusters 2 and 10 average doublet scores of 0.28 and 0.24 against 0.10 overall, with maxima of 0.32 and 0.26 and no extreme outliers; combined with their exclusive CDKN1C/TCF7L2 and IL3RA/LILRA4 programs, they read as real rare populations rather than hybrids, but both warrant doublet review in a real study.
- **Ambient RNA contaminates negative controls:** Cell lysis during tissue dissociation releases highly abundant transcripts (such as lysozyme, hemoglobin, or ribosomal proteins) into the cell suspension. These ambient transcripts enter droplets indiscriminately, meaning negative markers rarely display a literal mathematical zero (`frac_exp` = 0.00). This is why we use other metrics like `score` and `auc`, which become key in ensuring our negative controls stay negative.
- **Granularity depends on clustering resolution:** The number of clusters discovered is a mathematical function of graph resolution, not an objective count of biological lineages. Coarse resolutions will merge rare populations (such as pDCs or innate lymphoid cells) into dominant clusters, while fine resolutions will artificially fracture homogenous populations into arbitrary sub-clusters. Thus, probing across multiple clusters can be useful to determine the cell types that exist inside of your dataset. Subset graph construction and validation now live in {doc}`clustering`.
- **Marker statistics are not replicate-aware differential expression:** `p_value_adjusted` applies Benjamini-Hochberg correction within this one-versus-rest marker test over cells. It is useful for marker ranking but does not model biological replicates or study-level variation.

Keep the label map, marker reference, run ID, and review rationale in the study record. Cluster IDs can change when the graph or partition changes. Overlap-based `smart_label` helps compare two partitions, but it is not ontology annotation; that partition-comparison role belongs in {doc}`clustering`.

For scATAC-seq, {doc}`scatac_seq` uses GeneScores to display marker accessibility. GeneScores are accessibility summaries, not measured RNA expression. Use {doc}`mapping_and_label_transfer` when a query should inherit labels from a fixed reference rather than be annotated de novo.

## Annotation Resources

### Curated Single-Cell Marker Databases

- **[CellMarker 2.0](http://bio-bigdata.hrbmu.edu.cn/CellMarker/):** A comprehensive, manually curated database cataloging over 13,000 cell markers across human and mouse tissues, including both normal and clinical disease models.
- **[PanglaoDB](https://panglaodb.se/):** An open database of single-cell RNA-seq markers covering hundreds of cell types across major mammalian organs, providing computational specificity scores for each marker.
- **[Azimuth / HuBMAP Reference Atlases](https://azimuth.hubmapconsortium.org/):** Pre-annotated, expert-verified reference maps for single-cell data across organs (kidney, lung, pancreas, motor cortex, PBMC). Azimuth allows you to inspect canonical marker hierarchies directly.
- **[The Human Protein Atlas (Blood &amp; Single-Cell Atlas)](https://www.proteinatlas.org/):** Combines single-cell RNA sequencing data with antibody-based protein profiling across tissues and circulating blood compartments.

### Automated Annotation & Label-Transfer Frameworks

- **[ScType](https://github.com/IanevskiAleksandr/sc-type):** An automated marker-based annotation tool supported by a curated database that explicitly documents both **positive marker sets** and **negative control markers** for hundreds of cell lineages.
- **[CellTypist](https://www.celltypist.org/):** A machine-learning platform with specialized, pre-trained logistic regression models for immune cell phenotyping across healthy and diseased tissues.
- **[SingleR](https://bioconductor.org/packages/release/bioc/html/SingleR.html):** Performs unbiased, automated cell-type assignment by computing Spearman rank correlations between your single-cell clusters and bulk/microarray reference datasets (such as Blueprint-ENCODE and HPCA).
- **[CyteType](https://www.nygen.io/products/cytetype):** Automated annotation and clustering tool created by Nygen.

### Marker-Set Enrichment Platforms

If you have computed the top 10-20 marker genes for an uncharacterized cluster and need to query potential candidate identities:

- **[Enrichr](https://maayanlab.cloud/Enrichr/):** Paste your top marker gene list and evaluate over-representation against the **CellMarker Augmented**, **PanglaoDB Augmented**, or **ARCHS4 Tissues** gene-set libraries.
- **[ToppGene Suite (ToppFun)](https://toppgene.cchmc.org/):** Matches custom gene lists against cell-type specific signatures, Gene Ontology (GO) terms, and pathway databases to infer functional state and lineage.

### Immunophenotyping & Negative-Gating References

- **[BioLegend Cell Markers](https://www.biolegend.com/en-us/cell-markers) & [BD Biosciences CD Marker Handbooks](https://www.bdbiosciences.com/):** Reference posters and technical guides defining classical immunophenotyping panels, lineage-negative ({math}`\text{Lin}^-`) gating exclusion cocktails, and surface marker hierarchies.
- **[Optimized Multicolor Immunofluorescence Panels (OMIPs)](https://onlinelibrary.wiley.com/journal/15524930):** Peer-reviewed flow and mass cytometry gating panels published in *Cytometry Part A*, detailing validated gating trees and the negative markers used to dump non-target lineages.
