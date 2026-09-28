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

After we have the marker table, we can begin to assign our initial cell types. We can begin by first using our existing information of marker genes to see their spatial orientation in the UMAP embeddings. Comparing their orientation on the UMAP vs the clusters allows us to then target our search through the marker gene dataset, and thus we can identify if the metrics provided in the marker database support the marker genes we are using. If the metrics do support, then we can assign our initial cell identities; Later, we can dig into the data further to see if multiple markers support our annotations.

```{code-cell}
ds.plots.embedding(
    layout=run["umap"],
    color_by=["CD3D", "CD4", "CD8A", "MS4A1", "CD14", "NKG7", "IL3RA", "FCGR3A", clusters],
    n_columns=3,
    sort_values=True,
    legend_loc="on_data"
)
```

Here, we can see the UMAP of our select marker genes for our predicted cell types alongside the clusters they may be present inside of.

CD3D lights up clusters that may hold our candidate T cells. MS4A1 marks two separate blocks of potential B cells, CD14 marks the likely monocyte block, and NKG7 marks the NK-like block. The prescense of IL3RA also indicates  plasmacytoid dendritic cells (pDCs) being present. FCGR3A is also used to identify a specific type of monocyte, thus why we include it. One small cluster lights up none of the panel genes and stays unresolved for now; the heatmap below resolves it through its own top markers.

To confirm the visual readings on the UMAP, we can now utilize the marker table.

```{code-cell}
panel_genes = ["CD3D", "CD4", "CD8A", "MS4A1", "CD14", "NKG7", "IL3RA", "FCGR3A"]
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
    "2": "FCGR3A monocytes",
    "3": "B cells",
    "4": "T cells",
    "5": "T cells",
    "6": "T cells",
    "7": "NK cells",
    "8": "NK cells",
    "9": "T cells",
    "10": "B cells",
    "11": "pDC-like cells",
    "12": "unresolved",
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

In the heatmap, we can see that Cluster 1 displays a clear triplet of S100A12, VCAN, and CD14, which confirms our CD14-monocyte signature, while Cluster 2 expresses a distinct CTSL, TCF7L2, and SMIM25 program with only residual CD14, indicating the presence of a second monocyte state (potentially non-classical monocytes).

Cluster 10 shows a FCER2, IGHD, and TCL1A trio, which is a classic naive B-cell program, while Cluster 3 expresses a distinct IGHA1, TNFRSF13B, and IGHG1 program, indicating class-switched immunoglobulin heavy chains and a memory B-cell state rather than a second naive pool. Looking at cytotoxicity-linked genes, the signal spans clusters 6, 7, 8, and 9 with different leading genes; cluster 7 is dominated by FGFBP2, ADGRG1, and AKR1C3 (NK program), cluster 8 by XCL1, KLRC1, and XCL2 (a small second NK-like program), cluster 6 is led by GZMK, TRGC2, and KLRG1, and cluster 9 by CD8B with CD8A (cytotoxic-T programs), meaning the decision between an NK-cell and a cytotoxic T-cell identity rests on which exclusive markers lead each column, confirmed through the negative controls below.

The (large) Cluster 4 is dominated by ADTRP, ANKRD55, and TSHZ2; while these gene names may seem less familiar than canonical markers, cluster 4 can be identified as T cells by the process of elimination of B-cell, monocyte, and NK programs based on the genes above, supported by its CCR7 and IL7R signal. On the other hand, cluster 5 is led by TNFRSF4, NPDC1, and PI16 with IL7R and CD4 detection, indicating a CD4+ T-cell state that refines our initial broad T-cell call.

Lastly, clusters with unique marker combinations, like Cluster 11's LILRA4 and SERPINF1 signals alongside IL3RA, confirm the pDC-like program, while Cluster 12's F13A1, PTGS1, and PPBP triplet identifies platelets, which is why it stayed dark for every lymphoid and myeloid panel gene above. Cluster 9's NELL2 signal alongside CD8B provides a key example where our existing UMAPs can help in visualizing the spatial orientation of the expression of these genes.

```{code-cell}
heatmap_genes = [
    "CD14", "VCAN", "S100A12", "CTSL", "TCF7L2", "FCER2", "IGHD",
    "IGHA1", "TNFRSF13B", "KLRF1", "FGFBP2", "GNLY", "XCL1", "GZMK", "TRGC2", "CD8A",
    "CD8B", "CD3D", "CD3E", "ADTRP", "ANKRD55", "TNFRSF4",
    "NKG7", "TCL1A", "NELL2", "LILRA4", "IL3RA", "PPBP",
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

Going a step further, as you would in a real study, negative controls validate annotations by adding a layer of cell-type exclusivity, as confirming that a cluster not only has the right genes, but also properly silences the genes belonging to competing or mutually exclusive lineages. In the several markers section above, we hint at the idea of negative markers as a way to differentiate between different cell states, and even see it in use for cluster 4, in how we validate that cluster 4 is likely T cells, and clusters 7 and 8 NK cells, versus B-cell or monocyte alternatives.

**Negative controls can be verified by simply searching for them in our marker tables and analyzing their metrics:**

- `frac_exp` for low expression prevalence; our negative controls should have low detection within the target cluster (`frac_exp` {math}`\approx 0`).
- `score` for low exclusivity; a `score` near 0 means almost none of the gene's expression rank exists in the cluster you are studying.
- `auc` for exclusivity; AUC can be used as another metric for exclusivity of marker expression across multiple clusters.

For our examples, the negative controls we use are CD3D/CD3E in clusters 7 and 8 (NK cells), CD4 in cluster 9 (cytotoxic CD8+ T cells), and MS4A1/CD14 in clusters 6 and 9. This is because true NK cells lack T-cell receptors, thus we can use CD3D/CD3E for our negative controls. For our cytotoxic CD8+ T cells we attempt to identify, CD4 works because it is dominantly expressed in helper T cells, not cytotoxic (CD8+) T cells that we attempt to isolate here. And finally, our MS4A1 (CD20) and CD14 across both clusters.

```{code-cell}
neg_genes = ["CD3D", "CD3E", "CD4", "MS4A1", "CD14", ]
neg_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
neg_stats = neg_markers[neg_markers["feature_name"].isin(neg_genes)]
neg_stats[["group_id", "feature_name", "score", "frac_exp", "auc"]].sort_values(
    ["feature_name", "group_id"]
)
```

The table above shows the results of CD3D/CD3E across all twelve clusters: both genes hit ceiling detection almost everywhere T cells truly live, but in cluster 7, their exclusivity collapses to background with their low `score` (0.13/0.11) and `auc` below 0.5 (0.43/0.42), and in cluster 8 CD3D is fully silent (`frac_exp` = 0.00) while CD3E shows only low-exclusivity ambient detection (`score` 0.04, `auc` 0.31, `frac_exp` = 0.41). Set against other marker genes like NKG7, GNLY, KLRF1, and FGFBP2 (which can be seen below) that have detection of 0.97 or above in the same clusters, this rules out T-cell identity for clusters 7 and 8, which are now NK cells.

The same pattern exists with our other genes, with CD4 in cluster 9 sitting at `frac_exp` = 0.02 with score near zero, confirming the helper program is silenced while CD8A (`frac_exp` = 0.87) and CD8B (`frac_exp` = 0.95) support the cytotoxic CD8 identity (as shown below).

MS4A1 and CD14 sit near zero in clusters 6 and 9 with `frac_exp` detection at 0.01-0.08, with the `score` peaking at 0.01, confirming neither cluster carries B-cell or monocyte contamination.

```{code-cell}
anchor_genes = ["NKG7", "GNLY", "KLRF1", "FGFBP2", "CD8A"]
anchor_markers = ds.get_markers(marker=markers, min_score=-1, min_frac_exp=-1)
anchor_stats = anchor_markers[anchor_markers["feature_name"].isin(anchor_genes)]
anchor_stats[["group_id", "feature_name", "score", "frac_exp", "auc"]].sort_values(
    ["feature_name", "group_id"]
)
```

## Write & visualize the final reviewed mapping

With all of the analysis we performed above, we can now finalize our annotations as follows. This example records the broad teaching labels supported in {doc}`scrna_seq`.

```{code-cell}
final_labels = {
    "1": "CD14 monocytes",
    "2": "FCGR3A monocytes",
    "3": "memory B cells",
    "4": "T cells",
    "5": "CD4+ T cells",
    "6": "CD8+ T cells",
    "7": "NK cells",
    "8": "NK cells",
    "9": "CD8+ T cells",
    "10": "naive B cells",
    "11": "pDC-like cells",
    "12": "Platelets",
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

- **Heterotypic doublets may mimic "novel" transitional populations:** Droplets that capture two different cells (e.g., a T cell and a B cell) generate hybrid transcriptomes. Because they express moderate levels of conflicting marker programs, they frequently group into small, intermediate clusters. Always evaluate doublet scores and negative markers before proceeding, thus why quality control is so critical.
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
