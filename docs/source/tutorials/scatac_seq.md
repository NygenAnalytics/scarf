---
description: Interpret scATAC-seq clusters with GeneScore marker maps.
jupytext:
  formats: ipynb,md:myst
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
# sc-ATAC Seq Primer

Single cell Assay for Transposase-Accessible Chromatin with sequencing (sc-ATAC seq) is a method that is used to map open and accessible chromatin regions across the entire genome (i.e. where DNA is available for transcription factors and transcription machinery). In ATAC-seq, the features of the data are the **peaks**, which are segments of open chromatin. A good way to conceptualize this is that 1 peak is equivalent to 1 candidate regulatory element that was open in the sample. To identify where each peak is in the genome, you require a **genomic coordinate**, which is where the peak is located on the genome. Lots of peaks at the same range of genomic coordinates represent areas of high chromatin accessibility, vice versa for less peaks.

# Identify accessibility populations with scATAC-seq

Herein, we utilized a preran analysis to identify the accessibility states of Peripheral Blood Mononuclear Cells (PBMCs); We then validate our interpretation with gene-score marker maps, which are UMAPs of estimated gene activity, made by adding up nearby (peaks) open regions for each gene.

## Open the prepared ATAC result

```{code-cell}
import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_10K_pbmc-v1_atacseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="ATAC",
    nthreads=4,
)
```

Since the ATAC result comes with the primary analysis complete, we simply retrieve the information relevant for this tutorial: the UMAP & leiden clusters.

```{code-cell}
[clusters] = ds.list_artifacts(
    from_assay="ATAC",
    kind="cluster_labels",
    operation="run_leiden_clustering", 
    complete_only=True,
)
[umap] = ds.list_artifacts(
    from_assay="ATAC",
    kind="embedding",
    operation="run_umap", 
    complete_only=True,
)
```

### Does the accessibility graph contain distinct populations?

```{code-cell}
ds.plots.embedding(
    layout=umap,
    color_by=clusters,
    legend_loc="on_data",
)
```

Visualizing the cells based on their chromatin accessibility allows us to identify cells that share similar regulatory profiles. In this embedding above, each point is a cell, with cells being positioned close together if they share the same open genomic peaks.

Overlaying our cluster labels onto this map reveals distinct cell groupings, suggesting separate populations that may occupy distinct chromatin states.

But how do we move beyond arbitrary cluster numbers and determine the biological identities of these populations? For this, we use gene-score marker maps. By estimating the regulatory activity of canonical marker genes across the map, we can verify and identify whether each cluster corresponds to an expected immune cell type (such as T cells, B cells, or monocytes in our case).

## Build GeneScores from the coordinates of peaks

Building our gene-score marker maps require taking the physical coordinates of the peaks and translating them to their corresponding gene names. To do this, we use a BED (Browser Extensible Data) file, which acts as a dictionary that allows us to translate coordinates to known genes and promoter regions.

```{code-cell}
annotations = scarf.cytebase.connect("scarf_docs").download_dataset(
    "annotations",
    destination="scarf_datasets",
)

ds.add_melded_assay(
    from_assay="ATAC",
    external_bed_fn=(
        f"{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz" # Use the GRCH37 BED file
    ),
    peaks_col="ids",
    renormalization=False,
    assay_label="GeneScores",
    assay_type="RNA",
)
```

Some genes, however, in the BED file have no corresponding peak in the dataset, thus they are not added to matrix that holds the information regarding gene-scores. The genes that are successfully mapped can then be utilized to infer cell identities.

### What broad cell identities fit the observed accessibility regions?

With the computed gene-scores, we can now move towards identifying the identity of each cluster. To do this, we simply color our accessibility graph with specific genes that are used as proxies for cell identities.

```{code-cell}
ds.plots.embedding(
    layout=umap,
    from_assay="GeneScores",
    color_by=["CD3D", "MS4A1", "LEF1", "NKG7", "TREM1", "LYZ"],
    clip_fraction=0.01,
    n_columns=3,
    sort_values=True,
)
```

Our observation of CD3D and LEF1 appearing in the same clusters supports T-cell accessibility, with MS4A1 supporting B cells, and NKG7 highlights cytotoxic or NK-like regions. TREM1 alongside LYZ supports presence of myeloid populations. These maps support broad lineage
interpretation, but they do not assign every cluster a definitive cell type from this small panel of genes. To gain more confidence in assigning a definitive cell type, you can utilize a larger panel of genes alongside positive/negative controls (genes you would expect to observe in a cell type vs. genes you would not.)

## Substitute your own input

For substituting your own peak-count matrix for your analysis, the corresponding code below outlines this task briefly.

```python
cells = own_ds.auto_filter_cells(method="gaussian")
peaks = own_ds.select_prevalent_peaks(cells, top_n=25_000)
normalized = own_ds.run_normalization(cells, peaks)
lsi = own_ds.run_lsi(normalized, dims=50, skip_first=True)
initialization = own_ds.build_embedding_initialization(lsi)
neighbors = own_ds.query_neighbors(own_ds.build_ann_index(lsi), k=21)
graph = own_ds.build_connectivity_map(neighbors)
layout = own_ds.run_umap(graph, initialization)
clusters = own_ds.run_leiden_clustering(graph)
```

The general methodolgy to follow in SCARF is to convert and open the new count matrix as shown in {doc}`import_and_export`, then choose filtering, peak count, LSI dimensions, and clustering for your data. Some great resources during this process can be found in {doc}`quality_control`, {doc}`feature_selection`, {doc}`dimensionality_reduction`, {doc}`graph_construction`, and {doc}`clustering`. If you need to revert back to a previous point in the analysis, {doc}`reuse_and_tracing` is the best resource. 

## Limits of this result

- GeneScores are accessibility proxies, not measured RNA expression.
- Gene annotation and peak coordinates must use compatible genome builds.
- Marker accessibility supports broad states here; more loci and external evidence are needed for
  final annotation.
