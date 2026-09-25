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

Single cell Assay for Transposase-Accessible Chromatin with sequencing (sc-ATAC seq) is a method that is used to map open and accessible chromatin regions across the entire genome (i.e. where DNA is available for transcription factors and transcription machinery). In ATAC-seq, the features of the data are the **peaks**, which are segments of open chromatin. A good way to conceptualize this is that 1 peak is equivalent to 1 candidate regulatory element that was open in the sample. When you plot the peaks, you get the accessibility graph. To identify where each peak is in the genome, you require a **genomic coordinate**, which is where the peak is located on the genome. Lots of peaks at the same range of genomic coordinates represent areas of high chromatin accessibility, vice versa for less peaks.

# Identify accessibility populations with scATAC-seq

Herein, we utilized a preran analysis to identify the accessibility states of Periphera Blood Mononuclear Cells (PBMCs); We then validate our interpration with gene-score marker maps, which are UMAPs of estimated gene activity, made by adding up nearby (peaks) open regions for each gene.

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

The ATAC result comes with all quality control complete; here, we simply recompute the clusters & the UMAP for those clusters.

```{code-cell}
[clusters] = ds.list_artifacts(
    from_assay="ATAC",
    kind="cluster_labels",
    operation="run_leiden_clustering", # Recompute clusters
    complete_only=True,
)
[umap] = ds.list_artifacts(
    from_assay="ATAC",
    kind="embedding",
    operation="run_umap", # Recompute UMAP
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

GeneScores summarize TF-IDF-normalized accessibility over gene bodies and their promoter regions.
The downloaded BED file uses the same GRCh37 coordinate build as this peak matrix.

```{code-cell}
annotations = scarf.cytebase.connect("scarf_docs").download_dataset(
    "annotations",
    destination="scarf_datasets",
)

ds.add_melded_assay(
    from_assay="ATAC",
    external_bed_fn=(
        f"{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz"
    ),
    peaks_col="ids",
    renormalization=False,
    assay_label="GeneScores",
    assay_type="RNA",
)
```

Some annotation features have no overlapping peak in this matrix. They remain zero-count features;
the interpretation below relies only on the displayed loci with observed signal.

### Question: which broad lineages explain the accessibility regions?

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

CD3D and LEF1 support T-cell accessibility, MS4A1 supports B cells, NKG7 highlights cytotoxic or
NK-like regions, and TREM1 with LYZ supports myeloid populations. These maps justify broad lineage
interpretation, but they do not support assigning every cluster a definitive cell type from this
small panel alone.

## Substitute your own input

For another peak-count matrix, the corresponding atomic path is:

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

Convert and open the new count matrix as shown in {doc}`import_and_export`, then choose filtering,
peak count, LSI dimensions, and clustering policy for that dataset. The {doc}`quality_control`,
{doc}`feature_selection`, {doc}`dimensionality_reduction`,
{doc}`graph_construction`, and {doc}`clustering` guides own those decisions. Use
{doc}`reuse_and_tracing` when you need to inspect the exact artifact lineage.

## Limits of this result

- GeneScores are accessibility proxies, not measured RNA expression.
- Gene annotation and peak coordinates must use compatible genome builds.
- Marker accessibility supports broad states here; more loci and external evidence are needed for
  final annotation.
