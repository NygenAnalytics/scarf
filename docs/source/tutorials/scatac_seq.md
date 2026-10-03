---
description: Interpret scATAC-seq clusters with GeneScores marker maps.
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
# Interpret cell populations with scATAC-seq

Single-cell ATAC-seq measures chromatin accessibility: regions where DNA is accessible
to the assay. Its features are **peaks**, genomic intervals with accessibility signal.
A peak may overlap a gene, promoter, or other regulatory region; its coordinates tell us
where to look in the reference genome.

Here we use a prepared PBMC analysis to find populations with similar accessibility.
We then combine peaks overlapping gene regions into **GeneScores** and ask which broad
cell identities their marker patterns support.

## Open the prepared ATAC result

```{code-cell}
# Open count stores and run Scarf analyses.
import scarf

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_10K_pbmc-v1_atacseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", default_assay="ATAC", nthreads=4)
# Inspect the opened store's cells and features.
ds
```

Preprocessing is complete in this store. Retrieve its UMAP and Leiden clusters:

```{code-cell}
# Select the saved ATAC clusters; require exactly one match.
[clusters] = ds.list_artifacts(
    from_assay="ATAC",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    complete_only=True,
)
# Select the saved ATAC UMAP; require exactly one match.
[umap] = ds.list_artifacts(
    from_assay="ATAC",
    kind="embedding",
    operation="run_umap",
    complete_only=True,
)
# Inspect the saved ATAC clustering and layout.
{"clusters": clusters, "UMAP": umap}
```

## Find accessibility populations

```{code-cell}
# Locate the ATAC clusters before interpreting their markers.
ds.plots.embedding(layout=umap, color_by=clusters, legend_loc='on_data')
```

Each point is a cell. Nearby cells tend to share accessible peaks. The numbered clusters
give us groups to investigate, but we need marker evidence to attach biological names.

## Connect peaks to genes

A BED file describes genomic intervals and their gene names. Use an annotation built for
the same reference genome as the peaks; this dataset uses GRCh37. Scarf combines the
accessibility signal from overlapping peaks into a new assay named `GeneScores`.

Here `renormalization=False` keeps the combined scores without an extra rescaling during
assay construction. `assay_type="RNA"` gives the new assay RNA-style normalization for
plotting. This does not turn accessibility into measured RNA expression.

```{code-cell}
# Download gene intervals for the dataset's reference genome.
annotations = scarf.cytebase.connect("scarf_docs").download_dataset(
    "annotations",
    destination="scarf_datasets",
)
# Combine peaks overlapping gene intervals into GeneScores.
ds.add_melded_assay(
    from_assay="ATAC",
    external_bed_fn=f"{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz",
    renormalization=False,
    assay_label="GeneScores",
    assay_type="RNA",
)
# Inspect the opened store's cells and features.
ds
```

Annotation genes with no overlapping peaks remain as zero-count features and are marked
invalid. They are not evidence that a gene is inaccessible in every biological setting.

## Read the marker maps

```{code-cell}
# Compare GeneScores for lymphoid and myeloid markers.
ds.plots.embedding(
    layout=umap,
    from_assay="GeneScores",
    color_by=["CD3D", "MS4A1", "LEF1", "NKG7", "TREM1", "LYZ"],
    n_columns=3,
    sort_values=True,
)
```

CD3D and LEF1 support T-cell regions, MS4A1 supports B cells, and NKG7 highlights
cytotoxic or NK-like populations. TREM1 together with LYZ supports myeloid populations.
Look for agreement across markers rather than assigning a type from one bright panel.
See {doc}`annotation` for broader panels and checks against alternative identities.

## Start from your own peak counts

Convert and open a peak-count matrix with `default_assay="ATAC"` as shown in
{doc}`import_and_export`. After reviewing its quality measurements in
{doc}`quality_control`, this sequence uses the default settings to build a first result:

```python
# Open your own ATAC count store.
own_ds = scarf.DataStore("my_atac.zarr", default_assay="ATAC")
# Filter cells using the available quality measurements.
cells = own_ds.auto_filter_cells()
# Select peaks detected across the retained cells.
peaks = own_ds.select_prevalent_peaks(cells)
# Normalize counts over the selected features.
normalized = own_ds.run_normalization(cells, peaks)
# Reduce TF-IDF normalized accessibility with LSI.
lsi = own_ds.run_lsi(normalized)
# Build starting coordinates for the embedding.
initialization = own_ds.build_embedding_initialization(lsi)
# Build the nearest-neighbor search index from LSI.
neighbor_index = own_ds.build_ann_index(lsi)
# Find neighbors of each cell in the reduced space.
neighbors = own_ds.query_neighbors(neighbor_index)
# Convert neighbor distances into weighted connectivity.
graph = own_ds.build_connectivity_map(neighbors)
# Keep the saved UMAP coordinates for the figures.
layout = own_ds.run_umap(graph, initialization)
# Find groups of cells in the graph.
clusters = own_ds.run_leiden_clustering(graph)
```

For ATAC, normalization uses TF-IDF and dimensionality reduction uses LSI. Treat defaults
as a starting point: inspect the retained cells and marker patterns before adjusting
peak selection, LSI dimensions, or neighborhood size. The next steps are explained in
{doc}`feature_selection`, {doc}`dimensionality_reduction`, and {doc}`graph_construction`.

## What GeneScores leave out

- Accessible chromatin does not guarantee transcription or protein production.
- Scores depend on the annotation intervals. Gene-body scores do not capture every
  distant regulatory element that may affect a gene.
- A small marker panel supports broad lineages. Finer identities need additional
  markers and independent evidence.
