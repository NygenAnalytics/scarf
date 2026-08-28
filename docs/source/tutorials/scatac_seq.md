---
description: Analyze chromatin accessibility from peak counts through an LSI graph, clusters, and gene scores.
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

# Chromatin accessibility with scATAC-seq

This tutorial follows one recommended path from a peak-count matrix to broad chromatin-accessibility populations and interpretable gene scores.

## Prerequisites

- Scarf installed with the `extra` optional dependencies

## What you will learn

- Open a prepared 10x scATAC-seq Zarr store
- Follow an LSI-based graph and clustering chain by exact artifact reference
- Create a gene-score assay from peak annotations

## Dataset

```{code-cell} ipython3
import numpy as np

import scarf

scarf.configure_output(level="WARNING", progress=False)
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="tenx_10K_pbmc-v1_atacseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", default_assay="ATAC", nthreads=4)

```

## 1. Open the prepared peak-count analysis

The downloaded store was rebuilt with this Scarf version. It contains the raw peak counts and a
complete analysis chain, so this page can inspect exact lineage without refitting the expensive
wide-matrix reduction.

Start from the stored Leiden output and follow each recorded input upstream.

```{code-cell} ipython3
def input_ref(ref, name):
    return scarf.ArtifactRef.from_dict(ds.inspect_artifact(ref).inputs[name])


[clusters] = ds.list_artifacts(
    from_assay="ATAC", kind="cluster_labels", complete_only=True
)
graph = input_ref(clusters, "graph")
neighbors = input_ref(graph, "neighbors")
ann_index = input_ref(neighbors, "ann_index")
lsi = input_ref(ann_index, "coordinates")
normalized = input_ref(lsi, "normalized")
cell_selection = input_ref(normalized, "cell_selection")
peak_features = input_ref(normalized, "feature_selection")
input_cell_selection = input_ref(cell_selection, "prior_cell_selection")
[umap] = ds.list_artifacts(from_assay="ATAC", kind="embedding", complete_only=True)
assert input_ref(umap, "graph") == graph
```

## 2. Inspect cell filtering

Scarf initialization records accessible peaks per cell (`nFeatures`), total fragments or cut sites
per cell (`nCounts`), and per-peak detection counts used by explicit selection producers.

Inspect fragment and peak-count distributions before filtering.
Thresholds are dataset-specific.

```{code-cell} ipython3
qc_cols = [c for c in ("ATAC_nCounts", "ATAC_nFeatures") if c in ds.cells.columns]
ds.plots.distribution(
    keys=qc_cols,
    cell_selection=input_cell_selection,
    kind="violin",
    max_points=2000,
)
```

The prepared `auto_filter_cells` result stores Scarf's global automatic bounds and the exact input
selection. It did not delete rows.
The {doc}`quality_control` guide covers manual thresholds, sample-aware filtering, and the ATAC metrics Scarf does and does not provide.

```{code-cell} ipython3
input_mask = np.asarray(ds.load_artifact(input_cell_selection)["values"][:], dtype=bool)
cell_mask = np.asarray(ds.load_artifact(cell_selection)["values"][:], dtype=bool)
print(f"Cells in input selection: {int(input_mask.sum())}")
print(f"Cells after filter: {int(cell_mask.sum())}")
```

## 3. Select prevalent peaks

For scATAC-seq data, features are ranked by TF-IDF-normalized values summed across cells.
The top features are returned as an immutable selection artifact for downstream steps.
Here we retain 25,000 peaks, slightly more than one quarter of the available peaks.

```{code-cell} ipython3
peak_values = np.asarray(ds.load_artifact(peak_features)["values"][:])
print("Selected peaks:", int(peak_values.sum()))
```

The retained peaks should be present across enough cells to support stable neighbour comparisons without collapsing the assay onto only the most common open regions.

## 4. Inspect the LSI neighbourhood graph

For scATAC-Seq datasets, Scarf uses TF-IDF normalization during `run_normalization`.
The selected feature artifact is used for graph construction.
Dimension reduction uses LSI rather than PCA.
The stored index, neighbour, and connectivity artifacts match the RNA graph workflow.

LSI reduction of scATAC-Seq is known to capture the sequencing depth of cells in the first LSI dimension.
`run_lsi` skips that component by default (`skip_first=True`).

```{code-cell} ipython3
connectivity = ds.load_graph(graph)
connectivity.shape, connectivity.nnz
```

The sparse graph shape confirms that it covers the selected cells.

TF-IDF, LSI choices, and graph tuning are covered in {doc}`graph_construction` and {doc}`dimensionality_reduction`.

## 5. Inspect UMAP and clustering


UMAP and tSNE use the same neighbourhood-graph path as scRNA-seq, so the ATAC workflow looks the same after the graph is built.

The stored UMAP and Leiden artifacts both name the same exact graph input.

```{code-cell} ipython3
cluster_values = np.asarray(ds.load_artifact(clusters)["values"][:])
np.unique(cluster_values, return_counts=True)
```

Cluster sizes are worth a look before plotting: a resolution that is too high splits one accessibility state into several small clusters.

UMAP and Leiden results remain immutable artifacts aligned to the stored selection.

Plot the UMAP embedding colored by Leiden clusters:

```{code-cell} ipython3
ds.plots.embedding(layout=umap, color_by=clusters)
```

The embedding should separate several broad PBMC accessibility states rather than form one undifferentiated cloud.
Small groups dominated by low-fragment cells indicate that filtering or peak selection should be revisited.

Individual peak IDs are rarely interpretable on their own.
The next section maps peaks to gene scores so known marker genes can be plotted on this UMAP.

## 6. Calculate gene scores

Gene scores summarize accessible chromatin by TF-IDF-normalizing peaks, then summing overlaps with gene bodies and their promoter regions (optional renormalization).
Marker genes are often easier to use for cell-type annotation than individual peaks.

Prepared human and mouse BED files from GENCODE annotations are available in the `scarf_docs` Cytebase catalog (GFF3 primary-chromosome annotations converted with Scarf's `GffReader`, plus a 2 kb promoter offset).
Download them with `annotations`:

```{code-cell} ipython3
annotations = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="annotations", destination="scarf_datasets"
)
```

`add_melded_assay` maps genomic coordinates onto ATAC peaks.
For a given locus, values from all overlapping peaks are melded into one feature.
Peak values are TF-IDF normalized before melding.
The same method can meld motif or enhancer annotations, not only gene bodies.

This example uses the prepared gene-body annotation and peak coordinates stored in `ids`.
BED column requirements, promoter offsets, renormalization, and other coordinate-melding choices are covered in {doc}`annotation`.

```{code-cell} ipython3
ds.add_melded_assay(
    from_assay="ATAC",
    external_bed_fn=f"{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz",
    peaks_col="ids",
    renormalization=False,
    assay_label="GeneScores",
    assay_type="RNA",
)
print(
    "Valid GeneScores features:",
    int(ds.GeneScores.feats.fetch_all("I").sum()),
)
```

```{note}
This peak matrix has no `chrM` coordinates, so the chromosome warning is expected.
Some annotation features also do not overlap any peak.
They remain in `GeneScores` but are marked invalid, while overlapping genes retain their calculated scores.
```

Inspect GeneScores for known PBMC marker genes alongside the retained UMAP artifact.

The marker patterns should occupy coherent accessibility regions.
A uniformly flat score can indicate coordinate-build mismatch or poor overlap between peaks and annotations.

## Common mistakes and limitations

- Downloading raw HDF5 and repeating the wide LSI analysis when the prepared Zarr result is sufficient
- Using RNA normalization or PCA assumptions for ATAC data
- Setting `skip_first=False` on `run_lsi` without checking whether sequencing depth dominates the first LSI component
