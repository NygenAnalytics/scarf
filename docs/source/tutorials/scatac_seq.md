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
- A 10x scATAC-seq HDF5 count matrix

## What you will learn

- Convert a 10x scATAC-seq HDF5 matrix to Zarr
- Build an LSI-based graph and cluster cells
- Create a gene-score assay from peak annotations

## Dataset

```{code-cell} ipython3
import scarf

scarf.configure_output(level='WARNING', progress=True)
```

## 1. Import the peak-count matrix

We will use 10x Genomics single-cell ATAC-seq data from peripheral blood mononuclear cells.
Like single-cell RNA-seq, Scarf only needs a count matrix to start the analysis.
Use the `scarf_docs` Cytebase client to download the data in 10x HDF5 format.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='tenx_10K_pbmc-v1_atacseq',
    destination='scarf_datasets'
)
```

The `CrH5Reader` class provides access to the HDF5 file.
We can load the file and quickly check the number of features, and also verify that Scarf identified the assay as an ATAC assay.

```{code-cell} ipython3
reader = scarf.CrH5Reader(f'{dataset}/data.h5')
reader.assayFeats
```

This peak matrix is wide. `mem_budget="8G"` leaves room for one source row and one destination row band; the default 4 GiB budget is not enough for this file.

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc=f'{dataset}/data.zarr',
    mem_budget="8G",
)
writer.dump()
```

## 2. Filter cells

Load the Zarr store with `DataStore`, which is the main interface for the rest of the analysis.
On first load, Scarf streams the count matrix once to compute initialization statistics: accessible peaks per cell (`nFeatures`), total fragments or cut sites per cell (`nCounts`), and per-peak detection counts used by explicit selection producers.

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4
)
```

Inspect fragment and peak-count distributions before filtering.
Thresholds are dataset-specific.

```{code-cell} ipython3
qc_cols = [
    c for c in ('ATAC_nCounts', 'ATAC_nFeatures')
    if c in ds.cells.columns
]
ds.plots.distribution(
    keys=qc_cols,
    kind='violin',
    max_points=2000,
)
```

`auto_filter_cells` applies Scarf's global automatic bounds and marks outlying cells inactive in `I`; it does not delete them.
The {doc}`quality_control` guide covers manual thresholds, sample-aware filtering, and the ATAC metrics Scarf does and does not provide.

```{code-cell} ipython3
n_before = int(ds.cells.fetch_all('I').sum())
ds.auto_filter_cells(show_qc_plots=True)
n_after = int(ds.cells.fetch_all('I').sum())
print(f'Active cells before filter: {n_before}')
print(f'Active cells after filter: {n_after}')
```

## 3. Select prevalent peaks

For scATAC-seq data, features are ranked by TF-IDF-normalized values summed across cells.
The top features are marked as `prevalent_peaks` for downstream steps.
Here we retain 25,000 peaks, slightly more than one quarter of the available peaks.

```{code-cell} ipython3
peak_features = ds.mark_prevalent_peaks(top_n=25000)
print('Selected peaks:', int(ds.ATAC.feats.fetch_all('prevalent_peaks').sum()))
```

The retained peaks should be present across enough cells to support stable neighbour comparisons without collapsing the assay onto only the most common open regions.

## 4. Build an LSI neighbourhood graph

For scATAC-Seq datasets, Scarf uses TF-IDF normalization during `run_normalization`.
The selected features, marked as `prevalent_peaks` in feature metadata, are used for graph construction.
Dimension reduction uses LSI rather than PCA.
The remaining index, neighbour, and connectivity steps match the RNA graph workflow.

LSI reduction of scATAC-Seq is known to capture the sequencing depth of cells in the first LSI dimension.
`run_lsi` skips that component by default (`skip_first=True`).

```{code-cell} ipython3
ds.run_normalization(features=peak_features)
ds.run_lsi(dims=50, skip_first=True)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=21)
ds.build_connectivity_map()

ds.load_graph()
```

`load_graph` returns the graph as a sparse cell-by-cell matrix, which confirms it covers the active cells.

TF-IDF, LSI choices, and graph tuning are covered in {doc}`graph_construction` and {doc}`dimensionality_reduction`.

## 5. Run UMAP and clustering


UMAP and tSNE use the same neighbourhood-graph path as scRNA-seq, so the ATAC workflow looks the same after the graph is built.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=500,
    min_dist=0.1, 
    spread=1, 
    parallel=True
)
```

Leiden clustering also acts on the neighbourhood graph directly.

```{code-cell} ipython3
ds.run_leiden_clustering(resolution=0.6)
ds.cells.to_pandas_dataframe(
    columns=['ATAC_leiden_cluster'],
    key='I'
)['ATAC_leiden_cluster'].value_counts().sort_index()
```

Cluster sizes are worth a look before plotting: a resolution that is too high splits one accessibility state into several small clusters.

UMAP and Leiden results are stored in the cell attribute table with an `ATAC` prefix because they were run on the default ATAC assay.
Filtered cells (`I` is False) have NaN UMAP coordinates and cluster id `-1`.

Plot the UMAP embedding colored by Leiden clusters:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ATAC_UMAP',
    color_by='ATAC_leiden_cluster',
)
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
    name='annotations',
    destination='scarf_datasets'
)
```

----

`add_melded_assay` maps genomic coordinates onto ATAC peaks.
For a given locus, values from all overlapping peaks are melded into one feature.
Peak values are TF-IDF normalized before melding.
The same method can meld motif or enhancer annotations, not only gene bodies.

This example uses the prepared gene-body annotation and peak coordinates stored in `ids`.
BED column requirements, promoter offsets, renormalization, and other coordinate-melding choices are covered in {doc}`annotation`.

```{code-cell} ipython3
ds.add_melded_assay(
    from_assay='ATAC',
    external_bed_fn=f'{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz',
    peaks_col='ids',
    renormalization=False,
    assay_label='GeneScores',
    assay_type='RNA'
)
print(
    'Valid GeneScores features:',
    int(ds.GeneScores.feats.fetch_all('I').sum()),
)
```

```{note}
This peak matrix has no `chrM` coordinates, so the chromosome warning is expected.
Some annotation features also do not overlap any peak.
They remain in `GeneScores` but are marked invalid, while overlapping genes retain their calculated scores.
```

Plot GeneScores for known PBMC marker genes on the ATAC UMAP:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ATAC_UMAP',
    from_assay='GeneScores',
    color_by=['CD3D', 'MS4A1', 'LEF1', 'NKG7', 'TREM1', 'LYZ'],
    clip_fraction=0.01,
    n_columns=3,
    sort_values=True,
)
```

The marker patterns should occupy coherent accessibility regions.
A uniformly flat score can indicate coordinate-build mismatch or poor overlap between peaks and annotations.

## Common mistakes and limitations

- Assuming `tenx_10K_pbmc-v1_atacseq` has no prepared Zarr and skipping `zarr=True`
- Using RNA normalization or PCA assumptions for ATAC data
- Setting `skip_first=False` on `run_lsi` without checking whether sequencing depth dominates the first LSI component
