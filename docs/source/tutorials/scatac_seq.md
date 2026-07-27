---
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

# scATAC-seq analysis

Analyze a peak-count matrix by selecting prevalent peaks, building an LSI-based graph, and
clustering accessible chromatin profiles. See {doc}`scrna_seq` for the RNA workflow and
{doc}`data_organization` for store structure.

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

scarf.set_verbosity('WARNING')
```

## Guided steps

### 1. Fetch and convert data

We will use 10x Genomics single-cell ATAC-seq data from peripheral blood mononuclear
cells. Like single-cell RNA-seq, Scarf only needs a count matrix to start the analysis.
Use the `scarf_docs` Cytebase client to download the data in 10x HDF5 format.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='tenx_10K_pbmc-v1_atacseq',
    destination='scarf_datasets'
)
```

The `CrH5Reader` class provides access to the HDF5 file. We can load the file and quickly check the number of features, and also verify that Scarf identified the assay as an ATAC assay. 

```{code-cell} ipython3
reader = scarf.CrH5Reader(f'{dataset}/data.h5')
reader.assayFeats
```

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc=f'{dataset}/data.zarr',
)
writer.dump(batch_size=1000)
```

### 2. Create a DataStore and filter cells

+++

Load the Zarr store with `DataStore`, which is the main interface for the rest of the
analysis. On first load, Scarf calculates the number of cells in which each peak is present,
the number of accessible peaks per cell (`nFeatures`), and the total fragments or cut sites
per cell.

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4
)
```

Inspect fragment and peak-count distributions before filtering. Thresholds are
dataset-specific.

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

`auto_filter_cells` models each selected QC column with a normal distribution centered on its
median and using its standard deviation. The default 0.01 and 0.99 quantiles define the lower
and upper bounds. Cells outside those bounds are marked inactive in `I`; they are not deleted.

```{code-cell} ipython3
ds.auto_filter_cells(show_qc_plots=True)
```

### 3. Select features

For scATAC-seq data, features are ranked by TF-IDF-normalized values summed across cells.
The top features are marked as `prevalent_peaks` for downstream steps. Here we retain 25,000
peaks, slightly more than one quarter of the available peaks.

```{code-cell} ipython3
ds.mark_prevalent_peaks(top_n=25000)
```

```{code-cell} ipython3
ds.ATAC.feats.head()
```

### 4. Create a KNN graph

For scATAC-Seq datasets, Scarf uses TF-IDF normalization during `run_normalization`.
The selected features, marked as `prevalent_peaks` in feature metadata, are used for
graph creation. Dimension reduction uses LSI rather than PCA. The remaining ANN,
neighbor, and connectivity steps match the RNA atomic chain.

LSI reduction of scATAC-Seq is known to capture the sequencing depth of cells in the
first LSI dimension. `run_lsi` skips that component by default (`skip_first=True`).

```{code-cell} ipython3
ds.run_normalization(feat_key='prevalent_peaks')
ds.run_lsi(dims=50, skip_first=True)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=21)
ds.build_connectivity_map()

ds.load_graph()
```

`load_graph` returns the graph as a sparse cell-by-cell matrix, which confirms it covers the
active cells.


### 5. Run UMAP and clustering


UMAP and tSNE use the same neighbourhood-graph path as scRNA-seq, so the ATAC workflow
looks the same after the graph is built.

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
```

UMAP and Leiden results are stored in the cell attribute table with an `ATAC` prefix because
they were run on the default ATAC assay. Filtered cells (`I` is False) have NaN UMAP
coordinates and cluster id `-1`.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ['ATAC_UMAP1', 'ATAC_UMAP2', 'ATAC_leiden_cluster']
).head()
```

Plot the UMAP embedding colored by Leiden clusters:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ATAC_UMAP',
    color_by='ATAC_leiden_cluster',
)
```

Individual peak IDs are rarely interpretable on their own. The next section maps peaks to
gene scores so known marker genes can be plotted on this UMAP.

### 6. Calculate gene scores

Gene scores summarize accessible chromatin by summing peak fragments that overlap gene
bodies and their promoter regions. Marker genes are often easier to use for cell-type
annotation than individual peaks.

Provide a BED file of gene annotations. The file should have no header, be tab-separated, and
use this column order:
1) chromosome identifier
2) start coordinate
3) end coordinate 
4) Gene ID
5) Gene Name
6) Strand (Optional)

The start/end coordinate can extend through transcription start site (TSS) to include a portion of promoter.

Prepared human and mouse BED files from GENCODE annotations are available in the
`scarf_docs` Cytebase catalog (GFF3 primary-chromosome annotations converted with Scarf's
`GffReader`, plus a 2 kb promoter offset). Download them with `annotations`:

```{code-cell} ipython3
annotations = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='annotations',
    destination='scarf_datasets'
)
```

----

`add_melded_assay` maps genomic coordinates onto ATAC peaks. For a given locus, values from
all overlapping peaks are melded into one feature. Peak values are TF-IDF normalized before
melding. The same method can meld motif or enhancer annotations, not only gene bodies.

Common parameters:
- `from_assay`: Assay to act on. Optional when only one assay is present; shown here for clarity.
- `external_bed_fn`: Annotation BED. Here, human GRCh37/hg19 GENCODE v38 gene bodies.
- `peaks_col`: Column in `ds.ATAC.feats` with peak coordinates in `chr:start-end` format.
- `renormalization`: Set to False here. Renormalization would force each cell's melded feature
  values to the same sum; GeneScores is created as an `RNAassay`, which already applies
  library-size normalization.
- `assay_label`: Name of the output assay (`GeneScores`).
- `assay_type`: `'RNA'` treats the melded assay like scRNA-seq. Use `'Assay'` for a generic assay.

```{code-cell} ipython3
ds.add_melded_assay(
    from_assay='ATAC',
    external_bed_fn=f'{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz',
    peaks_col='ids',
    renormalization=False,
    assay_label='GeneScores',
    assay_type='RNA'
)
```

```{note}
It is expected that some annotation features do not overlap any peak. They
remain in `GeneScores` but are marked invalid, while overlapping genes retain
their calculated scores.
```

---

Print the DataStore to confirm that `GeneScores` was added. `add_melded_assay` also reports
how many gene bodies did not overlap any peak; those genes remain in the assay but are marked
invalid.

```{code-cell} ipython3
ds
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

The same melding approach maps any coordinate bed file onto peaks, including motif or enhancer annotations via `GffReader` and `coordinate_melding`. For cross-dataset integration using GeneScores, see {ref}`integration methods guide <integration_guide>` and {ref}`data projection <data_projection>`.

+++

## Common mistakes and limitations

- Requesting `zarr=True` for `tenx_10K_pbmc-v1_atacseq`, which has no prepared Zarr store
- Using RNA normalization or PCA assumptions for ATAC data
- Setting `skip_first=False` on `run_lsi` without checking whether sequencing depth dominates the first LSI component

## Saved results

The converted Zarr store retains the ATAC counts, cell metadata, graph, UMAP coordinates, Leiden
labels, and the optional `GeneScores` assay.

## Further reading

- [Cell Ranger ATAC](https://www.10xgenomics.com/support/software/cell-ranger-atac/latest)

## Next steps

- {doc}`data_organization`
- {doc}`plotting`
- {doc}`data_integration`

