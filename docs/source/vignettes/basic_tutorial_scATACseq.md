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

## Workflow for scATAC-Seq data

If you haven't already then we highly recommend that you please go the {ref}`Workflow for scRNA-Seq analysis <scrna_seq_workflow>` and also go through {ref}`this vignette <data_organization>` describing data organization and design principles of Scarf.

Here we will go through some of the basic steps of analysis of single-cell ATAC-seq (Assay for Transposase-Accessible Chromatin using sequencing) data.

```{code-cell} ipython3
%load_ext autotime

import scarf
scarf.__version__
```

---
### 1) Fetch and convert data

We will use 10x Genomics's singel-cell ATAC-Seq data from peripheral blood mononuclear cells. Like single-cell RNA-Seq, Scarf only needs a count matrix to start the analysis of scATAC-Seq data. We can use `fetch_dataset` to download the data in 10x's HDF5 format.

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='tenx_10K_pbmc-v1_atacseq',
    save_path='scarf_datasets'
)
```

The `CrH5Reader` class provides access to the HDF5 file. We can load the file and quickly check the number of features, and also verify that Scarf identified the assay as an ATAC assay. 

```{code-cell} ipython3
reader = scarf.CrH5Reader(
    'scarf_datasets/tenx_10K_pbmc-v1_atacseq/data.h5'
)
reader.assayFeats
```

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc=f'scarf_datasets/tenx_10K_pbmc-v1_atacseq/data.zarr',
    chunk_size=(1000, 2000)
)
writer.dump(batch_size=1000)
```

---
### 2) Create DataStore and filter cells

+++

We load the Zarr file on using `DataStore` class. The obtained `DataStore` object will be our single point on interaction for rest of this analysis. When loaded, Scarf will automatically calculate the number of cells where each peak is present and number of peaks that are accessible in cell (nFeatures). Scarf will also calculate the total number of fragments/cut sites within each cell.

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/tenx_10K_pbmc-v1_atacseq/data.zarr', 
    nthreads=4
)
```

We will use `auto_filter_cells` method which automatically remove outliers from the data. To identify outliers we generate a normal distribution using sample mean and variance. Using this normal distribution Scarf estimates the values with probability less 0.01 (default value) on both ends of the distribution and flags them for removal.

```{code-cell} ipython3
ds.auto_filter_cells()
```

---
### 3) Feature selection

For scATAC-Seq data, the features are ranked by their [TF-IDF](https://en.wikipedia.org/wiki/Tf-idf) normalized values, summed across all cells. The top n features are marked as `prevalent_peaks` and are used for downstream steps. Here we used top 25000 peaks, which makes for more than quarter of all the peaks which inline with the what has been suggested in other scATAC-Seq analysis protocols.

```{code-cell} ipython3
ds.mark_prevalent_peaks(top_n=25000)
```

```{code-cell} ipython3
ds.ATAC.feats.head()
```

---
### 4) KNN graph creation

For scATAC-Seq datasets, Scarf uses TF-IDF normalization. The normalization is automatically performed during the graph building step. The selected features, marked as `prevalent_peaks` in feature metadata, are used for graph creation. For the dimension reduction step, LSI (latent semantic indexing) is used rather than PCA. The rest of the steps are same as for scRNA-Seq data.

LSI reduction of scATAC-Seq is known to capture the sequencing depth of cells in the first LSI dimension. Hence, by default, the `lsi_skip_first` parameter is True but users can override it.

```{code-cell} ipython3
ds.make_graph(
    feat_key='prevalent_peaks',
    k=21,
    dims=50,
    n_centroids=100,
    lsi_skip_first=True,
)
```

---
### 5) UMAP reduction and clustering


Non-linear dimension reduction using UMAP and tSNE are performed in the same way as for scRNA-Seq data. Because, in Scarf the core UMAP step are run directly on the neighbourhood graph, the scATAC-Seq data is handled similar to any other data.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=500,
    min_dist=0.1, 
    spread=1, 
    parallel=True
)
```

Same goes for clustering as well. The leiden clustering acts on the neighbourhood graph directly.

```{code-cell} ipython3
ds.run_leiden_clustering(resolution=0.6)
```

Results from both UMAP and Leiden clustering are stored in cell attribute table. Here, because the UMAP and leiden clustering columns have 'ATAC' prefixes because they were executed on the default 'ATAC' assay. You can also see that the cells that were filtered out (marked by 'False' value in column 'I') have NaN value for UMAP axes and -1 for clustering

```{code-cell} ipython3
ds.cells.head()
```

We can visualize the UMAP embedding and the clusters of cells on the embedding

```{code-cell} ipython3
ds.plot_layout(
    layout_key='ATAC_UMAP', 
    color_by='ATAC_leiden_cluster'
)
```

Those familiar with PBMC datasets might already be able to identify different cell types in the UMAP plot.

+++

---
### 6) Calculating gene scores

The features in snATAC-Seq in form of peak coordinates are hard to interpret by themselves.
Hence, distilling the open chromatin regions in terms of accessible genes can help in identification of cell types.
`GeneScores` are simply the summation of all the peak fragments that are present within gene bodies and their corresponding promoter region.
Since, marker genes are often better understood than the non-coding regions the `GeneScores` can be used to annotate cell types.

In Scarf, the users can provide a BED file containing gene annotations. This BED should have no header, should be tab separated and must have the columns in following order:
1) chromosome identifier
2) start coordinate
3) end coordinate 
4) Gene ID
5) Gene Name
6) Strand (Optional)

The start/end coordinate can extend through transcription start site (TSS) to include a portion of promoter.

For convenience we have generated such BED files for human and mouse assemblies using the annotation information from GENCODE project. We downloaded the GFF3 format primary chromosome annotations and used Scarf's `GFFReader` to convert the files into BED and add promoter offset of 2KB. These BED files containing gene annotations can be downloaded using `fetch_dataset` command and passing 'annotations' parameter.

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='annotations', 
    save_path='scarf_datasets'
)
```

----

Now we have the annotations and are ready to calculate the 'GeneScores'. The `add_melded_assay` is the `DataStore` method that will be used for this purpose. The `add_melded_assay` method is actually designed to map any arbitrary genomic coordinate information (not just gene annotations) to the ATAC peaks. Here, we use the term 'melding' for the process wherein for a given loci the values from all the overlapping peaks are merged/melded into that feature. The peaks values are TF-IDF normalized before melding.

Now we explain the parameters that are usually passed to the `add_melded_assay`.
- `from_assay`: Name of the assay to be acted on. You can generally skip if you have only one assay. We use this parameter here only for demonstration puspose
- `external_bed_fn`: This is the annotation file. Here we pass the annotation for human GRCh37/hg19 assembly based GENCODE v38 annotations.
- `peaks_col`: Column in `ds.ATAC.feats` with peak coordinates in `chr:start-end` format.
- `renormalization`: By overriding the default value of True to False here, we turned off 'renormalization' step. The renormalization step makes sure that all the feature values for each cells in the melded assay sum up to the same value. Here we turned this off because we will have 'GeneScores' as an 'RNAassay' which uses library size normalization that has the same effect as 'renormalization'.
- `assay_label`: The label/name of the melded/output assay. Because we are using gene bodies as our input, 'GeneScores' is sensible choice.
- `assay_type`: Here we set the type of assay as 'RNA' which means that the new melded assay will be treated as if it was an scRNA-Seq assay. Alternatively, we could have set it as a generic 'Assay'

```{code-cell} ipython3
ds.add_melded_assay(
    from_assay='ATAC',
    external_bed_fn='scarf_datasets/annotations/human_GRCh37_gencode_v38_gene_body.bed.gz',
    peaks_col='ids',
    renormalization=False,
    assay_label='GeneScores',
    assay_type='RNA'
)
```

---

We can now print out the DataStore and see that the 'GeneScores' assay has indeed been added. The 'add_melded_assay' also printed a useful bit of information that almost half of features(genes bodies) did not overlap with even a single peak. The melded assay anyway contains all the genes but sets these 'empty' genes as invalid.

```{code-cell} ipython3
ds
```

Let's now visualize some 'GeneScores' for some of the known marker genes for PBMCs on the UMAP plot calculated on the ATAC assay

```{code-cell} ipython3
ds.plot_layout(
    layout_key='ATAC_UMAP', from_assay='GeneScores', 
    color_by=['CD3D', 'MS4A1', 'LEF1', 'NKG7', 'TREM1', 'LYZ'],
    clip_fraction=0.01, 
    n_columns=3,
    width=3,
    height=3,
    point_size=5,
    scatter_kwargs={'lw': 0.01},
)
```

The same melding approach maps any coordinate bed file onto peaks, including motif or enhancer annotations via `GffReader` and `coordinate_melding`. For cross-dataset integration using GeneScores, see {ref}`integration methods guide <integration_guide>` and {ref}`data projection <data_projection>`.

+++

---
That is all for this vignette.
