---
jupytext:
  cell_metadata_filter: -all
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

(scrna_seq_workflow)=

+++

# Scarf's basic workflow for scRNA-Seq

This workflow is meant to familiarize users with the Scarf API and how data is internally handled in Scarf. For a minimal pipeline see {ref}`Quick start <quickstart>`. For batch correction on merged data see {ref}`integration methods guide <integration_guide>`.

```{code-cell} ipython3
%load_ext autotime

import scarf
import scarf.plotting as splt
scarf.__version__
```

Download the count matrix from 10x's website using the `fetch_dataset` function. This is a convenience function that stores URLs of datasets that can be downloaded. The `save_path` parameter allows the data to be saved to a location of choice.

```{code-cell} ipython3
scarf.fetch_dataset(
    'tenx_5K_pbmc_rnaseq',
    save_path='scarf_datasets'
)
```

---
## 1) Format conversion

The first step of the analysis workflow is to convert the file into the Zarr format that is supported by Scarf. We read in the data using `CrH5Reader` (stands for cellranger H5 reader). The reader object allows quick investigation of the file before the format is converted.

```{code-cell} ipython3
reader = scarf.CrH5Reader('scarf_datasets/tenx_5K_pbmc_rnaseq/data.h5')
```

We can quickly check the number of cells and features (genes as well as ADT features in this case) present in the file.

```{code-cell} ipython3
reader.nCells, reader.nFeatures
```

Next we convert the data to the Zarr format that will later on be used by Scarf. For this we use Scarf's `CrToZarr` class. This class will first quickly ascertain the type of data to be written and then create a Zarr format file for the data to be written into. `CrToZarr` takes two mandatory arguments. The first is the cellranger reader, and the other is the name of the output file.

+++

<div class="alert alert-block alert-info">
NOTE: When we say zarr file, we actually mean zarr directory  because, unlike HDF5, Zarr hierarchy is organized as a directory structure.
</div>

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc='scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    chunk_size=(2000, 1000)
)
writer.dump(batch_size=1000)
```

The next step is to create a Scarf `DataStore` object. This object will be the primary way to interact with the data and all its constituent assays. When a Zarr file is loaded, Scarf checks if some per-cell statistics have been calculated. If not, then **nFeatures** (number of features per cell) and **nCounts** (total sum of feature counts per cell) are calculated. Scarf will also attempt to calculate the percent of mitochondrial and ribosomal content per cell. When we create a DataStore instance, we can decide to filter out low abundance genes with parameter `min_features_per_cell`. For example the value of 10 for `min_features_per_cell` below means that those genes that are present in less than 10 cells will be filtered out.

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr',
    nthreads=4,
    min_features_per_cell=10
)
```

---
## 2) Cell filtering

We can visualize the per-cell statistics in [violin plots](https://datavizcatalogue.com/methods/violin_plot.html) before we start filtering cells out.

```{code-cell} ipython3
ds.plot_cells_dists()
```

We can filter cells based on these cell attributes by providing upper and lower threshold values.

```{code-cell} ipython3
ds.filter_cells(
    attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0]
)
```

<div class="alert alert-block alert-info">
NOTE: One of the design philosophies in  Scarf is never to entirely delete any data. Hence, when cells are filtered out they are simply marked as inactive in the `I` column of cell metadata as False. 
    
All functions in the DataStore class, take a parameter called `cell_key`. The default value of this parameter is always `I` which means that the `I` column is always automatically looked up when deciding which cells to use. The added benefit of this approach is that a user can always switch the context of which cells by creating a new boolean column and providing the name of that column to `cell_key`. Read more about this in the 'Data organization' vignette.
</div>

+++

Now we visualize the attributes again after filtering the values. 

*Note: the 'I' value given as the `cell_key` attribute signifies the column of the table that is set to `False` for cells that were filtered out or `True` for cells that are kept.*

```{code-cell} ipython3
ds.plot_cells_dists(cell_key='I', color='coral')
```

The data stored under the 'cellData' level can easily be accessed using the `cells` attribute of the `DataStore` object.

```{code-cell} ipython3
ds.cells.head()
```

<div class="alert alert-block alert-info">
NOTE: We strongly discourage directly adding or removing the data from this table as Scarf will not be able to synchronize the changes to the disk. Instead use the methods of the <code>cells</code> attribute. Please refer to the <code>insert</code>, <code>fetch</code>, <code>fetch_all</code>, <code>drop</code> and <code>update_key</code> methods.
</div>

+++

One can now print the DataStore instance to get an overview of the data. The number of cells and genes post filtering are also indicated. The numbers in brackets indicate the total number of cells (5025 in this case) and total number of features in the RNA assay (33538 in this case).

```{code-cell} ipython3
ds
```

---
## 3) Feature selection

Similar to the cell table, Scarf also saves the feature level data that can be accessed as below:

```{code-cell} ipython3
ds.RNA.feats.head()
```

The feature selection step is performed on the normalized data. The default normalization method for `RNAassay`-type data is library-size normalization, wherein the count values are divided by the sum of total values for a cell. These values are then multiplied by a scalar factor. The default value of this scalar factor is 1000. However, if the total counts in a cell are less than this value, then on multiplication with this scalar factor the values will be 'scaled up' (which is not a desired behaviour). In the filtering step above, we set the `low` threshold for `RNA_nCounts` at 1000, and hence it is safe to use 1000 as a scalar factor. The scalar factor can be set by modifying the `sf` attribute of the assay. Let's print the default value of `sf`

```{code-cell} ipython3
ds.RNA.sf
```

Now the next step is to identify the highly variable genes in the dataset (for the RNA assay). This can be done using the `mark_hvgs` method of the assay. The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly variable. 

The variance is corrected by first dividing genes into bins based on their mean expression values. Genes with minimum variance is selected from each bin and a Lowess curve is fitted to the mean-variance trend of these genes. `mark_hvgs` will by default run on the default assay.

A plot is produced, that for each gene shows the corrected variance on the y-axis and the non-zero mean (means from cells where the gene had a non-zero value) on the x-axis. The genes are colored in two gradients which indicate the number of cells where the gene was expressed. The colors are yellow to dark red for HVGs, and blue to green for non-HVGs.

The `mark_hvgs` function has a parameter `cell_key` that dictates which cells to use to identify the HVGs. The default value of this parameter is `I`, which means it will use all the cells that were not filtered out.

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=20,
    top_n=500,
    min_mean=-3,
    max_mean=2,
    max_var=6
)
```

As a result of running `mark_hvgs`, the feature table now has an extra column **I\_\_hvgs** which contains a `True` value for genes marked HVGs. The naming rule in Scarf dictates that cells used to identify HVGs are prepended to the column name (with a double underscore delimiter). Since we did not provide any `cell_key` parameter the default value was used, i.e. the filtered cells. This resulted in **I** becoming the prefix.

```{code-cell} ipython3
ds.RNA.feats.head()
```

## 4) Graph creation

Creating a neighbourhood graph of cells is the most critical step in any Scarf workflow. This step internally involves multiple substeps: 

- data normalization for selected features
- linear dimensionality reduction using PCA
- creating an approximate nearest neighbour graph index (using the HNSWlib library)
- querying cells against this index to identify nearest neighbours for each cell
- edge weight computation using the `compute_membership_strengths` function from the UMAP package
- fitting MiniBatch Kmeans (The kmeans centers are used later, for UMAP initialization)

The `make_graph` method is responsible for graph construction. Its method takes a mandatory parameter: `feat_key`. This should be a column in the feature metadata table that indicates which genes to use to create the graph. Since we have already identified the `hvgs` in the step above, we use those genes. Note that we do not need to write *I\_\_hvgs* but just *hvgs* as the value of the parameter. We also supply values for two very important parameters here: `k` (number of nearest neighbours to be queried for each cell) and `dims` (number of PCA dimensions to use for graph construction). `n_centroids` parameter controls number of clusters to create for the data using the Kmeans algorithm. We perform a more accurate clustering of data in the later steps.

```{code-cell} ipython3
ds.make_graph(
    feat_key='hvgs',
    k=11,
    dims=15,
    n_centroids=100
)
```

---
## 5) Low dimensional embedding and clustering

Next we run UMAP on the graph calculated above. Here we will not provide which cell key or feature key to be used, because we want the UMAP to run on all the cells that were not filtered out and with the feature key used to calculate the latest graph. We can provide the parameter values for the UMAP algorithm here.

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
```

Optional: enable density-preserving UMAP (densMAP) when local density structure matters:

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    use_density_map=True,
    label='dUMAP'
)
```

The UMAP results are saved in the cell metadata table as seen below in columns: **RNA_UMAP1** and **RNA_UMAP2**

```{code-cell} ipython3
ds.cells.head()
```

`scarf.plotting.embedding` creates a [scatter plot](https://datavizcatalogue.com/methods/scatterplot.html) from a layout stored in cell metadata. Here we plot the UMAP coordinates of cells that passed filtering.

```{code-cell} ipython3
splt.embedding(ds, layout_key='RNA_UMAP', show=False).figure
```

The embedding can be colored by any cell-metadata column. Next, we visualize the total counts in each cell.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_nCounts',
    color_scale=splt.ColorScale(cmap='coolwarm'),
    show=False,
).figure
```

There has been a lot of discussion over the choice of non-linear dimensionality reduction for single-cell data. tSNE was initially considered an excellent solution, but has gradually lost out to UMAP because the magnitude of relations between the clusters cannot easily be discerned in a tSNE plot. Scarf contains an implementation of tSNE that runs directly on the graph structure of cells. So, essentially the same data that was used to create the UMAP and clustering is used.

```{code-cell} ipython3
ds.run_tsne(
    alpha=10,
    box_h=1,
    early_iter=250,
    max_iter=500,
    parallel=True
)
```

<div class="alert alert-block alert-info">
NOTE: The tSNE implementation is currently not supported on Windows.
</div>

```{code-cell} ipython3
# If run_tsne is unavailable, the tSNE columns will not exist.
try:
    splt.embedding(ds, layout_key='RNA_tSNE', show=False).figure
except KeyError:
    print ("'RNA_tSNE1' not found in MetaData")
```

---
## 6) Cell clustering

+++

Identifying clusters of cells is one of the central tenets of single cell approaches. Scarf includes two graph clustering methods and any (or even both) can be used on the dataset. The methods start with the same graph as the UMAP algorithm above to minimize the disparity between the UMAP and clustering results. The two clustering methods are:

- **Paris**: This is the default clustering algorithm.
- **Leiden**: Leiden is a widely used graph clustering algorithm in single-cell genomics.

Paris is the default algorithm in Scarf due to its ability to highlight cluster relationships. [Paris](https://github.com/tbonald/paris) is a hierarchical graph clustering algorithm that is based on node pair sampling. Paris creates a dendrogram of cells which can then be cut to obtain desired number of clusters. The advantage of using Paris, especially in the larger datasets, is that once the dendrogram has been created one can change the desired number of clusters with minimal computation overhead.

```{code-cell} ipython3
# We start with Leiden clustering
ds.run_leiden_clustering(resolution=0.5)
```

We can visualize the results by coloring the UMAP with Leiden cluster identity.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
    show=False,
).figure
```

The results of the clustering algorithm are saved in the cell metadata table. In this case, they have been saved under the column name **RNA_leiden_cluster**.

```{code-cell} ipython3
ds.cells.head()
```

We can export the Leiden cluster information into a pandas dataframe. Setting `key` to `I` makes sure that we do not include the data for cells that were filtered out.

```{code-cell} ipython3
leiden_clusters = ds.cells.to_pandas_dataframe(
    columns=['RNA_leiden_cluster'],
    key='I'
)
leiden_clusters.head()
```

Now we run Paris clustering algorithm using `run_clustering` method. Paris clustering requires only one parameter: `n_clusters`, which determines the number of clusters to create. Here we set the number of clusters same as that were obtained using Leiden clustering.

```{code-cell} ipython3
ds.run_clustering(n_clusters=leiden_clusters.nunique()[0])
```

Visualizing Paris clusters

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cluster',
    show=False,
).figure
```

Discerning similarity between clusters can be difficult from visual inspection alone, especially for tSNE plots. `plot_cluster_tree` function plots the relationship between clusters as a binary tree. This tree is simply a condensation of the dendrogram obtained using Paris clustering.

```{code-cell} ipython3
splt.cluster_tree(ds, cluster_key='RNA_cluster', width=1)
```

The tree is free form (i.e the position of clusters doesn't convey any meaning) but allows inspection of cluster similarity based on branching pattern. The sizes of clusters indicate the number of cells present in each cluster. The tree starts from the root node (black dot with no incoming edges). 

+++

---
## 7) Marker gene identification

Now we can identify the genes that are differentially expressed between the clusters using the `run_marker_search` method. The method to identify the differentially expressed genes in Scarf is optimized to obtain quick results. We have not compared the sensitivity of our method to other differential expression-detecting methods. We expect specialized methods to be more sensitive and accurate to varying degrees. Our method is designed to quickly obtain key marker genes for populations from a large dataset. For each gene individually, following steps are carried out:

- Expression values are converted to ranks (dense format) across cells.
- A mean of ranks is calculated for each group of cells.
- The mean value for each group is divided by the sum of mean values to obtain the 'specificity score'.
- The gene is saved as a marker gene if it's specificity score is higher than a given threshold.

This method does not perform any statistical test of significance and uses 'specificity score' as a measure of importance of each gene for a cluster.

```{code-cell} ipython3
ds.run_marker_search(
    group_key='RNA_cluster',
    gene_batch_size=100
)
```

Using the `plot_marker_heatmap` method, we can also plot a heatmap with the top marker genes from each cluster. The method will calculate the mean expression value for each gene from each cluster.

```{code-cell} ipython3
splt.marker_heatmap(
    ds,
    group_key='RNA_cluster',
    topn=5,
    figsize=(5, 9)
)
```

The markers list for specific clusters can be obtained using `get_markers`. The numbers on the left index (in bold) are the gene/feature index as present in the feature attribute table (`ds.RNA.feats`). The table below is sorted (descending) by scores and this means that more specific markers are placed higher on the list.

```{code-cell} ipython3
df = ds.get_markers(
    group_key='RNA_cluster',
    group_id='1',
    min_score=-1,
    min_frac_exp=-1
)
df
```

We can directly visualize the expression values for a gene of interest. It is usually a good idea to visually confirm the gene expression pattern across the cells at least this way.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='CD14',
    sort_values=True,
    show=False,
).figure
```

---
That is all for this vignette.
