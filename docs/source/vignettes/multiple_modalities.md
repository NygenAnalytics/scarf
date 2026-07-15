---
description: CITE-seq multimodal analysis with SNN and WNN integration for RNA and ADT.
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

# Handling datasets with multiple modalities

```{code-cell} ipython3
%load_ext autotime

import scarf
import scarf.plotting as splt
scarf.__version__
```

---
## 1) Fetch and convert data

For this tutorial we will use CITE-Seq data from 10x genomics. This dataset contains two modalities: gene expression and surface protein abundance. Throughout this tutorial we will refer to gene expression modality as `RNA` and surface protein as `ADT`. We start by downloading the data and converting it into Zarr format:

```{code-cell} ipython3
scarf.fetch_dataset(
    'tenx_8K_pbmc_citeseq',
    save_path='scarf_datasets'
)
```

```{code-cell} ipython3
reader = scarf.CrH5Reader(
    'scarf_datasets/tenx_8K_pbmc_citeseq/data.h5'
)
```

We can also quickly check the different kinds of assays present in the file and the number of features from each of them.

```{code-cell} ipython3
reader.assayFeats
```

The **nFeatures** column shows the number of features present in each assay. `CrH5Reader` will automatically pull this information from H5 file and rename the 'Gene Expression' assay to **RNA**. Here it also found another assay: 'Antibody Capture' and named it to **assay2**. We will rename this to **ADT**.

```{code-cell} ipython3
reader.rename_assays({'assay2': 'ADT'})
reader.assayFeats
```

Now the data is converted into Zarr format. Like single assay datasets, all the data is saved under one Zarr file.

```{code-cell} ipython3
writer = scarf.CrToZarr(
    reader,
    zarr_loc='scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr',
    chunk_size=(2000, 1000),
)
writer.dump(batch_size=1000)
```

---
## 2) Create a multimodal DataStore

The next step is to create a Scarf `DataStore` object. This object will be the primary way to interact with the data and all its constituent assays. The first time a Zarr file is loaded, we need to set the default assay. Here we set the 'RNA' assay as the default assay. When a Zarr file is loaded, Scarf checks if some per-cell statistics have been calculated. If not, then **nFeatures** (number of features per cell) and **nCounts** (total sum of feature counts per cell) are calculated. Scarf will also attempt to calculate the percent of mitochondrial and ribosomal content per cell.

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr',
    default_assay='RNA',
    nthreads=4
)
```

We can print out the DataStore object to get an overview of all the assays stored.

```{code-cell} ipython3
ds
```

Feature attribute tables for each of the assays can be accessed like this:

```{code-cell} ipython3
ds.RNA.feats.head()
```

```{code-cell} ipython3
ds.ADT.feats.head()
```

Cell filtering is performed based on the default assay. Here we use the `auto_filter_cells` method of the `DataStore` to filter low quality cells.

```{code-cell} ipython3
ds.auto_filter_cells()
```

---
## 3) Process gene expression modality

Now we process the RNA assay to perform feature selection, create KNN graph, run UMAP reduction and clustering. These steps are same as shown in the basic workflow for scRNA-Seq data.

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=20,
    top_n=1000,
    min_mean=-3,
    max_mean=2,
    max_var=6
)

ds.make_graph(
    feat_key='hvgs',
    k=21,
    dims=15,
    n_centroids=100
)

ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)

ds.run_leiden_clustering(resolution=1)
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
    show=False,
).figure
```

---
## 4) Process protein surface abundance modality

+++

We will now perform similar steps as RNA for the ADT data. Since ADT panels are often custom designed, we will not perform any feature selection step. This particular data contains some control antibodies which we should filter out before downstream analysis. 

```{code-cell} ipython3
ds.ADT.feats.head(n=ds.ADT.feats.N)
```

We can manually filter out the control antibodies by updating **I** to be False for those features. To do so we first extract the names of all the ADT features like below:

```{code-cell} ipython3
adt_names = ds.ADT.feats.to_pandas_dataframe(['names'])['names']
adt_names
```

The ADT features with 'control' in name are designated as control antibodies. You can have your own selection criteria here. The aim here is to create a boolean array that has `True` value for features to be removed.

```{code-cell} ipython3
is_control = adt_names.str.contains('control').values
is_control
```

Now we update `I` to remove the control features. `update_key` method takes a boolean array and disables the features that have `False` value. So we invert the above created array (using `~`) before providing it to `update_key`. The second parameter for `update_key` denotes which feature table boolean column to modify, `I` in this case.

```{code-cell} ipython3
ds.ADT.feats.update_key(~is_control, 'I')
ds.ADT.feats.head(n=ds.ADT.feats.N)
```

Assays named ADT are automatically created as objects of the `ADTassay` class, which uses CLR (centred log ratio) normalization as the default normalization method.

```{code-cell} ipython3
print (ds.ADT)
print (ds.ADT.normMethod.__name__)
```

Now we are ready to create a KNN graph of cells using only ADT data. Here we will use all the features (except those that were filtered out) and that is why we use `I` as value for `feat_key`. It is important to note the value for `from_assay` parameter which has now been set to `ADT`. If no value is provided for `from_assay` then it is automatically set to the default assay. By setting `dims` to 0 we disable dimension reduction.

```{code-cell} ipython3
ds.make_graph(
    from_assay='ADT',
    feat_key='I', 
    k=21,
    dims=0,
    n_centroids=100
)
```

UMAP and clustering can be run on ADT assay by simply setting `from_assay` parameter value to 'ADT':

```{code-cell} ipython3
ds.run_umap(
    from_assay='ADT',
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)

ds.run_leiden_clustering(
    from_assay='ADT',
    resolution=1
)
```

If we now check the cell attribute table, we will find the UMAP coordinates and clusters calculated using `ADT` assay:

```{code-cell} ipython3
ds.cells.head()
```

Visualizing the UMAP and clustering calcualted using `ADT` only:

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='ADT_UMAP',
    color_by='ADT_leiden_cluster',
    show=False,
).figure
```

---
## 5) Cross modality comparison

It is generally of interest to see how different modalities corroborate each other.

The examples below intentionally use the supported `plot_layout` compatibility
method because they combine several layout keys or CLR-normalized ADT features.
The `scarf.plotting` API raises for assay transformations it cannot reproduce
without changing their meaning.

```{code-cell} ipython3
# UMAP on RNA and coloured with clusters calculated on ADT
ds.plot_layout(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by=['ADT_leiden_cluster', 'RNA_leiden_cluster'],
    cmap='tab20',
    width=4,
    height=4, 
    n_columns=2,
    point_size=5,
    legend_onside=False
)
```

We can quantify the overlap of cells between RNA and ADT clusters. The following table has ADT clusters on columns and RNA clusters on rows. This table shows a cross tabulation of cells across the clustering from the two modalities.

```{code-cell} ipython3
import pandas as pd

df = pd.crosstab(
    ds.cells.fetch('RNA_leiden_cluster'),
    ds.cells.fetch('ADT_leiden_cluster')
)
df
```

There are possibly many interesting strategies to analyze this further. One simple way to summarize the above table can be quantify the transcriptomics 'purity' of ADT clusters:

```{code-cell} ipython3
(100 * df.max()/df.sum()).sort_values(ascending=False)
```

Individual ADT expression can be visualized in both UMAPs easily.

```{code-cell} ipython3
ds.plot_layout(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by='CD16_TotalSeqB',
    from_assay='ADT',
    width=4,
    height=4,
    n_columns=2,
    point_size=5
)
```

We can also query gene expression and visualize it on both RNA and ADT UMAPs. Here we query gene FCGR3A which codes for CD16:

```{code-cell} ipython3
ds.plot_layout(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by='FCGR3A',
    from_assay='RNA',
    width=4,
    height=4,
    n_columns=2,
    point_size=5
)
```

---
## 6) Integration of modalities

The KNN graphs created individually for each of the modalities can be merged together to provide an integrated mutimodal view of the data. Scarf takes the latest KNN graphs (continous form edge weight) generated for each of the user chosen modality and merges the edges from each modality. After first round of merging, Scarf performs edge pruning by penalizing those edges more that have lower number of shared nearest neighbors between the connected cells. For each cells edges are pruned until the same number of edges as in individual modalities' KNN graphs are left.

Here we will integrate the *RNA* and *ADT* assays and run UMAP and leiden clustering on the integrated graph.

```{code-cell} ipython3
ds.integrate_assays(
    assays=['RNA', 'ADT'],
    label='RNA+ADT'
)
```

`integrated_graph` parameter in `run_umap` and `run_leiden_clustering` allows running these steps on the integrated graph.

```{code-cell} ipython3
ds.run_umap(
    integrated_graph='RNA+ADT',
    n_epochs=500,
    spread=5,
    min_dist=0.5,
    parallel=True
)

ds.run_leiden_clustering(
    integrated_graph='RNA+ADT',
    resolution=1.75
)
```

Lets visualize the UMAPs created using the integrated manifolds from the two modalities. Here we label the cells based on their modality specific cluster identity as well as integrated manifold cluster identity

```{code-cell} ipython3
ds.plot_layout(
    layout_key=['RNA+ADT_UMAP'],
    color_by=['RNA_leiden_cluster',
              'ADT_leiden_cluster',
              'RNA+ADT_leiden_cluster'],
    cmap='tab20',
    legend_onside=False,
    point_size=5,
    width=4,
    height=4,
    n_columns=3
)
```

```{code-cell} ipython3
ds.cells.columns
```

The UMAP and clustering calculated on the integrated graph are here saved under cell attribute table with prefix *RNA+ADT*

---
(wnn_integration)=
## 7) WNN integration

The default `integrate_assays` method uses shared nearest neighbors (SNN). Scarf also supports weighted nearest neighbors (WNN), which can weight modalities differently. WNN requires exactly two assays:

```{code-cell} ipython3
ds.integrate_assays(
    assays=['RNA', 'ADT'],
    label='RNA+ADT_wnn',
    method='wnn'
)

ds.run_umap(
    integrated_graph='RNA+ADT_wnn',
    n_epochs=500,
    spread=5,
    min_dist=0.5,
    parallel=True
)

ds.run_leiden_clustering(
    integrated_graph='RNA+ADT_wnn',
    resolution=1.75
)
```

```{code-cell} ipython3
ds.plot_layout(
    layout_key=['RNA+ADT_UMAP', 'RNA+ADT_wnn_UMAP'],
    color_by='RNA+ADT_wnn_leiden_cluster',
    cmap='tab20',
    point_size=5,
    width=4,
    height=4,
    n_columns=2
)
```

SNN supports two or more assays; WNN is limited to two. Try WNN when one modality is sparse or weaker than the other.

---
## 8) HTO demultiplexing (hashtag oligos)

Multiplexed experiments with hashtag oligos (HTO) can be demultiplexed with `mark_hto_identities`. The assay should be named `HTO` (or pass `from_assay`). This CITE-seq tutorial dataset contains RNA and ADT only; below is the API pattern for multiplexed Cell Ranger output:

```python
# After loading a DataStore with an HTO assay:
ds.mark_hto_identities(from_assay='HTO', label='Hashtag_identity')
```

Demultiplexing follows a Seurat-style algorithm in `hto_demux`. A dedicated executable vignette will be added when a public HTO dataset is available on the Scarf OSF repository.

---
That is all for this vignette.
