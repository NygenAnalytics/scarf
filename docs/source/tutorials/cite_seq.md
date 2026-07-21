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

# CITE-seq analysis

CITE-seq combines RNA counts with antibody-derived tags (ADT) measured in the same cells.
This chapter processes each assay, then combines their cell graphs with SNN or WNN integration.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Python 3.12 or newer

## What you will learn

- Analyze RNA and ADT assays independently
- Compare modality-specific embeddings and clusters
- Integrate assay graphs with SNN and WNN

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')
scarf.__version__
```

## Guided steps

### 1. Fetch the prepared data

This dataset contains gene expression and surface protein abundance. The prepared Zarr store
contains these assays as `RNA` and `ADT`.

```{code-cell} ipython3
scarf.fetch_dataset(
    'tenx_8K_pbmc_citeseq',
    save_path='scarf_datasets',
    as_zarr=True,
)
```

### 2. Create a multimodal DataStore

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

### 3. Process the RNA assay

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
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

### 4. Process the ADT assay

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

Visualizing the UMAP and clustering calculated using `ADT` only:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ADT_UMAP',
    color_by='ADT_leiden_cluster',
)
```

### 5. Compare modalities

It is generally of interest to see how different modalities corroborate each other.

`ds.plots.embedding` can compare several layouts in one figure and uses the selected
assay's native normalization for feature values.

```{code-cell} ipython3
# UMAP on RNA and coloured with clusters calculated on ADT
ds.plots.embedding(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by=['ADT_leiden_cluster', 'RNA_leiden_cluster'],
    n_columns=2,
    point_size=5,
    legend_loc='on_data',
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
ds.plots.embedding(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by='CD16_TotalSeqB',
    from_assay='ADT',
    n_columns=2,
    point_size=5,
)
```

We can also query gene expression and visualize it on both RNA and ADT UMAPs. Here we query gene FCGR3A which codes for CD16:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by='FCGR3A',
    from_assay='RNA',
    n_columns=2,
    point_size=5,
)
```

### 6. Integrate assays with SNN

The KNN graphs created individually for each of the modalities can be merged together to provide an integrated mutimodal view of the data. Scarf takes the latest KNN graphs (continous form edge weight) generated for each of the user chosen modality and merges the edges from each modality. After first round of merging, Scarf performs edge pruning by penalizing those edges more that have lower number of shared nearest neighbors between the connected cells. For each cells edges are pruned until the same number of edges as in individual modalities' KNN graphs are left.

Here we will integrate the *RNA* and *ADT* assays and run UMAP and leiden clustering on the integrated graph.

```{code-cell} ipython3
ds.integrate_assays(
    assays=['RNA', 'ADT'],
    label='RNA+ADT',
    method='snn',
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
ds.plots.embedding(
    layout_key='RNA+ADT_UMAP',
    color_by=[
        'RNA_leiden_cluster',
        'ADT_leiden_cluster',
        'RNA+ADT_leiden_cluster',
    ],
    legend_loc='on_data',
    n_columns=3,
)
```

```{code-cell} ipython3
ds.cells.columns
```

The UMAP and clustering calculated on the integrated graph are here saved under cell attribute table with prefix *RNA+ADT*

(wnn_integration)=
### 7. Integrate assays with WNN

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

Compare each integration method on its matching layout and cluster labels:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA+ADT_UMAP',
    color_by='RNA+ADT_leiden_cluster',
    legend_loc='on_data',
)
ds.plots.embedding(
    layout_key='RNA+ADT_wnn_UMAP',
    color_by='RNA+ADT_wnn_leiden_cluster',
    legend_loc='on_data',
)
```

SNN supports two or more assays; WNN is limited to two. Try WNN when one modality is sparse or weaker than the other.

## Common mistakes

- Building assay graphs from different cell subsets before integration
- Using WNN with anything other than two assays
- Treating RNA and ADT clusters as interchangeable without examining cross-modality agreement

## Saved results

Each assay stores its own graph, UMAP, and cluster columns. SNN and WNN graphs are stored under
their `label` values and generate corresponding UMAP and Leiden columns.

## Next steps

- {doc}`choosing_integration_methods`
- {doc}`plotting`
- {doc}`data_organization`
