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

## Estimating pseudotime ordering and expression dynamics

```{code-cell} ipython3
%load_ext autotime

import scarf

scarf.__version__
```

---
### 1) Fetch pre-analyzed data

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors. 

We have stored this data on Scarf's online repository for quick access. We processed the data to identify the highly variable genes (top 2000) and create a neighbourhood graph of cells. A UMAP embedding was calculated for the cells. 

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    save_path='./scarf_datasets',
    as_zarr=True,
)
```

```{code-cell} ipython3
ds = scarf.DataStore(
    f"scarf_datasets/bastidas-ponce_4K_pancreas-d15_rnaseq/data.zarr",
    nthreads=4, 
    default_assay='RNA'
)
```

```{code-cell} ipython3
ds
```

```{code-cell} ipython3
ds.plot_layout(
    layout_key='RNA_UMAP',
    color_by=['RNA_cluster', 'clusters'],
    width=4,
    height=4, 
    legend_onside=False,
    cmap='tab20'
)
```

---
### 2) Estimate pseudotime ordering

In Scarf we use a memory efficient implementation of [PBA algorithm](https://github.com/AllonKleinLab/PBA) ([Weinreb et al. 2018, PNAS](https://www.pnas.org/content/115/10/E2467)) to estimate a pseudotime ordering of cells. The function `run_pseudotime_ordering` can be run on any Assay for which we have calculated a neighbourhood graph. The the pseudotime is estimated in a supervised manner and hence, the user needs to provide the source (stem/progenitor/precursor cells) and sink (differentiated cell states) cell clusters/groups. 

```{code-cell} ipython3
ds.run_pseudotime_scoring(
    source_sink_key="RNA_cluster",    # Column that contains cluster information 
    sources=[1],                      # Source clusters
    sinks=[3],                        # Sink clusters
)
```

By default, the calculated pseudotime values are saved under the cell attribute column **'RNA_pseudotime'**, where 'RNA' can be replaced by whatwever the name of the given assay is. Let's visualize these values on UMAP plot. The lighter color cells represent beginning of the pseudotime

```{code-cell} ipython3
ds.plot_layout(
    layout_key='RNA_UMAP',
    color_by='RNA_pseudotime',
)
```

---
### 3) Identify pseudotime correlated features

We can now identify the features that are correlated with pseudotime and hence increase or decrease along the pseudotime.`run_pseudotime_marker_search` function will calculate the correlation coefficient for each of the valid features/genes against the pseudotime. The only mandatory parameter that `run_pseudotime_marker_search` function needs is `pseudotime_key` the value of which should the cell attribute column that stores the pseudotime information

```{code-cell} ipython3
ds.run_pseudotime_marker_search(pseudotime_key='RNA_pseudotime')
```

Once calculated, the correlation values against pseudotime are saved under the feature attribute/metadata table ('I__RNA_pseudotime__p', here). The name of of the column is according to this pattern: `<cell_key>__<pseudotime_key>__<p>`. The corresponding p-value is saved under the same column name pattern with suffix `p`

```{code-cell} ipython3
ds.RNA.feats.head()
```

---
### 4) Visualize pseudotime correlated features

In this section will do deeper on how to use the pseudotime correlation values for further exploratory analysis.

The first step is to export the values in a convenient dataframe format. we can use the `to_pandas_dataframe` methods of the feature attribute table to export the dataframe containing only the columns of choice

```{code-cell} ipython3
corr_genes_df = ds.RNA.feats.to_pandas_dataframe(
    columns=[
        'names',
        'I__RNA_pseudotime__p',
        'I__RNA_pseudotime__r'
    ],
    key='I')

# Rename the columns to be shorter
corr_genes_df.columns = ['names', 'p_value', 'r_value']
```

Let's checkout the genes that are negatively correlated with the pseudotime. These genes increase in expression as the pseudotime progresses., i.e. as cells divide

```{code-cell} ipython3
corr_genes_df.sort_values('r_value')[:15]
```

Let's visualize the expression of some of these genes on the UMAP plot

```{code-cell} ipython3
ds.plot_layout(
    layout_key='RNA_UMAP',
    color_by=['Spp1', 'Dbi', 'Sparc'],
    width=3.5, 
    height=3.5,
    point_size=5,
)
```

Now let's checkout the genes that are most positively correlated with the pseudotime. These genes **decrease** in expression as the pseudotime progresses

```{code-cell} ipython3
corr_genes_df.sort_values('r_value', ascending=False)[:10]
```

```{code-cell} ipython3
ds.plot_layout(
    layout_key='RNA_UMAP',
    color_by=['Aplp1', 'Gnas', 'Cpe'],
    width=3.5,
    height=3.5, 
    point_size=5,
)
```

---
### 5) Identify feature modules based on pseudotime

`run_pseudotime_marker_search` is excellent to find the genes are linearly correlated with the pseudotime. This function provides us informative statistical metrics to identify genes that are most strongly correlated with the pseudotime. However, with these methods we do not recover all the dynamic patterns of expression along the pseudotime. For example, there might be certain genes that express only in the middle of the trajectory or in one branch of the trajectory.

`run_pseudotime_aggregation` performs two task: 1) It arranges cells along the pseudotime and creates a smoothened, scaled and binned matrix of data 2) Clustering (KNN+Paris) is performed on this matrix to identify the groups of features/genes that have similar expression patterns along the pseudotime.

```{code-cell} ipython3
ds.run_pseudotime_aggregation(
    pseudotime_key='RNA_pseudotime',
    cluster_label='pseudotime_clusters',
    n_clusters = 15,
    window_size=200,
    chunk_size=100,
)
```

There are two primary results of `run_pseudotime_aggregation`: 
1) The  binned matrix is saved under `aggregate_<cell_key>_<feat_key>_<pseudotime_key>`
2) Feature clusters are saved under feature attributes table

```{code-cell} ipython3
# The binned data matrix. Here we print the shape of the matrix indicating the number of features and numner of bins respectively
ds.RNA.z['aggregated_I_I_RNA_pseudotime/data'].shape
```

```{code-cell} ipython3
# Fetching pseudotime based cluster identity of features
ds.RNA.feats.fetch('pseudotime_clusters')
```

`plot_pseudotime_heatmap` allows visualizing the binned matrix along with the feature clusters

```{code-cell} ipython3
# Highlighting some marker genes
genes_to_label = ['S100b', 'Nrarp', 'Atoh8', 'Grin2c', 'Slc35d3',
                  'Sst', 'Mnx1', 'Ins2', 'Gm11837', 'Irx1']

ds.plot_pseudotime_heatmap(
    cell_key='I',
    feat_key='I',
    feature_cluster_key='pseudotime_clusters',
    pseudotime_key='RNA_pseudotime',
    show_features=genes_to_label
)
```

The heatmap above shows the gene expression dynamics as the cells progress throught the pseudotime. Here, cluster 1 captures the genes that have highest expression in early pseudotime while cluster 15 captures genes whose expression peak in the late pseudotime.

We can visualize the expression of the above selected genes on UMAP to check whether their cluster identity corroborates their expression pattern.

```{code-cell} ipython3
ds.plot_layout(
    layout_key='RNA_UMAP', 
    color_by=genes_to_label,
    width=3,
    height=3, 
    point_size=5,
    n_columns=5,
)
```

---
### 6) Merging pseudotime-based feature modules into a new assay

The pseudotime based clusters of features can be used create a new assay. `add_grouped_assay` will take each cluster and take the mean expression of genes from that cluster and add it to a new assay. The motivation behind this approach is that we do not have to add many columns to our cell metadata table and have the mean cluster values readily available for analysis.

Taking mean cluster values is a powerful approach that allows use to explore cumulative pattern of highly correlated genes. Here we create a new assay under title `PTIME_MODULES`

```{code-cell} ipython3
ds.add_grouped_assay(
    group_key='pseudotime_clusters',
    assay_label='PTIME_MODULES'
)
```

```{code-cell} ipython3
# DataStore summary showing `PTIME_MODULES` assay with 15 features (number of pseudotime based feature clusters)
ds
```

The mean values from each cluster are saved within the assay and tagged with names like `group_1`, `group_2`, etc

```{code-cell} ipython3
ds.PTIME_MODULES.feats.head()
```

We can visualize these cluster mean values directly on the UMAP like this:

```{code-cell} ipython3
n_clusters = 15
ds.plot_layout(
    from_assay='PTIME_MODULES',
    layout_key='RNA_UMAP', 
    color_by=[f"group_{i}" for i in range(1,n_clusters+1)],
    width=3, 
    height=3,
    point_size=5,
    n_columns=5,
    cmap='coolwarm',
)
```

This figure complements the heatmap we generated earlier very nicely. Using this approach we have clearly found **gene modules** that are restricted in expression to certain portion of the pseudotime and differentiation trajectory

+++

---
### 7) Comparing pseudotime based feature modules with cluster markers

Here we will compare the pseudotime based feature module extraction approach with classical cluster marker approach.

```{code-cell} ipython3
# Running marker search
ds.run_marker_search(group_key='RNA_cluster')
```

Here we extract features from pseudotime-based cluster/group 13. These genes are the ones that show high expressio in Beta cells. 

```{code-cell} ipython3
ptime_feat_clusts = ds.RNA.feats.to_pandas_dataframe(
    columns=['names', 'pseudotime_clusters']
)
ptime_based_markers = ptime_feat_clusts.names[ptime_feat_clusts.pseudotime_clusters == 13]
ptime_based_markers.head()
```

Now we extract all the marker genes for cell cluster 8, this cluster predominantly contains the Beta cells.

```{code-cell} ipython3
cell_cluster_markers = ds.get_markers(
    group_key='RNA_cluster',
    group_id='8'
).feature_name

cell_cluster_markers.head()
```

```{code-cell} ipython3
# let's checkout the number of Beta cell associated genes from both methods
ptime_based_markers.shape, cell_cluster_markers.shape
```

```{code-cell} ipython3
# let's checkout the number of common Beta cell associated genes from both methods
len(set(cell_cluster_markers.index).intersection(ptime_based_markers.index))
```

Let's visualize the cumulative expression of genes that are present only in cluster marker based approach

```{code-cell} ipython3
temp = list(set(cell_cluster_markers.index).difference(ptime_based_markers.index))
ds.cells.insert(
    column_name='Cell cluster based markers', 
    values=ds.RNA.normed(feat_idx=sorted(temp)).mean(axis=1).compute(),
    overwrite=True)

ds.plot_layout(
    layout_key='RNA_UMAP',
    color_by='Cell cluster based markers',
    width=4, height=4, point_size=5, n_columns=5, cmap='coolwarm',
)
```

Let's now do this the other way and visualize the cumulative expression of genes that are present only in pseudotime-based approach

```{code-cell} ipython3
temp = list(set(ptime_based_markers.index).difference(cell_cluster_markers.index))
ds.cells.insert(
    column_name='Pseudotime based markers',
    values=ds.RNA.normed(feat_idx=sorted(temp)).mean(axis=1).compute(),
    overwrite=True)

ds.plot_layout(
    layout_key='RNA_UMAP',
    color_by='Pseudotime based markers',
    width=4, height=4, point_size=5, n_columns=5, cmap='coolwarm',
)
```

The pseudotime-based approach clearly captures a lot of signal that would be otherwise missed by simply taking a cell cluster marker based approach. 

---
That is all for this vignette.
