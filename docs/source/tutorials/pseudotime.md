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

# Estimating pseudotime ordering and expression dynamics

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

scarf.__version__
```

---
## 1) Fetch pre-analyzed data

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
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=['RNA_cluster', 'clusters'],
    legend_loc='on_data',
).show();
```

---
## 2) Estimate pseudotime ordering

Scarf uses a memory-efficient implementation of the [PBA algorithm](https://github.com/AllonKleinLab/PBA) ([Weinreb et al. 2018, PNAS](https://www.pnas.org/content/115/10/E2467)) to estimate a pseudotime ordering. `run_pseudotime_scoring` works with any assay that has a neighborhood graph. The method is supervised, so provide source groups such as progenitor cells and sink groups such as differentiated cell states.

```{code-cell} ipython3
pseudotime = ds.run_pseudotime_scoring(
    source_sink_key="RNA_cluster",    # Column that contains cluster information 
    sources=[1],                      # Source clusters
    sinks=[3],                        # Sink clusters
)
```

By default, the calculated values are saved under **'RNA_pseudotime'**, where 'RNA' is replaced by the assay name. A companion boolean column **'RNA_pseudotime__valid'** is also written. The returned result exposes both names as `pseudotime_key` and `validity_key`. When the selected graph is fully connected, every cell is valid. If the graph has multiple components, only the largest one is scored by default. The remaining cells hold `NaN`, and downstream steps should use the validity column as `cell_key`. The UMAP below shows progression from 0 to 1.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=pseudotime.pseudotime_key,
).show();
```

---
## 3) Identify pseudotime correlated features

`run_pseudotime_marker_search` calculates a correlation coefficient and p-value for each selected feature against the pseudotime ordering.

```{code-cell} ipython3
markers = ds.run_pseudotime_marker_search(
    cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
)
```

The correlations and p-values are saved in feature metadata. Their generated column names are available as `markers.correlation_key` and `markers.p_value_key`. The same values, feature indices, and feature names are returned in `markers.table`.

```{code-cell} ipython3
markers.table.head()
```

---
## 4) Visualize pseudotime correlated features

This section uses the returned correlations for exploratory analysis.

The returned table can be used directly for sorting and filtering.

```{code-cell} ipython3
corr_genes_df = markers.table
```

Genes with a negative correlation decrease in expression as pseudotime progresses.

```{code-cell} ipython3
corr_genes_df.sort_values('r_value')[:15]
```

Let's visualize the expression of some of these genes on the UMAP plot

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=['Spp1', 'Dbi', 'Sparc'],
    sort_values=True,
).show();
```

Genes with a positive correlation increase in expression as pseudotime progresses.

```{code-cell} ipython3
corr_genes_df.sort_values('r_value', ascending=False)[:10]
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=['Aplp1', 'Gnas', 'Cpe'],
    sort_values=True,
).show();
```

---
## 5) Identify feature modules based on pseudotime

`run_pseudotime_marker_search` identifies features with a linear relationship to pseudotime. It does not capture every dynamic pattern. Some genes, for example, may peak only in the middle of a trajectory or along one branch.

`run_pseudotime_aggregation` orders cells by pseudotime and creates a smoothed, scaled, binned expression matrix. It then applies KNN and Paris clustering to identify features with similar expression patterns.

```{code-cell} ipython3
modules = ds.run_pseudotime_aggregation(
    cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
    cluster_label='pseudotime_clusters',
    n_clusters = 15,
    window_size=200,
    chunk_size=100,
)
```

The returned result contains the lazy binned matrix in `modules.data`, the aligned physical feature indices, and their cluster assignments. It also exposes the saved feature column as `modules.cluster_key` and the Zarr location as `modules.storage_path`.

Features with mean expression below `min_exp` or with no variation along the ordering are treated as invalid. They are excluded from the clustering and from the heatmap below, and they receive the unassigned cluster value (`-1`) in the feature table.

```{code-cell} ipython3
# Number of retained features and pseudotime bins
modules.data.shape
```

```{code-cell} ipython3
# Cluster assignment for each retained feature
modules.feature_clusters
```

`splt.pseudotime_heatmap` visualizes the binned matrix along with the feature clusters

```{code-cell} ipython3
# Highlighting some marker genes
genes_to_label = ['S100b', 'Nrarp', 'Atoh8', 'Grin2c', 'Slc35d3',
                  'Sst', 'Mnx1', 'Ins2', 'Gm11837', 'Irx1']

splt.pseudotime_heatmap(
    ds,
    cell_key=modules.cell_key,
    feat_key=modules.feature_key,
    feature_cluster_key=modules.cluster_key,
    pseudotime_key=modules.pseudotime_key,
    show_features=genes_to_label,
)
```

The heatmap above shows the gene expression dynamics as the cells progress through the pseudotime. Each block of rows is one feature module. Some modules capture genes that peak early in the pseudotime while others peak later. The module numbers are assigned by the clustering step and do not follow the pseudotime order.

We can visualize the expression of the above selected genes on UMAP to check whether their cluster identity corroborates their expression pattern.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=genes_to_label,
    n_columns=5,
    sort_values=True,
).show();
```

---
## 6) Merging pseudotime-based feature modules into a new assay

The pseudotime based clusters of features can be used create a new assay. `add_grouped_assay` will take each cluster and take the mean expression of genes from that cluster and add it to a new assay. The motivation behind this approach is that we do not have to add many columns to our cell metadata table and have the mean cluster values readily available for analysis.

Taking mean cluster values is a powerful approach that allows use to explore cumulative pattern of highly correlated genes. Here we create a new assay under title `PTIME_MODULES`

```{code-cell} ipython3
ds.add_grouped_assay(
    group_key=modules.cluster_key,
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
splt.embedding(
    ds,
    from_assay='PTIME_MODULES',
    layout_key='RNA_UMAP',
    color_by=[f"group_{i}" for i in range(1, n_clusters + 1)],
    n_columns=5,
    color_scale=splt.ColorScale(cmap='coolwarm'),
).show();
```

This figure complements the heatmap we generated earlier very nicely. Using this approach we have clearly found **gene modules** that are restricted in expression to certain portion of the pseudotime and differentiation trajectory

+++

---
## 7) Comparing pseudotime based feature modules with cluster markers

Here we will compare the pseudotime based feature module extraction approach with classical cluster marker approach.

```{code-cell} ipython3
# Running marker search
ds.run_marker_search(group_key='RNA_cluster')
```

First we extract the marker genes for cell cluster 8, which predominantly contains the Beta cells.

```{code-cell} ipython3
cell_cluster_markers = ds.get_markers(
    group_key='RNA_cluster',
    group_id='8',
).feature_name

cell_cluster_markers.head()
```

Next we pick the pseudotime feature module that shares the most genes with these Beta cell markers. The module numbering is assigned by clustering and can change between runs, so we select the module in a data-driven way instead of hard-coding a cluster id.

```{code-cell} ipython3
ptime_feat_clusts = ds.RNA.feats.to_pandas_dataframe(
    columns=['names', 'pseudotime_clusters']
)

beta_marker_names = set(cell_cluster_markers)
assigned = ptime_feat_clusts[ptime_feat_clusts.pseudotime_clusters != -1]
module_overlap = assigned.groupby('pseudotime_clusters')['names'].apply(
    lambda names: len(set(names) & beta_marker_names)
)
beta_module = int(module_overlap.idxmax())
beta_module
```

The genes belonging to this Beta associated module are:

```{code-cell} ipython3
ptime_based_markers = ptime_feat_clusts.names[
    ptime_feat_clusts.pseudotime_clusters == beta_module
]
ptime_based_markers.head()
```

```{code-cell} ipython3
# Number of genes captured by each approach
ptime_based_markers.shape, cell_cluster_markers.shape
```

```{code-cell} ipython3
# Number of genes shared by both approaches, compared by gene name
len(set(ptime_based_markers) & set(cell_cluster_markers))
```

Let's visualize the cumulative expression of genes that are present only in cluster marker based approach

```{code-cell} ipython3
available_names = set(ptime_feat_clusts.names)
cell_only = sorted(
    (set(cell_cluster_markers) - set(ptime_based_markers)) & available_names
)
ds.cells.insert(
    column_name='Cell cluster based markers',
    values=ds.RNA.mean_features(cell_only),
    overwrite=True,
)

splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='Cell cluster based markers',
    color_scale=splt.ColorScale(cmap='coolwarm'),
).show();
```

Let's now do this the other way and visualize the cumulative expression of genes that are present only in pseudotime-based approach

```{code-cell} ipython3
ptime_only = sorted(
    (set(ptime_based_markers) - set(cell_cluster_markers)) & available_names
)
ds.cells.insert(
    column_name='Pseudotime based markers',
    values=ds.RNA.mean_features(ptime_only),
    overwrite=True,
)

splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='Pseudotime based markers',
    color_scale=splt.ColorScale(cmap='coolwarm'),
).show();
```

The pseudotime-based approach clearly captures a lot of signal that would be otherwise missed by simply taking a cell cluster marker based approach. 

---
For marker-based annotation on cluster labels, see {doc}`annotation`.
