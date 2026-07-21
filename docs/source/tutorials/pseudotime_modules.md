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

# Pseudotime modules

Group features with similar pseudotime expression patterns, store module means as a new
assay, and compare those modules with classical cluster markers.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and UMAP embedding
- Complete {doc}`pseudotime` through correlated genes, or let the setup below score
  pseudotime when it is missing

## What you will learn

- Aggregate features into pseudotime expression modules
- Create a grouped assay from module means with `add_grouped_assay`
- Compare pseudotime modules with cluster marker genes

## Dataset

This page uses the Bastidas-Ponce pancreas store from {doc}`pseudotime`. The setup below
is standalone: it downloads the store, opens a `DataStore`, and runs pseudotime scoring
when needed.

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='./scarf_datasets',
    zarr=True,
)

ds = scarf.DataStore(
    'scarf_datasets/bastidas-ponce_4K_pancreas-d15_rnaseq/data.zarr',
    nthreads=4,
    default_assay='RNA',
)

pseudotime_key = 'RNA_pseudotime'
validity_key = 'RNA_pseudotime__valid'

if pseudotime_key not in ds.cells.columns:
    pseudotime = ds.run_pseudotime_scoring(
        source_sink_key='RNA_cluster',
        sources=[1],
        sinks=[3],
    )
    pseudotime_key = pseudotime.pseudotime_key
    validity_key = pseudotime.validity_key
```

## Guided steps

### 1. Identify feature modules based on pseudotime

`run_pseudotime_marker_search` identifies features with a linear relationship to pseudotime. It does not capture every dynamic pattern. Some genes, for example, may peak only in the middle of a trajectory or along one branch.

`run_pseudotime_aggregation` orders cells by pseudotime and creates a smoothed, scaled, binned expression matrix. It then applies KNN and Paris clustering to identify features with similar expression patterns.

```{code-cell} ipython3
modules = ds.run_pseudotime_aggregation(
    cell_key=validity_key,
    pseudotime_key=pseudotime_key,
    cluster_label='pseudotime_clusters',
    n_clusters=15,
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

`ds.plots.pseudotime_heatmap` visualizes the binned matrix along with the feature clusters.

```{code-cell} ipython3
# Highlighting some marker genes
genes_to_label = ['S100b', 'Nrarp', 'Atoh8', 'Grin2c', 'Slc35d3',
                  'Sst', 'Mnx1', 'Ins2', 'Gm11837', 'Irx1']

ds.plots.pseudotime_heatmap(
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
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=genes_to_label,
    n_columns=5,
    sort_values=True,
)
```

### 2. Merging pseudotime-based feature modules into a new assay

The pseudotime-based feature clusters can seed a new assay. `add_grouped_assay` takes the
mean expression of genes in each cluster and stores those means as features in a new assay.
That keeps many related genes out of the cell metadata table while still exposing one summary
value per module. Here we create an assay named `PTIME_MODULES`.

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
ds.plots.embedding(
    from_assay='PTIME_MODULES',
    layout_key='RNA_UMAP',
    color_by=[f"group_{i}" for i in range(1, n_clusters + 1)],
    n_columns=5,
    color_scale=splt.ColorScale(cmap='coolwarm'),
)
```

This figure complements the earlier heatmap. Several modules are restricted to parts of the
pseudotime and differentiation trajectory.

### 3. Comparing pseudotime based feature modules with cluster markers

Compare the pseudotime-based feature modules with classical cluster markers.

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

Visualize the cumulative expression of genes present only in the cluster-marker set:

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

ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='Cell cluster based markers',
    color_scale=splt.ColorScale(cmap='coolwarm'),
)
```

Do the reverse comparison for genes present only in the pseudotime-based module:

```{code-cell} ipython3
ptime_only = sorted(
    (set(ptime_based_markers) - set(cell_cluster_markers)) & available_names
)
ds.cells.insert(
    column_name='Pseudotime based markers',
    values=ds.RNA.mean_features(ptime_only),
    overwrite=True,
)

ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='Pseudotime based markers',
    color_scale=splt.ColorScale(cmap='coolwarm'),
)
```

The two approaches overlap but are not identical. Genes unique to the pseudotime module can
highlight trajectory signal that cluster markers alone miss.

## Common mistakes and limitations

- Interpreting linear correlation as the only form of expression dynamics along pseudotime
- Treating module numbers as ordered along pseudotime (they are clustering labels)
- Comparing module gene lists to cluster markers without accounting for unassigned features (`-1`)

## Saved results

Feature module labels are stored under `pseudotime_clusters` in feature metadata.
`PTIME_MODULES` holds mean expression per module.

## Further reading

- Weinreb et al. 2018, population balance analysis (PBA): https://doi.org/10.1073/pnas.1714723115
- [PBA reference implementation](https://github.com/AllonKleinLab/PBA)

## Next steps

- {doc}`fate_mapping`
- {doc}`annotation`
- {doc}`plotting`
