---
description: Merge scRNA-seq datasets with Harmony batch correction, partial PCA, and LISI integration metrics.
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

(harmony_batch_correction)=

# Merge, Harmony, and partial PCA

This tutorial merges datasets from different Zarr files, corrects batch effects with partial PCA or Harmony, and quantifies integration with LISI and related metrics. See also the {ref}`integration methods guide <integration_guide>`. Metrics details are in {doc}`integration_metrics`.

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')
scarf.__version__
```

---
## 1) Fetch datasets in Zarr format

Here we use the same datasets as in {ref}`mapping and label transfer <data_projection>`. We download the files in Zarr format.

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='kang_15K_pbmc_rnaseq',
    save_path='scarf_datasets',
    as_zarr=True
)

scarf.fetch_dataset(
    dataset_name='kang_14K_ifnb-pbmc_rnaseq', 
    save_path='scarf_datasets',
    as_zarr=True
)
```

The Zarr files need to be loaded as a DataStore before they can be merged:

```{code-cell} ipython3
ds_ctrl = scarf.DataStore(
    'scarf_datasets/kang_15K_pbmc_rnaseq/data.zarr/',
    nthreads=4
)

ds_ctrl
```

```{code-cell} ipython3
ds_stim = scarf.DataStore(
    'scarf_datasets/kang_14K_ifnb-pbmc_rnaseq/data.zarr',
    nthreads=4
)

ds_stim
```

---
## 2) Merging datasets

The merging step will make sure that the features are in the same order as in the merged file. The merged data will be dumped into a new Zarr file. Use `AssayMerge` to merge multiple samples (the `ZarrMerge` name is a deprecated alias that emits a warning).

```{code-cell} ipython3
scarf.AssayMerge(
    zarr_path='scarf_datasets/kang_merged_pbmc_rnaseq.zarr',
    assays=[ds_ctrl.RNA, ds_stim.RNA],
    names=['ctrl', 'stim'],
    merge_assay_name='RNA',
    prepend_text='orig',
    reset_cell_filter=False,
    source_column='sample_id',
    overwrite=True,
).dump()
```

Load the merged Zarr file as a DataStore:

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/kang_merged_pbmc_rnaseq.zarr',
    nthreads=4
)
```

The merge removes calculated graphs and embeddings. It keeps each input cell filter because `reset_cell_filter=False`, stores the input metadata with the `orig_` prefix, and writes the corresponding entry from `names` to `sample_id`. Counts and cell metadata receive the same row permutation, so these columns remain aligned.

```{code-cell} ipython3
ds
```

The cell table now contains `sample_id`, the aligned `orig_cluster_labels`, and the preserved `I` filter. The source name is also prepended to each barcode in `ids`.

```{code-cell} ipython3
ds.cells.head()
```

Now we can check the number of cells from each of the samples:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ['sample_id'],
    key='I'
)['sample_id'].value_counts()
```

---
## 3) Naive analysis of merged datasets

By naive, we mean that we make no attempt to remove/account for the latent factors that might contribute to batch effect or treatment-specific effect.
It is usually a good idea to perform a 'naive' pipeline to get an idea about the degree of batch effects.

+++

We start with detecting the highly variable genes:

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=10,
    top_n=2000,
    min_mean=-3, 
    max_mean=2,
    max_var=6
)
```

Next, we create a graph of cells in a standard way.

```{code-cell} ipython3
ds.make_graph(
    feat_key='hvgs',
    k=21, 
    dims=25,
    n_centroids=100
)
```

Calculating UMAP embedding of cells:

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250, 
    spread=5,
    min_dist=1,
    parallel=True
)
```

```{code-cell} ipython3
ds.cells.head()
```

Visualization of cells from the two samples in the 2D UMAP space:

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='sample_id',
    legend_loc='right',
)
```

Visualization of cluster labels in the 2D UMAP space:

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='orig_cluster_labels',
    legend_loc='right',
)
```

---
## 4) Partial PCA training to reduce batch effects

The plots above clearly show that the cells from the two samples are distinct on the UMAP space and have not integrated. This clearly indicates a treatment-specific or simply a batch effect between the cells from the two samples. Another interesting pattern in the UMAP plot above is the 'mirror effect', i.e. the equivalent clusters from the two samples look like mirror images. This is often seen in the datasets where the heterogenity/cell population composition is not strongly affected by the treatment.

We will now attempt to integrate the cells from the two samples so that we obtain same cell types that do not form separate clusters. One can do this by training the PCA on cells from only one of the samples. Training PCA on cells from only one of the samples will diminish the contribution of genes differentially expressed between the two samples.

+++

First, we need to create a boolean column in the cell attribute table. This column will indicate whether a cell belongs to one of the samples. Here we will create a new column `is_ctrl` and mark the values as True when a cell belongs to the `ctrl` sample.

```{code-cell} ipython3
ds.cells.insert(
    column_name=f'is_ctrl',
    values=(ds.cells.fetch_all('sample_id') == 'ctrl'),
    overwrite=True
)
```

The next step is to perform the partial PCA training. PCA is trained during the graph creation step. We will now use `pca_cell_key` parameter and set it to `is_ctrl` so that only 'ctrl' cells are used for PCA training.

```{code-cell} ipython3
ds.make_graph(
    feat_key='hvgs',
    k=21, 
    dims=25,
    n_centroids=100,
    pca_cell_key='is_ctrl'
)
```

We run UMAP as usual, but the UMAP embeddings are saved in a new cell attribute column so as to not overwrite the previous UMAP values. The new column will be called `RNA_pUMAP`; 'RNA' is automatically prepend because the assay name is `RNA`

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250, 
    spread=5,
    min_dist=1,
    parallel=True,
    label='pUMAP'
)
```

Visualize the new UMAP

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_pUMAP',
    color_by='sample_id',
    legend_loc='right',
)
```

Visualization of cluster labels in the new UMAP space shows that the cells from the same cell-type do not split into separate clusters like they did before.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_pUMAP',
    color_by='orig_cluster_labels',
    legend_loc='right',
)
```

---
## 5) Harmony batch correction

Harmony runs inside `make_graph` on the PCA embedding before KNN construction. Pass `harmonize=True` and the batch column name:

```{code-cell} ipython3
ds.make_graph(
    feat_key='hvgs',
    k=21,
    dims=25,
    n_centroids=100,
    harmonize=True,
    batch_columns=['sample_id'],
)

ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label='hUMAP'
)

ds.run_leiden_clustering(resolution=1.0)
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_hUMAP',
    color_by='sample_id',
    legend_loc='right',
)

splt.embedding(
    ds,
    layout_key='RNA_hUMAP',
    color_by='orig_cluster_labels',
    legend_loc='right',
)
```

Harmony is often preferred when multiple batches need to mix without choosing a single reference sample. Partial PCA (section 4) is lighter when one sample defines the embedding.

---
## 6) Quantifying integration quality

The UMAP plots suggest that partial PCA and Harmony mix the two samples, but a visual read is not enough. Scarf provides several metrics that quantify integration from different angles. See {doc}`integration_metrics` and the {ref}`integration methods guide <integration_guide>`. Re-run this section after the naive, partial PCA, and Harmony graphs to compare approaches on the same footing.

**LISI** measures how well a label mixes inside each cell's KNN neighborhood. Running it on `sample_id` tells us whether batches are mixed, while running it on `orig_cluster_labels` checks that cell types are still grouped. Good integration raises batch LISI while keeping cell-type LISI low. With `save_result=True` the per-cell scores are written back as `lisi__sample_id__*` columns, which you can overlay on the UMAP layouts.

Default `perplexity=30` needs a graph with at least about `3 * perplexity` neighbors. This tutorial builds graphs with `k=21`, so the default fits. If you use a smaller `k` (for example `k=11` in {doc}`integration_metrics`), Scarf lowers perplexity automatically and warns; scores remain valid for that smaller neighborhood.

```{code-cell} ipython3
ds.metric_lisi(
    label_colnames=['sample_id', 'orig_cluster_labels'],
    save_result=True,
)
```

**Batch mixing** condenses batch LISI into a single number in `[0, 1]` by rescaling the mean against the mixing perfectly integrated data would reach for these batch sizes. This makes it easy to compare across graphs. Higher is better.

```{code-cell} ipython3
ds.metric_batch_mixing(label_colname='sample_id')
```

**Silhouette** scores how separated each cluster is from its nearest neighboring cluster, from -1 to 1. Values near 1 mean distinct clusters. Read it alongside the batch metrics, since over-correction can mix genuinely different cell types.

```{code-cell} ipython3
ds.metric_silhouette(res_label='leiden_cluster')
```

**Label concordance** compares two labelings of the same cells with ARI or NMI. Here it checks how well the fresh Leiden clusters agree with the imported annotations. Note that this measures label agreement, not batch mixing.

```{code-cell} ipython3
ds.metric_label_concordance(
    label_columns=['RNA_leiden_cluster', 'orig_cluster_labels'],
    metric='ari'
)
```

---
See {doc}`integration_metrics` for more metric detail and {doc}`choosing_integration_methods` for method choice.
