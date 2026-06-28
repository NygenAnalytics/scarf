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

(data_projection)=

## Projection, label transfer, and unified embeddings

Scarf allows projections (aka mapping) of cells from one dataset to another. Such projection can help in  understanding how cells are related between the two datasets. Projection/mapping is a lightweight alternative to full-blown data integration which focuses on biological interpretation. In this notebook we use data from [Kang et. al.](https://www.nature.com/articles/nbt.4042). We have already preprocessed the raw count matrix to generate UMAPs and clustering of the data ([notebook here](https://github.com/parashardhapola/scarf_vignettes/blob/main/kang_et_al_processing.ipynb)). We will use two datasets from this study: control and IFN-B treated PBMCs.

```{code-cell} ipython3
%load_ext autotime

import scarf
scarf.__version__
```

---
### 1) Fetch datasets in Zarr format

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

```{code-cell} ipython3
# Control/untreated PBMC data
ds_ctrl = scarf.DataStore(
    'scarf_datasets/kang_15K_pbmc_rnaseq/data.zarr',
    nthreads=4
)

ds_ctrl.plot_layout(
    layout_key='RNA_UMAP',
    color_by='cluster_labels'
)
```

```{code-cell} ipython3
# Interferon beta stimulated PBMC data
ds_stim = scarf.DataStore(
    'scarf_datasets/kang_14K_ifnb-pbmc_rnaseq/data.zarr',
    nthreads=4
)

ds_stim.plot_layout(
    layout_key='RNA_UMAP',
    color_by='cluster_labels'
)
```

---
### 2) K-Nearest Neighbours (KNN) mapping

The ``run_mapping`` method of DataStore class performs KNN mapping/projection of target cells over reference cells. Reference cells are the ones that are present in the object where `run_mapping` is being called. The `Assay` object of target cells is provided as an argument to `run_mapping`. This step will load the latest graph of the reference cells and query the Approximate Nearest Neighbour (ANN) index of the reference cells for nearest neighbours of each target cell. Since the ANN index doesn't contain any target cells, nearest neighbours of target cells will exclusively be reference cells. Under the hood, `run_mapping` method makes sure that the feature order in the target cells is same as that in the reference cells. By default, `run_mapping` will impute zero values for missing features in the target order to preserve the feature order. Here we have set `run_coral` parameter to True which activates CORAL normalization of target data. CORAL aligns the the feature distribution between reference and target cells thus removing systemic difference between reference and target cells. Read more about CORAL [here](https://arxiv.org/pdf/1612.01939.pdf). Here we use control PMBCs as reference because we invoke `run_mapping` on control PBMCs' DataStore object and provide stimulated PBMC's `RNA` assay as target.

+++

<div class="alert alert-block alert-info">
<p>
   <b>Reference cells</b>: The cells from the dataset that forms the basis of mapping. A KNN graph must already be calculated for this dataset.
</p>
<p>
    <b>Target cells</b>: The cells to be projected onto reference cells. This dataset is not required to have a graph calculated.
</p>
</div>

```{code-cell} ipython3
# CORAL algorithm can be very slow with large number of features (> 5000).
# Hence here it is recommended for only scRNA-Seq datasets.

ds_ctrl.run_mapping(
    target_assay=ds_stim.RNA,
    target_name='stim',
    target_feat_key='hvgs_ctrl',
    save_k=5,
    run_coral=True
)
```

Key mapping parameters:
- `save_k`: number of reference neighbors stored per target cell (default 3)
- `run_coral`: align feature distributions between reference and target
- `exclude_missing`: drop target features absent from reference instead of imputing zeros
- `filter_null`: remove target cells with no valid features after alignment

---
### 3) Mapping scores

+++

We can use `mapping scores` to perform cross-dataset cluster similarity inspection. `mapping scores` are scores assigned to each reference cell based on how frequently it was identified as one of the nearest neighbour of the target cells. ``get_mapping_score`` method allows generating these scores. We use an optional parameter of `get_mapping_score`, `target_groups`. `target_groups` takes grouping information for target cells such that mapping scores are calculated for one group at a time. Here we provide the cluster information of stimulated cells as group information and mapping scores will be obtained for each target cluster independently. The UMAPs below show how much mapping score each control cell received upon mapping from cells from one of the IFN-B stimulated cell clusters.

```{code-cell} ipython3
# Here we will generate plots for IFB-B stimulated cells from NK  and CD14 monocyte clusters.

for g, ms in ds_ctrl.get_mapping_score(
    target_name='stim',
    target_groups=ds_stim.cells.fetch('cluster_labels'),
    log_transform=True
):
    
    if g in ['NK', 'CD 14 Mono']:
        print (f"Target cluster {g}")
        ds_ctrl.plot_layout(
            layout_key='RNA_UMAP',
            color_by='cluster_labels',
            size_vals=ms*10,
            height=4, 
            width=4,
            legend_onside=False
        )
```

---
### 4) Label transfer

Using the nearest neighbours of the target cells in the reference data, we can transfer labels from reference cells to target cells based on majority voting. This means that if a target cell has 'most' of its total edge weight shared with cells from one cell type, then that cell type label is tranferred to the target cell. The default threshold for 'most' is 0.5, i.e. half of all edge weight. `get_target_classes` method returns the transferred labels for each cell from a given mapped target dataset.

The `reference_class_group` parameter decides which labels to transfer. This can be any column from the cell attribute table that has categorical values, generally users would use `RNA_leiden_cluster` or `RNA_cluster` but they can also use other labels. Here, for example, we use the custom labels stored under `cluster_labels` column.

```{code-cell} ipython3
transferred_labels = ds_ctrl.get_target_classes(
    target_name='stim',
    reference_class_group='cluster_labels'
)

transferred_labels
```

We can now save these transferred labels into the stimulated cell dataset and visualize them of its UMAP.

```{code-cell} ipython3
ds_stim.cells.insert(
    'transferred_labels',
    transferred_labels.values,
    overwrite=True
)
```

```{code-cell} ipython3
ds_stim.plot_layout(
    layout_key='RNA_UMAP',
    color_by='transferred_labels'
)
```

It can be quite interesting to check how the predicted/transferred labels compare to the actual labels of the target cells:

```{code-cell} ipython3
import pandas as pd

df = pd.crosstab(
    ds_stim.cells.fetch('cluster_labels'),
    ds_stim.cells.fetch('transferred_labels')
)
df
```

This cross-tabulation can be presented as percentage accuracy, where the values indicate the percentage of the transferred values that were correct.

```{code-cell} ipython3
(100 * df / df.sum(axis=0)).round(1)
```

---
### 5) Unified UMAPs

Scarf introduces Unified UMAPs, a strategy to embed target cells onto the reference manifold. To do so, we take the results of KNN projection and spike the graph of reference cells with target cells. We can control the weight of target-reference edges also, as well as the number of edges per target to retain. We rerun UMAP on this 'unified graph' to obtain a unified embdding. Following code shows how to call `run_unified_umap` method.

```{code-cell} ipython3
ds_ctrl.run_unified_umap(
    target_names=['stim'],
    ini_embed_with='RNA_UMAP',
    target_weight=1,
    use_k=5,
    n_epochs=100
)
```

Since the results of unified embedding contain 'foreign' cells, `plot_layout` function cannot be used to visualize all the cells. A specialized method, `plot_unified_layout` takes care of this issue. The following example shows co-embedded control (reference) and stimulated  (target) PBMCs.

```{code-cell} ipython3
ds_ctrl.plot_unified_layout(
    layout_key='unified_UMAP',
    show_target_only=False,
    ref_name='ctrl'
)
```

We can visualize only the target cells, i.e IFN-B stimulated cells, in the unified embedding. The target cells can be colored based on their original cluster identity. Target cells of similar types are close together on the unified embedding and overlap with the cell types of the reference data

```{code-cell} ipython3
ds_ctrl.plot_unified_layout(
    layout_key='unified_UMAP',
    show_target_only=True, 
    legend_ondata=True,
    target_groups=ds_stim.cells.fetch('cluster_labels')
)
```

---
### 6) Unified tSNE

Unified tSNE co-embeds reference and target cells using the same unified graph as unified UMAP:

```{code-cell} ipython3
ds_ctrl.run_unified_tsne(
    target_names=['stim'],
    ini_embed_with='RNA_UMAP',
    target_weight=1,
    use_k=5,
    max_iter=500
)

ds_ctrl.plot_unified_layout(
    layout_key='unified_tSNE',
    show_target_only=False,
    ref_name='ctrl'
)
```

---
That is all for this vignette.
