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

# Projection, label transfer, and unified embeddings

Scarf can project (map) cells from one dataset onto another. Mapping is a lighter alternative
to merging both matrices into one batch-corrected object. This notebook uses data from
[Kang et al.](https://www.nature.com/articles/nbt.4042): control and IFN-B treated PBMCs.
The Scarf catalog already ships both samples as Zarr stores with UMAP and cluster labels.
For a reusable harmonized atlas workflow, see {doc}`reference_atlas`.

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')
scarf.__version__
```

---
## 1) Fetch datasets in Zarr format

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

splt.embedding(
    ds_ctrl,
    layout_key='RNA_UMAP',
    color_by='cluster_labels',
)
```

```{code-cell} ipython3
# Interferon beta stimulated PBMC data
ds_stim = scarf.DataStore(
    'scarf_datasets/kang_14K_ifnb-pbmc_rnaseq/data.zarr',
    nthreads=4
)

splt.embedding(
    ds_stim,
    layout_key='RNA_UMAP',
    color_by='cluster_labels',
)
```

---
## 2) K-Nearest Neighbours (KNN) mapping

The ``run_mapping`` method projects target cells onto the fixed reference graph. The reference cells are in the `DataStore` where `run_mapping` is called, and the target assay is supplied as an argument. Scarf aligns target features to the ordered reference feature set, applies the scaling stored with the reference PCA, and queries the reference ANN index. Target cells are never inserted into the reference index.

By default, `missing_feature_policy='zero'` fills features absent from the target with zero normalized expression. Earlier releases silently used one, so partial-overlap results can change after remapping. Use `missing_feature_policy='error'` when complete overlap is required. `missing_feature_policy='intersection'` creates an isolated, overlap-only index for a controlled comparison; it does not alter the authoritative reference graph. The deprecated `exclude_missing=True` alias also leaves its historical overlap feature key for one compatibility cycle.

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
ds_ctrl.run_mapping(
    target_assay=ds_stim.RNA,
    target_name='stim',
    target_feat_key='hvgs_ctrl',
    save_k=5
)
```

Key mapping parameters:
- `save_k`: number of reference neighbors stored per target cell (default 3)
- `missing_feature_policy`: choose `zero`, `error`, or `intersection` handling for absent target features
- `filter_null`: with `missing_feature_policy='intersection'`, removes target features with zero total counts
- `run_coral`: deprecated experimental feature-space correction. Build a Symphony-style mapping reference for harmonized atlas mapping.
- `ref_mu` and `ref_sigma`: deprecated compatibility flags. Reference statistics are always used.

---
## 3) Mapping scores

+++

Mapping scores support cross-dataset cluster similarity inspection. By default,
`get_mapping_score` accumulates distance-derived neighbor weights for each reference cell and
normalizes them by the number of mapped target cells and saved neighbors. Frequently selected,
nearby reference cells therefore receive higher scores. `target_groups` calculates a separate
score for each target group. Here the stimulated-cell clusters are used as groups, and the
UMAPs show where selected target clusters map onto the control reference.

```{code-cell} ipython3
# Generate plots for IFN-B stimulated cells from NK and CD14 monocyte clusters.

for g, ms in ds_ctrl.get_mapping_score(
    target_name='stim',
    target_groups=ds_stim.cells.fetch('cluster_labels'),
    log_transform=True
):
    
    if g in ['NK', 'CD 14 Mono']:
        print (f"Target cluster {g}")
        splt.embedding(
            ds_ctrl,
            layout_key='RNA_UMAP',
            color_by='cluster_labels',
            point_sizes=ms * 10,
            figsize=(4, 4),
        )
```

---
## 4) Label transfer

Using the nearest neighbours of the target cells in the reference data, we can transfer labels from reference cells to target cells based on majority voting. This means that if a target cell has 'most' of its total edge weight shared with cells from one cell type, then that cell type label is transferred to the target cell. The default threshold for 'most' is 0.5, i.e. half of all edge weight. `get_target_classes` method returns the transferred labels for each cell from a given mapped target dataset.

The `reference_class_group` parameter decides which labels to transfer. This can be any column from the cell attribute table that has categorical values, generally users would use `RNA_leiden_cluster` or `RNA_cluster` but they can also use other labels. Here, for example, we use the custom labels stored under `cluster_labels` column.

```{code-cell} ipython3
transferred_labels = ds_ctrl.get_target_classes(
    target_name='stim',
    reference_class_group='cluster_labels'
)

transferred_labels
```

Use `get_target_label_evidence` to inspect neighbor-vote fraction, entropy, margin, feature coverage, and a reference distance percentile. These are diagnostic quantities, not calibrated probabilities. The distance percentile uses the reference self-neighbor distribution rather than the query distribution. The method returns `NA` by default when the winning vote does not meet a chosen threshold, when top labels tie, or when a query has no directional information in reference PC space.

```{code-cell} ipython3
evidence = ds_ctrl.get_target_label_evidence(
    target_name='stim',
    reference_class_group='cluster_labels',
    threshold_fraction=0.6
)
evidence.head()
```

We can now save these transferred labels in the stimulated-cell dataset and visualize them on
its UMAP.

```{code-cell} ipython3
ds_stim.cells.insert(
    'transferred_labels',
    transferred_labels.values,
    overwrite=True
)
```

```{code-cell} ipython3
splt.embedding(
    ds_stim,
    layout_key='RNA_UMAP',
    color_by='transferred_labels',
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

The column-normalized table below shows the composition of each transferred label. Each column
sums to 100, and a diagonal entry is the precision for that predicted label. It is not an
overall accuracy score.

```{code-cell} ipython3
(100 * df / df.sum(axis=0)).round(1)
```

Overall label accuracy is the fraction of target cells whose transferred label matches the
stored target label:

```{code-cell} ipython3
round(
    100 * (
        ds_stim.cells.fetch('cluster_labels') == transferred_labels.to_numpy()
    ).mean(),
    1,
)
```

---
## 5) Unified UMAPs

Unified UMAP adds query-reference edges and reruns UMAP on the combined graph. It is useful for exploration, but it moves the reference coordinates and can change when a different query is supplied. Do not use it as a stable atlas coordinate system.

```{code-cell} ipython3
ds_ctrl.run_unified_umap(
    target_names=['stim'],
    ini_embed_with='RNA_UMAP',
    target_weight=1,
    use_k=5,
    n_epochs=100
)
```

Since unified embeddings contain cells from another dataset, use
`splt.unified_embedding`. Use `splt.embedding` for regular layouts stored in cell
metadata.

```{code-cell} ipython3
splt.unified_embedding(
    ds_ctrl,
    layout_key='unified_UMAP',
    ref_name='ctrl',
)
```

For stable placement into an existing reference layout, use deterministic neighbor-weighted projection. It writes only target coordinates and leaves reference coordinates unchanged. A read-only datastore returns the coordinate array in memory instead of attempting a write.

```{code-cell} ipython3
ds_ctrl.project_mapping_layout(
    target_name='stim',
    reference_layout_key='RNA_UMAP'
)
```

We can visualize only the target cells, i.e IFN-B stimulated cells, in the unified embedding. The target cells can be colored based on their original cluster identity. Target cells of similar types are close together on the unified embedding and overlap with the cell types of the reference data

```{code-cell} ipython3
splt.unified_embedding(
    ds_ctrl,
    layout_key='unified_UMAP',
    show_target_only=True,
    target_groups=ds_stim.cells.fetch('cluster_labels'),
    legend_loc='on_data',
)
```

---
## 6) Unified tSNE

Unified tSNE co-embeds reference and target cells using the same unified graph as unified UMAP:

```{code-cell} ipython3
ds_ctrl.run_unified_tsne(
    target_names=['stim'],
    ini_embed_with='RNA_UMAP',
    target_weight=1,
    use_k=5,
    max_iter=500
)

splt.unified_embedding(
    ds_ctrl,
    layout_key='unified_tSNE',
    ref_name='ctrl',
)
```

---
For building a reusable mapping reference from a harmonized atlas, see {doc}`reference_atlas`.
