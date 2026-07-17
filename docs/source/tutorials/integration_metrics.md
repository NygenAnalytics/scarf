---
description: Quantify batch mixing and label concordance with DataStore metric_* helpers.
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

(lisi_metrics)=
(integration_metrics)=

# Integration metrics

After merging batches and building a graph ({doc}`data_integration`), quantify mixing with
`DataStore.metric_*` helpers. Underlying implementations live in `scarf.metrics`.

## Prerequisites

- A merged multi-sample `DataStore` with a neighbourhood graph
- A batch/sample column and optionally paired label columns for concordance

## Dataset

```{code-cell} ipython3
import scarf

scarf.fetch_dataset('kang_15K_pbmc_rnaseq', save_path='scarf_datasets', as_zarr=True)
scarf.fetch_dataset('kang_14K_ifnb-pbmc_rnaseq', save_path='scarf_datasets', as_zarr=True)

ds_ctrl = scarf.DataStore('scarf_datasets/kang_15K_pbmc_rnaseq/data.zarr', nthreads=4)
ds_stim = scarf.DataStore('scarf_datasets/kang_14K_ifnb-pbmc_rnaseq/data.zarr', nthreads=4)

merged = 'scarf_datasets/kang_merged_pbmc_rnaseq.zarr'
scarf.AssayMerge(
    zarr_path=merged,
    assays=[ds_ctrl.RNA, ds_stim.RNA],
    names=['ctrl', 'stim'],
    merge_assay_name='RNA',
    overwrite=True,
).dump()

ds = scarf.DataStore(merged, nthreads=4)
ds.cells.insert(
    column_name='sample_id',
    values=[x.split('__')[0] for x in ds.cells.fetch_all('ids')],
    overwrite=True,
)
ds.cells.insert(
    column_name='imported_labels',
    values=list(ds_ctrl.cells.fetch_all('cluster_labels'))
    + list(ds_stim.cells.fetch_all('cluster_labels')),
    overwrite=True,
)
ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
ds.run_leiden_clustering(resolution=0.5)
```

## LISI

`metric_lisi` takes one or more cell-metadata label columns (batches, clusters, and similar).
Default `perplexity=30` expects roughly `3 * perplexity` graph neighbors. This page builds
the graph with `k=11`, so Scarf reduces perplexity automatically and prints a warning. Scores
still describe mixing inside that smaller neighborhood. Raise `k` or lower `perplexity=` if
you want an exact match to the default.

```{code-cell} ipython3
lisi = ds.metric_lisi(label_colnames=['sample_id'])
lisi
```

## Batch mixing summary

```{code-cell} ipython3
ds.metric_batch_mixing(label_colname='sample_id')
```

## Silhouette and label concordance

```{code-cell} ipython3
ds.metric_silhouette(res_label='leiden_cluster')
```

```{code-cell} ipython3
ds.metric_label_concordance(
    label_columns=['RNA_leiden_cluster', 'imported_labels']
)
```

On this naive merged graph (no batch correction), batch LISI and batch mixing are usually
low, and Leiden vs `imported_labels` concordance can also be low when clusters split by
sample. Compare the same metrics after partial PCA or Harmony in {doc}`data_integration`.

See {ref}`integration methods guide <integration_guide>` for how these scores relate to
method choice.

## Next steps

- {doc}`data_integration`
- {doc}`choosing_integration_methods`
- {doc}`../reference/api/integration`
