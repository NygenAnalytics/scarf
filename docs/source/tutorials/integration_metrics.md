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

scarf.set_verbosity('WARNING')

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
    prepend_text='orig',
    reset_cell_filter=False,
    source_column='sample_id',
    overwrite=True,
).dump()

ds = scarf.DataStore(merged, nthreads=4)
ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
ds.make_graph(feat_key='hvgs', k=11, dims=15, n_centroids=100)
ds.run_leiden_clustering(resolution=0.5)
ds.run_umap(n_epochs=100, parallel=True)
```

`source_column` records the source dataset, while `prepend_text='orig'` keeps the input labels as `orig_cluster_labels`. `reset_cell_filter=False` also preserves each input `I` filter. These columns stay aligned with the permuted count rows.

The naive merge (no batch correction) often separates by sample on UMAP. The metrics below
quantify that pattern.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='sample_id',
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='orig_cluster_labels',
)
```

## LISI

`metric_lisi` takes one or more cell-metadata label columns (batches, clusters, and similar).
It returns raw per-cell LISI values without reducing them to one benchmark score.
Default `perplexity=30` expects roughly `3 * perplexity` graph neighbors. This page builds
the graph with `k=11`, so Scarf reduces perplexity automatically and prints a warning. Scores
still describe mixing inside that smaller neighborhood. Raise `k` or lower `perplexity=` if
you want an exact match to the default.

```{code-cell} ipython3
lisi = ds.metric_lisi(label_colnames=['sample_id'])
lisi
```

## iLISI and cLISI

`metric_ilisi` summarizes batch mixing with the median and scIB scaling. Higher is better.
`metric_clisi` applies the complementary scaling to a biological label. Higher cLISI means
that neighborhoods preserve those labels. Their default perplexity is `floor(k / 3)`.
Both scores depend on graph `k`. The scIB benchmark convention compares neighborhoods with
15, 50, or 90 neighbors, so use the same `k` when comparing Scarf results to a benchmark.

```{code-cell} ipython3
ds.metric_ilisi(batch_colname='sample_id')
```

```{code-cell} ipython3
ds.metric_clisi(label_colname='orig_cluster_labels')
```

Scarf also provides a proportion-aware batch summary. It uses the mean LISI and the global
batch proportions, so it is intentionally different from iLISI.

```{code-cell} ipython3
ds.metric_proportional_batch_mixing(label_colname='sample_id')
```

## Graph connectivity, silhouette, and label concordance

Graph connectivity measures the fraction of each biological label retained in its largest
connected component on Scarf's symmetrized graph.

```{code-cell} ipython3
ds.metric_graph_connectivity(label_colname='orig_cluster_labels')
```

```{code-cell} ipython3
ds.metric_graph_silhouette(res_label='leiden_cluster')
```

```{code-cell} ipython3
ds.metric_label_concordance(
    label_columns=['RNA_leiden_cluster', 'orig_cluster_labels']
)
```

On this naive merged graph (no batch correction), batch LISI and iLISI are usually low.
Leiden vs `orig_cluster_labels` concordance can also be low when clusters split by sample.
Compare the same metrics after partial PCA or Harmony in {doc}`data_integration`.

See {ref}`integration methods guide <integration_guide>` for how these scores relate to
method choice.

## Next steps

- {doc}`data_integration`
- {doc}`choosing_integration_methods`
- {doc}`../reference/api/integration`
