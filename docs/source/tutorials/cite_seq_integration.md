---
description: SNN and WNN graph integration for CITE-seq RNA and ADT assays.
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

# SNN and WNN integration

Merge per-assay cell graphs from CITE-seq RNA and ADT into one multimodal graph with
SNN or WNN, then embed and cluster the result.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Python 3.12 or newer
- Complete {doc}`cite_seq` through cross-modality comparison, or let the setup below
  rebuild the RNA and ADT graphs

## What you will learn

- Integrate RNA and ADT graphs with shared nearest neighbors (SNN)
- Integrate the same assays with weighted nearest neighbors (WNN)
- Compare SNN and WNN embeddings and clusters

## Dataset

This page uses the same CITE-seq store as {doc}`cite_seq`. Both assay graphs are rebuilt
with the same `k` before integration so edge counts match.

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_8K_pbmc_citeseq',
    destination='scarf_datasets',
    zarr=True,
)

ds = scarf.DataStore(
    'scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr',
    default_assay='RNA',
    nthreads=4,
)

ds.auto_filter_cells()

ds.mark_hvgs(
    min_cells=20,
    top_n=1000,
    min_mean=-3,
    max_mean=2,
    max_var=6,
)
ds.make_graph(
    feat_key='hvgs',
    k=21,
    dims=15,
    n_centroids=100,
)
ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
)
ds.run_leiden_clustering(resolution=1)

adt_names = ds.ADT.feats.to_pandas_dataframe(['names'])['names']
is_control = adt_names.str.contains('control').values
ds.ADT.feats.update_key(~is_control, 'I')

ds.make_graph(
    from_assay='ADT',
    feat_key='I',
    k=21,
    dims=0,
    n_centroids=100,
)
ds.run_umap(
    from_assay='ADT',
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
)
ds.run_leiden_clustering(
    from_assay='ADT',
    resolution=1,
)
```

## Guided steps

### 1. Integrate assays with SNN

The per-modality KNN graphs can be merged into one multimodal graph. Scarf takes the latest
continuous-edge KNN graphs for the chosen assays, merges their edges, then prunes by shared
nearest neighbors until each cell keeps as many edges as in the individual graphs.

Integrate the *RNA* and *ADT* assays, then run UMAP and Leiden clustering on the merged graph.

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

Visualize the integrated UMAP. Color cells by modality-specific clusters and by the
integrated-graph clusters:

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
### 2. Integrate assays with WNN

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

## Common mistakes and limitations

- Building assay graphs from different cell subsets before integration
- Using WNN with anything other than two assays
- Treating RNA and ADT clusters as interchangeable without examining cross-modality agreement

## Saved results

SNN and WNN graphs are stored under their `label` values and generate corresponding UMAP and
Leiden columns (for example `RNA+ADT_UMAP` and `RNA+ADT_wnn_leiden_cluster`).

## Further reading

- Hao et al. 2021, weighted nearest neighbor analysis: https://doi.org/10.1016/j.cell.2021.04.048
- [Seurat WNN vignette](https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis)

## Next steps

- {doc}`multimodal_integration`
- {doc}`choosing_integration_methods`
- {doc}`plotting`
- {doc}`data_organization`
