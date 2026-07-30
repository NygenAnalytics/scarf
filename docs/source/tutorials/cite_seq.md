---
description: CITE-seq analysis in Scarf covering per-assay processing and multimodal graph integration.
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

CITE-seq measures RNA counts and antibody-derived tags (ADT) in the same cells. Scarf keeps
both as separate assays in one store, so each modality is processed with its own
normalization and graph, and the two can then be compared or merged.

This chapter processes RNA and ADT independently, compares what each modality sees, and
merges the two graphs with SNN and WNN.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- The single-assay workflow in {doc}`scrna_seq`

## What you will learn

- Open a store holding two assays and filter cells on the default assay
- Drop control antibodies from an ADT panel
- Build an ADT graph without dimension reduction
- Compare RNA and ADT clusters against each other
- Merge both graphs with shared nearest neighbors (SNN) and weighted nearest neighbors (WNN)

## Dataset

`tenx_8K_pbmc_citeseq` is distributed as a prepared Zarr store with the assays already named
`RNA` and `ADT`.

```{code-cell} ipython3
import numpy as np
import scarf

scarf.configure_output(level='WARNING', progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_8K_pbmc_citeseq',
    destination='scarf_datasets',
    zarr=True,
)
```

## 1) Open the multimodal store

`default_assay` decides which assay unqualified calls act on. Cell filtering and QC always
run on the default assay, so set it to `RNA` here.

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    default_assay='RNA',
    nthreads=4
)
ds
```

The summary lists both assays with their own feature counts, and the cell count reads as
active followed by total in brackets. This prepared store already carries a cell selection
and previously computed columns, which is why the two numbers differ before any filtering
happens here. Cell metadata is shared across assays: one row per cell, whichever assay wrote
the column.

```{code-cell} ipython3
ds.auto_filter_cells()
```

`auto_filter_cells` models each RNA QC column as a normal distribution, takes its 1st and
99th percentiles as bounds, and marks outliers inactive in cell key `I`. The two figures are
the QC distributions before and after that filter. Because the key is shared, the ADT assay
analyzes the same cells.

## 2) Process the RNA assay

These are the steps from {doc}`scrna_seq`, so the narrative here is brief.

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=20,
    top_n=1000,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=False,
)

ds.run_normalization(feat_key='hvgs')
ds.run_pca(dims=15)
ds.build_embedding_initialization()
ds.build_ann_index()
ds.query_neighbors(k=21)
ds.build_connectivity_map()

ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
ds.run_leiden_clustering(resolution=1)
ds.load_graph()
```

```{note}
Both assays are given `k=21` here. `integrate_assays` later merges the two graphs, and
matching `k` keeps one modality from contributing far more edges than the other.
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

## 3) Process the ADT assay

ADT panels hold tens of antibodies rather than thousands of genes, which changes two things:
there is no feature selection step, and control antibodies have to be removed by hand.

Scarf recognizes an assay named `ADT` as an `ADTassay`, which normalizes with a centred log
ratio rather than the library-size scaling used for RNA.

```{code-cell} ipython3
ds.ADT.normMethod.__name__
```

Controls in this panel carry `control` in their name. Other panels use other conventions, so
inspect the names before choosing a pattern.

```{code-cell} ipython3
adt_panel = ds.ADT.feats.to_pandas_dataframe(['names'])
adt_panel['is_control'] = adt_panel['names'].str.contains('control')
adt_panel
```

`update_key` takes a boolean array and marks features `False` as inactive, so pass the
inverse of the control flag.

```{code-cell} ipython3
ds.ADT.feats.update_key(~adt_panel['is_control'].values, 'I')
print(
    f"Active ADT features: {int(ds.ADT.feats.fetch_all('I').sum())}"
    f" of {len(adt_panel)}"
)
```

Now build the ADT graph. Arguments differ from the RNA chain:

- `from_assay='ADT'` targets the non-default assay on every step
- `feat_key='I'` uses every active antibody, since there is no feature selection column
- `run_custom_reduction` with an identity loading matrix keeps neighbours in the
  normalized antibody space (the former `make_graph(..., dims=0)` behaviour).
  `run_pca` no longer accepts `dims=0`

```{code-cell} ipython3
normalized_adt = ds.run_normalization(from_assay='ADT', feat_key='I')
n_adt_features = int(ds.load_artifact(normalized_adt)['data'].shape[1])
ds.run_custom_reduction(
    np.eye(n_adt_features, dtype=np.float64),
    normalized_adt,
    from_assay='ADT',
)
ds.build_embedding_initialization(from_assay='ADT', n_centroids=100)
ds.build_ann_index(from_assay='ADT')
ds.query_neighbors(from_assay='ADT', k=21)
ds.build_connectivity_map(from_assay='ADT')
ds.load_graph(from_assay='ADT')
```

```{note}
A PCA of 15 components over a panel of roughly 20 antibodies would discard little and cost
an extra fit, which is why an identity reduction is the sensible default for ADT. For RNA,
where thousands of genes are in play, PCA is what makes the neighbour search tractable.
```

UMAP and clustering take the same `from_assay` argument and write assay-prefixed columns.

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
sorted(c for c in ds.cells.columns if c.startswith('ADT_'))
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ADT_UMAP',
    color_by='ADT_leiden_cluster',
)
```

## 4) Compare the two modalities

Each modality now has its own embedding and its own clusters over the same cells. Plotting
one modality's clusters on the other's layout shows where they agree.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by=['ADT_leiden_cluster', 'RNA_leiden_cluster'],
    n_columns=2,
    point_size=5,
    legend_loc='on_data',
)
```

A cross tabulation counts cells shared by each pair of clusters, with RNA clusters on rows
and ADT clusters on columns.

```{code-cell} ipython3
import pandas as pd

overlap = pd.crosstab(
    ds.cells.fetch('RNA_leiden_cluster'),
    ds.cells.fetch('ADT_leiden_cluster')
)
overlap
```

One way to summarize that table is to ask, for each ADT cluster, what share of its cells
fall in a single RNA cluster. Values near 100 mean the ADT cluster maps onto one
transcriptional population.

```{code-cell} ipython3
(100 * overlap.max() / overlap.sum()).round(1).sort_values(ascending=False)
```

Individual antibodies and their coding genes can be placed on either layout. CD16 protein
and its gene `FCGR3A` are a useful pair to check.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by='CD16_TotalSeqB',
    from_assay='ADT',
    n_columns=2,
    point_size=5,
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=['RNA_UMAP', 'ADT_UMAP'],
    color_by='FCGR3A',
    from_assay='RNA',
    n_columns=2,
    point_size=5,
)
```

Protein signal is usually smoother than the matching transcript, which is one reason to
combine the modalities rather than pick one.

(multimodal_integration)=

## 5) Merge the graphs with SNN

Comparing clusters is descriptive. Integration goes further and produces a single graph, so
one embedding and one set of clusters describe both modalities.

`integrate_assays` takes the latest graph of each named assay, merges their edges, then
prunes by shared nearest neighbors until each cell keeps about as many edges as it had in
the per-assay graphs.

```{code-cell} ipython3
ds.integrate_assays(
    assays=['RNA', 'ADT'],
    label='RNA+ADT',
    method='snn',
)
```

The merged graph is stored under its `label`. Downstream steps reach it through
`integrated_graph` instead of `from_assay`, and write columns using the same label as prefix.

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
sorted(c for c in ds.cells.columns if c.startswith('RNA+ADT'))
```

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

The first two panels show where each modality alone would split these cells; the third shows
the partition the merged graph supports.

(wnn_integration)=

## 6) Merge the graphs with WNN

SNN treats both modalities equally and accepts two or more assays. Weighted nearest
neighbors instead learns a per-cell weight for each modality, so cells whose identity is
better resolved by protein lean on the ADT graph and the rest lean on RNA. WNN takes exactly
two assays.

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

```{code-cell} ipython3
ds.plots.embedding(
    layout_key=['RNA+ADT_UMAP', 'RNA+ADT_wnn_UMAP'],
    color_by=['RNA+ADT_leiden_cluster', 'RNA+ADT_wnn_leiden_cluster'],
    n_columns=2,
    legend_loc='on_data',
)
```

Reach for WNN when one modality is noticeably sparser or noisier than the other, and for SNN
when the modalities are comparable or when more than two are involved.

## HTO demultiplexing

`DataStore.mark_hto_identities` assigns hashtag identities when an HTO assay is present
(default assay name `HTO`). No public HTO dataset is in the Scarf catalog yet, so this page
cannot demonstrate it; see {doc}`../reference/api/datastore` for the signature.

## Common mistakes and limitations

- Filtering cells on one assay and then comparing modalities built from different cell sets
- Integrating per-assay graphs built with different `k`
- Leaving control antibodies active in the ADT panel
- Using WNN with anything other than two assays
- Reading RNA and ADT clusters as interchangeable labels for the same populations

## Summary of saved results

| Kind | Keys / location |
|---|---|
| RNA embedding and clusters | `RNA_UMAP1`, `RNA_UMAP2`, `RNA_leiden_cluster` |
| ADT embedding and clusters | `ADT_UMAP1`, `ADT_UMAP2`, `ADT_leiden_cluster` |
| Active ADT antibodies | feature key `I` in `ds.ADT.feats` |
| SNN integration | `RNA+ADT_UMAP1/2`, `RNA+ADT_leiden_cluster` |
| WNN integration | `RNA+ADT_wnn_UMAP1/2`, `RNA+ADT_wnn_leiden_cluster` |

## Further reading

- Stoeckius et al. 2017, CITE-seq: https://doi.org/10.1038/nmeth.4380
- Hao et al. 2021, weighted nearest neighbor analysis: https://doi.org/10.1016/j.cell.2021.04.048
- [Seurat WNN vignette](https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis)

## Next steps

- {doc}`plotting`
- {doc}`annotation`
- {doc}`data_organization`
- {doc}`data_integration`
