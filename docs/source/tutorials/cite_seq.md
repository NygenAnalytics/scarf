---
description: Process RNA and ADT assays, then integrate their neighbourhood graphs.
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

# RNA and protein integration with CITE-seq

CITE-seq measures RNA counts and antibody-derived tags (ADT) in the same cells.
Scarf keeps both as separate assays in one store, so each modality is processed with its own normalization and graph, and the two can then be compared or merged.

This tutorial processes RNA and ADT independently, then builds one integrated graph.
It shows SNN and WNN because combining modalities is the central outcome, while method comparison and tuning remain in {doc}`multimodal_diagnostics`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- The single-assay workflow in {doc}`scrna_seq`

## What you will learn

- Open a store holding two assays and filter cells on the default assay
- Drop control antibodies from an ADT panel
- Build an ADT graph without dimension reduction
- Inspect one coordinated RNA and ADT checkpoint
- Integrate both graphs with shared nearest neighbours (SNN) and weighted nearest neighbours (WNN)

## Dataset

This page builds the multimodal store from CellRanger counts so every step is visible.
`CrH5Reader` reports both libraries in the file, and `CrToZarr` writes them as separate assays named `RNA` and `ADT`.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scarf

scarf.configure_output(level='WARNING', progress=True)

counts = scarf.cytebase.connect("scarf_docs").download(
    'tenx_8K_pbmc_citeseq/data.h5',
    destination='scarf_datasets',
)[0]

store = counts.with_name('data.zarr')
reader = scarf.CrH5Reader(str(counts))
print(reader.assayFeats)
```

```{code-cell} ipython3
scarf.CrToZarr(
    reader,
    zarr_loc=str(store),
).dump()
```

## 1. Open and filter the multimodal store

`default_assay` decides which assay unqualified calls act on.
Cell filtering and QC always run on the default assay, so set it to `RNA` here.

```{code-cell} ipython3
ds = scarf.DataStore(
    str(store),
    default_assay='RNA',
    nthreads=4
)
ds
```

The summary lists both assays with their own feature counts, and the cell count reads as active followed by total in brackets.
Every barcode and feature is still active here because no filter has run yet.
Opening computes feature detection statistics but does not filter the physical feature column `I`.
Cell metadata is shared across assays: one row per cell, whichever assay wrote the column.

```{code-cell} ipython3
ds.auto_filter_cells()
print(
    f"Active cells: {int(ds.cells.fetch_all('I').sum())}"
    f" of {ds.cells.N}"
)
```

`auto_filter_cells` models each RNA QC column as a normal distribution, takes its 1st and 99th percentiles as bounds, and marks outliers inactive in {term}`cell key` `I`.
The two figures are the QC distributions before and after that filter.
Because the key is shared, the ADT assay analyzes the same cells.

## 2. Process the RNA assay

These are the steps from {doc}`scrna_seq`, so the narrative here is brief.

```{code-cell} ipython3
hvg_ref = ds.mark_hvgs(
    min_cells=20,
    top_n=1000,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=False,
)

ds.run_normalization(features=hvg_ref)
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
Both assays are given `k=21` here.
SNN merge requires identical per-cell neighbor degree and raises if the graphs differ; WNN allows different `k` values and keeps `min(k)` neighbors per cell.
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_nCounts',
)
```

The RNA layout should recover broad PBMC structure.
Colouring by `RNA_nCounts` shows whether a dominant region aligns with low library size; that pattern suggests the shared cell selection needs more inspection.

## 3. Process the ADT assay

ADT panels hold tens of antibodies rather than thousands of genes, so they usually need no variability model, while control antibodies still have to be removed explicitly.

Scarf recognizes an assay named `ADT` as an `ADTassay`, which normalizes with a centred log ratio rather than the library-size scaling used for RNA.

```{code-cell} ipython3
ds.ADT.normMethod.__name__
```

Controls in this panel carry `control` in their name.
Other panels use other conventions, so inspect the names before choosing a pattern.

```{code-cell} ipython3
adt_panel = ds.ADT.feats.to_pandas_dataframe(['names'])
adt_panel['is_control'] = adt_panel['names'].str.contains('control')
adt_panel[adt_panel['is_control']]
```

Publish the non-control mask as an immutable feature-selection artifact:

```{code-cell} ipython3
adt_features = ds.set_feature_selection(
    from_assay='ADT',
    mask=~adt_panel['is_control'].values,
    label='adt_panel',
)
print(
    f"Selected ADT features: {int(ds.ADT.feats.fetch_all('adt_panel').sum())}"
    f" of {len(adt_panel)}"
)
```

Now build the ADT graph.
Arguments differ from the RNA chain:

- `from_assay='ADT'` targets the non-default assay on every step
- `features=adt_features` pins the exact non-control antibody selection
- `run_custom_reduction` with an identity loading matrix keeps neighbours in the normalized antibody space

```{code-cell} ipython3
normalized_adt = ds.run_normalization(
    from_assay='ADT',
    features=adt_features,
)
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
```

```{note}
A PCA of 15 components over a panel of roughly 20 antibodies would discard little and cost an extra fit, which is why an identity reduction is the sensible default for ADT.
For RNA, where thousands of genes are in play, PCA is what makes the neighbour search tractable.
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
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ADT_UMAP',
    color_by='ADT_leiden_cluster',
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='ADT_UMAP',
    color_by=['CD3_TotalSeqB', 'CD14_TotalSeqB'],
    from_assay='ADT',
    n_columns=2,
    sort_values=True,
)
```

CD3 and CD14 antibodies should mark T-cell-like and monocyte-like regions when those lineages are present.
The ADT layout should resolve protein-defined populations without being dominated by control antibodies.

## 4. Integrate the modalities

Each modality now has its own embedding and its own clusters over the same cells.
Plotting one modality's clusters on the other's layout shows where they agree.

```{code-cell} ipython3
figure, axes = plt.subplots(2, 2, figsize=(9, 8))
modality_panels = (
    ("RNA layout, RNA clusters", "RNA_UMAP", "RNA_leiden_cluster"),
    ("RNA layout, ADT clusters", "RNA_UMAP", "ADT_leiden_cluster"),
    ("ADT layout, RNA clusters", "ADT_UMAP", "RNA_leiden_cluster"),
    ("ADT layout, ADT clusters", "ADT_UMAP", "ADT_leiden_cluster"),
)
for axis, (title, layout_key, color_by) in zip(
    axes.flat, modality_panels, strict=True
):
    ds.plots.embedding(
        layout_key=layout_key,
        color_by=color_by,
        point_size=5,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
```

The panels need not match one-to-one.
They should nevertheless show related broad populations.
A modality split that is completely absent from the other can reflect real complementarity, assay noise, or a graph mismatch.
The {doc}`multimodal_diagnostics` guide evaluates those possibilities with normalized overlap and protein-versus-RNA checks.

(multimodal_integration)=

### Shared nearest neighbours

Comparing clusters is descriptive.
Integration goes further and produces a single graph, so one embedding and one set of clusters describe both modalities.

`integrate_assays` captures each named assay's current connectivity-map reference once, merges their edges, then prunes by shared nearest neighbors until each cell keeps exactly the shared input degree `nk`.

```{code-cell} ipython3
snn_graph = ds.integrate_assays(
    assays=['RNA', 'ADT'],
    label='RNA+ADT',
    method='snn',
)
```

The merged graph is stored under its `label` and returned as an exact artifact reference.
Pass it through `graph=` to downstream steps; output columns still use the integrated label as their prefix.

```{code-cell} ipython3
ds.run_umap(
    graph=snn_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
ds.run_leiden_clustering(
    graph=snn_graph,
    resolution=1.75
)
```

```{code-cell} ipython3
snn_comparison = ds.plots.embedding(
    layout_key='RNA+ADT_UMAP',
    color_by=[
        'RNA_leiden_cluster',
        'ADT_leiden_cluster',
        'RNA+ADT_leiden_cluster',
    ],
    legend_loc='on_data',
    n_columns=3,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    snn_comparison.axes.values(),
    ("RNA clusters", "ADT clusters", "SNN clusters"),
    strict=True,
):
    axis.set_title(title)
snn_comparison.figure.tight_layout()
```

The first two panels show where each modality alone would split these cells; the third shows the partition the merged graph supports.

(wnn_integration)=

### Weighted nearest neighbours

SNN treats both modalities equally and accepts two or more assays.
Weighted nearest neighbors instead learns a per-cell weight for each modality, so cells whose identity is better resolved by protein lean on the ADT graph and the rest lean on RNA.
WNN also accepts two or more assays.
This RNA and ADT workflow is its two-modality special case.
Scarf implements the affinity and per-cell weighting equations from [Hao et al., Cell 2021](https://doi.org/10.1016/j.cell.2021.04.048), with the scaling choices described below.

```{code-cell} ipython3
wnn_graph = ds.integrate_assays(
    assays=['RNA', 'ADT'],
    label='RNA+ADT_wnn',
    method='wnn',
    l2_normalize=True,
)
ds.run_umap(
    graph=wnn_graph,
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True
)
ds.run_leiden_clustering(
    graph=wnn_graph,
    resolution=1.75
)
```

The integrated artifact stores blended affinities as graph weights.
It also publishes the per-cell columns `RNA+ADT_wnn_RNA_weight` and `RNA+ADT_wnn_ADT_weight`.
The weights are non-negative and sum to one for each selected cell.

```{code-cell} ipython3
weight_columns = [
    'RNA+ADT_wnn_RNA_weight',
    'RNA+ADT_wnn_ADT_weight',
]
weight_values = np.column_stack(
    [ds.cells.fetch(column, key='I') for column in weight_columns]
)

pd.Series(
    {
        'minimum weight': float(weight_values.min()),
        'maximum weight': float(weight_values.max()),
        'mean RNA weight': float(weight_values[:, 0].mean()),
        'mean ADT weight': float(weight_values[:, 1].mean()),
        'maximum row-sum error': float(
            np.abs(weight_values.sum(axis=1) - 1).max()
        ),
    }
)
```

```{code-cell} ipython3
wnn_weights = ds.plots.embedding(
    layout_key='RNA+ADT_wnn_UMAP',
    color_by=weight_columns,
    n_columns=2,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    wnn_weights.axes.values(),
    ('RNA weight', 'ADT weight'),
    strict=True,
):
    axis.set_title(title)
wnn_weights.figure.set_size_inches(9, 4)
```

Plotting both weights on the WNN layout shows where the graph leans on RNA or ADT local neighbourhoods.

Scarf WNN is Hao-inspired, but it is not bit-identical to Seurat's `FindMultiModalNeighbors`:

- Scarf scores the union of every existing KNN row, at most the sum of their degrees, and retains the smallest input degree.
  In this RNA and ADT example that means at most `2k` candidates and `min(k_RNA, k_ADT)` output neighbours.
  Seurat normally obtains a wider candidate pool with `knn.range=200`, then retains `k.nn=20`.
  Scarf therefore cannot tune candidate pool size independently of final graph degree.
- For this two-modality example, the wider search would require two more L2-space index builds and 20 million queries at ten million cells.
  Materializing 200 candidates for each modality would hold 4 billion neighbour records and scoring work would increase by roughly tenfold.
  Scarf keeps the existing graphs to avoid that cost.
- Scarf uses the distance span from each cell's nearest to its `k`-th stored nonself neighbour as the affinity bandwidth.
  This corresponds to Seurat's supported simple-bandwidth path, not its default SNN-far bandwidth.
- Scarf builds candidates in the PCA or Harmony geometry used by each source graph, then scores them after row-wise L2 normalization by default.
  Distance after normalization is monotone with cosine distance, but the candidate ordering is not guaranteed to match a KNN index built directly in the normalized space.
  Set `l2_normalize=False` only when this difference is intentional.
  The setting is part of artifact provenance.
- Prediction means exclude the query cell in both implementations.
  Scarf's stored rows contain `k` nonself cells, while Seurat's internal `k.nn` includes a self slot.
  Public Seurat `k.nn=20` therefore averages 19 nonself neighbours, while Scarf `k=20` averages 20.
- If the source graphs use different `k`, each modality predicts from its own row and the integrated graph retains the smaller degree.

Per-cell prediction work grows quadratically with the number of modalities, while blended edge scoring is linear in the union candidate pool.
Reading neighbour-index arrays directly and preallocating outputs keeps stored input and output memory linear in cell count.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
integration_panels = (
    ("SNN", "RNA+ADT_UMAP", "RNA+ADT_leiden_cluster"),
    ("WNN", "RNA+ADT_wnn_UMAP", "RNA+ADT_wnn_leiden_cluster"),
)
for axis, (title, layout_key, color_by) in zip(
    axes, integration_panels, strict=True
):
    ds.plots.embedding(
        layout_key=layout_key,
        color_by=color_by,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
```

SNN merges edge support across two or more assays.
WNN also accepts two or more assays and adjusts their contributions cell by cell.
In this two-assay dataset, both integrated layouts should preserve broad PBMC populations while resolving differences supported by the protein panel.
Use the advanced guide to compare them quantitatively rather than choosing from UMAP appearance alone.

(hto_demultiplexing)=
## 5. HTO demultiplexing

Hashtag assignment is a separate sample-identification task, not part of the RNA/ADT integration path.
See {doc}`hto_demultiplexing`.

## Common mistakes and limitations

- Filtering cells on one assay and then comparing modalities built from different cell sets
- Integrating per-assay graphs built with different `k`
- Leaving control antibodies active in the ADT panel
- Using WNN with fewer than two assays or with graphs built over different cells
- Reading RNA and ADT clusters as interchangeable labels for the same populations
