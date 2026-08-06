---
description: Select representative cells with TopACeDo and write any cell subset to Zarr.
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

# Cell downsampling

TopACeDo selects representative cells from a graph while retaining local structure.
Use the result to create a smaller Zarr store for workflows that do not need every cell.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- The separate `topacedo` package, installed with `uv pip install topacedo`

## What you will learn

- Select topology-preserving representatives with `run_topacedo_sampler`
- Export selected cells to a new Zarr store with `SubsetZarr`

## Dataset

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd

import scarf

scarf.configure_output(level='WARNING', progress=True)
```

## 1. Fetch prepared data

TopACeDo samples from a KNN graph and needs Paris cluster labels to balance across partitions, plus the Paris dendrogram for that graph.
Arbitrary cluster partitions or Leiden-only labels are not enough.
The published PBMC store already has the Paris cut and dendrogram, so nothing has to be recomputed here.
{doc}`scrna_seq` shows how the {term}`analysis chain` is built, and {doc}`clustering` covers the Paris cut.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='tenx_5K_pbmc_rnaseq',
    zarr=True,
    destination='scarf_datasets'
)
```

```{code-cell} ipython3
ds = scarf.DataStore(f'{dataset}/data.zarr')

ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
)
```

Paris clusters on the full UMAP before downsampling.

## 2. Run the TopACeDo downsampler

Some downstream steps remain expensive on every cell.
TopACeDo selects a topology-preserving subset from Scarf's KNN graph while keeping heterogeneous regions.

```{code-cell} ipython3
ds.run_topacedo_sampler(
    cluster_key='RNA_paris_cluster',
    max_sampling_rate=0.1
)
if 'RNA_sketched' not in ds.cells.columns:
    raise RuntimeError(
        "TopACeDo did not create RNA_sketched. Verify that topacedo is installed."
    )

{
    'active cells': int(ds.cells.fetch_all('I').sum()),
    'seed cells': int(ds.cells.fetch_all('RNA_sketch_seeds').sum()),
    'selected cells': int(ds.cells.fetch_all('RNA_sketched').sum()),
}
```

Selected cells are marked `True` under `RNA_sketched`.
Seed cells used to initialize the sampler are marked under `RNA_sketch_seeds`.
Per-cluster counts show that sampling stays balanced rather than draining one partition:

```{code-cell} ipython3
sampling = ds.cells.to_pandas_dataframe(
    columns=['RNA_paris_cluster', 'RNA_sketched', 'RNA_sketch_seeds'],
    key='I',
)
cluster_sampling = sampling.groupby('RNA_paris_cluster', sort=True).agg(
    cells=('RNA_paris_cluster', 'size'),
    seeds=('RNA_sketch_seeds', 'sum'),
    selected=('RNA_sketched', 'sum'),
)
cluster_sampling['sampling_rate'] = (
    cluster_sampling['selected'] / cluster_sampling['cells']
).round(3)
cluster_sampling
```

Compare the full manifold with the selected cells side by side.
The right panel should still cover the main clusters:

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
    legend_loc='on_data',
    show_titles=False,
    target=axes[0],
    show=False,
)
axes[0].set_title('Full')
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
    subset_by='RNA_sketched',
    legend_loc='on_data',
    show_titles=False,
    target=axes[1],
    show=False,
)
axes[1].set_title('Selected')
figure.tight_layout()
figure
```

Seed cells are a smaller set used to initialize the sampler:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
    subset_by='RNA_sketch_seeds',
)
```

## 3. Inspect downsampling parameters

To place seed cells, the sampler estimates density from neighbourhood degree.
Dense regions get a sampling penalty.
Per-cell values are stored in `RNA_cell_density`.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_cell_density',
)
```

Higher `RNA_cell_density` marks denser graph neighbourhoods.
Selection rate by density quantile links that penalty to who was kept:

```{code-cell} ipython3
density = ds.cells.to_pandas_dataframe(
    columns=['RNA_cell_density', 'RNA_sketched'],
    key='I',
)
density['density_bin'] = pd.qcut(
    density['RNA_cell_density'],
    q=5,
    duplicates='drop',
)
(
    density.groupby('density_bin', observed=True)
    .agg(
        cells=('RNA_sketched', 'size'),
        selected=('RNA_sketched', 'sum'),
    )
    .assign(
        selection_rate=lambda frame: (
            frame['selected'] / frame['cells']
        ).round(3)
    )
)
```

The sampler also scores tight connectivity via mean shared nearest neighbours of each cell's neighbours.
Tight regions get a sampling reward.
Values are stored in `RNA_snn_value`.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_snn_value',
)
```

Higher `RNA_snn_value` marks more tightly connected neighbourhoods.

## 4. Export downsampled data

TopACeDo writes cell-metadata columns such as `RNA_sketched` and a sampling artifact in the source store.
It does not create a smaller counts store.
`SubsetZarr` does that: it accepts any boolean cell-metadata column, not only a TopACeDo result.
Use it for a manually annotated population, a QC selection, or the `RNA_sketched` selection below.

```{code-cell} ipython3
subset_path = f'{dataset}/subset.zarr'
writer = scarf.SubsetZarr(
    zarr_loc=subset_path,
    assays=[ds.RNA],
    cell_key='RNA_sketched',
    reset_cell_filter=False,
    overwrite_existing_file=True,
)
writer.dump()
```

The output contains selected cells from every assay listed in `assays`.
`SubsetZarr` retains all features in those assays; it is cell-selective, not feature-selective.
Use `to_anndata(feature_names=...)` when both axes need to be reduced before disk export.

Open the downsampled store as a new `DataStore`:

```{code-cell} ipython3
ds2 = scarf.DataStore(subset_path)
{
    "source cells": ds.cells.N,
    "subset cells": ds2.cells.N,
    "source RNA features": ds.RNA.feats.N,
    "subset RNA features": ds2.RNA.feats.N,
}
```

Cell metadata, including UMAP coordinates and cluster labels, is copied into the subset store, so the smaller object plots without recomputing the graph:

```{code-cell} ipython3
ds2.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
)
```

When the subset fits in memory, export to AnnData for tools in the [scverse](https://scverse.org/) ecosystem:

```{code-cell} ipython3
adata = ds2.to_anndata()
adata.shape
```

## Common mistakes

- Calling `run_topacedo_sampler` without a Paris dendrogram for the graph; Leiden-only labels or other cluster partitions alone are not enough
- Passing a cluster column that was created from a different cell subset than the graph
- Treating the sampled cells as a replacement for the full dataset in quantitative analysis
- Expecting `SubsetZarr` to remove unselected features

TopACeDo writes its selection and diagnostic columns to cell metadata, plus a sampling artifact.
`SubsetZarr` writes selected cells to a new Zarr store.
