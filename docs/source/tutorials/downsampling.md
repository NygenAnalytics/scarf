---
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

TopACeDo selects representative cells from a graph while retaining local structure. Use the
result to create a smaller Zarr store for workflows that do not need every cell.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- The separate `topacedo` package, installed with `uv pip install topacedo`

## What you will learn

- Cluster cells, then select representatives with TopACeDo
- Select representative cells with `run_topacedo_sampler`
- Export the selected cells to a new Zarr store

## Dataset

```{code-cell} ipython3
import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')
scarf.__version__
```

## Guided steps

### 1. Fetch prepared data

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='tenx_5K_pbmc_rnaseq',
    as_zarr=True,
    save_path='scarf_datasets'
)
```

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr')

ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
ds.make_graph(feat_key='hvgs', k=11, dims=15)
ds.run_clustering(n_clusters=15)
ds.run_umap(n_epochs=250, parallel=True)

splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cluster',
)
```

---
### 2. Run the TopACeDo downsampler

+++

UMAP, clustering, and marker identification describe cellular diversity, but some downstream
analyses remain expensive on every cell. Scarf can select a topology-preserving subset from
the cell-neighborhood graph while retaining heterogeneous regions.

TopACeDo uses Scarf's KNN graph to perform manifold-preserving subsampling. Invoke it from
the `DataStore`:

```{code-cell} ipython3
ds.run_topacedo_sampler(
    cluster_key='RNA_cluster',
    max_sampling_rate=0.1
)
if 'RNA_sketched' not in ds.cells.columns:
    raise RuntimeError(
        "TopACeDo did not create RNA_sketched. Verify that topacedo is installed."
    )
```

As a result of subsampling the subsampled cells are marked True under the cell metadata column `RNA_sketched`. We can visualize these cells with `splt.embedding` and `subset_by`

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cluster',
    subset_by='RNA_sketched',
)
```

It may also be interesting to visualize the cells that were marked as `seed cells` used when PCST was run. These cells are marked under the column `RNA_sketch_seeds`.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cluster',
    subset_by='RNA_sketch_seeds',
)
```

---
### 3. Inspect downsampling parameters

+++

To identify the seed cells, the subsampling algorithm calculates cell densities based on neighbourhood degrees. Regions of higher cell density get a sampling penalty. The neighbourhood degree of individual cells are stored under the column `RNA_cell_density`.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_cell_density',
)
```

The downsampling algorithm also identifies regions of the graph where cells form tightly connected groups by calculating mean shared nearest neighbours of each cell's neighbours. The tightly connected regions get a sampling award. These values can be accessed from under the cell metadata column `RNA_snn_value`.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by='RNA_snn_value',
)
```

---
### 4. Export downsampled data

+++

TopACeDo marks representative cells but leaves the source store unchanged. Use `SubsetZarr`
to create a new Zarr store containing only the selected cells.

```{code-cell} ipython3
writer = scarf.SubsetZarr(
    zarr_loc='scarf_datasets/tenx_5K_pbmc_rnaseq/subset.zarr',
    assays=[ds.RNA],
    cell_key='RNA_sketched',
    reset_cell_filter=False
)
writer.dump()
```

The downsampled dataset can be loaded as a new DataStore

```{code-cell} ipython3
ds2 = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/subset.zarr')
```

```{code-cell} ipython3
ds2
```

It is expected the downsampled dataset will be small enough to fit in memory. Here it is exported
to AnnData for downstream analysis in the [scverse](https://scverse.org/) ecosystem.

```{code-cell} ipython3
adata = ds2.to_anndata()
```

```{code-cell} ipython3
adata
```

## Common mistakes

- Calling `run_topacedo_sampler` before clustering; pass a `cluster_key` column with cell partitions
- Passing a cluster column that was created from a different cell subset than the graph
- Treating the sampled cells as a replacement for the full dataset in quantitative analysis

## Saved results

TopACeDo writes `RNA_sketched`, `RNA_sketch_seeds`, `RNA_cell_density`, and `RNA_snn_value` to
cell metadata. `SubsetZarr` writes the selected cells to `subset.zarr`.

## Next steps

- {doc}`pseudotime`
- {doc}`import_and_export`
- {doc}`plotting`
