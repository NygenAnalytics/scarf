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

- Select topology-preserving representatives with `run_topacedo_sampler`
- Export selected cells to a new Zarr store with `SubsetZarr`

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')
```

## Guided steps

### 1. Fetch prepared data

```{code-cell} ipython3
scarf.cytebase.connect("scarf_docs").download_dataset(
    name='tenx_5K_pbmc_rnaseq',
    zarr=True,
    destination='scarf_datasets'
)
```

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr')

ds.mark_hvgs(min_cells=20, top_n=500, show_plot=False)
ds.make_graph(feat_key='hvgs', k=11, dims=15)
ds.run_paris_clustering(n_clusters=15)
ds.run_umap(n_epochs=250, parallel=True)

ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
)
```

Paris clusters on the full UMAP before downsampling.

---
### 2. Run the TopACeDo downsampler

+++

Some downstream steps remain expensive on every cell. TopACeDo selects a topology-preserving
subset from Scarf's KNN graph while keeping heterogeneous regions.

```{code-cell} ipython3
ds.run_topacedo_sampler(
    cluster_key='RNA_paris_cluster',
    max_sampling_rate=0.1
)
if 'RNA_sketched' not in ds.cells.columns:
    raise RuntimeError(
        "TopACeDo did not create RNA_sketched. Verify that topacedo is installed."
    )
```

Selected cells are marked `True` under `RNA_sketched`. Plot them with `subset_by`:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
    subset_by='RNA_sketched',
)
```

The subset should still cover the main clusters on the UMAP.

Seed cells used for PCST are marked under `RNA_sketch_seeds`:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_paris_cluster',
    subset_by='RNA_sketch_seeds',
)
```

Seed cells are a smaller set used to initialize the sampler.

---
### 3. Inspect downsampling parameters

+++

To place seed cells, the sampler estimates density from neighbourhood degree. Dense regions
get a sampling penalty. Per-cell values are stored in `RNA_cell_density`.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_cell_density',
)
```

Higher `RNA_cell_density` marks denser graph neighbourhoods.

The sampler also scores tight connectivity via mean shared nearest neighbours of each cell's
neighbours. Tight regions get a sampling reward. Values are stored in `RNA_snn_value`.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_snn_value',
)
```

Higher `RNA_snn_value` marks more tightly connected neighbourhoods.

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

Open the downsampled store as a new `DataStore`:

```{code-cell} ipython3
ds2 = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/subset.zarr')
```

```{code-cell} ipython3
ds2
```

When the subset fits in memory, export to AnnData for tools in the
[scverse](https://scverse.org/) ecosystem:

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

## Further reading

- [TopACeDo](https://github.com/parashardhapola/topacedo)
- [Seurat sketch-based analysis](https://satijalab.org/seurat/articles/seurat5_sketch_analysis) (analogous practice for large datasets; not equivalent to TopACeDo)

## Next steps

- {doc}`import_and_export`
- {doc}`data_organization`
- {doc}`plotting`
