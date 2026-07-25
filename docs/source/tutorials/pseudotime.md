---
jupytext:
  formats: ipynb,md:myst
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

# Estimating pseudotime ordering and expression dynamics

Order cells along a supervised trajectory and find genes correlated with that ordering.
For expression modules and comparisons to cluster markers, continue with
{doc}`pseudotime_modules`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and UMAP embedding

## What you will learn

- Score supervised pseudotime with source and sink clusters
- Find features correlated with the ordering
- Continue to pseudotime modules in {doc}`pseudotime_modules`

## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')
```

---
## 1) Fetch pre-analyzed data

Here we use the data from [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors.

The prepared Zarr store is available from the `scarf_docs` Cytebase catalog. It already
includes the top 2000 highly variable genes, a neighbourhood graph, and a UMAP embedding. 

```{code-cell} ipython3
scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='./scarf_datasets',
    zarr=True,
)
```

```{code-cell} ipython3
ds = scarf.DataStore(
    f"scarf_datasets/bastidas-ponce_4K_pancreas-d15_rnaseq/data.zarr",
    nthreads=4, 
    default_assay='RNA'
)
if 'RNA_paris_cluster' not in ds.cells.columns:
    ds.run_paris_clustering(n_clusters=10)
```

```{code-cell} ipython3
ds
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['RNA_paris_cluster', 'clusters'],
    legend_loc='on_data',
)
```

---
## 2) Estimate pseudotime ordering

Scarf uses a memory-efficient implementation of the [PBA algorithm](https://github.com/AllonKleinLab/PBA) ([Weinreb et al. 2018, PNAS](https://doi.org/10.1073/pnas.1714723115)) to estimate a pseudotime ordering. `run_pseudotime_scoring` works with any assay that has a neighborhood graph. The method is supervised, so provide source groups such as progenitor cells and sink groups such as differentiated cell states.

```{code-cell} ipython3
pseudotime = ds.run_pseudotime_scoring(
    source_sink_key="RNA_paris_cluster",  # Column that contains cluster information
    sources=[1],                      # Source clusters
    sinks=[3],                        # Sink clusters
)
```

By default, the calculated values are saved under **'RNA_pseudotime'**, where 'RNA' is replaced by the assay name. A companion boolean column **'RNA_pseudotime__valid'** is also written. The returned result exposes both names as `pseudotime_key` and `validity_key`. When the selected graph is fully connected, every cell is valid. If the graph has multiple components, only the largest one is scored by default. The remaining cells hold `NaN`, and downstream steps should use the validity column as `cell_key`. The UMAP below shows progression from 0 to 1.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=pseudotime.pseudotime_key,
)
```

---
## 3) Identify pseudotime correlated features

`run_pseudotime_marker_search` calculates a correlation coefficient and p-value for each selected feature against the pseudotime ordering.

```{code-cell} ipython3
markers = ds.run_pseudotime_marker_search(
    cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
)
```

The correlations and p-values are saved in feature metadata. Their generated column names are available as `markers.correlation_key` and `markers.p_value_key`. The same values, feature indices, and feature names are returned in `markers.table`.

```{code-cell} ipython3
markers.table.head()
```

---
## 4) Visualize pseudotime correlated features

This section uses the returned correlations for exploratory analysis.

The returned table can be used directly for sorting and filtering.

```{code-cell} ipython3
corr_genes_df = markers.table
```

Genes with a negative correlation decrease in expression as pseudotime progresses.

```{code-cell} ipython3
corr_genes_df.sort_values('r_value')[:15]
```

Visualize a few of these genes on the UMAP plot. Gene symbols come from the
correlation table above, not from a fixed list.

```{code-cell} ipython3
neg_genes = (
    corr_genes_df.sort_values('r_value')['feature_name']
    .head(3)
    .tolist()
)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=neg_genes,
    sort_values=True,
)
```

Genes with a positive correlation increase in expression as pseudotime progresses.

```{code-cell} ipython3
corr_genes_df.sort_values('r_value', ascending=False)[:10]
```

```{code-cell} ipython3
pos_genes = (
    corr_genes_df.sort_values('r_value', ascending=False)['feature_name']
    .head(3)
    .tolist()
)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=pos_genes,
    sort_values=True,
)
```

## Common mistakes and limitations

- Choosing source or sink clusters that do not sit at the intended ends of the trajectory
- Interpreting linear correlation as the only form of expression dynamics along pseudotime
- Ignoring `RNA_pseudotime__valid` when the graph has more than one connected component

## Saved results

Pseudotime and validity columns are written to cell metadata. Feature correlations and
p-values are stored in feature metadata.

## Further reading

- Weinreb et al. 2018, population balance analysis (PBA): https://doi.org/10.1073/pnas.1714723115
- [PBA reference implementation](https://github.com/AllonKleinLab/PBA)

## Next steps

- {doc}`pseudotime_modules`
- {doc}`fate_mapping`
- {doc}`annotation`
- {doc}`plotting`
