---
description: Build, combine, and save analysis figures with Scarf's plotting API.
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
(plotting_showcase)=

# Core plotting features

SCARF ships one plotting surface for the whole analysis through `scarf.plotting`, imported as `splt`. SCARF has a variety of plotting features available that compare against existing tools like Scanpy & Seurat. The most commonly viewed plots are ones that involve visualizing the 2D embeddings of the cells and their layout: for example, comparing the output of a UMAP or t-SNE colored by clusters, a gene, or any cell field. When a scatter plot carries too many overlapping points that convolute the expression of what you see, you can rasterize the image to instead see a less complex visual summary of the data.

You can also summarize the information about gene expression by transforming the information into compact tables. For example, dot plots and matrix plots compare expression and detection across groups, and a marker heatmap can show a large amount of information for multiple genes. Plotting the distributions of quality control metrics, or even gene expression, can allow you to diagnose multiple different observations you encounter during the analysis process. Distribution plots render violin, box, histogram, ecdf, or stacked-violin views for one or more columns, and standalone diagnostics cover quality control, graph structure, and feature selection.

Going further, if you want to see cluster relationships, multimodal signals, and trajectories, these branches all have their own style of figures. Cluster trees show a Paris hierarchy, cluster connectivity shows how clusters link across a graph, modality weights reveal the RNA and protein weights behind a WNN graph, and pseudotime heatmaps show gene dynamics along a pseudotime ordering.

Reference mapping and study-level comparison round out the broad overview of the plotting features. You cap map scores, together with their supporting evidence, confusion, and calibration views along with composition plots to compare cell type fractions across samples, including paired study designs.

While not all of these features are outlined in the tutorial today, more information about each and every one of these plotting options can be found in the SCARF API reference.

## Open the example data

For the purposes of this tutorial, we will be using an already analyzed dataset. It already contains the cluster information and the corresponding cell types, thus no recomputation of this is required.

```{code-cell} ipython3
from pathlib import Path

import matplotlib.pyplot as plt

import scarf
import scarf.plotting as splt

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
layout = run["umap"]
cell_types = splt.CellField("clusters", label="cell type")
ds
```

## Color an embedding

Pass the saved layout and the values you want to color for; in this graph, the colors identify specific cell types from this annotated dataset.

```{code-cell} ipython3
ds.plots.embedding(layout=layout, color_by=cell_types)
```

The gene name colors cells on the UMAP by expression, with a list of genes produces several UMAPs with each individual genes' expression.

```{code-cell} ipython3
ds.plots.embedding(layout=layout, color_by=["Gcg", "Ins2", "Sst"])
```

Compare the expression patterns with the annotated populations. The plot uses assay-normalized expression by default, which is often the scaled and then log1p normalized expression.

If you need to change the scale and make changes to how you visualize information on the UMAP, then you can parse parameters like `NormalizationSpec(transform="log1p")` to modulate the expression scale, and `sort_values=True` to draw in high-expressing cells last so other cells do not cover them.

```{code-cell}
ds.plots.embedding(
    layout=layout,
    color_by="Ins2",
    normalization=splt.NormalizationSpec(transform="log1p"),
    sort_values=True,
    color_scale=splt.ColorScale(quantiles=(0.0, 0.99)),
)
```

## Summarize markers across groups

To summarize the expression of a a marker gene across clusters for a purpose such as a annotation, we can use a dotplot. The dotplots generally follow the format of the color showing mean expression, and the dot size showing the fraction of cells in which the gene is detected.

```{code-cell} ipython3
ds.plots.dotplot(
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
)
```

An alternative way to also visualize this information is through the use of a matrixplot, which shows the mean expression of a gene as a heatmap.

```{code-cell} ipython3
ds.plots.matrixplot(
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
)
```

If you would like to visualize the amount of cells that express a gene instead through a matrixplot, then simply pass `value="fraction"` to show the detection rates of a gene. Both plots keep the gene order you supply. The `group_order` argument sets the left to right order of the groups on the axis, so you can arrange the clusters in any sequence you choose; when you leave it out, the groups fall back to their stored display order or a natural sorted order.

For `matrixplot`, `cluster_groups=True` is an alternative to a fixed order: it reorders the groups by hierarchical clustering, placing groups with similar profiles next to each other and drawing a dendrogram above them, which lets the layout reveal relationships instead of following a preset sequence. `dotplot` does not take `cluster_groups`, so its group order is controlled only by `group_order`.

## Inspect distributions

During the quality control process, it is often critical to visualize the distribution of the RNA counts in respect to the cells, and the same with the features per cell.  We can plot the distributions to see variation within groups that an average can hide.

```{code-cell} ipython3
ds.plots.distribution(
    keys=["RNA_nCounts", "RNA_nFeatures"],
    grouping=cell_types,
)
```

The default is a violin plot, but you can also plot box plots with `kind="box"`, histograms with`kind="hist"`, and `kind="ecdf"` provide other views.

If you want to plot the distribution of multiple marker genes in specific cell types, we can use stacked violin plots.

```{code-cell}
ds.plots.distribution(
    keys=["Gcg", "Ins2", "Sst"],
    grouping=cell_types,
    groups=["Alpha", "Beta", "Delta"],
    kind="stacked_violin",
)
```

Modifying the input for `grouping=cell_types` to something like your clusters input can make the input the clusters instead of the cell types, and modifying the input of `groups` can allow you to specify the subset of data that you want to visualize the distribution for.

If you want to see the expression of an individual gene across various different cell types, than you can also use a violin plot.

```{code-cell}
ds.plots.distribution(
    keys=["Sst"],
    grouping=cell_types,
    kind="violin",
)
```

If a box plot or a histogram better suits your visual taste, then simply change the input of `kind` with the available options we discuss above.

## Saving a figure

Since plots display automatically in a .ipynb notebook, you simply need to pass the `result.save` line.

```{code-cell} ipython3
output_directory = Path("figures")
output_directory.mkdir(exist_ok=True)

result = ds.plots.embedding(layout=layout, color_by=cell_types, show=False)
figure_path = output_directory / "pancreas_cell_types.png"
result.save(figure_path)
result.close()
{"file": str(figure_path), "bytes": figure_path.stat().st_size}
```

The file stays in `figures` after the notebook closes. If you need the figure to follow a specific extension, then you can save it as a PDF, SVG, or TIFF. For publication, if you need a higher quality, parse `dpi=300`, which controls raster resolution.  `exact_size=True` preserves the figure's physical dimensions. Add `provenance_sidecar=True` when you also need a JSON record of the selection and plot settings.

# Plotting Extensions

## Focus on selected groups

Facets show the same layout in separate panels; here each panel contains one annotated cell type,
colored by Ins2 expression. The panels use a shared expression scale so you can compare the expression across the figures.

```{code-cell} ipython3
ds.plots.embedding(
    layout=layout,
    color_by="Ins2",
    facet_by=cell_types.key,
    groups=["Alpha", "Beta", "Delta"],
)
```

A highlight keeps the other cells visible for context:

```{code-cell} ipython3
ds.plots.embedding(
    layout=layout,
    color_by=None,
    highlight=splt.Highlight(by=cell_types.key, groups=("Beta",)),
)
```

## Compose panels and choose a style

If you want to compose the plots in a specific way, you can also parse information from the matplotlib library, such as axes here. If you want to modify the arrangement of the panels in a custom way, pass the axes as `target`:

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
for axis, field, title in zip(
    axes, (cell_types, "Ins2"), ("Cell types", "Ins2"), strict=True
):
    ds.plots.embedding(
        layout=layout,
        color_by=field,
        target=axis,
        show_titles=False,
        show=False,
    )
    axis.set_title(title)
figure
```

The figure is now shaped this way because you created its axes; save it with matplotlib and close it once done

```{code-cell} ipython3
panel_path = output_directory / "pancreas_panels.pdf"
figure.savefig(panel_path)
plt.close(figure)
{"file": str(panel_path), "bytes": panel_path.stat().st_size}
```

Scarf's default theme suits notebooks in dark theme for the analysis; If you are not opparating in this format, you can use `theme="dark"` for a dark background. If you need smaller labels, you can parse  `theme="paper"`. Leave point sizes and legend placement at their defaults
unless they obscure the data; if these need to be fixed, then pass information with matplotlib. Furthermore, parsing `legend_loc="on_data"` puts category labels on the embedding.

```{code-cell} ipython3
ds.plots.embedding(layout=layout, color_by=cell_types)
ds.plots.embedding(layout=layout, color_by=cell_types, legend_loc="on_data")
```

## Plot large datasets as pixels

`embedding_raster` summarizes a continuous metadata column into pixels instead of large splots of each individual cell. It also avoids loading the full column into memory, and is useful when a scatter plot has too many overlapping points.

```{code-cell} ipython3
ds.plots.embedding_raster(layout=layout, color_by="RNA_nCounts")
```

Just for reference, empty raster pixels are white by default.

## Where to find analysis diagnostics

Keep each diagnostic close to the analysis it helps evaluate.

- {doc}`feature_selection` shows the mean-variance plot that guides how many genes to keep before a graph is built.
- {doc}`graph_construction` reports graph degree and edge-weight distributions, so you can confirm the neighborhood graph is connected before you cluster.
- {doc}`dimensionality_reduction` compares PCA and layout choices, which is where you judge how many components carry real structure.
- {doc}`clustering` covers membership strength and cluster relationships, helping you decide whether a cluster is a stable group or a boundary artifact.
- {doc}`annotation` presents marker heatmaps and cell-type evidence, which is how you check that a cluster label matches its biology.

Before interpreting a figure, check its cells, grouping, and expression scale. Add a panel when
it answers a new question, rather than only changing the decoration.
