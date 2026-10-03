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

# Plotting

Start with a few common plots: an embedding, a marker summary, and a QC distribution.
The later sections add options for particular questions. Most plots work with their defaults.

## Open the example data

This pancreas dataset contains an analysis saved as `docs_default`. Its live `clusters` metadata
column holds the published cell-type annotations. We use `CellField` to name that column
explicitly and avoid confusing it with the run's computed clusters.

```{code-cell} ipython3
# Manage local file and directory paths.
from pathlib import Path

# Arrange and save Matplotlib figures.
import matplotlib.pyplot as plt

# Open count stores and run Scarf analyses.
import scarf
# Use Scarf plotting options and diagnostics.
import scarf.plotting as splt

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Open the saved analysis and its exact results.
run = ds.pipeline.open(label="docs_default")
# Keep the saved UMAP coordinates for the figures.
layout = run["umap"]
# Use the published cell-type metadata for colors and groups.
cell_types = splt.CellField("clusters", label="cell type")
# Inspect the opened store's cells and features.
ds
```

## 1. Color an embedding

Pass the saved layout and the values you want to show. Here the colors identify cell types.

```{code-cell} ipython3
# Color the UMAP by the published cell-type annotations.
ds.plots.embedding(layout=layout, color_by=cell_types)
```

A gene name colors cells by expression. A list of genes produces several panels:

```{code-cell} ipython3
# Compare Gcg, Ins2, and Sst expression on separate UMAP panels.
ds.plots.embedding(layout=layout, color_by=["Gcg", "Ins2", "Sst"])
```

Compare the expression patterns with the annotated populations. The plot uses assay-normalized
expression by default. We will adjust the display scale later if a few high values hide the rest.

## 2. Summarize markers across groups

A dotplot shows two summaries: color is mean expression, and dot size is the fraction of cells
where the gene is detected.

```{code-cell} ipython3
# Compare marker expression and detection across the selected groups.
ds.plots.dotplot(
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
)
```

A matrixplot shows mean expression as a heatmap:

```{code-cell} ipython3
# Compare mean marker expression across cell types.
ds.plots.matrixplot(
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
)
```

Use `value="fraction"` to show detection rates instead. Both plots keep the supplied gene order;
`group_order` controls the group order. For `matrixplot`, add `cluster_groups=True` when you
want groups reordered by similarity; `dotplot` does not take this option.

## 3. Inspect distributions

Use distributions to see variation within groups that an average can hide.

```{code-cell} ipython3
# Compare count depth and detected genes across annotated cell types.
ds.plots.distribution(
    keys=["RNA_nCounts", "RNA_nFeatures"],
    grouping=cell_types,
)
```

The default is a violin plot. `kind="box"`, `kind="hist"`, and `kind="ecdf"` provide other views.
Use `groups=["Alpha", "Beta", "Delta"]` to focus on those cell types.
See {doc}`quality_control` for choosing thresholds from QC distributions.

## 4. Save a figure

Plots display automatically in a notebook. Set `show=False` when you want to save or change the
figure first. The returned `PlotResult` provides `save` and `close`.

```{code-cell} ipython3
# Choose a persistent directory for figures.
output_directory = Path("figures")
# Create the output directory if needed.
output_directory.mkdir(exist_ok=True)

# Create the figure without displaying it yet.
result = ds.plots.embedding(layout=layout, color_by=cell_types, show=False)
# Choose the filename for the cell-type figure.
figure_path = output_directory / "pancreas_cell_types.png"
# Save the figure to the chosen file.
result.save(figure_path)
# Close the figure after saving it.
result.close()
# Confirm the written filename and its size in bytes.
{"file": str(figure_path), "bytes": figure_path.stat().st_size}
```

The file stays in `figures` after the notebook closes. Change the extension to save PDF, SVG, or
TIFF. For publication, `dpi=300` controls raster resolution and `exact_size=True` preserves the
figure's physical dimensions. Add `provenance_sidecar=True` when you also need a JSON record of
the selection and plot settings.

## Optional: make expression easier to see

Use these display changes when the default gene panels are hard to read:

- `NormalizationSpec(transform="log1p")` compresses the expression scale.
- `sort_values=True` draws high-expressing cells last so other cells do not cover them.
- `ColorScale(quantiles=(0.0, 0.99))` caps the color range at the 99th percentile.

```{code-cell} ipython3
# Show log-transformed Ins2 expression with the upper color range clipped.
ds.plots.embedding(
    layout=layout,
    color_by="Ins2",
    normalization=splt.NormalizationSpec(transform="log1p"),
    sort_values=True,
    color_scale=splt.ColorScale(quantiles=(0.0, 0.99)),
)
```

These choices affect the display, not the saved counts. Clipping the color range makes all values
above the limit share the same color, so report that choice when presenting a figure.

## Optional: focus on selected groups

Facets show the same layout in separate panels. Here each panel contains one annotated cell type,
colored by Ins2 expression. The panels use a shared expression scale.

```{code-cell} ipython3
# Show Ins2 expression separately in Alpha, Beta, and Delta cells.
ds.plots.embedding(
    layout=layout,
    color_by="Ins2",
    facet_by=cell_types.key,
    groups=["Alpha", "Beta", "Delta"],
)
```

A highlight keeps the other cells visible for context:

```{code-cell} ipython3
# Highlight Beta cells while retaining the other cells for context.
ds.plots.embedding(
    layout=layout,
    color_by=None,
    highlight=splt.Highlight(by=cell_types.key, groups=("Beta",)),
)
```

For several marker distributions, stacked violins offer another compact view:

```{code-cell} ipython3
# Compare three endocrine markers with stacked violin plots.
ds.plots.distribution(
    keys=["Gcg", "Ins2", "Sst"],
    grouping=cell_types,
    groups=["Alpha", "Beta", "Delta"],
    kind="stacked_violin",
)
```

For real replicated studies, `sample_by` summarizes samples. Composition plots can compare cell
type fractions across samples, and a `StudyDesign` can connect paired observations.
See {doc}`condition_comparisons` for examples with actual study metadata.

## Optional: compose panels and choose a style

Pass Matplotlib axes as `target` when you need control over a figure's arrangement:

```{code-cell} ipython3
# Create axes for the comparison panels.
figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
# Draw each result on its comparison axes.
for axis, field, title in zip(
    axes, (cell_types, "Ins2"), ("Cell types", "Ins2"), strict=True
):
    # Draw cell-type labels or Ins2 expression in the corresponding panel.
    ds.plots.embedding(
        layout=layout,
        color_by=field,
        target=axis,
        show_titles=False,
        show=False,
    )
    # Label the panel with the result it shows.
    axis.set_title(title)
# Display the completed comparison figure.
figure
```

The figure belongs to you because you created its axes. Save it with Matplotlib and close it
afterward:

```{code-cell} ipython3
# Choose the filename for the composed figure.
panel_path = output_directory / "pancreas_panels.pdf"
# Save the figure to the chosen file.
figure.savefig(panel_path)
# Close the figure after saving it.
plt.close(figure)
# Confirm the written filename and its size in bytes.
{"file": str(panel_path), "bytes": panel_path.stat().st_size}
```

Scarf's default theme suits notebooks. Use `theme="paper"` for smaller labels or
`theme="dark"` for a dark background. Leave point sizes and legend placement at their defaults
unless they obscure the data. `legend_loc="on_data"` puts category labels on the embedding.
For composed figures that need a combined provenance record, see `scarf.plotting.compose_results`
in the {doc}`../reference/api/plotting` reference.

## Optional: plot large datasets as pixels

`embedding_raster` summarizes a continuous metadata column into pixels. It avoids loading the
full column into memory and is useful when a scatter plot has too many overlapping points.

```{code-cell} ipython3
# Summarize count depth into pixels on the embedding.
ds.plots.embedding_raster(layout=layout, color_by="RNA_nCounts")
```

This small dataset illustrates the call; the memory benefit matters on larger datasets.
Use `embedding` for gene expression. Empty raster pixels are white by default.

## Where to find analysis diagnostics

Keep diagnostics near the analysis they help evaluate:

- {doc}`feature_selection`: the mean-variance plot used to select genes.
- {doc}`graph_construction`: graph degree and edge-weight distributions.
- {doc}`dimensionality_reduction`: PCA and layout comparisons.
- {doc}`clustering`: membership strength and cluster relationships.
- {doc}`annotation`: marker heatmaps and cell-type evidence.

Before interpreting a figure, check its cells, grouping, and expression scale. Add a panel when
it answers a new question, rather than only changing the decoration.
