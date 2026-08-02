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

Use `ds.plots` for plots backed by a `DataStore`. The same store-first functions
remain available from `scarf.plotting`, which also provides reusable contracts
such as color and normalization scales.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- A dataset with an embedding and group labels

## What you will learn

- Draw embedding, dotplot, matrixplot, composition, and distribution figures
- Focus embedding regions with facets, highlights, and density contours
- Compose caller-owned panels with shared scales and legends
- Save figures with exact dimensions and provenance

## Dataset

```{code-cell} ipython3
from pathlib import Path

import matplotlib.pyplot as plt

import scarf
import scarf.plotting as splt

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
    default_assay="RNA",
)
```

## Guided steps

### 1. Plot embeddings

`ds.plots.embedding` colors cells on a layout such as UMAP. Pass a metadata column
or a gene name in `color_by`. Plot functions render in the current notebook
cell by default.

```{code-cell} ipython3
ds.plots.embedding(layout_key="RNA_UMAP", color_by="clusters");
```

Several genes become a row of panels. `NormalizationSpec(transform="log1p")`
compresses the expression scale. `sort_values=True` draws high-expressing cells
last so they sit on top of the cloud.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=["Gcg", "Ins2", "Sst"],
    normalization=splt.NormalizationSpec(transform="log1p"),
    sort_values=True,
);
```

Outliers can wash out a gene UMAP. `ColorScale(quantiles=(0.0, 0.99))` sets the
color limit from the 99th percentile instead of the absolute maximum.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="Gcg",
    normalization=splt.NormalizationSpec(transform="log1p"),
    color_scale=splt.ColorScale(cmap="viridis", quantiles=(0.0, 0.99)),
    sort_values=True,
);
```

For large datasets, `ds.plots.embedding_raster` builds a pixel image from continuous
cell metadata without loading full columns into memory. Empty pixels are white
by default. It does not color by gene; use `ds.plots.embedding` for that.

```{code-cell} ipython3
ds.plots.embedding_raster(
    layout_key="RNA_UMAP",
    color_by="RNA_nCounts",
    pixels=400,
);
```

### Facets and coordinated layouts

`facet_by` separates one layout by a categorical cell column. Use it when each
panel answers the same question for a defined group, and keep the colour scale
shared so intensities remain comparable.

```{code-cell} ipython3
shared_expression_scale = splt.ColorScale(
    cmap="magma",
    quantiles=(0.0, 0.99),
)
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="Ins2",
    facet_by="clusters",
    groups=["Alpha", "Beta", "Delta"],
    color_scale=shared_expression_scale,
    sort_values=True,
    n_columns=3,
);
```

Pass several layout keys when the same values need to be compared across
embeddings. The panels remain coordinated views of one cell table, not
independent analyses.

### Highlights and density contours

`Highlight` keeps context cells visible while emphasizing a selected group.
`DensityOverlay(statistic="mean")` adds contours around regions with high local
continuous signal.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=None,
    default_color="#bdbdbd",
    point_alpha=0.4,
    highlight=splt.Highlight(
        by="clusters",
        groups=("Beta",),
        color="#d62728",
        dim_alpha=0.12,
        size_multiplier=1.35,
        halo_width=0.4,
    ),
);
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="Ins2",
    color_scale=shared_expression_scale,
    sort_values=True,
    density_overlay=splt.DensityOverlay(
        statistic="mean",
        pixels=60,
        sigma=4.2,
        min_support=0.35,
        levels=(0.9,),
        max_hotspots=1,
    ),
);
```

Contours summarize a smoothed display layer. They do not alter stored values or
define a cluster boundary.

---

### 2. Choose a style

These options apply to embeddings. Defaults aim at compact figures. Override
them when the default is a poor fit for your number of clusters or for the
venue.

### Legends, frame, and theme

`legend_loc="auto"` (default) picks a placement from the number of categories:

- few categories: side legend
- many categories: labels drawn on the clusters
- very many categories: a wrapped side legend

`frame="minimal"` keeps a simple L-shaped edge without UMAP axis titles.
`frame="none"` removes the box for a Scanpy-like silhouette. `theme="paper"`
uses smaller fonts suited to multi-panel figures; `theme="dark"` is for dark
notebook themes.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="clusters",
    legend_loc="on_data",
    frame="none",
    theme="paper",
);
```

### Point size

Leave `point_size=None` (the default) so marker size follows the cell count.
Small datasets get larger points; dense clouds get smaller points and thinner
edges so clusters do not turn into a dark smudge. Pass an explicit `point_size`
only when you need a fixed look across figures.

---

### 3. Draw dotplots and matrixplots

A dotplot shows two summaries at once: color is mean expression in the group,
size is the fraction of cells above the detection cutoff. Pass an ordered
mapping if you want gene-group brackets. `sample_by` makes each sample
contribute equal weight instead of letting large samples dominate.

The `demo_sample` column below is synthetic demo metadata for equal sample
weighting in this showcase. It is not an experimental sample annotation.

```{code-cell} ipython3
n = len(ds.cells.active_index("I"))
ds.cells.insert(
    "demo_sample",
    [f"s{i % 8}" for i in range(n)],
    overwrite=True,
)

ds.plots.dotplot(
    features={"endocrine": ["Gcg", "Ins2", "Sst"]},
    group_by="clusters",
    sample_by="demo_sample",
);
```

A matrixplot is a heatmap of mean or fraction. Gene and group order are left as
you pass them by default. Use `feature_order` or `group_order` for explicit
orders, or `cluster_features=True` and `cluster_groups=True` to cluster them.

```{code-cell} ipython3
ds.plots.matrixplot(
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
    value="mean",
);
```

---

### 4. Plot composition

Use composition plots when you care about how cell types change across samples.
`kind="per_sample"` draws one point per sample. With subject and condition
fields, Scarf connects the same subject across conditions inside each category.

The `demo_subject` and `demo_condition` columns below are synthetic demo
metadata for this showcase. They are not experimental sample annotations.

```{code-cell} ipython3
ds.cells.insert(
    "demo_subject",
    [f"d{i % 4}" for i in range(n)],
    overwrite=True,
)
ds.cells.insert(
    "demo_condition",
    ["before" if i % 8 < 4 else "after" for i in range(n)],
    overwrite=True,
)

ds.plots.composition(
    category_by="clusters",
    study_design=splt.StudyDesign(
        sample_by="demo_sample",
        subject_by="demo_subject",
        condition_by="demo_condition",
    ),
    kind="per_sample",
);
```

---

### 5. Plot distributions

Violins (or boxes, histograms, ECDFs) are useful for QC metrics or genes split
by cluster. Groups are colored distinctly. `max_points` limits how many
individual cells are overlaid as points (`0` turns points off). Use the same
selection knobs as embeddings: `subset_by` for a boolean cell column and
`groups` to keep and order categories from `group_by`. Several gene keys share
a y-axis scale and wrap into a grid. Pass `sample_by` to plot biological-sample
summaries, or `split_by` for condition-split violins.

```{code-cell} ipython3
cluster_ids = sorted(set(ds.cells.fetch("clusters")), key=str)[:6]
ds.plots.distribution(
    keys=["RNA_nCounts", "RNA_nFeatures"],
    group_by="clusters",
    groups=cluster_ids,
    kind="violin",
    max_points=2000,
    seed=0,
);
```

Stacked violins align several marker distributions on one categorical axis.
This is useful when the question is whether a small marker panel separates the
annotated populations.

```{code-cell} ipython3
ds.plots.distribution(
    keys=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
    groups=["Alpha", "Beta", "Delta"],
    normalization=splt.NormalizationSpec(transform="log1p"),
    kind="stacked_violin",
    share_y=True,
    max_points=600,
);
```

For replicated studies, add `sample_by` to summarize biological samples rather
than displaying every cell as an independent replicate.

---

### 6. Save figures

Plot functions return a `PlotResult`. With the default `show=True`, a
Scarf-owned figure is rendered and then closed. Pass `show=False` when you need
to inspect or reuse `result.figure`, or save the figure. `PlotResult.save`
writes PNG, PDF, SVG, or TIFF. The default background is opaque white. Pass
`transparent=True` when you want the figure to sit on a dark notebook theme,
and call `close()` when you are done with an owned figure.

```{code-cell} ipython3
out = Path("scarf_datasets") / "plotting_showcase_embedding.png"
out.parent.mkdir(parents=True, exist_ok=True)
result = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="clusters",
    show=False,
)
result.save(
    out,
    dpi=200,
    exact_size=True,
    provenance_sidecar=True,
)
assert out.exists()
assert out.with_suffix(out.suffix + ".json").exists()
result.close()
```

`exact_size=True` preserves the figure's physical inch size. Set it to `False`
only when a tight crop is more important than exact dimensions.
`provenance_sidecar=True` writes the data selection, renderer, scales, and plot
settings to a sibling JSON file.

### Compose a publication-style figure

`target=` draws a plot into caller-owned Matplotlib axes. `compose_results`
collects the child results, can consolidate legends, and retains their
provenance. Here both continuous panels use the same `ColorScale`.

```{code-cell} ipython3
with splt.theme_context("paper"):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        layout="constrained",
    )
    children = {
        "annotation": ds.plots.embedding(
            layout_key="RNA_UMAP",
            color_by="clusters",
            legend_loc="on_data",
            show_legend=False,
            show_titles=False,
            target=axes[0],
            theme="paper",
            show=False,
        ),
        "expression": ds.plots.embedding(
            layout_key="RNA_UMAP",
            color_by="Ins2",
            color_scale=shared_expression_scale,
            sort_values=True,
            show_titles=False,
            target=axes[1],
            theme="paper",
            show=False,
        ),
    }
    composite = splt.compose_results(
        figure,
        children,
        panel_labels=False,
        shared_legends=True,
    )
    splt.label_panels(axes, labels=("A", "B"))
figure
```

The figure belongs to the caller because Matplotlib created it. Save through
`composite.save(...)` when provenance is needed, then close it explicitly.

```{code-cell} ipython3
composite_out = (
    Path("scarf_datasets") / "plotting_publication_composite.svg"
)
composite.save(
    composite_out,
    exact_size=True,
    provenance_sidecar=True,
)
plt.close(figure)
```

```{note}
For a fixed-reference mapping view, first persist query coordinates with
`query_ds.project_reference_embedding(...)`, then use
`query_ds.plots.mapping_projection(...)`. See
{doc}`mapping_and_label_transfer`.
```

### 7. Diagnostic plots used in workflow pages

Workflow chapters call a few standalone diagnostics:

- `DataStore.run_pca(..., show_elbow_plot=True)` plots PCA explained variance

- `scarf.plotting.graph_qc(graph)` plots degree and edge-weight distributions for a
  sparse graph from `load_graph`
- `mark_hvgs(..., show_plot=True)` or `scarf.plotting.highly_variable_features` shows the
  mean-variance relationship used for HVG selection

`marker_heatmap` chooses each group's top features by stored marker `score`,
with feature name as a deterministic tie-breaker. Adjusted p-values support
interpretation but do not control top-N selection.

See {doc}`dimensionality_reduction`, {doc}`clustering`, and {doc}`scrna_seq` for executable
examples. Keep diagnostic plots next to the analysis step that produces the values they
inspect.

## Common mistakes and limitations

- Passing a layout key or metadata column that is not present in the store
- Using a continuous color scale for categorical labels
- Calling `save` without closing figures in a long-running batch workflow
- Adding repeated UMAP panels that differ only by decoration and answer no new question
