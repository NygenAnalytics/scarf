---
description: Publication-oriented figures with scarf.plotting (embedding, dotplot, composition, export).
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

`scarf.plotting` provides the main figure types for Scarf analysis. Import the module as `splt`
and use it for new plotting code.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- A dataset with an embedding and group labels

## What you will learn

- Draw embedding, dotplot, matrixplot, composition, and distribution figures
- Control legends, themes, and point size
- Save publication-ready figures

## Dataset

```{code-cell} ipython3
from pathlib import Path

import scarf
import scarf.plotting as splt

scarf.set_verbosity('WARNING')

dataset = "bastidas-ponce_4K_pancreas-d15_rnaseq"
scarf.fetch_dataset(
    dataset_name=dataset,
    save_path="scarf_datasets",
    as_zarr=True,
)
ds = scarf.DataStore(
    str(Path("scarf_datasets") / dataset / "data.zarr"),
    nthreads=4,
    default_assay="RNA",
)
```

## Guided steps

### 1. Plot embeddings

`splt.embedding` colors cells on a layout such as UMAP. Pass a metadata column
or a gene name in `color_by`. The return value is a `PlotResult`; call
`.show()` to render it in the current notebook cell.

```{code-cell} ipython3
splt.embedding(ds, layout_key="RNA_UMAP", color_by="clusters").show();
```

Several genes become a row of panels. `NormalizationSpec(transform="log1p")`
compresses the expression scale. `sort_values=True` draws high-expressing cells
last so they sit on top of the cloud.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by=["Gcg", "Ins2", "Sst"],
    normalization=splt.NormalizationSpec(transform="log1p"),
    sort_values=True,
).show();
```

Outliers can wash out a gene UMAP. `ColorScale(quantiles=(0.0, 0.99))` sets the
color limit from the 99th percentile instead of the absolute maximum.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="Gcg",
    normalization=splt.NormalizationSpec(transform="log1p"),
    color_scale=splt.ColorScale(cmap="viridis", quantiles=(0.0, 0.99)),
    sort_values=True,
).show();
```

For large datasets, `embedding_raster` builds a pixel image from continuous
cell metadata without loading full columns into memory. Empty pixels are white
by default. It does not color by gene; use `embedding` for that.

```{code-cell} ipython3
splt.embedding_raster(
    ds,
    layout_key="RNA_UMAP",
    color_by="RNA_nCounts",
    pixels=400,
).show();
```

---

### 2. Choose a style

These options apply to embeddings (and to `unified_embedding` for mapping
layouts). Defaults aim at compact, journal-safe figures. Override them when the
default is a poor fit for your number of clusters or for the venue.

### Legends

`legend_loc="auto"` (default) picks a placement from the number of categories:

- few categories: side legend
- many categories: labels drawn on the clusters
- very many categories: no legend (label offline or subset the categories)

Force a placement when you know what you want.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="clusters",
    legend_loc="on_data",
).show();
```

### Frame and theme

`frame="minimal"` keeps a simple L-shaped edge without UMAP axis titles.
`frame="none"` removes the box for a Scanpy-like silhouette. `theme="paper"`
uses smaller fonts suited to multi-panel figures; `theme="dark"` is for dark
notebook themes.

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="clusters",
    legend_loc="on_data",
    frame="none",
    theme="paper",
).show();
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

```{code-cell} ipython3
n = len(ds.cells.active_index("I"))
ds.cells.insert(
    "demo_sample",
    [f"s{i % 8}" for i in range(n)],
    overwrite=True,
)

splt.dotplot(
    ds,
    features={"endocrine": ["Gcg", "Ins2", "Sst"]},
    group_by="clusters",
    sample_by="demo_sample",
).show();
```

A matrixplot is a plain heatmap of mean or fraction. Gene and group order are
left as you pass them; nothing is reclustered.

```{code-cell} ipython3
splt.matrixplot(
    ds,
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
    value="mean",
).show();
```

---

### 4. Plot composition

Use composition plots when you care about how cell types change across samples.
`kind="per_sample"` draws one point per sample. With subject and condition
fields, Scarf connects the same subject across conditions inside each category.

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

splt.composition(
    ds,
    category_by="clusters",
    study_design=splt.StudyDesign(
        sample_by="demo_sample",
        subject_by="demo_subject",
        condition_by="demo_condition",
    ),
    kind="per_sample",
).show();
```

---

### 5. Plot distributions

Violins (or boxes, histograms, ECDFs) are useful for QC metrics or genes split
by cluster. Groups are colored distinctly. `max_points` limits how many
individual cells are overlaid as points (`0` turns points off). Use the same
selection knobs as embeddings: `subset_by` for a boolean cell column and
`groups` to keep and order categories from `group_by`. Several gene keys share
a y-axis scale and wrap into a grid.

```{code-cell} ipython3
cluster_ids = sorted(set(ds.cells.fetch("clusters")), key=str)[:6]
splt.distribution(
    ds,
    keys=["RNA_nCounts", "RNA_nFeatures"],
    group_by="clusters",
    groups=cluster_ids,
    kind="violin",
    max_points=2000,
    seed=0,
).show();
```

---

### 6. Save figures

`PlotResult.save` writes PNG, PDF, SVG, or TIFF. The default background is
opaque white. Pass `transparent=True` when you want the figure to sit on a dark
notebook theme. Call `close()` when Scarf created the figure and you are done
with it.

```{code-cell} ipython3
from pathlib import Path

out = Path("scarf_datasets") / "plotting_showcase_embedding.png"
out.parent.mkdir(parents=True, exist_ok=True)
result = splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="clusters",
    show=False,
)
result.save(out, dpi=200)
assert out.exists()
result.close()
```

```{note}
Prefer `scarf.plotting` for new figures. `DataStore.plot_layout` and other
`DataStore.plot_*` helpers remain available compatibility APIs. Some older
tutorials still call them where behavior matches. For unified reference and
query layouts from mapping, use `splt.unified_embedding` (see
{doc}`mapping_and_label_transfer`).
```

## Common mistakes

- Passing a layout key or metadata column that is not present in the store
- Using a continuous color scale for categorical labels
- Calling `save` without closing figures in a long-running batch workflow

## Saved results

Plot functions return a `PlotResult`. `PlotResult.save` writes the requested file; plot calls do
not modify the Zarr store unless the preceding analysis code inserts metadata columns.

## Next steps

- {doc}`annotation`
- {doc}`data_organization`
- {doc}`import_and_export`
