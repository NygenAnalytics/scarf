---
description: Smooth sparse expression over a neighbourhood graph and compare it with observed values.
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

(imputation)=

# Imputation by graph diffusion

Dropout makes observed single-cell expression sparse.
Scarf can diffuse a feature over the neighbourhood graph to reveal coherent regional patterns.
Imputation is a visualization and exploratory-analysis aid.
It does not create new molecular observations and should not replace counts in differential expression.

## 1. Standalone setup

Diffusion needs a neighbourhood graph.
The catalog PBMC store supplies counts, literal metadata, and the UMAP used below.
This setup mounts those inputs into a temporary writable analysis store and builds a new explicit graph lineage locally.
The catalog snapshot is structurally repacked inside the temporary directory first so the mounted counts satisfy the current RNA layout.

```{code-cell} ipython3
from tempfile import TemporaryDirectory

import pandas as pd

import scarf
import scarf.plotting as splt
from scarf.tools.repack_zarr import repack_store

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
repacked_counts = f"{analysis_directory.name}/counts.zarr"
repack_store(
    f"{dataset}/data.zarr",
    repacked_counts,
    nthreads=2,
)
ds = scarf.mount_datastore(
    repacked_counts,
    at=f"{analysis_directory.name}/analysis.zarr",
    nthreads=4,
    default_assay="RNA",
)
preparation = ds.pipeline.run(
    filtering=False,
    hvg_count=2000,
    pca_dims=15,
    neighbors_k=11,
    umap=False,
    leiden=False,
    cell_cycle=False,
    paris=False,
    doublets=False,
    markers=False,
)
graph = preparation["connectivity_map"]
```

## 2. Diffuse one feature

`t` controls diffusion depth.
A larger value mixes information over more graph steps and can erase real boundaries.

```{code-cell} ipython3
diffusion_operators = {
    t: ds.run_diffusion_operator(graph, t=t)
    for t in (1, 2, 4)
}
diffused_by_t = {
    t: ds.get_imputed(feature_name="CD4", diffusion=diffusion)
    for t, diffusion in diffusion_operators.items()
}
for t, values in diffused_by_t.items():
    ds.cells.insert(
        f"CD4_imputed_t{t}",
        values,
        overwrite=True,
    )
```

```{code-cell} ipython3
observed = ds.get_cell_vals(from_assay="RNA", cell_key="I", k="CD4")
cd4_series = {
    "Observed CD4": observed,
    **{f"Diffusion t={t}": values for t, values in diffused_by_t.items()},
}
cd4_summary = pd.DataFrame(
    {
        label: {
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "zero_fraction": float((vals == 0).mean()),
            "filled_zeros": int(((observed == 0) & (vals > 0)).sum()),
        }
        for label, vals in cd4_series.items()
    }
).T
cd4_summary
```

## 3. Compare observed and diffused values

Mean stays near the observed level while max falls with `t` as diffusion spreads peak signal across neighbours.
`zero_fraction` also falls with `t`.
`filled_zeros` counts active cells that were zero for observed CD4 and became nonzero after diffusion.
That count is the size of the nonzero-as-detection mistake for this feature.

Literal Paris labels and UMAP coordinates from the catalog snapshot give the population context
for the CD4 panels below.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="RNA_paris_cluster",
)
```

```{code-cell} ipython3
imputation_comparison = ds.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=[
        "CD4",
        "CD4_imputed_t1",
        "CD4_imputed_t2",
        "CD4_imputed_t4",
    ],
    n_columns=4,
    color_scale=splt.ColorScale(scope="shared"),
    sort_values=True,
    show_titles=False,
    show=False,
)
for axis, title in zip(
    imputation_comparison.axes.values(),
    ("Observed CD4", "Diffusion t=1", "Diffusion t=2", "Diffusion t=4"),
    strict=True,
):
    axis.set_title(title)
imputation_comparison.figure
```

The imputed panels should fill gaps inside the same high-expression neighbourhoods visible in the observed panel.
Use the cluster map to check that high CD4 stays inside T-cell-like partitions rather than spreading into unrelated populations.
Signal across unrelated clusters indicates excessive diffusion or a graph that does not represent the intended biology.

## 4. Caveats

The result depends on the immutable cell selection and graph captured by each diffusion artifact.
Repeating `run_diffusion_operator` with the same graph and `t` reuses the complete stored artifact.
Pass that exact ref to `get_imputed`, or use `load_diffusion_operator` when direct sparse-matrix work
is needed.
Inserted columns such as `CD4_imputed_t2` are explicit cell metadata.
Do not interpret a nonzero imputed value as detection in that cell, use it for marker significance, or feed it to replicate-aware differential expression.
The `filled_zeros` column above is the concrete count of that mismatch for CD4 at each `t`.
