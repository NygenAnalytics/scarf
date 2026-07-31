---
description: Estimate cell-level probabilities for several supervised terminal states.
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

# Fate mapping across terminal states

Pseudotime places cells along one progression axis. Fate mapping complements that
ordering with a probability for each user-provided terminal state. This notebook
uses pancreatic endocrine differentiation to estimate Alpha, Beta, and Delta
fates.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and cluster annotations for sources and sinks

## What you will learn

- Define progenitor and terminal groups for multi-sink PBA
- Estimate a shared pseudotime across branches
- Compute absorption probabilities for each terminal fate

## Dataset

```{code-cell} ipython3
import matplotlib.pyplot as plt

import scarf
import scarf.plotting as splt

scarf.configure_output(level='WARNING', progress=True)
```

## 1. Load the preprocessed dataset

The prepared Zarr store from the `scarf_docs` Cytebase catalog contains a KNN graph,
UMAP coordinates, Scarf clusters, and the provided cell-type annotations.

```{code-cell} ipython3
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name='bastidas-ponce_4K_pancreas-d15_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
```

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    default_assay='RNA',
)
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='clusters',
    legend_loc='on_data',
)
```

## 2. Define progenitor and terminal groups

Fate mapping needs a starting point and a set of endpoints. The provided `clusters`
annotation names both: ductal cells are the progenitor pool of this stage, and the
hormone-expressing states are the terminal fates.

```{code-cell} ipython3
progenitors = ['Ductal']
terminal_cell_types = ['Alpha', 'Beta', 'Delta']

ds.cells.to_pandas_dataframe(
    ['clusters'],
    key='I'
)['clusters'].value_counts()
```

## 3. Estimate a shared pseudotime

PBA supplies the developmental direction. The terminal cell types are used as sinks so the
ordering covers the branches analyzed below. `label='fate_pseudotime'` keeps these columns
separate from any pseudotime already stored on the object.

```{code-cell} ipython3
pseudotime = ds.run_pseudotime_scoring(
    source_sink_key='clusters',
    sources=progenitors,
    sinks=terminal_cell_types,
    label='fate_pseudotime',
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=pseudotime.pseudotime_key,
    subset_by=pseudotime.validity_key,
)
```

## 4. Compute fate probabilities

`run_fate_mapping` biases the KNN graph toward increasing pseudotime and solves
the absorption probability for each terminal cell type. Every cell carrying an Alpha, Beta,
or Delta annotation defines the corresponding fate boundary. The PBA validity key excludes
any unscored graph components from the fate calculation. Since that subset key is
not `I`, Scarf includes it in the saved fate column names.

```{code-cell} ipython3
fate = ds.run_fate_mapping(
    cell_key='I',
    subset_cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
    sink_key='clusters',
    sinks=terminal_cell_types,
)
```

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(11, 4))
probability_scale = splt.ColorScale(vmin=0, vmax=1)
for index, (axis, sink, fate_key) in enumerate(
    zip(axes, fate.sink_labels, fate.fate_keys, strict=True)
):
    ds.plots.embedding(
        layout_key="RNA_UMAP",
        color_by=fate_key,
        subset_by=fate.validity_key,
        color_scale=probability_scale,
        sort_values=True,
        show_legend=index == 2,
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(f"{sink} fate probability")
for colorbar_axis in set(figure.axes) - set(axes):
    colorbar_axis.set_title(f"{fate.sink_labels[-1]} fate probability")
    colorbar_axis.set_xlabel("")
    colorbar_axis.set_ylabel("")
figure.tight_layout()
figure
```

The terminal regions should be dominated by their matching fate, while
intermediate cells can retain probability across several outcomes. A terminal
group with low probability for its own fate indicates a mismatch between the
annotations, graph, and selected boundaries.

## Interpretation and limits

The PBA score describes progress along the shared developmental direction.
The fate columns separate that direction into terminal outcomes and quantify
ambiguous intermediate cells. Sink identities remain supervised: this method
does not discover terminal states automatically and does not use RNA velocity.

## Common mistakes and limitations

- Choosing sink clusters that mix several terminal annotations
- Interpreting fate probabilities as lineage commitment without experimental support
- Ignoring the pseudotime validity key when the graph has multiple components
- Expecting the method to find terminal states on its own

Fate probabilities are stored under the keys returned in `fate.fate_keys`,
with a matching validity column. Probability-simplex checks, solver diagnostics,
and tuning belong in {doc}`trajectory_analysis`.
