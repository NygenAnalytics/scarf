---
description: Inspect a prepared three-way RNA, ATAC, and protein WNN result.
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

# Three-way WNN with TEA-seq

TEA-seq measures RNA, chromatin accessibility, and surface proteins in the same cells. This
advanced example asks one question: can Scarf retain interpretable populations while learning a
separate local contribution from all three modalities?

The expensive import and preprocessing are prepared. The core two-modality path is
{doc}`cite_seq`; method comparison belongs in {doc}`multimodal_diagnostics`.

## Open the exact publication matches

The prepared store contains all 7,069 cells from the checksum-pinned
[GSM5123951 Seurat object](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5123951).
Published Figure 4 labels cover 6,333 well-W3 cells. Exact barcode matching finds 6,194 of those
labels in the pinned object, and only those matches are active for this analysis. The missing 139
publication cells are not replaced with arbitrary unlabeled cells.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "swanson_7K_pbmc_teaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
```

The raw feature counts stay on disk. This summary reads only array shapes and active feature masks.

```{code-cell} ipython3
pd.DataFrame.from_dict(
    {
        assay: {
            "cells": ds.get_assay(assay).rawData.shape[0],
            "raw features": ds.get_assay(assay).rawData.shape[1],
            "active features": int(
                ds.get_assay(assay).feats.fetch_all("I").sum()
            ),
        }
        for assay in ("RNA", "ATAC", "ADT")
    },
    orient="index",
)
```

The prepared recipe uses library-size log1p RNA with 2,000 HVGs and 30-component PCA, TF-IDF ATAC
with 25,000 peaks and 30-component LSI, and CLR-normalized ADT with 15-component PCA. Each source
has a self-free 20-neighbour row over the same 6,194 cells.

## Compare the modality-specific views

```{code-cell} ipython3
modality_layouts = {}
for assay in ("RNA", "ATAC", "ADT"):
    [modality_layouts[assay]] = ds.list_artifacts(
        from_assay=assay,
        kind="embedding",
        operation="run_umap",
        complete_only=True,
    )
```

### Question: do all three assays retain the broad publication populations?

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, (assay, layout) in zip(
    axes,
    modality_layouts.items(),
    strict=True,
):
    ds.plots.embedding(
        layout=layout,
        color_by="tea_cell_type",
        point_size=5,
        legend_loc="right" if assay == "ADT" else "none",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(assay)
figure.tight_layout()
```

Broad agreement supports a shared signal. Local rearrangements can reflect complementary
measurement or modality-specific noise, so visual compactness alone is not validation.

## Reopen the three-way WNN result

```{code-cell} ipython3
[wnn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "wnn"},
    complete_only=True,
)
[wnn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": wnn_graph},
    complete_only=True,
)

wnn_status = ds.inspect_artifact(wnn_graph)
assert wnn_status.parameters["assays"] == ["RNA", "ATAC", "ADT"]
pd.Series(
    {
        "artifact": wnn_graph.artifact_id,
        "ordered assays": ", ".join(wnn_status.parameters["assays"]),
        "complete": wnn_status.complete,
    }
)
```

### Question: does the joint map preserve labels and protein landmarks?

```{code-cell} ipython3
figure, axes = plt.subplots(2, 3, figsize=(11, 7))
panels = (
    (None, None, "Publication cell type"),
    ("CD3", "ADT", None),
    ("CD19", "ADT", None),
    ("CD14", "ADT", None),
    ("CD56", "ADT", None),
)
for axis, (color_by, assay, title) in zip(
    axes.flat,
    panels,
    strict=False,
):
    ds.plots.embedding(
        layout=wnn_layout,
        color_by="tea_cell_type" if color_by is None else color_by,
        from_assay=assay,
        point_size=5,
        sort_values=color_by is not None,
        legend_loc="right" if color_by is None else "auto",
        show_titles=False,
        target=axis,
        show=False,
    )
    if title is not None:
        axis.set_title(title)
axes.flat[-1].set_visible(False)
figure.tight_layout()
```

CD3, CD19, CD14, and CD56 support T-cell, B-cell, monocyte, and NK-like regions on the integrated
layout. Agreement with imported publication labels is useful evidence, not proof that every local
WNN relationship is correct.

## Inspect all three modality weights

### Question: where does each local neighbourhood contribute most strongly?

```{code-cell} ipython3
ds.plots.modality_weights(
    graph=wnn_graph,
    layout=wnn_layout,
)
```

The plotting API validates that the stored weights are finite, non-negative, sum to one per cell,
follow the persisted assay order, and align to the exact WNN layout selection. A high weight reports
relative local predictability under these reductions and neighbourhoods. It is not molecular
abundance, global assay quality, or causal importance.

The N-way weighting equations, candidate construction, and differences from Seurat belong in the
{doc}`../reference/api/integration` contract. The prepared UMAP is a Scarf result and is not intended
to reproduce the publication UMAP.

Data attribution:

- Swanson et al. (2021), [TEA-seq](https://doi.org/10.7554/eLife.63632)
- [GEO GSM5123951](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5123951)
- eLife Figure 4 source data 2
