---
description: Inspect a prepared TEA-seq store and its three-way WNN integration artifact.
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

# Three-way RNA, ATAC, and protein integration with TEA-seq

TEA-seq measures gene expression, chromatin accessibility, and surface proteins in the same cells.
This tutorial opens a prepared store, inspects the three modality-specific analyses, and follows a three-way weighted nearest neighbour (WNN) artifact by reference.

The expensive import and preprocessing steps have already run.
The executable work here is limited to downloading the analyzed store, reading its artifacts, and plotting their results.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Familiarity with the two-modality workflow in {doc}`cite_seq`
- The integration diagnostics in {doc}`multimodal_diagnostics`

## What you will learn

- Inspect matched RNA, ATAC, and ADT assays without materializing their count matrices
- Compare modality-specific layouts with a three-way SNN layout
- Inspect a stored RNA plus ATAC plus ADT WNN graph and its exact inputs
- Validate and visualize one modality weight per cell and assay

## 1. Dataset and cell selection

The prepared store contains the 7,069 peripheral blood mononuclear cells and all raw features in the checksum-pinned [GSM5123951 TEA-seq Seurat object](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5123951).
Cell types come from [eLife Figure 4 source data](https://doi.org/10.7554/eLife.63632).

The two pinned sources are not the same revision.
Figure 4 lists 6,333 cells from well W3, but exact matching by `original_barcodes` and the W3 suffix finds 6,194 cells in the Seurat object.
The prepared analysis activates those 6,194 exact matches.
It does not replace the 139 missing publication cells with unlabelled cells.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
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

Read the stored shapes instead of loading a complete matrix.
The ATAC count array retains all 240,122 peaks, while the stored reduction lineage selects 25,000 peaks.

```{code-cell} ipython3
pd.DataFrame.from_dict(
    {
        name: {
            "cells": ds.get_assay(name).rawData.shape[0],
            "raw features": ds.get_assay(name).rawData.shape[1],
            "active features": int(ds.get_assay(name).feats.fetch_all("I").sum()),
        }
        for name in ("RNA", "ATAC", "ADT")
    },
    orient="index",
)
```

The active key identifies the exact publication matches used by every stored analysis artifact.

```{code-cell} ipython3
pd.Series(
    {
        "source cells retained": ds.cells.N,
        "active exact W3 matches": int(ds.cells.fetch_all("I").sum()),
        "Figure 4 W3 labels": 6_333,
        "Figure 4 labels absent from the pinned RDS": 139,
    }
)
```

Publication lineage labels on the active exact matches:

```{code-cell} ipython3
pd.Series(ds.cells.fetch("tea_cell_type", key="I")).value_counts()
```

## 2. Fixed preprocessing recipe

The recipe is recorded in the dataset manifest and represented by complete artifacts in the store.
This page does not refit normalization, reductions, neighbour graphs, UMAP, or clustering.

| Assay | Normalization | Feature selection | Reduction |
| --- | --- | --- | --- |
| RNA | Library-size log1p | 2,000 HVGs | 30-component PCA |
| ATAC | TF-IDF | 25,000 prevalent peaks | 30-component streaming LSI, first component skipped |
| ADT | CLR | Exclude isotype control | 15-component PCA |

Each modality has a 20-neighbour self-free row over the same active cells.
Those matched rows feed both integrations.
SNN merges shared edge support with equal standing for each source graph.
WNN estimates a separate local contribution for each assay and cell.
Integration provenance stores ordered `source_i` inputs matching the `assays` parameter.
SNN sources are exact connectivity-map references; WNN sources pair each neighbor reference with the exact native reduction or batch-correction coordinates it names.

## 3. Compare modality-specific and SNN layouts

These layouts are independent outputs.
Similar broad populations support a shared signal, while local differences can reflect complementary measurements or modality-specific noise.

```{code-cell} ipython3
modality_layouts = {}
for assay in ("RNA", "ATAC", "ADT"):
    [modality_layouts[assay]] = ds.list_artifacts(
        from_assay=assay,
        kind="embedding",
        complete_only=True,
    )

integrated_graphs = {
    ds.inspect_artifact(ref).parameters["method"]: ref
    for ref in ds.list_artifacts(
        scope="datastore",
        kind="integrated_graph",
        complete_only=True,
    )
}


def linked_output(graph, kind):
    [result] = [
        ref
        for ref in ds.list_artifacts(scope="datastore", kind=kind, complete_only=True)
        if scarf.ArtifactRef.from_dict(ds.inspect_artifact(ref).inputs["graph"])
        == graph
    ]
    return result


snn_ref = integrated_graphs["snn"]
wnn_ref = integrated_graphs["wnn"]
snn_layout = linked_output(snn_ref, "embedding")
wnn_layout = linked_output(wnn_ref, "embedding")
```

```{code-cell} ipython3
figure, axes = plt.subplots(2, 2, figsize=(10, 8))
for axis, (label, layout) in zip(
    axes.flat,
    (*modality_layouts.items(), ("SNN", snn_layout)),
    strict=True,
):
    ds.plots.embedding(
        layout=layout,
        color_by="tea_cell_type",
        point_size=5,
        legend_loc="right",
        target=axis,
        show=False,
    )
    axis.set_title(label)
figure.tight_layout()
figure
```

The SNN layout is a joint view, not a reference truth.
Differences between panels should be checked against markers and assay quality rather than judged from visual compactness alone.
This store keeps Ensembl IDs as RNA feature names, so the panel below uses ADT proteins for T, B, monocyte, and NK-like landmarks.

```{code-cell} ipython3
ds.plots.embedding(
    layout=snn_layout,
    from_assay="ADT",
    color_by=["CD3", "CD19", "CD14", "CD56"],
    n_columns=2,
    point_size=5,
    sort_values=True,
    show_titles=True,
    figsize=(10, 8),
)
```

## 4. Inspect the three-way WNN graph

The prepared WNN result has an immutable artifact reference and records its assays in order.
`integrate_assays` consumes exact neighbour artifacts, whose lineage includes their native
coordinate artifacts, and captures every source before planning.

```{code-cell} ipython3
wnn_status = ds.inspect_artifact(wnn_ref)
assays = list(wnn_status.parameters["assays"])

assert assays == ["RNA", "ATAC", "ADT"]
pd.Series(
    {
        "artifact kind": wnn_ref.kind,
        "artifact id": wnn_ref.artifact_id,
        "complete": wnn_status.complete,
        "ordered assays": ", ".join(wnn_status.parameters["assays"]),
        "input roles": ", ".join(sorted(wnn_status.inputs)),
    }
)
```

The artifact stores its modality weights in the same order as the input assays.

```{code-cell} ipython3
weight_values = np.asarray(ds.load_artifact(wnn_ref)["modality_weights"][:])

assert weight_values.shape == (6_194, 3)
assert np.isfinite(weight_values).all()
assert (weight_values >= 0).all()
assert np.allclose(weight_values.sum(axis=1), 1, atol=1e-6)

pd.Series(
    {
        "shape": str(weight_values.shape),
        "minimum weight": float(weight_values.min()),
        "maximum weight": float(weight_values.max()),
        "maximum row-sum error": float(np.abs(weight_values.sum(axis=1) - 1).max()),
    }
)
```

Mean modality weight by publication label shows which populations lean on RNA, ATAC, or ADT under this stored neighbourhood.

```{code-cell} ipython3
weight_frame = pd.DataFrame(
    weight_values,
    columns=[f"{assay} weight" for assay in assays],
)
weight_frame["tea_cell_type"] = ds.cells.fetch("tea_cell_type", key="I")
weight_columns = [f"{assay} weight" for assay in assays]
weight_frame.groupby("tea_cell_type")[weight_columns].agg(["count", "mean"]).round(3)
```

Plotting all three weights on the integrated layout shows where the graph relies more strongly on each local neighbourhood.

```{code-cell} ipython3
wnn_coordinates = np.asarray(ds.load_artifact(wnn_layout)["values"][:])
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, values, title in zip(
    axes,
    weight_values.T,
    weight_columns,
    strict=True,
):
    points = axis.scatter(
        wnn_coordinates[:, 0],
        wnn_coordinates[:, 1],
        c=values,
        s=5,
    )
    axis.set_title(title)
    figure.colorbar(points, ax=axis)
figure.tight_layout()
```

Place the three-way SNN and WNN layouts side by side under the same cell-type colouring.
Broad populations should agree; local rearrangements need marker and weight support before interpretation.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, (label, layout) in zip(
    axes,
    (("SNN", snn_layout), ("WNN", wnn_layout)),
    strict=True,
):
    ds.plots.embedding(
        layout=layout,
        color_by="tea_cell_type",
        point_size=5,
        legend_loc="right",
        target=axis,
        show=False,
    )
    axis.set_title(label)
figure.tight_layout()
figure
```

The same ADT panel on the WNN layout checks whether lineage landmarks remain coherent after per-cell reweighting.

```{code-cell} ipython3
ds.plots.embedding(
    layout=wnn_layout,
    from_assay="ADT",
    color_by=["CD3", "CD19", "CD14", "CD56"],
    n_columns=2,
    point_size=5,
    sort_values=True,
    show_titles=True,
    figsize=(10, 8),
)
```

## 5. How N-way WNN combines the assays

For cell \(i\), let \(\mathcal{N}_{m,i}\) be the stored self-free neighbour row for modality \(m\).
Scarf scores the bounded candidate set

\[ \mathcal{C}_i = \bigcup_m \mathcal{N}_{m,i} \]

and retains \(\min_m |\mathcal{N}_{m,i}|\) output neighbours.
Let \(\theta_{m \leftarrow n,i}\) be the affinity in target modality \(m\) between cell \(i\) and the mean target-modality coordinates of neighbours selected by source modality \(n\).
The directed cross-modality score is

\[ s_{m,n,i} = \operatorname{clip}\left( \frac{\theta_{m \leftarrow m,i}}
     {\theta_{m \leftarrow n,i} + 10^{-4}},
0, 200 \right), \qquad m \ne n. \]

For three or more assays, Scarf uses the grouped pairwise normalization from the Hao/current-Seurat formulation:

\[ w_{m,i} = \frac{\sum_{n \ne m}\exp(s_{m,n,i})}
     {\sum_a\sum_{b \ne a}\exp(s_{a,b,i})}.
\]

The largest directed score is subtracted before exponentiation for numerical stability.
This leaves the ratio unchanged and reduces to the existing two-modality softmax when there are two assays.
Candidate \(j\) then receives the blended affinity

\[ A_{i,j} = \sum_m w_{m,i} A^{(m)}_{i,j}. \]

Each target modality uses its own nearest-neighbour distance and bandwidth.
Scarf uses the union of existing KNN rows and the distance span from the nearest to the k-th neighbour as bandwidth.
It therefore follows the weighting equations but is not bit-identical to Seurat's default wider candidate search and SNN-far bandwidth, or to the TEA-seq fork's max-cross shortcut.

## Interpretation and limitations

WNN weights report relative local predictability under these stored embeddings and neighbourhoods.
They do not measure molecular abundance, global assay quality, or causal importance.
A high weight can reflect useful resolution, but it can also reflect preprocessing choices or noise.

The prepared UMAPs are Scarf results and are not intended to reproduce the published TEA-seq UMAP.
The exact-match selection also means that conclusions apply to the 6,194 cells represented in both pinned sources, not to all 6,333 Figure 4 labels.

Data attribution:

- Swanson et al. (2021), [TEA-seq](https://doi.org/10.7554/eLife.63632)
- [GEO GSM5123951](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5123951)
- eLife Figure 4 source data 2

## Common mistakes

- Recomputing the expensive assay pipelines when direct artifact inspection is sufficient
- Treating the three modality weights as expression, accessibility, or protein abundance
- Comparing graphs built over different cell selections or neighbour counts
- Assuming a visually cleaner integrated layout is the more accurate result
- Filling the 139-cell source mismatch with arbitrary unlabelled cells
