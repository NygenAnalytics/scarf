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
# Add a third modality with TEA-seq

TEA-seq measures RNA, chromatin accessibility, and surface proteins in the same cells.
After the two-modality example in {doc}`cite_seq`, we now ask whether a joint map retains
recognizable populations and how much each modality contributes locally.

## Open the prepared result

The data come from [Swanson et al. (2021)](https://doi.org/10.7554/eLife.63632).
The prepared store contains 7,069 cells from the
[GSM5123951 Seurat object](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5123951).
We use the 6,194 cells whose barcodes match the publication's Figure 4 labels. Of the
6,333 labeled well-W3 cells, 139 are absent from this source object.

```{code-cell}
import scarf
from scarf.plotting import CellField, FeatureRef

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "swanson_7K_pbmc_teaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", default_assay="RNA", nthreads=4)
```

Import and preprocessing are already complete. RNA uses log-transformed library-size
normalization, ATAC uses TF-IDF, and ADT uses CLR normalization. All three neighbor
results describe the same selected cells.

## Compare the separate views

First look for the labeled populations in each modality. Independent UMAPs can rotate
and rearrange, so compare population neighborhoods rather than absolute positions or
island sizes. A population's labels and cell count are the same in all three panels.

```{code-cell}
for assay in ("RNA", "ATAC", "ADT"):
    [layout] = ds.list_artifacts(
        from_assay=assay,
        kind="embedding",
        operation="run_umap",
        complete_only=True,
    )
    print(assay)
    ds.plots.embedding(layout=layout, color_by="tea_cell_type")
```

Differences between these views may reflect complementary measurements or technical
noise. A more compact population is not, by itself, evidence of a better analysis.

## Inspect the joint WNN map

Select the saved WNN graph and the UMAP made from it:

```{code-cell}
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
```

Now compare the publication labels with four measured protein markers on the joint map.

```{code-cell}
ds.plots.embedding(
    layout=wnn_layout,
    color_by=[
        CellField("tea_cell_type", label="Publication cell type"),
        FeatureRef("CD3", assay="ADT"),
        FeatureRef("CD19", assay="ADT"),
        FeatureRef("CD14", assay="ADT"),
        FeatureRef("CD56", assay="ADT"),
    ],
    n_columns=3,
    sort_values=True,
)
```

CD3, CD19, CD14, and CD56 support T-cell, B-cell, monocyte, and NK-like regions,
respectively. Agreement with the publication labels is useful evidence, but does not
validate every neighborhood. This UMAP is a Scarf analysis, not a reproduction of the
publication's layout.

## See which modality contributes locally

```{code-cell}
ds.plots.modality_weights(graph=wnn_graph, layout=wnn_layout)
```

A high weight means a modality contributes more to that cell's WNN neighborhood under
these preprocessing choices. It does not measure molecular abundance, overall assay
quality, or a cause of cell identity. Look for regions where the weights differ, then
return to the marker and single-modality maps to interpret those differences.

To check how sensitive the result is to integration choices, continue to
{doc}`multimodal_diagnostics`. The weighting equations and differences from Seurat are
covered in {doc}`../reference/api/integration`.
