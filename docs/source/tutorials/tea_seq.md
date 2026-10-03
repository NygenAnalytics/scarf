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
# Open count stores and run Scarf analyses.
import scarf
# Select explicit fields and display options for plots.
from scarf.plotting import CellField, FeatureRef

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "swanson_7K_pbmc_teaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(f"{dataset}/data.zarr", default_assay="RNA", nthreads=4)
# Inspect the opened store's cells and features.
ds
```

Import and preprocessing are already complete. RNA uses log-transformed library-size
normalization, ATAC uses TF-IDF, and ADT uses CLR normalization. All three neighbor
results describe the same selected cells.

## Compare the separate views

First look for the labeled populations in each modality. Independent UMAPs can rotate
and rearrange, so compare population neighborhoods rather than absolute positions or
island sizes. A population's labels and cell count are the same in all three panels.

```{code-cell}
# Inspect the RNA, ATAC, and protein views in turn.
for assay in ("RNA", "ATAC", "ADT"):
    # Select the saved layout for this assay; require exactly one match.
    [layout] = ds.list_artifacts(
        from_assay=assay,
        kind="embedding",
        operation="run_umap",
        complete_only=True,
    )
    # Name the modality on its UMAP and color cells by the publication labels.
    ds.plots.embedding(
        layout=layout,
        color_by=CellField("tea_cell_type", label=f"{assay}: publication cell type"),
    )
```

Differences between these views may reflect complementary measurements or technical
noise. A more compact population is not, by itself, evidence of a better analysis.

## Inspect the joint WNN map

Select the saved WNN graph and the UMAP made from it:

```{code-cell}
# Select the saved WNN graph; require exactly one match.
[wnn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "wnn"},
    complete_only=True,
)
# Select the saved WNN layout; require exactly one match.
[wnn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": wnn_graph},
    complete_only=True,
)
# Inspect the selected WNN graph and its matching layout.
{"WNN graph": wnn_graph, "WNN layout": wnn_layout}
```

Now compare the publication labels with four measured protein markers on the joint map.

```{code-cell}
# Compare publication labels with four protein markers on the WNN map.
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
# Show how much each modality contributes across the joint map.
ds.plots.modality_weights(graph=wnn_graph, layout=wnn_layout)
```

A high weight means a modality contributes more to that cell's WNN neighborhood under
these preprocessing choices. It does not measure molecular abundance, overall assay
quality, or a cause of cell identity. Look for regions where the weights differ, then
return to the marker and single-modality maps to interpret those differences.

To check how sensitive the result is to integration choices, continue to
{doc}`multimodal_diagnostics`. The weighting equations and differences from Seurat are
covered in {doc}`../reference/api/integration`.
