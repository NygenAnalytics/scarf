---
description: Interpret a prepared scRNA-seq workflow with marker-supported PBMC labels.
jupytext:
  cell_metadata_filter: -all
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

(scrna_seq_workflow)=

# Identify PBMC populations with scRNA-seq

Use one prepared RNA pipeline result to turn Leiden clusters into broad, marker-supported PBMC
labels. The objective is biological interpretation, not rebuilding stages already covered by the
focused QC, reduction, graph, and clustering guides.

This teaching dataset is useful for workflow familiarity. Scale claims come from
{doc}`../concepts/benchmarks`, not from this small matrix.

## Open the completed RNA workflow

```{code-cell} ipython3
import numpy as np

import scarf
from scarf.plotting import CategoricalScale, CellField

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
```

`run["clusters"]` is the Leiden partition selected by the pipeline. Alternative clustering
choices and selection diagnostics are covered in {doc}`clustering`.

The prepared run also binds the exact filtered cells, UMAP, and marker table used below.

## Assign dataset-specific teaching labels

The labels below summarize this dataset's marker evidence. They are not an automated annotation
algorithm or a reusable PBMC ontology. Several Leiden clusters are deliberately collapsed into
the same broad lineage.

```{code-cell} ipython3
cell_type_by_cluster = {
    "1": "CD14 monocytes",
    "2": "FCGR3A monocytes",
    "3": "B cells",
    "4": "T cells",
    "5": "NK cells",
    "6": "T cells",
    "7": "T cells",
    "8": "B cells",
    "9": "T cells",
    "10": "T cells",
    "11": "pDC-like cells",
    "12": "Platelets",
}
cell_type_order = (
    "CD14 monocytes",
    "FCGR3A monocytes",
    "B cells",
    "T cells",
    "NK cells",
    "pDC-like cells",
    "Platelets",
)

cluster_values = np.asarray(run.cells.fetch("clusters"))
observed_clusters = {str(value) for value in np.unique(cluster_values)}
assert observed_clusters == set(cell_type_by_cluster)

analysis_cells = np.asarray(run.cells.fetch_all("I"), dtype=bool)
cell_types = np.full(len(analysis_cells), "Not analyzed", dtype=object)
cell_types[analysis_cells] = [
    cell_type_by_cluster[str(value)] for value in cluster_values
]
ds.cells.insert("pbmc_cell_type", cell_types, overwrite=True)
ds.cells.insert("rna_analysis_cells", analysis_cells, overwrite=True)
```

### Question: where are the broad PBMC populations?

```{code-cell} ipython3
ds.plots.embedding(
    layout=run["umap"],
    color_by=CellField(
        "pbmc_cell_type",
        kind="categorical",
        label="Broad PBMC identity",
    ),
    categorical_scale=CategoricalScale(order=cell_type_order),
    legend_loc="right",
)
```

Monocytes, B cells, T cells, NK cells, a small pDC-like population, and platelets occupy coherent
neighbourhoods. UMAP position alone did not assign these names; the marker panel is the evidence.

### Question: which markers support each label?

```{code-cell} ipython3
marker_panel = {
    "Monocyte": ["LST1", "S100A8", "FCGR3A"],
    "B cell": ["MS4A1", "CD79A"],
    "T cell": ["CD3D", "IL7R"],
    "NK cell": ["NKG7", "GNLY"],
    "pDC-like": ["GZMB", "JCHAIN"],
    "Platelet": ["PPBP", "PF4"],
}
ds.plots.dotplot(
    features=marker_panel,
    group_by="pbmc_cell_type",
    cell_key="rna_analysis_cells",
    group_order=cell_type_order,
    standardize="feature",
    italicize_features=True,
)
```

The two monocyte labels share LST1 but separate along S100A8 and FCGR3A. MS4A1 and CD79A support
B cells; CD3D and IL7R support T cells; NKG7 and GNLY support NK cells; GZMB with JCHAIN motivates
the cautious pDC-like label; and PPBP with PF4 identifies platelets. These remain dataset-specific
teaching labels and should be reviewed with additional positive and negative markers in a real
study.

## Substitute your own input

For a filtered Cell Ranger H5 file, convert the counts and run the same default pipeline:

```python
reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
scarf.CrToZarr(reader, zarr_loc="analysis.zarr").dump()

own_ds = scarf.DataStore("analysis.zarr", nthreads=4)
own_run = own_ds.pipeline.run(label="baseline")
own_ds.plots.embedding(run=own_run, color_by="clusters")
```

Choose QC thresholds and review markers for the new dataset before assigning labels. Use
{doc}`quality_control` for alternative filtering policies, {doc}`feature_selection` and
{doc}`dimensionality_reduction` for representation choices, {doc}`graph_construction` for graph
diagnostics, {doc}`clustering` for partition choices, and {doc}`annotation` for a fuller annotation
workflow. Artifact identity and branching belong in {doc}`reuse_and_tracing`.

## Limits of this result

- The labels are curated for this prepared 5K PBMC result and are not transferred automatically.
- Marker tests compare cells, not biological replicates, so they are not condition-level DE.
- UMAP distances and empty space are visual summaries, not measured biological distances.
