---
description: Review immutable marker evidence and write deliberate cell-type annotations.
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

(annotation)=

# Review markers and assign cell types

Cluster IDs are not cell types. This recipe reads one immutable marker result, reviews several
forms of evidence, and writes a user-owned annotation only after the cluster-to-label mapping is
explicit. The core {doc}`scrna_seq` workflow shows the corresponding broad PBMC dotplot.

## Open the exact clustering and markers

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
run = ds.pipeline.open(label="docs_default")
clusters = run["clusters"]
markers = run["markers"]
cluster_values = np.asarray(run.cells.fetch("clusters"))
```

The run binds the marker table to the pipeline-selected Leiden partition and frozen feature
universe. Start by inspecting the strongest markers for one group rather than naming it from UMAP
position.

```{code-cell} ipython3
group_id = pd.Series(cluster_values).value_counts().index[0]
group_markers = ds.get_markers(
    marker=markers,
    group_id=group_id,
    min_score=-1,
    min_frac_exp=-1,
)
group_markers[
    [
        "feature_name",
        "score",
        "frac_exp",
        "fold_change",
        "auc",
        "p_value",
        "p_value_adjusted",
    ]
].head(12)
```

Use the columns together. `score` ranks specificity, `frac_exp` reports detection in the target,
`fold_change` compares target and reference means, and AUC summarizes cell-level separation.
`p_value_adjusted` is Benjamini-Hochberg adjustment within this one-versus-rest marker test. It is
not replicate-aware differential expression.

### Question: do several markers support each cluster interpretation?

```{code-cell} ipython3
ds.plots.marker_heatmap(
    marker=markers,
    topn=3,
    figsize=(6, 8),
)
```

Look for coherent programs rather than a single winning gene. In a real study, also inspect
expected negative markers, cluster size, technical covariates, donor coverage, and doublet scores.

## Write the reviewed mapping

This example records the broad teaching labels supported in {doc}`scrna_seq`. Multiple Leiden
clusters intentionally map to the same lineage. The mapping is tied to this run and should not be
copied to another graph or dataset.

```{code-cell} ipython3
label_map = {
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
observed = {str(value) for value in np.unique(cluster_values)}
assert observed == set(label_map)

analysis_cells = np.asarray(run.cells.fetch_all("I"), dtype=bool)
cell_type = np.full(len(analysis_cells), "Not analyzed", dtype=object)
cell_type[analysis_cells] = [label_map[str(value)] for value in cluster_values]
ds.cells.insert("reviewed_cell_type", cell_type, overwrite=True)
pd.Series(cell_type[analysis_cells]).value_counts()
```

The insertion is an explicit user metadata edit. It does not alter the immutable clustering or
marker artifacts.

### Question: does the reviewed annotation remain spatially coherent?

```{code-cell} ipython3
ds.plots.embedding(
    layout=run["umap"],
    color_by="reviewed_cell_type",
    legend_loc="right",
)
```

Keep the label map, marker ref, run ID, and review rationale in the study record. Cluster IDs can
change when the graph or partition changes. Overlap-based `smart_label` helps compare two
partitions, but it is not ontology annotation; that partition-comparison role belongs in
{doc}`clustering`.

For scATAC-seq, {doc}`scatac_seq` uses GeneScores to display marker accessibility. GeneScores are
accessibility summaries, not measured RNA expression. Use {doc}`mapping_and_label_transfer` when a
query should inherit labels from a fixed reference rather than be annotated de novo.
