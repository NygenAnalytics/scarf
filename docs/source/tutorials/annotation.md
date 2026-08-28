---
description: Interpret immutable marker tables and write deliberate user-owned cell annotations.
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

# Interpreting markers and assigning cell types

Cluster IDs are not cell types. This page reads one immutable marker artifact, combines several
forms of evidence, and writes a user-owned annotation only after the labels are reviewed.

## 1. Open the clustered baseline

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
cluster_ref = run["clusters"]
marker_ref = run["markers"]
cluster_values = np.asarray(run.cells.fetch("clusters"))
pd.Series(cluster_values).value_counts().sort_index()
```

The rebuilt store carries this completed run under the immutable label `docs_default`. Its frozen
analysis selection is also the store's active `I`, so user-owned columns written below align with
the run without rebuilding any analytical stage.

The run's cluster-selection stage scores its configured Leiden partition and records the decision
in `run["cluster_selection"]`. `cluster_ref` and `marker_ref` remain exact artifacts; neither result
is written to live metadata.

```{code-cell} ipython3
ds.plots.embedding(run=run, layout="umap", color_by="clusters")
```

## 2. Read marker evidence

```{code-cell} ipython3
group_id = pd.Series(cluster_values).value_counts().index[0]
group_markers = ds.get_markers(
    marker=marker_ref,
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
].head(10)
```

Interpret the columns together:

- `score` ranks group specificity;
- `frac_exp` is the detected fraction in the target group;
- `fold_change` compares average target and reference expression;
- `auc` measures target versus reference separation;
- `p_value` is the two-sided Mann-Whitney result;
- `p_value_adjusted` applies Benjamini-Hochberg correction within this one-versus-rest group.

These are cell-level marker statistics, not replicate-aware differential expression.

```{code-cell} ipython3
markers = ds.get_markers(
    marker=marker_ref,
    group_id=None,
    min_score=-1,
    min_frac_exp=-1,
)
panel_hits = (
    markers[markers["feature_name"].astype(str).isin(["CD14", "MS4A1", "CD3D"])]
    .sort_values(["feature_name", "score"], ascending=[True, False])
    .groupby("feature_name", as_index=False)
    .head(1)[["feature_name", "group_id", "score", "auc", "frac_exp"]]
)
panel_hits
```

Use multiple positive and negative markers, cluster size, technical covariates, and replicate
coverage before assigning a name.

## 3. Write a reviewed annotation

This example labels the cluster where each panel gene ranks highest and leaves other groups as
`Cluster {id}`. The `insert` call is an explicit user metadata edit, not an analytical side effect.

```{code-cell} ipython3
unique = sorted(pd.unique(cluster_values), key=str)
label_map = {str(value): f"Cluster {value}" for value in unique}
for gene, name in (
    ("CD14", "Monocytes"),
    ("MS4A1", "B cells"),
    ("CD3D", "T cells"),
):
    hit = panel_hits.loc[panel_hits["feature_name"].astype(str) == gene]
    if not hit.empty:
        label_map[str(hit.iloc[0]["group_id"])] = name

cell_type = np.asarray([label_map[str(value)] for value in cluster_values], dtype=object)
ds.cells.insert("cell_type", cell_type, key="I", overwrite=True)
pd.Series(cell_type).value_counts()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout=run["umap"],
    color_by="cell_type",
)
```

Keep `label_map`, the marker ref, and the run ID in the study record. Cluster IDs can change when
the graph or resolution changes.

## 4. Smart labeling between artifacts

`smart_label` consumes two compatible categorical artifacts and returns another artifact. It does
not rename a live metadata column.

```{code-cell} ipython3
fine_clusters = ds.run_leiden_clustering(
    run["connectivity_map"],
    resolution=1.5,
)
smart_labels = ds.smart_label(fine_clusters, cluster_ref)
smart_values = np.asarray(ds.load_artifact(smart_labels)["values"][:])
pd.crosstab(
    pd.Series(np.asarray(ds.load_artifact(fine_clusters)["values"][:]), name="fine"),
    pd.Series(smart_values, name="smart label"),
)
```

Smart labels summarize overlap. They are not ontology annotations and still require biological
review.

## 5. Annotate scATAC-seq with gene scores

Peak IDs are difficult to interpret directly. `add_melded_assay` can combine peaks overlapping
gene bodies and promoters into a `GeneScores` assay. See {doc}`scatac_seq` for the executable path.
Gene scores are accessibility summaries, not measured RNA expression.

## Common mistakes and limitations

- Treating cluster IDs as biologically stable across parameter changes
- Assigning a type from one marker gene or one statistic
- Claiming replicate-aware differential expression from marker search
- Overwriting a reviewed annotation without retaining its source run and marker refs

Use {doc}`mapping_and_label_transfer` when query cells should inherit evidence from a fixed
reference.
