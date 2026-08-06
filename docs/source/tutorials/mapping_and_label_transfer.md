---
description: Map query cells to a fixed reference, inspect mapping evidence, and transfer labels with abstention.
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

(data_projection)=

# Mapping cells and transferring labels

Mapping is the fixed-reference alternative to merging datasets and rebuilding a joint graph.
Keep one reference atlas unchanged, place new query cells onto it, and transfer labels from reference neighbours.

It does three things in order:

1. Align the query features to the reference feature panel.
2. Project each query cell into the reference PCA space and find nearest reference neighbours.
3. Use those neighbours to transfer labels and score how much of the query landed on each reference cell.

It does not merge count matrices, retrain the reference graph, or move reference cells.
When sources must be analysed together in one store, start with {doc}`dataset_merging` and {doc}`batch_correction` instead.

This tutorial maps interferon-stimulated PBMCs onto a control PBMC reference from the same Kang study.
The shared author labels let us evaluate the result.
For a reusable Symphony-style atlas, see {doc}`reference_atlases`.

Mapping currently supports RNA queries.
The prepared reference may be reopened read-only, but the query must be a different writable store.

## 1. Open the reference and query

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=True)

repository = scarf.cytebase.connect("scarf_docs")
ctrl_path = repository.download_dataset(
    name="kang_15K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
stim_path = repository.download_dataset(
    name="kang_14K_ifnb-pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)

ds_ctrl = scarf.DataStore(
    f"{ctrl_path}/data.zarr",
    nthreads=4,
    zarr_mode="r+",
)
ds_stim = scarf.DataStore(
    f"{stim_path}/data.zarr",
    nthreads=4,
    zarr_mode="r+",
)
```

```{code-cell} ipython3
ds_ctrl.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="cluster_labels",
)
ds_stim.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="cluster_labels",
)
```

These UMAP layouts were fitted independently, so their coordinates are not comparable.
Mapping keeps the control layout fixed and reports where query weight lands on reference cells.
The shared label vocabulary is what we use for evaluation.

## 2. Prepare a labelled reference

Build the reference graph with the standard RNA pipeline, then package the neighbour chain as an immutable mapping reference.

```{code-cell} ipython3
artifacts = ds_ctrl.pipeline.run(
    filtering=False,
    cell_cycle_scoring=False,
    highly_variable_features={
        "min_cells": 10,
        "top_n": 2000,
        "min_mean": -3,
        "max_mean": 2,
        "max_var": 6,
    },
    pca={"dims": 25},
    neighbors={"k": 17},
    umap=False,
    leiden={},
    paris=False,
    doublet_scoring=False,
    markers=False,
)
reference = ds_ctrl.build_mapping_reference(artifacts["neighbors"])
```

The completed `MappingReference` is immutable.
Its feature order, scaling, PCA loadings, neighbour index, and selected cells stay fixed in the reference datastore.
This example uses a plain PCA reference.
A Symphony reference instead passes Harmony-corrected neighbours into `build_mapping_reference`.

## 3. Map the query

`run_mapping` runs on the writable query datastore.
It aligns query features to the reference panel, applies the reference normalization and scaling, projects into the reference PCA space, and stores the nearest neighbours.
Query cells are never inserted into the reference index.

```{code-cell} ipython3
mapping = ds_stim.run_mapping(
    reference,
    "stim",
    query_assay="RNA",
    save_k=5,
    missing_feature_policy="reference_mean",
)
```

`reference_mean` fills an absent query feature with the reference mean, which becomes zero after reference scaling.
Use `zero` to fill with a normalized zero, or `error` when complete feature overlap is required.

Reload the projection with neighbour arrays to confirm the write: one row per mapped query cell, `save_k` reference neighbours, and finite distances.

```{code-cell} ipython3
peek = ds_stim.get_mapping_result(mapping, load_arrays=True)
peek.n_cells, int(peek.indices.shape[1]), peek.indices[:3], peek.distances[:3]
```

`mapping.diagnostics["queryScaledDispersion"]` compares query spread with the reference after scaling.
Values near 1 mean the query occupies a similar region of feature space.
Values much below 1 mean the query is compressed toward the centre of the reference cloud and neighbour labels become less trustworthy.

```{code-cell} ipython3
mapping.diagnostics
```

## 4. Where did the query land?

A mapping score tells you which reference cells received neighbour weight from the query.
Plot it on the reference UMAP.
One panel for the whole query is hard to read because the weight is spread across many cells.
Split by a few known query populations to see whether each population lands on the matching reference region.

```{code-cell} ipython3
query_labels = np.asarray(ds_stim.cells.fetch("cluster_labels")).astype(str)
focus = {"CD 14 Mono", "CD4 Memory T", "CD4 naive T", "NK"}
score_groups = np.array(
    [label if label in focus else "other" for label in query_labels],
    dtype=object,
)
ds_stim.plots.mapping_score(
    mapping,
    layout_key="RNA_UMAP",
    target_groups=score_groups,
    size_by_score=True,
    figsize=(14, 3.4),
)
```

Grey points received no weight from that query group.
Coloured points are the reference cells that attracted it; point size scales with score so sparse hits stay visible.
A useful map lights up the matching reference population.
Concentration in an unrelated pocket suggests a domain shift or a feature-alignment problem.

## 5. Transfer labels and inspect evidence

Label transfer aggregates neighbour weights for each query cell.
The winning label must clear `threshold_fraction`; otherwise Scarf returns `NA`.
A high vote fraction only means the neighbours agreed.
It is not a calibrated probability that the label is biologically correct.

```{code-cell} ipython3
transferred_labels = ds_stim.get_target_classes(
    mapping,
    reference_class_group="cluster_labels",
    threshold_fraction=0.6,
)
ds_stim.cells.insert(
    "transferred_labels",
    transferred_labels.to_numpy(),
    overwrite=True,
)
accepted = transferred_labels.notna() & transferred_labels.ne("NA")
accepted.value_counts().rename(
    index={True: "accepted", False: "abstained"}
).rename("query cells")
```

Plot the transferred labels on the query UMAP.
The `NA` category is abstention geography: those cells did not clear the vote threshold.

```{code-cell} ipython3
ds_stim.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=["cluster_labels", "transferred_labels"],
    figsize=(10, 4),
)
```

`get_target_classes` sets `NA` when the winning vote fraction is below `threshold_fraction`, when neighbour votes tie, or when the cell is uninformative.
`mapping_evidence` plots diagnostics from `get_target_label_evidence`; they do not trigger abstention on their own:

- `voteFraction`: how much neighbour weight supports the winning label
- `topTwoMargin`: how far the winner sits above the runner-up
- `referenceDistancePercentile`: how unusual the query cell is relative to reference neighbour distances

To force abstention by distance, pass `max_distance` to the evidence APIs.

```{code-cell} ipython3
ds_stim.plots.mapping_evidence(
    mapping,
    reference_class_group="cluster_labels",
    target_groups=query_labels,
    metrics=("voteFraction", "topTwoMargin", "referenceDistancePercentile"),
    kind="box",
    threshold_fraction=0.6,
    figsize=(14, 4),
)
```

Because this query carries author labels, we can compare known labels with transferred labels.

```{code-cell} ipython3
ds_stim.plots.mapping_confusion(
    mapping,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    normalize="true",
    threshold_fraction=0.6,
)
```

The diagonal is recall within each known query label.
Off-diagonal blocks are systematic swaps.
The `NA` column is abstention.
Watch the monocyte rows: stimulated CD14 Mono and DC often spill into CD16 Mono rather than a clean match, which is a domain-shift failure mode rather than a plotting artifact.

Because known labels are available, `mapping_calibration` shows how label accuracy trades off against retained coverage as the vote threshold rises.
The marker is the `threshold_fraction` used above.
Higher thresholds keep fewer cells and usually raise accuracy among the cells that remain.

```{code-cell} ipython3
ds_stim.plots.mapping_calibration(
    mapping,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    chosen_threshold=0.6,
)
```

## 6. Mapping scores by reference cluster

For a focused query population, ask which reference clusters absorbed the mapping weight.
Per-reference-cell scores are mostly zero, so cell-level box plots collapse to a flat line even when a few reference cells carry real weight.
Sum the raw (non-log) scores within each reference cluster instead.
That score mass is readable, and the sized embedding below keeps the same sparse scores on the reference UMAP.
The monocyte columns are the place to connect back to the confusion off-diagonals.

```{code-cell} ipython3
focus_labels = ("CD 14 Mono", "CD16 Mono", "NK")
focus_groups = np.array(
    [label if label in focus_labels else "other" for label in query_labels],
    dtype=object,
)
assert mapping.reference is not None
ref_classes = np.asarray(
    mapping.reference.fetch_cell_column("cluster_labels"),
    dtype=object,
)
score_mass: dict[str, pd.Series] = {}
for group, values in ds_stim.get_mapping_score(
    mapping,
    target_groups=focus_groups,
    log_transform=False,
):
    if group == "other":
        continue
    score_mass[str(group)] = (
        pd.Series(np.asarray(values, dtype=np.float64), index=ref_classes)
        .groupby(level=0, sort=False)
        .sum()
    )
score_mass_table = pd.DataFrame(
    {label: score_mass[label] for label in focus_labels if label in score_mass}
).fillna(0.0)
score_mass_table.loc[
    score_mass_table.max(axis=1).sort_values(ascending=False).index
].round(3)
```

```{code-cell} ipython3
ds_stim.plots.mapping_score(
    mapping,
    layout_key="RNA_UMAP",
    target_groups=focus_groups,
    size_by_score=True,
    figsize=(14, 3.4),
)
```

NK should put most score mass on the matching reference NK cluster and light up that pocket on the UMAP.
CD14 Mono often spreads toward CD16 Mono rather than a tight CD14-only peak; that matches the monocyte swaps in the confusion matrix.
Diffuse score mass across many unrelated clusters is a reason to inspect feature coverage or the query composition.

```{raw} html
<span id="reference-atlas-mapping"></span>
```

## 7. Reload a prepared mapping

In a later session you reopen the reference store and load the named mapping reference, then reload the query mapping by name:

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference()
reloaded_mapping = ds_stim.get_mapping_result(
    "stim",
    reference=reference,
    query_assay="RNA",
)
reloaded_mapping.mapping_name, reloaded_mapping.n_cells
```

Building and reusing a Symphony-style fixed reference is covered in {doc}`reference_atlases`.

Common failures include mapping before the reference exists, ignoring feature mismatch, treating vote support as a probability, using a biological condition as a correction batch, and transferring labels without an abstention path.

See {doc}`../reference/api/mapping` for method contracts and {doc}`../reference/api/plotting` for the diagnostic plotting signatures used above.
