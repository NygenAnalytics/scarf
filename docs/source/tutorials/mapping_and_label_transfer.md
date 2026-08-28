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
This allows you to keep one reference atlas unchanged, place new query cells onto it, and transfer labels from reference neighbours.

It does three things in order:

1. Align the query features to the reference feature panel.
2. Project each query cell into the reference PCA space and find nearest reference neighbours.
3. Use those neighbours to transfer labels and score how much of the query landed on each reference cell.

It does not merge count matrices, retrain the reference graph, or move reference cells.
When sources must be analysed together in one store, start with {doc}`dataset_merging` and {doc}`batch_correction` instead.

In this tutorial, we will be mapping interferon-stimulated PBMCs onto a control PBMC reference from the same Kang study.
The shared author labels let us evaluate the result.

Mapping currently supports RNA queries. The catalog stores were rebuilt with the current count
layout, and documentation execution downloads separate writable copies. The reference and query
must remain different stores.

For a reusable Symphony-style atlas, see {doc}`reference_atlases`.

## 1. Open the reference and query

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

repository = scarf.cytebase.connect("scarf_docs")

ctrl_path = repository.download_dataset(
    name="kang_15K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds_ctrl = scarf.DataStore(
    f"{ctrl_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
```

```{code-cell} ipython3
stim_path = repository.download_dataset(
    name="kang_14K_ifnb-pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds_stim = scarf.DataStore(
    f"{stim_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
```

The two page-local stores keep reference and query lineage isolated.

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

Package the frozen reference run's neighbour chain as an immutable mapping reference.

```{code-cell} ipython3
run = ds_ctrl.pipeline.open(label="docs_default")
reference_layout = run["umap"]
reference_ref = ds_ctrl.build_mapping_reference(run["neighbors"])
reference = ds_ctrl.get_mapping_reference(reference_ref)
```

The completed `MappingReference` is immutable.
Its `feature_selection` field pins the exact reference feature artifact rather than a metadata key.
Its feature order, scaling, PCA loadings, neighbour index, and selected cells stay fixed in the reference datastore.
`reference_layout` is the immutable UMAP from the same run and is used only to show where query
weight landed.
This example uses a plain PCA reference.
A Symphony reference instead passes Harmony-corrected neighbours into `build_mapping_reference`.

## 3. Map the query

`run_mapping` runs on the writable query datastore.
It aligns query features to the reference panel, applies the reference normalization and scaling, projects into the reference PCA space, and stores the nearest neighbours.
Query cells are never inserted into the reference index.

```{code-cell} ipython3
query_cell_selection = ds_stim.snapshot_cell_selection("I")
mapping_ref = ds_stim.run_mapping(
    reference,
    query_cell_selection,
    query_assay="RNA",
    save_k=5,
    missing_feature_policy="reference_mean",
)
```

To check the neighbour arrays, reload the projection with `get_mapping_result()`.
Use `load_arrays=True` to get two arrays with one row per query cell and `save_k` columns.
`indices` identifies the nearest reference cells.
`distances` contains finite distances between each query cell and its reference neighbours.

```{code-cell} ipython3
mapping = ds_stim.get_mapping_result(
    mapping_ref,
    reference=reference,
    load_arrays=True,
)
mapping.n_cells, int(mapping.indices.shape[1]), mapping.indices[:3], mapping.distances[:3]
```

`reference_mean` fills an absent query feature with the reference mean, which becomes zero after reference scaling.
Use `zero` to fill with a normalized zero, or `error` when complete feature overlap is required.

`mapping.diagnostics["queryScaledDispersion"]` is calculated from comparing query spread with the reference after scaling.
Values near 1 mean the query occupies a similar region of feature space.
Values much below 1 mean the query is compressed toward the centre of the reference cloud and neighbour labels become less trustworthy.

```{code-cell} ipython3
mapping.diagnostics
```

## 4. Where did the query land?

A mapping score tells you which reference cells received neighbour weight from the query. This can be plotted on the reference UMAP.
Since one panel for the whole query becomes hard to read due to the weight being spread across many cells,
we can split by a few known query populations to see whether each population lands on the matching reference region.

```{code-cell} ipython3
query_labels = np.asarray(ds_stim.cells.fetch("cluster_labels")).astype(str)
focus = {"CD 14 Mono", "CD4 Memory T", "CD4 naive T", "NK"}
score_groups = np.array(
    [label if label in focus else "other" for label in query_labels],
    dtype=object,
)
ds_stim.plots.mapping_score(
    mapping_ref,
    reference=reference,
    layout=reference_layout,
    target_groups=score_groups,
    size_by_score=True,
    figsize=(14, 3.4),
)
```

Each panel here shows grey points which received no weight from that query group.
Coloured points are the reference cells that are neighbours to the cells in the query group; point size scales with score so sparse hits stay visible.
A useful mapping of the query group will light up the matching reference population.
Alternatively, concentration in an unrelated pocket suggests a domain shift or a feature-alignment problem.

## 5. Transfer labels and inspect evidence

Label transfer aggregates neighbour weights for each query cell.
The winning label must clear `threshold_fraction`; otherwise Scarf returns `NA`.
A high vote fraction only means the neighbours agreed.
It is not a calibrated probability that the label is biologically correct.

```{code-cell} ipython3
transferred_labels = ds_stim.get_target_classes(
    mapping_ref,
    reference_class_group="cluster_labels",
    reference=reference,
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
In the plot below, the `NA` category is abstention geography, i.e. those cells did not clear the vote threshold.

`get_target_classes()` sets `NA` under the circumstances when:
- the winning vote fraction is below `threshold_fraction`
- neighbour votes tie, or
- the cell is uninformative.


```{code-cell} ipython3
ds_stim.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=["cluster_labels", "transferred_labels"],
    figsize=(10, 4),
)
```

`mapping_evidence` plots diagnostics from `get_target_label_evidence`; they do not trigger abstention on their own:

- `voteFraction`: how much neighbour weight supports the winning label
- `topTwoMargin`: how far the winner sits above the runner-up
- `referenceDistancePercentile`: how unusual the query cell is relative to reference neighbour distances

To force abstention by distance, pass `max_distance` to the evidence APIs.

```{code-cell} ipython3
ds_stim.plots.mapping_evidence(
    mapping_ref,
    reference=reference,
    reference_class_group="cluster_labels",
    target_groups=query_labels,
    metrics=("voteFraction", "topTwoMargin", "referenceDistancePercentile"),
    kind="box",
    threshold_fraction=0.6,
    figsize=(14, 4),
)
```

Because this query dataset also carries original author labels, we can compare these known labels
with the transferred labels.

```{code-cell} ipython3
ds_stim.plots.mapping_confusion(
    mapping_ref,
    reference=reference,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    normalize="true",
    threshold_fraction=0.6,
)
```

The diagonal is recall within each known query label.
Off-diagonal blocks are systematic swaps.
The `NA` predicted label here represents the abstention, cells that did not receive a transferred label.
Take note of the monocyte rows in this figure: stimulated CD14 Mono and DC often spill into CD16 Mono rather than a clean match, which is a domain-shift failure mode rather than a plotting artifact.

Because known labels are available, `mapping_calibration` shows how label accuracy trades off against retained coverage as the vote threshold rises.
The red marker is the `threshold_fraction` used above.
Higher thresholds keep fewer cells and usually raise accuracy among the cells that remain.

```{code-cell} ipython3
ds_stim.plots.mapping_calibration(
    mapping_ref,
    reference=reference,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    chosen_threshold=0.6,
)
```

## 6. Mapping scores by reference cluster

For a focused query population, you can find which reference clusters absorbed the mapping weight.
Per-reference-cell scores are mostly zero, so cell-level box plots collapse to a flat line even when a few reference cells carry real weight.
Instead, we can sum the raw (non-log) scores within each reference cluster.
That score now becomes readable, and still keep the same sparse scores on the reference UMAP as shown in the embeddings below.

Here we will focus on the monocyte groups we saw in the confusion matrix which were off-diagonal. We will also include NK group to use as a comparison.

```{code-cell} ipython3
focus_labels = ("CD 14 Mono", "CD16 Mono", "NK")
focus_groups = np.array(
    [label if label in focus_labels else "other" for label in query_labels],
    dtype=object,
)
ref_classes = np.asarray(
    reference.fetch_cell_column("cluster_labels"),
    dtype=object,
)
score_mass: dict[str, pd.Series] = {}
for group, values in ds_stim.get_mapping_score(
    mapping_ref,
    target_groups=focus_groups,
    reference=reference,
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
    mapping_ref,
    reference=reference,
    layout=reference_layout,
    target_groups=focus_groups,
    size_by_score=True,
    figsize=(14, 3.4),
)
```

Most of the NK scores matches the reference NK cluster and has clearly lit up in the reference NK cluster on the UMAP.
CD14 Mono scores have spread toward CD16 Mono rather than CD14 Mono; this matches the monocyte swaps seen previously in the confusion matrix.
If the scores were spread across multiple unrelated clusters, then it would be reasonable to inspect feature coverage or the composition of the query dataset.

```{raw} html
<span id="reference-atlas-mapping"></span>
```

## 7. Reload a prepared mapping

In a later session, retain the mapping-reference and projection artifact refs, reopen both stores, and reload the exact results:

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference(reference_ref)
reloaded_mapping = ds_stim.get_mapping_result(
    mapping_ref,
    reference=reference,
)
reloaded_mapping.n_cells, reloaded_mapping.correction_method
```

Building and reusing a Symphony-style fixed reference is covered in {doc}`reference_atlases`.

For troubleshooting, common failures include mapping before the reference exists, ignoring feature mismatch, treating vote support as a probability, using a biological condition as a correction batch, and transferring labels without an abstention path.

See {doc}`../reference/api/mapping` for method contracts and {doc}`../reference/api/plotting` for the diagnostic plotting signatures used above.
