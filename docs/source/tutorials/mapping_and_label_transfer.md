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

Mapping places query cells onto a fixed reference without merging count
matrices or moving reference cells. It is useful when new datasets should be
compared against the same graph and annotation. This guide performs direct KNN
mapping, examines evidence, transfers labels, and projects the query into the
reference layout.

Reference cells must already have a PCA model, neighbour index, graph, and
annotations. Query cells, called targets in the API, need a compatible assay but
do not need their own graph.

## Open the reference and query

```{code-cell} ipython3
import scarf

scarf.configure_output(level="WARNING", progress=False)

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

ds_ctrl = scarf.DataStore(f"{ctrl_path}/data.zarr", nthreads=4)
ds_stim = scarf.DataStore(f"{stim_path}/data.zarr", nthreads=4)
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

The layouts were fitted independently, so their coordinates are not directly
comparable. Their shared label vocabulary is used below to evaluate mapping.

## Map query cells

`run_mapping` aligns query features to the ordered reference feature set,
applies the reference normalization and PCA scaling, then queries the immutable
reference index. Query cells are never inserted into that index.

```{code-cell} ipython3
ds_ctrl.run_mapping(
    target_assay=ds_stim.RNA,
    target_name="stim",
    target_feat_key="hvgs_ctrl",
    save_k=5,
    missing_feature_policy="zero",
)
```

The default `zero` policy treats absent query features as zero normalized
expression. Use `error` when complete overlap is required. `intersection`
builds an isolated overlap-only index for a controlled comparison and does not
alter the authoritative reference graph.

Mapping scores show where query-neighbour evidence lands across the reference.
The public plotting method handles zero scores safely and uses colour rather
than invalid point sizes.

```{code-cell} ipython3
ds_ctrl.plots.mapping_score(
    target_name="stim",
    layout_key="RNA_UMAP",
)
```

A useful map distributes evidence over reference regions that match the query
composition. Concentration in a small, unrelated region suggests feature
alignment, scaling, or biological-compatibility problems.

## Transfer labels and inspect evidence

Label transfer aggregates edge weights from each query cell's saved reference
neighbours. The winning vote must pass `threshold_fraction`; otherwise the
result is `NA`. A vote fraction is neighbour support, not a calibrated class
probability.

```{code-cell} ipython3
transferred_labels = ds_ctrl.get_target_classes(
    target_name="stim",
    reference_class_group="cluster_labels",
    threshold_fraction=0.6,
)
ds_stim.cells.insert(
    "transferred_labels",
    transferred_labels.to_numpy(),
    overwrite=True,
)
```

`mapping_evidence` summarizes vote fraction, top-two margin, entropy, and
reference-distance percentile. Low vote fraction, a small margin, or a large
reference-distance percentile supports abstention rather than a forced label.

```{code-cell} ipython3
ds_ctrl.plots.mapping_evidence(
    target_name="stim",
    reference_class_group="cluster_labels",
    target_groups=ds_stim.cells.fetch("cluster_labels"),
    metrics=("voteFraction", "topTwoMargin"),
    kind="box",
    threshold_fraction=0.6,
    figsize=(12, 4),
)
```

The Kang query carries known labels, so confusion can be inspected directly.
This is evaluation data, not information available for an unlabelled query.

```{code-cell} ipython3
ds_ctrl.plots.mapping_confusion(
    target_name="stim",
    reference_class_group="cluster_labels",
    known_labels=ds_stim.cells.fetch("cluster_labels"),
    normalize="true",
    threshold_fraction=0.6,
)
```

The diagonal measures recall within each known query label. Off-diagonal blocks
show systematic transfer between label classes; an `NA` column represents
abstention.

## Calibrate an acceptance threshold

Calibration plots held-out label accuracy against retained coverage over a
range of evidence thresholds. It does not transform vote fractions into class
probabilities. The known stimulated labels serve as a held-out dataset relative
to the control reference. For deployment, calibrate on donors and batches that
are exchangeable with future queries.

```{code-cell} ipython3
ds_ctrl.plots.mapping_calibration(
    target_name="stim",
    reference_class_group="cluster_labels",
    known_labels=ds_stim.cells.fetch("cluster_labels"),
    metric="voteFraction",
    chosen_threshold=0.6,
)
```

In this held-out query, accuracy peaks near the marked threshold and falls when
the rule becomes either stricter or looser. Choose the operating point from the
cost of abstention and the required error rate, then validate it on a separate
evaluation set when possible.

## Project into the unchanged reference layout

Neighbour-weighted projection places query cells in an existing reference
layout without changing reference coordinates. The plot can colour the query by
known labels here or by transferred labels for an unlabelled query.

```{code-cell} ipython3
ds_ctrl.plots.mapping_projection(
    target_name="stim",
    reference_layout_key="RNA_UMAP",
    reference_groups="cluster_labels",
    target_groups=ds_stim.cells.fetch("cluster_labels"),
    ref_name="control",
)
```

Projection is a diagnostic view, not a new graph. For an exploratory joint
layout that allows the reference positions to move, use `run_unified_umap` and
`plots.unified_embedding`. Do not treat such a layout as a stable atlas
coordinate system.

```{raw} html
<span id="reference-atlas-mapping"></span>
```

## Reusable mapping references

Building, reloading, correcting, self-mapping, and validating a
Symphony-style fixed reference now lives in {doc}`reference_atlases`.

Common failures include mapping before the reference graph exists, ignoring
feature mismatch policy, treating vote support as a probability, using a
biological condition as a correction batch, and transferring labels without an
abstention path.

See the {doc}`../reference/api/mapping` for exact mapping method contracts and
the {doc}`../reference/api/plotting` for the diagnostic plotting signatures used
above.
