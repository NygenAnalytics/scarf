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

Mapping answers one question: where do new cells sit on a fixed reference atlas?

It does three things in order:

1. Align the query features to the reference feature panel.
2. Project each query cell into the reference PCA space and find nearest reference neighbours.
3. Use those neighbours to transfer labels and score how much of the query landed on each reference cell.

It does not merge count matrices, retrain the reference graph, or move reference cells.

This tutorial maps interferon-stimulated PBMCs onto a control PBMC reference from the same Kang study. The shared author labels let us evaluate the result. For a reusable Symphony-style atlas, see {doc}`reference_atlases`.

Mapping currently supports RNA queries. The prepared reference may be reopened read-only, but the query must be a different writable store.

## Open the reference and query

```{code-cell} ipython3
import numpy as np

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

These UMAP layouts were fitted independently, so their coordinates are not
comparable. Mapping will later place stimulated cells into the control layout.
The shared label vocabulary is what we use for evaluation.

## Prepare a labelled reference

The published Kang annotation leaves about 6,000 of the 14,526 active control
cells unlabelled. Those cells are stored as the string `nan`. Label transfer
treats every distinct string as a class, so unlabelled reference cells would
compete for votes and obscure the real cell-type transfer.

Restrict the reference to annotated cells, then select variable features on
that same cell key so the feature panel matches the cells used for mapping.

```{code-cell} ipython3
labelled = ds_ctrl.cells.fetch_all("cluster_labels").astype(str) != "nan"
ds_ctrl.cells.insert(
    "annotated",
    ds_ctrl.cells.fetch_all("I").astype(bool) & labelled,
    overwrite=True,
)
print(
    "annotated reference cells:",
    int(ds_ctrl.cells.fetch_all("annotated").sum()),
)
ds_ctrl.mark_hvgs(
    cell_key="annotated",
    top_n=2000,
    min_cells=10,
    min_mean=-3,
    max_mean=2,
    max_var=6,
    show_plot=False,
)
```

```{code-cell} ipython3
normalized = ds_ctrl.run_normalization(cell_key="annotated", feat_key="hvgs")
pca = ds_ctrl.run_pca(normalized, dims=25, feat_scaling=True)
ann_index = ds_ctrl.build_ann_index(pca)
neighbors = ds_ctrl.query_neighbors(ann_index, k=17)
reference = ds_ctrl.build_mapping_reference(neighbors)
reference
```

The completed `MappingReference` is immutable. Its feature order, scaling,
PCA loadings, neighbour index, and selected cells stay fixed in the reference
datastore. This example uses a plain PCA reference. A Symphony reference instead
passes `run_harmony` output to `build_ann_index` before querying neighbours.

## Map the query

`run_mapping` runs on the writable query datastore. It aligns query features to
the reference panel, applies the reference normalization and scaling, projects
into the reference PCA space, and stores the nearest neighbours. Query cells are
never inserted into the reference index.

```{code-cell} ipython3
mapping = ds_stim.run_mapping(
    reference,
    "stim",
    query_assay="RNA",
    save_k=5,
    missing_feature_policy="reference_mean",
)
mapping
```

`reference_mean` fills an absent query feature with the reference mean, which
becomes zero after reference scaling. Use `zero` to fill with a normalized zero,
or `error` when complete feature overlap is required.

`mapping.diagnostics["queryScaledDispersion"]` compares query spread with the
reference after scaling. Values near 1 mean the query occupies a similar region
of feature space. Values much below 1 mean the query is compressed toward the
centre of the reference cloud and neighbour labels become less trustworthy.

```{code-cell} ipython3
mapping.diagnostics
```

## Where did the query land?

A mapping score tells you which reference cells received neighbour weight from
the query. Plot it on the reference UMAP. One panel for the whole query is hard
to read because the weight is spread across many cells. Split by a few known
query populations to see whether each population lands on the matching
reference region.

```{code-cell} ipython3
query_labels = np.asarray(ds_stim.cells.fetch("cluster_labels")).astype(str)
focus = {"CD 14 Mono", "CD4 naive T", "NK"}
score_groups = np.array(
    [label if label in focus else "other" for label in query_labels],
    dtype=object,
)
ds_stim.plots.mapping_score(
    mapping,
    layout_key="RNA_UMAP",
    target_groups=score_groups,
    figsize=(11, 3.4),
)
```

Grey points received no weight from that query group. Coloured points are the
reference cells that attracted it. A useful map lights up the matching
reference population. Concentration in an unrelated pocket suggests a domain
shift or a feature-alignment problem.

## Transfer labels and inspect evidence

Label transfer aggregates neighbour weights for each query cell. The winning
label must clear `threshold_fraction`; otherwise Scarf returns `NA`. A high
vote fraction only means the neighbours agreed. It is not a calibrated
probability that the label is biologically correct.

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

`mapping_evidence` summarizes vote fraction, top-two margin, entropy, and
reference-distance percentile. Low vote fraction, a small margin, or a large
reference-distance percentile supports abstention rather than a forced label.

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

Because this query carries author labels, we can compare known labels with
transferred labels. The `nan` row is unlabelled query cells, so it has no
correct answer.

```{code-cell} ipython3
ds_stim.plots.mapping_confusion(
    mapping,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    normalize="true",
    threshold_fraction=0.6,
)
```

The diagonal is recall within each known query label. Off-diagonal blocks are
systematic swaps. The `NA` column is abstention.

## Calibrate an acceptance threshold

Calibration plots held-out accuracy against retained coverage over a range of
evidence thresholds. It does not turn vote fractions into class probabilities.
Here the stimulated labels act as a held-out evaluation set. For deployment,
calibrate on donors and batches that match future queries.

```{code-cell} ipython3
ds_stim.plots.mapping_calibration(
    mapping,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    metric="voteFraction",
    chosen_threshold=0.6,
)
```

Choose the operating point from the cost of abstention and the required error
rate, then validate it on a separate evaluation set when possible.

## Project into the unchanged reference layout

Neighbour-weighted projection places query cells into the existing reference
UMAP without moving reference coordinates. The plot below keeps the reference
as a light background and colours only the labelled query cells.

```{code-cell} ipython3
projected_embedding = ds_stim.project_reference_embedding(
    mapping,
    reference_layout_key="RNA_UMAP",
    label="ref_UMAP",
)
projected_embedding
```

```{code-cell} ipython3
labelled_query = np.array(
    ["unlabelled" if label == "nan" else label for label in query_labels],
    dtype=object,
)
ds_stim.plots.mapping_projection(
    mapping,
    reference_layout_key="RNA_UMAP",
    target_groups=labelled_query,
    ref_name="control reference",
    reference_mode="background",
    figsize=(7.2, 5.2),
)
```

Projection is a diagnostic view, not a new graph. Stimulated monocytes and T
cells should land near their control counterparts. Residual shifts are expected
because interferon response moves cells off the control manifold.

```{raw} html
<span id="reference-atlas-mapping"></span>
```

## Reload a prepared mapping

Reload the reference by `ArtifactRef`. Reload the query projection by name with
an explicit reference and query assay:

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference(reference.ref)
reloaded_mapping = ds_stim.get_mapping_result(
    "stim",
    reference=reference,
    query_assay="RNA",
)
reloaded_mapping
```

Consumers also accept `reloaded_mapping.ref` when `reference=reference` is
provided. Building and reusing a Symphony-style fixed reference is covered in
{doc}`reference_atlases`.

Common failures include mapping before the reference exists, ignoring feature
mismatch, treating vote support as a probability, using a biological condition
as a correction batch, and transferring labels without an abstention path.

See {doc}`../reference/api/mapping` for method contracts and
{doc}`../reference/api/plotting` for the diagnostic plotting signatures used
above.
