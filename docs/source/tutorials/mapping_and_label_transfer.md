---
description: Prepare and reuse a fixed reference, map query cells, and transfer labels with abstention.
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
The prepared mapping reference is also the reusable atlas: later queries use the same feature panel,
projection model, reference coordinates, and neighbour index.

It does three things in order:

1. Align the query features to the reference feature panel.
2. Project each query cell into the reference PCA space and find nearest reference neighbours.
3. Use those neighbours to transfer labels and score how much of the query landed on each reference cell.

It does not merge count matrices, retrain the reference graph, or move reference cells.
When sources must be analysed together in one store, start with {doc}`dataset_merging` and {doc}`batch_correction` instead.

In this tutorial, we will be mapping interferon-stimulated PBMCs onto a control PBMC reference from the same Kang study.
The shared author labels let us evaluate the result.

Mapping currently supports RNA queries. Keep the reference and query in separate stores;
the query store must be writable so Scarf can save the mapping and transferred labels.

## 1. Open the reference and query

```{code-cell} ipython3
# Work with numeric arrays and cell masks.
import numpy as np
# Summarize the query populations in a labeled table.
import pandas as pd

# Open count stores and run Scarf analyses.
import scarf
# Give the reference and query plots descriptive labels.
from scarf.plotting import CellField

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Connect to the public example-data repository.
repository = scarf.cytebase.connect("scarf_docs")

# Download the unstimulated reference cells.
ctrl_path = repository.download_dataset(
    name="kang_15K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the reference count store.
ds_ctrl = scarf.DataStore(
    f"{ctrl_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
# Inspect the opened store's cells and features.
ds_ctrl
```

```{code-cell} ipython3
# Download the interferon-stimulated query cells.
stim_path = repository.download_dataset(
    name="kang_14K_ifnb-pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the query store where mapping results will be saved.
ds_stim = scarf.DataStore(
    f"{stim_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
# Inspect the opened store's cells and features.
ds_stim
```

First, look at the author labels in both datasets.

```{code-cell} ipython3
# Inspect the published labels in the control reference.
ds_ctrl.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=CellField("cluster_labels", label="Control reference"),
)
# Inspect the published labels in the stimulated query.
ds_stim.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=CellField("cluster_labels", label="Stimulated query"),
)
```

These UMAP layouts were fitted independently, so their coordinates are not comparable.
Mapping keeps the control layout fixed and reports where query weight lands on reference cells.
The shared label vocabulary is what we use for evaluation.

## 2. Prepare a labelled, reusable reference

The control store already contains an analysis named `docs_default`. Use its neighbours to
prepare a reference that later queries can share.

```{code-cell} ipython3
# Open the saved analysis and its exact results.
run = ds_ctrl.pipeline.open(label="docs_default")
# Keep the reference UMAP fixed for mapping-score plots.
reference_layout = run["umap"]
# Prepare a reusable reference from the saved neighbors.
reference_ref = ds_ctrl.build_mapping_reference(run["neighbors"])
# Load the saved reference model.
reference = ds_ctrl.get_mapping_reference(reference_ref)
# Inspect the reference saved for future query datasets.
reference_ref
```

The completed `MappingReference` is immutable.
Its `feature_selection` field pins the exact reference feature artifact rather than a metadata key.
Its feature order, scaling, PCA loadings, neighbour index, and selected cells stay fixed in the reference datastore.
`reference_layout` is the immutable UMAP from the same run and is used only to show where query
weight landed.
The mapping reference does not contain that layout or the reference labels. Label transfer
freezes the labels it uses when it runs (section 5).

## 3. Map the query

`run_mapping` runs on the writable query datastore.
It aligns query features to the reference panel, applies the reference normalization and scaling, projects into the reference PCA space, and stores the nearest neighbours.
Query cells are never inserted into the reference index.

We map the cells of the query's own pipeline run, so that the results can be drawn on that run's
UMAP later. The default mapping keeps the three nearest reference neighbours for each query cell.

```{code-cell} ipython3
# Open the saved analysis of the query cells.
query_run = ds_stim.pipeline.open(label="docs_default")
# Keep the query UMAP for displaying transferred labels.
query_layout = query_run["umap"]
# Map the selected query cells onto the fixed reference.
mapping_ref = ds_stim.run_mapping(reference, query_run['analysis_cell_selection'])
# Inspect the saved query-mapping reference.
mapping_ref
```

Reload the saved mapping to inspect its diagnostics:

```{code-cell} ipython3
# Load the saved mapping and its diagnostics.
mapping = ds_stim.get_mapping_result(mapping_ref, reference=reference)
# Inspect feature coverage and query mapping diagnostics.
mapping.diagnostics
```

By default, an absent query feature is filled with the reference mean, which becomes zero after
reference scaling. Check `featureCoverage` in the diagnostics before interpreting transferred labels.

Cells with no counts in the measured reference features are flagged as uninformative.
They cannot provide evidence for a transferred label.

## 4. Where did the query land?

A mapping score tells you which reference cells received neighbour weight from the query. This can be plotted on the reference UMAP.
Since one panel for the whole query becomes hard to read due to the weight being spread across many cells,
we can split by a few known query populations to see whether each population lands on the matching reference region.
The author labels of the mapped cells are read in the order of the run's cells.

```{code-cell} ipython3
# Find the query cell rows included in the saved analysis.
mapped_rows = np.flatnonzero(query_run.cells.fetch_all("I"))
# Read the published labels in mapped-cell order.
query_labels = np.asarray(ds_stim.cells.fetch_all("cluster_labels"))[mapped_rows]
# Use text labels consistently when forming plot groups.
query_labels = query_labels.astype(str)
# Choose a few known populations for separate mapping-score panels.
focus = {"CD 14 Mono", "CD4 Memory T", "CD4 naive T", "NK"}
# Keep the selected populations and group the remaining labels as other.
score_groups = np.array(
    [label if label in focus else "other" for label in query_labels],
    dtype=object,
)
# Count cells in each mapping-score group.
pd.Series(score_groups, name="query population").value_counts().to_frame("cells")
```

Show where each query population contributes weight on the reference map:

```{code-cell} ipython3
# Show where each query group contributes weight on the reference.
ds_stim.plots.mapping_score(
    mapping_ref,
    reference=reference,
    layout=reference_layout,
    target_groups=score_groups,
    size_by_score=True,
    figsize=(12, 7),
)
```

Each panel here shows grey points which received no weight from that query group.
Coloured points are the reference cells that are neighbours to the cells in the query group; point size scales with score so sparse hits stay visible.
A useful mapping of the query group will light up the matching reference population.
Alternatively, concentration in an unrelated pocket suggests a domain shift or a feature-alignment problem.

## 5. Transfer labels and inspect evidence

Label transfer aggregates neighbour weights for each query cell.
By default, a unique winning label needs at least half of the total vote weight.
Otherwise the cell abstains and gets no label.
A high vote fraction only means the neighbours agreed.
It is not a calibrated probability that the label is biologically correct.

`run_label_transfer` saves the result as an immutable `label_transfer` artifact in the query
datastore and returns its reference. It first freezes the reference labels it reads into the query
datastore, so later edits to the reference annotations cannot change this result.

```{code-cell} ipython3
# Transfer reference labels with the default abstention threshold.
transfer_ref = ds_stim.run_label_transfer(
    mapping_ref,
    reference=reference,
    reference_labels="cluster_labels",
)
# Load the transferred labels and their supporting evidence.
transfer = ds_stim.get_label_transfer(transfer_ref)
# Count cells in each reported category.
transfer.labels.notna().value_counts().rename(
    index={True: "labelled", False: "abstained"}
).rename("query cells")
```

An abstained cell has a missing label, and the `abstentionReason` column of `transfer.evidence`
says why:

- `below_threshold`: the winning vote fraction is below `threshold_fraction`
- `tied_vote`: two labels received the same neighbour weight
- `uninformative_cell`: the cell has no counts in any reference feature that the query measured
- `no_labeled_neighbors`: no neighbour has a usable reference label
- `beyond_max_distance`: the nearest reference neighbour is farther than `max_distance`, when one is set

Uninformative cells also add nothing to mapping scores.

```{code-cell} ipython3
# Count cells in each reported category.
transfer.evidence["abstentionReason"].value_counts()
```

Plot the transferred labels on the query UMAP.
A label artifact colours an embedding directly, so nothing is written into the query cell metadata.
Abstained cells are drawn as missing, which shows the geography of abstention.

```{code-cell} ipython3
# Compare published and transferred labels on the query UMAP.
ds_stim.plots.embedding(
    layout=query_layout,
    color_by=["cluster_labels", transfer_ref],
    figsize=(10, 4),
)
```

`mapping_evidence` plots the evidence saved with the transfer. These metrics do not trigger
abstention on their own:

- `voteFraction`: how much neighbour weight supports the winning label
- `topTwoMargin`: how far the winner sits above the runner-up
- `referenceDistancePercentile`: how unusual the query cell is relative to reference neighbour distances

To also abstain by distance, pass `max_distance` to `run_label_transfer`. That saves a second
transfer and leaves this one unchanged.

```{code-cell} ipython3
# Inspect the evidence supporting transferred labels.
ds_stim.plots.mapping_evidence(
    transfer_ref,
    target_groups=query_labels,
    metrics=("voteFraction", "topTwoMargin", "referenceDistancePercentile"),
    kind="box",
    figsize=(14, 4),
)
```

Because this query dataset also carries original author labels, we can compare these known labels
with the transferred labels.

```{code-cell} ipython3
# Compare transferred labels with the known query labels.
ds_stim.plots.mapping_confusion(
    transfer_ref,
    known_labels=query_labels,
    normalize="true",
)
```

Cells where the known and predicted labels match show recall within each known query label.
Blocks between different labels show systematic swaps.
The `Abstained` column holds the cells that did not receive a transferred label.
Pay particular attention to the monocyte rows. Stimulation can change expression enough that a
query population maps to another reference label. Inspect such swaps before accepting the labels.

Because known labels are available, `mapping_calibration` shows how label accuracy trades off against retained coverage as the vote threshold rises.
It applies each threshold to the candidate labels saved with the transfer, so nothing is recomputed.
Coverage is the share of cells with a known label that a threshold keeps. Cells that abstained
without evidence stay in that share and are never kept, so the curve counts the same cells as the
confusion matrix.
The red marker is the transfer's own `threshold_fraction`.
Higher thresholds keep fewer cells. Use this curve to check whether the retained labels are
also more accurate in this dataset.

```{code-cell} ipython3
# Compare label accuracy and retained coverage across thresholds.
ds_stim.plots.mapping_calibration(transfer_ref, known_labels=query_labels)
```

## 6. Choose stricter settings when needed

The first pass used the defaults. If too many uncertain labels remain, raise
`threshold_fraction` in `run_label_transfer`. For example, `threshold_fraction=0.6` requires
60% of the neighbour weight to support the winning label. It saves a separate transfer,
so you can compare it with the first one. Use the confusion matrix and calibration curve to
judge the tradeoff between coverage and agreement with known labels.

You can also change `save_k` in `run_mapping` to consider more reference neighbours.
This creates a new mapping and can change which populations receive weight. More neighbours
are not automatically better, especially near boundaries between cell types.

For missing features, `missing_feature_policy="error"` requires complete overlap;
`"zero"` fills an absent feature with a normalized zero. The default, `"reference_mean"`,
is the path used above.

`mapping.diagnostics["queryScaledDispersion"]` is calculated from comparing query spread with the reference after scaling.
Only informative query cells and the reference features the query measured enter it, so the missing-feature fill does not change it.
Values near 1 mean a similar average scaled distance from the reference centre;
they do not establish matching cell types.
Values much below 1 mean the query is compressed toward the centre of the reference cloud and neighbour labels become less trustworthy.
RNA normalization renormalizes counts over the selected features by default (`renormalize_subset=True`).
With that setting and `featureCoverage` below 1, query cells are renormalized over fewer features than the reference used, so the value is not comparable with 1.

A query cell is uninformative when its raw counts are zero in every reference feature that the query measured.
Such a cell carries no query evidence. It is flagged in `mapping.uninformative` and counted by `mapping.diagnostics["uninformativeCellCount"]`.

This example uses a plain PCA reference. Use a Harmony-backed Symphony reference only when
both reference and query have defensible technical-batch metadata. Stimulation and disease
are biological conditions, so they should not be substituted for technical batches.

```{raw} html
<span id="reference-atlas-mapping"></span>
```

## 7. Reuse the reference and saved labels

In a later session, retain the mapping-reference, projection, and label-transfer artifact refs,
reopen both stores, and reload the exact results. A saved transfer loads from the query datastore
alone:

```{code-cell} ipython3
# Load the saved reference model.
reference = ds_ctrl.get_mapping_reference(reference_ref)
# Reload the exact mapping without recomputing it.
reloaded_mapping = ds_stim.get_mapping_result(mapping_ref, reference=reference)
# Reload the saved labels from the query store.
reloaded_transfer = ds_stim.get_label_transfer(transfer_ref)
# Check the reloaded cell count, correction, label source, and threshold.
(
    reloaded_mapping.n_cells,
    reloaded_mapping.correction_method,
    reloaded_transfer.reference_label_source,
    reloaded_transfer.threshold_fraction,
)
```

For repeated use, reopen the prepared reference datastore read-only and run each mapping in a
separate writable query datastore. If query counts come from a read-only source or the same physical
store used to prepare the reference, use `mount_datastore` to create that separate query store.

Reference labels are frozen when labels are transferred. `run_label_transfer` copies the labels it
reads into the query datastore as a `reference_labels` artifact, which records their source column
or artifact and a fingerprint of their values. Transferring again with unchanged labels and settings
reuses the saved transfer. After the reference annotations change, the same call saves a new
transfer next to the old one, and the old transfer keeps its labels. Reference labels can also be a
cell-label artifact of the reference datastore, such as a clustering or a `smart_label`
relabelling. Retain the layout artifact separately when mapping scores must be displayed on the
original reference UMAP.

The lineage of a transfer follows its inputs into the reference datastore: from the query labels
through the threshold, the frozen reference labels, and the projection, to the reference model and
the cells it was built from.

```{code-cell} ipython3
# Display the saved result and its upstream inputs.
ds_stim.lineage(transfer_ref, references=reference)
```

Validate a reused atlas with feature coverage, mapping evidence, abstention, and score concentration.
When independent query labels exist, also inspect confusion and threshold calibration. A visually
plausible embedding alone does not validate transferred labels.

For troubleshooting, common failures include mapping before the reference exists, ignoring feature mismatch, treating vote support as a probability, using a biological condition as a correction batch, and transferring labels without an abstention path.

See {doc}`../reference/api/mapping` for method contracts and {doc}`../reference/api/plotting` for the diagnostic plotting signatures used above.
