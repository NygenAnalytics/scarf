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

Mapping answers a different question than merging. When two sources must be analyzed together in one store, you typically merge them and rebuild a joint graph ({doc}`dataset_merging`), but when you already hold a finished reference atlas for your dataset of interest, rebuilding everything around each new query throws away the work frozen in that reference. Mapping instead keeps the reference atlas fixed and places each new query cell onto it: the query is aligned to the reference feature panel, projected into the reference PCA space, and labeled from its nearest reference neighbors, while the reference cells never move and its graph is never retrained. The prepared reference is thus a reusable atlas, so later queries reuse the same feature panel, projection model, coordinates, and neighbor index without repeating the preparation.

In simple terms, this assumes you have an existing, high-quality dataset, and use that as a reference map for your new dataset. Here, we map interferon-stimulated PBMCs onto a control PBMC reference from the same study, where the shared cell type labels let us check the transferred labels against known answers.

Mapping currently supports only RNA-seq-based queries. For the analysis, keep the reference and query in separate stores; the query store must be writable so Scarf can save the mapping and transferred labels.

## Open the reference and query

```{code-cell} ipython3
import numpy as np
import pandas as pd

import scarf
from scarf.plotting import CellField

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
ds_ctrl
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
ds_stim
```

First, look at the author labels in both datasets for what cell types are listed.

```{code-cell} ipython3
ds_ctrl.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=CellField("cluster_labels", label="Control reference"),
)
ds_stim.plots.embedding(
    layout_key="RNA_UMAP",
    color_by=CellField("cluster_labels", label="Stimulated query"),
)
```

These UMAP layouts were fitted independently, so their coordinates are not comparable between the two datasets. Mapping keeps the control layout fixed and reports where query weight lands on reference cells. The shared annotation labels are what we use for evaluating our results.

## Prepare a labeled, reusable reference

The control store already contains the completed analysis, so we use its neighbors to prepare a reference that later queries can share.

```{code-cell} ipython3
run = ds_ctrl.pipeline.open(label="docs_default")
reference_layout = run["umap"]
reference_ref = ds_ctrl.build_mapping_reference(run["neighbors"])
reference = ds_ctrl.get_mapping_reference(reference_ref)
reference_ref
```

The completed `MappingReference` is unchangeable, with information like its `feature_selection` being artifacts to use for the analysis instead of selection keys in the metadata. Its feature order, scaling, PCA loadings, neighbor index, and selected cells stay fixed in the same reference datastore. The `reference_layout` is likewise an unchangeable artifact (UMAP in our case) from the same run and is used only to show where query weight landed. The mapping reference does not contain that layout or the reference labels, which live outside the reference itself and are supplied separately where needed.

## Map the query

To actually map the query dataset, we run the `run_mapping` function on the query datastore.
It aligns query features to the reference panel, applies the reference normalization and scaling, projects into the reference PCA space, and stores the nearest neighbors. From this, we can then visualize the results on the UMAP. For your own understanding, query cells are never inserted into the reference index; the reference cells are simply used for the mapping. The default mapping keeps the three nearest reference neighbors for each query cell, but this number can be modulated by you.

```{code-cell} ipython3
query_run = ds_stim.pipeline.open(label="docs_default")
query_layout = query_run["umap"]
mapping_ref = ds_stim.run_mapping(reference, query_run['analysis_cell_selection'])
mapping_ref
```

Reload the saved mapping to inspect its diagnostics:

```{code-cell} ipython3
mapping = ds_stim.get_mapping_result(mapping_ref, reference=reference)
mapping.diagnostics
```

By default, an absent query feature (gene) is filled with the reference mean, which becomes zero after reference scaling. This results in the missing feature contributing nothing to the projection, neither pulling the cell toward nor pushing it away from any reference neighborhood.

Cells with no counts in the measured reference features are flagged as uninformative. They cannot provide evidence for a transferred label, so their neighbor rows are stored but skipped, and they abstain with `uninformative_cell` rather than receiving an invented label.

## Understanding where query landed

A mapping score tells you which reference cells received neighbor weight from the query, but to visualize this, you can plot this onto the reference dataset's UMAP.
Since one panel for the whole query becomes hard to read due to the weight being spread across many cells, we can split by a few known query populations to see whether each population lands on the matching reference region.

```{code-cell} ipython3
mapped_rows = np.flatnonzero(query_run.cells.fetch_all("I"))
query_labels = np.asarray(ds_stim.cells.fetch_all("cluster_labels"))[mapped_rows]
query_labels = query_labels.astype(str)
focus = {"CD 14 Mono", "CD4 Memory T", "CD4 naive T", "NK"}
score_groups = np.array(
    [label if label in focus else "other" for label in query_labels],
    dtype=object,
)
pd.Series(score_groups, name="query population").value_counts().to_frame("cells")
```

Show where each query population contributes on the reference map:

```{code-cell} ipython3
ds_stim.plots.mapping_score(
    mapping_ref,
    reference=reference,
    layout=reference_layout,
    target_groups=score_groups,
    size_by_score=True,
    figsize=(12, 7),
)
```

Each panel here shows gray points which received no weight from that query group, with colored points representing the reference cells that are neighbors to the cells in the query group; point size scales with score so sparse hits stay visible. A useful mapping of the query group will light up the matching reference population. Alternatively, concentration in an unrelated pocket suggests a domain shift (as in cells being different) or a feature-alignment problem.

## Transfer labels and inspect evidence

Since label transfer aggregates neighbor weights for each query cell, each neighbor essentially votes for its own reference annotation, and the winning label is simply whichever reference annotation collects the most vote weight: that annotation is what the query cell gets mapped to. By default, a unique winning label needs at least half of the total vote weight (weight needs to be at least 50%), otherwise the cell abstains and gets no label.

`run_label_transfer` saves the result as an unchangeable `label_transfer` artifact in the query
datastore and returns its reference. It first freezes the reference labels it reads into the query
datastore, so later edits to the reference annotations cannot change the end result.

```{code-cell} ipython3
transfer_ref = ds_stim.run_label_transfer(
    mapping_ref,
    reference=reference,
    reference_labels="cluster_labels",
)
transfer = ds_stim.get_label_transfer(transfer_ref)
transfer.labels.notna().value_counts().rename(
    index={True: "labelled", False: "abstained"}
).rename("query cells")
```

Any cells that didn't get mapped can be analyzed as to why they weren't mapped by looking at the `abstentionReason` section, with the column of `transfer.evidence` says why:

- `below_threshold`: This indicates that the winning vote (weight) fraction is below `threshold_fraction`
- `tied_vote`: two labels received the same neighbor weight
- `uninformative_cell`: the cell has no counts in any reference feature that the query measured
- `no_labeled_neighbors`: no neighbor has a usable reference label
- `beyond_max_distance`: the nearest reference neighbor is farther than `max_distance`, when one is set

Uninformative cells also add nothing to mapping scores.

```{code-cell} ipython3
transfer.evidence["abstentionReason"].value_counts()
```

We can now plot the transferred labels on the query UMAP and later compare them against our existing labels. A label artifact colors an embedding directly, so nothing is written into the query cell metadata. Abstained cells (cells assigned no labels) are drawn as missing, which shows the geography of abstention.

```{code-cell} ipython3
ds_stim.plots.embedding(
    layout=query_layout,
    color_by=["cluster_labels", transfer_ref],
    figsize=(10, 4),
)
```

`mapping_evidence` plots the evidence saved with the transfer. If you want to set other criteria for the mapping, such as the max distance in the PCA space, then pass `max_distance` to `run_label_transfer`. That saves a second transfer and leaves this one unchanged.

```{code-cell} ipython3
ds_stim.plots.mapping_evidence(
    transfer_ref,
    target_groups=query_labels,
    metrics=("voteFraction", "topTwoMargin", "referenceDistancePercentile"),
    kind="box",
    figsize=(14, 4),
)
```

Because this query dataset also carries original author labels, we can compare these known labels
with the transferred labels to see how mapping performed.

```{code-cell} ipython3
ds_stim.plots.mapping_confusion(
    transfer_ref,
    known_labels=query_labels,
    normalize="true",
)
```

Cells where the known and predicted labels match show recall within each known query label.
Blocks between different labels show systematic swaps. In our example here, we can see that there is a difference in the monocytes. This is because one dataset has stimulated PBMCs, whereas the other has non-stimulated PBMCs. Stimulation can change expression enough that a
query population maps to another reference label. Inspect such swaps before accepting the labels, as the direct biological differences present in your data can influence how the mapping performs.

Because known labels are available, we can use `mapping_calibration` to show how label accuracy trades off against retained coverage as the vote threshold rises. It applies each threshold to the candidate labels saved with the transfer, so nothing is recomputed. The red marker is the transfer's own `threshold_fraction`.

You can use the curve below to check whether the retained labels are also more accurate in this dataset.

```{code-cell} ipython3
ds_stim.plots.mapping_calibration(transfer_ref, known_labels=query_labels)
```

## Choose stricter settings when needed

The first pass uses the defaults, like `threshold_fraction=0.5`, `save_k=3`, `missing_feature_policy="reference_mean"`, and no `max_distance`. If too many assigned labels rest on weak vote support, raise `threshold_fraction` (e.g. 0.6 needs 60% of the neighbor weight); weakly supported cells abstain instead of keeping shaky labels, and the second transfer saves separately for comparison. More neighbors via `save_k` in `run_mapping` create a new mapping and can shift weight near cell type boundaries, and `missing_feature_policy="error"` or `"zero"` replace the default mean-fill when you need strict overlap or true zeros.

## Reuse the reference and saved labels

If you return for your analysis at a later date, you can still retain the mapping-reference, projection, and label-transfer artifact refs. You can do this simply by reopening both stores, and reloading the exact results. A saved transfer loads from the query datastore alone:

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference(reference_ref)
reloaded_mapping = ds_stim.get_mapping_result(mapping_ref, reference=reference)
reloaded_transfer = ds_stim.get_label_transfer(transfer_ref)
(
    reloaded_mapping.n_cells,
    reloaded_mapping.correction_method,
    reloaded_transfer.reference_label_source,
    reloaded_transfer.threshold_fraction,
)
```

For repeated use, reopen the prepared reference datastore read-only and run each mapping in a
separate writable query datastore. If query counts come from a read-only source or the same physical store used to prepare the reference, use `mount_datastore` to create that separate query store. Reference labels are frozen when labels are transferred.

The lineage of a transfer follows its inputs into the reference datastore: from the query labels
through the threshold, the frozen reference labels, and the projection, to the reference model and
the cells it was built from.

```{code-cell} ipython3
ds_stim.lineage(transfer_ref, references=reference)
```

Validate a reused atlas with feature coverage, mapping evidence, abstention, and score concentration. When independent query labels (annotations) exist, also inspect confusion and threshold calibration. A visually plausible embedding alone does not validate transferred labels.

For troubleshooting, common failures include mapping before the reference exists, ignoring feature mismatch, treating vote support as a probability, using a biological condition as a correction batch, and transferring labels without an abstention path.
