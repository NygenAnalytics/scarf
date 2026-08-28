---
description: Build, reload, validate, and reuse a fixed Symphony-style reference for mapping.
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

(reference_atlas_mapping)=

# Building reusable reference atlases

A reusable atlas keeps its feature set, normalization, PCA basis, corrected coordinates, and neighbour index fixed while new queries arrive.
Scarf stores those components as a content-addressed mapping reference and applies a Symphony-style correction without moving the reference cells.

Scarf's `symphony` path follows a fixed-reference PCA, soft-assignment, and ridge-correction contract.
It is not a complete reimplementation of every option in the Symphony R package.

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
stim_path = repository.download_dataset(
    name="kang_14K_ifnb-pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds_ctrl = scarf.DataStore(
    f"{ctrl_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
ds_stim = scarf.DataStore(
    f"{stim_path}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
```

Both catalog stores were rebuilt from their declared sources with the current RNA count layout.
Documentation execution downloads separate page-local copies, so the new reference and mapping
artifacts remain isolated from the published archives.

`reference_batch` must represent technical structure such as donor, preparation, or sequencing batch.
The single control label below only exercises the API.
It is confounded with the biological condition in the stimulated query, so correction magnitude must not be given a biological interpretation.

```{code-cell} ipython3
ds_ctrl.cells.insert(
    "reference_batch",
    np.repeat("control", ds_ctrl.cells.N),
    overwrite=True,
)
```

## 2. Build and reload the reference

Build the Harmony-backed neighbour chain with one pipeline call, then package it as the mapping reference.

```{code-cell} ipython3
run = ds_ctrl.pipeline.run(
    label="harmony_reference",
    filtering=False,
    cell_cycle=False,
    hvg_count=2000,
    pca_dims=25,
    harmony_batch_columns=["reference_batch"],
    neighbors_k=17,
    leiden=False,
    paris=False,
    doublets=False,
    markers=False,
    snapshot_columns=["cluster_labels"],
)
reference_layout = run["umap"]
reference_ref = ds_ctrl.build_mapping_reference(run["neighbors"])
reference_ref
```

`method="symphony"` means the neighbour chain used Harmony-corrected coordinates and the reference carries soft-assignment state for query correction.
The feature count and PCA dimensions stay fixed for every later query.

```{code-cell} ipython3
ds_ctrl.plots.embedding(
    run=run,
    layout="umap",
    color_by="cluster_labels",
)
```

This plot uses the Harmony run's immutable UMAP and its snapshot of `cluster_labels`.
`reference_layout` is reused for mapping-score plots below. `MappingReference` stores
the exact immutable `feature_selection` reference, PCA basis, corrected coordinates, and neighbour
index; it does not store an embedding.
Mapping places query weight onto reference cells without moving those cells.

In a later session, reopen the reference store and load the exact mapping-reference artifact.
The prepared reference datastore may be opened read-only.
Mapping still requires a separate writable query datastore.
Use `mount_datastore` when the query counts come from a read-only source or from the same source used to prepare the reference.

```{code-cell} ipython3
reference_store = scarf.DataStore(
    f"{ctrl_path}/data.zarr",
    zarr_mode="r",
    nthreads=4,
)
reference = reference_store.get_mapping_reference(reference_ref)
pd.Series(
    {
        "method": reference.method,
        "assay": reference.assay_name,
        "selected_cells": reference.selected_cell_count,
        "n_features": reference.model.n_features,
        "n_dims": reference.model.n_dims,
        "has_symphony_state": reference.symphony_state is not None,
    },
    name="reloaded_mapping_reference",
)
```

## 3. Map and inspect a shifted query

Mapping runs in streaming passes so the query matrix does not need to be materialized.
Missing reference features fall back to the reference mean, which becomes zero after reference scaling.

```{code-cell} ipython3
query_batches = pd.DataFrame(
    {
        "reference_batch": np.repeat(
            "stimulated",
            len(ds_stim.cells.fetch("ids", key="I")),
        )
    }
)
query_cell_selection = ds_stim.snapshot_cell_selection("I")
mapping_ref = ds_stim.run_mapping(
    reference,
    query_cell_selection,
    query_assay="RNA",
    save_k=5,
    query_batches=query_batches,
)
mapping_ref
```

The mapping result is stored in `ds_stim`.
The reference model and reference coordinates remain unchanged.
`correction_method="symphony"` confirms the query used the fixed-reference ridge correction rather than a plain PCA neighbour lookup.

```{code-cell} ipython3
mapped = ds_stim.get_mapping_result(
    mapping_ref,
    reference=reference,
    load_arrays=True,
)
mapped.diagnostics
```

Diagnostics report feature coverage, query batch count, the mapping algorithm, and uninformative-cell count.
Mapping rejects using the same physical datastore as both query and reference.
Use a separate writable query datastore for controls.

## 4. Transfer labels and retain abstention

```{code-cell} ipython3
transferred = ds_stim.get_target_classes(
    mapping_ref,
    reference_class_group="cluster_labels",
    reference=reference,
    threshold_fraction=0.6,
)
ds_stim.cells.insert(
    "atlas_labels",
    transferred.to_numpy(),
    overwrite=True,
)
accepted = transferred.notna() & transferred.ne("NA")
accepted.value_counts().rename(
    index={True: "accepted", False: "abstained"}
).rename("query cells")
```

Cells below `threshold_fraction` abstain as `NA` instead of taking a weak majority label.
Compare the accepted and abstained counts with the `NA` column in the confusion matrix below.
The mapping reference freezes its model and selected cell axis, while `reference_class_group`
reads the named column from the reference store at use time. Keep a reusable atlas read-only, or
version externally revised labels under a new column name.

```{code-cell} ipython3
query_labels = np.asarray(ds_stim.cells.fetch("cluster_labels")).astype(str)
ds_stim.plots.mapping_confusion(
    mapping_ref,
    reference=reference,
    reference_class_group="cluster_labels",
    known_labels=query_labels,
    normalize="true",
    threshold_fraction=0.6,
)
```

## 5. Mapping scores by reference cluster

After label transfer, inspect where query weight landed on the reference.
Per-reference-cell scores are mostly zero, so cell-level box plots look flat even when the embedding shows clear hotspots.
Sum the raw (non-log) scores within each reference cluster, then check the sized embedding.
Focused populations should put the highest score mass on matching reference clusters.

```{code-cell} ipython3
focus_labels = ("CD4 Memory T", "CD 14 Mono", "NK")
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
    log_transform=False,
    figsize=(14, 3.4),
)
```

Systematic off-diagonal confusion or diffuse score mass across unrelated reference clusters is a reason to inspect feature coverage, batch design, and whether the query contains populations absent from the atlas.
Compare the confusion pattern with the direct KNN workflow in {doc}`mapping_and_label_transfer`; different off-diagonal errors show how the reference model and correction assumptions affect transfer.

Split-conformal prediction sets are available for label transfer, but their coverage claim requires calibration examples exchangeable with future queries.
The Kang catalog data does not provide a donor-level calibration design, so this page does not manufacture a prediction-set demonstration.
Use the mapping API reference when a defensible calibration cohort is available.
