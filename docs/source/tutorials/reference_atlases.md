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
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import scarf
from scarf.tools.repack_zarr import repack_store

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
analysis_directory = TemporaryDirectory()
ctrl_counts_path = Path(analysis_directory.name) / "control_counts.zarr"
stim_counts_path = Path(analysis_directory.name) / "stimulated_counts.zarr"
ctrl_analysis_path = Path(analysis_directory.name) / "reference.zarr"
stim_analysis_path = Path(analysis_directory.name) / "query.zarr"
repack_store(
    f"{ctrl_path}/data.zarr",
    str(ctrl_counts_path),
    nthreads=2,
)
repack_store(
    f"{stim_path}/data.zarr",
    str(stim_counts_path),
    nthreads=2,
)
ds_ctrl = scarf.mount_datastore(
    str(ctrl_counts_path),
    at=str(ctrl_analysis_path),
    default_assay="RNA",
    nthreads=4,
)
ds_stim = scarf.mount_datastore(
    str(stim_counts_path),
    at=str(stim_analysis_path),
    default_assay="RNA",
    nthreads=4,
)
```

Both published stores remain unchanged.
Each is structurally repacked into its own temporary source with the current RNA count layout.
Mounting those sources copies literal cell metadata into two separate page-local targets, while the current reference and mapping artifacts are written only there.

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
    harmony={"batch_columns": ["reference_batch"]},
    neighbors={"k": 17},
    umap=False,
    leiden={},
    paris=False,
    doublet_scoring=False,
    markers=False,
)
reference = ds_ctrl.build_mapping_reference(artifacts["neighbors"])
pd.Series(
    {
        "method": reference.method,
        "assay": reference.assay_name,
        "feature_selection": reference.feature_selection,
        "selected_cells": reference.selected_cell_count,
        "n_features": reference.model.n_features,
        "n_dims": reference.model.n_dims,
        "has_symphony_state": reference.symphony_state is not None,
    },
    name="mapping_reference",
)
```

`method="symphony"` means the neighbour chain used Harmony-corrected coordinates and the reference carries soft-assignment state for query correction.
The feature count and PCA dimensions stay fixed for every later query.

```{code-cell} ipython3
ds_ctrl.plots.embedding(
    layout_key="RNA_UMAP",
    color_by="cluster_labels",
)
```

This plot uses the pre-published `RNA_UMAP` already on the datastore.
`umap=False` above skipped recomputing UMAP for the Harmony neighbour chain, so the layout is a viewing aid rather than part of the packaged mapping reference.
`MappingReference` stores the exact immutable `feature_selection` reference, PCA basis, corrected coordinates, and neighbour index; it does not store an embedding.
Mapping places query weight onto reference cells without moving those cells.

In a later session, reopen the reference store and load the named mapping reference.
The prepared reference datastore may be opened read-only.
Mapping still requires a separate writable query datastore.
Use `mount_datastore` when the query counts come from a read-only source or from the same source used to prepare the reference.

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference()
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
mapping = ds_stim.run_mapping(
    reference,
    "stim_atlas",
    query_assay="RNA",
    save_k=5,
    query_batches=query_batches,
)
mapping.mapping_name, mapping.n_cells, mapping.correction_method
```

The mapping result is stored in `ds_stim`.
The reference model and reference coordinates remain unchanged.
`correction_method="symphony"` confirms the query used the fixed-reference ridge correction rather than a plain PCA neighbour lookup.

```{code-cell} ipython3
mapped = ds_stim.get_mapping_result(
    "stim_atlas",
    reference=reference,
    query_assay="RNA",
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
    mapped,
    reference_class_group="cluster_labels",
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

```{code-cell} ipython3
query_labels = np.asarray(ds_stim.cells.fetch("cluster_labels")).astype(str)
ds_stim.plots.mapping_confusion(
    mapped,
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
assert mapped.reference is not None
ref_classes = np.asarray(
    mapped.reference.fetch_cell_column("cluster_labels"),
    dtype=object,
)
score_mass: dict[str, pd.Series] = {}
for group, values in ds_stim.get_mapping_score(
    mapped,
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
    mapped,
    layout_key="RNA_UMAP",
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
