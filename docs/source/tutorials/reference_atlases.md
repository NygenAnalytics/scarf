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

A reusable atlas keeps its feature set, normalization, PCA basis, corrected
coordinates, and neighbour index fixed while new queries arrive. Scarf stores
those components as a content-addressed mapping reference and applies a
Symphony-style correction without moving the reference cells.

Scarf's `symphony` path follows a fixed-reference PCA, soft-assignment, and
ridge-correction contract. It is not a complete reimplementation of every
option in the Symphony R package.

## Open the reference and query

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
ds_ctrl = scarf.DataStore(f"{ctrl_path}/data.zarr", nthreads=4)
ds_stim = scarf.DataStore(f"{stim_path}/data.zarr", nthreads=4)
```

`reference_batch` must represent technical structure such as donor,
preparation, or sequencing batch. The single control label below only exercises
the API. It is confounded with the biological condition in the stimulated
query, so correction magnitude must not be given a biological interpretation.

```{code-cell} ipython3
ds_ctrl.cells.insert(
    "reference_batch",
    np.repeat("control", ds_ctrl.cells.N),
    overwrite=True,
)
```

## Build and reload the reference

```{code-cell} ipython3
reference = ds_ctrl.build_mapping_reference(
    feat_key="hvgs",
    batch_columns=["reference_batch"],
)
reference
```

The artifact records the active cells, selected features, normalization, PCA
loadings, corrected latent coordinates, ANN contract, and batch metadata.
Building a different contract creates another artifact rather than mutating the
completed reference.

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference(feat_key="hvgs")
```

Reloading validates that the stored inputs still match the datastore. This is
the normal entry point in later sessions.

## Map and inspect a shifted query

Mapping runs in streaming passes so the query matrix does not need to be
materialized. Missing reference features fall back to the reference mean, which
becomes zero after reference scaling.

```{code-cell} ipython3
query_batches = pd.DataFrame(
    {
        "reference_batch": np.repeat(
            "stimulated",
            len(ds_stim.cells.fetch("ids", key="I")),
        )
    }
)
reference.map_query(
    target_assay=ds_stim.RNA,
    target_name="stim_atlas",
    target_feat_key="hvgs_stim_atlas",
    save_k=5,
    query_batches=query_batches,
)
```

`mapping_correction` compares query latent coordinates before and after
fixed-reference correction. A nonzero shift is expected for this deliberately
confounded example.

```{code-cell} ipython3
ds_ctrl.plots.mapping_correction(
    target_name="stim_atlas",
    batch_labels=ds_stim.cells.fetch("cluster_labels"),
)
```

```{code-cell} ipython3
mapped = ds_ctrl.get_mapping_result(
    "stim_atlas",
    load_arrays=True,
)
mapped.diagnostics
```

Diagnostics should report finite corrected coordinates, valid neighbour
indices, and the requested feature coverage. A large correction is not
automatically good; it must be assessed with label evidence and an
uncorrected control.

## Self-map as a control

The reference can map its own cells through the same contract. With matching
batch labels, self-map correction should remain close to zero and reference
coordinates must stay unchanged.

```{code-cell} ipython3
reference.map_query(
    target_assay=ds_ctrl.RNA,
    target_name="control_self_map",
    target_feat_key="hvgs_control_self_map",
    save_k=5,
    query_batches=pd.DataFrame(
        {
            "reference_batch": ds_ctrl.cells.fetch(
                "reference_batch",
                key="I",
            )
        }
    ),
)
self_mapped = ds_ctrl.get_mapping_result(
    "control_self_map",
    load_arrays=True,
)
float(
    np.linalg.norm(
        self_mapped.corrected_latent
        - self_mapped.uncorrected_latent,
        axis=1,
    ).max()
)
```

## Transfer labels and retain abstention

```{code-cell} ipython3
transferred = ds_ctrl.get_target_classes(
    target_name="stim_atlas",
    reference_class_group="cluster_labels",
    threshold_fraction=0.6,
)
ds_stim.cells.insert(
    "atlas_labels",
    transferred.to_numpy(),
    overwrite=True,
)
```

```{code-cell} ipython3
ds_ctrl.plots.mapping_confusion(
    target_name="stim_atlas",
    reference_class_group="cluster_labels",
    known_labels=ds_stim.cells.fetch("cluster_labels"),
    normalize="true",
    threshold_fraction=0.6,
)
```

```{code-cell} ipython3
ds_ctrl.plots.mapping_projection(
    target_name="stim_atlas",
    reference_layout_key="RNA_UMAP",
    reference_groups="cluster_labels",
    target_groups=transferred.to_numpy(),
    ref_name="control atlas",
)
```

The projected query should land near compatible reference regions while the
reference layout remains unchanged. Systematic off-diagonal confusion or large
unmapped regions is a reason to inspect feature coverage, batch design, and
whether the query contains populations absent from the atlas.
Compare the confusion pattern with the direct KNN workflow in
{doc}`mapping_and_label_transfer`; different off-diagonal errors show how the
reference model and correction assumptions affect transfer.

Split-conformal prediction sets are available for label transfer, but their
coverage claim requires calibration examples exchangeable with future queries.
The Kang catalog data does not provide a donor-level calibration design, so
this page does not manufacture a prediction-set demonstration. Use the mapping
API reference when a defensible calibration cohort is available.
