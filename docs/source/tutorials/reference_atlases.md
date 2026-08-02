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
normalized = ds_ctrl.run_normalization(feat_key="hvgs")
pca = ds_ctrl.run_pca(normalized, dims=25, feat_scaling=True)
corrected = ds_ctrl.run_harmony(["reference_batch"], pca)
ann_index = ds_ctrl.build_ann_index(corrected)
neighbors = ds_ctrl.query_neighbors(ann_index, k=17)
reference = ds_ctrl.build_mapping_reference(neighbors)
reference
```

The {term}`artifact` records the active cells, selected features, normalization, PCA
loadings, corrected latent coordinates, ANN contract, and batch metadata.
Building a different contract creates another artifact rather than mutating the
completed reference.

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference(reference.ref)
```

Reloading validates that the stored inputs still match the datastore. This is
the normal entry point in later sessions, and the prepared reference datastore
may be opened read-only. Mapping still requires a separate writable query
datastore. Use `mount_datastore` to create that writable analysis layer when
the query counts come from a read-only source or from the same source used to
prepare the reference.

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
mapping = ds_stim.run_mapping(
    reference,
    "stim_atlas",
    query_assay="RNA",
    save_k=5,
    query_batches=query_batches,
)
mapping
```

The projection is stored in `ds_stim`. The reference model and reference
coordinates remain unchanged.

```{code-cell} ipython3
mapped = ds_stim.get_mapping_result(
    "stim_atlas",
    reference=reference,
    query_assay="RNA",
    load_arrays=True,
)
mapped.diagnostics
```

Diagnostics report feature coverage, query batch count, the mapping algorithm,
and uninformative-cell count. Mapping rejects using the same physical datastore
as both query and reference. Use a separate writable query datastore for
controls.

## Transfer labels and retain abstention

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
```

```{code-cell} ipython3
ds_stim.plots.mapping_confusion(
    mapped,
    reference_class_group="cluster_labels",
    known_labels=ds_stim.cells.fetch("cluster_labels"),
    normalize="true",
    threshold_fraction=0.6,
)
```

```{code-cell} ipython3
projected_embedding = ds_stim.project_reference_embedding(
    mapped,
    reference_layout_key="RNA_UMAP",
    label="atlas_UMAP",
)
projected_embedding
```

The call above persists an embedding artifact in the query datastore and links
`RNA_atlas_UMAP1` and `RNA_atlas_UMAP2` into query cell metadata. Cells with
uninformative projected PCA coordinates receive `NaN` embedding coordinates
and retain an abstained label.

```{code-cell} ipython3
ds_stim.plots.mapping_projection(
    mapped,
    reference_layout_key="RNA_UMAP",
    target_groups=transferred.to_numpy(),
    ref_name="control atlas",
    reference_mode="background",
    figsize=(7.2, 5.2),
)
```

The projected query should land near compatible reference regions while the
reference stays as a light background. Systematic off-diagonal confusion or
large unmapped regions is a reason to inspect feature coverage, batch design,
and whether the query contains populations absent from the atlas.
Compare the confusion pattern with the direct KNN workflow in
{doc}`mapping_and_label_transfer`; different off-diagonal errors show how the
reference model and correction assumptions affect transfer.

Split-conformal prediction sets are available for label transfer, but their
coverage claim requires calibration examples exchangeable with future queries.
The Kang catalog data does not provide a donor-level calibration design, so
this page does not manufacture a prediction-set demonstration. Use the mapping
API reference when a defensible calibration cohort is available.
