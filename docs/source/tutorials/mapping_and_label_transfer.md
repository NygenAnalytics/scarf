---
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

# Projection, label transfer, and unified embeddings

Mapping places the cells of one dataset onto the graph of another without merging the two
count matrices. It is lighter than merge plus batch correction, and it leaves the reference
untouched, which is what makes a reference reusable.

This chapter covers two levels of that idea:

- Direct KNN mapping, where query cells are matched against a reference graph as it stands
- A versioned mapping reference with Symphony-style correction, for a reference you intend
  to publish and map against repeatedly

Both use data from [Kang et al.](https://www.nature.com/articles/nbt.4042): control and
IFN-B treated PBMCs, distributed as prepared Zarr stores that already carry UMAP coordinates
and curated `cluster_labels`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- The single-dataset workflow in {doc}`scrna_seq`

## What you will learn

- Map query cells onto a fixed reference graph
- Transfer categorical labels and judge them with vote evidence
- Choose an abstention threshold from held-out correctness
- Build unified embeddings and project into a stable reference layout
- Store a versioned mapping reference and map against it with batch correction

## Terminology

```{note}
**Reference cells** come from the dataset that mapping is based on. That dataset must
already have a neighbourhood graph.

**Query cells**, called target cells in the API, are the cells being placed onto the
reference. They do not need a graph of their own.
```

## Dataset

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scarf

scarf.configure_output(level='WARNING', progress=False)

repository = scarf.cytebase.connect("scarf_docs")
ctrl_path = repository.download_dataset(
    name='kang_15K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True
)
stim_path = repository.download_dataset(
    name='kang_14K_ifnb-pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True
)
```

The control sample is the reference and the stimulated sample is the query.

```{code-cell} ipython3
ds_ctrl = scarf.DataStore(f'{ctrl_path}/data.zarr', nthreads=4)
ds_stim = scarf.DataStore(f'{stim_path}/data.zarr', nthreads=4)

ds_ctrl.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='cluster_labels',
)
ds_stim.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='cluster_labels',
)
```

The two layouts are computed independently, so their coordinates are not comparable. Only
the cell-type labels are shared, and the rest of this page is about relating the two.

---
## 1) K-Nearest Neighbours (KNN) mapping

The ``run_mapping`` method projects target cells onto the fixed reference graph. The reference cells are in the `DataStore` where `run_mapping` is called, and the target assay is supplied as an argument. Scarf aligns target features to the ordered reference feature set, applies the scaling stored with the reference PCA, and queries the reference ANN index. Target cells are never inserted into the reference index.

By default, `missing_feature_policy='zero'` fills features absent from the target with zero normalized expression. Earlier releases silently used one, so partial-overlap results can change after remapping. Use `missing_feature_policy='error'` when complete overlap is required. `missing_feature_policy='intersection'` creates an isolated, overlap-only index for a controlled comparison; it does not alter the authoritative reference graph. The deprecated `exclude_missing=True` alias also leaves its historical overlap feature key for one compatibility cycle.

```{code-cell} ipython3
ds_ctrl.run_mapping(
    target_assay=ds_stim.RNA,
    target_name='stim',
    target_feat_key='hvgs_ctrl',
    save_k=5
)
print(f"Mapped {ds_stim.cells.N} query cells onto the control reference")
```

Key mapping parameters:
- `save_k`: number of reference neighbors stored per target cell (default 3)
- `missing_feature_policy`: choose `zero`, `error`, or `intersection` handling for absent target features
- `filter_null`: with `missing_feature_policy='intersection'`, removes target features with zero total counts
- `run_coral`: deprecated experimental feature-space correction. Build a Symphony-style mapping reference for harmonized atlas mapping.
- `ref_mu` and `ref_sigma`: deprecated compatibility flags. Reference statistics are always used.

---
## 2) Mapping scores

+++

Mapping scores support cross-dataset cluster similarity inspection. By default,
`get_mapping_score` accumulates distance-derived neighbor weights for each reference cell and
normalizes them by the number of mapped target cells and saved neighbors. Frequently selected,
nearby reference cells therefore receive higher scores. `target_groups` calculates a separate
score for each target group. Here the stimulated-cell clusters are used as groups, and the
UMAPs show where selected target clusters map onto the control reference.

```{code-cell} ipython3
# Generate plots for IFN-B stimulated cells from NK and CD14 monocyte clusters.

for g, ms in ds_ctrl.get_mapping_score(
    target_name='stim',
    target_groups=ds_stim.cells.fetch('cluster_labels'),
    log_transform=True
):
    if g in ['NK', 'CD 14 Mono']:
        print(f"Target cluster {g}")
        ds_ctrl.plots.embedding(
            layout_key='RNA_UMAP',
            color_by='cluster_labels',
            point_sizes=ms * 10,
            figsize=(4, 4),
        )
```

---
## 3) Label transfer

Using the nearest neighbours of the target cells in the reference data, we can transfer labels from reference cells to target cells based on majority voting. This means that if a target cell has 'most' of its total edge weight shared with cells from one cell type, then that cell type label is transferred to the target cell. The default threshold for 'most' is 0.5, i.e. half of all edge weight. `get_target_classes` method returns the transferred labels for each cell from a given mapped target dataset.

The `reference_class_group` parameter decides which labels to transfer. This can be any column from the cell attribute table that has categorical values, generally users would use `RNA_leiden_cluster` or `RNA_paris_cluster` but they can also use other labels. Here, for example, we use the custom labels stored under `cluster_labels` column.

```{code-cell} ipython3
transferred_labels = ds_ctrl.get_target_classes(
    target_name='stim',
    reference_class_group='cluster_labels'
)
transferred_labels.value_counts()
```

Use `get_target_label_evidence` to inspect neighbor-vote fraction, entropy, margin, feature coverage, and a reference distance percentile. These are diagnostic quantities, not calibrated probabilities. The distance percentile uses the reference self-neighbor distribution rather than the query distribution. The method returns `NA` by default when the winning vote does not meet a chosen threshold, when top labels tie, or when a query has no directional information in reference PC space.

```{code-cell} ipython3
evidence = ds_ctrl.get_target_label_evidence(
    target_name='stim',
    reference_class_group='cluster_labels',
    threshold_fraction=0.6
)
evidence.head()
```

We can now save these transferred labels in the stimulated-cell dataset and visualize them on
its UMAP.

```{code-cell} ipython3
ds_stim.cells.insert(
    'transferred_labels',
    transferred_labels.values,
    overwrite=True
)
```

```{code-cell} ipython3
ds_stim.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='transferred_labels',
)
```

Compare transferred labels to the stored target labels:

```{code-cell} ipython3
import pandas as pd

df = pd.crosstab(
    ds_stim.cells.fetch('cluster_labels'),
    ds_stim.cells.fetch('transferred_labels')
)
df
```

The column-normalized table below shows the composition of each transferred label. Each column
sums to 100, and a diagonal entry is the precision for that predicted label. It is not an
overall accuracy score.

```{code-cell} ipython3
(100 * df / df.sum(axis=0)).round(1)
```

Overall label accuracy is the fraction of target cells whose transferred label matches the
stored target label:

```{code-cell} ipython3
matches = ds_stim.cells.fetch('cluster_labels') == transferred_labels.to_numpy()
print(f'Labels matching the stored target label: {100 * matches.mean():.1f}%')
```

That figure is modest, and the reason is biological rather than a mapping failure:
interferon stimulation shifts expression enough that several activated populations no longer
sit nearest their untreated counterparts. The correction covered in section 5 is one way to
address exactly this.

### Calibrate a vote-fraction threshold

`calibrate_label_transfer_threshold` chooses a vote-fraction cutoff from held-out correctness
labels so that a target fraction of correct cells remain above the cutoff. Here the stored
stimulated `cluster_labels` stand in for held-out truth. Prefer donor-level held-out splits
in real atlases rather than labels from the query being mapped.

```{code-cell} ipython3
evidence_for_cal = ds_ctrl.get_target_label_evidence(
    target_name='stim',
    reference_class_group='cluster_labels',
    threshold_fraction=0.0,
)
correct = (
    ds_stim.cells.fetch('cluster_labels') == transferred_labels.to_numpy()
)
calibration = ds_ctrl.calibrate_label_transfer_threshold(
    vote_fractions=evidence_for_cal['voteFraction'].to_numpy(dtype=float),
    correct=np.asarray(correct, dtype=bool),
    target_coverage=0.9,
)
calibration
```

The returned `voteThreshold` is a cutoff on `voteFraction`, not a calibrated probability.
`validationCoverage` and `validationAccuracy` summarize the held-out set at that cutoff.

---
## 4) Unified UMAPs

Unified UMAP adds query-reference edges and reruns UMAP on the combined graph. It is useful for exploration, but it moves the reference coordinates and can change when a different query is supplied. Do not use it as a stable atlas coordinate system.

```{code-cell} ipython3
ds_ctrl.run_unified_umap(
    target_names=['stim'],
    ini_embed_with='RNA_UMAP',
    target_weight=1,
    use_k=5,
    n_epochs=100
)
```

Since unified embeddings contain cells from another dataset, use
`ds_ctrl.plots.unified_embedding`. Use `ds_ctrl.plots.embedding` for regular layouts
stored in cell metadata.

```{code-cell} ipython3
ds_ctrl.plots.unified_embedding(
    layout_key='unified_UMAP',
    ref_name='ctrl',
)
```

For stable placement into an existing reference layout, use deterministic neighbor-weighted projection. It writes only target coordinates and leaves reference coordinates unchanged. A read-only datastore returns the coordinate array in memory instead of attempting a write.

```{code-cell} ipython3
projected_layout = ds_ctrl.project_mapping_layout(
    target_name='stim',
    reference_layout_key='RNA_UMAP'
)
```

On a writable store the call returns the Zarr path it wrote and leaves every reference
coordinate untouched. A read-only store returns the coordinates as an array instead.

Show only the IFN-B stimulated target cells in the unified embedding, colored by their
original cluster identity. Similar types often sit near each other and near matching
reference populations, but unified UMAP is exploratory and not a fixed atlas layout.

```{code-cell} ipython3
ds_ctrl.plots.unified_embedding(
    layout_key='unified_UMAP',
    show_target_only=True,
    target_groups=ds_stim.cells.fetch('cluster_labels'),
    legend_loc='on_data',
)
```

---
## 5) Unified tSNE (optional)

Unified tSNE co-embeds reference and target cells on the same unified graph. Prefer
`project_mapping_layout` when you need stable coordinates; use unified embeddings only for
exploration.

```{code-cell} ipython3
ds_ctrl.run_unified_tsne(
    target_names=['stim'],
    ini_embed_with='RNA_UMAP',
    target_weight=1,
    use_k=5,
    max_iter=500,
    verbose=False,
)

ds_ctrl.plots.unified_embedding(
    layout_key='unified_tSNE',
    ref_name='ctrl',
)
```

(reference_atlas_mapping)=

---
## 6) A reusable mapping reference

Everything so far mapped against the reference graph as it happened to be built. When a
reference is meant to be published and mapped against repeatedly, two more things matter:
the reference should be stored as a versioned artifact so results are reproducible, and
query cells should be corrected onto it so that batch differences do not masquerade as
biological ones.

`build_mapping_reference` writes that artifact, and `map_query` applies a Symphony-style
fixed-reference correction: the reference coordinates never move, and each query cell is
corrected using fixed soft cluster assignments and a scalar ridge term.

```{note}
Scarf's `symphony` path is not a full reimplementation of the Symphony R model. Its shared
PCA, soft-assignment, and correction contracts are checked against a static fixture from
Symphony R 0.1.3, including a nonzero correction case.
```

Batch labels are the part most often got wrong, so it is worth being explicit.

```{warning}
`reference_batch` must describe technical structure such as donor, preparation, or
sequencing batch. The control and stimulated labels used below are confounded with the
biological condition and serve only to exercise the API. Read nothing biological into this
example, and in a real atlas use batches that appear on both sides of the comparison.
```

```{code-cell} ipython3
ds_ctrl.cells.insert(
    'reference_batch',
    np.repeat('control', ds_ctrl.cells.N),
    overwrite=True
)

reference = ds_ctrl.build_mapping_reference(
    feat_key='hvgs',
    batch_columns=['reference_batch']
)
```

Reload the artifact when the build and the mapping happen in different sessions. Scarf
validates the feature set, active cells, normalization, PCA loadings, corrected coordinates,
ANN contract, and batch metadata before handing it back. Rebuilding produces a new
content-addressed artifact and leaves earlier complete ones intact.

```{code-cell} ipython3
reference = ds_ctrl.get_mapping_reference(feat_key='hvgs')
```

Mapping runs in two streaming passes. The first estimates batch and soft-cluster statistics,
the second corrects the query coordinates and queries the immutable reference index. Missing
reference features fall back to `reference_mean`, which contributes zero after reference
scaling. `query_batches` has one row per active query cell, and its columns are combined into
query-specific batch groups rather than matched against reference category names.

```{code-cell} ipython3
result = reference.map_query(
    target_assay=ds_stim.RNA,
    target_name='stim_symphony',
    target_feat_key='hvgs_symphony',
    save_k=5,
    query_batches=pd.DataFrame(
        {
            'reference_batch': np.repeat(
                'stimulated',
                len(ds_stim.cells.fetch('ids', key='I'))
            )
        }
    )
)
result
```

`get_mapping_result` reloads the stored arrays. Both the uncorrected and corrected latent
coordinates are kept so the correction can be inspected; neighbours come from the corrected
ones.

```{code-cell} ipython3
mapped = ds_ctrl.get_mapping_result('stim_symphony', load_arrays=True)
mapped.diagnostics
```

How far did the correction actually move each query cell? For a shifted query it should be
clearly nonzero.

```{code-cell} ipython3
shift = np.linalg.norm(
    mapped.corrected_latent - mapped.uncorrected_latent,
    axis=1,
)

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.hist(shift, bins=40, color='#1f4e79', edgecolor='white')
ax.set_xlabel(r'$\|corrected - uncorrected\|$')
ax.set_ylabel('Query cells')
ax.set_title('Symphony correction magnitude (stimulated query)')
fig
```

The same reference can map the control cells back onto themselves. That is the control
experiment for the correction: with no batch shift to remove, the histogram should sit near
zero, and the reference coordinates must not change.

```{code-cell} ipython3
control_result = reference.map_query(
    target_assay=ds_ctrl.RNA,
    target_name='control_symphony',
    target_feat_key='hvgs_control_symphony',
    save_k=5,
    query_batches=pd.DataFrame(
        {'reference_batch': ds_ctrl.cells.fetch('reference_batch', key='I')}
    )
)
control_mapped = ds_ctrl.get_mapping_result(
    'control_symphony',
    load_arrays=True,
)
ctrl_shift = np.linalg.norm(
    control_mapped.corrected_latent - control_mapped.uncorrected_latent,
    axis=1,
)

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.hist(ctrl_shift, bins=40, color='#5b7c99', edgecolor='white')
ax.set_xlabel(r'$\|corrected - uncorrected\|$')
ax.set_ylabel('Reference cells (self-map)')
ax.set_title('Correction should stay near zero on control')
fig
```

Labels transfer from a corrected mapping exactly as they did in section 3.

```{code-cell} ipython3
transferred = ds_ctrl.get_target_classes(
    target_name='stim_symphony',
    reference_class_group='cluster_labels',
)
ds_stim.cells.insert('symphony_labels', transferred.values, overwrite=True)

ds_stim.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='symphony_labels',
)
```

```{code-cell} ipython3
pd.crosstab(
    ds_stim.cells.fetch('cluster_labels'),
    ds_stim.cells.fetch('symphony_labels'),
)
```

Vote-fraction evidence is available here too, and the same caution applies: choose an
abstention threshold from held-out donors, not from the query being mapped.

```{code-cell} ipython3
symphony_evidence = ds_ctrl.get_target_label_evidence(
    target_name='stim_symphony',
    reference_class_group='cluster_labels',
    threshold_fraction=0.6
)

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.hist(
    symphony_evidence['voteFraction'].dropna(),
    bins=30,
    color='#1f4e79',
    edgecolor='white',
)
ax.axvline(0.6, color='#c45c26', linestyle='--', label='threshold 0.6')
ax.set_xlabel('Vote fraction')
ax.set_ylabel('Query cells')
ax.set_title('Label-transfer vote fractions')
ax.legend(frameon=False)
fig
```

Split-conformal prediction sets are also supported, but they need calibration examples that
are exchangeable with future queries, which this page does not set up. Opening a reference
read-only returns arrays in memory; pass a writable Zarr group as `result_store` to keep
neighbours and latent coordinates out of core for large queries.

## Common mistakes and limitations

- Mapping before the reference has a neighbourhood graph and PCA loadings
- Treating transferred labels as ground truth without checking vote evidence
- Reading vote fractions as calibrated probabilities
- Using biological condition as `reference_batch` or as a query batch
- Using unified UMAP as a stable atlas coordinate system
- Ignoring `missing_feature_policy` when reference and query gene sets differ
- Expecting the `symphony` path to match every option of the Symphony R package

## Saved results

`run_mapping` stores neighbor relations under the chosen `target_name`. Transferred labels
and evidence live in query cell metadata after `insert`. Unified layouts are written under
keys such as `unified_UMAP`; `project_mapping_layout` writes target coordinates only.
`build_mapping_reference` writes a content-addressed mapping artifact on the reference store,
and `map_query` stores neighbours and latent arrays under its `target_name`.

## Further reading

- Kang et al. 2018, IFN-stimulated PBMCs used in this chapter: https://doi.org/10.1038/nbt.4042
- Kang et al. 2021, Symphony: https://doi.org/10.1038/s41467-021-25957-x

## Next steps

- {doc}`data_integration`
- {doc}`annotation`
- {doc}`plotting`

