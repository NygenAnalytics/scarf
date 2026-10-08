# Plotting API reference

`scarf.plotting` is Scarf's plotting API.
Import it as `splt` and call functions such as `splt.embedding(...)`, `splt.dotplot(...)`, and `splt.cluster_tree(...)`.
For functions whose first argument is a datastore, `ds.plots.embedding(...)` and related accessor methods provide the same behavior with that argument already bound.
Store-backed graph plotters mirror their `DataStore` counterparts and require an exact `graph=`
reference.
`dotplot` and `matrixplot` take `features=` as gene names or `FeatureRef` values, or as a mapping of group names to such lists.
Diagnostics remain standalone: `qc` takes a DataFrame, `elbow` and `highly_variable_features` take arrays, and `graph_qc` takes a sparse graph.

A completed {py:class}`~scarf.PipelineRun` can be passed to
`ds.plots.embedding(run=run, layout="umap", color_by="clusters")`. The run supplies the layout
(`layout` names a run output, `"umap"` by default) and the cells, and every cell input comes from
the run. There is no live fallback in run mode:

- `color_by` takes one item or a list, drawn as one panel each. A string or `CellField` names a
  frozen run cell field from `run.cells.columns`, such as `clusters`, `doublet_score`, or a column
  frozen with `snapshot_columns`. An `ArtifactRef` must be an output of the run, such as
  `run["leiden_1.0"]`. Any other ref raises `ValueError`; color by it with the granular
  `layout=run["umap"]` route instead.
- `facet_by`, `subset_by`, `Highlight(by=...)` and `DensityOverlay(group_by=...)` name frozen run
  cell fields. A name that is neither a frozen field nor, for `color_by`, a gene raises `KeyError`
  listing the run's cell fields.
- A gene name or `FeatureRef` reads the live counts of the run's assay over the run's cells and
  normalizes them with the assay's live normalizer. A run freezes no gene values, so a gene color
  requires an explicit `normalization=NormalizationSpec(...)`, which provenance records; without
  one it raises `ValueError`.
- `cell_key` other than `"I"`, a `from_assay` other than the run's assay, and `point_sizes` raise
  `ValueError`. For per-cell sizes, pass `layout=run["umap"]` with sizes in that layout's cell
  order.

Run plots record the run as `provenance.extras["run"]`, a mapping with `runId` and `label`.
`layout_key=` remains the live-metadata source. Mixed source modes are rejected. For large
continuous fields, `ds.plots.embedding_raster(run=run, layout="umap",
color_by="doublet_score")` uses the same frozen run selection blockwise; its `color_by` (a string
or `CellField`) and `subset_by` name frozen run cell fields.

Granular workflows pass exact refs to the same datastore-owned plotting surface:

| Plot | Explicit artifact inputs |
|---|---|
| embedding | `layout=embedding_ref`, optionally `color_by=cluster_ref` |
| dot or matrix plot | `groups=cluster_ref` |
| composition | `categories=cluster_ref` |
| distribution | `grouping=cluster_ref` |
| cluster connectivity | `graph=graph_ref`, `groups=cluster_ref`, `layout=embedding_ref` |
| modality weights | `graph=wnn_graph_ref`, `layout=embedding_ref` |
| Paris hierarchy | `graph=graph_ref`, `clusters=paris_ref` |
| marker heatmap | `marker=marker_ref` |
| pseudotime heatmap | `aggregation=aggregation_ref` |
| mapping score | `mapping_score(result_ref, reference=reference, layout=embedding_ref)` |
| label-transfer evidence, confusion, and calibration | `mapping_evidence(transfer_ref)`, `mapping_confusion(transfer_ref, known_labels=...)`, `mapping_calibration(transfer_ref, known_labels=...)` |

`layout_key` and string forms on plotters that still accept them refer to deliberate live metadata
inputs. Distribution grouping instead requires either an exact categorical artifact or an explicit
`CellField`, with `cell_selection=` when a frozen metadata subset is intended.

Dot plots and matrix plots name the grouping columns of their tables by role, whatever the grouping
columns are called: `group` holds the first `group_by` key's values or the `groups=` labels,
`subgroup` the second key's values, and `sample` the `sample_by` value. They are followed by
`feature`, `feature_group`, `mean`, `fraction`, `n_cells` and `variance`, and an aggregate over
samples adds `n_samples`. A grouping column can therefore be named `feature`, `mean` or `sample`,
and `provenance.extras["group_by"]` keeps the source column names. With `standardize="feature"`,
`mean` stays raw and a `zscore` column of `tables["aggregate"]` holds the plotted values, while
`tables["per_sample"]` keeps the per-sample means without it; `provenance.extras["color_values"]`
names the plotted column. A matrix plot's `tables["matrix"]` is the drawn matrix, like the matrix
of `marker_heatmap`: one row per feature (index `feature`) and one column per group label (columns
`group`). `marker_heatmap`'s matrix holds the standardized values of its marker features, one
column per group of the marker search. The annotation tables of both are indexed by the same
labels.

Plots treat rows that a nullable metadata column or label artifact flags in its linked missing mask
as missing, never as their stored placeholder. Dot plots and matrix plots leave those cells out of
every group and report their number as `dropped_group_cells`. Composition plots count a missing
category in the NA category and leave cells with a missing sample label out of every sample.
Embeddings and cluster trees show missing colors or fill values as missing, and
`cluster_connectivity` rejects missing group labels and coordinates.

A feature has no value in a cell that its assay did not measure, which the assay's membership
column `<assay>_I` marks False ({ref}`assay_membership`); such a cell holds zero counts, which are no
measurement. Plots read only the measured cells and normalize only them, so a normalizer fitted over
the cells that it reads, such as ATAC TF-IDF, gives a measured cell the value of a read of the
measured cells alone. Embeddings draw an unmeasured cell in the scale's `missing_color` and leave it
out of the color limits, and `provenance.extras["color_limits"]` holds None for a color or panel
without a value among the drawn cells. Distribution plots leave it out of the feature's panel and
table, and a feature panel whose assay measured none of the selected cells raises `ValueError` that
names the membership column. Dot plots and matrix plots compute `mean`, `fraction`, and `variance`
over the cells of a group with a value, which `n_cells` counts, so `fraction` is the share of
measured cells above `expression_cutoff`; a group without a measured cell keeps its row with
`n_cells` 0 and NaN statistics and draws no dot or a missing matrix cell, and an aggregate over
samples skips such a sample in every statistic and in `n_samples`. Cluster-tree fill values and the
group means of `marker_heatmap`, such as those of a marker search that an earlier release ran over
unmeasured cells, leave it out. Dot plots, matrix plots, embeddings, distribution plots, and
`marker_heatmap` record `provenance.extras["unmeasured_cells"]`, a mapping from each feature assay
to the number of plotted cells that it did not measure, when that number is not zero.

WNN is the default for `DataStore.integrate_assays`. Its integrated graph stores one weight per
input assay and cell. Use `ds.plots.modality_weights(graph=wnn_graph, layout=embedding)` or
{py:func}`scarf.plotting.modality_weights` to show those weights over an explicit embedding. The
graph and layout must have the exact same cell-selection artifact. Explicit SNN graphs do not
contain modality weights.

Store-backed plotters and diagnostics generally return a `PlotResult` and render by default with `show=True`.
Pass `show=False` before accessing, saving, or reusing an owned figure.
`run_recipe` returns a `PlotRecipeResult` and defaults to `show=False`.
Helpers differ: `label_panels` returns `None`, `theme_context` is an iterator, and `compose_results` returns a `PlotResult` without a `show` parameter.

```{eval-rst}
.. automodule:: scarf.plotting
    :members: embedding, embedding_raster, dotplot, matrixplot, modality_weights, composition, distribution, cluster_connectivity, mapping_score, mapping_evidence, mapping_confusion, mapping_calibration, qc, graph_qc, elbow, highly_variable_features, label_panels, compose_results, theme_context, marker_heatmap, cluster_tree, pseudotime_heatmap, run_recipe
    :imported-members:
    :undoc-members:
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: scarf.plotting.PlotResult
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.FeatureRef
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.CellField
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.StudyDesign
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.NormalizationSpec
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.ColorScale
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.CategoricalScale
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.SizeScale
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.DensityOverlay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.Highlight
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.PlotRecipe
    :members:

.. autoclass:: scarf.plotting.PlotStep
    :members:

.. autoclass:: scarf.plotting.PlotPanelTarget
    :members:

.. autoclass:: scarf.plotting.PlotOutputSettings
    :members:

.. autoclass:: scarf.plotting.PlotOutput
    :members:

.. autoclass:: scarf.plotting.PlotRecipeResult
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.PlotProvenance
    :members:
```

```{eval-rst}
.. autoclass:: scarf.plotting.LegendSpec
    :members:
```

```{eval-rst}
.. py:data:: THEMES

    Registry of built-in plotting theme definitions.
```
