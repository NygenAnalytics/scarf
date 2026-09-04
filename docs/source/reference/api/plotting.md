# Plotting API reference

`scarf.plotting` is Scarf's plotting API.
Import it as `splt` and call functions such as `splt.embedding(...)`, `splt.dotplot(...)`, and `splt.cluster_tree(...)`.
For functions whose first argument is a datastore, `ds.plots.embedding(...)` and related accessor methods provide the same behavior with that argument already bound.
Store-backed graph plotters mirror their `DataStore` counterparts and require an exact `graph=`
reference.
Feature-consuming plotters require `features=` as a feature-selection ref.
Diagnostics remain standalone: `qc` takes a DataFrame, `elbow` and `highly_variable_features` take arrays, and `graph_qc` takes a sparse graph.

A completed {py:class}`~scarf.PipelineRun` can be passed to
`ds.plots.embedding(run=run, layout="umap", color_by="clusters")`. The datastore accessor reads only
frozen run cell fields and does not require live metadata columns. A string `color_by` names a
frozen run field such as `clusters` or `doublet_score`. To color by another artifact or a gene, use
the granular `layout=ArtifactRef` route with typed refs. There is no live fallback in run mode.
`layout_key=` remains the live-metadata source. Mixed source modes are rejected. For large
continuous fields, `ds.plots.embedding_raster(run=run, layout="umap",
color_by="doublet_score")` uses the same frozen run selection blockwise.

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

`layout_key` and string forms on plotters that still accept them refer to deliberate live metadata
inputs. Distribution grouping instead requires either an exact categorical artifact or an explicit
`CellField`, with `cell_selection=` when a frozen metadata subset is intended.

WNN is the default for `DataStore.integrate_assays`. Its integrated graph stores one weight per
input assay and cell. Use `ds.plots.modality_weights(graph=wnn_graph, layout=embedding)` or
{py:func}`scarf.plotting.modality_weights` to show those weights over an explicit embedding. The
graph and layout must have the exact same cell-selection artifact. Explicit SNN graphs do not
contain modality weights.

Store-backed plotters and diagnostics generally return a `PlotResult` and render by default with `show=True`.
Pass `show=False` before accessing, saving, or reusing an owned figure.
`run_recipe` returns a `PlotRecipeResult` and defaults to `show=False`.
Helpers differ: `label_panels` and `register_theme` return `None`, `collect_legends` returns a tuple, `theme_context` is an iterator, and `compose_results` returns a `PlotResult` without a `show` parameter.

```{eval-rst}
.. automodule:: scarf.plotting
    :members: embedding, embedding_raster, dotplot, matrixplot, modality_weights, composition, distribution, cluster_connectivity, mapping_score, mapping_evidence, mapping_confusion, mapping_calibration, qc, graph_qc, elbow, highly_variable_features, label_panels, collect_legends, compose_results, register_theme, theme_context, marker_heatmap, cluster_tree, pseudotime_heatmap, run_recipe
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
.. autoclass:: scarf.plotting.FeatureSummary
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
