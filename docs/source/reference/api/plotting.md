# Plotting API reference

`scarf.plotting` is Scarf's plotting API. Import it as `splt` and call functions such as
`splt.embedding(...)`, `splt.dotplot(...)`, and `splt.cluster_tree(...)`.
For functions whose first argument is a datastore, `ds.plots.embedding(...)`
and related accessor methods provide the same behavior with that argument
already bound. Array and DataFrame diagnostics remain standalone.

Plot functions return a `PlotResult` and render by default with `show=True`. Pass
`show=False` before accessing, saving, or reusing an owned figure.

```{eval-rst}
.. automodule:: scarf.plotting
    :members: embedding, embedding_raster, dotplot, matrixplot, composition, distribution, cluster_connectivity, mapping_score, mapping_evidence, mapping_confusion, mapping_calibration, mapping_projection, qc, graph_qc, elbow, highly_variable_features, label_panels, collect_legends, compose_results, register_theme, theme_context, marker_heatmap, cluster_tree, pseudotime_heatmap, run_recipe
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
