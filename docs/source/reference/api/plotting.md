# Plotting

`scarf.plotting` is Scarf's plotting API. Import it as `splt` and call functions such as
`splt.embedding(...)`, `splt.dotplot(...)`, and `splt.cluster_tree(...)`.

Plot functions return a `PlotResult` and render by default with `show=True`. Pass
`show=False` before accessing, saving, or reusing an owned figure.

```{eval-rst}
.. automodule:: scarf.plotting
    :members: embedding, embedding_raster, unified_embedding, dotplot, matrixplot, composition, distribution, qc, graph_qc, elbow, highly_variable_features, label_panels, collect_legends, theme_context, marker_heatmap, cluster_tree, pseudotime_heatmap
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
.. autoclass:: scarf.plotting.PlotProvenance
    :members:
```
