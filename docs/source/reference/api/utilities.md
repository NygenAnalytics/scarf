# Cytebase and utilities

```{eval-rst}
.. autofunction:: scarf.cytebase.list_repositories
```

```{eval-rst}
.. autofunction:: scarf.cytebase.connect
```

```{eval-rst}
.. autoclass:: scarf.cytebase.Repository
   :members:
```

```{eval-rst}
.. autofunction:: scarf.load_zarr
```

```{eval-rst}
.. autofunction:: scarf.controlled_compute
```

```{eval-rst}
.. autofunction:: scarf.read_gmt
```

## Logging and progress

Scarf starts with `INFO` logs and progress enabled, without timestamps. Notebook
sessions do not need an output setup call. For a batch pipeline, configure the
independent settings once at startup:

```python
scarf.configure_output(progress=False, timestamps=True)
```

Arguments that are omitted retain their current values.

```{eval-rst}
.. autofunction:: scarf.configure_output
```

```{eval-rst}
.. autofunction:: scarf.set_verbosity
```

```{eval-rst}
.. autofunction:: scarf.get_log_level
```

```{eval-rst}
.. py:data:: scarf.logger

    Scarf's `loguru` logger. Library code logs through this object, so
    :func:`scarf.configure_output` controls the Scarf-owned sink while preserving
    sinks added by callers.
```

```{eval-rst}
.. autofunction:: scarf.tqdmbar
```

```{eval-rst}
.. py:data:: scarf.tqdm_params

    Default keyword arguments applied by :func:`scarf.tqdmbar`, including bar
    format, dynamic width, and colour. Override any of them per call.
```

```{eval-rst}
.. autofunction:: scarf.show_dask_progress
```

```{eval-rst}
.. autofunction:: scarf.system_call
```

## Array helpers

These operate on plain arrays and are used by Scarf's own normalization and feature
selection code.

```{eval-rst}
.. autofunction:: scarf.clean_array
```

```{eval-rst}
.. autofunction:: scarf.rescale_array
```

```{eval-rst}
.. autofunction:: scarf.rolling_window
```

```{eval-rst}
.. autofunction:: scarf.permute_into_chunks
```
