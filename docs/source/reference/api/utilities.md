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

```{eval-rst}
.. autofunction:: scarf.set_verbosity
```

```{eval-rst}
.. autofunction:: scarf.get_log_level
```

```{eval-rst}
.. py:data:: scarf.logger

    Scarf's `loguru` logger. Library code logs through this object, so
    :func:`scarf.set_verbosity` controls what user sessions see.
```

```{eval-rst}
.. autofunction:: scarf.tqdmbar
```

```{eval-rst}
.. py:data:: scarf.tqdm_params

    Default keyword arguments applied by :func:`scarf.tqdmbar` (bar format, width,
    and colour). Override any of them per call.
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

```{eval-rst}
.. autofunction:: scarf.prefetch_blocks
```
