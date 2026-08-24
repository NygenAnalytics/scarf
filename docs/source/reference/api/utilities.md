# Datasets and utilities API reference

## Cytebase and store helpers

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

Scarf starts with `INFO` logs and progress enabled, without timestamps.
Notebook sessions do not need an output setup call.
For a batch pipeline, configure the independent settings once at startup:

```python
scarf.configure_output(progress=False, timestamps=True)
```

Arguments that are omitted retain their current values.
`configure_output` is the primary runtime interface:

- `level` controls Scarf's log severity.
- `progress` controls progress bars, including `compute_with_progress`.
- `timestamps` controls timestamps in Scarf's installed log sink.

Progress is independent of log level.
Setting a quiet level such as `WARNING` does not disable progress.
Code that previously relied on log level to suppress bars should call `configure_output(progress=False)` explicitly.
Interactive notebooks animate these bars.
Read the Docs shows deterministic completed snapshots from the committed notebook cache instead of replaying an animation.

```{eval-rst}
.. autofunction:: scarf.configure_output
```

```{eval-rst}
.. autofunction:: scarf.set_verbosity
```

`set_verbosity(level=..., filepath=...)` remains the interface for selecting a log level and optional file destination.
Use it when a durable batch log is needed; use `configure_output` for ordinary notebook and console behavior.

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
.. autofunction:: scarf.compute_with_progress
```

```{eval-rst}
.. autofunction:: scarf.system_call
```

## Array helpers

These operate on plain arrays and are used by Scarf's own embeddings, graph construction, trajectory dynamics, and merge code.

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
