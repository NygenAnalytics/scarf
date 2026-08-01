# Repository instructions

These instructions apply to the whole repository.

## Source and conventions

- Python 3.12 or newer is required. Use `uv` for every Python command and
  dependency change.
- Edit package code in `scarf/`, tests in `tests/`, documentation in `docs/`,
  and profiling tools in `profiling/`.
- Preserve unrelated working-tree changes. Trace callers and tests before
  changing a public function or persisted result.
- Use built-in generics and `X | None`. Do not add `typing.List`,
  `typing.Optional`, or `from __future__ import annotations`.
- Pydantic model attributes use camel case.
- Do not add implicit compatibility branches, ad hoc schema versions, or silent
  migrations. Make a compatibility policy explicit before implementing it.
- Do not commit credentials, local object-store endpoints, private bucket
  names, generated benchmark configs, or result identifiers.
- Avoid em dashes in public prose, prompts, docstrings, and comments.

## Environment and checks

Install the development, test, and profiling dependencies:

```bash
uv sync --group dev --group profiling --extra test --extra extra
```

On a clean checkout, download the test fixtures before running the complete
suite:

```bash
uv run python -m tests.download_fixtures --with-h5ad
```

Run a focused test without pytest-xdist:

```bash
uv run pytest -n 0 tests/test_file.py::test_name
```

Run the quick local suite while iterating, then the complete suite before handoff:

```bash
uv run pytest -m "not slow and not integration"
uv run pytest
```

Run the same static checks used by CI:

```bash
uv run ruff check scarf profiling tests
uv run ruff format --check scarf profiling tests
uv run mypy scarf profiling
```

Visual references are intentional test artifacts. Do not regenerate them unless
the requested change affects rendering:

```bash
MPLBACKEND=Agg SCARF_RUN_VISUAL_REGRESSION=1 \
  uv run pytest -n 0 -m visual tests/test_plotting_showcase.py
```

## Documentation

Documentation sources are MyST Markdown under `docs/source/`. Executed code-cell
outputs live in the committed `docs/.jupyter_cache/`. Sphinx builds HTML from
that cache and does not execute the pages.

Install documentation dependencies and run the strict local build:

```bash
uv sync --extra docs --extra extra
uv run pytest -n 0 tests/test_docs_cache.py
make -C docs check-reference
```

Execute pages locally with one worker. Each page runs in its own subprocess and
can take several GB, so keep `JOBS` at 1 even for a full rebuild, especially on
WSL. Refresh a single changed page, or rebuild every page in one command:

```bash
make -C docs execute-page PAGE=scrna_seq JOBS=1
make -C docs execute-docs JOBS=1
```

Modal executes pages in parallel instead, one single-use container per page with
2 CPUs and 8 GiB. Omit `PAGES` to run every page, or name pages for a targeted
run. Modal is optional. Use it only when the user confirms that Modal
authentication and the `scarf_profiling` environment are available:

```bash
uv sync --group docs-modal --extra docs --extra extra
make -C docs execute-docs-modal
make -C docs execute-docs-modal PAGES="scrna_seq cite_seq"
```

That fan-out is safe because one local coordinator merges and publishes the
cache once. The command uses an ephemeral Modal app and needs no deployment. If
Modal is unavailable, use the local targets.

Do not start two documentation execution, resume, prune, or publication commands
at the same time, including from separate agents. They share one cache and one
resume area. The runner serializes publication and rejects source changes made
during execution, so concurrent invocations waste work or invalidate a run. Run
one full command rather than several page commands at once.

Resume a failed run with the matching `resume-docs` or `resume-docs-modal`
target, repeating the `PAGES` scope the failed run used. After execution,
validate and build:

```bash
make -C docs validate-cache
make -C docs check-reference
```

Do not publish example datasets or use write credentials unless the user
explicitly requests it.

## Profiling

`profiling/` contains the current end-to-end cloud profiler and targeted
diagnostics. Public reference results are in `profiling/BENCHMARKS.md`.

Run its local tests before using cloud resources:

```bash
uv run pytest -n 0 tests/test_profiling_*.py
```

Create a local config from `profiling/config.example.toml`. Keep
`profiling/config.toml` and any machine-specific variants untracked. Set a fresh,
non-empty `runTag` for every end-to-end measurement.

Profiling uses a deployed Modal app so spawned jobs survive a local disconnect.
Deployment is a user action. Never run `modal deploy`; ask the user to run:

```bash
uv run --group profiling modal deploy --env scarf_profiling -m profiling.modal_app
```

After the user confirms deployment, prepare deterministic CELLxGENE samples and
spawn the current end-to-end funnel:

```bash
uv run --group profiling modal run --env scarf_profiling -m profiling.modal_app -- prepare --config profiling/config.toml
uv run --group profiling modal run --env scarf_profiling -m profiling.modal_app -- run-e2e --config profiling/config.toml --size 1000000
```

`prepare` returns after spawning. Confirm its result before starting `run-e2e`.
Use `run --stage ...` for a targeted stage, including rewriting an incomplete
`countsT`, `run-local` for the Modal ephemeral-disk comparison, and
`io-baseline` for read-pattern diagnostics.

Treat local connectivity as unreliable. A long job must survive a dropped
laptop connection:

- Ask before starting a paid or long-running Modal job.
- Do not create Modal environments, secrets, or credentials.
- Long jobs must use `.spawn(...)`, never `.remote()`. A blocking call can
  cancel its own input when the client gRPC session dies, which lost a
  completed 1M measurement. Wait through short `FunctionCall.get(timeout=...)`
  polls or durable result JSON instead.
- Give coordinators about 1 CPU and 2 to 4 GiB with `retries=0`. Spawning them
  with stage memory makes them compete with real workers for scarce
  high-memory capacity.
- Treat `FAILURE`, `INIT_FAILURE`, `TERMINATED`, and `TIMEOUT` as terminal when
  polling. Modal can raise an empty `TimeoutError` for a failed input, so
  waiting on that alone hangs until the stage deadline.
- Prefer the broad `eu` region over a narrow one, and leave the Modal `cloud`
  option unset. Pinning a provider shrinks the schedulable pool.
- Log a start line, a plan line, periodic progress, and a done line. A silent
  multi-hour run cannot be diagnosed from `modal app logs`.
- Do not run two jobs with the same `runTag`.
- Persist stage and funnel JSON before treating a run as complete.
- Change one measured variable at a time and keep workflow seeds fixed.
- Do not present different machine sizes as one scaling curve.
