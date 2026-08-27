# Repository instructions

These instructions apply to the whole repository.

## Start here

- Edit package code in `scarf/`, tests in `tests/`, documentation in `docs/`, and profiling tools
  in `profiling/`.
- Read `docs/AGENTS.md` before executing or publishing documentation.
- Read `profiling/AGENTS.md` before using profiling tools or cloud resources.
- Use `docs/source/developers/architecture.md` for code placement and dependency boundaries, and
  `docs/source/developers/contributing.md` for the human contribution workflow.
- Biological decision guidance belongs in `docs/source/analysis_with_agents.md`, not in repository
  instructions.

## Scarf developer mental model

- Scarf streams Zarr-backed matrices in bounded blocks. Do not materialize a complete matrix when
  a blockwise path exists.
- `DataStore` orchestrates workflows. Reusable computation belongs in its concrete domain package
  before a thin `datastore._operations` method exposes it.
- Dependencies flow from `storage`, `matrix`, and `utils`, through data models and domain
  algorithms, to I/O, datastore orchestration, and plotting.
- Saved computations are immutable artifacts. `ArtifactRef`, provenance, and durable `PipelineRun`
  records connect frozen selections, results, reuse, and complete workflow invocations.
- The lazy public facades are compatibility boundaries. Trace the public call, operation
  implementation, domain algorithm, persisted result, callers, tests, and documentation before
  changing a contract.
- Plotting consumes narrow adapters and must not gain a module-load dependency on `datastore`.

## Source and conventions

- Python 3.12 or newer is required. Use `uv` for every Python command and dependency change.
- Preserve unrelated working-tree changes. Trace callers and tests before changing a public
  function or persisted result.
- Use built-in generics and `X | None`. Do not add `typing.List`,
  `typing.Optional`, or `from __future__ import annotations`.
- Pydantic model attributes use camel case.
- Do not add implicit compatibility branches, ad hoc schema versions, or silent migrations. Make a
  compatibility policy explicit before implementing it.
- Do not commit credentials, local object-store endpoints, private bucket names, generated
  benchmark configs, or result identifiers.
- Avoid em dashes in public prose, prompts, docstrings, and comments.

## Environment and checks

```bash
uv sync --group dev --group profiling --extra test --extra extra
uv run python -m tests.download_fixtures --with-h5ad
```

Run focused tests without pytest-xdist. Run the quick suite while iterating and the complete suite before handoff:

```bash
uv run pytest -n 0 tests/test_file.py::test_name
uv run pytest -m "not slow and not integration"
uv run pytest
```

Run the same static checks as CI:

```bash
uv run ruff check scarf profiling tests
uv run ruff format --check scarf profiling tests
uv run mypy scarf profiling
```

Visual references are intentional artifacts. Regenerate them only when rendering changes:

```bash
MPLBACKEND=Agg SCARF_RUN_VISUAL_REGRESSION=1 \
  uv run pytest -n 0 -m visual tests/test_plotting_showcase.py
```

## Troubleshooting router

- Wrong code location or import cycle: read the architecture placement rules and run
  `uv run pytest -n 0 tests/test_import_architecture.py`.
- Missing, stale, or incompatible result: inspect the pipeline-run report, artifact status, and
  lineage before reading private Zarr paths.
- Public API or persisted contract: inspect the public API reference and the public,
  compatibility, signature, and frozen-store tests.
- On-disk layout: read `docs/source/developers/zarr_internals.md`.
- Documentation cache or build: follow `docs/AGENTS.md`.
- Profiling, Modal, or benchmark interpretation: follow `profiling/AGENTS.md`.

## Universal safety

- Never run two documentation execute, resume, prune, or publication commands concurrently. They
  share one cache and resume area.
- Never deploy Modal. Deployment is a user action. Ask before any paid or long-running cloud job.
- Do not publish example datasets or use write credentials unless the user explicitly requests it.
