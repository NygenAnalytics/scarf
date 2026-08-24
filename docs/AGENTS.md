# Documentation instructions

These instructions cover documentation sources, execution, cache management, and publication under `docs/`.

## Model

- Sources are MyST Markdown under `docs/source/`; executable cells use `{code-cell}` blocks.
- Executed outputs live in the committed `docs/.jupyter_cache/`. Sphinx reads that cache and does
  not execute pages during a normal build.
- Prose-only edits do not change the notebook execution hash. Re-execute a page when its code cells
  or execution inputs change. Execution inputs include `scarf/`, `uv.lock`, `pyproject.toml`,
  `docs/source/conf.py`, and the documentation runner.
- Cache validation compares executable source hashes, not the execution-input fingerprint.
  Re-execute affected pages after an execution-input change even if `validate-cache` passes.
- `docs/source/developers/contributing.md` is the canonical tutorial-authoring guide.

## Dependencies and validation

Install the same documentation and test extras used by CI:

```bash
uv sync --extra docs --extra test --extra extra
```

Run the cache tests and strict reference build:

```bash
uv run pytest -n 0 tests/test_docs_cache.py tests/test_docs_structure.py
make -C docs validate-cache
make -C docs check-reference
```

`check-reference` performs a nitpicky, warnings-as-errors Sphinx build. Its coverage check requires
every public `DataStore` method exactly once and each top-level `scarf.__all__` export at least
once. It does not exhaustively check every module-level public surface.

## Local execution

Each page can use several GiB. Keep local execution at one worker, especially on WSL:

```bash
make -C docs execute-page PAGE=scrna_seq JOBS=1
make -C docs execute-docs JOBS=1
```

Resume a failed local run with the same scope:

```bash
make -C docs resume-docs JOBS=1
```

## Optional Modal execution

Use Modal only after the user confirms authentication and the `scarf_profiling` environment are
available. This path uses an ephemeral app and does not require deployment:

```bash
uv sync --group docs-modal --extra docs --extra extra
make -C docs execute-docs-modal
make -C docs execute-docs-modal PAGES="scrna_seq cite_seq"
```

Resume with `make -C docs resume-docs-modal`, repeating the original `PAGES` scope. If Modal is
unavailable, use the local targets.

## Concurrency and publication safety

- Never start two execute, resume, prune, or publication commands at once, including from separate
  agents. They share one cache and one resume area.
- Prefer one command covering the full changed scope over independent page commands.
- The runner rejects code-cell and execution-input changes during execution. Prose-only edits are
  not part of that comparison, but do not edit executable pages while a run is active.
- After execution, run `validate-cache` and `check-reference`.
- Commit changed executable sources and the matching cache update together.
- Do not run dataset publication tools or use write credentials unless the user explicitly
  requests publication.

## Troubleshooting

- Missing or stale cache record: run `make -C docs validate-cache` before attempting execution.
- Failed execution with unchanged source and runtime fingerprint: resume the matching local or
  Modal scope.
- Broken `{doc}`, `{ref}`, or Python references: run `make -C docs check-reference`.
- Wrong tutorial values: fact-check method defaults against `scarf/`, dataset identifiers against
  manifests, and metadata keys against executed output.
