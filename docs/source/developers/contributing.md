# Contributing

## Contributions through pull requests

If you would like to add a new feature, fix a bug or make some improvements, please follow this [guideline].
When planning a new feature, introduce the proposal and discuss it on the [discussion page].
Automated contributors must also follow the repository `AGENTS.md` and any instructions scoped to the directory being changed.

The project uses [Ruff] for formatting and linting.
Before opening a pull request, run the same static checks as CI:

```bash
uv run ruff check scarf profiling tests
uv run ruff format --check scarf profiling tests
uv run mypy scarf profiling
```

## Testing locally

You can run the tests locally on your branch with [pytest].
Configurations are in `pyproject.toml`.
Install the development, profiling, and test dependencies with:

```bash
uv sync --group dev --group profiling --extra test --extra extra
uv run python -m tests.download_fixtures --with-h5ad
```

Python 3.12 or newer is required (`requires-python >=3.12`).

Two markers select the expensive parts of the suite: `slow` for tests that build neighbourhood graphs or run iterative numerical workflows, and `integration` for tests that need live network access.
While iterating, skip both:

    uv run pytest -m "not slow and not integration"

CI runs the whole suite, so run `uv run pytest` before opening a pull request.
For full CI parity, also run the visual regression step (or see `AGENTS.md`):

    MPLBACKEND=Agg SCARF_RUN_VISUAL_REGRESSION=1 \
      uv run pytest -n 0 -m visual tests/test_plotting_showcase.py

### Performance benchmarks

`tests/benchmarks/` times the operations that dominate large runs and projects each timing onto production sizes, such as one and ten million cells.
`test_kernels.py` covers the compute kernels behind the slowest pipeline stages, `test_stages.py` the `DataStore` stages of the recorded benchmark funnel, `test_cold_start.py` the import and compilation cost of a new process, and the `test_scaling_*.py` files known hot spots in readers, graphs, features, plotting, merge and export, and the agent.
Every benchmark also checks its result against an independent reference.
A normal test run executes each benchmark once at a small size, without timing it.
Timed runs need a quiet machine and their own process:

    SCARF_RUN_BENCHMARKS=1 uv run pytest -n 0 tests/benchmarks

Each benchmark times a ladder of input sizes and fits one of three models: a power law for kernels, a fixed overhead plus a per-unit rate for calls whose overhead hides their per-unit work at small sizes, and a constant for per-call overheads and the cold start of a new process.
Repeats are interleaved across sizes, so a change in machine load during the run slows every size alike instead of bending the fitted scaling.
The run prints each benchmark's largest measured time, its fitted scaling, and its projected single-thread time at the production sizes, with the change from the baseline.
A benchmark fails only when its slowdown exceeds timing noise (30% by default, after calibrating for machine speed) and projects to a meaningful delay at some production size: 2 s per million cells by default, so 2 s at one million and 20 s at ten million, or 0.5 s of fixed overhead per call.
Judging every size catches a superlinear cost that is still small at one million cells.
A growing power-law exponent also fails when it projects to a meaningful delay at ten million cells, even if small inputs look unchanged.
Scaling is judged only when both fits are tight (r-squared at least 0.97); a noisier fit is judged at its largest measured size alone.
Before failing, a benchmark measures its ladder again and keeps the fastest time at each size, so a regression must survive two measurements and transient noise cannot fail it.

The committed baseline, `tests/benchmarks/baseline.json`, comes from one machine.
The most reliable comparison measures both revisions on the same machine, for example from a worktree of the base revision:

    SCARF_RUN_BENCHMARKS=1 SCARF_BENCHMARK_OUTPUT=/tmp/base.json uv run pytest -n 0 tests/benchmarks
    # then, in the changed checkout:
    SCARF_RUN_BENCHMARKS=1 SCARF_BENCHMARK_BASELINE=/tmp/base.json uv run pytest -n 0 tests/benchmarks

Regenerate the committed baseline on a quiet machine after an intended performance change:

    SCARF_RUN_BENCHMARKS=1 SCARF_BENCHMARK_UPDATE=1 uv run pytest -n 0 tests/benchmarks

`SCARF_BENCHMARK_SLOWDOWN` sets the tolerated slowdown ratio, for example `1.5` when comparing across machines.
`SCARF_BENCHMARK_THREADS` sets the Numba thread count, one by default.
`SCARF_BENCHMARK_OUTPUT` sets where results are written, `build/benchmarks/latest.json` by default.

### Operation revisions

A fix that changes what an analysis operation computes for unchanged parameters and inputs must also stop Scarf from reusing the results that earlier code stored ({doc}`operation_revisions`).
To change an operation's outputs:

1. Decide with the ladder in {doc}`operation_revisions` whether the change needs no revision, a scoped revision, or a revision of every artifact of the operation.
2. Append the `OperationRevision` entry to the operation's tuple in `scarf/storage/operation_revisions.py`.
3. Add a test that a new artifact of the operation records the revision and that an artifact of the earlier revision is recomputed, not reused.
4. A change that needs no revision but still moves results is listed under {ref}`stable_identity_result_changes` on the operation revisions page.

## Contributions to the documentation

You may contribute to the documentation by either adding new sections or modifying existing sections.
Install the documentation and test dependencies with `uv sync --extra agent --extra docs --extra test --extra extra --extra cytebase`.

Executable docs are MyST markdown files with `{code-cell}` blocks, not standalone `.ipynb` files.
Sources live in `docs/source/quickstart.md` and `docs/source/tutorials/`.
Executed outputs are stored in `docs/.jupyter_cache/` and committed to the repo so Read the Docs can build HTML without re-running notebooks on every build.

### Refresh the docs cache

Prose-only edits do not change the notebook execution hash.
Refresh affected pages after changing a code cell or another execution input.
The execution fingerprint includes `scarf/`, `uv.lock`, `pyproject.toml`, `docs/source/conf.py`, and the documentation runner.
`validate-cache` compares executable source hashes, so it can pass for outputs made stale by another execution input.

Execute one affected page locally with one worker:

    make -C docs execute-page PAGE=scrna_seq JOBS=1

Modal can execute pages in parallel when authentication and the `scarf_profiling` environment are available:

    uv sync --group docs-modal --extra agent --extra docs --extra extra
    make -C docs execute-docs-modal PAGES="scrna_seq"

Both paths preserve matching outputs for other sources, validate a complete candidate, and publish through a recoverable backup-and-rename sequence.
If execution, import, validation, or publication fails, the committed cache remains unchanged.
Resume a failed run with the matching `resume-docs` or `resume-docs-modal` target and the same scope.

Never run two execute, resume, prune, or publication commands concurrently.

The executor converts completed live progress widgets into static, accessible bars before caching them.
Commit both the edited `.md` files and `docs/.jupyter_cache/`.

### Other doc commands

For local verification that matches CI, validate the committed cache and run the strict reference build (see also `docs/AGENTS.md`):

    make -C docs validate-cache
    make -C docs check-reference

`check-reference` is a nitpicky, warnings-as-errors Sphinx build plus reference coverage.
A plain `make -C docs html` build is fine for browsing output, but it is not the CI-matching path.

Rebuild the cache from outputs that still match current sources:

    make -C docs prune-stale-cache

Force every page and run a strict Sphinx build:

    make -C docs execute-notebooks-all JOBS=1

### Adding a new tutorial

1. Add `docs/source/tutorials/your_tutorial.md` with MyST `{code-cell}` blocks, or convert from Jupyter with [Jupytext].
2. Register it in `docs/source/toctree.yml`.
3. Execute the page locally with `JOBS=1`, or use the optional Modal target when its environment is available.
4. Commit the `.md` file and `docs/.jupyter_cache/`.

For Cytebase, `Catalog()` defaults to the public `Nygen/cytebase` bucket, which
needs no credentials. To refresh the saved outputs, leave `CYTEBASE_BUCKET` and
Hugging Face tokens unset so the page reads the public bucket anonymously, then run
`make -C docs execute-page PAGE=cytebase JOBS=1`.
The tutorial reads the selected bucket; it does not invoke the
ingestion pipeline or write to remote stores. Keep private connection values
out of sources and cached outputs. Its explicit notebook download link uses
MyST-NB's generated `docs/build/jupyter_execute/tutorials/cytebase.ipynb`, so the
MyST source and executed cache remain the maintained tutorial. Read the Docs
renders the matching cache without needing bucket credentials.

Suggested chapter outline:

1. Short intro and when to use the page
2. Prerequisites
3. What you will learn
4. Dataset
5. Guided analysis (numbered steps)
6. Common mistakes and limitations

Fact-check before merging: method names and defaults against `scarf/`, dataset IDs against the `scarf_docs` Cytebase repository, metadata keys against executed output, and method claims against the capabilities Scarf ships.

### Documentation tooling

The documentation uses [Sphinx], the [MyST] parser, and [myst_nb] for notebook execution.
Sphinx reads the committed cache via `nb_execution_mode = "cache"` in `docs/source/conf.py`.

Use `scarf.configure_output(level='DEBUG', progress=True)` when debugging tutorial execution.
Tutorials download datasets over the network.
Timeout per code cell is 600 seconds (`nb_execution_timeout` in `conf.py`).

### Republishing the example stores

Pages that are not about building an artifact lineage open a pre-analyzed store with `download_dataset(..., zarr=True)`.
Source stores are rebuilt from raw counts, while declared derived stores are rebuilt from their published inputs.
`scripts/regenerate_docs_datasets.py` writes both kinds and creates a manifest under
`docs/source/developers/dataset_manifests/` recording the recipe, cell counts, artifact and
pipeline-run inventories, and archive checksum:

    uv run python scripts/regenerate_docs_datasets.py --all

`--all` excludes checksum-pinned external recipes so routine regeneration does not trigger large third-party downloads.
Rebuild one of those recipes by name:

    uv run python scripts/regenerate_docs_datasets.py swanson_7K_pbmc_teaseq

Rebuild a store whenever its recipe changes, or whenever the stored layout changes in a way that would stop the published artifacts from being reused.
Nothing leaves `build/cytebase` until you publish:

    uv run python scripts/publish_docs_datasets.py            # print the plan
    uv run python scripts/publish_docs_datasets.py --apply    # needs a write token

Publishing swaps `<dataset>/data.zarr.tar.gz` in place and first preserves the archive it replaces as `<dataset>_legacy_master/data.zarr.tar.gz`.
Preservation is a server-side copy by content hash, and it never overwrites a legacy snapshot that already exists.
Those snapshots are the pre-1.0 Zarr v2 corpus that `tests/test_frozen_master_compat.py` reads; no documentation page opens them.

## Releasing

Publishing a GitHub release runs `.github/workflows/publish.yml`.
Every job checks out the exact commit that the release tag named when the release was published (`github.sha`), and the upload to PyPI waits for four gates on that commit:

- `verify` runs the test workflow, `pytest.yml`: the static checks, the visual regression comparison, and the complete suite on Python 3.12, 3.13, and 3.14 and with the lowest direct dependency versions. A release uploads no coverage report.
- `docs` runs the documentation workflow, `docs.yml`: the documentation tests, the committed notebook cache check, and the nitpicky Sphinx build with reference coverage.
- `build` first requires the release tag to still name that commit, then builds the wheel and the source distribution from a clean checkout and checks their metadata against the tag.
- `smoke` runs `tests/smoke_wheel.py` on the built wheel on Linux with Python 3.12, 3.13, and 3.14 and on Windows with Python 3.12.

`tests/smoke_wheel.py` checks that the wheel is pure Python and complete, installs it into a clean environment with `uv venv` and `uv pip install`, imports the public modules from the installed wheel, and runs `tests/smoke_workflow.py` with that environment's interpreter in isolated mode.
That script imports a synthetic three-population count matrix with `SparseToZarr` and analyzes it with `pipeline.run`; the run must complete, its UMAP coordinates must be finite, and Leiden must find at least two clusters.
It then exports the run's raw counts and normalized values with `to_h5ad` in that environment, which has no `anndata`, and reads both files back with h5py.
On Linux x86_64 the environment installs the `tsne` extra and the run must also produce finite t-SNE coordinates; on every other platform the environment has no `sgtsnepi`, and `run_tsne` must raise the `ImportError` that names the extra.
No smoke environment may have an `sgtsne` executable on `PATH`.

Nothing is uploaded when a gate fails, and runs for the same tag publish one at a time.
Fix the cause on `master`, then publish a release whose tag names the fixed commit.
Re-running a failed workflow tests the same commit again, which helps only when the failure came from outside the commit, such as a network error.

Run the smoke locally against a wheel you build. It installs the wheel's dependencies, so it needs network access or a warm uv cache:

    uv build --wheel --clear --out-dir dist
    uv run --no-project python tests/smoke_wheel.py dist/scarf-*.whl

## Acknowledgements

### Contributors

Contributors to the Scarf repository.
Thank you everyone!

```{eval-rst}
.. include:: ../contributors.rst
```

### Open-source stack

A diverse number of open-source packages in Python scientific stack are being used to build Scarf.
Here we acknowledge some of them (at least those with pretty logos).

```{eval-rst}
.. include:: ../logos.rst
```

[guideline]: https://www.dataschool.io/how-to-contribute-on-github
[discussion page]: https://github.com/NygenAnalytics/scarf/discussions
[Ruff]: https://docs.astral.sh/ruff/
[Sphinx]: https://www.sphinx-doc.org
[MyST]: https://myst-parser.readthedocs.io/en/latest/index.html
[myst_nb]: https://myst-nb.readthedocs.io/
[Jupytext]: https://jupytext.readthedocs.io/en/latest/index.html
[pytest]: https://docs.pytest.org/
