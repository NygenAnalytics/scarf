# Contributing

## Contributions through pull requests

If you would like to add a new feature, fix a bug or make some improvements, please follow this
[guideline]. When planning a new feature, introduce the proposal and discuss it on the
[discussion page]. Automated contributors must also follow the repository `AGENTS.md` and any
instructions scoped to the directory being changed.

The project uses [Ruff] for formatting and linting. Before opening a pull
request, run the same static checks as CI:

```bash
uv run ruff check scarf profiling tests
uv run ruff format --check scarf profiling tests
uv run mypy scarf profiling
```

## Testing locally

You can run the tests locally on your branch with [pytest]. Configurations are in `pyproject.toml`.
Install the development, profiling, and test dependencies with:

```bash
uv sync --group dev --group profiling --extra test --extra extra
```

Python 3.12 or newer is required (`requires-python >=3.12`).

Two markers select the expensive parts of the suite: `slow` for tests that build
neighbourhood graphs or run iterative numerical workflows, and `integration` for tests
that need live network access. While iterating, skip both:

    uv run pytest -m "not slow and not integration"

CI runs the whole suite, so run `uv run pytest` before opening a pull request.

## Contributions to the documentation

You may contribute to the documentation by either adding new sections or modifying existing
sections. Install the documentation and test dependencies with
`uv sync --extra docs --extra test --extra extra`.

Executable docs are MyST markdown files with `{code-cell}` blocks, not standalone `.ipynb` files.
Sources live in `docs/source/quickstart.md` and `docs/source/tutorials/`. Executed outputs
are stored in `docs/.jupyter_cache/` and committed to the repo so Read the Docs can build HTML
without re-running notebooks on every build.

### Refresh the docs cache

Prose-only edits do not change the notebook execution hash. Refresh affected pages after changing
a code cell or another execution input. The execution fingerprint includes `scarf/`, `uv.lock`,
`pyproject.toml`, `docs/source/conf.py`, and the documentation runner. `validate-cache` compares
executable source hashes, so it can pass for outputs made stale by another execution input.

Execute one affected page locally with one worker:

    make -C docs execute-page PAGE=scrna_seq JOBS=1

Modal can execute pages in parallel when authentication and the
`scarf_profiling` environment are available:

    uv sync --group docs-modal --extra docs --extra extra
    make -C docs execute-docs-modal PAGES="scrna_seq"

Both paths preserve matching outputs for other sources, validate a complete
candidate, and publish through a recoverable backup-and-rename sequence. If
execution, import, validation, or publication fails, the committed cache remains
unchanged. Resume a failed run with the matching `resume-docs` or
`resume-docs-modal` target and the same scope.

Never run two execute, resume, prune, or publication commands concurrently.

The executor converts completed live progress widgets into static, accessible
bars before caching them. Commit both the edited `.md` files and
`docs/.jupyter_cache/`.

### Other doc commands

Validate the committed cache without changing it:

    make -C docs validate-cache

Build HTML locally after cache validation:

    make -C docs html

Rebuild the cache from outputs that still match current sources:

    make -C docs prune-stale-cache

Force every page and run a strict Sphinx build:

    make -C docs execute-notebooks-all JOBS=1

### Adding a new tutorial

1. Add `docs/source/tutorials/your_tutorial.md` with MyST `{code-cell}` blocks, or convert from
   Jupyter with [Jupytext].
2. Register it in `docs/source/toctree.yml`.
3. Execute the page locally with `JOBS=1`, or use the optional Modal target when
   its environment is available.
4. Commit the `.md` file and `docs/.jupyter_cache/`.

Suggested chapter outline:

1. Short intro and when to use the page
2. Prerequisites
3. What you will learn
4. Dataset
5. Guided analysis (numbered steps)
6. Common mistakes and limitations

Fact-check before merging: method names and defaults against `scarf/`, dataset IDs against the
`scarf_docs` Cytebase repository, metadata keys against executed output, and method claims against
the capabilities Scarf ships.

The documentation uses [Sphinx], the [MyST] parser, and [myst_nb] for notebook execution.
Sphinx reads the committed cache via `nb_execution_mode = "cache"` in `docs/source/conf.py`.

Use `scarf.configure_output(level='DEBUG', progress=True)` when debugging
tutorial execution. Tutorials download datasets over the network. Timeout per
page is 600 seconds (`nb_execution_timeout` in `conf.py`).

### Republishing the example stores

Pages that are not about building an analysis chain open a pre-analyzed store
with `download_dataset(..., zarr=True)`. Those stores are rebuilt from each
dataset's raw counts by `scripts/regenerate_docs_datasets.py`, which also writes
a manifest under `docs/source/developers/dataset_manifests/` recording the
recipe, the cell counts, the artifact inventory, and the archive checksum:

    uv run python scripts/regenerate_docs_datasets.py --all

Rebuild a store whenever its recipe changes, or whenever the stored layout
changes in a way that would stop the published artifacts from being reused.
Nothing leaves `build/cytebase` until you publish:

    uv run python scripts/publish_docs_datasets.py            # print the plan
    uv run python scripts/publish_docs_datasets.py --apply    # needs a write token

Publishing swaps `<dataset>/data.zarr.tar.gz` in place and first preserves the archive it replaces
as `<dataset>_legacy_master/data.zarr.tar.gz`. Preservation is a server-side copy by content hash,
and it never overwrites a legacy snapshot that already exists. Those snapshots are the pre-1.0
Zarr v2 corpus that `tests/test_frozen_master_compat.py` reads; no documentation page opens them.

## Acknowledgements

### Contributors

Contributors to the Scarf repository. Thank you everyone!

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
[discussion page]: https://github.com/parashardhapola/scarf/discussions
[Ruff]: https://docs.astral.sh/ruff/
[Sphinx]: https://www.sphinx-doc.org
[MyST]: https://myst-parser.readthedocs.io/en/latest/index.html
[myst_nb]: https://myst-nb.readthedocs.io/
[Jupytext]: https://jupytext.readthedocs.io/en/latest/index.html
[pytest]: https://docs.pytest.org/
