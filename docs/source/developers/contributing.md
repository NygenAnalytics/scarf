# Contributing

## Contributions through pull requests

If you would like to add a new feature, fix a bug or make some improvements please
follow this [guideline]. Usually when planning to add a new feature it is a good
idea to introduce the proposed feature and discuss it on the [discussion page].
The project uses [Ruff] for formatting and linting. Before opening a pull request, run
`uv run ruff format .` and `uv run ruff check .`.

## Testing locally

You can run the tests locally on your branch with [pytest]. Configurations are in `pyproject.toml`.
Install dependencies with `uv sync --extra test --extra extra`, then run `uv run pytest`.
Python 3.12 or newer is required (`requires-python >=3.12`).

Two markers select the expensive parts of the suite: `slow` for tests that build
neighbourhood graphs or run iterative numerical workflows, and `integration` for tests
that need live network access. While iterating, skip both:

    uv run pytest -m "not slow and not integration"

CI runs the whole suite, so run `uv run pytest` before opening a pull request.

## Contributions to the documentation

You may contribute to the documentation by either adding new sections or modifying existing
sections. Install doc dependencies with `uv sync --extra docs --extra extra`.

Executable docs are MyST markdown files with `{code-cell}` blocks, not standalone `.ipynb` files.
Sources live in `docs/source/quickstart.md` and `docs/source/tutorials/`. Executed outputs
are stored in `docs/.jupyter_cache/` and committed to the repo so Read the Docs can build HTML
without re-running notebooks on every build.

### Refresh the docs cache

After editing a tutorial or quickstart page:

    uv sync --group docs-modal --extra docs --extra extra
    make -C docs execute-docs-modal PAGES="scrna_seq"

The Make target uses the `scarf_profiling` Modal environment and starts an
ephemeral container for each requested page. The local process preserves
matching outputs for other current sources, validates a complete candidate, and
publishes it through a recoverable backup-and-rename sequence. If execution,
import, validation, or publication fails, the committed cache remains unchanged.

Omit `PAGES` to force every executable page. A failed run retains successful
staged pages. Resume the same scope with
`make -C docs resume-docs-modal`; staged outputs are reused only while their
source hash, execution fingerprint, and Modal runner identity still match.
`make -C docs execute-page PAGE=scrna_seq` remains available when Modal access
is not configured.

The executor converts completed live progress widgets into static, accessible
bars before caching them. Commit both the edited `.md` files and
`docs/.jupyter_cache/`.

### Other doc commands

Validate the committed cache without changing it:

    cd docs && make validate-cache

Build HTML locally after cache validation:

    cd docs && make html

Rebuild the cache from outputs that still match current sources:

    cd docs && make prune-stale-cache

Force every page and run a strict Sphinx build:

    cd docs && make execute-notebooks-all

### Adding a new tutorial

1. Add `docs/source/tutorials/your_tutorial.md` with MyST `{code-cell}` blocks (or convert from Jupyter with [Jupytext]).
2. Register it in `docs/source/toctree.yml`.
3. Run `make -C docs execute-docs-modal PAGES="your_tutorial"`.
4. Commit the `.md` file and `docs/.jupyter_cache/`.

Suggested chapter outline:

1. Short intro and when to use the page
2. Prerequisites
3. What you will learn
4. Dataset
5. Guided analysis (numbered steps)
6. Common mistakes and limitations

Fact-check before merging: method names and defaults against `scarf/`, dataset IDs against the `scarf_docs` Cytebase repository, metadata keys against executed output, and no claims for methods Scarf does not ship.

The documentation uses [Sphinx], the [MyST] parser, and [myst_nb] for notebook execution.
Sphinx reads the committed cache via `nb_execution_mode = "cache"` in `docs/source/conf.py`.

Use `scarf.configure_output(level='DEBUG', progress=True)` when debugging
tutorial execution. Tutorials download datasets over the network. Timeout per
page is 600 seconds (`nb_execution_timeout` in `conf.py`).

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
