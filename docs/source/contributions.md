# How to contribute

## Contributions through pull requests

If you would like to add a new feature, fix a bug or make some improvements please
follow this [guideline]. Usually when planning to add a new feature it is a good
idea to introduce the proposed feature and discuss it on the [discussion page].
The code is written using [black] style. Please make sure you blacken any edited files.

## Testing locally

You can run the tests locally on your branch with [pytest]. Configurations are in `pyproject.toml`.
Install dependencies with `uv sync --extra test --extra extra`, then run `uv run pytest`.
Python 3.12 or newer is required (`requires-python >=3.12`).

## Contributions to the documentation

You may contribute to the documentation by either adding new sections or modifying existing
sections. Install doc dependencies with `uv sync --extra docs --extra extra`.

Executable docs are MyST markdown files with `{code-cell}` blocks, not standalone `.ipynb` files.
Sources live in `docs/source/quickstart.md` and `docs/source/vignettes/`. Executed outputs
are stored in `docs/.jupyter_cache/` and committed to the repo so Read the Docs can build HTML
without re-running notebooks on every build.

### Refresh the docs cache

After editing a vignette or quickstart page:

    uv sync --extra docs --extra extra
    cd docs && make execute-docs

This runs all executable pages in parallel, updates `docs/.jupyter_cache/`, and prunes stale
cache entries for deleted source files. Commit both the edited `.md` files and `docs/.jupyter_cache/`.

### Other doc commands

Run one page only:

    cd docs && make execute-vignette VIGNETTE=basic_tutorial_scRNAseq

Build HTML locally without re-executing notebooks:

    cd docs && make html

Remove orphaned cache entries only:

    cd docs && make prune-stale-cache

Full sequential re-execution via Sphinx (slower fallback):

    cd docs && make execute-notebooks-all

### Adding a new vignette

1. Add `docs/source/vignettes/your_vignette.md` with MyST `{code-cell}` blocks (or convert from Jupyter with [Jupytext]).
2. Register it in `docs/source/toctree.yml`.
3. Run `cd docs && make execute-docs`.
4. Commit the `.md` file and `docs/.jupyter_cache/`.

The documentation uses [Sphinx], the [MyST] parser, and [myst_nb] for notebook execution.
Sphinx reads the committed cache via `nb_execution_mode = "cache"` in `docs/source/conf.py`.

Use `scarf.set_verbosity('DEBUG')` when debugging vignette execution locally. Vignettes
download datasets over the network. Timeout per page is 200 seconds (`nb_execution_timeout` in `conf.py`).

# Acknowledgements

## Contributors

Contributors to the Scarf repository. Thank you everyone!

```{eval-rst}
.. include:: contributors.rst
```

## Open-source stack

A diverse number of open-source packages in Python scientific stack are being used to build Scarf.
Here we acknowledge some of them (atleast those with pretty logos..)

```{eval-rst}
.. include:: logos.rst
```

[guideline]: https://www.dataschool.io/how-to-contribute-on-github
[discussion page]: https://github.com/parashardhapola/scarf/discussions
[black]: https://black.readthedocs.io/en/stable
[Sphinx]: https://www.sphinx-doc.org
[MyST]: https://myst-parser.readthedocs.io/en/latest/index.html
[myst_nb]: https://myst-nb.readthedocs.io/
[Jupytext]: https://jupytext.readthedocs.io/en/latest/index.html
[pytest]: https://docs.pytest.org/
