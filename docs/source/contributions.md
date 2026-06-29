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
sections. Install doc dependencies with `uv pip install -e ".[extra,docs]"`.

Vignettes are MyST markdown files under `docs/source/vignettes/` with executable `{code-cell}` blocks.
After editing a vignette, execute it locally with:

    cd docs && make execute-notebooks

Then commit both the source `.md` files and `docs/.jupyter_cache/`. Read the Docs
builds HTML with Sphinx and reuses that cache instead of re-running notebooks on every build.

The documentation uses [Sphinx], the [MyST] parser, and [myst_nb] for notebook execution.
To add a tutorial from an existing Jupyter notebook, convert it with [Jupytext] to MyST markdown.

Use `scarf.set_verbosity('DEBUG')` when debugging vignette execution locally.

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
