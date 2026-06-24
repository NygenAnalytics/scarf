# How to contribute

## Contributions through pull requests:
If you would like to add a new feature, fix a bug or make some improvements please
follow this [guideline]. Usually when planning to add a new feature it is a good
idea to introduce the proposed feature and discuss it. This can be done on the [discussion
page]. The code is written using [black] style. Please make sure you blacken any edited files. 

## Testing locally
You can run the tests locally on your branch with [pytest]. Configurations are in `pyproject.toml`.
Install dependencies with `uv sync --extra test --extra extra`, then run `uv run pytest`.
Python 3.14 is required (`requires-python >=3.14,<3.15`).

## Contributions to the documentation
You may contribute to the documentation by either adding new sections or modifying existing
sections. All the required packages for building the documentation locally can be installed
with `uv pip install -e ".[extra,docs]"`. The documentation is built
using [Sphinx]. The markdown files are parsed using [MyST] parser. If you want to add a notebook
to  the documentation then please convert your `ipynb` files to `markdown` format using [Jupytext].
The markdown files are saved with extension `mdnb` so that they can be recognized by [nbshpinx]
extension. 

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
[discussion page]: https://www.dataschool.io/how-to-contribute-on-github
[black]: https://black.readthedocs.io/en/stable
[Sphinx]: https://www.sphinx-doc.org
[MyST]: https://myst-parser.readthedocs.io/en/latest/index.html
[Jupytext]: https://jupytext.readthedocs.io/en/latest/index.html
[nbsphinx]: https://nbsphinx.readthedocs.io
