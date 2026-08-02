import os
import sys

import matplotlib

sys.path.insert(0, os.path.abspath("../.."))

project = "Scarf"
copyright = "2020-2026, Parashar Dhapola"
author = "Parashar Dhapola"

extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_external_toc",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_reredirects",
    "sphinxcontrib.mermaid",
    "myst_nb",
]
# Explicit autosummary/autodoc on reference pages only. Do not enable recursive
# generation over the whole package until public __all__ boundaries are complete.
autosummary_generate = False

autodoc_type_aliases = {
    "DataStore": "scarf.datastore.datastore.DataStore",
    "DataStorePlotAccessor": "scarf.datastore.plot_accessor.DataStorePlotAccessor",
}

templates_path = ["_templates"]
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "vignettes/dev",
    "tutorials/dev",
    "scarf_datasets",
    "**/scarf_datasets",
]

pygments_style = "sphinx"
language = "en"

external_toc_path = "toctree.yml"
external_toc_exclude_missing = False
redirects = {
    "about": "index.html",
    "install": "installation.html",
    "tutorials/atomic_graph_operations": "custom_graph_construction.html",
    "concepts/graph_and_state": "../tutorials/custom_graph_construction.html",
    "reference/api/graph_ops": "graph_construction.html",
    "tutorials/dimensionality_reduction_and_clustering": (
        "dimensionality_reduction.html"
    ),
}
myst_enable_extensions = [
    "colon_fence",
]


html_theme = "sphinx_book_theme"
html_favicon = "favicon.ico"
html_logo = "_static/scarf-logo-black.png"
html_title = "Scarf documentation"
html_baseurl = "https://scarf.readthedocs.io/en/latest/"
html_theme_options = {
    "repository_url": "https://github.com/NygenAnalytics/scarf",
    "home_page_in_toc": False,
    "path_to_docs": "docs/source",
    "show_navbar_depth": 2,
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "navigation_with_keys": False,
    "toc_title": "Sections",
    "logo": {
        "image_light": "_static/scarf-logo-black.png",
        "image_dark": "_static/scarf-logo-white.png",
    },
}
html_static_path = ["_static"]
html_extra_path = ["llms.txt"]
html_css_files = ["styles.css"]

htmlhelp_basename = "Scarf Documentation"

man_pages = [(master_doc, "scarf", "Scarf Documentation", [author], 1)]

nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_mode = "cache"
nb_execution_cache_path = os.path.join(
    os.path.dirname(__file__), "..", ".jupyter_cache"
)
nb_execution_timeout = 600
nb_merge_streams = True

matplotlib.use("agg")

# Suppress noisy autodoc type cross-refs until intersphinx inventories are complete (P7).
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.dtype"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "scipy.sparse._csr.csr_matrix"),
    ("py:class", "scipy.sparse._coo.coo_matrix"),
    ("py:class", "zarr.abc.store.Store"),
    ("py:class", "zarr.core.group.Group"),
    ("py:class", "zarr.core.array.Array"),
    ("py:class", "pathlib.Path"),
    ("py:class", "os.PathLike"),
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Iterable"),
    ("py:class", "collections.abc.Iterator"),
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "collections.abc.Generator"),
    ("py:class", "scarf.matrix.ChunkedArray"),
    ("py:class", "scarf.storage.partition.IndexBlock"),
    ("py:class", "scarf.neighbors.stream.AnnStream"),
    ("py:class", "scarf.merge.DummyAssay"),
    ("py:class", "scarf.readers.CrReader"),
    ("py:class", "scarf.readers.h5ad._H5adAssayFeatures"),
    ("py:class", "scarf.plotting._figure.LegendSpec"),
    ("py:class", "scarf.metadata.MetaDataRowBlock"),
    ("py:class", "scarf.datastore.mapping_datastore.MappingDatastore"),
    ("py:obj", "numpy.typing.DTypeLike"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.Literal"),
    ("py:data", "typing.Optional"),
    ("py:data", "typing.Union"),
    ("py:data", "Ellipsis"),
]
