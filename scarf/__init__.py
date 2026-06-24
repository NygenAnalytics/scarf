"""
====================================
Scarf - Single-cell atlases reformed
====================================

Scarf is a Python package that performs memory-efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with up to 4 million cells)
- Perform analysis of scATAC-Seq data (datasets with up to 700K cells with 1 million peaks tested) under 10 GB RAM
- Make parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art method TopACeDo
- Perform quick and accurate projections of cells from one dataset onto another or integrate multiple datasets.

Exports:
--------

- Modules
    - datastore: Contains the primary interface to interact with data (i.e. DataStore) and its superclasses.
    - readers: A collection of classes for reading in different data formats.
    - writers: Methods and classes for writing data to disk.
    - meld_assay:
    - utils: Utility methods.
    - downloader: Used to download datasets included in Scarf.

GitHub: https://github.com/parashardhapola/scarf

Documentation: https://scarf.readthedocs.io/en/latest/index.html

Pre-print: https://www.biorxiv.org/content/10.1101/2021.05.02.441899v1

PyPI: https://pypi.org/project/scarf/
"""

import warnings

from importlib.metadata import version

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    __version__ = version("scarf")
except ImportError:
    print("Scarf is not installed", flush=True)

from .datastore.datastore import DataStore
from .readers import *
from .writers import *
from .meld_assay import *
from .utils import *
from .downloader import *
from .merge import *