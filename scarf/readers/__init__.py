"""A collection of classes for reading in different data formats.

- Classes:
    - CrH5Reader: A class to read in CellRanger (Cr) data, in the form of an H5 file.
    - CrDirReader: A class to read in CellRanger (Cr) data, in the form of a directory.
    - CrReader: A class to read in CellRanger (Cr) data.
    - H5adReader: A class to read in data in the form of a H5ad file (h5 file with AnnData information).
    - LoomReader: A class to read in data in the form of a Loom file.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._text import get_file_handle as get_file_handle
from ._text import read_file as read_file

if TYPE_CHECKING:
    from .cellranger import CrDirReader, CrH5Reader, CrReader
    from .csv import CSVReader
    from .h5ad import H5adReader
    from .loom import LoomReader

__all__ = [
    "CrH5Reader",
    "CrDirReader",
    "CrReader",
    "H5adReader",
    "LoomReader",
    "CSVReader",
]

_LAZY_EXPORTS = {
    "CrDirReader": "cellranger",
    "CrH5Reader": "cellranger",
    "CrReader": "cellranger",
    "CSVReader": "csv",
    "H5adReader": "h5ad",
    "LoomReader": "loom",
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name


def _normalize_class_metadata(reader_class: type[Any]) -> None:
    reader_class.__module__ = __name__
    for descriptor in reader_class.__dict__.values():
        if isinstance(descriptor, staticmethod):
            descriptor.__func__.__module__ = __name__
        elif callable(descriptor) and hasattr(descriptor, "__module__"):
            descriptor.__module__ = __name__


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    for export_name, export_module in _LAZY_EXPORTS.items():
        if export_module != module_name:
            continue
        reader_class = getattr(module, export_name)
        _normalize_class_metadata(reader_class)
        globals()[export_name] = reader_class
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


for _function in (get_file_handle, read_file):
    _function.__module__ = __name__
del _function
