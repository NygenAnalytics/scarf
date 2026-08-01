from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harmony import (
        Harmony as Harmony,
        HarmonyResult as HarmonyResult,
        fit_harmony as fit_harmony,
        run_harmony as run_harmony,
    )
    from .initialization import initial_embedding as initial_embedding
    from .imported import (
        validate_imported_embedding_artifact as validate_imported_embedding_artifact,
        write_imported_coordinates as write_imported_coordinates,
        write_imported_embedding as write_imported_embedding,
    )
    from .sgtsne import (
        export_knn_to_mtx as export_knn_to_mtx,
        run_sgtsne as run_sgtsne,
    )
    from .umap import (
        calc_dens_map_params as calc_dens_map_params,
        fit_transform as fit_transform,
        fuzzy_simplicial_set as fuzzy_simplicial_set,
        simplicial_set_embedding as simplicial_set_embedding,
    )

__all__ = [
    "Harmony",
    "HarmonyResult",
    "calc_dens_map_params",
    "export_knn_to_mtx",
    "fit_harmony",
    "fit_transform",
    "fuzzy_simplicial_set",
    "initial_embedding",
    "run_harmony",
    "run_sgtsne",
    "simplicial_set_embedding",
    "validate_imported_embedding_artifact",
    "write_imported_coordinates",
    "write_imported_embedding",
]

_LAZY_EXPORTS = {
    "Harmony": (".harmony", "Harmony"),
    "HarmonyResult": (".harmony", "HarmonyResult"),
    "calc_dens_map_params": (".umap", "calc_dens_map_params"),
    "export_knn_to_mtx": (".sgtsne", "export_knn_to_mtx"),
    "fit_harmony": (".harmony", "fit_harmony"),
    "fit_transform": (".umap", "fit_transform"),
    "fuzzy_simplicial_set": (".umap", "fuzzy_simplicial_set"),
    "initial_embedding": (".initialization", "initial_embedding"),
    "validate_imported_embedding_artifact": (
        ".imported",
        "validate_imported_embedding_artifact",
    ),
    "write_imported_coordinates": (".imported", "write_imported_coordinates"),
    "write_imported_embedding": (".imported", "write_imported_embedding"),
    "run_harmony": (".harmony", "run_harmony"),
    "run_sgtsne": (".sgtsne", "run_sgtsne"),
    "simplicial_set_embedding": (".umap", "simplicial_set_embedding"),
}

for _export_name in _LAZY_EXPORTS:
    globals().pop(_export_name, None)
del _export_name


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(_import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_EXPORTS))
