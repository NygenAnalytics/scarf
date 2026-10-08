from typing import TYPE_CHECKING

from .._facade import lazy_facade as _lazy_facade

if TYPE_CHECKING:
    from .harmony import (
        Harmony as Harmony,
        HarmonyResult as HarmonyResult,
        fit_harmony as fit_harmony,
    )
    from .initialization import initial_embedding as initial_embedding
    from .imported import (
        validate_imported_embedding_artifact as validate_imported_embedding_artifact,
        write_imported_coordinates as write_imported_coordinates,
        write_imported_embedding as write_imported_embedding,
    )
    from .sgtsne import run_sgtsne as run_sgtsne
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
    "fit_harmony",
    "fit_transform",
    "fuzzy_simplicial_set",
    "initial_embedding",
    "run_sgtsne",
    "simplicial_set_embedding",
    "validate_imported_embedding_artifact",
    "write_imported_coordinates",
    "write_imported_embedding",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "Harmony": ".harmony",
        "HarmonyResult": ".harmony",
        "calc_dens_map_params": ".umap",
        "fit_harmony": ".harmony",
        "fit_transform": ".umap",
        "fuzzy_simplicial_set": ".umap",
        "initial_embedding": ".initialization",
        "validate_imported_embedding_artifact": ".imported",
        "write_imported_coordinates": ".imported",
        "write_imported_embedding": ".imported",
        "run_sgtsne": ".sgtsne",
        "simplicial_set_embedding": ".umap",
    },
)
