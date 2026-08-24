from math import prod
from typing import Literal, cast

import numpy as np
import zarr

from ...clustering._paris_core import ParisHierarchy
from ...clustering.paris_multiscale import PlateauForest
from ...storage.arrays import create_zarr_dataset
from ...storage.budget import ResourceBudget
from ...storage.types import as_zarr_array, as_zarr_group

PARIS_HIERARCHY_ROOT = "paris_hierarchy"
LATEST_PARIS_GENERATION = "latest_paris_generation"
_PARIS_HIERARCHY_ARRAYS = (
    "children",
    "heights",
    "sizes",
    "component_roots",
    "synthetic_joins",
)
_PARIS_PLATEAU_ARRAYS = (
    "representatives",
    "heights",
    "sizes",
    "parent_events",
    "child_offsets",
    "child_refs",
    "min_leaves",
    "component_roots",
)
_MEMORY_HEADROOM = 1.35
_CACHED_FIXED_TRANSIENT_BYTES_PER_CELL = 128
_CACHED_ADAPTIVE_TRANSIENT_BYTES_PER_CELL = 96


def generation_location(graph_loc: str, generation_id: str) -> str:
    return f"{graph_loc}/{PARIS_HIERARCHY_ROOT}/{generation_id}"


def _paris_memory_components(
    n_cells: int,
    edge_count: int,
    edge_itemsize: int,
    weight_itemsize: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    largest_index = max(2 * n_cells - 1, 2 * edge_count)
    index_bytes = 4 if largest_index <= np.iinfo(np.int32).max else 8
    pointer_bytes = (n_cells + 1) * index_bytes
    stored_edges = edge_count * (2 * edge_itemsize + weight_itemsize)
    directed_csr = edge_count * (8 + index_bytes) + pointer_bytes
    canonical_edges = 2 * edge_count
    canonical_csr = canonical_edges * (8 + index_bytes) + pointer_bytes
    load_peak = stored_edges + directed_csr
    symmetrize_peak = 2 * directed_csr + 2 * canonical_csr
    hierarchy_bytes = (n_cells - 1) * (2 * index_bytes + 8 + 4 + 1)
    plateau_bytes = n_cells * (6 * index_bytes + 8 + 4)
    modularity_guard_bytes = n_cells * 128
    return (
        index_bytes,
        load_peak,
        symmetrize_peak,
        directed_csr,
        canonical_csr,
        hierarchy_bytes,
        plateau_bytes,
        modularity_guard_bytes,
    )


def estimate_paris_peak_bytes(
    n_cells: int,
    edge_count: int,
    edge_itemsize: int,
    weight_itemsize: int,
    *,
    nthreads: int = 1,
) -> int:
    """Conservatively estimate peak bytes for canonical Paris fitting."""
    (
        index_bytes,
        load_peak,
        symmetrize_peak,
        directed_csr,
        canonical_csr,
        hierarchy_bytes,
        plateau_bytes,
        modularity_guard_bytes,
    ) = _paris_memory_components(
        n_cells,
        edge_count,
        edge_itemsize,
        weight_itemsize,
    )
    contraction_workspaces = canonical_csr
    contraction_layout = n_cells * (6 * index_bytes)
    contraction_thread_tables = max(1, nthreads) * n_cells * 8
    contraction_peak = (
        directed_csr
        + canonical_csr
        + contraction_layout
        + contraction_thread_tables
        + contraction_workspaces
    )

    fit_node_buffers = n_cells * (8 + 8 + 8 + 8 + 8 + index_bytes)
    fit_peak = contraction_peak + hierarchy_bytes + fit_node_buffers
    cut_peak = (
        directed_csr
        + canonical_csr
        + hierarchy_bytes
        + plateau_bytes
        + modularity_guard_bytes
    )
    return int(_MEMORY_HEADROOM * max(load_peak + fit_peak, symmetrize_peak, cut_peak))


def estimate_paris_adaptive_cut_peak_bytes(
    n_cells: int,
    edge_count: int,
    edge_itemsize: int,
    weight_itemsize: int,
) -> int:
    """Estimate peak bytes for a guarded cut over a cached hierarchy."""
    (
        _index_bytes,
        load_peak,
        symmetrize_peak,
        directed_csr,
        canonical_csr,
        hierarchy_bytes,
        plateau_bytes,
        modularity_guard_bytes,
    ) = _paris_memory_components(
        n_cells,
        edge_count,
        edge_itemsize,
        weight_itemsize,
    )
    cut_peak = (
        directed_csr
        + canonical_csr
        + hierarchy_bytes
        + plateau_bytes
        + modularity_guard_bytes
    )
    return int(_MEMORY_HEADROOM * max(load_peak, symmetrize_peak, cut_peak))


def _zarr_array_nbytes(group: zarr.Group, name: str) -> int:
    values = as_zarr_array(group[name], name=name)
    return prod(values.shape) * np.dtype(values.dtype).itemsize


def estimate_cached_paris_peak_bytes(
    root: zarr.Group,
    graph_loc: str,
    generation_id: str,
    cut_mode: Literal["adaptive", "fixed"],
) -> int:
    """Estimate hierarchy loading plus fixed or cached-adaptive cut buffers."""
    location = generation_location(graph_loc, generation_id)
    generation = as_zarr_group(root[location], name=location)
    return estimate_hierarchy_group_peak_bytes(generation, cut_mode)


def estimate_hierarchy_group_peak_bytes(
    generation: zarr.Group,
    cut_mode: Literal["adaptive", "fixed"],
) -> int:
    """Estimate loading and cutting an independent hierarchy artifact."""
    n_cells = cast(int, generation.attrs["n_leaves"])
    plateau = as_zarr_group(generation["plateau"], name="plateau")
    hierarchy_bytes = sum(
        _zarr_array_nbytes(generation, name)
        for name in (
            "children",
            "heights",
            "sizes",
            "component_roots",
            "synthetic_joins",
        )
    )
    plateau_bytes = sum(
        _zarr_array_nbytes(plateau, name)
        for name in (
            "representatives",
            "heights",
            "sizes",
            "parent_events",
            "child_offsets",
            "child_refs",
            "min_leaves",
            "component_roots",
        )
    )
    if cut_mode == "fixed":
        transient_bytes = n_cells * _CACHED_FIXED_TRANSIENT_BYTES_PER_CELL
    elif cut_mode == "adaptive":
        transient_bytes = n_cells * _CACHED_ADAPTIVE_TRANSIENT_BYTES_PER_CELL
    else:
        raise ValueError("cut_mode must be 'adaptive' or 'fixed'")
    return int(_MEMORY_HEADROOM * (hierarchy_bytes + plateau_bytes + transient_bytes))


def _raise_if_over_budget(
    estimate: int,
    budget: ResourceBudget,
    operation: str,
) -> None:
    if estimate <= budget.memoryBytes:
        return
    required_gib = estimate / 1024**3
    budget_gib = budget.memoryBytes / 1024**3
    raise MemoryError(
        f"{operation} is estimated to require "
        f"{required_gib:.2f} GiB including headroom, but the active "
        f"resource budget is {budget_gib:.2f} GiB"
    )


def preflight_paris_fit(
    graph_group: zarr.Group,
    n_cells: int,
    budget: ResourceBudget,
) -> int:
    """Fail before loading graph arrays when the Paris estimate exceeds budget."""
    edges = as_zarr_array(graph_group["edges"], name="edges")
    weights = as_zarr_array(graph_group["weights"], name="weights")
    edge_count = int(edges.shape[0])
    estimate = estimate_paris_peak_bytes(
        n_cells,
        edge_count,
        np.dtype(edges.dtype).itemsize,
        np.dtype(weights.dtype).itemsize,
        nthreads=budget.workers,
    )
    _raise_if_over_budget(estimate, budget, "Paris hierarchy fit")
    return estimate


def preflight_cached_paris_cut(
    root: zarr.Group,
    graph_loc: str,
    generation_id: str,
    cut_mode: Literal["adaptive", "fixed"],
    budget: ResourceBudget,
) -> int:
    """Fail before loading a cached hierarchy and cut workspaces."""
    estimate = estimate_cached_paris_peak_bytes(
        root,
        graph_loc,
        generation_id,
        cut_mode,
    )
    _raise_if_over_budget(estimate, budget, f"Cached Paris {cut_mode} cut")
    return estimate


def preflight_hierarchy_artifact_cut(
    hierarchy_group: zarr.Group,
    cut_mode: Literal["adaptive", "fixed"],
    budget: ResourceBudget,
) -> int:
    estimate = estimate_hierarchy_group_peak_bytes(
        hierarchy_group,
        cut_mode,
    )
    _raise_if_over_budget(
        estimate,
        budget,
        f"Cached Paris {cut_mode} cut",
    )
    return estimate


def preflight_paris_adaptive_cut(
    graph_group: zarr.Group,
    n_cells: int,
    budget: ResourceBudget,
) -> int:
    """Fail before loading graph arrays for a cached-hierarchy adaptive cut."""
    edges = as_zarr_array(graph_group["edges"], name="edges")
    weights = as_zarr_array(graph_group["weights"], name="weights")
    estimate = estimate_paris_adaptive_cut_peak_bytes(
        n_cells,
        int(edges.shape[0]),
        np.dtype(edges.dtype).itemsize,
        np.dtype(weights.dtype).itemsize,
    )
    _raise_if_over_budget(estimate, budget, "Paris adaptive cut")
    return estimate


def _array_chunks(values: np.ndarray) -> tuple[int, ...]:
    first = max(1, min(100_000, values.shape[0]))
    return (first, *values.shape[1:])


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    target = create_zarr_dataset(
        group,
        name,
        _array_chunks(values),
        values.dtype,
        values.shape,
    )
    target[:] = values


def write_hierarchy_group(
    generation: zarr.Group,
    hierarchy: ParisHierarchy,
    plateau_forest: PlateauForest,
) -> None:
    generation.attrs.update(
        {
            "n_leaves": hierarchy.n_leaves,
            "total_weight": hierarchy.total_weight,
        }
    )
    _write_array(generation, "children", hierarchy.children)
    _write_array(generation, "heights", hierarchy.heights)
    _write_array(generation, "sizes", hierarchy.sizes)
    _write_array(generation, "component_roots", hierarchy.component_roots)
    _write_array(generation, "synthetic_joins", hierarchy.synthetic_joins)

    plateau = generation.create_group("plateau", overwrite=True)
    _write_array(plateau, "representatives", plateau_forest.representatives)
    _write_array(plateau, "heights", plateau_forest.heights)
    _write_array(plateau, "sizes", plateau_forest.sizes)
    _write_array(plateau, "parent_events", plateau_forest.parent_events)
    _write_array(plateau, "child_offsets", plateau_forest.child_offsets)
    _write_array(plateau, "child_refs", plateau_forest.child_refs)
    _write_array(plateau, "min_leaves", plateau_forest.min_leaves)
    _write_array(plateau, "component_roots", plateau_forest.component_roots)

    if hierarchy.diagnostics is not None:
        generation.attrs.update(
            {
                "preprocessing_seconds": hierarchy.diagnostics.preprocessing_seconds,
                "component_seconds": hierarchy.diagnostics.component_seconds,
                "fit_seconds": hierarchy.diagnostics.fit_seconds,
                "reciprocal_rounds": len(hierarchy.diagnostics.rounds),
            }
        )


def _read_array(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])


def load_hierarchy_generation(
    root: zarr.Group,
    graph_loc: str,
    generation_id: str,
) -> tuple[ParisHierarchy, PlateauForest]:
    """Load and validate a completed hierarchy generation."""
    location = generation_location(graph_loc, generation_id)
    generation = as_zarr_group(root[location], name=location)
    if generation.attrs.get("complete") is not True:
        raise ValueError(f"Paris hierarchy generation {generation_id!r} is incomplete")
    return load_hierarchy_group(generation, location)


def load_hierarchy_group(
    generation: zarr.Group,
    label: str,
) -> tuple[ParisHierarchy, PlateauForest]:
    missing = [name for name in _PARIS_HIERARCHY_ARRAYS if name not in generation]
    if missing or "plateau" not in generation:
        raise ValueError(f"Paris hierarchy {label!r} is missing required arrays")
    plateau_group = as_zarr_group(generation["plateau"], name=f"{label}/plateau")
    missing_plateau = [
        name for name in _PARIS_PLATEAU_ARRAYS if name not in plateau_group
    ]
    if missing_plateau:
        raise ValueError(f"Paris hierarchy {label!r} is missing plateau arrays")
    n_leaves = cast(int, generation.attrs["n_leaves"])
    hierarchy = ParisHierarchy(
        children=_read_array(generation, "children"),
        heights=_read_array(generation, "heights"),
        sizes=_read_array(generation, "sizes"),
        component_roots=_read_array(generation, "component_roots"),
        synthetic_joins=_read_array(generation, "synthetic_joins").astype(
            bool,
            copy=False,
        ),
        n_leaves=n_leaves,
        total_weight=cast(float, generation.attrs["total_weight"]),
    )
    plateau_forest = PlateauForest(
        representatives=_read_array(plateau_group, "representatives"),
        heights=_read_array(plateau_group, "heights"),
        sizes=_read_array(plateau_group, "sizes"),
        parent_events=_read_array(plateau_group, "parent_events"),
        child_offsets=_read_array(plateau_group, "child_offsets"),
        child_refs=_read_array(plateau_group, "child_refs"),
        min_leaves=_read_array(plateau_group, "min_leaves"),
        component_roots=_read_array(plateau_group, "component_roots"),
        n_leaves=n_leaves,
    )
    return hierarchy, plateau_forest


def ensure_compatibility_dendrogram(
    root: zarr.Group,
    graph_loc: str,
    generation_id: str,
    hierarchy: ParisHierarchy,
    *,
    update_alias: bool = True,
) -> str:
    """Materialize the compatibility linkage and update its alias."""
    from ...clustering.paris import hierarchy_to_dendrogram

    location = generation_location(graph_loc, generation_id)
    dendrogram_loc = f"{location}/dendrogram"
    generation = as_zarr_group(root[location], name=location)
    if (
        dendrogram_loc not in root
        or generation.attrs.get("dendrogram_complete") is not True
    ):
        dendrogram = hierarchy_to_dendrogram(hierarchy, compatibility=True)
        target = create_zarr_dataset(
            generation,
            "dendrogram",
            (min(5000, dendrogram.shape[0]), 4),
            "f8",
            dendrogram.shape,
        )
        target[:] = dendrogram
        generation.attrs["dendrogram_complete"] = True
    if update_alias:
        graph_group = as_zarr_group(root[graph_loc], name=graph_loc)
        graph_group.attrs["latest_dendrogram"] = dendrogram_loc
    return dendrogram_loc


def resolve_compatibility_dendrogram(
    root: zarr.Group,
    graph_loc: str,
    budget: ResourceBudget,
) -> tuple[str, str | None]:
    """Resolve a linkage, falling back only for legacy stores."""
    graph_group = as_zarr_group(root[graph_loc], name=graph_loc)
    latest_value = graph_group.attrs.get(LATEST_PARIS_GENERATION)
    generation_id = None if latest_value is None else str(latest_value)
    # A stale generation pointer (for example an old paris_hierarchy/v2 layout)
    # must fall through to the legacy alias or dendrogram rather than crash.
    if (
        generation_id is not None
        and generation_location(graph_loc, generation_id) in root
    ):
        location = generation_location(graph_loc, generation_id)
        dendrogram_loc = f"{location}/dendrogram"
        generation = as_zarr_group(root[location], name=location)
        if (
            dendrogram_loc in root
            and generation.attrs.get("dendrogram_complete") is True
        ):
            graph_group.attrs["latest_dendrogram"] = dendrogram_loc
            return dendrogram_loc, generation_id
        preflight_cached_paris_cut(
            root,
            graph_loc,
            generation_id,
            "fixed",
            budget,
        )
        hierarchy, _plateau_forest = load_hierarchy_generation(
            root,
            graph_loc,
            generation_id,
        )
        return (
            ensure_compatibility_dendrogram(
                root,
                graph_loc,
                generation_id,
                hierarchy,
                update_alias=True,
            ),
            generation_id,
        )

    alias = graph_group.attrs.get("latest_dendrogram")
    if alias is not None and str(alias) in root:
        return str(alias), None
    legacy_location = f"{graph_loc}/dendrogram"
    if legacy_location in root:
        return legacy_location, None
    raise KeyError(
        "No Paris hierarchy is available for this graph. "
        "Run run_paris_clustering first."
    )
