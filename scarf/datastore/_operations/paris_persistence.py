import hashlib
import json
from math import prod
from typing import Literal, cast
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import zarr

from ...clustering._paris_core import ParisHierarchy
from ...clustering.paris_multiscale import (
    ParisClusterDiagnostic,
    ParisClusteringResult,
    PlateauForest,
    labels_from_selected_nodes,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.budget import ResourceBudget, get_resource_budget
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
_ADAPTIVE_RESULT_ARRAYS = (
    "selected_nodes",
    "parent_events",
    "components",
    "sizes",
    "resolution_lower",
    "resolution_upper",
    "persistence",
    "margins",
    "forced",
    "blocking_child_counts",
    "folded_cell_counts",
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
    n_threads: int = 1,
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
    contraction_thread_tables = max(1, n_threads) * n_cells * 8
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
    n_cells = cast(int, generation.attrs["n_leaves"])
    plateau = as_zarr_group(generation["plateau"], name=f"{location}/plateau")
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
        n_threads=budget.workers,
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


def write_hierarchy_generation(
    root: zarr.Group,
    graph_loc: str,
    hierarchy: ParisHierarchy,
    plateau_forest: PlateauForest,
) -> tuple[str, str]:
    """Write an immutable hierarchy generation and mark it complete."""
    generation_id = uuid4().hex
    location = generation_location(graph_loc, generation_id)
    generation = root.create_group(location, overwrite=False)
    generation.attrs.update(
        {
            "complete": False,
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
    generation.attrs["complete"] = True
    return generation_id, location


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
    missing = [name for name in _PARIS_HIERARCHY_ARRAYS if name not in generation]
    if missing or "plateau" not in generation:
        raise ValueError(
            f"Paris hierarchy generation {generation_id!r} is missing required arrays"
        )
    plateau_group = as_zarr_group(generation["plateau"], name=f"{location}/plateau")
    missing_plateau = [
        name for name in _PARIS_PLATEAU_ARRAYS if name not in plateau_group
    ]
    if missing_plateau:
        raise ValueError(
            f"Paris hierarchy generation {generation_id!r} is missing plateau arrays"
        )
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
    *,
    final_label_key: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a label-compatible linkage, falling back only for legacy stores."""
    graph_group = as_zarr_group(root[graph_loc], name=graph_loc)
    latest_value = graph_group.attrs.get(LATEST_PARIS_GENERATION)
    latest_generation = None if latest_value is None else str(latest_value)
    label_generation = (
        None
        if final_label_key is None
        else resolve_adaptive_label_generation(
            root,
            graph_loc,
            final_label_key,
        )
    )
    generation_id = label_generation or latest_generation
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
            if generation_id == latest_generation:
                graph_group.attrs["latest_dendrogram"] = dendrogram_loc
            return dendrogram_loc, generation_id
        preflight_cached_paris_cut(
            root,
            graph_loc,
            generation_id,
            "fixed",
            get_resource_budget(),
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
                update_alias=generation_id == latest_generation,
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


def adaptive_config_digest(generation_id: str, min_cluster_size: int) -> str:
    payload = json.dumps(
        {
            "hierarchy_generation_id": generation_id,
            "min_cluster_size": min_cluster_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _adaptive_label_location(graph_loc: str, label: str) -> str:
    return f"{graph_loc}/adaptive_clustering/{quote(label, safe='')}"


def persist_adaptive_result(
    root: zarr.Group,
    graph_loc: str,
    label: str,
    digest: str,
    result: ParisClusteringResult,
    *,
    generation_id: str,
    final_label_key: str,
    hierarchy_cache_hit: bool,
    cut_seconds: float,
) -> str:
    """Write a complete inactive adaptive result cache."""
    if result.mode != "auto" or result.min_cluster_size is None:
        raise ValueError("Only adaptive Paris results can be persisted here")
    location = f"{_adaptive_label_location(graph_loc, label)}/{digest}"
    config = root.create_group(location, overwrite=True)
    config.attrs.update(
        {
            "complete": False,
            "hierarchy_generation_id": generation_id,
            "min_cluster_size": result.min_cluster_size,
            "final_label_key": final_label_key,
            "hierarchy_cache_hit": hierarchy_cache_hit,
            "cut_seconds": cut_seconds,
            "metadata_write_seconds": np.nan,
        }
    )
    diagnostics = result.diagnostics
    _write_array(
        config,
        "selected_nodes",
        np.asarray([item.selected_node for item in diagnostics], dtype=np.int64),
    )
    _write_array(
        config,
        "parent_events",
        np.asarray([item.parent_event for item in diagnostics], dtype=np.int64),
    )
    _write_array(
        config,
        "components",
        np.asarray([item.component for item in diagnostics], dtype=np.int32),
    )
    _write_array(
        config,
        "sizes",
        np.asarray([item.size for item in diagnostics], dtype=np.int64),
    )
    for name, values in (
        (
            "resolution_lower",
            [item.resolution_lower for item in diagnostics],
        ),
        (
            "resolution_upper",
            [item.resolution_upper for item in diagnostics],
        ),
        ("persistence", [item.persistence for item in diagnostics]),
        ("margins", [item.decision_margin for item in diagnostics]),
    ):
        _write_array(
            config,
            name,
            np.asarray(
                [np.nan if value is None else value for value in values],
                dtype=np.float64,
            ),
        )
    _write_array(
        config,
        "forced",
        np.asarray([item.forced for item in diagnostics], dtype=bool),
    )
    _write_array(
        config,
        "blocking_child_counts",
        np.asarray(
            [item.blocking_child_count for item in diagnostics],
            dtype=np.int32,
        ),
    )
    _write_array(
        config,
        "folded_cell_counts",
        np.asarray(
            [item.folded_cell_count for item in diagnostics],
            dtype=np.int64,
        ),
    )
    config.attrs["complete"] = True
    return location


def _optional_float(value: np.floating) -> float | None:
    return None if np.isnan(value) else float(value)


def load_adaptive_result(
    root: zarr.Group,
    graph_loc: str,
    label: str,
    digest: str,
    hierarchy: ParisHierarchy,
) -> ParisClusteringResult | None:
    """Load adaptive diagnostics and regenerate their O(N) label vector."""
    location = f"{_adaptive_label_location(graph_loc, label)}/{digest}"
    if location not in root:
        return None
    config = as_zarr_group(root[location], name=location)
    if config.attrs.get("complete") is not True:
        return None
    if any(name not in config for name in _ADAPTIVE_RESULT_ARRAYS):
        return None

    selected_nodes = _read_array(config, "selected_nodes")
    labels = labels_from_selected_nodes(hierarchy, selected_nodes)
    parent_events = _read_array(config, "parent_events")
    components = _read_array(config, "components")
    sizes = _read_array(config, "sizes")
    resolution_lower = _read_array(config, "resolution_lower")
    resolution_upper = _read_array(config, "resolution_upper")
    persistence = _read_array(config, "persistence")
    margins = _read_array(config, "margins")
    forced = _read_array(config, "forced")
    blocking_counts = _read_array(config, "blocking_child_counts")
    folded_counts = _read_array(config, "folded_cell_counts")
    n_clusters = selected_nodes.size
    arrays = (
        parent_events,
        components,
        sizes,
        resolution_lower,
        resolution_upper,
        persistence,
        margins,
        forced,
        blocking_counts,
        folded_counts,
    )
    if any(values.shape != (n_clusters,) for values in arrays):
        raise ValueError(f"Adaptive Paris diagnostics are misaligned at {location!r}")

    diagnostics = tuple(
        ParisClusterDiagnostic(
            label=index + 1,
            selected_node=int(selected_nodes[index]),
            parent_event=int(parent_events[index]),
            component=int(components[index]),
            size=int(sizes[index]),
            resolution_lower=_optional_float(resolution_lower[index]),
            resolution_upper=_optional_float(resolution_upper[index]),
            persistence=_optional_float(persistence[index]),
            forced=bool(forced[index]),
            blocking_child_count=int(blocking_counts[index]),
            folded_cell_count=int(folded_counts[index]),
            decision_margin=_optional_float(margins[index]),
        )
        for index in range(n_clusters)
    )
    return ParisClusteringResult(
        labels=labels,
        mode="auto",
        n_clusters=n_clusters,
        diagnostics=diagnostics,
        min_cluster_size=cast(int, config.attrs["min_cluster_size"]),
        label_key=str(config.attrs["final_label_key"]),
        hierarchy_generation_id=str(config.attrs["hierarchy_generation_id"]),
    )


def activate_adaptive_result(
    root: zarr.Group,
    graph_loc: str,
    label: str,
    digest: str,
    *,
    metadata_write_seconds: float,
) -> None:
    label_location = _adaptive_label_location(graph_loc, label)
    config_location = f"{label_location}/{digest}"
    config = as_zarr_group(root[config_location], name=config_location)
    if config.attrs.get("complete") is not True:
        raise ValueError("Cannot activate an incomplete adaptive Paris result")
    config.attrs["metadata_write_seconds"] = metadata_write_seconds
    label_group = as_zarr_group(root[label_location], name=label_location)
    label_group.attrs["active_digest"] = digest


def clear_active_adaptive_result(
    root: zarr.Group,
    graph_loc: str,
    label: str,
) -> None:
    label_location = _adaptive_label_location(graph_loc, label)
    if label_location not in root:
        return
    label_group = as_zarr_group(root[label_location], name=label_location)
    if "active_digest" in label_group.attrs:
        del label_group.attrs["active_digest"]


def resolve_adaptive_label_generation(
    root: zarr.Group,
    graph_loc: str,
    final_label_key: str,
) -> str | None:
    adaptive_location = f"{graph_loc}/adaptive_clustering"
    if adaptive_location not in root:
        return None
    adaptive_group = as_zarr_group(root[adaptive_location], name=adaptive_location)
    for label in adaptive_group.group_keys():
        label_location = f"{adaptive_location}/{label}"
        label_group = as_zarr_group(root[label_location], name=label_location)
        active_digest = label_group.attrs.get("active_digest")
        if active_digest is None:
            continue
        config_location = f"{label_location}/{active_digest}"
        if config_location not in root:
            continue
        config = as_zarr_group(root[config_location], name=config_location)
        if (
            config.attrs.get("complete") is True
            and config.attrs.get("final_label_key") == final_label_key
        ):
            return str(config.attrs["hierarchy_generation_id"])
    return None


def _reusable_adaptive_generation(
    config: zarr.Group,
    digest: str,
    available_generations: set[str],
) -> str | None:
    if config.attrs.get("complete") is not True:
        return None
    if any(name not in config for name in _ADAPTIVE_RESULT_ARRAYS):
        return None
    generation_value = config.attrs.get("hierarchy_generation_id")
    min_cluster_size = config.attrs.get("min_cluster_size")
    final_label_key = config.attrs.get("final_label_key")
    if (
        generation_value is None
        or isinstance(min_cluster_size, (bool, np.bool_))
        or not isinstance(min_cluster_size, (int, np.integer))
        or not isinstance(final_label_key, str)
    ):
        return None
    generation_id = str(generation_value)
    if generation_id not in available_generations:
        return None
    if digest != adaptive_config_digest(generation_id, int(min_cluster_size)):
        return None
    return generation_id


def garbage_collect_hierarchy_generations(
    root: zarr.Group,
    graph_loc: str,
) -> None:
    graph_group = as_zarr_group(root[graph_loc], name=graph_loc)
    latest = graph_group.attrs.get(LATEST_PARIS_GENERATION)
    hierarchy_location = f"{graph_loc}/{PARIS_HIERARCHY_ROOT}"
    if hierarchy_location in root:
        hierarchy_group = as_zarr_group(
            root[hierarchy_location], name=hierarchy_location
        )
        available_generations = set(hierarchy_group.group_keys())
    else:
        hierarchy_group = None
        available_generations = set()
    retained = (
        {str(latest)}
        if latest is not None and str(latest) in available_generations
        else set()
    )
    adaptive_location = f"{graph_loc}/adaptive_clustering"
    if adaptive_location in root:
        adaptive_group = as_zarr_group(root[adaptive_location], name=adaptive_location)
        for label in adaptive_group.group_keys():
            label_location = f"{adaptive_location}/{label}"
            label_group = as_zarr_group(root[label_location], name=label_location)
            active_digest = label_group.attrs.get("active_digest")
            if active_digest is None:
                continue
            config_location = f"{label_location}/{active_digest}"
            if config_location in root:
                config = as_zarr_group(root[config_location], name=config_location)
                generation_id = _reusable_adaptive_generation(
                    config,
                    str(active_digest),
                    available_generations,
                )
                if generation_id is not None:
                    retained.add(generation_id)

    if hierarchy_group is not None:
        for generation_id in tuple(hierarchy_group.group_keys()):
            if generation_id not in retained:
                del root[f"{hierarchy_location}/{generation_id}"]

    if adaptive_location not in root:
        return
    adaptive_group = as_zarr_group(root[adaptive_location], name=adaptive_location)
    for label in tuple(adaptive_group.group_keys()):
        label_location = f"{adaptive_location}/{label}"
        label_group = as_zarr_group(root[label_location], name=label_location)
        active_value = label_group.attrs.get("active_digest")
        active_digest = None if active_value is None else str(active_value)
        for digest in tuple(label_group.group_keys()):
            config_location = f"{label_location}/{digest}"
            config = as_zarr_group(root[config_location], name=config_location)
            if _reusable_adaptive_generation(config, digest, retained) is not None:
                continue
            del root[config_location]
            if active_digest == digest and "active_digest" in label_group.attrs:
                del label_group.attrs["active_digest"]
        if not tuple(label_group.group_keys()):
            del root[label_location]
    if not tuple(adaptive_group.group_keys()):
        del root[adaptive_location]
