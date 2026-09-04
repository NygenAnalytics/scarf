"""Unit tests for Paris persistence helpers."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.paris_persistence import (
    estimate_hierarchy_group_peak_bytes,
    estimate_paris_peak_bytes,
    load_hierarchy_group,
)
from scarf.storage.budget import ResourceBudget


def test_estimate_hierarchy_group_peak_bytes_rejects_unknown_cut_mode():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    generation = root.create_group("gen")
    generation.attrs["n_leaves"] = 2
    for name, values in (
        ("children", np.array([[0, 1]], dtype=np.int32)),
        ("heights", np.array([1.0])),
        ("sizes", np.array([2], dtype=np.int32)),
        ("component_roots", np.array([2], dtype=np.int32)),
        ("synthetic_joins", np.array([False])),
    ):
        generation.create_array(name, data=values)
    plateau = generation.create_group("plateau")
    for name, values in (
        ("representatives", np.array([2], dtype=np.int32)),
        ("heights", np.array([1.0])),
        ("sizes", np.array([2], dtype=np.int32)),
        ("parent_events", np.array([-1], dtype=np.int32)),
        ("child_offsets", np.array([0, 2], dtype=np.int32)),
        ("child_refs", np.array([-1, -2], dtype=np.int32)),
        ("min_leaves", np.array([2], dtype=np.int32)),
        ("component_roots", np.array([0], dtype=np.int32)),
    ):
        plateau.create_array(name, data=values)

    fixed = estimate_hierarchy_group_peak_bytes(generation, "fixed")
    adaptive = estimate_hierarchy_group_peak_bytes(generation, "adaptive")
    # Fixed cuts allocate more transient scratch than adaptive cuts.
    assert fixed > adaptive > 0
    assert estimate_hierarchy_group_peak_bytes(generation, "fixed") == fixed

    generation.attrs["n_leaves"] = 200
    scaled = estimate_hierarchy_group_peak_bytes(generation, "fixed")
    assert scaled > fixed
    generation.attrs["n_leaves"] = 2

    with pytest.raises(ValueError, match="cut_mode must be"):
        estimate_hierarchy_group_peak_bytes(generation, "mystery")  # type: ignore[arg-type]


def test_load_hierarchy_group_rejects_missing_arrays():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    location = "hierarchy"
    generation = root.create_group(location)
    generation.attrs["n_leaves"] = 2
    generation.attrs["total_weight"] = 1.0

    with pytest.raises(ValueError, match="missing required arrays"):
        load_hierarchy_group(generation, location)

    for name, values in (
        ("children", np.array([[0, 1]], dtype=np.int32)),
        ("heights", np.array([1.0])),
        ("sizes", np.array([2], dtype=np.int32)),
        ("component_roots", np.array([2], dtype=np.int32)),
        ("synthetic_joins", np.array([False])),
    ):
        generation.create_array(name, data=values)
    with pytest.raises(ValueError, match="missing required arrays"):
        load_hierarchy_group(generation, location)

    plateau = generation.create_group("plateau")
    with pytest.raises(ValueError, match="missing plateau arrays"):
        load_hierarchy_group(generation, location)

    for name, values in (
        ("representatives", np.array([2], dtype=np.int32)),
        ("heights", np.array([1.0])),
        ("sizes", np.array([2], dtype=np.int32)),
        ("parent_events", np.array([-1], dtype=np.int32)),
        ("child_offsets", np.array([0, 2], dtype=np.int32)),
        ("child_refs", np.array([-1, -2], dtype=np.int32)),
        ("min_leaves", np.array([2], dtype=np.int32)),
        ("component_roots", np.array([0], dtype=np.int32)),
    ):
        plateau.create_array(name, data=values)

    hierarchy, forest = load_hierarchy_group(generation, location)
    assert hierarchy.n_leaves == 2
    assert forest.n_leaves == 2
    assert hierarchy.children.tolist() == [[0, 1]]


def test_preflight_paris_fit_and_artifact_cut_respect_budget():
    from scarf.datastore._operations.paris_persistence import (
        preflight_hierarchy_artifact_cut,
        preflight_paris_fit,
    )

    root = zarr.open_group(store=MemoryStore(), mode="w")
    graph = root.create_group("RNA/graph")
    graph.create_array("edges", data=np.array([[0, 1], [1, 0]], dtype=np.int32))
    graph.create_array("weights", data=np.array([1.0, 1.0], dtype=np.float64))

    tiny = ResourceBudget(memoryBytes=1, workers=1)
    with pytest.raises(MemoryError, match="Paris hierarchy fit"):
        preflight_paris_fit(graph, n_cells=2, budget=tiny)

    generation = root.create_group("hierarchy")
    generation.attrs["n_leaves"] = 2
    generation.attrs["total_weight"] = 1.0
    for name, values in (
        ("children", np.array([[0, 1]], dtype=np.int32)),
        ("heights", np.array([1.0])),
        ("sizes", np.array([2], dtype=np.int32)),
        ("component_roots", np.array([2], dtype=np.int32)),
        ("synthetic_joins", np.array([False])),
    ):
        generation.create_array(name, data=values)
    plateau = generation.create_group("plateau")
    for name, values in (
        ("representatives", np.array([2], dtype=np.int32)),
        ("heights", np.array([1.0])),
        ("sizes", np.array([2], dtype=np.int32)),
        ("parent_events", np.array([-1], dtype=np.int32)),
        ("child_offsets", np.array([0, 2], dtype=np.int32)),
        ("child_refs", np.array([-1, -2], dtype=np.int32)),
        ("min_leaves", np.array([2], dtype=np.int32)),
        ("component_roots", np.array([0], dtype=np.int32)),
    ):
        plateau.create_array(name, data=values)

    with pytest.raises(MemoryError, match="Cached Paris fixed cut"):
        preflight_hierarchy_artifact_cut(
            generation,
            "fixed",
            tiny,
        )

    ok = ResourceBudget(memoryBytes=1024**3, workers=1)
    fit_bytes = preflight_paris_fit(graph, n_cells=2, budget=ok)
    cut_bytes = preflight_hierarchy_artifact_cut(
        generation,
        "fixed",
        ok,
    )
    assert fit_bytes == estimate_paris_peak_bytes(
        2,
        2,
        np.dtype(np.int32).itemsize,
        np.dtype(np.float64).itemsize,
        nthreads=1,
    )
    assert cut_bytes == estimate_hierarchy_group_peak_bytes(generation, "fixed")
