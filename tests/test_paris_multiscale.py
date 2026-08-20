import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering.paris_multiscale import (
    adaptive_cut,
    collapse_equal_height_plateaus,
)
from scarf.clustering.paris import fit_paris_hierarchy


def _hierarchy(
    children: list[tuple[int, int]],
    heights: list[float],
    *,
    component_roots: list[int] | None = None,
    synthetic_joins: list[bool] | None = None,
) -> ParisHierarchy:
    n_leaves = len(children) + 1
    child_array = np.asarray(children, dtype=np.int32)
    sizes = np.ones(2 * n_leaves - 1, dtype=np.int32)
    merge_sizes = np.empty(n_leaves - 1, dtype=np.int32)
    for merge_index, (left, right) in enumerate(children):
        size = int(sizes[left]) + int(sizes[right])
        merge_sizes[merge_index] = size
        sizes[n_leaves + merge_index] = size
    synthetic = (
        np.isinf(heights)
        if synthetic_joins is None
        else np.asarray(synthetic_joins, dtype=bool)
    )
    roots = (
        np.asarray([2 * n_leaves - 2], dtype=np.int32)
        if component_roots is None
        else np.asarray(component_roots, dtype=np.int32)
    )
    return ParisHierarchy(
        children=child_array,
        heights=np.asarray(heights, dtype=np.float64),
        sizes=merge_sizes,
        component_roots=roots,
        synthetic_joins=np.asarray(synthetic, dtype=bool),
        n_leaves=n_leaves,
        total_weight=1.0,
    )


def _nested_block_graph(seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    block_size = 40
    n_cells = 4 * block_size
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for left in range(n_cells):
        for right in range(left + 1, n_cells):
            left_block = left // block_size
            right_block = right // block_size
            if left_block == right_block:
                probability, weight = 0.28, 1.0
            elif left_block // 2 == right_block // 2:
                probability, weight = 0.12, 0.15
            else:
                probability, weight = 0.04, 0.002
            if rng.random() >= probability:
                continue
            rows.extend((left, right))
            columns.extend((right, left))
            weights.extend((weight, weight))
    return csr_matrix(
        (weights, (rows, columns)),
        shape=(n_cells, n_cells),
    )


def _plateau_variants() -> tuple[ParisHierarchy, ParisHierarchy]:
    first = _hierarchy(
        [(0, 1), (2, 3), (6, 7), (4, 5), (8, 9)],
        [1, 1, 1, 2, 10],
    )
    second = _hierarchy(
        [(0, 2), (1, 6), (3, 7), (4, 5), (8, 9)],
        [1, 1, 1, 2, 10],
    )
    return first, second


def _event_leaf_sets(
    hierarchy: ParisHierarchy,
    representatives: np.ndarray,
) -> list[set[int]]:
    n_leaves = hierarchy.n_leaves
    members = {leaf: {leaf} for leaf in range(n_leaves)}
    for merge_index, children in enumerate(hierarchy.children):
        members[n_leaves + merge_index] = (
            members[int(children[0])] | members[int(children[1])]
        )
    return [members[int(node)] for node in representatives]


def test_equal_height_binary_refinements_have_the_same_cut() -> None:
    first, second = _plateau_variants()
    first_forest = collapse_equal_height_plateaus(first)
    second_forest = collapse_equal_height_plateaus(second)
    first_result = adaptive_cut(first, 2, plateau_forest=first_forest)
    second_result = adaptive_cut(second, 2, plateau_forest=second_forest)

    assert len(first_forest.representatives) == 3
    assert len(second_forest.representatives) == 3
    assert np.array_equal(first_result.labels, [1, 1, 1, 1, 2, 2])
    assert np.array_equal(second_result.labels, first_result.labels)


def test_adaptive_score_is_optimal_over_small_event_antichains() -> None:
    hierarchy, _other = _plateau_variants()
    forest = collapse_equal_height_plateaus(hierarchy)
    result = adaptive_cut(hierarchy, 2, plateau_forest=forest)
    leaf_sets = _event_leaf_sets(hierarchy, forest.representatives)
    best_score = -np.inf

    for mask in itertools.product((False, True), repeat=len(leaf_sets)):
        selected = [index for index, keep in enumerate(mask) if keep]
        if not selected:
            continue
        coverage = np.zeros(hierarchy.n_leaves, dtype=np.int8)
        for event in selected:
            coverage[list(leaf_sets[event])] += 1
        if not np.all(coverage == 1):
            continue
        if any(int(forest.sizes[event]) < 2 for event in selected):
            continue
        score = 0.0
        for event in selected:
            parent = int(forest.parent_events[event])
            if parent >= 0:
                score += int(forest.sizes[event]) * (
                    1.0 / (1.0 + float(forest.heights[event]))
                    - 1.0 / (1.0 + float(forest.heights[parent]))
                )
        best_score = max(best_score, score)

    selected_score = sum(
        0.0 if item.persistence is None else item.persistence
        for item in result.diagnostics
    )
    assert selected_score == pytest.approx(best_score)


def test_extreme_coarse_intervals_do_not_hide_canonical_scale_substructure() -> None:
    hierarchy = _hierarchy(
        [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (12, 13),
        ],
        [1, 1, 1, 1, 10, 10, 10_000],
    )
    result = adaptive_cut(hierarchy, 2)

    assert np.bincount(result.labels)[1:].tolist() == [2, 2, 2, 2]
    assert [item.selected_node for item in result.diagnostics] == [8, 9, 10, 11]
    assert all(item.persistence == pytest.approx(9 / 11) for item in result.diagnostics)


def test_nested_graph_keeps_four_durable_subcommunities() -> None:
    hierarchy = fit_paris_hierarchy(_nested_block_graph(), nthreads=2)
    result = adaptive_cut(hierarchy, 10)

    assert result.n_clusters == 4
    block_labels = result.labels.reshape(4, 40)
    assert all(np.unique(block).size == 1 for block in block_labels)
    assert np.unique(block_labels[:, 0]).size == 4


def test_unequal_depth_tree_keeps_durable_and_splits_transient_branches() -> None:
    hierarchy = _hierarchy(
        [
            (0, 1),
            (2, 3),
            (8, 9),
            (4, 5),
            (6, 7),
            (11, 12),
            (10, 13),
        ],
        [1, 1, 2, 1, 1, 50, 100],
    )
    result = adaptive_cut(hierarchy, 2)

    assert np.bincount(result.labels)[1:].tolist() == [4, 2, 2]
    assert [item.selected_node for item in result.diagnostics] == [10, 11, 12]
    assert all(not item.forced for item in result.diagnostics)


def test_exact_score_ties_keep_the_parent_event() -> None:
    hierarchy = _hierarchy(
        [
            (0, 1),
            (2, 3),
            (8, 9),
            (4, 5),
            (6, 7),
            (11, 12),
            (10, 13),
        ],
        [1 / 3, 1 / 3, 1, 1 / 3, 1 / 3, 1, 3],
    )
    result = adaptive_cut(hierarchy, 2)

    assert np.bincount(result.labels)[1:].tolist() == [4, 4]
    assert [item.selected_node for item in result.diagnostics] == [10, 13]
    assert all(
        item.decision_margin == pytest.approx(0.0) for item in result.diagnostics
    )


def test_resolution_bounds_and_zero_height_events_are_finite_safe() -> None:
    hierarchy = _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [0, 1, 10],
    )
    result = adaptive_cut(hierarchy, 2)
    zero_event = result.diagnostics[0]

    assert zero_event.resolution_lower == 0
    assert zero_event.resolution_upper == 10
    assert zero_event.persistence == pytest.approx(20 / 11)
    assert np.isfinite(
        [
            item.persistence
            for item in result.diagnostics
            if item.persistence is not None
        ]
    ).all()


@pytest.mark.parametrize("height", [np.nan, -1.0])
def test_nan_and_negative_heights_are_rejected(height: float) -> None:
    hierarchy = _hierarchy([(0, 1), (2, 3), (4, 5)], [1, 1, height])
    with pytest.raises(ValueError, match="non-negative"):
        adaptive_cut(hierarchy, 2)


def test_strict_multiway_folding_selects_the_nearest_valid_ancestor() -> None:
    hierarchy = _hierarchy(
        [(0, 1), (2, 3), (6, 7), (4, 5), (8, 9)],
        [0.5, 0.5, 1, 2, 10],
    )
    result = adaptive_cut(hierarchy, 3)

    assert result.n_clusters == 1
    assert np.array_equal(result.labels, np.ones(6, dtype=np.int32))
    diagnostic = result.diagnostics[0]
    assert diagnostic.forced
    assert diagnostic.blocking_child_count == 1
    assert diagnostic.folded_cell_count == 2
    assert diagnostic.persistence is None
    assert diagnostic.decision_margin is None


def test_disconnected_components_and_isolates_are_forced_and_relabelled() -> None:
    graph = csr_matrix(
        np.asarray(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
    )
    result = adaptive_cut(fit_paris_hierarchy(graph), 2)

    assert result.labels.tolist() == [1, 1, 2, 2, 3]
    assert all(item.forced for item in result.diagnostics)
    assert all(item.persistence is None for item in result.diagnostics)
    assert [item.component for item in result.diagnostics] == [0, 1, 2]


def test_labels_are_contiguous_read_only_and_cover_every_leaf() -> None:
    hierarchy, _other = _plateau_variants()
    result = adaptive_cut(hierarchy, 2)

    assert result.labels.dtype == np.int32
    assert set(result.labels) == set(range(1, result.n_clusters + 1))
    assert not result.labels.flags.writeable
    with pytest.raises(ValueError):
        result.labels[0] = 99


def test_split_gate_rejects_misaligned_or_non_finite_values() -> None:
    hierarchy, _other = _plateau_variants()
    forest = collapse_equal_height_plateaus(hierarchy)
    n_events = forest.representatives.size

    with pytest.raises(ValueError, match="one value per plateau event"):
        adaptive_cut(
            hierarchy,
            2,
            plateau_forest=forest,
            split_gate=np.ones(n_events + 1),
        )
    with pytest.raises(TypeError, match="real numbers"):
        adaptive_cut(
            hierarchy,
            2,
            plateau_forest=forest,
            split_gate=np.full(n_events, "positive"),
        )
    with pytest.raises(ValueError, match="finite"):
        adaptive_cut(
            hierarchy,
            2,
            plateau_forest=forest,
            split_gate=np.full(n_events, np.nan),
        )


def test_adaptive_cut_accepts_numpy_integer_minimum_size() -> None:
    hierarchy, _other = _plateau_variants()

    result = adaptive_cut(hierarchy, np.int64(2))

    assert result.labels.tolist() == [1, 1, 1, 1, 2, 2]
    with pytest.raises(TypeError, match="min_cluster_size"):
        adaptive_cut(hierarchy, np.bool_(True))


def test_plateau_storage_scales_linearly() -> None:
    def forest_shape_and_bytes(n_leaves: int) -> tuple[int, int, int]:
        children: list[tuple[int, int]] = [(0, 1)]
        heights = [1.0]
        for leaf in range(2, n_leaves):
            children.append((n_leaves + leaf - 2, leaf))
            heights.append(float(leaf))
        forest = collapse_equal_height_plateaus(_hierarchy(children, heights))
        stored_bytes = sum(
            values.nbytes
            for values in (
                forest.representatives,
                forest.heights,
                forest.sizes,
                forest.parent_events,
                forest.child_offsets,
                forest.child_refs,
                forest.min_leaves,
                forest.component_roots,
            )
        )
        return forest.representatives.size, forest.child_refs.size, stored_bytes

    small_events, small_children, small_bytes = forest_shape_and_bytes(1_000)
    large_events, large_children, large_bytes = forest_shape_and_bytes(2_000)

    assert (small_events, large_events) == (999, 1_999)
    assert (small_children, large_children) == (1_998, 3_998)
    assert 1.9 * small_bytes < large_bytes < 2.1 * small_bytes
    assert large_bytes < 100 * 2_000
