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


def test_hierarchy_validation_allows_one_ulp_height_roundoff() -> None:
    parent_height = 1.0
    child_height = np.nextafter(parent_height, np.inf)
    hierarchy = _hierarchy(
        [(0, 1), (3, 2)],
        [child_height, parent_height],
    )

    forest = collapse_equal_height_plateaus(hierarchy)

    assert forest.representatives.size == 2


def test_hierarchy_validation_rejects_more_than_one_ulp_height_inversion() -> None:
    parent_height = 1.0
    child_height = np.nextafter(
        np.nextafter(parent_height, np.inf),
        np.inf,
    )
    hierarchy = _hierarchy(
        [(0, 1), (3, 2)],
        [child_height, parent_height],
    )

    with pytest.raises(ValueError, match="merge distances must be monotone"):
        collapse_equal_height_plateaus(hierarchy)


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


def test_clustering_result_validates_its_labels_and_mode() -> None:
    from scarf.clustering.paris_multiscale import ParisClusteringResult

    hierarchy, _other = _plateau_variants()
    adaptive = adaptive_cut(hierarchy, 2)

    with pytest.raises(ValueError, match="one-dimensional int32 array"):
        ParisClusteringResult(
            labels=np.ones(4, dtype=np.int64), mode="fixed", n_clusters=1
        )
    with pytest.raises(ValueError, match="one-dimensional int32 array"):
        ParisClusteringResult(
            labels=np.ones((2, 2), dtype=np.int32), mode="fixed", n_clusters=1
        )
    with pytest.raises(ValueError, match="n_clusters must be positive"):
        ParisClusteringResult(
            labels=np.ones(4, dtype=np.int32), mode="fixed", n_clusters=0
        )
    with pytest.raises(ValueError, match="fixed cuts cannot contain adaptive"):
        ParisClusteringResult(
            labels=np.ones(6, dtype=np.int32),
            mode="fixed",
            n_clusters=2,
            diagnostics=adaptive.diagnostics,
        )


def test_plateau_forest_requires_aligned_event_arrays() -> None:
    from dataclasses import replace

    hierarchy, _other = _plateau_variants()
    forest = collapse_equal_height_plateaus(hierarchy)
    n_events = forest.representatives.size

    with pytest.raises(ValueError, match="heights must have one value per event"):
        replace(forest, heights=np.ones(n_events + 1))
    with pytest.raises(ValueError, match="one more item than events"):
        replace(forest, child_offsets=forest.child_offsets[:-1].copy())
    with pytest.raises(ValueError, match="do not span child_refs"):
        replace(forest, child_refs=forest.child_refs[:-1].copy())


def _with(hierarchy: ParisHierarchy, **changes: object) -> ParisHierarchy:
    from dataclasses import replace

    return replace(hierarchy, **changes)


@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        (
            lambda h: ParisHierarchy(
                children=np.empty((0, 2), dtype=np.int32),
                heights=np.empty(0),
                sizes=np.empty(0, dtype=np.int32),
                component_roots=np.zeros(1, dtype=np.int32),
                synthetic_joins=np.empty(0, dtype=bool),
                n_leaves=1,
                total_weight=0.0,
            ),
            ValueError,
            "at least two leaves",
        ),
        (
            lambda h: _with(h, children=h.children.astype(np.float64)),
            TypeError,
            "child references must be integers",
        ),
        (
            lambda h: _with(h, sizes=h.sizes.astype(np.float64)),
            TypeError,
            "subtree sizes must be integers",
        ),
        (
            lambda h: _with(h, heights=np.array([1.0, 1.0, np.inf])),
            ValueError,
            "Only synthetic component joins may have infinite distance",
        ),
        (
            lambda h: _with(h, synthetic_joins=np.array([False, False, True])),
            ValueError,
            "Synthetic component joins must have infinite distance",
        ),
        (
            lambda h: _with(h, children=np.array([[0, 1], [2, 2], [4, 5]], np.int32)),
            ValueError,
            "distinct topological references",
        ),
        (
            lambda h: _with(h, children=np.array([[0, 1], [1, 2], [4, 5]], np.int32)),
            ValueError,
            "more than one parent",
        ),
        (
            lambda h: _with(h, sizes=np.array([2, 2, 3], dtype=np.int32)),
            ValueError,
            "subtree size does not match its children",
        ),
        (
            lambda h: _with(
                h,
                heights=np.array([np.inf, 1.0, 2.0]),
                synthetic_joins=np.array([True, False, False]),
            ),
            ValueError,
            "finite merge cannot contain a synthetic join",
        ),
    ],
    ids=[
        "one_leaf",
        "float_children",
        "float_sizes",
        "infinite_finite_merge",
        "finite_synthetic_join",
        "repeated_child",
        "shared_child",
        "wrong_size",
        "synthetic_inside_finite",
    ],
)
def test_hierarchy_validation_rejects_malformed_hierarchies(
    damage, error, message
) -> None:
    # Four leaves: (0, 1) -> 4, (2, 3) -> 5, (4, 5) -> 6.
    valid = _hierarchy([(0, 1), (2, 3), (4, 5)], [1.0, 1.0, 2.0])
    collapse_equal_height_plateaus(valid)

    with pytest.raises(error, match=message):
        collapse_equal_height_plateaus(damage(valid))


def test_events_above_a_rounding_inversion_have_no_persistence() -> None:
    from scarf.clustering.paris_multiscale import _event_keep_score

    # The child merge sits one ulp above its parent, which validation accepts.
    hierarchy = _hierarchy([(0, 1), (3, 2)], [np.nextafter(1.0, np.inf), 1.0])
    forest = collapse_equal_height_plateaus(hierarchy)

    assert forest.parent_events.tolist() == [1, -1]
    assert _event_keep_score(forest, 0) == 0.0
    result = adaptive_cut(hierarchy, 2)
    # Leaf 2 alone cannot form a cluster, so the root is kept whole.
    assert result.labels.tolist() == [1, 1, 1]
    assert result.diagnostics[0].forced


def test_a_root_smaller_than_the_minimum_size_is_kept_whole() -> None:
    hierarchy = _hierarchy([(0, 1), (2, 3), (4, 5)], [1.0, 1.0, 2.0])

    result = adaptive_cut(hierarchy, 5)

    assert result.labels.tolist() == [1, 1, 1, 1]
    (diagnostic,) = result.diagnostics
    assert diagnostic.forced
    assert diagnostic.selected_node == 6
    assert diagnostic.persistence is None


def test_adaptive_cut_rejects_a_forest_of_another_hierarchy() -> None:
    hierarchy, _other = _plateau_variants()
    smaller = collapse_equal_height_plateaus(
        _hierarchy([(0, 1), (2, 3), (4, 5)], [1.0, 1.0, 2.0])
    )

    with pytest.raises(ValueError, match="different leaf counts"):
        adaptive_cut(hierarchy, 2, plateau_forest=smaller)


def test_labels_from_selected_nodes_regenerates_and_validates_a_cut() -> None:
    from scarf.clustering.paris_multiscale import labels_from_selected_nodes

    # Four leaves: (0, 1) -> 4, (2, 3) -> 5, (4, 5) -> 6.
    hierarchy = _hierarchy([(0, 1), (2, 3), (4, 5)], [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        labels_from_selected_nodes(hierarchy, np.array([5, 0, 1])), [2, 3, 1, 1]
    )

    for selected, error, message in (
        (np.array([], dtype=np.int64), ValueError, "non-empty one-dimensional"),
        (np.array([[4, 5]]), ValueError, "non-empty one-dimensional"),
        (np.array([4.0, 5.0]), TypeError, "must contain integers"),
        (np.array([4, 4, 5]), ValueError, "must not contain duplicates"),
        (np.array([4, 7]), ValueError, "outside the hierarchy"),
        (np.array([4, -1]), ValueError, "outside the hierarchy"),
        (np.array([6, 0]), RuntimeError, "clusters overlap"),
        (np.array([4]), RuntimeError, "did not cover every leaf"),
    ):
        with pytest.raises(error, match=message):
            labels_from_selected_nodes(hierarchy, selected)

    # Two pairs joined only synthetically: the join node cannot be selected.
    disconnected = _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [1.0, 1.0, np.inf],
        component_roots=[4, 5],
    )
    with pytest.raises(ValueError, match="synthetic component joins cannot"):
        labels_from_selected_nodes(disconnected, np.array([6]))
