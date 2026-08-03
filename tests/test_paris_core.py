import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sknetwork.hierarchy import Paris

from scarf.clustering._paris_core import (
    ParisHierarchy,
    _contract_graph,
    _nearest_neighbors,
    canonicalize_paris_graph,
)
from scarf.clustering.paris import (
    fit_paris_hierarchy,
    hierarchy_to_dendrogram,
    paris_dendrogram,
)


def _leaf_sets(dendrogram: np.ndarray) -> dict[frozenset[int], float]:
    n_leaves = dendrogram.shape[0] + 1
    leaves = {node: frozenset({node}) for node in range(n_leaves)}
    clades: dict[frozenset[int], float] = {}
    for merge_index, row in enumerate(dendrogram):
        members = leaves[int(row[0])] | leaves[int(row[1])]
        leaves[n_leaves + merge_index] = members
        clades[members] = float(row[2])
    return clades


def _tie_free_graph() -> csr_matrix:
    rng = np.random.default_rng(742)
    n_vertices = 40
    rows = np.repeat(np.arange(n_vertices), 5)
    columns = rng.integers(0, n_vertices, size=rows.size)
    weights = rng.uniform(0.1, 2.0, size=rows.size)
    return csr_matrix(
        (weights, (rows, columns)),
        shape=(n_vertices, n_vertices),
    )


def test_merge_distances_match_aggregated_graph_weights() -> None:
    graph = canonicalize_paris_graph(_tie_free_graph())
    hierarchy = fit_paris_hierarchy(graph, n_threads=2)
    total_weight = float(graph.data.sum())
    members = {
        node: np.asarray([node], dtype=np.int64) for node in range(hierarchy.n_leaves)
    }
    volumes = np.asarray(graph.sum(axis=1)).ravel()

    for merge_index, children in enumerate(hierarchy.children):
        left, right = map(int, children)
        left_members = members[left]
        right_members = members[right]
        between_weight = float(graph[left_members][:, right_members].sum())
        left_volume = float(volumes[left_members].sum())
        right_volume = float(volumes[right_members].sum())
        expected = (left_volume / total_weight) * (right_volume / between_weight)
        assert hierarchy.heights[merge_index] == pytest.approx(expected)
        members[hierarchy.n_leaves + merge_index] = np.concatenate(
            (left_members, right_members)
        )


def test_tie_free_topology_matches_scikit_network() -> None:
    canonical = canonicalize_paris_graph(_tie_free_graph())
    native = hierarchy_to_dendrogram(fit_paris_hierarchy(canonical, n_threads=4))
    reference = np.asarray(Paris(reorder=False).fit_transform(canonical))
    native_clades = _leaf_sets(native)
    reference_clades = _leaf_sets(reference)

    assert native_clades.keys() == reference_clades.keys()
    for clade in native_clades:
        assert native_clades[clade] == pytest.approx(
            reference_clades[clade],
            abs=1e-6,
            rel=1e-6,
        )


def test_output_is_identical_across_thread_counts() -> None:
    graph = _tie_free_graph()
    reference = fit_paris_hierarchy(graph, n_threads=1)
    for n_threads in (2, 4, 8):
        result = fit_paris_hierarchy(graph, n_threads=n_threads)
        assert np.array_equal(result.children, reference.children)
        assert np.array_equal(result.heights, reference.heights)
        assert np.array_equal(result.sizes, reference.sizes)
        assert np.array_equal(result.component_roots, reference.component_roots)
        assert np.array_equal(result.synthetic_joins, reference.synthetic_joins)


def test_contraction_diagnostics_account_for_each_phase() -> None:
    hierarchy = fit_paris_hierarchy(_tie_free_graph(), n_threads=2)
    diagnostics = hierarchy.diagnostics
    assert diagnostics is not None
    assert diagnostics.rounds

    for round_ in diagnostics.rounds:
        phases = (
            round_.contraction_remap_seconds,
            round_.contraction_filter_seconds,
            round_.contraction_build_seconds,
            round_.contraction_cleanup_seconds,
        )
        assert all(seconds >= 0 for seconds in phases)
        assert sum(phases) <= round_.contraction_seconds + 1e-9


def test_parallel_contraction_matches_scipy_reference() -> None:
    graph = canonicalize_paris_graph(_tie_free_graph())
    group_ids = np.arange(graph.shape[0], dtype=graph.indices.dtype)
    group_ids[20:30] = group_ids[:10]
    logical_groups = np.unique(group_ids)
    mapping = np.searchsorted(logical_groups, group_ids).astype(
        graph.indices.dtype,
        copy=False,
    )
    n_vertices = logical_groups.size

    rows = np.repeat(
        np.arange(graph.shape[0], dtype=graph.indices.dtype),
        np.diff(graph.indptr),
    )
    mapped_rows = mapping[rows]
    mapped_columns = mapping[graph.indices]
    keep = mapped_rows != mapped_columns
    expected = csr_matrix(
        (
            graph.data[keep],
            (mapped_rows[keep], mapped_columns[keep]),
        ),
        shape=(n_vertices, n_vertices),
    )
    expected.sum_duplicates()
    expected.eliminate_zeros()
    expected.sort_indices()

    values = np.full((2, graph.shape[0]), np.nan, dtype=np.float64)
    contracted, *_timings = _contract_graph(
        graph,
        mapping,
        n_vertices,
        values,
    )

    assert np.array_equal(contracted.indptr, expected.indptr)
    assert np.array_equal(contracted.indices, expected.indices)
    assert np.array_equal(contracted.data, expected.data)
    assert contracted.has_canonical_format
    assert np.isnan(values).all()

    repeated, *_timings = _contract_graph(
        graph,
        mapping,
        n_vertices,
        values,
    )
    assert np.array_equal(repeated.indptr, contracted.indptr)
    assert np.array_equal(repeated.indices, contracted.indices)
    assert np.array_equal(repeated.data, contracted.data)


def test_ties_use_smallest_logical_tree_id() -> None:
    graph = csr_matrix(
        np.asarray(
            [
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
    )
    hierarchy = fit_paris_hierarchy(graph, n_threads=4)
    assert hierarchy.children[0].tolist() == [0, 1]
    assert hierarchy.children.tolist() == [[0, 1], [2, 4], [3, 5]]


def test_canonicalization_sums_duplicates_sorts_and_removes_diagonal() -> None:
    indptr = np.asarray([0, 4, 5, 6], dtype=np.int32)
    indices = np.asarray([2, 1, 1, 0, 0, 1], dtype=np.int32)
    data = np.asarray([2.0, 1.0, 3.0, 9.0, 4.0, 5.0])
    graph = csr_matrix((data, indices, indptr), shape=(3, 3))
    canonical = canonicalize_paris_graph(graph)

    assert canonical.has_sorted_indices
    assert canonical.diagonal().tolist() == [0.0, 0.0, 0.0]
    assert np.array_equal(
        canonical.toarray(),
        np.asarray(
            [
                [0.0, 8.0, 2.0],
                [8.0, 0.0, 5.0],
                [2.0, 5.0, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_invalid_weights_are_rejected(weight: float) -> None:
    graph = csr_matrix(np.asarray([[0.0, weight], [1.0, 0.0]]))
    error = "non-negative" if weight < 0 else "finite"
    with pytest.raises(ValueError, match=error):
        fit_paris_hierarchy(graph)


def test_canonicalize_rejects_non_square_and_tiny_graphs() -> None:
    with pytest.raises(ValueError, match="square"):
        canonicalize_paris_graph(csr_matrix(np.ones((2, 3), dtype=np.float64)))
    with pytest.raises(ValueError, match="at least two vertices"):
        canonicalize_paris_graph(csr_matrix([[0.0]], dtype=np.float64))


def test_paris_hierarchy_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="children must have shape"):
        ParisHierarchy(
            children=np.zeros((1, 2), dtype=np.int32),
            heights=np.array([1.0, 2.0], dtype=np.float64),
            sizes=np.array([2, 3], dtype=np.int32),
            component_roots=np.array([4], dtype=np.int32),
            synthetic_joins=np.array([False, False]),
            n_leaves=3,
            total_weight=1.0,
        )
    with pytest.raises(ValueError, match="heights must have length"):
        ParisHierarchy(
            children=np.array([[0, 1], [2, 3]], dtype=np.int32),
            heights=np.array([1.0], dtype=np.float64),
            sizes=np.array([2, 3], dtype=np.int32),
            component_roots=np.array([4], dtype=np.int32),
            synthetic_joins=np.array([False, False]),
            n_leaves=3,
            total_weight=1.0,
        )


def test_components_isolates_and_synthetic_joins_are_explicit() -> None:
    graph = csr_matrix(
        np.asarray(
            [
                [0, 2, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
    )
    hierarchy = fit_paris_hierarchy(graph)
    raw = hierarchy_to_dendrogram(hierarchy)
    compatibility = hierarchy_to_dendrogram(hierarchy, compatibility=True)

    assert hierarchy.component_roots.tolist() == [5, 6, 4]
    assert hierarchy.synthetic_joins.tolist() == [False, False, True, True]
    assert np.isinf(raw[hierarchy.synthetic_joins, 2]).all()
    assert np.all(compatibility[hierarchy.synthetic_joins, 2] == 0)
    assert np.array_equal(paris_dendrogram(graph), compatibility)
    assert hierarchy.children.dtype == np.int32
    assert hierarchy.sizes.dtype == np.int32
    assert hierarchy.heights.dtype == np.float64


def test_nearest_neighbors_skips_self_loops_nonpositive_and_ties_on_id() -> None:
    indptr = np.asarray([0, 4, 5, 6, 6], dtype=np.int64)
    indices = np.asarray([0, 1, 2, 3, 0, 0], dtype=np.int64)
    data = np.asarray([9.0, 0.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
    volumes = np.ones(4, dtype=np.float64)
    logical_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)

    nearest, between = _nearest_neighbors(
        indptr,
        indices,
        data,
        volumes,
        logical_ids,
        1,
    )

    np.testing.assert_array_equal(nearest, [2, 0, 0, -1])
    np.testing.assert_allclose(between, [2.0, 1.0, 1.0, 0.0])


def test_contract_graph_sums_parallel_mapped_edges_and_drops_intra_group() -> None:
    graph = csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 2.0, 0.0],
                [1.0, 0.0, 0.0, 3.0],
                [2.0, 0.0, 0.0, 4.0],
                [0.0, 3.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    mapping = np.asarray([0, 0, 1, 1], dtype=np.int64)
    values = np.full((1, graph.shape[0]), np.nan, dtype=np.float64)
    contracted, *_ = _contract_graph(graph, mapping, 2, values)

    np.testing.assert_array_equal(
        contracted.toarray(),
        np.asarray([[0.0, 5.0], [5.0, 0.0]]),
    )
    assert contracted.has_sorted_indices


def test_canonicalize_rejects_symmetrization_overflow() -> None:
    huge = np.finfo(np.float64).max
    graph = csr_matrix(
        (
            np.asarray([huge, huge], dtype=np.float64),
            (np.asarray([0, 1]), np.asarray([1, 0])),
        ),
        shape=(2, 2),
    )
    with pytest.raises(ValueError, match="overflowed during symmetrization"):
        canonicalize_paris_graph(graph)
