import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sknetwork.hierarchy import Paris

from scarf.clustering._paris_core import _contract_graph, canonicalize_paris_graph
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
