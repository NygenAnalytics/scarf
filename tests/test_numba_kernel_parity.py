import numpy as np
import pytest
from scipy.linalg import solve_triangular
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from scarf.clustering import _paris_core as paris_core
from scarf.clustering import _paris_modularity as paris_modularity
from scarf.features.enrichment import aucell
from scarf.features.genomic import intervals
from scarf.metrics import connectivity
from scarf.neighbors import diffusion
from scarf.trajectory import fate


def test_paris_python_nearest_neighbor_kernel_matches_compiled_kernel() -> None:
    graph = csr_matrix(
        np.array(
            [
                [9.0, 2.0, 2.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    volumes = np.ones(4, dtype=np.float64)
    logical_ids = np.array([9, 5, 3, 1], dtype=np.int64)

    expected = paris_core._nearest_neighbors(
        graph.indptr,
        graph.indices,
        graph.data,
        volumes,
        logical_ids,
        2,
    )
    actual = paris_core._nearest_neighbors.py_func(
        graph.indptr,
        graph.indices,
        graph.data,
        volumes,
        logical_ids,
        2,
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[0], [2, 0, -1, -1])


def test_paris_python_contraction_kernels_match_sparse_aggregation() -> None:
    dense = np.zeros((6, 6), dtype=np.float64)
    for left, right, weight in (
        (0, 1, 5.0),
        (0, 2, 1.0),
        (1, 2, 2.0),
        (0, 3, 4.0),
        (1, 3, 8.0),
        (2, 4, 16.0),
        (3, 4, 7.0),
    ):
        dense[left, right] = dense[right, left] = weight
    graph = csr_matrix(dense)
    mapping = np.array([0, 0, 1, 2, 2, 3], dtype=graph.indices.dtype)

    member_offsets, members, workspace_offsets = paris_core._contraction_layout.py_func(
        graph.indptr, mapping, 4
    )
    workspace_indices, workspace_data, unique_counts = (
        paris_core._aggregate_contracted_rows.py_func(
            graph.indptr,
            graph.indices,
            graph.data,
            mapping,
            member_offsets,
            members,
            workspace_offsets,
            4,
            2,
            np.full((2, 4), np.nan, dtype=np.float64),
        )
    )
    indptr, indices_out, data_out = paris_core._compact_contracted_rows.py_func(
        workspace_offsets,
        workspace_indices,
        workspace_data,
        unique_counts,
    )
    actual = csr_matrix((data_out, indices_out, indptr), shape=(4, 4))

    expected = np.array(
        [
            [0.0, 3.0, 12.0, 0.0],
            [3.0, 0.0, 16.0, 0.0],
            [12.0, 16.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(actual.toarray(), expected)
    np.testing.assert_array_equal(unique_counts, [2, 2, 2, 0])


def test_paris_python_contraction_group_kernel_covers_merged_and_free_nodes() -> None:
    mapping = np.empty(5, dtype=np.int32)
    logical_ids, volumes, sizes = paris_core._prepare_contraction_groups.py_func(
        np.array([10, 11, 12, 13, 14], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1, 1, 2, 3, 4], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
        np.array([1, 4], dtype=np.int64),
        np.array([20, 21], dtype=np.int64),
        mapping,
    )

    np.testing.assert_array_equal(mapping, [1, 1, 0, 2, 2])
    np.testing.assert_array_equal(logical_ids, [12, 20, 21])
    np.testing.assert_array_equal(volumes, [3.0, 3.0, 9.0])
    np.testing.assert_array_equal(sizes, [2, 2, 7])


def test_paris_python_contraction_group_kernel_rejects_overlapping_pair() -> None:
    with np.testing.assert_raises_regex(RuntimeError, "assembly failed"):
        paris_core._prepare_contraction_groups.py_func(
            np.array([0, 1, 2], dtype=np.int64),
            np.ones(3),
            np.ones(3, dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.empty(3, dtype=np.int32),
        )


def test_paris_python_symmetry_kernel_reports_each_failure_mode() -> None:
    valid = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    self_loop = csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    directed = csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    unequal = csr_matrix(np.array([[0.0, 1.0], [2.0, 0.0]]))

    assert (
        paris_modularity._csr_symmetry_error.py_func(
            valid.indptr, valid.indices, valid.data
        )
        == paris_modularity._csr_symmetry_error(valid.indptr, valid.indices, valid.data)
        == 0
    )
    assert (
        paris_modularity._csr_symmetry_error.py_func(
            self_loop.indptr, self_loop.indices, self_loop.data
        )
        == 1
    )
    assert (
        paris_modularity._csr_symmetry_error.py_func(
            directed.indptr, directed.indices, directed.data
        )
        == 2
    )
    assert (
        paris_modularity._csr_symmetry_error.py_func(
            unequal.indptr, unequal.indices, unequal.data
        )
        == 3
    )


def test_paris_python_union_find_kernels_cover_compression_and_rank_cases() -> None:
    parents = np.array([1, 2, 2], dtype=np.int64)
    assert paris_modularity._union_find_root.py_func(parents, 0) == 2
    np.testing.assert_array_equal(parents, [2, 2, 2])

    parents = np.array([0, 0], dtype=np.int64)
    ranks = np.zeros(2, dtype=np.uint8)
    assert paris_modularity._union_sets.py_func(parents, ranks, 0, 1) == 0

    parents = np.arange(2, dtype=np.int64)
    ranks = np.array([0, 1], dtype=np.uint8)
    assert paris_modularity._union_sets.py_func(parents, ranks, 0, 1) == 1
    np.testing.assert_array_equal(parents, [1, 1])

    parents = np.arange(2, dtype=np.int64)
    ranks = np.zeros(2, dtype=np.uint8)
    assert paris_modularity._union_sets.py_func(parents, ranks, 0, 1) == 0
    np.testing.assert_array_equal(parents, [0, 0])
    np.testing.assert_array_equal(ranks, [1, 0])


def test_paris_python_offline_lca_kernel_counts_edges() -> None:
    children = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    graph = csr_matrix(
        np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )

    counts, component_counts, error = paris_modularity._offline_lca_edge_counts.py_func(
        graph.indptr,
        graph.indices,
        children,
        np.array([6], dtype=np.int64),
        np.zeros(4, dtype=np.int32),
    )

    np.testing.assert_array_equal(counts, [1, 1, 2])
    np.testing.assert_array_equal(component_counts, [4])
    assert error == 0


def test_paris_python_offline_lca_kernel_reports_invalid_component_and_lca() -> None:
    children = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    cross_component = csr_matrix(
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    _, _, error = paris_modularity._offline_lca_edge_counts.py_func(
        cross_component.indptr,
        cross_component.indices,
        children,
        np.array([4, 5], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int32),
    )
    assert error == 1

    self_loop = csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    _, _, error = paris_modularity._offline_lca_edge_counts.py_func(
        self_loop.indptr,
        self_loop.indices,
        np.array([[0, 1]], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.zeros(2, dtype=np.int32),
    )
    assert error == 2


def test_fate_python_weight_kernel_matches_compiled_kernel() -> None:
    assert fate._log_biased_weight.py_func(2.0, -1.0, 10.0) == (
        fate._log_biased_weight(2.0, -1.0, 10.0)
    )
    assert fate._log_biased_weight.py_func(2.0, 0.5, 10.0) == (
        fate._log_biased_weight(2.0, 0.5, 10.0)
    )


def test_fate_python_row_bias_kernel_handles_self_loops_underflow_and_isolates() -> (
    None
):
    graph = csr_matrix(
        np.array(
            [
                [4.0, 1.0, 1.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    pseudotime = np.array([1.0, 0.0, 1.0, 0.5])
    absorbing = np.array([False, False, True, False])

    python_data = graph.data.copy()
    compiled_data = graph.data.copy()
    python_count = fate._bias_and_normalize_rows.py_func(
        python_data,
        graph.indices,
        graph.indptr,
        pseudotime,
        absorbing,
        1000.0,
    )
    compiled_count = fate._bias_and_normalize_rows(
        compiled_data,
        graph.indices,
        graph.indptr,
        pseudotime,
        absorbing,
        1000.0,
    )

    assert python_count == compiled_count == 1
    np.testing.assert_array_equal(python_data, compiled_data)
    row_zero = python_data[graph.indptr[0] : graph.indptr[1]]
    assert row_zero[0] == 0.0
    assert 0.0 < row_zero[1] < np.finfo(np.float64).eps
    assert row_zero[1:].sum() == 1.0


def test_fate_python_support_kernel_accepts_symmetric_and_rejects_directed() -> None:
    symmetric = csr_matrix(np.array([[2, 1, 0], [1, 0, 1], [0, 1, 0]]))
    directed = csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))

    assert fate._has_symmetric_support.py_func(symmetric.indices, symmetric.indptr)
    assert not fate._has_symmetric_support.py_func(directed.indices, directed.indptr)


@pytest.mark.parametrize("descending", [False, True])
def test_fate_ordered_sweeps_match_triangular_solve(descending):
    transition = np.array(
        [
            [0, 0.2, 0.3, 0.1, 0.4, 0],
            [0.3, 0, 0.2, 0.1, 0, 0.4],
            [0, 0, 0, 0, 0, 0],
            [0.1, 0.2, 0, 0, 0.3, 0.4],
            [0.2, 0.3, 0.1, 0.4, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    graph = csr_matrix(transition)
    order = np.array([4, 1, 3, 0], dtype=np.int64)
    rank = np.full(6, -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    residual = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.5])
    permutation = np.concatenate(([2, 5], order if descending else order[::-1]))
    system = np.eye(6) - transition
    expected = np.empty(6)
    expected[permutation] = solve_triangular(
        system[np.ix_(permutation, permutation)],
        residual[permutation],
        lower=True,
    )

    for kernel in (fate._ordered_sweep, fate._ordered_sweep.py_func):
        actual = np.full(6, np.nan)
        kernel(
            graph.data,
            graph.indices,
            graph.indptr,
            order,
            rank,
            residual,
            actual,
            descending,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual[[2, 5]], residual[[2, 5]])


@pytest.mark.parametrize("n_bins", [1, 2, 3, 7])
def test_fate_aggregates_match_connected_components_within_bins(n_bins):
    adjacency = np.zeros((8, 8))
    for left, right in [(0, 3), (1, 2), (2, 3), (4, 5), (5, 6), (3, 7), (6, 7)]:
        adjacency[left, right] = adjacency[right, left] = 1
    graph = csr_matrix(adjacency)
    rank = np.array([0, 1, 2, 3, 4, 5, 6, -1])
    bins = rank[:7] * n_bins // 7
    expected = np.full(8, -1)
    count = 0
    for current in np.unique(bins):
        cells = np.flatnonzero(bins == current)
        n_components, labels = connected_components(
            graph[cells][:, cells], directed=False
        )
        expected[cells] = labels + count
        count += n_components

    for kernel in (
        fate._pseudotime_bin_aggregates,
        fate._pseudotime_bin_aggregates.py_func,
    ):
        actual, n_aggregates = kernel(graph.indptr, graph.indices, rank, n_bins)
        assert n_aggregates == count
        assert actual[7] == -1
        np.testing.assert_array_equal(np.unique(actual[:7]), np.arange(count))
        np.testing.assert_array_equal(
            actual[:, None] == actual[None, :],
            expected[:, None] == expected[None, :],
        )


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_diffusion_product_counts_match_sparse_multiplication(seed):
    rng = np.random.default_rng(seed)
    left_values = (rng.random((7, 5)) < 0.35).astype(np.float64)
    right_values = (rng.random((5, 9)) < 0.35).astype(np.float64)
    left_values[0] = 1
    left_values[-1] = 0
    right_values[1] = right_values[0]
    right_values[:, -1] = 0
    left, right = csr_matrix(left_values), csr_matrix(right_values)
    expected = (left @ right).nnz

    for kernel in (diffusion._product_nnz, diffusion._product_nnz.py_func):
        assert (
            kernel(left.indptr, left.indices, right.indptr, right.indices, 9)
            == expected
        )
    bounds = [
        kernel(left.indptr, left.indices, np.diff(right.indptr), 9)
        for kernel in (
            diffusion._product_nnz_bound,
            diffusion._product_nnz_bound.py_func,
        )
    ]
    assert bounds[0] == bounds[1]
    assert expected < bounds[0] <= 7 * 9


def test_interval_python_kernel_matches_compiled_overlap_search() -> None:
    ranges = np.array([[0, 5], [10, 15], [20, 25], [21, 30]], dtype=np.int64)
    queries = np.array([[1, 2], [5, 10], [12, 22], [40, 45], [0, 100]], dtype=np.int64)

    expected = intervals.binary_search(ranges, queries)
    actual = intervals.binary_search.py_func(ranges, queries)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual,
        [[0, 1], [-1, -1], [1, 4], [-1, -1], [0, 4]],
    )


def test_aucell_python_scoring_kernel_matches_compiled_kernel() -> None:
    ranks = np.array([1, 4, 2, 6, 3, 5], dtype=np.int64)
    connections = np.array([0, 2, 4, 1, 3, 0], dtype=np.int64)
    starts = np.array([0, 3, 5], dtype=np.int64)
    offsets = np.array([3, 2, 1], dtype=np.int64)

    expected = aucell._score_ranked_row(ranks, connections, starts, offsets, 3)
    actual = aucell._score_ranked_row.py_func(ranks, connections, starts, offsets, 3)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual, [1.0, 0.0, 1.0])


def test_aucell_python_block_kernel_matches_compiled_kernel() -> None:
    matrix = np.array(
        [[3, 0, 3, 1, 0, 2], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=np.uint32,
    )
    permutation = np.array([2, 0, 5, 1, 4, 3], dtype=np.int64)
    connections = np.array([0, 2, 4, 1, 3, 0], dtype=np.int64)
    starts = np.array([0, 3, 5], dtype=np.int64)
    offsets = np.array([3, 2, 1], dtype=np.int64)

    expected = aucell._score_ranked_block(
        matrix, permutation, connections, starts, offsets, 3
    )
    actual = aucell._score_ranked_block.py_func(
        matrix, permutation, connections, starts, offsets, 3
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[1], [0.0, 0.0, 0.0])


def test_connectivity_python_union_find_kernels_cover_all_edge_cases() -> None:
    parents = np.array([1, 2, 2], dtype=np.int64)
    assert connectivity._find_root.py_func(parents, 0) == 2
    np.testing.assert_array_equal(parents, [2, 2, 2])

    parents = np.arange(4, dtype=np.int64)
    component_sizes = np.array([1, 3, 1, 1], dtype=np.int64)
    connectivity._union_same_label_edges.py_func(
        np.array([[0, 3], [0, 1], [0, 1], [2, 0]], dtype=np.int64),
        np.array([0, 0, 0, 1], dtype=np.int64),
        parents,
        component_sizes,
    )
    np.testing.assert_array_equal(parents, [1, 1, 1, 3])
    assert component_sizes[1] == 5

    parents = np.array([1, 2, 2, 3], dtype=np.int64)
    connectivity._compress_paths.py_func(parents)
    np.testing.assert_array_equal(parents, [2, 2, 2, 3])


def test_transpose_kernel_fills_ragged_tiles_of_a_strided_target() -> None:
    from scarf.utils.strided import transpose_into

    source = np.random.default_rng(3).integers(0, 50, (70, 45), dtype=np.uint16)
    compiled = np.zeros((60, 90), dtype=np.uint16)
    python = np.zeros_like(compiled)
    transpose_into(source, compiled[5:50, 10:80])
    transpose_into.py_func(source, python[5:50, 10:80])

    np.testing.assert_array_equal(compiled[5:50, 10:80], source.T)
    np.testing.assert_array_equal(python, compiled)
    compiled[5:50, 10:80] = 0
    assert not compiled.any()


def test_row_summary_python_kernel_matches_compiled_kernel(monkeypatch) -> None:
    from scarf.utils import digest

    values = np.array([[0, 3, 0, 1], [0, 0, 0, 0], [2, -1, 0, 5]], dtype=np.int32)

    def run(kernel):
        outputs = (
            np.zeros((3, 2), dtype=np.uint64),
            np.zeros(3),
            np.zeros(3, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
        )
        kernel(values, values.view(np.uint32), *outputs)
        return outputs

    compiled = run(digest.summarize_rows)
    # Called from Python, the compiled mixer would type small ints as int64.
    monkeypatch.setattr(digest, "_mix", digest._mix.py_func)
    with np.errstate(over="ignore"):
        python = run(digest.summarize_rows.py_func)

    for expected, actual in zip(compiled, python, strict=True):
        np.testing.assert_array_equal(actual, expected)
    digests, row_sums, row_positive, column_positive = compiled
    np.testing.assert_array_equal(row_sums, [4, 0, 6])
    np.testing.assert_array_equal(row_positive, [2, 0, 2])
    np.testing.assert_array_equal(column_positive, [1, 1, 0, 2])
    assert len({tuple(row) for row in digests.tolist()}) == 3


def test_column_copy_kernel_matches_fancy_assignment() -> None:
    from scarf.utils.strided import copy_columns

    source = np.random.default_rng(4).integers(0, 50, (6, 9), dtype=np.uint16)
    columns = np.array([0, 2, 3, 7], dtype=np.int64)
    positions = np.array([5, 1, 2, 9], dtype=np.int64)
    expected = np.zeros((6, 12), dtype=np.uint16)
    expected[:, positions] = source[:, columns]
    for kernel in (copy_columns, copy_columns.py_func):
        target = np.zeros_like(expected)
        kernel(source, columns, target, positions)
        np.testing.assert_array_equal(target, expected)


def test_radix_argsort_matches_stable_argsort_for_positive_values() -> None:
    from scarf.features.markers.rank import _argsort_positive

    rng = np.random.default_rng(5)
    values = np.concatenate(
        [
            rng.random(300).astype(np.float32) * 20 + 1e-3,
            np.full(40, 2.5, dtype=np.float32),
            np.float32([1e-30, 3e38, 7.0, 7.0]),
        ]
    )
    rng.shuffle(values)
    expected = np.argsort(values, kind="stable")
    for kernel in (_argsort_positive, _argsort_positive.py_func):
        order = np.full(values.size + 3, -1, dtype=np.int64)
        kernel(
            values,
            values.size,
            order,
            np.empty_like(order),
            np.empty(2048, dtype=np.int64),
        )
        np.testing.assert_array_equal(order[: values.size], expected)
