import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering._paris_modularity import (
    aggregate_plateau_statistics,
    collect_topology_statistics,
    modularity_split_gains,
)
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
    raw_sizes = np.ones(2 * n_leaves - 1, dtype=np.int32)
    merge_sizes = np.empty(n_leaves - 1, dtype=np.int32)
    for merge_index, (left, right) in enumerate(children):
        size = int(raw_sizes[left]) + int(raw_sizes[right])
        raw_sizes[n_leaves + merge_index] = size
        merge_sizes[merge_index] = size
    roots = [2 * n_leaves - 2] if component_roots is None else component_roots
    synthetic = (
        np.isinf(heights).tolist() if synthetic_joins is None else synthetic_joins
    )
    return ParisHierarchy(
        children=child_array,
        heights=np.asarray(heights, dtype=np.float64),
        sizes=merge_sizes,
        component_roots=np.asarray(roots, dtype=np.int32),
        synthetic_joins=np.asarray(synthetic, dtype=bool),
        n_leaves=n_leaves,
        total_weight=1.0,
    )


def _canonical_graph(
    n_vertices: int,
    edges: list[tuple[int, int]],
) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for left, right in edges:
        rows.extend((left, right))
        columns.extend((right, left))
    graph = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float64),
            (rows, columns),
        ),
        shape=(n_vertices, n_vertices),
    )
    graph.sum_duplicates()
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _erdos_renyi_null_graph(seed: int) -> csr_matrix:
    n_vertices = 1_000
    rows, columns = np.triu_indices(n_vertices, k=1)
    keep = np.random.default_rng(seed).random(rows.size) < 0.01
    upper = csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float64),
            (rows[keep], columns[keep]),
        ),
        shape=(n_vertices, n_vertices),
    )
    return csr_matrix(upper + upper.T)


def _balanced_four_hierarchy(*, plateau: bool = False) -> ParisHierarchy:
    return _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [1.0, 1.0, 1.0 if plateau else 2.0],
    )


def _balanced_eight_hierarchy() -> ParisHierarchy:
    return _hierarchy(
        [
            (0, 1),
            (2, 3),
            (8, 9),
            (4, 5),
            (6, 7),
            (11, 12),
            (10, 13),
        ],
        [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0],
    )


def test_offline_lca_and_event_statistics_match_scalars() -> None:
    hierarchy = _balanced_four_hierarchy()
    graph = _canonical_graph(4, [(0, 1), (2, 3), (0, 2), (1, 2)])
    forest = collapse_equal_height_plateaus(hierarchy)

    topology = collect_topology_statistics(graph, hierarchy)
    repeated = collect_topology_statistics(graph, hierarchy)
    assert topology.leaf_degrees.tolist() == [2, 2, 3, 1]
    assert topology.lca_edge_counts.tolist() == [1, 1, 2]
    assert topology.component_edge_counts.tolist() == [4]
    assert topology.leaf_degrees.tobytes() == repeated.leaf_degrees.tobytes()
    assert topology.lca_edge_counts.tobytes() == repeated.lca_edge_counts.tobytes()
    assert not topology.lca_edge_counts.flags.writeable

    events = aggregate_plateau_statistics(hierarchy, forest, topology)
    assert events.cross_edges.tolist() == [1, 1, 2]
    assert events.volumes.tolist() == [4, 4, 8]
    assert not events.cross_edges.flags.writeable
    assert not events.volumes.flags.writeable


def test_plateau_statistics_ignore_binary_refinement() -> None:
    first = _balanced_four_hierarchy(plateau=True)
    second = _hierarchy(
        [(0, 2), (1, 3), (4, 5)],
        [1.0, 1.0, 1.0],
    )
    graph = _canonical_graph(4, [(0, 1), (2, 3), (0, 2), (1, 2)])
    first_forest = collapse_equal_height_plateaus(first)
    second_forest = collapse_equal_height_plateaus(second)
    first_events = aggregate_plateau_statistics(
        first,
        first_forest,
        collect_topology_statistics(graph, first),
    )
    second_events = aggregate_plateau_statistics(
        second,
        second_forest,
        collect_topology_statistics(graph, second),
    )

    assert first_forest.representatives.size == 1
    assert second_forest.representatives.size == 1
    assert np.array_equal(first_events.cross_edges, second_events.cross_edges)
    assert np.array_equal(first_events.volumes, second_events.volumes)


def test_modularity_split_gain_matches_scalar_delta_q() -> None:
    hierarchy = _balanced_four_hierarchy()
    forest = collapse_equal_height_plateaus(hierarchy)
    graph = _canonical_graph(4, [(0, 1), (2, 3), (0, 2), (1, 2)])
    gains = modularity_split_gains(hierarchy, forest, graph)

    two_m = 8.0
    expected_root = (8.0**2 - (4.0**2 + 4.0**2)) / two_m**2 - 2.0 * 2.0 / two_m
    root = int(forest.component_roots[0])
    assert gains[root] == pytest.approx(expected_root)
    expected_pair = (4.0**2 - (2.0**2 + 2.0**2)) / two_m**2 - 2.0 * 1.0 / two_m
    assert gains[0] == pytest.approx(expected_pair)


def test_split_gate_vetoes_exact_non_positive_gain_splits() -> None:
    hierarchy = _balanced_eight_hierarchy()
    graph = _canonical_graph(
        8,
        [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (5, 6), (3, 4), (0, 7)],
    )
    forest = collapse_equal_height_plateaus(hierarchy)
    gains = modularity_split_gains(hierarchy, forest, graph)

    unguarded = adaptive_cut(hierarchy, 2, plateau_forest=forest)
    guarded = adaptive_cut(hierarchy, 2, plateau_forest=forest, split_gate=gains)
    root_veto = np.ones(forest.representatives.size, dtype=np.float64)
    root_veto[int(forest.component_roots[0])] = 0.0
    coarsened = adaptive_cut(
        hierarchy,
        2,
        plateau_forest=forest,
        split_gate=root_veto,
    )
    permissive = adaptive_cut(
        hierarchy,
        2,
        plateau_forest=forest,
        split_gate=np.ones(forest.representatives.size),
    )

    assert unguarded.labels.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert guarded.labels.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert [item.selected_node for item in guarded.diagnostics] == [10, 13]
    assert coarsened.labels.tolist() == [1] * 8
    assert np.array_equal(permissive.labels, unguarded.labels)


def test_guard_reduces_erdos_renyi_null_fragmentation() -> None:
    unguarded_counts: list[int] = []
    guarded_counts: list[int] = []
    for seed in range(3):
        graph = _erdos_renyi_null_graph(seed)
        hierarchy = fit_paris_hierarchy(graph, n_threads=2)
        forest = collapse_equal_height_plateaus(hierarchy)
        assert hierarchy.component_roots.size == 1

        unguarded = adaptive_cut(hierarchy, 11, plateau_forest=forest)
        guarded = adaptive_cut(
            hierarchy,
            11,
            plateau_forest=forest,
            split_gate=modularity_split_gains(hierarchy, forest, graph),
        )
        unguarded_counts.append(unguarded.n_clusters)
        guarded_counts.append(guarded.n_clusters)

    assert all(
        guarded < unguarded
        for guarded, unguarded in zip(
            guarded_counts,
            unguarded_counts,
            strict=True,
        )
    )
    assert 2 * sum(guarded_counts) <= sum(unguarded_counts)
