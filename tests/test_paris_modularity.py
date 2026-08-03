import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering._paris_modularity import (
    PlateauModularityStatistics,
    TopologyStatistics,
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


def test_topology_statistics_reject_invalid_arrays() -> None:
    with pytest.raises(TypeError, match="one-dimensional integer array"):
        TopologyStatistics(
            leaf_degrees=np.array([[1]], dtype=np.int64),
            lca_edge_counts=np.array([0], dtype=np.int64),
            component_edge_counts=np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        TopologyStatistics(
            leaf_degrees=np.array([-1], dtype=np.int64),
            lca_edge_counts=np.array([0], dtype=np.int64),
            component_edge_counts=np.array([0], dtype=np.int64),
        )
    with pytest.raises(TypeError, match="integer array over events"):
        PlateauModularityStatistics(
            cross_edges=np.array([1.0]),
            volumes=np.array([2], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        PlateauModularityStatistics(
            cross_edges=np.array([1], dtype=np.int64),
            volumes=np.array([-2], dtype=np.int64),
        )


def test_collect_topology_rejects_non_canonical_graphs() -> None:
    hierarchy = _balanced_four_hierarchy()

    with pytest.raises(TypeError, match="csr_matrix"):
        collect_topology_statistics(np.ones((4, 4)), hierarchy)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        collect_topology_statistics(
            csr_matrix(np.ones((4, 3), dtype=np.float64)),
            hierarchy,
        )
    with pytest.raises(ValueError, match="different leaf counts"):
        collect_topology_statistics(_canonical_graph(3, [(0, 1)]), hierarchy)

    unsorted = csr_matrix(
        (
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([2, 1], dtype=np.int32),
            np.array([0, 2, 2, 2, 2], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    with pytest.raises(ValueError, match="sorted canonical CSR"):
        collect_topology_statistics(unsorted, hierarchy)

    nonpositive = _canonical_graph(4, [(0, 1), (2, 3)])
    nonpositive.data[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        collect_topology_statistics(nonpositive, hierarchy)

    with_self_loop = csr_matrix(
        (
            np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            np.array([0, 1, 0, 2, 3], dtype=np.int32),
            np.array([0, 2, 3, 4, 5], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    with_self_loop = with_self_loop + with_self_loop.T
    with_self_loop.sum_duplicates()
    with_self_loop.sort_indices()
    with pytest.raises(ValueError, match="self-loops"):
        collect_topology_statistics(with_self_loop, hierarchy)

    asymmetric = _canonical_graph(4, [(0, 1), (2, 3)])
    asymmetric.data[0] = 2.0
    with pytest.raises(ValueError, match="exactly symmetric"):
        collect_topology_statistics(asymmetric, hierarchy)

    directed_only = csr_matrix(
        (
            np.array([1.0], dtype=np.float64),
            np.array([1], dtype=np.int32),
            np.array([0, 1, 1, 1, 1], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    directed_only.sort_indices()
    with pytest.raises(ValueError, match="structurally symmetric"):
        collect_topology_statistics(directed_only, hierarchy)


def test_collect_topology_rejects_edges_across_hierarchy_components() -> None:
    hierarchy = _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [1.0, 1.0, np.inf],
        component_roots=[4, 5],
        synthetic_joins=[False, False, True],
    )
    graph = _canonical_graph(4, [(0, 1), (2, 3), (0, 2)])
    with pytest.raises(ValueError, match="across hierarchy components"):
        collect_topology_statistics(graph, hierarchy)


def test_collect_topology_rejects_synthetic_join_component_root() -> None:
    hierarchy = _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [1.0, 1.0, np.inf],
        component_roots=[6],
        synthetic_joins=[False, False, True],
    )
    graph = _canonical_graph(4, [(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="do not match synthetic joins"):
        collect_topology_statistics(graph, hierarchy)


def test_aggregate_plateau_statistics_rejects_inconsistent_topology() -> None:
    hierarchy = _balanced_four_hierarchy()
    forest = collapse_equal_height_plateaus(hierarchy)
    graph = _canonical_graph(4, [(0, 1), (2, 3), (0, 2)])
    topology = collect_topology_statistics(graph, hierarchy)

    with pytest.raises(ValueError, match="leaf degree statistics"):
        aggregate_plateau_statistics(
            hierarchy,
            forest,
            TopologyStatistics(
                leaf_degrees=np.zeros(3, dtype=np.int64),
                lca_edge_counts=topology.lca_edge_counts,
                component_edge_counts=topology.component_edge_counts,
            ),
        )
    with pytest.raises(ValueError, match="LCA edge statistics"):
        aggregate_plateau_statistics(
            hierarchy,
            forest,
            TopologyStatistics(
                leaf_degrees=topology.leaf_degrees,
                lca_edge_counts=np.zeros(2, dtype=np.int64),
                component_edge_counts=topology.component_edge_counts,
            ),
        )
    with pytest.raises(ValueError, match="component edge statistics"):
        aggregate_plateau_statistics(
            hierarchy,
            forest,
            TopologyStatistics(
                leaf_degrees=topology.leaf_degrees,
                lca_edge_counts=topology.lca_edge_counts,
                component_edge_counts=np.zeros(2, dtype=np.int64),
            ),
        )
    with pytest.raises(ValueError, match="disagree on the total edge count"):
        aggregate_plateau_statistics(
            hierarchy,
            forest,
            TopologyStatistics(
                leaf_degrees=topology.leaf_degrees,
                lca_edge_counts=topology.lca_edge_counts,
                component_edge_counts=topology.component_edge_counts + 1,
            ),
        )

    synthetic = _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [1.0, 1.0, np.inf],
        component_roots=[4, 5],
        synthetic_joins=[False, False, True],
    )
    synthetic_forest = collapse_equal_height_plateaus(synthetic)
    synthetic_graph = _canonical_graph(4, [(0, 1), (2, 3)])
    synthetic_topology = collect_topology_statistics(synthetic_graph, synthetic)
    poisoned_counts = np.asarray(synthetic_topology.lca_edge_counts).copy()
    poisoned_counts[np.asarray(synthetic.synthetic_joins)] = 1
    component_counts = np.zeros_like(synthetic_topology.component_edge_counts)
    component_counts[0] = int(poisoned_counts.sum())
    with pytest.raises(ValueError, match="synthetic joins cannot contain"):
        aggregate_plateau_statistics(
            synthetic,
            synthetic_forest,
            TopologyStatistics(
                leaf_degrees=np.asarray(synthetic_topology.leaf_degrees).copy(),
                lca_edge_counts=poisoned_counts,
                component_edge_counts=component_counts,
            ),
        )
