"""Benchmarks of graph integration and of the work that follows a Paris fit.

Each benchmark times the function a production stage calls, on inputs shaped
like the recorded 1M-cell funnel with 11 neighbours per cell, and checks the
result against an independent oracle, so it doubles as a test at the smoke
size. The integration benchmarks merge an RNA modality with a protein
modality of the same cells, as ``integrate_assays`` does. The Paris
benchmarks start from a fitted hierarchy, which ``test_kernels`` times, and
cover the cuts and the coalesced cluster tree built from it.
"""

import math
import sys
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse

from . import inputs
from .harness import Ladder

if TYPE_CHECKING:
    from scarf.clustering._paris_core import ParisHierarchy

pytestmark = pytest.mark.benchmark

# Seurat's CITE-seq WNN vignette keeps 18 protein components.
PROTEIN_DIMS = 18
# WNN scores every cell in a Python loop, so its ladder stops earlier.
WNN_CELLS = Ladder(sizes=(500, 1_000, 2_000, 4_000), smoke=200)
SNN_CELLS = Ladder(sizes=(2_000, 4_000, 8_000, 16_000), smoke=300)
PARIS_CELLS = Ladder(sizes=(4_000, 8_000, 16_000, 32_000), smoke=600)
# A user-requested cluster count above the eight drawn groups.
FIXED_CLUSTERS = 20
# Integrated rows rebuilt from scratch by each check.
CHECKED_ROWS = 48


def _nearest_other_cells(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest other cells and L2 distances, as inputs.neighbors does."""
    from scarf.neighbors.index import fix_knn_query, instantiate_knn_index

    n_cells, dims = values.shape
    index = instantiate_knn_index(
        "l2",
        dims,
        n_cells,
        ef_construction=100,
        M=32,
        random_seed=inputs.SEED,
        ef=100,
        nthreads=1,
    )
    index.add_items(values, num_threads=1)
    indices, distances = index.knn_query(
        values, k=inputs.N_NEIGHBORS + 1, num_threads=1
    )
    indices, distances, _missed = fix_knn_query(
        indices, np.sqrt(distances), np.arange(n_cells)
    )
    return np.ascontiguousarray(indices, dtype=np.int64), np.ascontiguousarray(
        distances, dtype=np.float32
    )


@lru_cache(maxsize=8)
def _protein_modality(n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return protein coordinates of the RNA cells, their neighbours and distances.

    Every cell keeps its RNA group, so both modalities agree on cell identity,
    while the protein coordinates have their own centers and noise.
    """
    _coordinates, labels = inputs.labelled_coordinates(n_cells)
    rng = np.random.default_rng(inputs.SEED + 1)
    centers = rng.normal(0.0, 3.0, size=(inputs.N_GROUPS, PROTEIN_DIMS))
    values = centers[labels] + rng.normal(size=(n_cells, PROTEIN_DIMS))
    values = values.astype(np.float32)
    indices, distances = _nearest_other_cells(values)
    return values, indices, distances


def _wnn_modalities(n_cells: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return both modalities as WNN loads them: uint32 neighbours, float32 values."""
    rna, _labels = inputs.labelled_coordinates(n_cells)
    rna_indices, _distances = inputs.neighbors(n_cells)
    protein, protein_indices, _protein_distances = _protein_modality(n_cells)
    return [
        ("RNA", rna_indices.astype(np.uint32), rna),
        ("ADT", protein_indices.astype(np.uint32), protein),
    ]


def _unit_vector(values: np.ndarray) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else [0.0] * len(vector)


def _euclidean(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))


def _affinity(distance: float, nearest: float, span: float) -> float:
    """One at the nearest neighbour, decaying over the span to the k-th one."""
    excess = max(distance - nearest, 0.0)
    tolerance = 8.0 * sys.float_info.epsilon * nearest
    if span <= tolerance:
        return 1.0 if excess <= tolerance else 0.0
    return math.exp(-excess / span)


def _wnn_reference_row(
    cell: int,
    modalities: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[int, float], tuple[float, float]]:
    """Rebuild one cell's WNN candidate affinities and modality weights in scalars.

    Candidates are the union of the cell's two neighbour rows. Each modality
    L2-normalizes its coordinates and scores distances over the span from its
    nearest to its k-th own neighbour. A modality's weight is the softmax of
    how much better its own neighbours predict the cell than the other
    modality's neighbours do, and a candidate's affinity is the weighted sum
    of its two modality affinities.
    """
    own = [[int(node) for node in indices[cell]] for _name, indices, _ in modalities]
    pool = sorted(set(own[0]) | set(own[1]))
    units = [
        {node: _unit_vector(values[node]) for node in [cell, *pool]}
        for _name, _indices, values in modalities
    ]
    distances = [
        {node: _euclidean(unit[cell], unit[node]) for node in pool} for unit in units
    ]
    nearest = [
        min(distance[node] for node in rows)
        for distance, rows in zip(distances, own, strict=True)
    ]
    spans = [
        max(distance[node] for node in rows) - low
        for distance, rows, low in zip(distances, own, nearest, strict=True)
    ]

    def predicted(modality: int, rows: list[int]) -> float:
        unit = units[modality]
        mean = [
            sum(unit[node][axis] for node in rows) / len(rows)
            for axis in range(len(unit[cell]))
        ]
        return _affinity(
            _euclidean(unit[cell], mean), nearest[modality], spans[modality]
        )

    scores = []
    for modality in (0, 1):
        within = predicted(modality, own[modality])
        cross = predicted(modality, own[1 - modality])
        scores.append(min(max(within / (cross + 1e-4), 0.0), 200.0))
    rna_weight = 1.0 / (1.0 + math.exp(scores[1] - scores[0]))
    weights = (rna_weight, 1.0 - rna_weight)
    affinities = {
        node: sum(
            weights[modality]
            * _affinity(distances[modality][node], nearest[modality], spans[modality])
            for modality in (0, 1)
        )
        for node in pool
    }
    return affinities, weights


def _checked_rows(n_cells: int) -> list[int]:
    return np.unique(np.linspace(0, n_cells - 1, CHECKED_ROWS).astype(int)).tolist()


def test_wnn_two_modalities(bench) -> None:
    from scarf.neighbors.integration import _wnn_integration_many

    def make(n_cells: int):
        modalities = _wnn_modalities(n_cells)
        return lambda: _wnn_integration_many(modalities, 1)

    def check(n_cells: int, value) -> None:
        graph, modality_weights = value
        modalities = _wnn_modalities(n_cells)
        _coordinates, truth = inputs.labelled_coordinates(n_cells)
        k = inputs.N_NEIGHBORS
        assert graph.shape == (n_cells, n_cells)
        np.testing.assert_array_equal(graph.row, np.repeat(np.arange(n_cells), k))
        columns = graph.col.reshape(n_cells, k).astype(np.int64)
        weights = graph.data.reshape(n_cells, k)
        own_rows = np.hstack([indices for _name, indices, _ in modalities])
        # Every edge is a distinct candidate from the cell's own neighbour rows.
        assert (columns[:, :, None] == own_rows[:, None, :]).any(axis=2).all()
        ordered = np.sort(columns, axis=1)
        assert np.all(ordered[:, 1:] != ordered[:, :-1])
        assert np.all((weights > 0) & (weights <= 1))
        assert modality_weights.shape == (n_cells, 2)
        np.testing.assert_allclose(modality_weights.sum(axis=1), 1.0, rtol=1e-6)
        # Both modalities agree on the drawn groups, so edges stay inside them.
        assert np.mean(truth[columns] == truth[:, None]) > 0.99
        tiny = float(np.finfo(np.float32).tiny)
        for cell in _checked_rows(n_cells):
            affinities, expected_weights = _wnn_reference_row(cell, modalities)
            kept = columns[cell].tolist()
            # The kept candidates are k of the best ones, allowing ties at k.
            kth = sorted(affinities.values(), reverse=True)[k - 1]
            assert all(affinities[node] >= kth - 1e-9 for node in kept)
            np.testing.assert_allclose(
                weights[cell],
                [max(min(affinities[node], 1.0), tiny) for node in kept],
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                modality_weights[cell], expected_weights, rtol=1e-5
            )

    bench("integration.wnn_two_modalities", make, WNN_CELLS, check=check)


def _connectivity(indices: np.ndarray, distances: np.ndarray) -> sparse.csr_matrix:
    """Return the connectivity map of these neighbours as the datastore loads it."""
    from scarf.neighbors.graph import build_connectivity_arrays

    edges, weights = build_connectivity_arrays(
        indices, distances, local_connectivity=1.0, bandwidth=1.5
    )
    n_cells = indices.shape[0]
    return sparse.csr_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells)
    )


def _snn_graphs(n_cells: int) -> list[sparse.csr_matrix]:
    _values, indices, distances = _protein_modality(n_cells)
    return [inputs.connectivity_graph(n_cells), _connectivity(indices, distances)]


def _snn_reference_row(
    row: int,
    graphs: list[sparse.csr_matrix],
) -> dict[int, tuple[float, set[float]]]:
    """Score one row's candidates as edge weight plus shared-neighbour fraction.

    Each candidate keeps its best score over the graphs, with the edge
    weights that reach that score.
    """
    best: dict[int, tuple[float, set[float]]] = {}
    for graph in graphs:

        def neighbours(node: int, graph: sparse.csr_matrix = graph) -> list[int]:
            return graph.indices[graph.indptr[node] : graph.indptr[node + 1]].tolist()

        own = neighbours(row)
        weights = graph.data[graph.indptr[row] : graph.indptr[row + 1]].tolist()
        for node, weight in zip(own, weights, strict=True):
            shared = len(set(own) & set(neighbours(node)))
            score = weight + shared / (len(own) - 1)
            previous = best.get(node)
            if previous is None or score > previous[0]:
                best[node] = (score, {weight})
            elif score == previous[0]:
                previous[1].add(weight)
    return best


def test_snn_merge(bench) -> None:
    from scarf.neighbors.graph import merge_graphs

    def make(n_cells: int):
        graphs = _snn_graphs(n_cells)
        return lambda: merge_graphs(graphs)

    def check(n_cells: int, merged) -> None:
        graphs = _snn_graphs(n_cells)
        _coordinates, truth = inputs.labelled_coordinates(n_cells)
        k = inputs.N_NEIGHBORS
        assert merged.shape == (n_cells, n_cells)
        np.testing.assert_array_equal(merged.row, np.repeat(np.arange(n_cells), k))
        columns = merged.col.reshape(n_cells, k)
        weights = merged.data.reshape(n_cells, k)
        own_rows = np.hstack([graph.indices.reshape(n_cells, k) for graph in graphs])
        assert (columns[:, :, None] == own_rows[:, None, :]).any(axis=2).all()
        assert np.mean(truth[columns] == truth[:, None]) > 0.99
        for row in _checked_rows(n_cells):
            best = _snn_reference_row(row, graphs)
            kept = columns[row].tolist()
            assert len(set(kept)) == k
            scores = [best[node][0] for node in kept]
            # Kept edges run from the best score down, and none dropped beats them.
            assert scores == sorted(scores, reverse=True)
            dropped = [score for node, (score, _) in best.items() if node not in kept]
            assert max(dropped, default=-np.inf) <= min(scores)
            # Each edge keeps the weight of its best-scoring occurrence.
            for node, weight in zip(kept, weights[row].tolist(), strict=True):
                assert weight in best[node][1]

    bench("integration.snn_merge", make, SNN_CELLS, check=check)


@lru_cache(maxsize=5)
def _paris_hierarchy(n_cells: int) -> "ParisHierarchy":
    from scarf.clustering._paris_core import fit_paris_hierarchy

    return fit_paris_hierarchy(inputs.connectivity_graph(n_cells), nthreads=1)


def _leaves_below(hierarchy: "ParisHierarchy", node: int) -> list[int]:
    n_leaves = hierarchy.n_leaves
    stack, leaves = [int(node)], []
    while stack:
        current = stack.pop()
        if current < n_leaves:
            leaves.append(current)
        else:
            stack.extend(hierarchy.children[current - n_leaves].tolist())
    return leaves


def _lowest_merges_partition(
    hierarchy: "ParisHierarchy", n_clusters: int
) -> np.ndarray:
    """Return each leaf's set root after the lowest merges leave ``n_clusters``.

    Merges apply by increasing height, and by row within one height, in a
    disjoint-set forest over the leaves.
    """
    from scipy.cluster.hierarchy import DisjointSet

    n_leaves = hierarchy.n_leaves
    children = np.asarray(hierarchy.children, dtype=np.int64)
    some_leaf = np.arange(2 * n_leaves - 1)
    for merge_index, left in enumerate(children[:, 0].tolist()):
        some_leaf[n_leaves + merge_index] = some_leaf[left]
    order = np.argsort(hierarchy.heights, kind="stable")[: n_leaves - n_clusters]
    disjoint = DisjointSet(range(n_leaves))
    for left, right in children[order].tolist():
        disjoint.merge(int(some_leaf[left]), int(some_leaf[right]))
    return np.asarray([disjoint[leaf] for leaf in range(n_leaves)])


def _same_partition(first: np.ndarray, second: np.ndarray) -> bool:
    pairs = np.unique(np.column_stack([first, second]), axis=0)
    return len(pairs) == len(np.unique(first)) == len(np.unique(second))


def test_paris_fixed_cut(bench) -> None:
    from scarf.clustering.paris import fixed_cut

    def make(n_cells: int):
        hierarchy = _paris_hierarchy(n_cells)
        return lambda: fixed_cut(hierarchy, FIXED_CLUSTERS)

    def check(n_cells: int, labels) -> None:
        hierarchy = _paris_hierarchy(n_cells)
        _coordinates, truth = inputs.labelled_coordinates(n_cells)
        assert labels.shape == (n_cells,)
        np.testing.assert_array_equal(
            np.unique(labels), np.arange(1, FIXED_CLUSTERS + 1)
        )
        # Labels count from 1 by decreasing cluster size.
        assert np.all(np.diff(np.bincount(labels)[1:]) <= 0)
        assert _same_partition(
            labels, _lowest_merges_partition(hierarchy, FIXED_CLUSTERS)
        )
        assert inputs.cluster_purity(labels, truth) > 0.99

    bench("clustering.paris_fixed_cut", make, PARIS_CELLS, check=check)


def _topology(graph: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return the unweighted undirected graph without self-loops."""
    both = (graph + graph.T).tocoo()
    keep = (both.row != both.col) & (both.data > 0)
    return sparse.csr_matrix(
        (np.ones(np.count_nonzero(keep)), (both.row[keep], both.col[keep])),
        shape=graph.shape,
    )


def _plateau_parts(hierarchy: "ParisHierarchy", node: int) -> list[int]:
    """Return the nodes just below the equal-height region that ``node`` heads."""
    n_leaves = hierarchy.n_leaves
    height = hierarchy.heights[node - n_leaves]
    stack, parts = [node], []
    while stack:
        for child in hierarchy.children[stack.pop() - n_leaves].tolist():
            merge_index = child - n_leaves
            if (
                child >= n_leaves
                and not hierarchy.synthetic_joins[merge_index]
                and hierarchy.heights[merge_index] == height
            ):
                stack.append(child)
            else:
                parts.append(child)
    return parts


def _split_gain(
    hierarchy: "ParisHierarchy",
    topology: sparse.csr_matrix,
    parts: list[int],
) -> float:
    """Newman-Girvan modularity gained by splitting the union of parts into them."""
    owner = np.full(hierarchy.n_leaves, -1)
    for index, part in enumerate(parts):
        owner[_leaves_below(hierarchy, part)] = index
    members = np.flatnonzero(owner >= 0)
    block = topology[members].tocoo()
    sources = owner[members[block.row]]
    targets = owner[block.col]
    cross_edges = np.count_nonzero((targets >= 0) & (targets != sources)) / 2
    degrees = np.diff(topology.indptr).astype(np.float64)
    volumes = np.bincount(
        owner[members], weights=degrees[members], minlength=len(parts)
    )
    two_m = degrees.sum()
    return (volumes.sum() ** 2 - np.sum(volumes**2)) / two_m**2 - (
        2 * cross_edges / two_m
    )


def test_paris_adaptive_cut(bench) -> None:
    from scarf.clustering._paris_modularity import modularity_split_gains
    from scarf.clustering.paris_multiscale import (
        adaptive_cut,
        collapse_equal_height_plateaus,
    )

    # The datastore default: one more cell than the graph's neighbour count.
    min_cluster_size = inputs.N_NEIGHBORS + 1

    def make(n_cells: int):
        hierarchy = _paris_hierarchy(n_cells)
        graph = inputs.connectivity_graph(n_cells)

        # A fit collapses plateaus, and the default cut then gates and cuts.
        def call():
            forest = collapse_equal_height_plateaus(hierarchy)
            gains = modularity_split_gains(hierarchy, forest, graph)
            result = adaptive_cut(
                hierarchy,
                min_cluster_size,
                plateau_forest=forest,
                split_gate=gains,
            )
            return forest, gains, result

        return call

    def check(n_cells: int, value) -> None:
        forest, gains, result = value
        hierarchy = _paris_hierarchy(n_cells)
        _coordinates, truth = inputs.labelled_coordinates(n_cells)
        n_leaves = hierarchy.n_leaves
        children = np.asarray(hierarchy.children, dtype=np.int64)
        merges = np.arange(n_leaves - 1)
        parents = np.full(2 * n_leaves - 1, -1)
        parents[children[:, 0]] = n_leaves + merges
        parents[children[:, 1]] = n_leaves + merges
        parent_merge = parents[n_leaves:] - n_leaves
        has_parent = parent_merge >= 0
        safe_parent = np.where(has_parent, parent_merge, 0)
        # An event heads each finite merge whose parent is missing, synthetic,
        # or at a different height.
        heads = ~hierarchy.synthetic_joins & (
            ~has_parent
            | hierarchy.synthetic_joins[safe_parent]
            | (hierarchy.heights[safe_parent] != hierarchy.heights)
        )
        np.testing.assert_array_equal(
            forest.representatives, n_leaves + np.flatnonzero(heads)
        )
        np.testing.assert_array_equal(forest.sizes, hierarchy.sizes[heads])

        topology = _topology(inputs.connectivity_graph(n_cells))
        n_events = forest.representatives.size
        sampled = set(np.linspace(0, n_events - 1, 16).astype(int).tolist())
        sampled.update(int(root) for root in forest.component_roots if root >= 0)
        for event in sorted(sampled):
            parts = _plateau_parts(hierarchy, int(forest.representatives[event]))
            refs = forest.child_refs[
                forest.child_offsets[event] : forest.child_offsets[event + 1]
            ].tolist()
            assert sorted(parts) == sorted(
                -ref - 1 if ref < 0 else int(forest.representatives[ref])
                for ref in refs
            )
            np.testing.assert_allclose(
                gains[event],
                _split_gain(hierarchy, topology, parts),
                rtol=1e-9,
                atol=1e-12,
            )

        # Each drawn group is a component and the gate keeps it whole.
        assert result.n_clusters == inputs.N_GROUPS
        assert inputs.cluster_purity(result.labels, truth) > 0.99
        for diagnostic in result.diagnostics:
            np.testing.assert_array_equal(
                np.flatnonzero(result.labels == diagnostic.label),
                np.sort(_leaves_below(hierarchy, diagnostic.selected_node)),
            )

    bench("clustering.paris_adaptive_cut", make, PARIS_CELLS, check=check)


def _coalesced_reference(
    dendrogram: np.ndarray, clusters: np.ndarray
) -> tuple[set[int], set[tuple[int, int]], dict[int, int]]:
    """Return the nodes, edges, and holding-node clusters of the coalesced tree.

    Labels propagate up the linkage: a node keeps a label both children carry
    and is mixed otherwise. A cluster's holding node is its highest node with
    that label, and the tree keeps holding nodes and the mixed nodes above.
    """
    n_leaves = clusters.size
    children = dendrogram[:, :2].astype(np.int64)
    labels = np.zeros(2 * n_leaves - 1, dtype=np.int64)
    labels[:n_leaves] = clusters
    for merge_index, (left, right) in enumerate(children.tolist()):
        if labels[left] == labels[right]:
            labels[n_leaves + merge_index] = labels[left]
    parents = np.full(2 * n_leaves - 1, -1, dtype=np.int64)
    parents[children.ravel()] = np.repeat(np.arange(n_leaves, 2 * n_leaves - 1), 2)
    parent_labels = np.where(parents >= 0, labels[np.maximum(parents, 0)], 0)
    holding = (labels > 0) & (parent_labels == 0)
    nodes = set(np.flatnonzero(holding | (labels == 0)).tolist())
    edges = {(int(parents[node]), node) for node in nodes if parents[node] >= 0}
    return nodes, edges, {node: int(labels[node]) for node in np.flatnonzero(holding)}


def test_coalesced_cluster_tree(bench) -> None:
    from scarf.clustering.cluster_tree import CoalesceTree, make_digraph
    from scarf.clustering.paris import fixed_cut, hierarchy_to_dendrogram

    def tree_inputs(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
        # The cluster-tree plot reads a stored dendrogram and int32 cut labels.
        hierarchy = _paris_hierarchy(n_cells)
        dendrogram = hierarchy_to_dendrogram(hierarchy, compatibility=True)
        clusters = fixed_cut(hierarchy, FIXED_CLUSTERS).astype(np.int32)
        return dendrogram, clusters

    def make(n_cells: int):
        dendrogram, clusters = tree_inputs(n_cells)
        return lambda: CoalesceTree(make_digraph(dendrogram), clusters)

    def check(n_cells: int, tree) -> None:
        dendrogram, clusters = tree_inputs(n_cells)
        nodes, edges, holding = _coalesced_reference(dendrogram, clusters)
        assert len(holding) == FIXED_CLUSTERS
        assert set(tree.nodes) == nodes
        assert set(tree.edges) == edges
        assert {
            node: data["partition_id"]
            for node, data in tree.nodes(data=True)
            if "partition_id" in data
        } == holding
        for node, data in tree.nodes(data=True):
            expected = 0 if node < n_cells else int(dendrogram[node - n_cells, 3])
            assert data["nleaves"] == expected

    bench("clustering.coalesced_cluster_tree", make, PARIS_CELLS, check=check)
