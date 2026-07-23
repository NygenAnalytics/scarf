from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from ._paris_core import ParisHierarchy, canonicalize_paris_graph
from .paris_multiscale import PlateauForest, _validate_hierarchy


def _freeze_arrays(*arrays: np.ndarray) -> None:
    for values in arrays:
        values.setflags(write=False)


@dataclass(frozen=True, slots=True)
class TopologyStatistics:
    """Raw topology statistics aligned to leaves and binary merges."""

    leaf_degrees: np.ndarray
    lca_edge_counts: np.ndarray
    component_edge_counts: np.ndarray

    def __post_init__(self) -> None:
        for name, values in (
            ("leaf_degrees", self.leaf_degrees),
            ("lca_edge_counts", self.lca_edge_counts),
            ("component_edge_counts", self.component_edge_counts),
        ):
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{name} must be a one-dimensional integer array")
            if np.any(values < 0):
                raise ValueError(f"{name} must be non-negative")
        _freeze_arrays(
            self.leaf_degrees,
            self.lca_edge_counts,
            self.component_edge_counts,
        )


@dataclass(frozen=True, slots=True)
class PlateauModularityStatistics:
    """Configuration-null statistics aligned to plateau events."""

    cross_edges: np.ndarray
    volumes: np.ndarray

    def __post_init__(self) -> None:
        n_events = self.cross_edges.size
        for name, values in (
            ("cross_edges", self.cross_edges),
            ("volumes", self.volumes),
        ):
            if values.shape != (n_events,) or not np.issubdtype(
                values.dtype,
                np.integer,
            ):
                raise TypeError(f"{name} must be an integer array over events")
            if np.any(values < 0):
                raise ValueError(f"{name} must be non-negative")
        _freeze_arrays(
            self.cross_edges,
            self.volumes,
        )


@njit(cache=True, nogil=True)
def _csr_symmetry_error(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> int:
    n_vertices = indptr.size - 1
    for source in range(n_vertices):
        for offset in range(indptr[source], indptr[source + 1]):
            target = int(indices[offset])
            if target == source:
                return 1
            lower = int(indptr[target])
            upper = int(indptr[target + 1])
            while lower < upper:
                middle = (lower + upper) // 2
                candidate = int(indices[middle])
                if candidate < source:
                    lower = middle + 1
                else:
                    upper = middle
            if lower >= indptr[target + 1] or int(indices[lower]) != source:
                return 2
            if data[lower] != data[offset]:
                return 3
    return 0


def _validate_canonical_csr(graph: csr_matrix, n_vertices: int | None = None) -> None:
    if not isinstance(graph, csr_matrix):
        raise TypeError("graph must be a scipy.sparse.csr_matrix")
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("canonical graph must be square")
    if graph.shape[0] < 1:
        raise ValueError("canonical graph must contain at least one vertex")
    if n_vertices is not None and graph.shape != (n_vertices, n_vertices):
        raise ValueError("canonical graph and hierarchy have different leaf counts")
    if not graph.has_canonical_format or not graph.has_sorted_indices:
        raise ValueError("graph must be a sorted canonical CSR without duplicates")
    if not np.isfinite(graph.data).all() or np.any(graph.data <= 0):
        raise ValueError("canonical graph entries must be finite and positive")
    symmetry_error = _csr_symmetry_error(
        graph.indptr,
        graph.indices,
        graph.data,
    )
    if symmetry_error == 1:
        raise ValueError("canonical graph must not contain self-loops")
    if symmetry_error == 2:
        raise ValueError("canonical graph must be structurally symmetric")
    if symmetry_error == 3:
        raise ValueError("canonical graph must be exactly symmetric")
    if graph.nnz % 2:
        raise ValueError("canonical graph must contain paired undirected entries")


def _leaf_component_ids(hierarchy: ParisHierarchy) -> np.ndarray:
    _validate_hierarchy(hierarchy)
    n_leaves = hierarchy.n_leaves
    if not np.issubdtype(hierarchy.component_roots.dtype, np.integer):
        raise TypeError("component roots must contain integers")

    n_nodes = 2 * n_leaves - 1
    owners = np.full(n_nodes, -1, dtype=np.int64)
    stack = np.empty(n_nodes, dtype=np.int64)
    for component, raw_root in enumerate(hierarchy.component_roots):
        root = int(raw_root)
        if root < 0 or root >= n_nodes:
            raise ValueError("component root lies outside the hierarchy")
        stack_size = 1
        stack[0] = root
        while stack_size:
            stack_size -= 1
            node = int(stack[stack_size])
            if owners[node] >= 0:
                raise ValueError("finite hierarchy components overlap")
            if node >= n_leaves and hierarchy.synthetic_joins[node - n_leaves]:
                raise ValueError("component roots cannot contain synthetic joins")
            owners[node] = component
            if node < n_leaves:
                continue
            left, right = hierarchy.children[node - n_leaves]
            stack[stack_size] = int(left)
            stack[stack_size + 1] = int(right)
            stack_size += 2

    finite_nodes = np.ones(n_nodes, dtype=bool)
    finite_nodes[n_leaves:] = ~hierarchy.synthetic_joins
    if np.any(owners[finite_nodes] < 0) or np.any(owners[~finite_nodes] >= 0):
        raise ValueError("component roots do not cover exactly the finite hierarchy")
    owner_dtype = (
        np.int32
        if hierarchy.component_roots.size <= np.iinfo(np.int32).max
        else np.int64
    )
    return owners[:n_leaves].astype(owner_dtype, copy=True)


@njit(cache=True, nogil=True, inline="always")
def _union_find_root(parents: np.ndarray, node: int) -> int:
    root = node
    while parents[root] != root:
        root = parents[root]
    while parents[node] != node:
        parent = parents[node]
        parents[node] = root
        node = parent
    return root


@njit(cache=True, nogil=True, inline="always")
def _union_sets(
    parents: np.ndarray,
    ranks: np.ndarray,
    first: int,
    second: int,
) -> int:
    first_root = int(_union_find_root(parents, first))
    second_root = int(_union_find_root(parents, second))
    if first_root == second_root:
        return first_root
    if ranks[first_root] < ranks[second_root]:
        parents[first_root] = second_root
        return second_root
    parents[second_root] = first_root
    if ranks[first_root] == ranks[second_root]:
        ranks[first_root] += 1
    return first_root


@njit(cache=True, nogil=True)
def _offline_lca_edge_counts(
    indptr: np.ndarray,
    indices: np.ndarray,
    children: np.ndarray,
    component_roots: np.ndarray,
    leaf_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_leaves = indptr.size - 1
    n_nodes = 2 * n_leaves - 1
    lca_edge_counts = np.zeros(n_leaves - 1, dtype=np.int64)
    component_edge_counts = np.zeros(component_roots.size, dtype=np.int64)
    parents = np.full(n_nodes, -1, dtype=np.int64)
    ranks = np.zeros(n_nodes, dtype=np.uint8)
    ancestors = np.full(n_nodes, -1, dtype=np.int64)
    black = np.zeros(n_nodes, dtype=np.bool_)
    stack_nodes = np.empty(n_nodes, dtype=np.int64)
    stack_states = np.empty(n_nodes, dtype=np.uint8)

    for component in range(component_roots.size):
        stack_size = 1
        stack_nodes[0] = component_roots[component]
        stack_states[0] = 0
        while stack_size:
            stack_size -= 1
            node = int(stack_nodes[stack_size])
            state = int(stack_states[stack_size])
            if state == 0:
                parents[node] = node
                ancestors[node] = node
                if node < n_leaves:
                    black[node] = True
                    for offset in range(indptr[node], indptr[node + 1]):
                        neighbor = int(indices[offset])
                        if not black[neighbor]:
                            continue
                        if leaf_components[neighbor] != component:
                            return lca_edge_counts, component_edge_counts, 1
                        root = _union_find_root(parents, neighbor)
                        lca = int(ancestors[root])
                        if lca < n_leaves:
                            return lca_edge_counts, component_edge_counts, 2
                        lca_edge_counts[lca - n_leaves] += 1
                        component_edge_counts[component] += 1
                    continue

                left = int(children[node - n_leaves, 0])
                stack_nodes[stack_size] = node
                stack_states[stack_size] = 1
                stack_nodes[stack_size + 1] = left
                stack_states[stack_size + 1] = 0
                stack_size += 2
            elif state == 1:
                left = int(children[node - n_leaves, 0])
                root = _union_sets(parents, ranks, node, left)
                ancestors[root] = node
                right = int(children[node - n_leaves, 1])
                stack_nodes[stack_size] = node
                stack_states[stack_size] = 2
                stack_nodes[stack_size + 1] = right
                stack_states[stack_size + 1] = 0
                stack_size += 2
            else:
                right = int(children[node - n_leaves, 1])
                root = _union_sets(parents, ranks, node, right)
                ancestors[root] = node
                black[node] = True

    return lca_edge_counts, component_edge_counts, 0


def collect_topology_statistics(
    canonical_graph: csr_matrix,
    hierarchy: ParisHierarchy,
) -> TopologyStatistics:
    """Collect deterministic unweighted topology statistics."""
    _validate_canonical_csr(canonical_graph, hierarchy.n_leaves)
    leaf_components = _leaf_component_ids(hierarchy)
    leaf_degrees = np.diff(canonical_graph.indptr.astype(np.int64, copy=False)).astype(
        np.int64, copy=False
    )
    (
        lca_edge_counts,
        component_edge_counts,
        error_code,
    ) = _offline_lca_edge_counts(
        canonical_graph.indptr,
        canonical_graph.indices,
        hierarchy.children,
        hierarchy.component_roots,
        leaf_components,
    )
    if error_code == 1:
        raise ValueError("canonical graph contains an edge across hierarchy components")
    if error_code == 2:
        raise RuntimeError("offline LCA resolved an edge to a leaf")
    expected_edges = canonical_graph.nnz // 2
    if int(lca_edge_counts.sum()) != expected_edges:
        raise RuntimeError("offline LCA did not account for every topology edge")
    if int(leaf_degrees.sum()) != 2 * expected_edges:
        raise RuntimeError("leaf degrees do not account for every topology edge")
    return TopologyStatistics(
        leaf_degrees=np.ascontiguousarray(leaf_degrees),
        lca_edge_counts=lca_edge_counts,
        component_edge_counts=component_edge_counts,
    )


def _event_components(
    hierarchy: ParisHierarchy,
    forest: PlateauForest,
) -> np.ndarray:
    if forest.n_leaves != hierarchy.n_leaves:
        raise ValueError("plateau forest and hierarchy have different leaf counts")
    if forest.component_roots.size != hierarchy.component_roots.size:
        raise ValueError("plateau forest has the wrong number of components")

    n_events = forest.representatives.size
    n_leaves = hierarchy.n_leaves
    incoming = np.zeros(n_events, dtype=np.int64)
    for event in range(n_events):
        representative = int(forest.representatives[event])
        if (
            representative < n_leaves
            or representative >= 2 * n_leaves - 1
            or hierarchy.synthetic_joins[representative - n_leaves]
        ):
            raise ValueError("plateau representative must be a finite internal node")
        if int(forest.sizes[event]) != int(hierarchy.sizes[representative - n_leaves]):
            raise ValueError("plateau event size does not match its representative")
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0:
                leaf = -child_ref - 1
                if leaf < 0 or leaf >= n_leaves:
                    raise ValueError("plateau child leaf lies outside the hierarchy")
                continue
            if child_ref >= event:
                raise ValueError("plateau child events must precede their parent")
            incoming[child_ref] += 1
            if int(forest.parent_events[child_ref]) != event:
                raise ValueError("plateau parent and child references disagree")

    for event, parent in enumerate(forest.parent_events):
        if int(parent) < 0:
            if incoming[event] != 0:
                raise ValueError("plateau root has an incoming event reference")
        elif incoming[event] != 1:
            raise ValueError("non-root plateau event must have one parent")

    component_ids = np.full(n_events, -1, dtype=np.int32)
    stack = np.empty(max(1, n_events), dtype=np.int64)
    for component, (forest_root, hierarchy_root) in enumerate(
        zip(
            forest.component_roots,
            hierarchy.component_roots,
            strict=True,
        )
    ):
        root_ref = int(forest_root)
        expected_raw_root = int(hierarchy_root)
        actual_raw_root = (
            -root_ref - 1 if root_ref < 0 else int(forest.representatives[root_ref])
        )
        if actual_raw_root != expected_raw_root:
            raise ValueError("plateau and hierarchy component roots disagree")
        if root_ref < 0:
            continue
        if root_ref >= n_events or int(forest.parent_events[root_ref]) >= 0:
            raise ValueError("plateau component root is not a root event")
        stack_size = 1
        stack[0] = root_ref
        while stack_size:
            stack_size -= 1
            event = int(stack[stack_size])
            if component_ids[event] >= 0:
                raise ValueError("plateau components overlap")
            component_ids[event] = component
            for offset in range(
                int(forest.child_offsets[event]),
                int(forest.child_offsets[event + 1]),
            ):
                child_ref = int(forest.child_refs[offset])
                if child_ref >= 0:
                    stack[stack_size] = child_ref
                    stack_size += 1
    if np.any(component_ids < 0):
        raise ValueError("plateau component roots do not cover every event")
    return component_ids


def aggregate_plateau_statistics(
    hierarchy: ParisHierarchy,
    forest: PlateauForest,
    topology: TopologyStatistics,
) -> PlateauModularityStatistics:
    """Aggregate edge counts and degree volumes onto plateau events."""
    leaf_components = _leaf_component_ids(hierarchy)
    component_ids = _event_components(hierarchy, forest)
    n_leaves = hierarchy.n_leaves
    n_nodes = 2 * n_leaves - 1
    if topology.leaf_degrees.shape != (n_leaves,):
        raise ValueError("leaf degree statistics have the wrong length")
    if topology.lca_edge_counts.shape != (n_leaves - 1,):
        raise ValueError("LCA edge statistics have the wrong length")
    if topology.component_edge_counts.shape != hierarchy.component_roots.shape:
        raise ValueError("component edge statistics have the wrong length")
    if int(topology.lca_edge_counts.sum()) != int(topology.component_edge_counts.sum()):
        raise ValueError("raw edge statistics disagree on the total edge count")
    if np.any(
        topology.lca_edge_counts[np.asarray(hierarchy.synthetic_joins, dtype=bool)] != 0
    ):
        raise ValueError("synthetic joins cannot contain topology edges")

    raw_sizes = np.ones(n_nodes, dtype=np.int64)
    raw_volumes = np.zeros(n_nodes, dtype=np.int64)
    raw_volumes[:n_leaves] = topology.leaf_degrees
    raw_internal_edges = np.zeros(n_nodes, dtype=np.int64)
    for merge_index in range(n_leaves - 1):
        node = n_leaves + merge_index
        left = int(hierarchy.children[merge_index, 0])
        right = int(hierarchy.children[merge_index, 1])
        raw_sizes[node] = raw_sizes[left] + raw_sizes[right]
        raw_volumes[node] = raw_volumes[left] + raw_volumes[right]
        raw_internal_edges[node] = (
            raw_internal_edges[left]
            + raw_internal_edges[right]
            + topology.lca_edge_counts[merge_index]
        )
        if int(raw_sizes[node]) != int(hierarchy.sizes[merge_index]):
            raise ValueError("hierarchy sizes changed during event aggregation")

    for component, root in enumerate(hierarchy.component_roots):
        raw_root = int(root)
        edge_count = int(topology.component_edge_counts[component])
        if int(raw_internal_edges[raw_root]) != edge_count:
            raise ValueError("component root does not contain its declared edges")
        if int(raw_volumes[raw_root]) != 2 * edge_count:
            raise ValueError("component volume is not twice its edge count")
        if raw_root < n_leaves and int(leaf_components[raw_root]) != component:
            raise ValueError("isolated component ownership is inconsistent")

    n_events = forest.representatives.size
    cross_edges = np.empty(n_events, dtype=np.int64)
    volumes = np.empty(n_events, dtype=np.int64)

    for event, raw_representative in enumerate(forest.representatives):
        raw_node = int(raw_representative)
        size = int(raw_sizes[raw_node])
        volume = int(raw_volumes[raw_node])
        internal_edge_count = int(raw_internal_edges[raw_node])

        child_size_sum = 0
        child_volume_sum = 0
        child_internal_edge_sum = 0
        component = int(component_ids[event])

        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            child_node = (
                -child_ref - 1
                if child_ref < 0
                else int(forest.representatives[child_ref])
            )
            child_component = (
                int(leaf_components[child_node])
                if child_node < n_leaves
                else int(component_ids[child_ref])
            )
            if child_component != component:
                raise ValueError("a plateau event crosses hierarchy components")
            child_size = int(raw_sizes[child_node])
            child_volume = int(raw_volumes[child_node])
            child_internal_edge = int(raw_internal_edges[child_node])

            child_size_sum += child_size
            child_volume_sum += child_volume
            child_internal_edge_sum += child_internal_edge

        if child_size_sum != size or child_volume_sum != volume:
            raise ValueError("plateau children do not partition their event")

        cross_edges[event] = internal_edge_count - child_internal_edge_sum
        volumes[event] = volume

    return PlateauModularityStatistics(
        cross_edges=cross_edges,
        volumes=volumes,
    )


def modularity_split_gains(
    hierarchy: ParisHierarchy,
    forest: PlateauForest,
    graph: csr_matrix,
) -> np.ndarray:
    """Return the configuration-null modularity gain of splitting each event.

    The gain for event ``p`` is the change in Newman-Girvan modularity when the
    single block ``p`` is replaced by its immediate plateau children, using
    unweighted graph topology. A positive value means the split beats the
    degree-preserving null. A non-positive value means the split only fragments
    structure the null already explains. Leaf children contribute their degree
    as volume and no internal edges. The graph is canonicalized to the additive
    undirected form Paris fits.
    """
    topology = collect_topology_statistics(
        canonicalize_paris_graph(graph),
        hierarchy,
    )
    events = aggregate_plateau_statistics(hierarchy, forest, topology)
    leaf_degrees = topology.leaf_degrees.astype(np.float64, copy=False)
    two_m = float(leaf_degrees.sum())
    n_events = forest.representatives.size
    gains = np.zeros(n_events, dtype=np.float64)
    if two_m <= 0:
        return gains

    volumes = events.volumes.astype(np.float64, copy=False)
    cross_edges = events.cross_edges.astype(np.float64, copy=False)
    for event in range(n_events):
        parent_volume = float(volumes[event])
        child_volume_square_sum = 0.0
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0:
                child_volume = float(leaf_degrees[-child_ref - 1])
            else:
                child_volume = float(volumes[child_ref])
            child_volume_square_sum += child_volume * child_volume
        attraction = 2.0 * float(cross_edges[event]) / two_m
        repulsion = (parent_volume * parent_volume - child_volume_square_sum) / (
            two_m * two_m
        )
        gains[event] = repulsion - attraction
    return gains
