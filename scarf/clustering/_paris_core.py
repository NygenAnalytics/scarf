import time
from dataclasses import dataclass

import numpy as np
from numba import (
    config,
    get_num_threads,
    njit,
    prange,
    set_num_threads,
)
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ParisRoundDiagnostics:
    """Measurements for one reciprocal-neighbor contraction round."""

    component: int
    round_index: int
    active_vertices: int
    active_edges: int
    merges: int
    scan_seconds: float
    sort_seconds: float
    contraction_seconds: float
    contraction_remap_seconds: float
    contraction_filter_seconds: float
    contraction_build_seconds: float
    contraction_cleanup_seconds: float


@dataclass(frozen=True)
class ParisFitDiagnostics:
    """Measurements from a Paris hierarchy fit."""

    preprocessing_seconds: float
    component_seconds: float
    fit_seconds: float
    rounds: tuple[ParisRoundDiagnostics, ...]


@dataclass(frozen=True)
class ParisHierarchy:
    """Flat-array representation of a Paris hierarchy."""

    children: np.ndarray
    heights: np.ndarray
    sizes: np.ndarray
    component_roots: np.ndarray
    synthetic_joins: np.ndarray
    n_leaves: int
    total_weight: float
    diagnostics: ParisFitDiagnostics | None = None

    def __post_init__(self) -> None:
        n_merges = self.n_leaves - 1
        if self.children.shape != (n_merges, 2):
            raise ValueError("children must have shape (n_leaves - 1, 2)")
        for name, values in (
            ("heights", self.heights),
            ("sizes", self.sizes),
            ("synthetic_joins", self.synthetic_joins),
        ):
            if values.shape != (n_merges,):
                raise ValueError(f"{name} must have length n_leaves - 1")
        for values in (
            self.children,
            self.heights,
            self.sizes,
            self.component_roots,
            self.synthetic_joins,
        ):
            values.setflags(write=False)


@njit(cache=True, nogil=True, parallel=True)
def _nearest_neighbors(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    volumes: np.ndarray,
    logical_ids: np.ndarray,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_vertices = volumes.shape[0]
    nearest = np.full(n_vertices, -1, dtype=np.int64)
    between_weights = np.zeros(n_vertices, dtype=np.float64)
    for worker in prange(n_workers):
        for row in range(worker, n_vertices, n_workers):
            best_neighbor = -1
            best_logical_id = np.iinfo(np.int64).max
            best_similarity = -np.inf
            row_volume = volumes[row]
            for offset in range(indptr[row], indptr[row + 1]):
                neighbor = indices[offset]
                weight = data[offset]
                if neighbor == row or weight <= 0:
                    continue
                similarity = weight / (row_volume * volumes[neighbor])
                logical_id = logical_ids[neighbor]
                if similarity > best_similarity or (
                    similarity == best_similarity and logical_id < best_logical_id
                ):
                    best_neighbor = neighbor
                    best_logical_id = logical_id
                    best_similarity = similarity
                    between_weights[row] = weight
            nearest[row] = best_neighbor
    return nearest, between_weights


@njit(cache=True, nogil=True)
def _contraction_layout(
    indptr: np.ndarray,
    mapping: np.ndarray,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_old_vertices = mapping.size
    member_counts = np.zeros(n_vertices, dtype=mapping.dtype)
    for old_vertex in range(n_old_vertices):
        member_counts[mapping[old_vertex]] += 1

    member_offsets = np.empty(n_vertices + 1, dtype=mapping.dtype)
    member_offsets[0] = 0
    for row in range(n_vertices):
        member_offsets[row + 1] = member_offsets[row] + member_counts[row]

    cursors = member_offsets[:-1].copy()
    members = np.empty(n_old_vertices, dtype=mapping.dtype)
    for old_vertex in range(n_old_vertices):
        group = mapping[old_vertex]
        members[cursors[group]] = old_vertex
        cursors[group] += 1

    workspace_offsets = np.empty(n_vertices + 1, dtype=indptr.dtype)
    workspace_offsets[0] = 0
    for row in range(n_vertices):
        degree = 0
        for position in range(member_offsets[row], member_offsets[row + 1]):
            old_vertex = members[position]
            degree += indptr[old_vertex + 1] - indptr[old_vertex]
        workspace_offsets[row + 1] = workspace_offsets[row] + degree

    return member_offsets, members, workspace_offsets


@njit(cache=True, nogil=True, parallel=True)
def _aggregate_contracted_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    mapping: np.ndarray,
    member_offsets: np.ndarray,
    members: np.ndarray,
    workspace_offsets: np.ndarray,
    n_vertices: int,
    n_workers: int,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workspace_size = workspace_offsets[-1]
    workspace_indices = np.empty(workspace_size, dtype=indices.dtype)
    workspace_data = np.empty(workspace_size, dtype=data.dtype)
    unique_counts = np.zeros(n_vertices, dtype=indices.dtype)

    for worker in prange(n_workers):
        for row in range(worker, n_vertices, n_workers):
            output_offset = workspace_offsets[row]
            unique_count = 0
            for position in range(member_offsets[row], member_offsets[row + 1]):
                old_vertex = members[position]
                for edge in range(indptr[old_vertex], indptr[old_vertex + 1]):
                    destination = mapping[indices[edge]]
                    if destination == row:
                        continue
                    value = values[worker, destination]
                    if np.isnan(value):
                        values[worker, destination] = data[edge]
                        workspace_indices[output_offset + unique_count] = destination
                        unique_count += 1
                    else:
                        values[worker, destination] = value + data[edge]

            if unique_count:
                workspace_indices[output_offset : output_offset + unique_count].sort()
                for position in range(unique_count):
                    destination = workspace_indices[output_offset + position]
                    workspace_data[output_offset + position] = values[
                        worker,
                        destination,
                    ]
                    values[worker, destination] = np.nan
            unique_counts[row] = unique_count

    return workspace_indices, workspace_data, unique_counts


@njit(cache=True, nogil=True)
def _compact_contracted_rows(
    workspace_offsets: np.ndarray,
    workspace_indices: np.ndarray,
    workspace_data: np.ndarray,
    unique_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vertices = unique_counts.size
    indptr = np.empty(n_vertices + 1, dtype=workspace_offsets.dtype)
    indptr[0] = 0
    for row in range(n_vertices):
        indptr[row + 1] = indptr[row] + unique_counts[row]

    for row in range(n_vertices):
        source_offset = workspace_offsets[row]
        target_offset = indptr[row]
        count = unique_counts[row]
        workspace_indices[target_offset : target_offset + count] = workspace_indices[
            source_offset : source_offset + count
        ]
        workspace_data[target_offset : target_offset + count] = workspace_data[
            source_offset : source_offset + count
        ]
    return (
        indptr,
        workspace_indices[: indptr[-1]],
        workspace_data[: indptr[-1]],
    )


def canonicalize_paris_graph(graph: spmatrix) -> csr_matrix:
    """Build the canonical additive, loop-free Paris graph."""
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Paris requires a square adjacency matrix")
    if graph.shape[0] < 2:
        raise ValueError("Paris requires at least two vertices")

    directed = csr_matrix(graph, dtype=np.float64, copy=True)
    directed.sum_duplicates()
    if not np.isfinite(directed.data).all():
        raise ValueError("Paris graph weights must be finite")
    if np.any(directed.data < 0):
        raise ValueError("Paris graph weights must be non-negative")

    additive = csr_matrix(directed + directed.T, dtype=np.float64)
    additive.sum_duplicates()
    additive.setdiag(0)
    additive.eliminate_zeros()
    additive.sort_indices()
    if not np.isfinite(additive.data).all():
        raise ValueError("Paris graph weights overflowed during symmetrization")
    return additive


def _reciprocal_pairs(
    nearest: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.arange(nearest.size, dtype=np.int64)
    valid = nearest >= 0
    reciprocal = np.zeros(nearest.size, dtype=bool)
    reciprocal[valid] = nearest[nearest[valid]] == vertices[valid]
    first = np.flatnonzero(reciprocal & (vertices < nearest))
    second = nearest[first]
    return first, second


@njit(cache=True, nogil=True)
def _prepare_contraction_groups(
    logical_ids: np.ndarray,
    volumes: np.ndarray,
    sizes: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    new_ids: np.ndarray,
    mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vertices = logical_ids.size
    merge_count = first.size
    unmerged_count = n_vertices - 2 * merge_count
    new_count = n_vertices - merge_count
    pair_by_vertex = np.full(n_vertices, -1, dtype=mapping.dtype)
    for pair in range(merge_count):
        pair_by_vertex[first[pair]] = pair
        pair_by_vertex[second[pair]] = pair

    new_logical_ids = np.empty(new_count, dtype=logical_ids.dtype)
    new_volumes = np.empty(new_count, dtype=volumes.dtype)
    new_sizes = np.empty(new_count, dtype=sizes.dtype)
    output = 0
    for vertex in range(n_vertices):
        if pair_by_vertex[vertex] < 0:
            mapping[vertex] = output
            new_logical_ids[output] = logical_ids[vertex]
            new_volumes[output] = volumes[vertex]
            new_sizes[output] = sizes[vertex]
            output += 1

    if output != unmerged_count:
        raise RuntimeError("Paris contraction group assembly failed")
    for pair in range(merge_count):
        output = unmerged_count + pair
        left = first[pair]
        right = second[pair]
        mapping[left] = output
        mapping[right] = output
        new_logical_ids[output] = new_ids[pair]
        new_volumes[output] = volumes[left] + volumes[right]
        new_sizes[output] = sizes[left] + sizes[right]

    return new_logical_ids, new_volumes, new_sizes


def _contract_graph(
    graph: csr_matrix,
    mapping: np.ndarray,
    n_vertices: int,
    values: np.ndarray,
) -> tuple[csr_matrix, float, float, float, float]:
    remap_start = time.perf_counter()
    member_offsets, members, workspace_offsets = _contraction_layout(
        graph.indptr,
        mapping,
        n_vertices,
    )
    remap_seconds = time.perf_counter() - remap_start

    filter_start = time.perf_counter()
    n_workers = min(values.shape[0], n_vertices)
    workspace_indices, workspace_data, unique_counts = _aggregate_contracted_rows(
        graph.indptr,
        graph.indices,
        graph.data,
        mapping,
        member_offsets,
        members,
        workspace_offsets,
        n_vertices,
        n_workers,
        values,
    )
    filter_seconds = time.perf_counter() - filter_start

    build_start = time.perf_counter()
    indptr, indices, data = _compact_contracted_rows(
        workspace_offsets,
        workspace_indices,
        workspace_data,
        unique_counts,
    )
    build_seconds = time.perf_counter() - build_start

    cleanup_start = time.perf_counter()
    contracted = csr_matrix(
        (data, indices, indptr),
        shape=(n_vertices, n_vertices),
        copy=False,
    )
    cleanup_seconds = time.perf_counter() - cleanup_start
    return (
        contracted,
        remap_seconds,
        filter_seconds,
        build_seconds,
        cleanup_seconds,
    )


def _fit_component(
    graph: csr_matrix,
    leaf_ids: np.ndarray,
    *,
    component: int,
    next_node: int,
    total_weight: float,
    round_diagnostics: list[ParisRoundDiagnostics],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int, int]:
    logical_ids = np.asarray(leaf_ids, dtype=np.int64)
    volumes = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64, copy=False)
    sizes = np.ones(logical_ids.size, dtype=np.int64)
    contraction_values = np.full(
        (min(get_num_threads(), logical_ids.size), logical_ids.size),
        np.nan,
        dtype=np.float64,
    )
    child_blocks: list[np.ndarray] = []
    height_blocks: list[np.ndarray] = []
    size_blocks: list[np.ndarray] = []
    round_index = 0

    while logical_ids.size > 1:
        active_edges = graph.nnz
        scan_start = time.perf_counter()
        n_workers = min(contraction_values.shape[0], logical_ids.size)
        nearest, between_weights = _nearest_neighbors(
            graph.indptr,
            graph.indices,
            graph.data,
            volumes,
            logical_ids,
            n_workers,
        )
        first, second = _reciprocal_pairs(nearest)
        scan_seconds = time.perf_counter() - scan_start
        if first.size == 0:
            raise RuntimeError(
                "Paris reciprocal-neighbor scan made no progress in a connected component"
            )

        merge_count = first.size
        new_ids = np.arange(next_node, next_node + merge_count, dtype=np.int64)
        first_ids = logical_ids[first]
        second_ids = logical_ids[second]
        children = np.column_stack(
            (np.minimum(first_ids, second_ids), np.maximum(first_ids, second_ids))
        )
        heights = (volumes[first] / total_weight) * (
            volumes[second] / between_weights[first]
        )
        if not np.isfinite(heights).all():
            raise ValueError("Paris merge distances must be finite")
        merged_sizes = sizes[first] + sizes[second]
        child_blocks.append(children)
        height_blocks.append(heights)
        size_blocks.append(merged_sizes)

        sort_start = time.perf_counter()
        index_dtype = graph.indices.dtype
        mapping = np.empty(logical_ids.size, dtype=index_dtype)
        new_logical_ids, new_volumes, new_sizes = _prepare_contraction_groups(
            logical_ids,
            volumes,
            sizes,
            first,
            second,
            new_ids,
            mapping,
        )
        new_count = new_logical_ids.size
        sort_seconds = time.perf_counter() - sort_start
        contract_start = time.perf_counter()
        (
            graph,
            contraction_remap_seconds,
            contraction_filter_seconds,
            contraction_build_seconds,
            contraction_cleanup_seconds,
        ) = _contract_graph(
            graph,
            mapping,
            new_count,
            contraction_values,
        )
        contract_seconds = time.perf_counter() - contract_start

        round_diagnostics.append(
            ParisRoundDiagnostics(
                component=component,
                round_index=round_index,
                active_vertices=logical_ids.size,
                active_edges=active_edges,
                merges=merge_count,
                scan_seconds=scan_seconds,
                sort_seconds=sort_seconds,
                contraction_seconds=contract_seconds,
                contraction_remap_seconds=contraction_remap_seconds,
                contraction_filter_seconds=contraction_filter_seconds,
                contraction_build_seconds=contraction_build_seconds,
                contraction_cleanup_seconds=contraction_cleanup_seconds,
            )
        )
        logical_ids = new_logical_ids
        volumes = new_volumes
        sizes = new_sizes
        next_node += merge_count
        round_index += 1

    return (
        child_blocks,
        height_blocks,
        size_blocks,
        int(logical_ids[0]),
        next_node,
    )


def fit_paris_hierarchy(
    graph: spmatrix,
    *,
    nthreads: int | None = None,
) -> ParisHierarchy:
    """Fit a deterministic Paris hierarchy with reciprocal-neighbor rounds."""
    start = time.perf_counter()
    canonical = canonicalize_paris_graph(graph)
    preprocessing_seconds = time.perf_counter() - start
    n_leaves = canonical.shape[0]
    total_weight = float(canonical.data.sum(dtype=np.float64))
    if not np.isfinite(total_weight):
        raise ValueError("Paris graph total weight must be finite")

    component_start = time.perf_counter()
    n_components, labels = connected_components(
        canonical,
        directed=False,
        return_labels=True,
    )
    component_mins = np.full(n_components, n_leaves, dtype=np.int64)
    np.minimum.at(component_mins, labels, np.arange(n_leaves, dtype=np.int64))
    component_order = np.argsort(component_mins, kind="stable")
    vertices_by_component = np.argsort(labels, kind="stable")
    component_counts = np.bincount(labels, minlength=n_components)
    component_offsets = np.empty(n_components + 1, dtype=np.int64)
    component_offsets[0] = 0
    np.cumsum(component_counts, out=component_offsets[1:])
    component_seconds = time.perf_counter() - component_start

    requested_threads = get_num_threads() if nthreads is None else max(1, int(nthreads))
    previous_threads = get_num_threads()
    set_num_threads(min(requested_threads, config.NUMBA_NUM_THREADS))
    fit_start = time.perf_counter()
    child_blocks: list[np.ndarray] = []
    height_blocks: list[np.ndarray] = []
    size_blocks: list[np.ndarray] = []
    component_roots: list[int] = []
    component_sizes: list[int] = []
    round_diagnostics: list[ParisRoundDiagnostics] = []
    next_node = n_leaves
    try:
        for component_index, component_label in enumerate(component_order):
            start_offset = component_offsets[component_label]
            end_offset = component_offsets[component_label + 1]
            leaf_ids = vertices_by_component[start_offset:end_offset].astype(
                np.int64,
                copy=False,
            )
            component_sizes.append(leaf_ids.size)
            if leaf_ids.size == 1:
                component_roots.append(int(leaf_ids[0]))
                continue
            if leaf_ids.size == n_leaves:
                component_graph = canonical
            else:
                component_graph = canonical[leaf_ids][:, leaf_ids].tocsr()
            (
                component_children,
                component_heights,
                component_merge_sizes,
                component_root,
                next_node,
            ) = _fit_component(
                component_graph,
                leaf_ids,
                component=component_index,
                next_node=next_node,
                total_weight=total_weight,
                round_diagnostics=round_diagnostics,
            )
            child_blocks.extend(component_children)
            height_blocks.extend(component_heights)
            size_blocks.extend(component_merge_sizes)
            component_roots.append(component_root)
    finally:
        set_num_threads(previous_threads)

    finite_merge_count = sum(block.shape[0] for block in child_blocks)
    synthetic_count = n_components - 1
    index_dtype = np.int32 if 2 * n_leaves - 1 <= np.iinfo(np.int32).max else np.int64
    children = np.empty((n_leaves - 1, 2), dtype=index_dtype)
    heights = np.empty(n_leaves - 1, dtype=np.float64)
    size_dtype = np.int32 if n_leaves <= np.iinfo(np.int32).max else np.int64
    sizes = np.empty(n_leaves - 1, dtype=size_dtype)
    synthetic_joins = np.zeros(n_leaves - 1, dtype=bool)

    offset = 0
    for child_block, height_block, size_block in zip(
        child_blocks,
        height_blocks,
        size_blocks,
        strict=True,
    ):
        block_size = child_block.shape[0]
        children[offset : offset + block_size] = child_block
        heights[offset : offset + block_size] = height_block
        sizes[offset : offset + block_size] = size_block
        offset += block_size
    if offset != finite_merge_count:
        raise RuntimeError("Paris hierarchy assembly failed")

    if synthetic_count:
        root = component_roots[0]
        root_size = component_sizes[0]
        for component_root, component_size in zip(
            component_roots[1:],
            component_sizes[1:],
            strict=True,
        ):
            children[offset] = (root, component_root)
            root_size += component_size
            heights[offset] = np.inf
            sizes[offset] = root_size
            synthetic_joins[offset] = True
            root = next_node
            next_node += 1
            offset += 1

    if offset != n_leaves - 1 or next_node != 2 * n_leaves - 1:
        raise RuntimeError(
            "Paris hierarchy does not contain exactly n_leaves - 1 merges"
        )

    fit_seconds = time.perf_counter() - fit_start
    diagnostics = ParisFitDiagnostics(
        preprocessing_seconds=preprocessing_seconds,
        component_seconds=component_seconds,
        fit_seconds=fit_seconds,
        rounds=tuple(round_diagnostics),
    )
    return ParisHierarchy(
        children=children,
        heights=heights,
        sizes=sizes,
        component_roots=np.asarray(component_roots, dtype=index_dtype),
        synthetic_joins=synthetic_joins,
        n_leaves=n_leaves,
        total_weight=total_weight,
        diagnostics=diagnostics,
    )
