"""Construct, parse, and look up parameter-encoded assay graph paths.

This module is the sole parser for the assay graph chain: ``normed__``,
reduction, ANN (optional ``__unscaled`` / ``__harmony_{hash}``), KNN, graph,
k-means, and ``latest_*`` traversal. It does not mutate stores.
"""

from typing import Any

from .paths import (
    AssayGraphPaths,
    AssayNearestNeighborPaths,
    StoredAssayGraph,
    StoredIntegratedGraph,
)


def make_normalized_leaf_name(cell_key: str, feat_key: str) -> str:
    """Return ``normed__{cell_key}__{feat_key}``."""
    return f"normed__{cell_key}__{feat_key}"


def make_normalized_group_path(from_assay: str, cell_key: str, feat_key: str) -> str:
    """Return ``{from_assay}/normed__{cell_key}__{feat_key}``."""
    return f"{from_assay}/{make_normalized_leaf_name(cell_key, feat_key)}"


def make_reduction_group_path(
    normalized_group_path: str,
    reduction_method: str,
    dims: int,
    pca_cell_key: str,
) -> str:
    return (
        f"{normalized_group_path}/reduction__{reduction_method}__{dims}__{pca_cell_key}"
    )


def make_neighbor_index_group_path(
    reduction_group_path: str,
    ann_metric: str,
    ann_efc: int,
    ann_ef: int,
    ann_m: int,
    rand_state: int,
    *,
    feat_scaling: bool = True,
    harmony_contract_hash: str | None = None,
) -> str:
    path = (
        f"{reduction_group_path}/ann__{ann_metric}__{ann_efc}__{ann_ef}__"
        f"{ann_m}__{rand_state}"
    )
    if not feat_scaling:
        path = f"{path}__unscaled"
    if harmony_contract_hash is not None:
        path = f"{path}__harmony_{harmony_contract_hash}"
    return path


def make_nearest_neighbors_group_path(neighbor_index_group_path: str, k: int) -> str:
    return f"{neighbor_index_group_path}/knn__{k}"


def make_cell_graph_group_path(
    nearest_neighbors_group_path: str,
    local_connectivity: float,
    bandwidth: float,
) -> str:
    return f"{nearest_neighbors_group_path}/graph__{local_connectivity}__{bandwidth}"


def make_kmeans_initialization_group_path(
    reduction_group_path: str, n_centroids: int, rand_state: int
) -> str:
    return f"{reduction_group_path}/kmeans__{n_centroids}__{rand_state}"


def make_integrated_graph_path(integrated_graphs_loc: str, label: str) -> str:
    return f"{integrated_graphs_loc}/{label}"


def is_integrated_graph_path(graph_loc: str, integrated_graphs_loc: str) -> bool:
    return graph_loc == integrated_graphs_loc or graph_loc.startswith(
        f"{integrated_graphs_loc}/"
    )


def nearest_neighbors_group_path_from_cell_graph(cell_graph_group_path: str) -> str:
    """Return the parent KNN group path for a cell-graph leaf."""
    if "/" not in cell_graph_group_path:
        raise ValueError(
            f"Cell graph path has no parent group: {cell_graph_group_path}"
        )
    return cell_graph_group_path.rsplit("/", 1)[0]


def parse_normalized_leaf_name(name: str) -> tuple[str, str]:
    if not name.startswith("normed__"):
        raise ValueError(f"Not a normalized group name: {name}")
    rest = name.removeprefix("normed__")
    cell_key, feat_key = rest.split("__", 1)
    return cell_key, feat_key


def parse_reduction_leaf_name(name: str) -> tuple[str, int, str]:
    if not name.startswith("reduction__"):
        raise ValueError(f"Not a reduction group name: {name}")
    body = name.removeprefix("reduction__")
    reduction_method, dims_s, pca_cell_key = body.split("__", 2)
    return reduction_method, int(dims_s), pca_cell_key


def parse_reduction_group_path(
    reduction_group_path: str,
) -> tuple[str, int, str]:
    """Parse the reduction leaf of a complete reduction group path."""
    return parse_reduction_leaf_name(_leaf(reduction_group_path))


def parse_ann_leaf_name(
    name: str,
) -> tuple[str, int, int, int, int, bool, str | None]:
    """Parse ANN leaf name; ``__unscaled`` and ``__harmony_{hash}`` are optional."""
    if not name.startswith("ann__"):
        raise ValueError(f"Not an ANN group name: {name}")
    parts = name.split("__")
    if len(parts) < 6:
        raise ValueError(f"ANN group name missing required fields: {name}")
    ann_metric = parts[1]
    ann_efc, ann_ef, ann_m, rand_state = (
        int(parts[2]),
        int(parts[3]),
        int(parts[4]),
        int(parts[5]),
    )
    feat_scaling = True
    harmony_contract_hash: str | None = None
    for token in parts[6:]:
        if token == "unscaled":
            feat_scaling = False
        elif token.startswith("harmony_"):
            harmony_contract_hash = token.removeprefix("harmony_")
        else:
            raise ValueError(f"Unrecognized ANN path suffix {token!r} in {name}")
    return (
        ann_metric,
        ann_efc,
        ann_ef,
        ann_m,
        rand_state,
        feat_scaling,
        harmony_contract_hash,
    )


def parse_neighbor_index_group_path(
    neighbor_index_group_path: str,
) -> tuple[str, int, int, int, int, bool, str | None]:
    """Parse the ANN leaf of a complete neighbor-index group path."""
    return parse_ann_leaf_name(_leaf(neighbor_index_group_path))


def parse_knn_leaf_name(name: str) -> int:
    if not name.startswith("knn__"):
        raise ValueError(f"Not a KNN group name: {name}")
    return int(name.removeprefix("knn__"))


def parse_nearest_neighbors_group_path(
    nearest_neighbors_group_path: str,
) -> int:
    """Parse the KNN leaf of a complete nearest-neighbors group path."""
    return parse_knn_leaf_name(_leaf(nearest_neighbors_group_path))


def parse_graph_leaf_name(name: str) -> tuple[float, float]:
    if not name.startswith("graph__"):
        raise ValueError(f"Not a cell-graph group name: {name}")
    body = name.removeprefix("graph__")
    local_connectivity_s, bandwidth_s = body.split("__", 1)
    return float(local_connectivity_s), float(bandwidth_s)


def parse_kmeans_leaf_name(name: str) -> tuple[int, int]:
    if not name.startswith("kmeans__"):
        raise ValueError(f"Not a k-means group name: {name}")
    body = name.removeprefix("kmeans__")
    n_centroids_s, rand_state_s = body.split("__", 1)
    return int(n_centroids_s), int(rand_state_s)


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def parse_assay_keys_from_nearest_neighbors_path(
    nearest_neighbors_group_path: str,
) -> tuple[str, str, str]:
    """Return assay, cell key, and feature key encoded by a KNN path."""
    paths = nearest_neighbor_paths_from_loc(nearest_neighbors_group_path)
    cell_key, feat_key = parse_normalized_leaf_name(_leaf(paths.normalized_group_path))
    if "/" not in paths.normalized_group_path:
        raise ValueError(
            f"Normalized path has no assay parent: {paths.normalized_group_path}"
        )
    from_assay = paths.normalized_group_path.rsplit("/", 1)[0]
    return from_assay, cell_key, feat_key


def stored_assay_graph_from_paths(
    paths: AssayGraphPaths,
    *,
    from_assay: str | None = None,
    cell_key: str | None = None,
    feat_key: str | None = None,
) -> StoredAssayGraph:
    """Fill cheap scientific parameters from encoded leaf names when possible."""
    resolved_assay = from_assay
    resolved_cell = cell_key
    resolved_feat = feat_key
    try:
        parsed_cell, parsed_feat = parse_normalized_leaf_name(
            _leaf(paths.normalized_group_path)
        )
        if resolved_cell is None:
            resolved_cell = parsed_cell
        if resolved_feat is None:
            resolved_feat = parsed_feat
        if resolved_assay is None:
            parent = paths.normalized_group_path.rsplit("/", 1)[0]
            resolved_assay = parent
    except ValueError:
        pass

    reduction_method = dims = pca_cell_key = None
    try:
        reduction_method, dims, pca_cell_key = parse_reduction_leaf_name(
            _leaf(paths.reduction_group_path)
        )
    except ValueError:
        pass

    ann_metric = ann_efc = ann_ef = ann_m = rand_state = None
    feat_scaling = harmony_contract_hash = None
    try:
        (
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            feat_scaling,
            harmony_contract_hash,
        ) = parse_ann_leaf_name(_leaf(paths.neighbor_index_group_path))
    except ValueError:
        pass

    k = None
    try:
        k = parse_knn_leaf_name(_leaf(paths.nearest_neighbors_group_path))
    except ValueError:
        pass

    local_connectivity = bandwidth = None
    try:
        local_connectivity, bandwidth = parse_graph_leaf_name(
            _leaf(paths.cell_graph_group_path)
        )
    except ValueError:
        pass

    n_centroids = None
    if paths.kmeans_initialization_group_path is not None:
        try:
            n_centroids, kmeans_rand = parse_kmeans_leaf_name(
                _leaf(paths.kmeans_initialization_group_path)
            )
            if rand_state is None:
                rand_state = kmeans_rand
        except ValueError:
            pass

    if resolved_assay is None or resolved_cell is None or resolved_feat is None:
        raise ValueError(
            "Could not determine assay, cell_key, and feat_key for stored graph"
        )

    return StoredAssayGraph(
        paths=paths,
        from_assay=resolved_assay,
        cell_key=resolved_cell,
        feat_key=resolved_feat,
        reduction_method=reduction_method,
        dims=dims,
        pca_cell_key=pca_cell_key,
        ann_metric=ann_metric,
        ann_efc=ann_efc,
        ann_ef=ann_ef,
        ann_m=ann_m,
        rand_state=rand_state,
        k=k,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
        feat_scaling=feat_scaling,
        harmony_contract_hash=harmony_contract_hash,
        n_centroids=n_centroids,
    )


def parse_assay_graph_paths(
    graph_loc: str,
    *,
    kmeans_initialization_group_path: str | None = None,
) -> StoredAssayGraph:
    """Parse a parameter-encoded cell-graph path into a ``StoredAssayGraph``."""
    segments = graph_loc.strip("/").split("/")
    if len(segments) < 6:
        raise ValueError(f"Not an encoded assay graph path: {graph_loc}")

    graph_name = segments[-1]
    knn_name = segments[-2]
    ann_name = segments[-3]
    reduction_name = segments[-4]
    normed_name = segments[-5]
    from_assay = "/".join(segments[:-5])
    if not from_assay:
        raise ValueError(f"Assay graph path missing assay prefix: {graph_loc}")

    # Validate leaf encodings early so malformed paths fail here.
    parse_normalized_leaf_name(normed_name)
    parse_reduction_leaf_name(reduction_name)
    parse_ann_leaf_name(ann_name)
    parse_knn_leaf_name(knn_name)
    parse_graph_leaf_name(graph_name)

    normalized = "/".join(segments[:-4])
    reduction = "/".join(segments[:-3])
    neighbor_index = "/".join(segments[:-2])
    nearest_neighbors = "/".join(segments[:-1])
    paths = AssayGraphPaths(
        normalized_group_path=normalized,
        reduction_group_path=reduction,
        neighbor_index_group_path=neighbor_index,
        nearest_neighbors_group_path=nearest_neighbors,
        cell_graph_group_path=graph_loc.strip("/"),
        kmeans_initialization_group_path=kmeans_initialization_group_path,
    )
    return stored_assay_graph_from_paths(paths, from_assay=from_assay)


def _group_attrs(group: Any) -> Any:
    return group.attrs


def lookup_latest_reduction_group_path(zw: Any, normalized_group_path: str) -> str:
    """Return the existing reduction selected by a normalized group."""
    normalized_group = zw[normalized_group_path]
    reduction_group_path = str(_group_attrs(normalized_group)["latest_reduction"])
    return reduction_group_path


def lookup_latest_neighbor_index_group_path(zw: Any, reduction_group_path: str) -> str:
    """Return the existing neighbor index selected by a reduction group."""
    reduction_group = zw[reduction_group_path]
    neighbor_index_group_path = str(_group_attrs(reduction_group)["latest_ann"])
    return neighbor_index_group_path


def lookup_latest_nearest_neighbors_group_path(
    zw: Any, neighbor_index_group_path: str
) -> str:
    """Return the existing KNN group selected by a neighbor-index group."""
    neighbor_index_group = zw[neighbor_index_group_path]
    nearest_neighbors_group_path = str(_group_attrs(neighbor_index_group)["latest_knn"])
    return nearest_neighbors_group_path


def lookup_latest_cell_graph_group_path(
    zw: Any, nearest_neighbors_group_path: str
) -> str:
    """Return the existing cell graph selected by a KNN group."""
    nearest_neighbors_group = zw[nearest_neighbors_group_path]
    cell_graph_group_path = str(_group_attrs(nearest_neighbors_group)["latest_graph"])
    return cell_graph_group_path


def lookup_latest_kmeans_group_path(zw: Any, reduction_group_path: str) -> str | None:
    """Return the k-means group selected by a reduction, when recorded."""
    reduction_group = zw[reduction_group_path]
    reduction_attrs = _group_attrs(reduction_group)
    if "latest_kmeans" not in reduction_attrs:
        return None
    return str(reduction_attrs["latest_kmeans"])


def lookup_latest_assay_paths(
    zw: Any,
    from_assay: str,
    cell_key: str,
    feat_key: str,
) -> AssayGraphPaths:
    """Follow ``latest_*`` pointers through a complete assay graph chain."""
    nearest_neighbor_paths = lookup_latest_nearest_neighbor_paths(
        zw, from_assay, cell_key, feat_key
    )
    reduction = nearest_neighbor_paths.reduction_group_path
    nearest_neighbors = nearest_neighbor_paths.nearest_neighbors_group_path
    cell_graph = lookup_latest_cell_graph_group_path(zw, nearest_neighbors)
    kmeans = lookup_latest_kmeans_group_path(zw, reduction)
    return AssayGraphPaths(
        normalized_group_path=nearest_neighbor_paths.normalized_group_path,
        reduction_group_path=reduction,
        neighbor_index_group_path=nearest_neighbor_paths.neighbor_index_group_path,
        nearest_neighbors_group_path=nearest_neighbors,
        cell_graph_group_path=cell_graph,
        kmeans_initialization_group_path=kmeans,
    )


def lookup_latest_nearest_neighbor_paths(
    zw: Any,
    from_assay: str,
    cell_key: str,
    feat_key: str,
) -> AssayNearestNeighborPaths:
    """Follow ``latest_*`` pointers only through the nearest-neighbors group."""
    normalized = make_normalized_group_path(from_assay, cell_key, feat_key)
    reduction = lookup_latest_reduction_group_path(zw, normalized)
    neighbor_index = lookup_latest_neighbor_index_group_path(zw, reduction)
    nearest_neighbors = lookup_latest_nearest_neighbors_group_path(zw, neighbor_index)
    _ = zw[nearest_neighbors]
    return AssayNearestNeighborPaths(
        normalized_group_path=normalized,
        reduction_group_path=reduction,
        neighbor_index_group_path=neighbor_index,
        nearest_neighbors_group_path=nearest_neighbors,
    )


def lookup_latest_kmeans_path(
    zw: Any, from_assay: str, cell_key: str, feat_key: str
) -> str | None:
    """Return the latest k-means group path without requiring a graph.

    Follows ``normed -> latest_reduction -> latest_kmeans`` only. Does not
    require an ANN index, KNN graph, or cell graph to exist, matching the
    minimal inputs needed for initial embedding. Returns None when the
    reduction has no k-means initialization recorded.
    """
    normalized = make_normalized_group_path(from_assay, cell_key, feat_key)
    reduction = lookup_latest_reduction_group_path(zw, normalized)
    return lookup_latest_kmeans_group_path(zw, reduction)


def lookup_latest_assay_graph(
    zw: Any, from_assay: str, cell_key: str, feat_key: str
) -> StoredAssayGraph:
    """Follow ``latest_*`` pointers for an assay graph without mutating the store."""
    paths = lookup_latest_assay_paths(zw, from_assay, cell_key, feat_key)
    return stored_assay_graph_from_paths(
        paths,
        from_assay=from_assay,
        cell_key=cell_key,
        feat_key=feat_key,
    )


def lookup_stored_integrated_graph(zw: Any, graph_loc: str) -> StoredIntegratedGraph:
    """Read an integrated graph path and its stored metadata only."""
    group = zw[graph_loc]
    attrs = _group_attrs(group)
    n_cells = int(attrs["n_cells"]) if "n_cells" in attrs else None
    n_neighbors = int(attrs["n_neighbors"]) if "n_neighbors" in attrs else None
    return StoredIntegratedGraph(
        cell_graph_group_path=graph_loc,
        n_cells=n_cells,
        n_neighbors=n_neighbors,
    )


def nearest_neighbor_paths_from_loc(knn_loc: str) -> AssayNearestNeighborPaths:
    """Parse a KNN location into exact parent-chain paths."""
    neighbor_index = knn_loc.rsplit("/", 1)[0]
    reduction = neighbor_index.rsplit("/", 1)[0]
    normalized = reduction.rsplit("/", 1)[0]
    parse_normalized_leaf_name(_leaf(normalized))
    parse_reduction_leaf_name(_leaf(reduction))
    parse_ann_leaf_name(_leaf(neighbor_index))
    parse_knn_leaf_name(_leaf(knn_loc))
    return AssayNearestNeighborPaths(
        normalized_group_path=normalized,
        reduction_group_path=reduction,
        neighbor_index_group_path=neighbor_index,
        nearest_neighbors_group_path=knn_loc,
    )
