"""Parse parameter-encoded assay graph paths for diagnostics.

This module is the sole parser for the assay graph chain: ``normed__``,
reduction, ANN (optional ``__unscaled`` / ``__harmony_{hash}``), KNN, graph,
k-means. Runtime graph resolution does not use these legacy paths.
"""

from .paths import (
    AssayGraphPaths,
    AssayNearestNeighborPaths,
    StoredAssayGraph,
)


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
