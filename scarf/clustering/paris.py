import numpy as np
from scipy.sparse import spmatrix

from ._paris_core import (
    ParisHierarchy,
    fit_paris_hierarchy as _fit_paris_hierarchy,
)
from .balanced_cut import BalancedCut


def fit_paris_hierarchy(
    graph: spmatrix,
    *,
    nthreads: int | None = None,
) -> ParisHierarchy:
    """Fit the canonical component-aware Paris hierarchy."""
    return _fit_paris_hierarchy(graph, nthreads=nthreads)


def hierarchy_to_dendrogram(
    hierarchy: ParisHierarchy,
    *,
    compatibility: bool = False,
) -> np.ndarray:
    """Convert a typed Paris hierarchy to a SciPy linkage matrix."""
    dendrogram = np.empty((hierarchy.n_leaves - 1, 4), dtype=np.float64)
    dendrogram[:, :2] = hierarchy.children
    dendrogram[:, 2] = hierarchy.heights
    dendrogram[:, 3] = hierarchy.sizes
    if compatibility:
        dendrogram[hierarchy.synthetic_joins, 2] = 0.0
    return dendrogram


def paris_dendrogram(
    graph: spmatrix,
    *,
    nthreads: int | None = None,
) -> np.ndarray:
    """Fit Paris and map synthetic component joins to zero."""
    hierarchy = fit_paris_hierarchy(graph, nthreads=nthreads)
    return hierarchy_to_dendrogram(hierarchy, compatibility=True)


def _validate_linkage_children(dendrogram: np.ndarray) -> np.ndarray:
    raw_children = np.asarray(dendrogram[:, :2])
    if np.iscomplexobj(raw_children):
        raise ValueError("dendrogram child IDs must be finite integers")
    try:
        child_values = np.asarray(raw_children, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("dendrogram child IDs must be finite integers") from None
    if not np.all(np.isfinite(child_values)) or not np.all(
        child_values == np.floor(child_values)
    ):
        raise ValueError("dendrogram child IDs must be finite integers")
    if np.any(child_values < 0):
        raise ValueError("dendrogram child IDs must be non-negative")

    n_leaves = dendrogram.shape[0] + 1
    for merge_index, (left, right) in enumerate(child_values):
        node = n_leaves + merge_index
        if left == right:
            raise ValueError(
                f"dendrogram row {merge_index + 1} must reference two distinct children"
            )
        if left >= node or right >= node:
            raise ValueError(
                f"dendrogram row {merge_index + 1} must reference only earlier nodes"
            )

    children = child_values.astype(np.int64, copy=False)
    n_nodes = 2 * n_leaves - 1
    consumption_counts = np.bincount(children.ravel(), minlength=n_nodes)
    repeated = np.flatnonzero(consumption_counts > 1)
    if repeated.size > 0:
        raise ValueError(
            f"dendrogram child node {int(repeated[0])} is consumed more than once"
        )
    missing = np.flatnonzero(consumption_counts[:-1] == 0)
    if missing.size > 0:
        raise ValueError(
            "dendrogram must consume every non-root node exactly once; "
            f"missing node {int(missing[0])}"
        )
    return children


def straight_cut(dendrogram: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut a Paris dendrogram into a fixed number of clusters."""
    if dendrogram.ndim != 2 or dendrogram.shape[1] != 4:
        raise ValueError("dendrogram must have shape (n_leaves - 1, 4)")
    if isinstance(n_clusters, bool) or not isinstance(
        n_clusters,
        (int, np.integer),
    ):
        raise TypeError("n_clusters must be an integer")

    n_leaves = dendrogram.shape[0] + 1
    cluster_count = int(n_clusters)
    if cluster_count < 1 or cluster_count > n_leaves:
        raise ValueError("n_clusters must be between 1 and the number of leaves")
    children = _validate_linkage_children(dendrogram)
    if cluster_count == 1:
        return np.ones(n_leaves, dtype=np.int64)

    cut_index = n_leaves - cluster_count
    heights = np.asarray(dendrogram[:, 2])
    cut_height = np.partition(heights, cut_index)[cut_index]
    n_nodes = 2 * n_leaves - 1
    active = np.zeros(n_nodes, dtype=bool)
    active[:n_leaves] = True
    sizes = np.zeros(n_nodes, dtype=np.int64)
    sizes[:n_leaves] = 1

    for merge_index, (left, right) in enumerate(children):
        node = n_leaves + merge_index
        if (
            heights[merge_index] < cut_height
            and 0 <= left < node
            and 0 <= right < node
            and active[left]
            and active[right]
        ):
            active[left] = False
            active[right] = False
            active[node] = True
            sizes[node] = sizes[left] + sizes[right]

    roots = np.flatnonzero(active)
    roots = roots[np.argsort(-sizes[roots])]
    labels_by_node = np.full(n_nodes, -1, dtype=np.int64)
    labels_by_node[roots] = np.arange(roots.size, dtype=np.int64)
    for merge_index in range(n_leaves - 2, -1, -1):
        node = n_leaves + merge_index
        label = labels_by_node[node]
        if label >= 0:
            left, right = children[merge_index]
            labels_by_node[left] = label
            labels_by_node[right] = label

    labels = labels_by_node[:n_leaves]
    if np.any(labels < 0):
        raise ValueError("dendrogram contains invalid child references")
    return labels + 1


def balanced_cut(
    dendrogram: np.ndarray,
    max_size: int,
    min_size: int,
    max_distance_fc: float,
) -> np.ndarray:
    """Cut a hierarchy using balanced size and distance constraints."""
    return BalancedCut(
        dendrogram,
        max_size,
        min_size,
        max_distance_fc,
    ).get_clusters()
