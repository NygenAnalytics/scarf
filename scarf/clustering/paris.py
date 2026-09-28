import numpy as np

from ._paris_core import (
    ParisHierarchy,
    fit_paris_hierarchy as fit_paris_hierarchy,
)


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


def _apply_merge(
    children: np.ndarray,
    active: np.ndarray,
    sizes: np.ndarray,
    merge_index: int,
) -> bool:
    """Merge the two active children of one row into its node."""
    left, right = children[merge_index]
    if not (active[left] and active[right]):
        return False
    node = children.shape[0] + 1 + merge_index
    active[left] = False
    active[right] = False
    active[node] = True
    sizes[node] = sizes[left] + sizes[right]
    return True


def _merges_below(
    children: np.ndarray,
    heights: np.ndarray,
    cut_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply, in row order, every merge strictly below ``cut_height``.

    Returns which nodes head a current cluster and each merged node's size;
    a merge that was not applied keeps size zero.
    """
    n_leaves = children.shape[0] + 1
    active = np.zeros(2 * n_leaves - 1, dtype=bool)
    active[:n_leaves] = True
    sizes = np.zeros(2 * n_leaves - 1, dtype=np.int64)
    sizes[:n_leaves] = 1
    for merge_index in range(n_leaves - 1):
        if heights[merge_index] < cut_height:
            _apply_merge(children, active, sizes, merge_index)
    return active, sizes


def _size_ordered_labels(
    children: np.ndarray,
    active: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    """Label clusters from 1 by decreasing size, breaking ties by node order."""
    n_leaves = children.shape[0] + 1
    roots = np.flatnonzero(active)
    roots = roots[np.argsort(-sizes[roots], kind="stable")]
    labels_by_node = np.full(active.size, -1, dtype=np.int64)
    labels_by_node[roots] = np.arange(roots.size, dtype=np.int64)
    for merge_index in range(n_leaves - 2, -1, -1):
        label = labels_by_node[n_leaves + merge_index]
        if label >= 0:
            labels_by_node[children[merge_index]] = label
    labels = labels_by_node[:n_leaves]
    if np.any(labels < 0):
        raise ValueError("dendrogram contains invalid child references")
    return labels + 1


def straight_cut(dendrogram: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut a Paris dendrogram into a fixed number of clusters.

    As in scikit-network, only merges strictly below the cut height are
    applied, so tied heights can leave more clusters than requested. Clusters
    are labelled from 1 by decreasing size, and equal sizes by node order.
    """
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
    active, sizes = _merges_below(children, heights, cut_height)
    return _size_ordered_labels(children, active, sizes)


def _merge_tied_overflow(
    children: np.ndarray,
    heights: np.ndarray,
    synthetic_joins: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Apply tied component-internal merges until ``n_clusters`` remain."""
    children = np.asarray(children, dtype=np.int64)
    heights = np.asarray(heights, dtype=np.float64)
    n_leaves = children.shape[0] + 1
    cut_index = n_leaves - n_clusters
    active, sizes = _merges_below(
        children,
        heights,
        np.partition(heights, cut_index)[cut_index],
    )
    n_active = int(np.count_nonzero(active))
    candidates = np.argsort(heights, kind="stable")
    candidates = candidates[~np.asarray(synthetic_joins, dtype=bool)[candidates]]
    progressed = True
    while n_active > n_clusters and progressed:
        progressed = False
        for merge_index in candidates:
            if n_active == n_clusters:
                break
            if sizes[n_leaves + merge_index] == 0 and _apply_merge(
                children,
                active,
                sizes,
                int(merge_index),
            ):
                n_active -= 1
                progressed = True
    if n_active != n_clusters:
        raise ValueError(f"The hierarchy cannot be cut into {n_clusters} clusters")
    return _size_ordered_labels(children, active, sizes)


def fixed_cut(hierarchy: ParisHierarchy, n_clusters: int) -> np.ndarray:
    """Cut a Paris hierarchy into exactly ``n_clusters`` clusters labelled from 1.

    :func:`straight_cut` can leave extra clusters when merge heights tie at the
    cut. Tied merges inside components are then applied in ascending height,
    and in hierarchy row order within one height. Synthetic joins between
    components are never applied, so callers must reject
    ``1 < n_clusters < n_components``. Labels follow :func:`straight_cut`.
    """
    labels = straight_cut(hierarchy_to_dendrogram(hierarchy), n_clusters)
    if int(labels.max()) <= n_clusters:
        return labels
    return _merge_tied_overflow(
        hierarchy.children,
        hierarchy.heights,
        hierarchy.synthetic_joins,
        n_clusters,
    )
