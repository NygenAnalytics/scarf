import numpy as np
from scipy.sparse import spmatrix

from .hierarchy import BalancedCut


def paris_dendrogram(graph: spmatrix) -> np.ndarray:
    """Fit a Paris hierarchy and replace disconnected merge distances."""
    import sknetwork as skn

    dendrogram = np.asarray(skn.hierarchy.Paris(reorder=False).fit_transform(graph))
    dendrogram[dendrogram == np.inf] = 0
    return dendrogram


def straight_cut(dendrogram: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut a Paris dendrogram into a fixed number of clusters."""
    import sknetwork as skn

    return (
        np.asarray(
            skn.hierarchy.cut_straight(
                dendrogram,
                n_clusters=n_clusters,
            )
        )
        + 1
    )


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
