import numpy as np

from ..utils.arrays import rescale_array


def initial_embedding(
    cluster_centers: np.ndarray,
    cluster_labels: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Create cell coordinates from PCA-projected seed partitions."""
    from sklearn.decomposition import PCA

    centers = np.asarray(cluster_centers)
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1:
        raise ValueError("Cluster labels must be one-dimensional")
    if np.issubdtype(labels.dtype, np.floating):
        if not np.all(np.isfinite(labels)) or not np.array_equal(
            labels,
            np.floor(labels),
        ):
            raise ValueError("Floating cluster labels must be finite integers")
    elif not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Cluster labels must contain numeric integers")
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError("Cluster labels are outside the center range")
    if n_components < 1 or n_components > min(centers.shape):
        raise ValueError(
            "Embedding components cannot exceed the center count or dimensions"
        )
    index_labels = labels.astype(np.intp, copy=False)
    principal_components = PCA(n_components=n_components).fit_transform(centers)
    for component in range(n_components):
        principal_components[:, component] = rescale_array(
            principal_components[:, component]
        )
    return np.array([principal_components[label] for label in index_labels]).astype(
        np.float32, order="C"
    )
