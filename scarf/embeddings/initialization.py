import numpy as np

from ..utils.arrays import rescale_array


def initial_embedding(
    cluster_centers: np.ndarray,
    cluster_labels: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Create cell coordinates from PCA-projected seed partitions."""
    from sklearn.decomposition import PCA

    principal_components = PCA(n_components=n_components).fit_transform(
        np.asarray(cluster_centers)
    )
    for component in range(n_components):
        principal_components[:, component] = rescale_array(
            principal_components[:, component]
        )
    labels = np.asarray(cluster_labels).astype(np.uint32)
    return np.array([principal_components[label] for label in labels]).astype(
        np.float32, order="C"
    )
