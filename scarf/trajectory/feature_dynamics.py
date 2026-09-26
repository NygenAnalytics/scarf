from typing import Any

import numpy as np
import scarf

from ..utils.arrays import rolling_window
from ..utils.logging import logger
from .parameters import resolve_aggregation_ann_params

__all__ = [
    "aggregate_feature_profiles",
    "knn_clustering",
    "scatter_feature_clusters",
    "validate_pseudotime_regressor",
]

# Scratch bytes per input value in aggregate_feature_profiles: the ordered
# copy, its smoothed replacement, and the temporary used by the standard
# deviation, with room for the boolean finiteness checks.
AGGREGATION_SCRATCH_ITEMSIZE = 3 * np.dtype(np.float64).itemsize


def validate_pseudotime_regressor(
    values: object,
    expected_size: int,
    pseudotime_key: str,
    cell_key: str,
    *,
    has_validity_column: bool,
) -> np.ndarray:
    try:
        pseudotime = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Pseudotime column '{pseudotime_key}' must be numeric"
        ) from exc

    if pseudotime.ndim != 1:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' must be one-dimensional"
        )
    if pseudotime.shape[0] != expected_size:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' has {pseudotime.shape[0]} values, "
            f"but cell_key '{cell_key}' selects {expected_size} cells"
        )
    if not np.isfinite(pseudotime).all():
        validity_key = f"{pseudotime_key}__valid"
        if has_validity_column:
            raise ValueError(
                f"Pseudotime column '{pseudotime_key}' contains unscored cells. "
                f"Use cell_key='{validity_key}' for downstream analysis"
            )
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' contains non-finite values"
        )
    if pseudotime.size < 2 or np.unique(pseudotime).size < 2:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' must contain at least two distinct values"
        )
    return pseudotime


def scatter_feature_clusters(
    n_features: int,
    feature_indices: np.ndarray,
    clusters: np.ndarray,
    unassigned_value: int,
) -> np.ndarray:
    feature_indices = np.asarray(feature_indices, dtype=int)
    clusters = np.asarray(clusters, dtype=int)
    if feature_indices.shape != clusters.shape:
        raise ValueError("Feature indices and cluster assignments are misaligned")
    if unassigned_value in clusters:
        raise ValueError("unassigned_value conflicts with an assigned feature cluster")
    values = np.full(n_features, unassigned_value, dtype=int)
    values[feature_indices] = clusters
    return values


def aggregate_feature_profiles(
    values: np.ndarray,
    ordering_indices: np.ndarray,
    feature_indices: np.ndarray,
    *,
    min_expression: float,
    window_size: int,
    n_bins: int,
    smooth: bool,
    z_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Order, smooth, filter, and bin normalized feature profiles.

    ``values`` is cells by features and is not modified. Besides the input,
    the calculation holds at most :data:`AGGREGATION_SCRATCH_ITEMSIZE` bytes
    per value, because smoothing and scaling reuse one ordered copy.
    """
    values = np.asarray(values, dtype=float)
    mean_expression = values.mean(axis=0)
    # Fancy indexing returns a new C-order array, which can be updated in
    # place and gives the column reductions a fixed summation order.
    ordered = np.ascontiguousarray(values[ordering_indices])
    finite_columns = np.isfinite(ordered).all(axis=0)
    if not finite_columns.all():
        invalid_columns = np.asarray(feature_indices)[~finite_columns]
        raise ValueError(
            f"Normalized features contain non-finite values: {invalid_columns.tolist()}"
        )
    if smooth:
        ordered = rolling_window(ordered, window_size)
        if not np.isfinite(ordered).all():
            raise ValueError("Smoothed feature profiles contain non-finite values")

    standard_deviation = ordered.std(axis=0)
    valid_features = (
        (mean_expression > min_expression)
        & np.isfinite(standard_deviation)
        & (standard_deviation > np.finfo(float).eps)
    )
    if z_scale:
        # Scale in place. Reducing only the retained columns keeps the
        # established floating-point summation order of their means.
        column_mean = np.zeros(ordered.shape[1], dtype=np.float64)
        if valid_features.any():
            column_mean[valid_features] = ordered[:, valid_features].mean(axis=0)
        ordered -= column_mean
        ordered /= np.where(valid_features, standard_deviation, 1.0)
    ordered[:, ~valid_features] = 0.0

    binned = np.stack(
        [
            bin_values.mean(axis=0)
            for bin_values in np.array_split(ordered, n_bins, axis=0)
        ],
        axis=1,
    )
    if not np.isfinite(binned).all():
        raise ValueError("Binned feature profiles contain non-finite values")
    return binned, valid_features


def knn_clustering(
    d_array: "scarf.matrix.ChunkedArray",
    n_neighbours: int,
    n_clusters: int,
    nthreads: int,
    ann_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Cluster pseudotime-ordered feature profiles."""
    import pandas as pd
    from scipy.sparse import csr_matrix

    from ..neighbors.index import fix_knn_query, instantiate_knn_index
    from ..utils.compute import compute_with_progress

    if len(d_array.shape) != 2:
        raise ValueError("d_array must be two-dimensional")
    n_genes = int(d_array.shape[0])
    if n_genes < 2:
        raise ValueError("At least two retained genes are required for clustering")
    if isinstance(n_neighbours, (bool, np.bool_)) or not isinstance(
        n_neighbours,
        (int, np.integer),
    ):
        raise TypeError("n_neighbours must be an integer")
    n_neighbours = int(n_neighbours)
    if not 1 <= n_neighbours < n_genes:
        raise ValueError(f"n_neighbours must satisfy 1 <= n_neighbours < {n_genes}")
    if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(
        n_clusters,
        (int, np.integer),
    ):
        raise TypeError("n_clusters must be an integer")
    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= n_genes:
        raise ValueError(f"n_clusters must satisfy 1 <= n_clusters <= {n_genes}")

    def make_knn_mat(
        data: "scarf.matrix.ChunkedArray",
        k: int,
        threads: int,
    ) -> "csr_matrix":
        logger.debug("Pseudotime modules: fitting feature KNN")
        for block in data.stream_blocks(nthreads=threads, msg="Fitting feature KNN"):
            if not np.isfinite(block).all():
                raise ValueError("Feature profiles must contain only finite values")
            ann_idx.add_items(block)
        start, end = 0, 0
        neighbor_indices: list[np.ndarray] = []
        logger.debug("Pseudotime modules: querying feature KNN")
        for block in data.stream_blocks(
            nthreads=threads,
            msg="Identifying feature neighbors",
        ):
            end += block.shape[0]
            indices, distances = ann_idx.knn_query(block, k=k + 1)
            indices, _, _ = fix_knn_query(
                indices,
                distances,
                np.arange(start, end),
            )
            neighbor_indices.append(indices)
            start = end
        indices_mat = np.vstack(neighbor_indices)
        assert indices_mat.shape[0] == data.shape[0]

        return csr_matrix(
            (
                np.ones(indices_mat.shape[0] * indices_mat.shape[1]),
                (
                    np.repeat(
                        np.arange(indices_mat.shape[0]),
                        indices_mat.shape[1],
                    ),
                    indices_mat.flatten(),
                ),
            ),
            shape=(indices_mat.shape[0], indices_mat.shape[0]),
        )

    def make_clusters(matrix: "csr_matrix", n_cluster: int) -> np.ndarray:
        from ..clustering.paris import fit_paris_hierarchy, fixed_cut

        logger.debug("Pseudotime modules: clustering modules")
        hierarchy = fit_paris_hierarchy(
            matrix,
            nthreads=nthreads,
        )
        n_components = int(np.count_nonzero(hierarchy.synthetic_joins)) + 1
        if 1 < n_cluster < n_components:
            raise ValueError(
                f"The feature KNN graph has {n_components} disconnected "
                f"components, which cannot form n_clusters={n_cluster} modules. "
                "Increase n_neighbours or n_clusters."
            )
        return fixed_cut(hierarchy, n_cluster)

    def fix_cluster_order(
        data: "scarf.matrix.ChunkedArray",
        clusters: np.ndarray,
        threads: int,
    ) -> np.ndarray:
        idxmax = compute_with_progress(
            data.argmax(axis=1),
            "Sorting clusters",
            threads,
        )
        cluster_medians = (
            pd.DataFrame([idxmax, clusters]).T.groupby(1).median()[0].sort_values()
        )
        return np.asarray(
            pd.Series(clusters)
            .replace(
                dict(
                    zip(
                        cluster_medians.index,
                        range(1, 1 + len(cluster_medians)),
                        strict=True,
                    )
                )
            )
            .values
        )

    default_ann_params = resolve_aggregation_ann_params(
        ann_params,
        dim=int(d_array.shape[1]),
    )
    default_ann_params.setdefault("max_elements", d_array.shape[0])
    if int(default_ann_params["max_elements"]) < int(d_array.shape[0]):
        raise ValueError("ann_params.max_elements is smaller than the feature count")
    ann_idx = instantiate_knn_index(
        space=str(default_ann_params["space"]),
        dim=int(default_ann_params["dim"]),
        max_elements=int(default_ann_params["max_elements"]),
        ef_construction=int(default_ann_params["ef_construction"]),
        M=int(default_ann_params["M"]),
        random_seed=int(default_ann_params["random_seed"]),
        ef=int(default_ann_params["ef"]),
        nthreads=int(default_ann_params["num_threads"]),
    )
    return fix_cluster_order(
        d_array,
        make_clusters(
            make_knn_mat(d_array, n_neighbours, nthreads),
            n_clusters,
        ),
        nthreads,
    )
