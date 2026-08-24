from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._rows import read_matrix_rows
from ._types import MatrixData

_CLUSTERING_SCORE_COLUMNS = (
    "clustering",
    "n_sampled_cells",
    "n_clusters",
    "macro_f1_mean",
    "macro_f1_standard_error",
    "weighted_f1_mean",
    "silhouette_score",
    "status",
    "status_reason",
)
_CLUSTER_SCORE_COLUMNS = (
    "clustering",
    "cluster_label",
    "n_sampled_cells",
    "f1_score",
)
_CONFUSION_COLUMNS = (
    "clustering",
    "true_cluster",
    "predicted_cluster",
    "n_cells",
    "fraction_of_true_cluster",
)


@dataclass(frozen=True, slots=True, eq=False)
class ClusterSeparabilityResult:
    clustering_scores: pd.DataFrame = field(repr=False)
    cluster_scores: pd.DataFrame = field(repr=False)
    confusion: pd.DataFrame = field(repr=False)
    sample_indices: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        expected_columns = (
            (self.clustering_scores, _CLUSTERING_SCORE_COLUMNS),
            (self.cluster_scores, _CLUSTER_SCORE_COLUMNS),
            (self.confusion, _CONFUSION_COLUMNS),
        )
        for table, columns in expected_columns:
            if tuple(table.columns) != columns:
                raise ValueError("Cluster separability result columns are invalid")
        indices = np.asarray(self.sample_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("sample_indices must be a one-dimensional integer array")
        indices.setflags(write=False)
        object.__setattr__(self, "sample_indices", indices)


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _validated_clusterings(
    clusterings: Mapping[str, Sequence[object] | np.ndarray],
    n_cells: int,
) -> dict[str, np.ndarray]:
    if not isinstance(clusterings, Mapping):
        raise TypeError("clusterings must be a mapping")
    if not clusterings:
        raise ValueError("clusterings must be non-empty")

    validated: dict[str, np.ndarray] = {}
    for name, labels in clusterings.items():
        if not isinstance(name, str) or not name:
            raise TypeError("clustering names must be non-empty strings")
        values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError(f"Clustering {name!r} labels must be one-dimensional")
        if len(values) != n_cells:
            raise ValueError(
                f"Clustering {name!r} labels must match the coordinate rows"
            )
        if pd.isna(values).any():
            raise ValueError(
                f"Clustering {name!r} labels must not contain missing values"
            )
        validated[name] = values
    return validated


def _stratified_sample_indices(
    labels: np.ndarray,
    max_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_cells = len(labels)
    if n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)

    codes, categories = pd.factorize(labels, sort=False)
    n_clusters = len(categories)
    if max_cells < n_clusters:
        raise ValueError(
            "max_sample_cells must be at least the number of clusters in the "
            "finest clustering"
        )

    counts = np.bincount(codes, minlength=n_clusters)
    targets = counts * (max_cells / n_cells)
    quotas = np.minimum(counts, np.maximum(1, np.floor(targets).astype(np.int64)))

    while int(quotas.sum()) > max_cells:
        candidates = np.flatnonzero(quotas > 1)
        excess = quotas[candidates] - targets[candidates]
        quotas[candidates[int(np.argmax(excess))]] -= 1
    while int(quotas.sum()) < max_cells:
        candidates = np.flatnonzero(quotas < counts)
        deficits = targets[candidates] - quotas[candidates]
        quotas[candidates[int(np.argmax(deficits))]] += 1

    sampled = []
    for cluster_code, quota in enumerate(quotas):
        cluster_indices = np.flatnonzero(codes == cluster_code)
        sampled.append(
            rng.choice(cluster_indices, size=int(quota), replace=False).astype(
                np.int64,
                copy=False,
            )
        )
    return np.sort(np.concatenate(sampled))


def _silhouette_score(
    coordinates: np.ndarray,
    cluster_codes: np.ndarray,
    *,
    max_cells: int,
    random_seed: int,
) -> float:
    from sklearn.metrics import silhouette_score

    if len(cluster_codes) > max_cells:
        if max_cells < int(np.unique(cluster_codes).size):
            return float("nan")
        indices = _stratified_sample_indices(
            cluster_codes,
            max_cells,
            np.random.default_rng(random_seed),
        )
        coordinates = coordinates[indices]
        cluster_codes = cluster_codes[indices]

    n_clusters = int(np.unique(cluster_codes).size)
    if n_clusters < 2 or n_clusters >= len(cluster_codes):
        return float("nan")
    return float(silhouette_score(coordinates, cluster_codes))


def evaluate_cluster_separability(
    coordinates: MatrixData,
    clusterings: Mapping[str, Sequence[object] | np.ndarray],
    *,
    n_folds: int = 5,
    max_sample_cells: int = 50_000,
    max_silhouette_cells: int = 10_000,
    random_seed: int = 4444,
    svm_c: float = 1.0,
    svm_max_iter: int = 10_000,
) -> ClusterSeparabilityResult:
    """Evaluate how well the coordinates recover alternative clusterings."""
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if len(coordinates.shape) != 2:
        raise ValueError("coordinates must be two-dimensional")
    n_cells, n_dimensions = (int(value) for value in coordinates.shape)
    if n_cells < 1 or n_dimensions < 1:
        raise ValueError("coordinates must contain cells and dimensions")

    n_folds = _positive_integer(n_folds, "n_folds", minimum=2)
    max_sample_cells = _positive_integer(max_sample_cells, "max_sample_cells")
    max_silhouette_cells = _positive_integer(
        max_silhouette_cells,
        "max_silhouette_cells",
    )
    random_seed = _positive_integer(random_seed, "random_seed", minimum=0)
    svm_max_iter = _positive_integer(svm_max_iter, "svm_max_iter")
    if isinstance(svm_c, bool) or not isinstance(
        svm_c,
        int | float | np.integer | np.floating,
    ):
        raise TypeError("svm_c must be numeric")
    svm_c = float(svm_c)
    if not np.isfinite(svm_c) or svm_c <= 0:
        raise ValueError("svm_c must be finite and greater than zero")

    clustering_values = _validated_clusterings(clusterings, n_cells)
    finest_name = max(
        clustering_values,
        key=lambda name: int(pd.unique(clustering_values[name]).size),
    )
    sample_indices = _stratified_sample_indices(
        clustering_values[finest_name],
        min(max_sample_cells, n_cells),
        np.random.default_rng(random_seed),
    )
    sampled_coordinates = np.asarray(
        read_matrix_rows(coordinates, sample_indices),
        dtype=np.float64,
    )
    if sampled_coordinates.shape != (len(sample_indices), n_dimensions):
        raise ValueError("Sampled coordinate rows have an unexpected shape")
    if not np.all(np.isfinite(sampled_coordinates)):
        raise ValueError("coordinates must contain only finite values")

    clustering_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for clustering_name, full_labels in clustering_values.items():
        sampled_labels = full_labels[sample_indices]
        cluster_codes, cluster_labels = pd.factorize(sampled_labels, sort=False)
        cluster_codes = np.asarray(cluster_codes, dtype=np.int64)
        cluster_labels = np.asarray(cluster_labels)
        n_clusters = len(cluster_labels)
        cluster_counts = np.bincount(cluster_codes, minlength=n_clusters)
        silhouette = _silhouette_score(
            sampled_coordinates,
            cluster_codes,
            max_cells=max_silhouette_cells,
            random_seed=random_seed,
        )

        reason: str | None = None
        if n_clusters < 2:
            reason = "fewer than two clusters"
        elif int(cluster_counts.min()) < n_folds:
            reason = f"a cluster has fewer than {n_folds} sampled cells"

        if reason is not None:
            clustering_rows.append(
                {
                    "clustering": clustering_name,
                    "n_sampled_cells": len(sample_indices),
                    "n_clusters": n_clusters,
                    "macro_f1_mean": np.nan,
                    "macro_f1_standard_error": np.nan,
                    "weighted_f1_mean": np.nan,
                    "silhouette_score": silhouette,
                    "status": "unscorable",
                    "status_reason": reason,
                }
            )
            for cluster_label, count in zip(
                cluster_labels,
                cluster_counts,
                strict=True,
            ):
                cluster_rows.append(
                    {
                        "clustering": clustering_name,
                        "cluster_label": cluster_label,
                        "n_sampled_cells": int(count),
                        "f1_score": np.nan,
                    }
                )
            continue

        cross_validation = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        out_of_fold = np.full(len(cluster_codes), -1, dtype=np.int64)
        macro_f1: list[float] = []
        weighted_f1: list[float] = []
        for train_indices, test_indices in cross_validation.split(
            sampled_coordinates,
            cluster_codes,
        ):
            estimator = make_pipeline(
                StandardScaler(),
                LinearSVC(
                    C=svm_c,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=svm_max_iter,
                    random_state=random_seed,
                ),
            )
            estimator.fit(
                sampled_coordinates[train_indices],
                cluster_codes[train_indices],
            )
            predictions = estimator.predict(sampled_coordinates[test_indices])
            out_of_fold[test_indices] = predictions
            macro_f1.append(
                float(
                    f1_score(
                        cluster_codes[test_indices],
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            weighted_f1.append(
                float(
                    f1_score(
                        cluster_codes[test_indices],
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                )
            )
        if np.any(out_of_fold < 0):
            raise RuntimeError("Cross-validation did not predict every sampled cell")

        per_cluster = precision_recall_fscore_support(
            cluster_codes,
            out_of_fold,
            labels=np.arange(n_clusters),
            zero_division=0,
        )
        for cluster_label, count, score in zip(
            cluster_labels,
            per_cluster[3],
            per_cluster[2],
            strict=True,
        ):
            cluster_rows.append(
                {
                    "clustering": clustering_name,
                    "cluster_label": cluster_label,
                    "n_sampled_cells": int(count),
                    "f1_score": float(score),
                }
            )

        counts = confusion_matrix(
            cluster_codes,
            out_of_fold,
            labels=np.arange(n_clusters),
        )
        row_totals = counts.sum(axis=1)
        rates = np.divide(
            counts,
            row_totals[:, None],
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_totals[:, None] > 0,
        )
        for true_code, true_label in enumerate(cluster_labels):
            for predicted_code, predicted_label in enumerate(cluster_labels):
                confusion_rows.append(
                    {
                        "clustering": clustering_name,
                        "true_cluster": true_label,
                        "predicted_cluster": predicted_label,
                        "n_cells": int(counts[true_code, predicted_code]),
                        "fraction_of_true_cluster": float(
                            rates[true_code, predicted_code]
                        ),
                    }
                )

        macro_values = np.asarray(macro_f1)
        clustering_rows.append(
            {
                "clustering": clustering_name,
                "n_sampled_cells": len(sample_indices),
                "n_clusters": n_clusters,
                "macro_f1_mean": float(macro_values.mean()),
                "macro_f1_standard_error": float(
                    macro_values.std(ddof=1) / np.sqrt(n_folds)
                ),
                "weighted_f1_mean": float(np.mean(weighted_f1)),
                "silhouette_score": silhouette,
                "status": "scored",
                "status_reason": None,
            }
        )

    return ClusterSeparabilityResult(
        clustering_scores=pd.DataFrame(
            clustering_rows,
            columns=_CLUSTERING_SCORE_COLUMNS,
        ),
        cluster_scores=pd.DataFrame(
            cluster_rows,
            columns=_CLUSTER_SCORE_COLUMNS,
        ),
        confusion=pd.DataFrame(
            confusion_rows,
            columns=_CONFUSION_COLUMNS,
        ),
        sample_indices=sample_indices,
    )
