from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
import zarr

from ._rows import read_matrix_rows
from ._types import MatrixData

type ClusterLabels = np.ndarray | zarr.Array
type ClusterCandidate = tuple[str, ClusterLabels]

SHARED_CLUSTER_QUOTA_STRATEGY = "sharedClusterQuota"
DEFAULT_MIN_CLUSTER_QUOTA = 2


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _read_all_labels(
    labels: ClusterLabels,
    *,
    expected_rows: int,
) -> np.ndarray:
    if len(labels.shape) != 1:
        raise ValueError("candidate labels must be one-dimensional")
    if int(labels.shape[0]) != expected_rows:
        raise ValueError(
            f"candidate has {labels.shape[0]} labels for {expected_rows} coordinate rows"
        )
    return np.asarray(labels[:])


def shared_cluster_quota_sample_indices(
    candidates: Sequence[ClusterCandidate],
    *,
    n_cells: int,
    seed: int,
    max_sample_size: int,
    min_cluster_quota: int = DEFAULT_MIN_CLUSTER_QUOTA,
    checkpoint: Callable[[], None] | None = None,
) -> np.ndarray:
    """Return one seeded sample that covers every cluster in every candidate."""
    n_cells = _integer(n_cells, "n_cells", minimum=1)
    seed = _integer(seed, "seed", minimum=0)
    max_sample_size = _integer(max_sample_size, "max_sample_size", minimum=1)
    min_cluster_quota = _integer(
        min_cluster_quota,
        "min_cluster_quota",
        minimum=1,
    )
    if n_cells <= max_sample_size:
        return np.arange(n_cells, dtype=np.int64)

    rng = np.random.default_rng(seed)
    required: set[int] = set()
    for candidate in candidates:
        if checkpoint is not None:
            checkpoint()
        if not isinstance(candidate, tuple) or len(candidate) != 2:
            raise TypeError("candidates must contain (key, labels) tuples")
        _key, labels = candidate
        values = _read_all_labels(labels, expected_rows=n_cells)
        unique_labels = np.unique(values)
        for label in unique_labels:
            cluster_indices = np.flatnonzero(values == label)
            quota = min(int(cluster_indices.size), min_cluster_quota)
            chosen = rng.choice(cluster_indices, size=quota, replace=False)
            required.update(int(index) for index in chosen)
    if len(required) > max_sample_size:
        raise ValueError(
            "Shared cluster-quota sample requires "
            f"{len(required)} cells, which exceeds max_sample_size={max_sample_size}"
        )
    remaining = max_sample_size - len(required)
    if remaining > 0:
        pool = np.fromiter(
            (index for index in range(n_cells) if index not in required),
            dtype=np.int64,
            count=n_cells - len(required),
        )
        extra = rng.choice(pool, size=remaining, replace=False)
        required.update(int(index) for index in extra)
    return np.sort(np.fromiter(required, dtype=np.int64, count=len(required)))


@dataclass(frozen=True, slots=True, eq=False)
class ClusterSelectionResult:
    """Immutable outcome of a bounded silhouette comparison."""

    candidate_keys: tuple[str, ...]
    sample_indices: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)
    invalid_reasons: tuple[str | None, ...]
    selected_key: str
    seed: int
    population_size: int
    max_sample_size: int
    working_memory_mib: int
    sample_strategy: str = SHARED_CLUSTER_QUOTA_STRATEGY
    min_cluster_quota: int = DEFAULT_MIN_CLUSTER_QUOTA

    def __post_init__(self) -> None:
        candidate_keys = tuple(self.candidate_keys)
        if not candidate_keys:
            raise ValueError("candidate_keys must be non-empty")
        if any(not isinstance(key, str) or not key for key in candidate_keys):
            raise TypeError("candidate_keys must contain non-empty strings")
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("candidate_keys must be unique")

        raw_indices = np.asarray(self.sample_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(
            raw_indices.dtype,
            np.integer,
        ):
            raise TypeError("sample_indices must be a one-dimensional integer array")
        copied_indices = np.array(raw_indices, dtype=np.int64, copy=True)
        sample_indices = np.frombuffer(
            copied_indices.tobytes(),
            dtype=np.int64,
        )

        raw_scores = np.asarray(self.scores)
        if raw_scores.ndim != 1 or not np.issubdtype(
            raw_scores.dtype,
            np.floating,
        ):
            raise TypeError("scores must be a one-dimensional floating-point array")
        copied_scores = np.array(raw_scores, dtype=np.float64, copy=True)
        scores = np.frombuffer(copied_scores.tobytes(), dtype=np.float64)
        if len(scores) != len(candidate_keys):
            raise ValueError("scores must align with candidate_keys")

        invalid_reasons = tuple(self.invalid_reasons)
        if len(invalid_reasons) != len(candidate_keys):
            raise ValueError("invalid_reasons must align with candidate_keys")
        for score, reason in zip(scores, invalid_reasons, strict=True):
            if np.isfinite(score):
                if reason is not None:
                    raise ValueError("finite scores must not have an invalid reason")
            elif not np.isnan(score):
                raise ValueError("invalid scores must be represented by NaN")
            elif not isinstance(reason, str) or not reason:
                raise ValueError("invalid scores must have a non-empty reason")

        seed = _integer(self.seed, "seed", minimum=0)
        population_size = _integer(
            self.population_size,
            "population_size",
            minimum=1,
        )
        max_sample_size = _integer(
            self.max_sample_size,
            "max_sample_size",
            minimum=1,
        )
        working_memory_mib = _integer(
            self.working_memory_mib,
            "working_memory_mib",
            minimum=1,
        )
        if (
            not isinstance(self.sample_strategy, str)
            or self.sample_strategy != SHARED_CLUSTER_QUOTA_STRATEGY
        ):
            raise ValueError(
                f"sample_strategy must be {SHARED_CLUSTER_QUOTA_STRATEGY!r}"
            )
        min_cluster_quota = _integer(
            self.min_cluster_quota,
            "min_cluster_quota",
            minimum=1,
        )
        expected_sample_size = min(population_size, max_sample_size)
        if len(sample_indices) != expected_sample_size:
            raise ValueError("sample_indices has an unexpected size")
        if (
            np.any(sample_indices < 0)
            or np.any(sample_indices >= population_size)
            or np.any(sample_indices[1:] <= sample_indices[:-1])
        ):
            raise ValueError(
                "sample_indices must be sorted, unique, and within the population"
            )

        valid_indices = np.flatnonzero(np.isfinite(scores))
        if valid_indices.size == 0:
            raise ValueError("scores must contain at least one finite value")
        selected_index = int(valid_indices[0])
        for index in valid_indices[1:]:
            if scores[int(index)] > scores[selected_index]:
                selected_index = int(index)
        if self.selected_key != candidate_keys[selected_index]:
            raise ValueError(
                "selected_key must be the first candidate with the maximum score"
            )

        object.__setattr__(self, "candidate_keys", candidate_keys)
        object.__setattr__(self, "sample_indices", sample_indices)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "invalid_reasons", invalid_reasons)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "population_size", population_size)
        object.__setattr__(self, "max_sample_size", max_sample_size)
        object.__setattr__(self, "working_memory_mib", working_memory_mib)
        object.__setattr__(self, "min_cluster_quota", min_cluster_quota)

    @property
    def metric(self) -> Literal["euclidean"]:
        return "euclidean"

    @property
    def sample_size(self) -> int:
        return len(self.sample_indices)

    @property
    def sample_definition(self) -> Mapping[str, int | str]:
        return MappingProxyType(
            {
                "seed": self.seed,
                "populationSize": self.population_size,
                "sampleSize": self.sample_size,
                "maxSampleSize": self.max_sample_size,
                "sampleStrategy": self.sample_strategy,
                "minClusterQuota": self.min_cluster_quota,
            }
        )

    @property
    def tie_order(self) -> tuple[str, ...]:
        return self.candidate_keys


def _read_label_rows(
    labels: ClusterLabels,
    sample_indices: np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    if len(labels.shape) != 1:
        raise ValueError("candidate labels must be one-dimensional")
    if int(labels.shape[0]) != expected_rows:
        raise ValueError(
            f"candidate has {labels.shape[0]} labels for {expected_rows} coordinate rows"
        )
    if isinstance(labels, zarr.Array):
        return np.asarray(labels.get_orthogonal_selection((sample_indices,)))
    return np.asarray(labels[sample_indices])


def select_clusters_by_silhouette(
    coordinates: MatrixData,
    candidates: Sequence[ClusterCandidate],
    *,
    seed: int = 4466,
    max_sample_size: int = 10_000,
    working_memory_mib: int = 1024,
    min_cluster_quota: int = DEFAULT_MIN_CLUSTER_QUOTA,
    checkpoint: Callable[[], None] | None = None,
) -> ClusterSelectionResult:
    """Choose the first maximum silhouette on one shared cluster-quota sample."""
    if len(coordinates.shape) != 2:
        raise ValueError("Coordinates must be two-dimensional")
    n_cells = int(coordinates.shape[0])
    if n_cells <= 0:
        raise ValueError("Cluster selection has no coordinate rows to sample")
    if int(coordinates.shape[1]) <= 0:
        raise ValueError("Coordinates must contain at least one dimension")

    seed = _integer(seed, "seed", minimum=0)
    max_sample_size = _integer(
        max_sample_size,
        "max_sample_size",
        minimum=1,
    )
    working_memory_mib = _integer(
        working_memory_mib,
        "working_memory_mib",
        minimum=1,
    )
    min_cluster_quota = _integer(
        min_cluster_quota,
        "min_cluster_quota",
        minimum=1,
    )
    resolved_candidates = tuple(candidates)
    if not resolved_candidates:
        raise ValueError("Cluster selection requires at least one candidate")
    candidate_keys: list[str] = []
    for candidate in resolved_candidates:
        if not isinstance(candidate, tuple) or len(candidate) != 2:
            raise TypeError("candidates must contain (key, labels) tuples")
        key, _labels = candidate
        if not isinstance(key, str) or not key:
            raise TypeError(
                "Cluster selection candidate keys must be non-empty strings"
            )
        candidate_keys.append(key)
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Cluster selection candidate keys must be unique")

    sample_indices = shared_cluster_quota_sample_indices(
        resolved_candidates,
        n_cells=n_cells,
        seed=seed,
        max_sample_size=max_sample_size,
        min_cluster_quota=min_cluster_quota,
        checkpoint=checkpoint,
    )
    sample_size = len(sample_indices)
    sampled_coordinates = np.asarray(
        read_matrix_rows(coordinates, sample_indices),
        dtype=np.float64,
    )
    if sampled_coordinates.shape != (sample_size, int(coordinates.shape[1])):
        raise ValueError("Sampled coordinates have an unexpected shape")
    coordinate_error = (
        None
        if np.all(np.isfinite(sampled_coordinates))
        else "sampled coordinates contain non-finite values"
    )
    scores = np.full(len(resolved_candidates), np.nan, dtype=np.float64)
    invalid_reasons: list[str | None] = []
    if coordinate_error is None:
        from sklearn import config_context
        from sklearn.metrics import silhouette_score

    for index, (_key, labels) in enumerate(resolved_candidates):
        if checkpoint is not None:
            checkpoint()
        reason = coordinate_error
        try:
            sampled_labels = _read_label_rows(
                labels,
                sample_indices,
                expected_rows=n_cells,
            )
            unique_count = int(np.unique(sampled_labels).size)
            if unique_count < 2:
                reason = "sample contains fewer than two clusters"
            elif unique_count >= sample_size:
                reason = "every sampled cell has a distinct cluster"
            elif reason is None:
                if checkpoint is not None:
                    checkpoint()
                with config_context(working_memory=working_memory_mib):
                    score = float(
                        silhouette_score(
                            sampled_coordinates,
                            sampled_labels,
                            metric="euclidean",
                        )
                    )
                if checkpoint is not None:
                    checkpoint()
                if not np.isfinite(score):
                    reason = "silhouette score is not finite"
                else:
                    scores[index] = score
        except (TypeError, ValueError, RuntimeError) as error:
            reason = str(error) or type(error).__name__
        invalid_reasons.append(reason)

    valid_indices = np.flatnonzero(np.isfinite(scores))
    if valid_indices.size == 0:
        details = "; ".join(
            f"{key}: {reason or 'not scoreable'}"
            for key, reason in zip(candidate_keys, invalid_reasons, strict=True)
        )
        raise ValueError(f"No clustering candidate is silhouette-scoreable: {details}")
    selected_index = int(valid_indices[0])
    for index in valid_indices[1:]:
        if scores[int(index)] > scores[selected_index]:
            selected_index = int(index)

    return ClusterSelectionResult(
        candidate_keys=tuple(candidate_keys),
        sample_indices=sample_indices,
        scores=scores,
        invalid_reasons=tuple(invalid_reasons),
        selected_key=candidate_keys[selected_index],
        seed=seed,
        population_size=n_cells,
        max_sample_size=max_sample_size,
        working_memory_mib=working_memory_mib,
        min_cluster_quota=min_cluster_quota,
    )
