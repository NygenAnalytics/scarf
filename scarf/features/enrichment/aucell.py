from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .net import PreparedNetwork

__all__ = [
    "AUCELL_ALGORITHM_VERSION",
    "GeneSetIndex",
    "build_gene_set_index",
    "make_rank_permutation",
    "resolve_n_up",
    "score_aucell_block",
]

AUCELL_ALGORITHM_VERSION = 1


def _owned_readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GeneSetIndex:
    connections: np.ndarray
    starts: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.connections.ndim != 1
            or self.starts.ndim != 1
            or self.offsets.ndim != 1
        ):
            raise ValueError("Gene-set index arrays must be one-dimensional")
        if len(self.starts) == 0 or len(self.starts) != len(self.offsets):
            raise ValueError(
                "Gene-set starts and offsets must be non-empty and aligned"
            )
        if any(
            not np.issubdtype(values.dtype, np.integer)
            for values in (self.connections, self.starts, self.offsets)
        ):
            raise ValueError("Gene-set index arrays must have integer dtypes")
        if np.any(self.offsets <= 0):
            raise ValueError("Gene-set offsets must be positive")
        if np.any(self.connections < 0) or np.any(self.starts < 0):
            raise ValueError("Gene-set indices must be non-negative")
        if int(self.offsets.sum()) != len(self.connections):
            raise ValueError("Gene-set connections do not match offsets")
        expected_starts = np.zeros(len(self.offsets), dtype=np.int64)
        expected_starts[1:] = np.cumsum(self.offsets[:-1])
        if not np.array_equal(self.starts, expected_starts):
            raise ValueError("Gene-set starts are invalid")
        for start, size in zip(self.starts, self.offsets, strict=True):
            values = self.connections[start : start + size]
            if np.unique(values).size != len(values):
                raise ValueError("Gene sets must not contain duplicate connections")


def resolve_n_up(n_features: int, n_up: int | None) -> int:
    """Resolve the number of top-ranked features used for AUCell."""
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise TypeError("n_features must be an integer")
    if n_features < 2:
        raise ValueError("AUCell requires at least two ranking features")
    if n_up is None:
        return int(np.clip(np.ceil(0.05 * n_features), 2, n_features))
    if isinstance(n_up, bool) or not isinstance(n_up, int):
        raise TypeError("n_up must be an integer or None")
    if not 1 < n_up <= n_features:
        raise ValueError(
            f"n_up must be greater than 1 and at most {n_features}, got {n_up}"
        )
    return n_up


def make_rank_permutation(n_features: int, tie_seed: int) -> np.ndarray:
    """Create the one global feature permutation used to break rank ties."""
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise TypeError("n_features must be an integer")
    if n_features < 2:
        raise ValueError("AUCell requires at least two ranking features")
    if isinstance(tie_seed, bool) or not isinstance(tie_seed, int):
        raise TypeError("tie_seed must be an integer")
    if tie_seed < 0:
        raise ValueError("tie_seed must be non-negative")
    permutation = np.random.default_rng(tie_seed).permutation(n_features)
    return _owned_readonly(permutation)


def build_gene_set_index(
    network: PreparedNetwork,
    rank_feature_index: np.ndarray,
) -> GeneSetIndex:
    """Index source targets by position in the permuted ranking universe."""
    rank_index = np.asarray(rank_feature_index)
    if rank_index.ndim != 1 or len(rank_index) < 2:
        raise ValueError("rank_feature_index must contain at least two features")
    if not np.issubdtype(rank_index.dtype, np.integer):
        raise ValueError("rank_feature_index must have an integer dtype")
    rank_index = np.asarray(rank_index, dtype=np.int64)
    if np.any(rank_index < 0):
        raise ValueError("rank_feature_index must be non-negative")
    if np.unique(rank_index).size != len(rank_index):
        raise ValueError("rank_feature_index must contain unique feature indices")
    positions = {int(feature): pos for pos, feature in enumerate(rank_index)}
    per_source: list[list[int]] = [[] for _ in network.source_names]
    for source, feature in zip(
        network.edge_source_index,
        network.edge_feature_index,
        strict=True,
    ):
        try:
            position = positions[int(feature)]
        except KeyError as exc:
            raise ValueError(
                "Network target is absent from the AUCell ranking universe"
            ) from exc
        per_source[int(source)].append(position)

    offsets = np.asarray([len(values) for values in per_source], dtype=np.int64)
    if not np.array_equal(offsets, network.source_sizes):
        raise ValueError("Gene-set sizes do not align with prepared network sources")
    starts = np.zeros(len(offsets), dtype=np.int64)
    starts[1:] = np.cumsum(offsets[:-1])
    connections = np.concatenate(
        [np.asarray(sorted(values), dtype=np.int64) for values in per_source]
    )
    return GeneSetIndex(
        connections=_owned_readonly(connections),
        starts=_owned_readonly(starts),
        offsets=_owned_readonly(offsets),
    )


@njit(cache=True, parallel=True)
def _score_ranked_row(
    ranks: np.ndarray,
    connections: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int,
) -> np.ndarray:
    n_sources = starts.size
    scores = np.zeros(n_sources, dtype=np.float64)
    for source in prange(n_sources):
        start = starts[source]
        size = offsets[source]

        max_auc = 0.0
        threshold_count = min(size, n_up - 1)
        for threshold in range(1, threshold_count + 1):
            next_threshold = threshold + 1
            if threshold == threshold_count:
                next_threshold = n_up
            max_auc += (next_threshold - threshold) * threshold

        selected = np.empty(size, dtype=np.int64)
        selected_count = 0
        for offset in range(size):
            rank = ranks[connections[start + offset]]
            if rank <= n_up:
                selected[selected_count] = rank
                selected_count += 1
        if selected_count == 0:
            continue
        selected_values = np.sort(selected[:selected_count])
        area = 0.0
        for index in range(selected_count):
            left = selected_values[index]
            right = n_up
            if index + 1 < selected_count:
                right = selected_values[index + 1]
            area += (right - left) * (index + 1)
        scores[source] = area / max_auc
    return scores


def score_aucell_block(
    values: np.ndarray,
    permutation: np.ndarray,
    sets: GeneSetIndex,
    *,
    n_up: int,
) -> np.ndarray:
    """Score one raw count block using AUCell recovery curves."""
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("AUCell values must be two-dimensional")
    if (
        not np.issubdtype(matrix.dtype, np.number)
        or np.issubdtype(matrix.dtype, np.complexfloating)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("AUCell values must be finite and numeric")
    permutation_array = np.asarray(permutation)
    if permutation_array.ndim != 1 or len(permutation_array) != matrix.shape[1]:
        raise ValueError("AUCell permutation must align with matrix features")
    if not np.issubdtype(permutation_array.dtype, np.integer):
        raise ValueError("AUCell permutation must have an integer dtype")
    permutation_array = np.asarray(permutation_array, dtype=np.int64)
    if not np.array_equal(
        np.sort(permutation_array), np.arange(matrix.shape[1], dtype=np.int64)
    ):
        raise ValueError("AUCell permutation must contain every feature position once")
    resolved_n_up = resolve_n_up(matrix.shape[1], n_up)
    if np.any(sets.connections < 0) or np.any(sets.connections >= matrix.shape[1]):
        raise ValueError("Gene-set connections are outside the ranking universe")

    scores = np.zeros((matrix.shape[0], len(sets.starts)), dtype=np.float64)
    ordinal = np.arange(1, matrix.shape[1] + 1, dtype=np.int64)
    for row_index, row in enumerate(matrix):
        if not np.any(row != 0):
            continue
        permuted = np.asarray(row[permutation_array], dtype=np.float64)
        order = np.argsort(-permuted, kind="stable")
        ranks = np.empty(matrix.shape[1], dtype=np.int64)
        ranks[order] = ordinal
        scores[row_index] = _score_ranked_row(
            ranks,
            sets.connections,
            sets.starts,
            sets.offsets,
            resolved_n_up,
        )

    tolerance = 1e-12
    if np.any(scores < -tolerance) or np.any(scores > 1.0 + tolerance):
        raise ValueError("AUCell produced scores outside the expected [0, 1] range")
    return np.clip(scores, 0.0, 1.0)
