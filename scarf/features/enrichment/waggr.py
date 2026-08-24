from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csc_matrix

from .net import PreparedNetwork

__all__ = [
    "WAGGR_ALGORITHM_VERSION",
    "WaggrModel",
    "build_waggr_model",
    "score_waggr_block",
]

WAGGR_ALGORITHM_VERSION = 1


@dataclass(frozen=True, slots=True)
class WaggrModel:
    adjacency: csc_matrix
    denominator: np.ndarray

    def __post_init__(self) -> None:
        if self.adjacency.ndim != 2:
            raise ValueError("WAGGR adjacency matrix must be two-dimensional")
        if self.denominator.ndim != 1:
            raise ValueError("WAGGR denominator must be one-dimensional")
        if self.adjacency.shape[1] != len(self.denominator):
            raise ValueError("WAGGR adjacency and denominator must be aligned")
        if np.any(self.denominator <= 0) or not np.isfinite(self.denominator).all():
            raise ValueError("WAGGR denominators must be finite and positive")


def build_waggr_model(network: PreparedNetwork) -> WaggrModel:
    """Build a sparse target-by-source matrix for weighted aggregation."""
    feature_positions = np.searchsorted(
        network.matched_feature_index, network.edge_feature_index
    )
    if np.any(feature_positions >= len(network.matched_feature_index)) or np.any(
        network.matched_feature_index[feature_positions] != network.edge_feature_index
    ):
        raise ValueError("Network edges do not align with matched features")
    adjacency = csc_matrix(
        (
            np.asarray(network.edge_weight, dtype=np.float64),
            (
                np.asarray(feature_positions, dtype=np.int64),
                np.asarray(network.edge_source_index, dtype=np.int64),
            ),
        ),
        shape=(len(network.matched_feature_index), len(network.source_names)),
        dtype=np.float64,
    )
    denominator = np.asarray(np.abs(adjacency).sum(axis=0)).reshape(-1)
    denominator = np.asarray(denominator, dtype=np.float64)
    denominator.setflags(write=False)
    return WaggrModel(adjacency=adjacency, denominator=denominator)


def score_waggr_block(
    values: np.ndarray,
    model: WaggrModel,
    *,
    mode: Literal["wmean", "wsum"],
) -> np.ndarray:
    """Score one normalized cell block with weighted sums or means."""
    if mode not in ("wmean", "wsum"):
        raise ValueError("mode must be 'wmean' or 'wsum'")
    raw_values = np.asarray(values)
    if raw_values.ndim != 2:
        raise ValueError("WAGGR values must be two-dimensional")
    if (
        not np.issubdtype(raw_values.dtype, np.number)
        or np.issubdtype(raw_values.dtype, np.complexfloating)
        or not np.isfinite(raw_values).all()
    ):
        raise ValueError("WAGGR values must be finite and numeric")
    matrix = np.asarray(raw_values, dtype=np.float64)
    if matrix.shape[1] != model.adjacency.shape[0]:
        raise ValueError("WAGGR values and adjacency features are not aligned")
    scores = np.asarray(model.adjacency.T.dot(matrix.T).T, dtype=np.float64)
    if mode == "wmean":
        scores = scores / model.denominator.reshape(1, -1)
    if scores.shape != (matrix.shape[0], model.adjacency.shape[1]):
        raise ValueError("WAGGR output shape is invalid")
    if not np.isfinite(scores).all():
        raise ValueError("WAGGR produced non-finite scores")
    return scores
