from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..matrix import ChunkedArray
from ..storage.refs import ArtifactRef


def _immutable_array(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values))
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


@dataclass(frozen=True, slots=True)
class _PseudotimeAggregationFeatureIdentity:
    names: np.ndarray
    ids: np.ndarray

    def __post_init__(self) -> None:
        names = np.asarray(self.names)
        ids = np.asarray(self.ids)
        if names.ndim != 1 or ids.ndim != 1 or names.shape != ids.shape:
            raise ValueError("Frozen feature names and IDs must align")
        object.__setattr__(self, "names", _immutable_array(names))
        object.__setattr__(self, "ids", _immutable_array(ids))


class _PseudotimeAggregationIdentityCarrier:
    __slots__ = ("_feature_identity",)


@dataclass(frozen=True, slots=True, eq=False)
class FateMappingResult:
    """Fate probabilities loaded from an immutable artifact."""

    ref: ArtifactRef
    graph: ArtifactRef
    pseudotime: ArtifactRef
    sink_labels_artifact: ArtifactRef
    cell_selection: ArtifactRef
    sink_labels: tuple[Any, ...]
    values: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Fate probabilities must be two-dimensional")
        if self.valid.ndim != 1:
            raise ValueError("Fate validity must be one-dimensional")
        if self.values.shape[0] != self.valid.shape[0]:
            raise ValueError("Fate probabilities and validity rows must align")
        n_sinks = self.values.shape[1]
        if n_sinks != len(self.sink_labels):
            raise ValueError("Fate probability columns and sink labels must align")


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeScoreResult:
    """Pseudotime values loaded from an immutable artifact."""

    ref: ArtifactRef
    graph: ArtifactRef
    cell_selection: ArtifactRef
    values: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.valid.ndim != 1:
            raise ValueError("Pseudotime values and validity must be one-dimensional")
        if self.values.shape != self.valid.shape:
            raise ValueError("Pseudotime values and validity must have the same shape")


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeMarkerResult:
    """Pseudotime correlation table loaded from an immutable artifact."""

    ref: ArtifactRef
    table: pd.DataFrame = field(repr=False)
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    pseudotime: ArtifactRef

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_selection, ArtifactRef)
            or self.feature_selection.kind != "feature_selection"
            or self.feature_selection.scope != "assay"
            or self.feature_selection.assay != self.assay
        ):
            raise ValueError(
                "Pseudotime marker feature selection must belong to its assay"
            )
        required = {
            "feature_index",
            "feature_name",
            "r_value",
            "p_value",
        }
        missing = required.difference(self.table.columns)
        if missing:
            raise ValueError(
                "Pseudotime marker table is missing columns: "
                + ", ".join(sorted(missing))
            )


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeAggregationResult(_PseudotimeAggregationIdentityCarrier):
    """Lazy pseudotime aggregation loaded from an immutable artifact."""

    ref: ArtifactRef
    data: ChunkedArray = field(repr=False)
    feature_indices: np.ndarray = field(repr=False)
    feature_clusters: np.ndarray = field(repr=False)
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    pseudotime: ArtifactRef

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_selection, ArtifactRef)
            or self.feature_selection.kind != "feature_selection"
            or self.feature_selection.scope != "assay"
            or self.feature_selection.assay != self.assay
        ):
            raise ValueError(
                "Pseudotime aggregation feature selection must belong to its assay"
            )
        if len(self.data.shape) != 2:
            raise ValueError("Pseudotime aggregation data must be two-dimensional")
        if self.feature_indices.ndim != 1 or self.feature_clusters.ndim != 1:
            raise ValueError("Feature indices and clusters must be one-dimensional")
        n_features = int(self.data.shape[0])
        if len(self.feature_indices) != n_features:
            raise ValueError("Feature indices do not align with aggregation rows")
        if len(self.feature_clusters) != n_features:
            raise ValueError("Feature clusters do not align with aggregation rows")

    def _attach_feature_identity(
        self,
        names: np.ndarray,
        ids: np.ndarray,
    ) -> None:
        if hasattr(self, "_feature_identity"):
            raise RuntimeError("Frozen feature identity is already attached")
        identity = _PseudotimeAggregationFeatureIdentity(names=names, ids=ids)
        if identity.names.shape != (int(self.data.shape[0]),):
            raise ValueError(
                "Frozen feature identity does not align with aggregation rows"
            )
        object.__setattr__(self, "_feature_identity", identity)

    @property
    def feature_names(self) -> np.ndarray:
        """Frozen feature names aligned to the aggregation rows."""
        identity = getattr(self, "_feature_identity", None)
        if not isinstance(identity, _PseudotimeAggregationFeatureIdentity):
            raise RuntimeError(
                "Frozen feature names are available on results loaded with "
                "DataStore.load_pseudotime_aggregation"
            )
        return identity.names

    @property
    def feature_ids(self) -> np.ndarray:
        """Frozen feature IDs aligned to the aggregation rows."""
        identity = getattr(self, "_feature_identity", None)
        if not isinstance(identity, _PseudotimeAggregationFeatureIdentity):
            raise RuntimeError(
                "Frozen feature IDs are available on results loaded with "
                "DataStore.load_pseudotime_aggregation"
            )
        return identity.ids
