from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..matrix import ChunkedArray
from ..storage.refs import ArtifactRef


@dataclass(frozen=True, slots=True, eq=False)
class FateMappingResult:
    """Saved fate probabilities aligned to cells selected by result_cell_key."""

    fate_keys: tuple[str, ...]
    validity_key: str
    sink_labels: tuple[Any, ...]
    assay: str
    graph_cell_key: str
    result_cell_key: str
    graph: ArtifactRef
    pseudotime_key: str
    sink_key: str
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
        if n_sinks != len(self.fate_keys) or n_sinks != len(self.sink_labels):
            raise ValueError(
                "Fate probability columns, keys, and sink labels must align"
            )


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeScoreResult:
    """Saved pseudotime values and their metadata keys."""

    pseudotime_key: str
    validity_key: str
    assay: str
    graph_cell_key: str
    result_cell_key: str
    graph: ArtifactRef
    values: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.valid.ndim != 1:
            raise ValueError("Pseudotime values and validity must be one-dimensional")
        if self.values.shape != self.valid.shape:
            raise ValueError("Pseudotime values and validity must have the same shape")


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeMarkerResult:
    """Pseudotime correlation table and saved feature-metadata keys."""

    table: pd.DataFrame = field(repr=False)
    correlation_key: str
    p_value_key: str
    assay: str
    cell_key: str
    feature_selection: ArtifactRef
    pseudotime_key: str

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

    @property
    def p_value_adjusted_key(self) -> str | None:
        if (
            "p_value_adjusted" not in self.table.columns
            or not self.p_value_key.endswith("__p")
        ):
            return None
        return f"{self.p_value_key[:-3]}__padj"


@dataclass(frozen=True, slots=True, eq=False)
class PseudotimeAggregationResult:
    """Lazy pseudotime aggregation with aligned feature metadata."""

    data: ChunkedArray = field(repr=False)
    feature_indices: np.ndarray = field(repr=False)
    feature_clusters: np.ndarray = field(repr=False)
    cluster_key: str
    storage_path: str
    assay: str
    cell_key: str
    feature_selection: ArtifactRef
    pseudotime_key: str

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
