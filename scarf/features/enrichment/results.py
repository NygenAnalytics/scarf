from dataclasses import dataclass, field

import numpy as np

from ...matrix import ChunkedArray
from ...storage.refs import ArtifactRef

__all__ = ["EnrichmentResult"]


@dataclass(frozen=True, slots=True, eq=False)
class EnrichmentResult:
    """A persisted gene-set enrichment result.

    Attributes:
        data: Lazy score matrix with cells as rows and sources as columns.
        source_names: Source names aligned to the score columns.
        source_sizes: Matched target counts aligned to ``source_names``.
        cell_index: Assay cell indices aligned to the score rows.
        artifact: Immutable enrichment artifact backing the result.
        storage_path: Zarr path that owns the result.
        assay: Name of the RNA assay used for scoring.
        cell_selection: Immutable cell selection used for scoring.
        feature_selection: Feature-selection artifact used for scoring.
        method: Enrichment method, either ``"waggr"`` or ``"aucell"``.
    """

    data: ChunkedArray = field(repr=False)
    source_names: np.ndarray = field(repr=False)
    source_sizes: np.ndarray = field(repr=False)
    cell_index: np.ndarray = field(repr=False)
    artifact: ArtifactRef
    storage_path: str
    assay: str
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    method: str

    def __post_init__(self) -> None:
        if (
            self.artifact.kind != "enrichment_scores"
            or self.artifact.scope != "assay"
            or self.artifact.assay != self.assay
        ):
            raise ValueError("Enrichment artifact must belong to the result assay")
        if (
            not isinstance(self.feature_selection, ArtifactRef)
            or self.feature_selection.kind != "feature_selection"
            or self.feature_selection.scope != "assay"
            or self.feature_selection.assay != self.assay
        ):
            raise ValueError(
                "Enrichment feature_selection must belong to the result assay"
            )
        if (
            not isinstance(self.cell_selection, ArtifactRef)
            or self.cell_selection.kind != "cell_selection"
            or self.cell_selection.scope != "datastore"
            or self.cell_selection.assay is not None
        ):
            raise ValueError(
                "Enrichment cell_selection must be a datastore cell selection"
            )
        if len(self.data.shape) != 2:
            raise ValueError("Enrichment data must be two-dimensional")
        if (
            self.source_names.ndim != 1
            or self.source_sizes.ndim != 1
            or self.cell_index.ndim != 1
        ):
            raise ValueError("Enrichment metadata arrays must be one-dimensional")
        if self.data.shape[0] != len(self.cell_index):
            raise ValueError("Enrichment cell indices do not align with score rows")
        if self.data.shape[1] != len(self.source_names):
            raise ValueError("Enrichment source names do not align with score columns")
        if len(self.source_names) != len(self.source_sizes):
            raise ValueError("Enrichment source names and sizes must be aligned")
        if np.unique(self.source_names).size != len(self.source_names):
            raise ValueError("Enrichment source names must be unique")
        if np.unique(self.cell_index).size != len(self.cell_index):
            raise ValueError("Enrichment cell indices must be unique")
        if np.any(self.source_sizes <= 0):
            raise ValueError("Enrichment source sizes must be positive")
        if self.method not in {"waggr", "aucell"}:
            raise ValueError(f"Unknown enrichment method: {self.method!r}")
