from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..storage.refs import ArtifactRef

if TYPE_CHECKING:
    from .reference import MappingReference


@dataclass(frozen=True)
class ScaledPCAProjectionModel:
    feature_means: np.ndarray
    feature_scales: np.ndarray
    loadings: np.ndarray

    def __post_init__(self) -> None:
        if self.loadings.ndim != 2:
            raise ValueError("Reference PCA loadings must be two-dimensional")
        n_features = self.loadings.shape[0]
        if self.feature_means.shape != (n_features,):
            raise ValueError("Reference feature means have incompatible dimensions")
        if self.feature_scales.shape != (n_features,):
            raise ValueError("Reference feature scales have incompatible dimensions")
        if np.any(self.feature_scales <= 0):
            raise ValueError("Reference feature scales must be positive")
        for values in (
            self.feature_means,
            self.feature_scales,
            self.loadings,
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "Reference projection model contains non-finite values"
                )

    @property
    def n_features(self) -> int:
        return int(self.loadings.shape[0])

    @property
    def n_dims(self) -> int:
        return int(self.loadings.shape[1])


@dataclass(frozen=True)
class SymphonyCorrectionModel:
    centroids: np.ndarray
    raw_centroids: np.ndarray
    corrected_centroids: np.ndarray
    cluster_mass: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        if self.centroids.ndim != 2:
            raise ValueError("Reference centroids must be two-dimensional")
        n_clusters = self.centroids.shape[0]
        n_dims = self.centroids.shape[1]
        if self.raw_centroids.shape != (n_clusters, n_dims):
            raise ValueError("Reference raw centroids have incompatible dimensions")
        if self.corrected_centroids.shape != (n_clusters, n_dims):
            raise ValueError(
                "Reference corrected centroids have incompatible dimensions"
            )
        if self.cluster_mass.shape != (n_clusters,) or np.any(self.cluster_mass <= 0):
            raise ValueError("Reference cluster masses must be positive")
        if self.sigma.shape != (n_clusters,) or np.any(self.sigma <= 0):
            raise ValueError("Reference kernel widths must be positive")
        for values in (
            self.centroids,
            self.raw_centroids,
            self.corrected_centroids,
            self.cluster_mass,
            self.sigma,
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError("Symphony correction model contains non-finite values")

    @property
    def n_dims(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])


@dataclass(frozen=True)
class QueryCorrection:
    batch_offsets: np.ndarray
    batch_counts: np.ndarray

    def __post_init__(self) -> None:
        if self.batch_offsets.ndim != 3:
            raise ValueError(
                "Batch offsets must have batch, cluster, and dimension axes"
            )
        if self.batch_counts.shape != self.batch_offsets.shape[:2]:
            raise ValueError("Batch counts must match batch offsets")
        if not np.all(np.isfinite(self.batch_offsets)):
            raise ValueError("Batch offsets contain non-finite values")


@dataclass(frozen=True)
class MappingResult:
    ref: ArtifactRef
    mapping_name: str
    n_cells: int
    correction_method: str
    diagnostics: dict[str, float | int | str]
    indices: np.ndarray | None = None
    distances: np.ndarray | None = None
    uninformative: np.ndarray | None = None
    reference: "MappingReference | None" = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        loaded = [
            name
            for name in ("indices", "distances", "uninformative")
            if getattr(self, name) is not None
        ]
        return (
            f"MappingResult(ref={self.ref!r}, "
            f"mapping_name={self.mapping_name!r}, "
            f"n_cells={self.n_cells}, "
            f"correction_method={self.correction_method!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"arrays={loaded or 'not loaded'})"
        )
