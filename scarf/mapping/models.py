from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymphonyReferenceModel:
    feature_means: np.ndarray
    feature_scales: np.ndarray
    loadings: np.ndarray
    centroids: np.ndarray
    raw_centroids: np.ndarray
    corrected_centroids: np.ndarray
    cluster_mass: np.ndarray
    sigma: np.ndarray
    correction_ridge: float

    def __post_init__(self) -> None:
        n_features, n_dims = self.loadings.shape
        n_clusters = self.centroids.shape[0]
        if self.feature_means.shape != (n_features,):
            raise ValueError("Reference feature means have incompatible dimensions")
        if self.feature_scales.shape != (n_features,):
            raise ValueError("Reference feature scales have incompatible dimensions")
        if np.any(self.feature_scales <= 0):
            raise ValueError("Reference feature scales must be positive")
        if self.centroids.shape != (n_clusters, n_dims):
            raise ValueError("Reference centroids have incompatible dimensions")
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
        if self.correction_ridge < 0:
            raise ValueError("Reference correction ridge must be non-negative")
        for values in (
            self.feature_means,
            self.feature_scales,
            self.loadings,
            self.centroids,
            self.raw_centroids,
            self.corrected_centroids,
            self.cluster_mass,
            self.sigma,
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError("Reference model contains non-finite values")

    @property
    def n_features(self) -> int:
        return int(self.loadings.shape[0])

    @property
    def n_dims(self) -> int:
        return int(self.loadings.shape[1])

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
    projection_path: str
    n_cells: int
    correction_method: str
    diagnostics: dict[str, float | str]
    indices: np.ndarray | None = None
    distances: np.ndarray | None = None
    uncorrected_latent: np.ndarray | None = None
    corrected_latent: np.ndarray | None = None
    uninformative: np.ndarray | None = None

    def __repr__(self) -> str:
        # The default dataclass repr prints every loaded array in full.
        loaded = [
            name
            for name in (
                "indices",
                "distances",
                "uncorrected_latent",
                "corrected_latent",
                "uninformative",
            )
            if getattr(self, name) is not None
        ]
        return (
            f"MappingResult(n_cells={self.n_cells}, "
            f"correction_method={self.correction_method!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"arrays={loaded or 'not loaded'}, "
            f"projection_path='{self.projection_path[:40]}...')"
        )
