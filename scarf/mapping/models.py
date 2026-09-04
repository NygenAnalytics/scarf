from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..storage.refs import ArtifactRef

if TYPE_CHECKING:
    from .reference import MappingReference


def _immutable_array(values: np.ndarray) -> np.ndarray:
    """Own one C-contiguous array through an immutable bytes buffer."""
    array = np.ascontiguousarray(np.asarray(values))
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


@dataclass(frozen=True)
class ScaledPCAProjectionModel:
    feature_means: np.ndarray
    feature_scales: np.ndarray
    loadings: np.ndarray

    def __post_init__(self) -> None:
        feature_means = np.asarray(self.feature_means)
        feature_scales = np.asarray(self.feature_scales)
        loadings = np.asarray(self.loadings)
        if loadings.ndim != 2:
            raise ValueError("Reference PCA loadings must be two-dimensional")
        n_features = loadings.shape[0]
        if feature_means.shape != (n_features,):
            raise ValueError("Reference feature means have incompatible dimensions")
        if feature_scales.shape != (n_features,):
            raise ValueError("Reference feature scales have incompatible dimensions")
        if np.any(feature_scales <= 0):
            raise ValueError("Reference feature scales must be positive")
        for values in (
            feature_means,
            feature_scales,
            loadings,
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "Reference projection model contains non-finite values"
                )
        object.__setattr__(self, "feature_means", _immutable_array(feature_means))
        object.__setattr__(self, "feature_scales", _immutable_array(feature_scales))
        object.__setattr__(self, "loadings", _immutable_array(loadings))

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
        centroids = np.asarray(self.centroids)
        raw_centroids = np.asarray(self.raw_centroids)
        corrected_centroids = np.asarray(self.corrected_centroids)
        cluster_mass = np.asarray(self.cluster_mass)
        sigma = np.asarray(self.sigma)
        if centroids.ndim != 2:
            raise ValueError("Reference centroids must be two-dimensional")
        n_clusters = centroids.shape[0]
        n_dims = centroids.shape[1]
        if raw_centroids.shape != (n_clusters, n_dims):
            raise ValueError("Reference raw centroids have incompatible dimensions")
        if corrected_centroids.shape != (n_clusters, n_dims):
            raise ValueError(
                "Reference corrected centroids have incompatible dimensions"
            )
        if cluster_mass.shape != (n_clusters,) or np.any(cluster_mass <= 0):
            raise ValueError("Reference cluster masses must be positive")
        if sigma.shape != (n_clusters,) or np.any(sigma <= 0):
            raise ValueError("Reference kernel widths must be positive")
        for values in (
            centroids,
            raw_centroids,
            corrected_centroids,
            cluster_mass,
            sigma,
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError("Symphony correction model contains non-finite values")
        object.__setattr__(self, "centroids", _immutable_array(centroids))
        object.__setattr__(self, "raw_centroids", _immutable_array(raw_centroids))
        object.__setattr__(
            self,
            "corrected_centroids",
            _immutable_array(corrected_centroids),
        )
        object.__setattr__(self, "cluster_mass", _immutable_array(cluster_mass))
        object.__setattr__(self, "sigma", _immutable_array(sigma))

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
        batch_offsets = np.asarray(self.batch_offsets)
        batch_counts = np.asarray(self.batch_counts)
        if batch_offsets.ndim != 3:
            raise ValueError(
                "Batch offsets must have batch, cluster, and dimension axes"
            )
        if batch_counts.shape != batch_offsets.shape[:2]:
            raise ValueError("Batch counts must match batch offsets")
        if not np.all(np.isfinite(batch_offsets)):
            raise ValueError("Batch offsets contain non-finite values")
        object.__setattr__(self, "batch_offsets", _immutable_array(batch_offsets))
        object.__setattr__(self, "batch_counts", _immutable_array(batch_counts))


@dataclass(frozen=True, slots=True)
class _MappingResultAxes:
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef


@dataclass(frozen=True)
class MappingResult:
    ref: ArtifactRef
    n_cells: int
    correction_method: str
    diagnostics: dict[str, float | int | str]
    reference: "MappingReference" = field(repr=False, compare=False)
    indices: np.ndarray | None = None
    distances: np.ndarray | None = None
    uninformative: np.ndarray | None = None

    @property
    def cell_selection(self) -> ArtifactRef:
        """Exact frozen query-row selection for this loaded projection."""
        axes = getattr(self, "_axes", None)
        if not isinstance(axes, _MappingResultAxes):
            raise RuntimeError(
                "Query axes are available on results loaded with "
                "DataStore.get_mapping_result"
            )
        return axes.cell_selection

    @property
    def feature_selection(self) -> ArtifactRef:
        """Exact frozen query-feature selection for this loaded projection."""
        axes = getattr(self, "_axes", None)
        if not isinstance(axes, _MappingResultAxes):
            raise RuntimeError(
                "Query axes are available on results loaded with "
                "DataStore.get_mapping_result"
            )
        return axes.feature_selection

    def __repr__(self) -> str:
        loaded = [
            name
            for name in ("indices", "distances", "uninformative")
            if getattr(self, name) is not None
        ]
        return (
            f"MappingResult(ref={self.ref!r}, "
            f"n_cells={self.n_cells}, "
            f"correction_method={self.correction_method!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"arrays={loaded or 'not loaded'})"
        )
