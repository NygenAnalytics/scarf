from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..storage.refs import ArtifactRef, ExternalArtifactRef
from ..utils.arrays import read_only_copy
from .confidence import _conformal_membership, _validated_conformal_calibration

if TYPE_CHECKING:
    from .reference import MappingReference


@dataclass(frozen=True)
class ScaledPCAProjectionModel:
    feature_means: np.ndarray
    feature_scales: np.ndarray
    center: np.ndarray
    loadings: np.ndarray

    def __post_init__(self) -> None:
        feature_means = np.asarray(self.feature_means)
        feature_scales = np.asarray(self.feature_scales)
        center = np.asarray(self.center)
        loadings = np.asarray(self.loadings)
        if loadings.ndim != 2:
            raise ValueError("Reference PCA loadings must be two-dimensional")
        n_features = loadings.shape[0]
        if feature_means.shape != (n_features,):
            raise ValueError("Reference feature means have incompatible dimensions")
        if feature_scales.shape != (n_features,):
            raise ValueError("Reference feature scales have incompatible dimensions")
        if center.shape != (n_features,):
            raise ValueError("Reference PCA center has incompatible dimensions")
        if np.any(feature_scales <= 0):
            raise ValueError("Reference feature scales must be positive")
        for values in (
            feature_means,
            feature_scales,
            center,
            loadings,
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "Reference projection model contains non-finite values"
                )
        object.__setattr__(self, "feature_means", read_only_copy(feature_means))
        object.__setattr__(self, "feature_scales", read_only_copy(feature_scales))
        object.__setattr__(self, "center", read_only_copy(center))
        object.__setattr__(self, "loadings", read_only_copy(loadings))

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
        object.__setattr__(self, "centroids", read_only_copy(centroids))
        object.__setattr__(self, "raw_centroids", read_only_copy(raw_centroids))
        object.__setattr__(
            self,
            "corrected_centroids",
            read_only_copy(corrected_centroids),
        )
        object.__setattr__(self, "cluster_mass", read_only_copy(cluster_mass))
        object.__setattr__(self, "sigma", read_only_copy(sigma))

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
        object.__setattr__(self, "batch_offsets", read_only_copy(batch_offsets))
        object.__setattr__(self, "batch_counts", read_only_copy(batch_counts))


@dataclass(frozen=True, slots=True)
class _MappingResultAxes:
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef


@dataclass(frozen=True)
class MappingResult:
    """One loaded query projection and its optional neighbor arrays.

    ``uninformative`` marks query cells whose raw counts are zero in every
    reference feature that the query measured. Their neighbor rows are stored
    but carry no query evidence, so label transfer and mapping scores skip them.
    """

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


_VOTE_BLOCK_ROWS = 65_536


def _frozen_array(values: Any) -> np.ndarray:
    """Return ``values`` as a read-only array, copying only changeable data.

    An array whose underlying data no owner can change, such as one a loader
    froze, is kept as a read-only view, so large loaded arrays are not copied.
    """
    array = np.asarray(values)
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if owner.flags.writeable:
        return read_only_copy(array)
    frozen = array.view()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class LabelTransferResult:
    """One loaded label transfer: transferred query labels and their evidence.

    ``evidence`` has one row per projected query cell, in the order of
    ``cell_selection``; ``cell_idx`` gives the matching query cell rows.
    ``label`` is the transferred label and is missing where the cell
    abstained. ``abstentionReason`` says why a cell abstained:
    ``uninformative_cell`` (no counts in any measured reference feature),
    ``no_labeled_neighbors``, ``tied_vote``, ``below_threshold``
    (``voteFraction`` below ``threshold_fraction``), or
    ``beyond_max_distance`` (``nearestDistance`` above ``max_distance``).
    ``candidateLabel`` is the label that the vote favored before the threshold
    and distance rules, so other thresholds can be compared without a new
    transfer.

    ``reference_labels`` is the frozen copy of the reference labels in the
    query datastore, and ``reference_label_source`` names where they were
    read: a reference cell-metadata column, or a reference label artifact.
    ``vote_class_codes`` and ``vote_class_fractions`` hold, for each cell, the
    reference classes that its neighbors voted for, as positions in
    ``categories``, padded with -1, and each class's share of the neighbor
    weight. These matrices hold one column per saved neighbor, so they are
    loaded only when requested, with ``get_label_transfer(transfer,
    load_votes=True)``; :meth:`label_vote_shares` and :meth:`prediction_sets`
    need them.
    """

    ref: ArtifactRef
    projection: ArtifactRef
    reference_labels: ArtifactRef
    reference_label_source: "str | ExternalArtifactRef"
    cell_selection: ArtifactRef
    cell_idx: np.ndarray = field(repr=False, compare=False)
    threshold_fraction: float
    max_distance: float | None
    categories: np.ndarray = field(repr=False, compare=False)
    evidence: pd.DataFrame = field(repr=False, compare=False)
    vote_class_codes: np.ndarray | None = field(default=None, repr=False, compare=False)
    vote_class_fractions: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("cell_idx", "categories"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name)))
        n_cells = len(self.evidence)
        if self.cell_idx.shape != (n_cells,):
            raise ValueError("cell_idx must have one row per projected query cell")
        if (self.vote_class_codes is None) != (self.vote_class_fractions is None):
            raise ValueError("Vote class codes and fractions are loaded together")
        if self.vote_class_codes is None or self.vote_class_fractions is None:
            return
        codes = _frozen_array(self.vote_class_codes)
        fractions = _frozen_array(self.vote_class_fractions)
        if (
            codes.ndim != 2
            or codes.shape[0] != n_cells
            or fractions.shape != codes.shape
        ):
            raise ValueError("Vote arrays must have one row per projected query cell")
        object.__setattr__(self, "vote_class_codes", codes)
        object.__setattr__(self, "vote_class_fractions", fractions)

    def _votes(self) -> tuple[np.ndarray, np.ndarray]:
        if self.vote_class_codes is None or self.vote_class_fractions is None:
            raise ValueError(
                "This label transfer was loaded without its votes. Load it with "
                "get_label_transfer(transfer, load_votes=True)"
            )
        return self.vote_class_codes, self.vote_class_fractions

    @property
    def labels(self) -> pd.Series:
        """Transferred labels, missing where a cell abstained."""
        return self.evidence["label"].copy()

    @property
    def n_cells(self) -> int:
        """Number of projected query cells."""
        return len(self.evidence)

    def label_vote_shares(self, labels: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Return each cell's share of neighbor vote weight for a given label.

        ``labels`` holds one label per projected query cell, such as the known
        labels of held-out calibration cells. A label that is not a reference
        class has a share of zero. Missing labels and uninformative cells give
        NaN. One minus these shares is the nonconformity that
        :meth:`prediction_sets` calibrates with.
        """
        values = np.asarray(labels, dtype=object)
        if values.shape != (self.n_cells,):
            raise ValueError("labels must have one value per projected query cell")
        categories = self.categories.tolist()
        missing = np.asarray(pd.isna(values), dtype=bool)
        codes = np.asarray(
            pd.Index(categories, dtype=object).get_indexer(values),
            dtype=np.int64,
        )
        codes[missing] = -1
        unmatched = (codes < 0) & ~missing
        if unmatched.any():
            category_texts = pd.Index([str(category) for category in categories])
            if (category_texts.get_indexer(values[unmatched].astype(str)) >= 0).any():
                raise ValueError(
                    "Some labels match a reference class only as text. Convert "
                    "labels to the value type of the reference labels"
                )
        vote_codes, vote_fractions = self._votes()
        shares = np.zeros(self.n_cells, dtype=np.float64)
        # Bounded row blocks keep the temporaries small next to the votes.
        for start in range(0, self.n_cells, _VOTE_BLOCK_ROWS):
            stop = min(start + _VOTE_BLOCK_ROWS, self.n_cells)
            label_codes = codes[start:stop, np.newaxis]
            voted = (vote_codes[start:stop] == label_codes) & (label_codes >= 0)
            shares[start:stop] = np.where(voted, vote_fractions[start:stop], 0.0).sum(
                axis=1
            )
        uninformative = (
            self.evidence["abstentionReason"].to_numpy(dtype=object)
            == "uninformative_cell"
        )
        shares[missing | uninformative] = np.nan
        return shares

    def prediction_sets(
        self,
        calibration_nonconformity: np.ndarray,
        alpha: float = 0.1,
    ) -> pd.Series:
        """Return split-conformal prediction sets from the stored votes.

        ``calibration_nonconformity`` holds one minus the vote share of the
        true label for held-out calibration cells, which must be exchangeable
        with these query cells; :meth:`label_vote_shares` computes those
        shares. A class joins a cell's set when its nonconformity is not
        exceeded by more than a fraction ``alpha`` of the calibration cells.
        Cells without labeled votes get an empty set.
        """
        calibration, resolved_alpha = _validated_conformal_calibration(
            calibration_nonconformity,
            alpha,
        )
        vote_codes, vote_fractions = self._votes()
        sets: list[tuple[Any, ...]] = [()] * self.n_cells
        vote_fraction = self.evidence["voteFraction"].to_numpy(dtype=np.float64)
        for row in np.flatnonzero(vote_fraction > 0):
            # One row of class scores at a time bounds memory by the class count.
            scores = np.zeros(len(self.categories), dtype=np.float64)
            voted = vote_codes[row] >= 0
            scores[vote_codes[row, voted]] = vote_fractions[row, voted]
            members = _conformal_membership(scores, calibration, resolved_alpha)
            sets[row] = tuple(self.categories[members].tolist())
        return pd.Series(sets, name="predictionSet")

    def __repr__(self) -> str:
        abstained = int(self.evidence["abstained"].sum())
        return (
            f"LabelTransferResult(ref={self.ref!r}, n_cells={self.n_cells}, "
            f"abstained={abstained}, threshold_fraction={self.threshold_fraction!r}, "
            f"max_distance={self.max_distance!r}, "
            f"reference_label_source={self.reference_label_source!r})"
        )
