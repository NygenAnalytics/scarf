"""Public contracts for scarf.plotting."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

LookupBy = Literal["name", "id", "index"]
FeatureReduction = Literal["mean", "sum"]
CellFieldKind = Literal["auto", "categorical", "continuous"]
NormSource = Literal["assay", "raw"]
NormTransform = Literal["none", "log1p"]
Standardize = Literal["none", "feature"]


@dataclass(frozen=True, slots=True)
class FeatureRef:
    """Reference to one assay feature.

    Parameters:
        value: Feature name, id, or physical index (see ``by``).
        assay: Assay name. Defaults to the store default assay when omitted.
        by: How to look up ``value``: ``name``, ``id``, or ``index``.
        label: Optional display label.
        reduction: Required when multiple features match; ``mean`` or ``sum``.
    """

    value: str | int
    assay: str | None = None
    by: LookupBy = "name"
    label: str | None = None
    reduction: FeatureReduction | None = None

    def __post_init__(self) -> None:
        if self.by not in ("name", "id", "index"):
            raise ValueError("by must be 'name', 'id', or 'index'")
        if self.reduction not in (None, "mean", "sum"):
            raise ValueError("reduction must be 'mean', 'sum', or None")


@dataclass(frozen=True, slots=True)
class CellField:
    """Reference to a cell-metadata column."""

    key: str
    kind: CellFieldKind = "auto"
    label: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("auto", "categorical", "continuous"):
            raise ValueError("kind must be 'auto', 'categorical', or 'continuous'")


@dataclass(frozen=True, slots=True)
class StudyDesign:
    """How cells relate to biological samples and conditions.

    Supports ``sample_by`` and optional ``condition_by``. Composition pairing
    uses ``condition_by`` with ``subject_by`` or ``pair_by``. Technical-replicate
    collapse is deferred.
    """

    sample_by: str
    condition_by: str | None = None
    subject_by: str | None = None
    pair_by: str | None = None
    technical_replicate_by: str | None = None
    technical_replicate_reduction: Literal["sum", "mean"] | None = None

    def __post_init__(self) -> None:
        unsupported = [
            name
            for name, value in (
                ("technical_replicate_by", self.technical_replicate_by),
                (
                    "technical_replicate_reduction",
                    self.technical_replicate_reduction,
                ),
            )
            if value is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "These StudyDesign fields are not supported yet: "
                + ", ".join(unsupported)
                + ". Collapse technical replicates into sample_by first, "
                "or omit them."
            )


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    source: NormSource = "assay"
    transform: NormTransform = "none"

    def __post_init__(self) -> None:
        if self.source not in ("assay", "raw"):
            raise ValueError("source must be 'assay' or 'raw'")
        if self.transform not in ("none", "log1p"):
            raise ValueError("transform must be 'none' or 'log1p'")


@dataclass(frozen=True, slots=True)
class ColorScale:
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    vcenter: float | None = None
    quantiles: tuple[float, float] | None = None
    missing_color: str = "#bdbdbd"
    scope: Literal["feature", "panel", "shared"] = "feature"

    def __post_init__(self) -> None:
        if self.quantiles is not None:
            low, high = self.quantiles
            if not (0.0 <= low < high <= 1.0):
                raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
        if self.vmin is not None and self.vmax is not None and self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        if (
            self.vcenter is not None
            and self.vmin is not None
            and self.vmax is not None
            and not self.vmin < self.vcenter < self.vmax
        ):
            raise ValueError("vcenter must be strictly between vmin and vmax")
        if self.scope not in ("feature", "panel", "shared"):
            raise ValueError("scope must be 'feature', 'panel', or 'shared'")


@dataclass(frozen=True, slots=True)
class CategoricalScale:
    order: tuple[Any, ...] | None = None
    palette: dict[Any, str] | None = None
    missing_color: str = "#bdbdbd"
    missing_label: str = "NA"


@dataclass(frozen=True, slots=True)
class SizeScale:
    """Maps numeric values to marker area. Detection fraction uses domain [0, 1]."""

    vmin: float = 0.0
    vmax: float = 1.0
    size_min: float = 10.0
    size_max: float = 200.0

    def __post_init__(self) -> None:
        if self.size_min < 0 or self.size_max < self.size_min:
            raise ValueError("size range must satisfy 0 <= size_min <= size_max")

    def areas(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        span = self.vmax - self.vmin
        if span <= 0:
            return np.full(v.shape, self.size_min, dtype=np.float64)
        t = np.clip((v - self.vmin) / span, 0.0, 1.0)
        return self.size_min + t * (self.size_max - self.size_min)


@dataclass(frozen=True, slots=True)
class PlotProvenance:
    scarf_version: str
    schema_version: str = "1"
    assay: str | None = None
    cell_key: str | None = None
    n_cells: int = 0
    n_samples: int | None = None
    renderer: str = "matplotlib"
    notes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureSummary:
    """Bounded tables behind summary plots."""

    aggregate: pd.DataFrame
    per_sample: pd.DataFrame | None = None
