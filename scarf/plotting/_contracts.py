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
LegendLoc = Literal["auto", "right", "on_data", "none"]
FrameStyle = Literal["axes", "minimal", "none"]
DistKind = Literal["violin", "box", "hist", "ecdf"]


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
    """Point ``color_by`` or similar arguments at a cell-metadata column.

    Use this when the column name alone is ambiguous. ``kind="categorical"``
    forces discrete colors (useful for integer cluster ids).
    ``kind="continuous"`` forces a colorbar. ``kind="auto"`` chooses from the
    dtype and number of unique values.
    """

    key: str
    kind: CellFieldKind = "auto"
    label: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("auto", "categorical", "continuous"):
            raise ValueError("kind must be 'auto', 'categorical', or 'continuous'")


@dataclass(frozen=True, slots=True)
class StudyDesign:
    """Describe samples and conditions for composition and summary plots.

    ``sample_by`` is the column that identifies biological samples.
    ``condition_by`` is the experimental condition (for example treatment).
    For paired composition plots, also set ``subject_by`` or ``pair_by`` so the
    same donor or pair can be connected across conditions.
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
    """How feature values are read for gene-colored plots.

    ``source="assay"`` uses the assay's current normalization settings.
    ``source="raw"`` reads raw counts. ``transform="log1p"`` applies log1p
    after that fetch, which is the usual choice for gene UMAPs and dotplots
    when you want a compressed expression scale.
    """

    source: NormSource = "assay"
    transform: NormTransform = "none"

    def __post_init__(self) -> None:
        if self.source not in ("assay", "raw"):
            raise ValueError("source must be 'assay' or 'raw'")
        if self.transform not in ("none", "log1p"):
            raise ValueError("transform must be 'none' or 'log1p'")


@dataclass(frozen=True, slots=True)
class ColorScale:
    """Continuous color mapping for embeddings and summary plots.

    Set ``vmin`` / ``vmax`` for fixed limits, or ``quantiles=(low, high)`` to
    ignore extreme tails (for example ``(0.0, 0.99)`` clips the top 1%).
    ``vcenter`` is for diverging maps such as scaled fold changes.
    ``scope`` controls whether limits are computed per feature (default), per
    panel, or shared across every continuous panel in the figure.
    """

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
    """Category order and colors for discrete embeddings and compositions.

    ``order`` sets legend and axis order. ``palette`` maps each category to a
    color string. Categories missing from ``palette`` raise an error, so pass
    a complete map when you customize colors.
    """

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
