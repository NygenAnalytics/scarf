"""Value-selection contracts shared by analysis and presentation layers."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

LookupBy = Literal["name", "id", "index"]
FeatureReduction = Literal["mean", "sum"]
CellFieldKind = Literal["auto", "categorical", "continuous"]
NormSource = Literal["assay", "raw"]
NormTransform = Literal["none", "log1p"]
Standardize = Literal["none", "feature"]

__all__ = [
    "CellField",
    "CellFieldKind",
    "FeatureReduction",
    "FeatureRef",
    "LookupBy",
    "NormalizationSpec",
    "NormSource",
    "NormTransform",
    "Standardize",
    "StudyDesign",
    "valid_category_mask",
]


def _is_missing_label(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def valid_category_mask(
    values: Any,
    *,
    missing_mask: Any | None = None,
) -> np.ndarray:
    """Mark non-missing, non-blank values that can identify categories.

    ``missing_mask`` carries the explicit missingness stored alongside typed
    metadata columns, whose placeholder values may otherwise look valid.
    """
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError("category values must be one-dimensional")
    valid = np.fromiter(
        (
            not _is_missing_label(value)
            and (not isinstance(value, str) or value.strip() != "")
            for value in array
        ),
        dtype=bool,
        count=array.size,
    )
    if missing_mask is not None:
        missing = np.asarray(missing_mask, dtype=bool)
        if missing.shape != array.shape:
            raise ValueError("missing mask must align with category values")
        valid &= ~missing
    return valid


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
