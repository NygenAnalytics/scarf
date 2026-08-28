"""Value-selection contracts shared by analysis and presentation layers."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ..storage.selections import read_stored_selection_indices
from .artifacts import artifact_values
from .rows import read_metadata_missing_rows, read_metadata_rows

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
    "GROUPING_VALUE_NAMES",
    "LookupBy",
    "NormalizationSpec",
    "NormSource",
    "NormTransform",
    "ResolvedGrouping",
    "Standardize",
    "StudyDesign",
    "grouping_value_name",
    "resolve_grouping",
    "valid_category_mask",
]

GROUPING_VALUE_NAMES: dict[str, str] = {
    "cell_cycle": "phase",
    "cluster_cut": "labels",
    "cluster_labels": "values",
    "hto_identity": "values",
    "smart_label": "values",
}


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
class ResolvedGrouping:
    """Categorical labels aligned to exact physical cell rows."""

    source: ArtifactRef | CellField
    labels: np.ndarray
    cell_idx: np.ndarray
    cell_selection: ArtifactRef | None
    missing_mask: np.ndarray | None


def grouping_value_name(kind: str) -> str:
    """Return the canonical categorical-label array for an artifact kind."""
    try:
        return GROUPING_VALUE_NAMES[kind]
    except KeyError as exc:
        raise ValueError(
            "Grouping artifacts must contain categorical cell labels"
        ) from exc


def _selection_indices(root: Any, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)


def resolve_grouping(
    root: Any,
    cells: Any,
    grouping: ArtifactRef | CellField,
    *,
    cell_selection: ArtifactRef | None = None,
) -> ResolvedGrouping:
    """Resolve categorical labels from an artifact or an explicit metadata field."""
    if isinstance(grouping, CellField):
        if grouping.kind == "continuous":
            raise ValueError("Grouping CellField must be categorical")
        cell_idx = (
            np.arange(cells.N, dtype=np.int64)
            if cell_selection is None
            else _selection_indices(root, cell_selection)
        )
        labels = np.asarray(read_metadata_rows(cells, grouping.key, cell_idx))
        missing = read_metadata_missing_rows(cells, grouping.key, cell_idx)
        missing_mask = None if missing is None else np.asarray(missing, dtype=bool)
        if labels.shape != (len(cell_idx),):
            raise ValueError("Grouping metadata does not align with selected cells")
        if missing_mask is not None and missing_mask.shape != labels.shape:
            raise ValueError("Grouping missing mask does not align with selected cells")
        return ResolvedGrouping(
            source=grouping,
            labels=labels,
            cell_idx=cell_idx,
            cell_selection=cell_selection,
            missing_mask=missing_mask,
        )

    if not isinstance(grouping, ArtifactRef):
        raise TypeError("grouping must be an ArtifactRef or CellField")
    value_name = grouping_value_name(grouping.kind)
    status = inspect_artifact(root, grouping)
    if not status.complete:
        raise ValueError("Grouping artifact is unavailable or incomplete")
    raw_selection = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw_selection, dict):
        raise ValueError("Grouping artifact has no cell-selection input")
    stored_selection = ArtifactRef.from_dict(raw_selection)
    stored_idx = _selection_indices(root, stored_selection)
    labels = np.asarray(artifact_values(artifact_group(root, grouping), value_name))
    if labels.shape != (len(stored_idx),):
        raise ValueError("Grouping labels do not align with selected cells")

    resolved_selection = stored_selection
    cell_idx = stored_idx
    if cell_selection is not None and cell_selection != stored_selection:
        requested_idx = _selection_indices(root, cell_selection)
        keep = np.isin(stored_idx, requested_idx, assume_unique=True)
        if int(keep.sum()) != len(requested_idx):
            raise ValueError("cell_selection must be a subset of the grouping artifact")
        selected_idx = stored_idx[keep]
        if not np.array_equal(selected_idx, requested_idx):
            raise ValueError(
                "cell_selection must preserve the grouping artifact's cell order"
            )
        labels = labels[keep]
        cell_idx = selected_idx
        resolved_selection = cell_selection

    return ResolvedGrouping(
        source=grouping,
        labels=labels,
        cell_idx=cell_idx,
        cell_selection=resolved_selection,
        missing_mask=None,
    )


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
