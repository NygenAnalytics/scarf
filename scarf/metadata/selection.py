"""Value-selection contracts shared by analysis and presentation layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ..storage.selections import read_stored_selection_indices
from ..storage.types import as_zarr_array
from .rows import (
    read_array_rows_chunkwise,
    read_metadata_missing_rows,
    read_metadata_rows,
)

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
    "NamedCellArtifact",
    "ResolvedCellArtifact",
    "ResolvedGrouping",
    "Standardize",
    "StudyDesign",
    "grouping_value_name",
    "resolve_cell_aligned_artifact",
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


def _is_blank_label(value: object) -> bool:
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, bytes | np.bytes_):
        return value.strip() == b""
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
            not _is_missing_label(value) and not _is_blank_label(value)
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
class NamedCellArtifact:
    """One semantic name bound to an exact cell-aligned artifact."""

    name: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Named cell artifacts require a non-empty name")
        if self.name != self.name.strip():
            raise ValueError(
                "Named cell artifact names cannot have surrounding whitespace"
            )
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("Named cell artifacts require an ArtifactRef")


@dataclass(frozen=True, slots=True)
class ResolvedCellArtifact:
    """Artifact values aligned to one validated cell selection."""

    source: ArtifactRef
    values: np.ndarray
    cell_idx: np.ndarray
    source_cell_selection: ArtifactRef
    cell_selection: ArtifactRef


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


def resolve_cell_aligned_artifact(
    root: Any,
    artifact: ArtifactRef,
    *,
    cell_selection: ArtifactRef | None = None,
    value_name: str = "values",
    expected_kind: str | None = None,
) -> ResolvedCellArtifact:
    """Read one artifact vector in the exact requested cell order."""
    if not isinstance(artifact, ArtifactRef):
        raise TypeError("artifact must be an ArtifactRef")
    if expected_kind is not None and artifact.kind != expected_kind:
        raise ValueError(
            f"Expected a {expected_kind!r} artifact, received {artifact.kind!r}"
        )
    if not isinstance(value_name, str) or not value_name:
        raise ValueError("value_name must be a non-empty string")

    status = inspect_artifact(root, artifact)
    if not status.exists or not status.complete:
        raise ValueError("Cell-aligned artifact is unavailable or incomplete")
    raw_selection = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ValueError("Cell-aligned artifact has no cell-selection input")
    try:
        source_selection = ArtifactRef.from_dict(raw_selection)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cell-aligned artifact cell selection is malformed") from exc
    source_idx = _selection_indices(root, source_selection)

    target_selection = source_selection if cell_selection is None else cell_selection
    if not isinstance(target_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    target_idx = (
        source_idx
        if target_selection == source_selection
        else _selection_indices(root, target_selection)
    )
    if target_selection == source_selection:
        compact_idx = np.arange(len(source_idx), dtype=np.int64)
    else:
        compact_idx = np.searchsorted(source_idx, target_idx).astype(
            np.int64,
            copy=False,
        )
        in_bounds = compact_idx < len(source_idx)
        if not bool(in_bounds.all()):
            raise ValueError(
                "cell_selection must be a subset of the artifact cell selection"
            )
        if not np.array_equal(source_idx[compact_idx], target_idx):
            raise ValueError(
                "cell_selection must be a subset of the artifact cell selection"
            )

    group = artifact_group(root, artifact)
    if value_name not in group:
        raise ValueError(f"Cell-aligned artifact has no {value_name!r} value array")
    values_array = as_zarr_array(group[value_name], name=value_name)
    if values_array.ndim != 1 or int(values_array.shape[0]) != len(source_idx):
        raise ValueError(
            "Cell-aligned artifact must contain one value per source-selected cell"
        )
    values = read_array_rows_chunkwise(values_array, compact_idx)
    if values.shape != (len(target_idx),):
        raise ValueError("Cell-aligned artifact values do not match the selection")
    return ResolvedCellArtifact(
        source=artifact,
        values=values,
        cell_idx=target_idx,
        source_cell_selection=source_selection,
        cell_selection=target_selection,
    )


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
    resolved = resolve_cell_aligned_artifact(
        root,
        grouping,
        cell_selection=cell_selection,
        value_name=value_name,
        expected_kind=grouping.kind,
    )

    return ResolvedGrouping(
        source=grouping,
        labels=resolved.values,
        cell_idx=resolved.cell_idx,
        cell_selection=resolved.cell_selection,
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
