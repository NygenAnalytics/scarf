"""Value-selection contracts shared by analysis and presentation layers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..storage.arrays import linked_missing_mask
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_group,
    inspect_artifact,
    parse_artifact_ref,
)
from ..storage.geometry import array_geometry
from ..storage.partition import scan_band
from ..storage.selections import read_stored_selection_indices, validate_cell_selection
from ..storage.types import as_zarr_array
from ..storage.validation_scope import validation_scope
from ..utils.arguments import integer_argument
from .rows import (
    apply_missing_mask,
    array_row_selection_parts,
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
    "CELL_VALUE_NAMES",
    "CellField",
    "CellFieldKind",
    "CellValueSpec",
    "CellValues",
    "FeatureReduction",
    "FeatureRef",
    "LookupBy",
    "NormalizationSpec",
    "NormSource",
    "NormTransform",
    "NamedCellArtifact",
    "ResolvedCellArtifact",
    "ResolvedGrouping",
    "Standardize",
    "StudyDesign",
    "cell_value_array",
    "cell_value_spec",
    "grouping_value_name",
    "resolve_cell_aligned_artifact",
    "resolve_grouping",
    "valid_category_mask",
]


@dataclass(frozen=True, slots=True)
class CellValueSpec:
    """The per-cell arrays of one cell-aligned artifact kind."""

    name: str
    categorical: bool
    alternatives: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alternatives = dict(self.alternatives)
        if not all(
            isinstance(flag, bool)
            for flag in (self.categorical, *alternatives.values())
        ):
            raise TypeError("Cell value categorical flags must be booleans")
        if self.name in alternatives:
            raise ValueError(
                "The canonical cell value array cannot also be an alternative"
            )
        object.__setattr__(self, "alternatives", MappingProxyType(alternatives))


#: The per-cell arrays of every cell-aligned artifact kind, a read-only
#: mapping from each kind to its ``CellValueSpec``: the canonical array, whether
#: it holds labels, and the kind's other per-cell arrays. A kind is
#: cell-aligned exactly when it is listed, and readers refuse every other
#: kind, whatever array they are asked for. Reductions and Harmony corrections
#: hold a row per cell too, but their cells follow the cell selection of their
#: lineage rather than a ``cell_selection`` input, so they are not listed.
#:
#: The canonical array names are part of the identity of the results that
#: read them: ``select_cells``, plot groupings, and other consumers record
#: the artifact that they read but not which of its arrays. Changing a kind's
#: canonical array therefore changes what those consumers compute from the
#: same recorded inputs, and needs an operation revision for each of them.
CELL_VALUE_NAMES: Mapping[str, CellValueSpec] = MappingProxyType(
    {
        "cell_cycle": CellValueSpec(
            "phase", True, {"s_score": False, "g2m_score": False}
        ),
        "cluster_cut": CellValueSpec("labels", True),
        "cluster_labels": CellValueSpec("values", True),
        "doublet_score": CellValueSpec("values", False),
        "embedding": CellValueSpec("values", False),
        "enrichment_scores": CellValueSpec("scores", False),
        "fate_map": CellValueSpec("probabilities", False, {"valid": True}),
        "hto_identity": CellValueSpec("values", True),
        "imported_coordinates": CellValueSpec("data", False),
        "label_transfer": CellValueSpec(
            "labels",
            True,
            {
                "abstention_reason": True,
                "candidate_codes": True,
                "vote_class_codes": True,
                "vote_class_fractions": False,
                "vote_fraction": False,
                "top_two_margin": False,
                "vote_entropy": False,
                "nearest_distance": False,
                "reference_distance_percentile": False,
            },
        ),
        "membership_strength": CellValueSpec("values", False),
        # A metadata snapshot is cell-aligned when it records a cell
        # selection, as the custom source/sink vector of pseudotime scoring
        # does. A snapshot of whole metadata columns, such as a pipeline
        # run's, records none: readers refuse it with a ValueError that names
        # the run's frozen fields, and label it corrupt only when its
        # recorded cell selection is malformed.
        "metadata_snapshot": CellValueSpec("values", False),
        "pseudotime": CellValueSpec("pseudotime", False, {"valid": True}),
        "quality_metric": CellValueSpec("values", False),
        "sampling": CellValueSpec(
            "sampled", True, {"seeds": True, "density": False, "mean_snn": False}
        ),
        "smart_label": CellValueSpec("values", True),
    }
)


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
        if self.by == "index":
            # Resolution checks the index against the assay's feature count.
            value: str | int = integer_argument(self.value, "FeatureRef value")
        elif isinstance(self.value, str):
            value = str(self.value)
        else:
            raise TypeError(f"FeatureRef value must be a string when by is {self.by!r}")
        object.__setattr__(self, "value", value)


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
    """Artifact values aligned to one validated cell selection.

    ``missing_mask`` flags rows whose stored value is a placeholder for a
    missing entry. It is None when the artifact records no missing values.
    """

    source: ArtifactRef
    values: np.ndarray
    cell_idx: np.ndarray
    source_cell_selection: ArtifactRef
    cell_selection: ArtifactRef
    missing_mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ResolvedGrouping:
    """Categorical labels aligned to exact physical cell rows."""

    source: ArtifactRef | CellField
    labels: np.ndarray
    cell_idx: np.ndarray
    cell_selection: ArtifactRef | None
    missing_mask: np.ndarray | None


@dataclass(frozen=True, slots=True, eq=False)
class CellValues:
    """The per-cell values of one cell-aligned artifact, aligned to cells.

    Attributes:
        source: The artifact that holds the values.
        value: The name of the array that was read.
        values: One value, or one row of values, per cell.
        cell_ids: The ``ids`` of the cells, aligned to ``values``.
        cell_idx: The row of each cell in the cell table, in ascending order.
        cell_selection: The cell selection that the rows follow.
        missing: True for each cell whose value is missing, or None.
        categorical: Whether the values are labels that group cells.
    """

    source: ArtifactRef
    value: str
    values: np.ndarray
    cell_ids: np.ndarray
    cell_idx: np.ndarray
    cell_selection: ArtifactRef
    missing: np.ndarray | None
    categorical: bool

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Return the values indexed by cell id, showing missing rows as missing."""
        index = pd.Index(self.cell_ids, name="ids")
        values = np.asarray(self.values)
        if values.ndim == 1:
            return pd.Series(
                apply_missing_mask(values, self.missing, labels=self.categorical),
                index=index,
                name=self.value,
            )
        if values.ndim != 2:
            raise ValueError("to_pandas needs one or two dimensions of values")
        missing = (
            None
            if self.missing is None
            else np.broadcast_to(np.asarray(self.missing)[:, None], values.shape)
        )
        return pd.DataFrame(
            apply_missing_mask(values, missing, labels=self.categorical),
            index=index,
            columns=pd.RangeIndex(values.shape[1], name=self.value),
        )


def cell_value_spec(kind: str) -> CellValueSpec:
    """Return the per-cell arrays of a cell-aligned artifact kind."""
    try:
        return CELL_VALUE_NAMES[kind]
    except KeyError:
        raise ValueError(
            f"{kind!r} is not a cell-aligned artifact kind; the cell-aligned "
            f"kinds are {', '.join(sorted(CELL_VALUE_NAMES))}"
        ) from None


def cell_value_array(kind: str, value: str | None = None) -> tuple[str, bool]:
    """Return the per-cell array of ``kind`` to read and whether it holds labels."""
    if value is not None and not isinstance(value, str):
        raise TypeError("value must be a string or None")
    spec = cell_value_spec(kind)
    if value is None or value == spec.name:
        return spec.name, spec.categorical
    if value in spec.alternatives:
        return value, spec.alternatives[value]
    choices = ", ".join((spec.name, *spec.alternatives))
    raise ValueError(
        f"{kind!r} artifacts have no per-cell array {value!r}; their per-cell "
        f"arrays are {choices}"
    )


def grouping_value_name(kind: str) -> str:
    """Return the canonical categorical-label array for an artifact kind."""
    spec = CELL_VALUE_NAMES.get(kind)
    if spec is None or not spec.categorical:
        raise ValueError("Grouping artifacts must contain categorical cell labels")
    return spec.name


def require_complete_cluster_labels(
    group: Any,
    value_name: str,
    *,
    name: str,
    values: Any | None = None,
) -> None:
    """Reject stored cluster labels whose linked mask flags a missing label.

    Imported clusterings store a placeholder for each missing label, so
    consumers that treat every row as a member of a cluster must refuse them.
    """
    try:
        mask = linked_missing_mask(group, value_name, values=values)
    except ValueError as error:
        raise ValueError(f"{name} has a malformed missing-label mask") from error
    if mask is None:
        return
    n_rows = int(mask.shape[0])
    band = scan_band(array_geometry(mask), fallback=max(1, n_rows))
    if any(
        np.asarray(mask[start : start + band]).any() for start in range(0, n_rows, band)
    ):
        raise ValueError(f"{name} contains missing cluster labels")


def _selection_indices(root: Any, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)


def _selection_count(root: Any, selection: ArtifactRef) -> int:
    """Validate a cell selection and count its cells without reading its rows."""
    return int(validate_cell_selection(root, selection).selected_count)


# Bytes of one int64 cell row or position.
_INDEX_BYTES = np.dtype(np.int64).itemsize


def _chunkwise_read_bytes(array: Any, n_rows: int) -> int:
    """Bytes that reading ``n_rows`` rows of ``array`` chunk by chunk holds."""
    fixed, per_row = array_row_selection_parts(array)
    return int(fixed) + int(n_rows) * int(per_row)


def _unaligned_snapshot_error(
    artifact: ArtifactRef,
    status: ArtifactStatus,
) -> ValueError:
    """Refuse a metadata snapshot that records no cell selection.

    Such a snapshot is valid but holds whole metadata columns, one row per
    row of its table, rather than a row per cell of a recorded selection.
    """
    parameters = status.parameters or {}
    axis = parameters.get("axis")
    raw_columns = parameters.get("ordered_columns")
    columns = (
        ", ".join(repr(column) for column in raw_columns)
        if isinstance(raw_columns, list)
        else None
    )
    if status.operation == "snapshot_run_metadata" and axis == "cell":
        what = f"the cell table columns {columns}"
        remedy = (
            "Read them as the frozen fields of the pipeline run that recorded "
            "the snapshot, with run.cells.fetch(column) or "
            "run.cells.to_pandas_dataframe(columns); a plot with run=run takes "
            "a frozen field by name, such as color_by=column"
        )
    elif status.operation == "snapshot_run_metadata" and axis == "feature":
        what = f"the {artifact.assay} feature table columns {columns}"
        remedy = (
            "Read them as the frozen feature fields of the pipeline run that "
            "recorded the snapshot, with run.features.fetch(column) or "
            "run.features.to_pandas_dataframe(columns)"
        )
    else:
        what = "metadata values"
        remedy = "Open it with load_artifact"
    return ValueError(
        f"The metadata_snapshot artifact {artifact.artifact_id} holds {what} "
        "and records no cell selection, so its rows are not aligned to a "
        "recorded cell selection and it holds no cell-aligned values. "
        f"{remedy}."
    )


def _recorded_cell_selection(
    artifact: ArtifactRef,
    status: ArtifactStatus,
) -> ArtifactRef:
    """Return the cell selection that a cell-aligned artifact records.

    A metadata snapshot without a ``cell_selection`` input is not
    cell-aligned and raises ``ValueError``. For every other kind, and for a
    malformed recorded selection, the one strict reader of recorded inputs
    raises ``ArtifactResolutionError`` with code ``corrupt_payload``.
    """
    inputs = status.inputs or {}
    if artifact.kind == "metadata_snapshot" and "cell_selection" not in inputs:
        raise _unaligned_snapshot_error(artifact, status)
    # The one strict reader of recorded inputs, as ArtifactStatus.input_ref.
    return parse_artifact_ref(
        inputs.get("cell_selection"),
        "cell_selection",
        owner=artifact,
    )


def _row_missing_mask(mask: np.ndarray, value_name: str) -> np.ndarray:
    """Reduce a linked mask to one entry per row, which must flag whole rows."""
    if mask.ndim == 1:
        return mask
    flat = mask.reshape(mask.shape[0], -1)
    flagged = flat.any(axis=1)
    if not bool(flat[flagged].all()):
        raise ValueError(
            f"Cell-aligned artifact array {value_name!r} marks only some values "
            "of a cell as missing"
        )
    return flagged


def resolve_cell_aligned_artifact(
    root: Any,
    artifact: ArtifactRef,
    *,
    cell_selection: ArtifactRef | None = None,
    value_name: str | None = None,
    expected_kind: str | None = None,
    ndim: int | None = 1,
    max_bytes: int | None = None,
    caller_reads: Sequence[Any] = (),
) -> ResolvedCellArtifact:
    """Read one artifact array in the exact requested cell order.

    Raises:
        MemoryError: If ``max_bytes`` is set and the read needs more bytes.
    """
    if not isinstance(artifact, ArtifactRef):
        raise TypeError("artifact must be an ArtifactRef")
    if expected_kind is not None and artifact.kind != expected_kind:
        raise ValueError(
            f"Expected a {expected_kind!r} artifact, received {artifact.kind!r}"
        )
    if value_name is not None and (not isinstance(value_name, str) or not value_name):
        raise ValueError("value_name must be a non-empty string")
    if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    if ndim is not None and (type(ndim) is not int or ndim < 1):
        raise ValueError("ndim must be a positive integer or None")
    value_name, _ = cell_value_array(artifact.kind, value_name)

    status = inspect_artifact(root, artifact)
    if not status.exists or not status.complete:
        raise ValueError("Cell-aligned artifact is unavailable or incomplete")
    source_selection = _recorded_cell_selection(artifact, status)
    target_selection = source_selection if cell_selection is None else cell_selection
    same_selection = target_selection == source_selection

    group = artifact_group(root, artifact)
    if value_name not in group:
        raise ValueError(f"Cell-aligned artifact has no {value_name!r} value array")
    values_array = as_zarr_array(group[value_name], name=value_name)
    shape = tuple(int(extent) for extent in values_array.shape)
    # The selections are validated once: counting their cells here and
    # reading their rows below share one validation.
    with validation_scope():
        n_source = _selection_count(root, source_selection)
        if (
            not shape
            or (ndim is not None and len(shape) != ndim)
            or shape[0] != n_source
        ):
            unit = "value" if ndim == 1 else "row"
            raise ValueError(
                f"Cell-aligned artifact must contain one {unit} per "
                "source-selected cell"
            )
        missing_array = linked_missing_mask(
            group,
            value_name,
            label=f"Cell-aligned artifact array {value_name!r}",
            values=values_array,
        )
        if max_bytes is not None:
            n_target = (
                n_source if same_selection else _selection_count(root, target_selection)
            )
            row_arrays = [values_array, *caller_reads]
            if missing_array is not None:
                row_arrays.append(missing_array)
            required = _INDEX_BYTES * (
                n_source + n_target + (0 if same_selection else n_target)
            ) + sum(_chunkwise_read_bytes(array, n_target) for array in row_arrays)
            if required > max_bytes:
                raise MemoryError(
                    f"Reading {value_name!r} of the {artifact.kind} artifact for "
                    f"{n_target} cells needs about {required} bytes, but the "
                    f"memory budget is {max_bytes} bytes. Read fewer cells with "
                    "cell_selection, or raise mem_budget."
                )

        source_idx = _selection_indices(root, source_selection)
        if same_selection:
            target_idx = source_idx
            compact_idx = np.arange(len(source_idx), dtype=np.int64)
        else:
            target_idx = _selection_indices(root, target_selection)
            compact_idx = np.searchsorted(source_idx, target_idx).astype(
                np.int64,
                copy=False,
            )
            in_bounds = compact_idx < len(source_idx)
            if not bool(in_bounds.all()) or not np.array_equal(
                source_idx[compact_idx], target_idx
            ):
                raise ValueError(
                    "cell_selection must be a subset of the artifact cell selection"
                )
            # Only the positions of the requested cells are read from here.
            del source_idx, in_bounds
    values = read_array_rows_chunkwise(values_array, compact_idx)
    if values.shape != (len(target_idx), *shape[1:]):
        raise ValueError("Cell-aligned artifact values do not match the selection")
    missing = (
        None
        if missing_array is None
        else _row_missing_mask(
            np.asarray(
                read_array_rows_chunkwise(missing_array, compact_idx), dtype=bool
            ),
            value_name,
        )
    )
    return ResolvedCellArtifact(
        source=artifact,
        values=values,
        cell_idx=target_idx,
        source_cell_selection=source_selection,
        cell_selection=target_selection,
        missing_mask=missing,
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
        missing_mask=resolved.missing_mask,
    )


def resolve_complete_labels(
    root: Any,
    labels: ArtifactRef,
    *,
    name: str,
    remedy: str | None = None,
) -> ResolvedCellArtifact:
    """Resolve artifact labels that must assign every selected cell to a group.

    Producers that persist a per-group result cannot treat a stored
    placeholder as a label, so labels whose linked mask flags a row are
    rejected before any result is reused or written. The error ends with
    ``remedy``, which says how to make labels that the caller accepts. The
    default narrows the labels to their labelled cells, which suits a
    caller that takes labels over any cell selection.
    """
    if not isinstance(labels, ArtifactRef):
        raise TypeError(f"{name} must be an ArtifactRef")
    resolved = resolve_cell_aligned_artifact(
        root,
        labels,
        value_name=grouping_value_name(labels.kind),
        expected_kind=labels.kind,
    )
    if resolved.missing_mask is not None and bool(resolved.missing_mask.any()):
        if remedy is None:
            remedy = (
                f"Select the labelled cells with select_cells({name}, include=[...]) "
                f"and freeze their labels with snapshot_cluster_labels({name}, "
                "cell_selection=...)"
            )
        raise ValueError(f"{name} contains missing labels. {remedy}")
    return resolved


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
    when you want a compressed expression scale. With ``source="assay"`` it
    needs a normalizer that supports ``log_transform``.
    """

    source: NormSource = "assay"
    transform: NormTransform = "none"

    def __post_init__(self) -> None:
        if self.source not in ("assay", "raw"):
            raise ValueError("source must be 'assay' or 'raw'")
        if self.transform not in ("none", "log1p"):
            raise ValueError("transform must be 'none' or 'log1p'")
