import colorsys
import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import zarr

from ..storage.arrays import (
    _decode_metadata_values,
    _measured_text_dtype,
    create_metadata_column,
    create_zarr_dataset,
)
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
    reused_artifact_group,
)
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    fingerprint_array,
    fingerprint_stored_arrays,
)
from ..storage.selections import validate_cell_selection
from ..storage.types import as_zarr_array, as_zarr_group
from .selection import CellField, resolve_grouping, valid_category_mask

_CATEGORY_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
)
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_TEXT_LABEL_KINDS = frozenset({"O", "S", "T", "U"})
_LABEL_KINDS = _TEXT_LABEL_KINDS | {"b", "i", "u"}
# int64 holds the whole numbers in [-_INT64_BOUND, _INT64_BOUND). A float64
# bound, unlike a Python float, compares with float16 labels without overflow.
_INT64_BOUND = np.float64(2.0**63)


def validate_display_metadata(
    display: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(display)
    kind = value.get("kind")
    if kind == "continuous":
        required = {"kind", "colormap", "minimum", "maximum", "scale"}
        if not required.issubset(value):
            raise ValueError("Continuous display metadata is incomplete")
        if set(value) != required:
            raise ValueError("Continuous display metadata has unknown fields")
        if not isinstance(value.get("colormap"), str):
            raise TypeError("Continuous display colormap must be a string")
        if value.get("scale") not in {"linear", "log", "symlog"}:
            raise ValueError("Continuous display scale is invalid")
        minimum = value.get("minimum")
        maximum = value.get("maximum")
        if minimum is not None and (
            isinstance(minimum, bool)
            or not isinstance(minimum, int | float)
            or not np.isfinite(float(minimum))
        ):
            raise TypeError("Continuous display minimum must be numeric or null")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int | float)
            or not np.isfinite(float(maximum))
        ):
            raise TypeError("Continuous display maximum must be numeric or null")
        if (
            minimum is not None
            and maximum is not None
            and float(minimum) > float(maximum)
        ):
            raise ValueError("Continuous display minimum exceeds maximum")
        return value
    if kind == "categorical":
        if not {"kind", "categories"}.issubset(value):
            raise ValueError("Categorical display metadata is incomplete")
        allowed = {
            "kind",
            "categories",
            "missing_label",
            "missing_color",
        }
        if not set(value).issubset(allowed):
            raise ValueError("Categorical display metadata has unknown fields")
        categories = value.get("categories")
        if not isinstance(categories, list):
            raise TypeError("Categorical display categories must be a list")
        seen_values: set[tuple[str, Any]] = set()
        mapping_values: list[Any] = []
        for category in categories:
            if not isinstance(category, dict):
                raise TypeError("Each display category must be a mapping")
            if set(category) != {"value", "label", "color"}:
                raise ValueError(
                    "Each display category requires value, label, and color"
                )
            category_value = category["value"]
            if category_value is None or (
                not isinstance(
                    category_value,
                    bool | int | float | str,
                )
                or (
                    isinstance(category_value, float)
                    and not np.isfinite(category_value)
                )
            ):
                raise TypeError("Display category value must be a JSON scalar")
            typed_value = (
                type(category_value).__name__,
                category_value,
            )
            if typed_value in seen_values:
                raise ValueError("Display category values must be unique")
            if any(category_value == existing for existing in mapping_values):
                raise ValueError("Display category values collide as mapping keys")
            seen_values.add(typed_value)
            mapping_values.append(category_value)
            if not isinstance(category.get("label"), str):
                raise TypeError("Display category label must be a string")
            color = category.get("color")
            if not isinstance(color, str) or _HEX_COLOR.fullmatch(color) is None:
                raise ValueError("Display category color must be a hex color")
        missing_label = value.get("missing_label")
        if missing_label is not None and not isinstance(missing_label, str):
            raise TypeError("Categorical missing_label must be a string")
        missing_color = value.get("missing_color")
        if missing_color is not None and (
            not isinstance(missing_color, str)
            or _HEX_COLOR.fullmatch(missing_color) is None
        ):
            raise ValueError("Categorical missing_color must be a hex color")
        return value
    raise ValueError("Display kind must be continuous or categorical")


def plan_cell_data_artifact(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    execution_options: dict[str, Any],
    cell_selection: ArtifactRef,
    arrays: Mapping[str, tuple[tuple[int, ...], str | None]],
    assay: str | None = None,
    invalidate_cache: bool = False,
    required_attributes: tuple[str | AttributeRequirement, ...] = (),
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
) -> PlannedArtifact:
    if cell_selection.kind != "cell_selection":
        raise ValueError("cell_selection must reference a cell-selection artifact")
    selected_count = int(validate_cell_selection(root, cell_selection).selected_count)
    if any(shape[0] != selected_count for shape, _kind in arrays.values()):
        raise ValueError("Artifact arrays must align with the selected cell count")
    artifact_inputs = {**inputs, "cell_selection": cell_selection}
    requirements = tuple(
        ArrayRequirement(name, shape=shape, dtype_kind=dtype_kind)
        for name, (shape, dtype_kind) in arrays.items()
    )
    return plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters=parameters,
        inputs=artifact_inputs,
        execution_options=execution_options,
        invalidate_cache=invalidate_cache,
        required_arrays=requirements,
        required_attributes=required_attributes,
        reuse_validator=reuse_validator,
    )


def write_cell_data_artifact(
    root: zarr.Group,
    planned: PlannedArtifact,
    arrays: Mapping[str, np.ndarray],
    *,
    fingerprint_payload: bool = False,
) -> zarr.Group:
    if planned.reused:
        return reused_artifact_group(root, planned)
    with artifact_transaction(root, planned) as group:
        for name, raw_values in arrays.items():
            values = np.asarray(raw_values)
            if values.ndim < 1:
                raise ValueError("Artifact arrays must have at least one dimension")
            if values.dtype.kind in {"O", "S", "U"}:
                if values.ndim != 1:
                    raise ValueError("String artifact arrays must be one-dimensional")
                create_metadata_column(
                    group,
                    name,
                    data=values.astype(str),
                    overwrite=True,
                    chunkSize=min(max(int(values.shape[0]), 1), 100_000),
                )
            else:
                chunks = (
                    min(max(int(values.shape[0]), 1), 100_000),
                    *values.shape[1:],
                )
                output = create_zarr_dataset(
                    group,
                    name,
                    chunks,
                    values.dtype,
                    values.shape,
                )
                output[...] = values
        if fingerprint_payload:
            group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
                group,
                tuple(arrays),
            )
    return group


def snapshot_cluster_labels(
    root: zarr.Group,
    cells: Any,
    labels: str | ArtifactRef,
    *,
    cell_selection: ArtifactRef,
) -> ArtifactRef:
    """Freeze one label per selected cell as a ``cluster_labels`` artifact.

    ``labels`` names a column of ``cells`` or references a cell-label
    artifact, whose own cell selection must contain ``cell_selection``.
    Labels must be text, integer, or boolean values, or floating-point whole
    numbers within the int64 range, such as the float64 ids that pandas
    writes for integer ids with missing values. Other floating-point labels
    are rejected whether or not they hold NaN, and every selected cell needs
    a label that is not missing, masked, or blank. Text is stored at the
    width of the selected labels, so the declared width and encoding of the
    source do not change the identity; floating-point whole numbers are
    stored as int64, and integer and boolean labels keep their dtype. The
    identity holds the source column name or source artifact, the cell
    selection, and a fingerprint of the stored labels, and a match is
    reused only while its payload is exactly those labels.

    Raises:
        TypeError: If ``labels`` or ``cell_selection`` has the wrong type, or
            if a label is not a text, integer, boolean, or whole-number
            value.
        ValueError: If a selected cell has no label.
        PermissionError: If no snapshot matches and ``root`` is read-only.
    """
    source: ArtifactRef | CellField
    parameters: dict[str, Any]
    inputs: dict[str, Any]
    if isinstance(labels, str):
        source = CellField(labels, kind="categorical")
        described = f"column {labels!r}"
        example = "snapshot_cell_selection of a boolean column that marks them"
        parameters, inputs = {"source_column": labels}, {}
    elif isinstance(labels, ArtifactRef):
        source = labels
        described = f"the {labels.kind} artifact"
        example = "select_cells(labels, include=[...])"
        parameters, inputs = {}, {"source_labels": labels}
    else:
        raise TypeError(
            "labels must be a cell metadata column name or a cell-label ArtifactRef"
        )
    if not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    resolved = resolve_grouping(root, cells, source, cell_selection=cell_selection)
    valid = valid_category_mask(resolved.labels, missing_mask=resolved.missing_mask)
    kind = resolved.labels.dtype.kind
    if kind == "f":
        # Checked before completeness but over labelled cells only, so neither
        # NaN nor a masked placeholder decides whether the floats are ids.
        labelled = resolved.labels[valid]
        whole = (labelled >= -_INT64_BOUND) & (labelled < _INT64_BOUND)
        whole &= labelled == np.trunc(labelled)
        if not whole.all():
            raise TypeError(
                "Floating-point cluster labels must be whole numbers within the "
                f"int64 range, but {described} holds {labelled[~whole][0]!s}"
            )
    elif kind not in _LABEL_KINDS:
        raise TypeError(
            "Cluster labels must be text, integer, boolean, or whole-number "
            f"values, but {described} holds {resolved.labels.dtype} values"
        )
    unlabelled = int(np.count_nonzero(~valid))
    if unlabelled:
        raise ValueError(
            f"{unlabelled} of {valid.size} selected cells have no label in "
            f"{described}. Pass a cell_selection of labelled cells only, such "
            f"as {example}"
        )
    values = _decode_metadata_values(resolved.labels)
    if kind == "f":
        values = values.astype(np.int64)
    elif kind in _TEXT_LABEL_KINDS:
        values = values.astype(_measured_text_dtype((values,)))
    fingerprint = fingerprint_array(values)
    inputs["values_fingerprint"] = fingerprint

    def payload_matches(_ref: ArtifactRef, group: zarr.Group) -> bool:
        # Reuse and completion checks call this only after the values array
        # matched the planned shape and dtype kind.
        return (
            not set(group.group_keys())
            and set(group.array_keys()) == {"values"}
            and fingerprint_stored_arrays(group, ("values",)) == fingerprint
        )

    planned = plan_cell_data_artifact(
        root,
        scope="datastore",
        kind="cluster_labels",
        operation="snapshot_cluster_labels",
        parameters=parameters,
        inputs=inputs,
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (values.shape, values.dtype.kind)},
        reuse_validator=payload_matches,
    )
    write_cell_data_artifact(root, planned, {"values": values})
    return planned.ref


def artifact_values(
    group: zarr.Group,
    name: str,
    value_index: int | None = None,
) -> np.ndarray:
    values = np.asarray(as_zarr_array(group[name], name=name)[:])
    return values if value_index is None else values[:, value_index]


def column_display(
    root: zarr.Group,
    column: str,
) -> dict[str, Any] | None:
    cell_data = as_zarr_group(root["cellData"], name="cellData")
    if column not in cell_data:
        return None
    attrs = as_zarr_array(
        cell_data[column],
        name=column,
    ).attrs
    if "display" not in attrs:
        return None
    raw_display = attrs["display"]
    if not isinstance(raw_display, Mapping):
        raise TypeError("Existing display metadata must be a mapping")
    return validate_display_metadata(raw_display)


def continuous_display(
    values: np.ndarray,
    *,
    colormap: str = "viridis",
    scale: str = "linear",
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    minimum = float(finite.min()) if len(finite) else None
    maximum = float(finite.max()) if len(finite) else None
    return {
        "kind": "continuous",
        "colormap": colormap,
        "minimum": minimum,
        "maximum": maximum,
        "scale": scale,
    }


def _category_color(index: int, count: int) -> str:
    if count <= len(_CATEGORY_COLORS):
        return _CATEGORY_COLORS[index]
    red, green, blue = colorsys.hsv_to_rgb(
        (index * 0.618033988749895) % 1.0,
        0.55,
        0.85,
    )
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def categorical_display(values: np.ndarray) -> dict[str, Any]:
    categories: list[Any] = []
    seen: set[tuple[str, str]] = set()
    has_missing = False
    for raw_value in np.asarray(values).reshape(-1):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, float) and np.isnan(value):
            value = None
        if value is None:
            has_missing = True
            continue
        key = (type(value).__name__, repr(value))
        if key not in seen:
            seen.add(key)
            categories.append(value)
    try:
        categories = sorted(categories)
    except TypeError:
        pass
    count = len(categories)
    display = {
        "kind": "categorical",
        "categories": [
            {
                "value": value,
                "label": "NA" if value is None else str(value),
                "color": _category_color(index, count),
            }
            for index, value in enumerate(categories)
        ],
    }
    if has_missing:
        display["missing_label"] = "NA"
        display["missing_color"] = "#bdbdbd"
    return display
