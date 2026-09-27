"""Small helpers shared by Scarf domain-agent tools."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ...metadata.rows import (
    apply_missing_mask,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ..types import ArtifactReferenceModel

__all__ = [
    "artifact_reference",
    "bounded_list",
    "core_artifact_reference",
    "label_filter_bound",
    "mark_missing_rows",
    "persisted_assay_types",
    "read_marked_metadata_rows",
]


def mark_missing_rows(values: Any, missing: Any | None) -> np.ndarray:
    """Replace the placeholders on rows that a linked missing mask flags.

    Floating values become NaN and other values become None in an object
    array. Values without flagged rows are returned unchanged, so every agent
    reader sees the same missing rows as the evidence it was checked against.
    """
    array = np.asarray(values)
    return apply_missing_mask(array, missing, labels=array.dtype.kind != "f")


def read_marked_metadata_rows(
    metadata: Any,
    column: str,
    rows: np.ndarray,
) -> np.ndarray:
    """Read one metadata column on selected rows with missing rows marked."""
    values = np.asarray(read_metadata_rows_chunkwise(metadata, column, rows))
    if values.shape != (len(rows),):
        raise ValueError(f"Metadata column {column!r} does not align with its rows")
    return mark_missing_rows(
        values,
        read_metadata_missing_rows_chunkwise(metadata, column, rows),
    )


def bounded_list(values: Iterable[Any], *, limit: int) -> list[Any]:
    """Return at most ``limit`` JSON-facing values."""
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    output: list[Any] = []
    for value in values:
        output.append(value)
        if len(output) == limit:
            break
    return output


def label_filter_bound(value: Any) -> Any:
    """Return a ``filter_cells`` bound that selects rows equal to ``value``.

    Manual filter bounds reject booleans, so a boolean label becomes 0 or 1,
    which selects the same rows of a boolean column.
    """
    return int(value) if isinstance(value, bool | np.bool_) else value


def artifact_reference(ref: Any) -> ArtifactReferenceModel:
    """Convert a core artifact reference into an agent Pydantic model."""
    if isinstance(ref, ArtifactReferenceModel):
        return ArtifactReferenceModel.model_validate(ref.model_dump())
    return ArtifactReferenceModel.from_artifact_ref(ref)


def core_artifact_reference(ref: Any) -> Any:
    """Convert an agent artifact model back to Scarf's exact core reference."""
    if not isinstance(ref, ArtifactReferenceModel):
        return ref
    return ref.to_artifact_ref()


def persisted_assay_types(store: Any) -> dict[str, str]:
    """Read each assay's persisted type without building a datastore summary."""
    from ...assay.classification import lookup_persisted_assay_type

    raw = store.zw.attrs.get("assayTypes")
    assay_types = raw if isinstance(raw, Mapping) else None
    return {
        str(name): lookup_persisted_assay_type(str(name), assay_types)
        for name in store.assay_names
    }
