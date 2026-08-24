"""Raw-Zarr contracts for mandatory RNA ``countsT``.

These helpers inspect and gate stores without constructing ``RNAassay`` or
``DataStore``, so migration and mount paths can fail early with a concrete
repack remedy. Callers pass the resolved assay type; this module does not
import assay classification.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import zarr

from .count_matrix import REBUILD_REMEDY, require_count_matrix_layout
from .types import as_zarr_array, as_zarr_group


CountsTInspectStatus = Literal[
    "not-rna",
    "ready",
    "missing",
    "incomplete",
    "zarr-v2",
    "unsupported-layout",
    "shape-dtype-mismatch",
    "missing-layout-metadata",
    "layout-mismatch",
]

_RNA_ASSAY_TYPES = frozenset({"RNA", "GeneActivity", "GeneScores", "URNA"})


@dataclass(frozen=True, slots=True)
class CountsTInspectResult:
    """Raw-Zarr view of RNA ``countsT`` readiness without constructing assays."""

    assayName: str
    assayType: str
    status: CountsTInspectStatus
    reason: str


def _workspace_root(root: zarr.Group, workspace: str | None) -> zarr.Group:
    if workspace is None:
        return root
    return as_zarr_group(root[workspace], name=workspace)


def _resolve_assay_type(
    assay_name: str,
    attr_root: zarr.Group,
    assay_type: str | None,
) -> str:
    if assay_type is not None:
        return str(assay_type)
    raw_types = attr_root.attrs.get("assayTypes", {})
    if isinstance(raw_types, dict) and assay_name in raw_types:
        return str(raw_types[assay_name])
    return assay_name


def inspect_counts_t(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None = None,
    *,
    assay_type: str | None = None,
) -> CountsTInspectResult:
    """Classify ``countsT`` readiness from raw Zarr metadata."""
    attr_root = _workspace_root(root, workspace)
    type_name = _resolve_assay_type(assay_name, attr_root, assay_type)
    if type_name not in _RNA_ASSAY_TYPES:
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="not-rna",
            reason=f"{assay_name!r} is typed as {type_name!r}, not RNA",
        )
    matrix_path = assay_name if workspace is None else f"matrices/{assay_name}"
    if matrix_path not in root:
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="missing",
            reason=f"matrix group {matrix_path!r} is missing",
        )
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    if "countsT" not in matrix_group:
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="missing",
            reason=f"countsT is missing for {assay_name!r}",
        )
    counts_t = as_zarr_array(
        matrix_group["countsT"],
        name=f"{matrix_path}/countsT",
    )
    if counts_t.attrs.get("complete") is not True:
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="incomplete",
            reason=f"countsT is not complete for {assay_name!r}",
        )
    zarr_format = int(getattr(counts_t.metadata, "zarr_format", 3) or 3)
    if zarr_format < 3:
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="zarr-v2",
            reason=(
                f"RNA assay {assay_name!r} requires Zarr v3 for sharded "
                "countsT. Repack the store to Zarr v3."
            ),
        )
    from .schema import load_count_array

    try:
        counts = load_count_array(root, assay_name, workspace)
    except (KeyError, FileNotFoundError, TypeError):
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="missing",
            reason=f"counts is missing for {assay_name!r}",
        )
    expected_shape = (int(counts.shape[1]), int(counts.shape[0]))
    actual_shape = tuple(int(v) for v in counts_t.shape)
    if actual_shape != expected_shape or np.dtype(counts_t.dtype) != np.dtype(
        counts.dtype
    ):
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="shape-dtype-mismatch",
            reason=(
                f"countsT for {assay_name!r} has shape {actual_shape} "
                f"dtype {np.dtype(counts_t.dtype)}, expected shape "
                f"{expected_shape} dtype {np.dtype(counts.dtype)}"
            ),
        )
    try:
        require_count_matrix_layout(matrix_group, counts, counts_t)
    except ValueError as exc:
        message = str(exc)
        status: CountsTInspectStatus = (
            "missing-layout-metadata"
            if "missing" in message or "retired keys" in message
            else "layout-mismatch"
        )
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status=status,
            reason=f"{message} {REBUILD_REMEDY}"
            if REBUILD_REMEDY not in message
            else message,
        )
    return CountsTInspectResult(
        assayName=assay_name,
        assayType=type_name,
        status="ready",
        reason=f"countsT for {assay_name!r} is complete sharded Zarr v3",
    )


def require_rna_counts_t_ready(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None = None,
    *,
    assay_type: str | None = None,
) -> None:
    """Raise when an RNA assay lacks a mount-safe sharded ``countsT``."""
    result = inspect_counts_t(
        root,
        assay_name,
        workspace,
        assay_type=assay_type,
    )
    if result.status in {"not-rna", "ready"}:
        return
    raise ValueError(result.reason)
