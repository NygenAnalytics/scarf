"""Raw-Zarr contracts for mandatory RNA strip ``countsT``.

These helpers inspect and gate stores without constructing ``RNAassay`` or
``DataStore``, so migration and mount paths can fail early with a concrete
repack remedy.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import zarr

from ..assay.classification import is_rna_assay_type, lookup_persisted_assay_type
from .sharding import is_strip_counts_t_layout
from .types import array_metadata_shards, as_zarr_array, as_zarr_group


CountsTInspectStatus = Literal[
    "not-rna",
    "ready",
    "missing",
    "incomplete",
    "zarr-v2",
    "unsupported-layout",
    "shape-dtype-mismatch",
]


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


def inspect_counts_t(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None = None,
    *,
    assay_type: str | None = None,
) -> CountsTInspectResult:
    """Classify ``countsT`` readiness from raw Zarr metadata."""
    attr_root = _workspace_root(root, workspace)
    raw_types = attr_root.attrs.get("assayTypes", {})
    type_map = (
        {str(k): str(v) for k, v in raw_types.items()}
        if isinstance(raw_types, dict)
        else {}
    )
    type_name = lookup_persisted_assay_type(
        assay_name,
        type_map,
        assay_type=assay_type,
    )
    if not is_rna_assay_type(type_name):
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
                f"RNA assay {assay_name!r} requires Zarr v3 for strip-sharded "
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
    shards = array_metadata_shards(counts_t)
    shape = tuple(int(v) for v in counts_t.shape)
    chunks = tuple(int(v) for v in counts_t.chunks)
    shard_shape = None if shards is None else tuple(int(v) for v in shards)
    if shards is None or not is_strip_counts_t_layout(
        shape=shape,
        chunks=chunks,
        shards=shard_shape,
        dtype=counts_t.dtype,
    ):
        return CountsTInspectResult(
            assayName=assay_name,
            assayType=type_name,
            status="unsupported-layout",
            reason=(
                f"RNA assay {assay_name!r} has an unsupported countsT layout "
                "(unsharded or non-strip). Rebuild with repack_zarr or "
                "write_counts_t."
            ),
        )
    return CountsTInspectResult(
        assayName=assay_name,
        assayType=type_name,
        status="ready",
        reason=f"countsT for {assay_name!r} is complete strip-sharded Zarr v3",
    )


def require_rna_counts_t_ready(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None = None,
    *,
    assay_type: str | None = None,
) -> None:
    """Raise when an RNA assay lacks a mount-safe strip ``countsT``."""
    result = inspect_counts_t(
        root,
        assay_name,
        workspace,
        assay_type=assay_type,
    )
    if result.status in {"not-rna", "ready"}:
        return
    raise ValueError(result.reason)
