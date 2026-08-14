"""Helpers for mandatory RNA strip ``countsT`` at ingest and subset."""

from typing import Any

import zarr

from ..assay.classification import (
    is_rna_assay_type,
    resolve_persisted_assay_type,
)
from ..storage.budget import ResourceBudget
from ..storage.counts_t_contract import (
    CountsTInspectResult as CountsTInspectResult,
)
from ..storage.counts_t_contract import (
    CountsTInspectStatus as CountsTInspectStatus,
)
from ..storage.counts_t_contract import inspect_counts_t as inspect_counts_t
from ..storage.counts_t_contract import (
    require_rna_counts_t_ready as require_rna_counts_t_ready,
)
from ..storage.profiles import StorageProfile
from ..storage.schema import load_count_array
from ..storage.sharding import finalize_rna_counts_t
from ..storage.types import as_zarr_group
from ..utils.logging import logger


def _workspace_root(z: zarr.Group, workspace: str | None) -> zarr.Group:
    if workspace is None:
        return z
    return as_zarr_group(z[workspace], name=workspace)


def seed_assay_type(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    assay_type: str,
) -> None:
    """Persist ``assayTypes[assay_name]`` using a recognized preset key only."""
    type_name = resolve_persisted_assay_type(assay_name, assay_type)
    root = _workspace_root(z, workspace)
    raw = root.attrs.get("assayTypes", {})
    types = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    if types.get(assay_name) == type_name:
        return
    types[assay_name] = type_name
    root.attrs["assayTypes"] = types


def matrix_group_for_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
) -> zarr.Group:
    if workspace is None:
        return as_zarr_group(z[assay_name], name=assay_name)
    return as_zarr_group(
        z[f"matrices/{assay_name}"],
        name=f"matrices/{assay_name}",
    )


def finalize_writer_counts_t(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    assay_type: str | None = None,
    resources: ResourceBudget | None = None,
    profile: StorageProfile | None = None,
    mem_budget: int | str | None = None,
    nthreads: int | None = None,
) -> zarr.Array | None:
    """Write strip ``countsT`` when the assay type is RNA; seed ``assayTypes``.

    When ``assay_type`` is omitted, ``assay_name`` is used only if it is a
    recognized preset (``RNA``, ``ADT``, …). Unknown names persist as
    ``Assay`` and skip ``countsT``. Pass an explicit preset ``assay_type`` to
    declare a custom assay group as RNA (or another modality).

    Returns ``None`` when skipped or when the store is Zarr format < 3.
    """
    type_name = resolve_persisted_assay_type(assay_name, assay_type)
    seed_assay_type(z, assay_name, workspace, type_name)
    if not is_rna_assay_type(type_name):
        return None
    counts = load_count_array(z, assay_name, workspace)
    group = matrix_group_for_assay(z, assay_name, workspace)
    counts_t = finalize_rna_counts_t(
        counts,
        group,
        profile=profile,
        resources=resources,
        mem_budget=mem_budget,
        nthreads=nthreads,
    )
    if counts_t is None:
        raise ValueError(
            f"RNA assay {assay_name!r} requires Zarr v3 for strip-sharded "
            "countsT. Repack the store to Zarr v3, or create the store with "
            "zarr_format=3."
        )
    logger.debug(f"Wrote strip countsT for RNA assay {assay_name}")
    return counts_t


def finalize_writer_counts_t_many(
    z: zarr.Group,
    assay_names: tuple[str, ...] | list[str],
    workspace: str | None,
    *,
    assay_types: dict[str, str] | None = None,
    resources: ResourceBudget | None = None,
    profile: StorageProfile | None = None,
) -> dict[str, Any]:
    """Finalize ``countsT`` for each assay name (RNA only)."""
    written: dict[str, Any] = {}
    type_map = assay_types or {}
    for name in assay_names:
        result = finalize_writer_counts_t(
            z,
            name,
            workspace,
            assay_type=type_map.get(name),
            resources=resources,
            profile=profile,
        )
        if result is not None:
            written[name] = result
    return written
