"""Helpers for mandatory RNA ``countsT`` at ingest and subset."""

from typing import Any

import zarr

from ..assay.classification import (
    default_feature_sets,
    is_rna_assay_type,
    resolve_persisted_assay_type,
)
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import StorageProfile
from ..storage.schema import load_count_array
from ..storage.sharding import write_counts_t
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
    """Persist ``assayTypes[assay_name]`` using a recognized preset key only.

    Args:
        z: Root Zarr group.
        assay_name: Assay group name.
        workspace: Workspace name. None uses the legacy layout.
        assay_type: Preset type to store. Unrecognized values become ``Assay``.
    """
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
    """Return the Zarr group that owns ``counts`` and RNA ``countsT``.

    Args:
        z: Root Zarr group.
        assay_name: Assay group name.
        workspace: Workspace name. None uses the legacy layout.

    Returns:
        The assay matrix group.
    """
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
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
) -> zarr.Array | None:
    """Write paired ``countsT`` when the assay type is RNA; seed ``assayTypes``.

    When ``assay_type`` is omitted, ``assay_name`` is used only if it is a
    recognized preset (``RNA``, ``ADT``, …). Unknown names persist as
    ``Assay`` and skip ``countsT``. Pass an explicit preset ``assay_type`` to
    declare a custom assay group as RNA (or another modality).

    Returns ``None`` when skipped or when the store is Zarr format < 3.

    Args:
        z: Root Zarr group.
        assay_name: Assay group to finalize.
        workspace: Workspace name. None uses the legacy layout.
        assay_type: Optional preset used to seed ``assayTypes``.
        resources: Optional resolved memory and worker budget.
        profile: Zarr encoding profile. When None, chosen from the store.
        mem_budget: Memory budget used when ``resources`` is omitted.
        nthreads: Worker budget used when ``resources`` is omitted.
        policy: Count-matrix geometry policy. When None, the default plan
                is used.
        io: Optional explicit read, compute, and write widths.

    Returns:
        The ``countsT`` array, or None when the assay is not RNA.
    """
    logical = as_zarr_group(_workspace_root(z, workspace)[assay_name], name=assay_name)
    if logical.attrs.get("prepared") is True:
        raise ValueError(
            "Prepared counts cannot be finalized again; rebuild into a fresh destination"
        )
    type_name = resolve_persisted_assay_type(assay_name, assay_type)
    seed_assay_type(z, assay_name, workspace, type_name)
    counts = load_count_array(z, assay_name, workspace)
    resources = resources or resolve_budget(mem_budget, nthreads)
    if not is_rna_assay_type(type_name):
        return None
    group = matrix_group_for_assay(z, assay_name, workspace)
    counts_t = write_counts_t(
        counts,
        group,
        profile=profile,
        resources=resources,
        policy=policy,
        io=io,
        overwrite="countsT" in group,
        featureSets=default_feature_sets(logical),
    )
    logger.debug(f"Wrote paired countsT for RNA assay {assay_name}")
    return counts_t


def finalize_writer_counts_t_many(
    z: zarr.Group,
    assay_names: tuple[str, ...] | list[str],
    workspace: str | None,
    *,
    assay_types: dict[str, str] | None = None,
    resources: ResourceBudget | None = None,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
) -> dict[str, Any]:
    """Finalize ``countsT`` for each assay name (RNA only).

    Args:
        z: Root Zarr group.
        assay_names: Assay groups to consider.
        workspace: Workspace name. None uses the legacy layout.
        assay_types: Optional mapping of assay name to preset type.
        resources: Optional resolved memory and worker budget.
        profile: Zarr encoding profile. When None, chosen from the store.
        policy: Count-matrix geometry policy. When None, the default plan
                is used.
        io: Optional explicit read, compute, and write widths.

    Returns:
        Mapping of RNA assay name to the written ``countsT`` array.
    """
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
            policy=policy,
            io=io,
        )
        if result is not None:
            written[name] = result
    return written
