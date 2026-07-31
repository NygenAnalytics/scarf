import os
from typing import Any

import numpy as np
import zarr
from zarr.abc.store import Store

from .types import ZarrMode, as_zarr_array, as_zarr_group
from .profiles import (
    ZarrLocation,
    is_remote_zarr_location,
)

MATRIX_SOURCE_ATTR = "matrixSource"
_ASSAY_COPY_ATTRS = ("is_assay", "misc", "percentFeatures", "size_factor")
_WORKSPACE_COPY_ATTRS = ("defaultAssay", "assayTypes")


def zarr_group_root(group: zarr.Group, mode: ZarrMode = "r+") -> zarr.Group:
    """Open the root Zarr group sharing the same store as ``group``."""
    return zarr.open_group(store=group.store, mode=mode)


def zarr_root_path(node: zarr.Group | zarr.Array) -> str | None:
    """Return the filesystem path for a Zarr node when available."""
    store = node.store
    root = getattr(store, "root", None)
    if root is not None:
        return str(root)
    storePath = getattr(node, "store_path", None)
    if storePath and str(storePath).startswith("file://"):
        return str(storePath)[7:]
    return None


def is_remote_datastore(
    zarr_loc: ZarrLocation | None,
    node: zarr.Group | zarr.Array,
) -> bool:
    """Return whether a datastore uses a remote or object backend."""
    if isinstance(zarr_loc, str) and zarr_loc:
        return is_remote_zarr_location(zarr_loc)
    if zarr_root_path(node) is not None:
        return False
    store_name = type(node.store).__name__
    if store_name in ("MemoryStore", "LocalStore"):
        return False
    return True


def _is_obstore_native_store(obj: object) -> bool:
    return type(obj).__module__.startswith("obstore.")


def make_store(
    location: ZarrLocation,
    *,
    storage_options: dict[str, Any] | None = None,
    read_only: bool = False,
) -> str | Store:
    """Resolve a path, URI, or store for use with ``zarr.open_group``."""
    if isinstance(location, Store):
        return location

    if _is_obstore_native_store(location):
        from zarr.storage import ObjectStore

        return ObjectStore(store=location, read_only=read_only)  # type: ignore[type-var]

    if isinstance(location, str):
        if is_remote_zarr_location(location):
            if location.startswith("hf://"):
                from zarr.storage import FsspecStore

                return FsspecStore.from_url(
                    location,
                    storage_options=storage_options,
                    read_only=read_only,
                )
            try:
                from obstore.store import from_url as obstore_from_url
                from zarr.storage import ObjectStore
            except ImportError as exc:
                raise ImportError("Remote Zarr stores require obstore.") from exc
            obstore = obstore_from_url(location, **(storage_options or {}))
            return ObjectStore(store=obstore, read_only=read_only)  # type: ignore[type-var]
        return location

    raise TypeError(
        f"zarr location must be a path string or zarr Store, got {type(location)!r}"
    )


def open_store(
    path: ZarrLocation,
    mode: ZarrMode = "r",
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Open a Zarr group from a path, URI, or store object."""
    store = make_store(path, storage_options=storage_options, read_only=(mode == "r"))
    if isinstance(store, str):
        return zarr.open_group(store, mode=mode)
    return zarr.open_group(store=store, mode=mode)


def load_zarr(
    zarr_loc: ZarrLocation,
    mode: ZarrMode,
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Open a Zarr group through the compatibility entry point."""
    return open_store(zarr_loc, mode=mode, storage_options=storage_options)


def _persistable_location(source: str) -> str:
    """Return a location that resolves identically from any working directory."""
    if "://" in source:
        return source
    return os.path.abspath(source)


def _discard_target(target: zarr.Group, at: ZarrLocation) -> None:
    """Delete a target created by this call so the mount can be retried."""
    from zarr.core.sync import sync

    from ..utils.logging import logger

    try:
        sync(target.store_path.delete_dir())
    except Exception as exc:
        logger.warning(
            f"Could not remove the incomplete mount target at {at}: {exc}. "
            "Delete it before mounting again."
        )


def _workspace_group(root: zarr.Group, workspace: str | None) -> zarr.Group:
    if workspace is None:
        return root
    return as_zarr_group(root[workspace], name=workspace)


def _list_assay_names(root: zarr.Group, workspace: str | None) -> list[str]:
    zw = _workspace_group(root, workspace)
    assays: list[str] = []
    for name in sorted(dict.fromkeys(zw.group_keys())):
        node = zw[name]
        if isinstance(node, zarr.Group) and "is_assay" in node.attrs:
            assays.append(name)
    return assays


def _assay_identity(
    source_root: zarr.Group,
    assay: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    cell_ids_fingerprint: str | None,
) -> dict[str, Any]:
    from .artifacts import fingerprint_stored_strings
    from .schema import load_count_array

    counts = load_count_array(source_root, assay_name, workspace)
    entry: dict[str, Any] = {
        "shape": [int(counts.shape[0]), int(counts.shape[1])],
        "dtype": np.dtype(counts.dtype).str,
        "datasetFingerprint": None,
        "cellIdsFingerprint": None,
        "featureIdsFingerprint": None,
    }
    fingerprint = assay.attrs.get("dataset_fingerprint")
    if fingerprint is not None:
        entry["datasetFingerprint"] = str(fingerprint)
        return entry
    if cell_ids_fingerprint is None:
        raise RuntimeError("Cell identifier fingerprint was not calculated")
    feature_data = as_zarr_group(
        assay["featureData"],
        name=f"{assay_name}/featureData",
    )
    feature_ids = as_zarr_array(feature_data["ids"], name="ids")
    entry["cellIdsFingerprint"] = cell_ids_fingerprint
    entry["featureIdsFingerprint"] = fingerprint_stored_strings(feature_ids)
    return entry


def _validate_assay_identity(
    source_root: zarr.Group,
    assay: zarr.Group,
    assay_name: str,
    workspace: str | None,
    expected: dict[str, Any],
    *,
    cell_ids_fingerprint: str | None,
) -> None:
    from .artifacts import fingerprint_stored_strings
    from .schema import load_count_array

    counts = load_count_array(source_root, assay_name, workspace)
    shape = [int(counts.shape[0]), int(counts.shape[1])]
    dtype = np.dtype(counts.dtype).str
    if shape != list(expected.get("shape", [])) or dtype != expected.get("dtype"):
        raise ValueError(
            f"Matrix source assay {assay_name!r} no longer matches the mounted "
            "count matrix identity"
        )
    expected_fingerprint = expected.get("datasetFingerprint")
    if expected_fingerprint is not None:
        current = assay.attrs.get("dataset_fingerprint")
        if current is None or str(current) != str(expected_fingerprint):
            raise ValueError(
                f"Matrix source assay {assay_name!r} dataset fingerprint no longer "
                "matches the mounted store"
            )
        return
    if cell_ids_fingerprint is None:
        raise RuntimeError("Cell identifier fingerprint was not calculated")
    feature_data = as_zarr_group(
        assay["featureData"],
        name=f"{assay_name}/featureData",
    )
    feature_ids = as_zarr_array(feature_data["ids"], name="ids")
    if cell_ids_fingerprint != expected.get(
        "cellIdsFingerprint"
    ) or fingerprint_stored_strings(feature_ids) != expected.get(
        "featureIdsFingerprint"
    ):
        raise ValueError(
            f"Matrix source assay {assay_name!r} cell or feature identifiers no "
            "longer match the mounted store"
        )


def create_matrix_source(
    source: str,
    at: ZarrLocation,
    *,
    workspace: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Create a writable store that mounts count matrices from ``source``."""
    from .copy import copy_zarr_group_tree

    if not isinstance(source, str) or not source:
        raise TypeError("Matrix source location must be a non-empty string")
    source = _persistable_location(source)
    source_root = load_zarr(
        source,
        mode="r",
        storage_options=storage_options,
    )
    assay_names = _list_assay_names(source_root, workspace)
    if not assay_names:
        raise ValueError("No assays found in the matrix source")

    source_zw = _workspace_group(source_root, workspace)
    source_cell_data = as_zarr_group(
        source_zw["cellData"],
        name="cellData",
    )
    cell_ids = as_zarr_array(source_cell_data["ids"], name="ids")
    source_assays: dict[str, zarr.Group] = {}
    needs_id_fingerprint = False
    for assay_name in assay_names:
        source_assay = as_zarr_group(source_zw[assay_name], name=assay_name)
        feature_data = as_zarr_group(
            source_assay["featureData"],
            name=f"{assay_name}/featureData",
        )
        as_zarr_array(feature_data["ids"], name="ids")
        source_assays[assay_name] = source_assay
        needs_id_fingerprint = (
            needs_id_fingerprint
            or source_assay.attrs.get("dataset_fingerprint") is None
        )

    cell_ids_fingerprint: str | None = None
    if needs_id_fingerprint:
        from .artifacts import fingerprint_stored_strings

        cell_ids_fingerprint = fingerprint_stored_strings(cell_ids)

    assay_manifest = {
        assay_name: _assay_identity(
            source_root,
            source_assays[assay_name],
            assay_name,
            workspace,
            cell_ids_fingerprint=cell_ids_fingerprint,
        )
        for assay_name in assay_names
    }

    target = load_zarr(at, mode="w-", storage_options=storage_options)
    try:
        target_zw = target if workspace is None else target.create_group(workspace)
        for key in _WORKSPACE_COPY_ATTRS:
            if key in source_zw.attrs:
                target_zw.attrs[key] = source_zw.attrs[key]

        cell_data = target_zw.create_group("cellData")
        copy_zarr_group_tree(source_cell_data, cell_data)
        for assay_name in assay_names:
            source_assay = source_assays[assay_name]
            target_assay = target_zw.create_group(assay_name)
            for key in _ASSAY_COPY_ATTRS:
                if key in source_assay.attrs:
                    target_assay.attrs[key] = source_assay.attrs[key]
            feature_data = target_assay.create_group("featureData")
            copy_zarr_group_tree(
                as_zarr_group(source_assay["featureData"], name="featureData"),
                feature_data,
            )

        target.attrs[MATRIX_SOURCE_ATTR] = {
            "location": source,
            "workspace": workspace,
            "assays": assay_manifest,
        }
    except BaseException:
        _discard_target(target, at)
        raise
    return target


def resolve_matrix_source(
    root: zarr.Group,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[zarr.Group, str | None] | None:
    """Open and validate a mounted matrix source, if present."""
    raw = root.attrs.get(MATRIX_SOURCE_ATTR)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("matrixSource attribute must be a mapping")
    location = raw.get("location")
    if not isinstance(location, str) or not location:
        raise ValueError("matrixSource.location must be a non-empty string")
    workspace = raw.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        raise ValueError("matrixSource.workspace must be a string or null")
    assays = raw.get("assays")
    if not isinstance(assays, dict) or not assays:
        raise ValueError("matrixSource.assays must be a non-empty mapping")

    source_root = load_zarr(
        location,
        mode="r",
        storage_options=storage_options,
    )
    entries: list[tuple[str, dict[str, Any]]] = []
    for assay_name, expected in assays.items():
        if not isinstance(assay_name, str):
            raise ValueError("matrixSource assay names must be strings")
        if not isinstance(expected, dict):
            raise ValueError(
                f"matrixSource assay entry for {assay_name!r} must be a mapping"
            )
        entries.append((assay_name, expected))

    source_zw = _workspace_group(source_root, workspace)
    cell_ids_fingerprint: str | None = None
    if any(expected.get("datasetFingerprint") is None for _, expected in entries):
        from .artifacts import fingerprint_stored_strings

        cell_data = as_zarr_group(source_zw["cellData"], name="cellData")
        cell_ids = as_zarr_array(cell_data["ids"], name="ids")
        cell_ids_fingerprint = fingerprint_stored_strings(cell_ids)

    for assay_name, expected in entries:
        source_assay = as_zarr_group(source_zw[assay_name], name=assay_name)
        _validate_assay_identity(
            source_root,
            source_assay,
            assay_name,
            workspace,
            expected,
            cell_ids_fingerprint=cell_ids_fingerprint,
        )
    return source_root, workspace
