import os
import posixpath
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import zarr
from zarr.abc.store import Store

from .types import ZarrMode, as_zarr_array, as_zarr_group
from .profiles import (
    StorageProfile,
    ZarrLocation,
    is_remote_zarr_location,
    resolve_storage_profile,
)

MATRIX_SOURCE_ATTR = "matrixSource"
_ASSAY_COPY_ATTRS = ("is_assay", "misc", "percentFeatures", "size_factor")
_WORKSPACE_COPY_ATTRS = ("defaultAssay", "assayTypes")


def _location_identity(location: str) -> tuple[str, str]:
    parsed = urlsplit(location)
    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else location
        return "file", str(Path(path).expanduser().resolve())
    normalized_path = posixpath.normpath(parsed.path or "/")
    return "uri", urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            normalized_path,
            parsed.query,
            parsed.fragment,
        )
    )


def locations_overlap(first: str, second: str) -> bool:
    first_kind, first_identity = _location_identity(first)
    second_kind, second_identity = _location_identity(second)
    if (first_kind, first_identity) == (second_kind, second_identity):
        return True
    if first_kind != second_kind:
        return False
    if first_kind == "file":
        first_path: Path | PurePosixPath = Path(first_identity)
        second_path: Path | PurePosixPath = Path(second_identity)
    else:
        first_uri = urlsplit(first_identity)
        second_uri = urlsplit(second_identity)
        if (first_uri.scheme, first_uri.netloc) != (
            second_uri.scheme,
            second_uri.netloc,
        ):
            return False
        first_path = PurePosixPath(first_uri.path)
        second_path = PurePosixPath(second_uri.path)
    if first_path == second_path:
        return True
    return first_path in second_path.parents or second_path in first_path.parents


def zarr_group_root(group: zarr.Group, mode: ZarrMode = "r+") -> zarr.Group:
    """Open the root Zarr group sharing the same store as ``group``."""
    return open_store(group.store, mode=mode)


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


# Small requests worth overlapping against an object store, where every group,
# array, or attribute access pays a network round trip.
REMOTE_METADATA_WORKERS = 32


def metadata_workers(node: zarr.Group | zarr.Array) -> int:
    """Return how many small metadata requests to overlap on ``node``'s store."""
    return REMOTE_METADATA_WORKERS if is_remote_datastore(None, node) else 1


def run_concurrently[T](tasks: Sequence[Callable[[], T]], *, workers: int) -> list[T]:
    """Run independent storage tasks in order, overlapping them when useful."""
    if workers <= 1 or len(tasks) <= 1:
        return [task() for task in tasks]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = [pool.submit(task) for task in tasks]
        return [future.result() for future in futures]


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
    from .async_execution import ensure_zarr_host_ceiling

    ensure_zarr_host_ceiling()
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


def zarr_location_has_content(
    location: ZarrLocation,
    *,
    storage_options: dict[str, Any] | None = None,
) -> bool:
    """Return whether a Zarr location already holds content.

    Local filesystem paths, including ``file://`` URIs, use path existence.
    Remote and in-memory stores are probed through the Zarr store API. Probe
    failures raise so callers can fail closed instead of overwriting blindly.
    """
    if isinstance(location, str) and not is_remote_zarr_location(location):
        path = location[7:] if location.startswith("file://") else location
        return os.path.lexists(path)

    store = make_store(location, storage_options=storage_options, read_only=True)
    if isinstance(store, str):
        return os.path.lexists(store)

    from zarr.core.sync import sync

    return not bool(sync(store.is_empty("")))


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


def create_matrix_source(
    source: str,
    at: ZarrLocation,
    *,
    workspace: str | None = None,
    required_transposes: frozenset[str],
    storage_options: dict[str, Any] | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    """Create a writable store that mounts count matrices from ``source``."""
    from .arrays import create_metadata_column
    from .copy import copy_zarr_group_tree

    if not isinstance(source, str) or not source:
        raise TypeError("Matrix source location must be a non-empty string")
    source = _persistable_location(source)
    if isinstance(at, str) and locations_overlap(source, at):
        raise ValueError("Source and destination must not overlap")
    source_root = load_zarr(
        source,
        mode="r",
        storage_options=storage_options,
    )
    if MATRIX_SOURCE_ATTR in source_root.attrs:
        raise ValueError(
            "Mounting a mounted target requires repacking it into a store that owns its counts first"
        )
    assay_names = _list_assay_names(source_root, workspace)
    if not assay_names:
        raise ValueError("No assays found in the matrix source")

    from .identity import count_fingerprint, validate_preparation, publish_preparation
    from .copy import validate_metadata_dependencies

    source_zw = _workspace_group(source_root, workspace)
    source_cell_data = as_zarr_group(source_zw["cellData"], name="cellData")
    validate_metadata_dependencies(source_cell_data)
    source_assays = {}
    source_matrices = {}
    assay_manifest = {}
    for name in assay_names:
        assay = as_zarr_group(source_zw[name], name=name)
        path = name if workspace is None else f"matrices/{name}"
        matrix = as_zarr_group(source_root[path], name=path)
        required = name in required_transposes
        fingerprint = validate_preparation(
            assay, source_cell_data, matrix, require_transpose=required
        )
        validate_metadata_dependencies(
            as_zarr_group(assay["featureData"], name="featureData")
        )
        counts = as_zarr_array(matrix["counts"], name="counts")
        source_assays[name] = assay
        source_matrices[name] = matrix
        assay_manifest[name] = {
            "datasetFingerprint": fingerprint,
            "countsFingerprint": count_fingerprint(counts),
            "requiresTranspose": required,
        }

    profile = resolve_storage_profile(at, profile)
    target = load_zarr(at, mode="w-", storage_options=storage_options)
    try:
        target_zw = target if workspace is None else target.create_group(workspace)
        for key in _WORKSPACE_COPY_ATTRS:
            if key in source_zw.attrs:
                target_zw.attrs[key] = source_zw.attrs[key]

        cell_data = target_zw.create_group("cellData")
        copy_zarr_group_tree(source_cell_data, cell_data, profile=profile)
        for assay_name in assay_names:
            source_assay = source_assays[assay_name]
            target_assay = target_zw.create_group(assay_name)
            target_assay.attrs["prepared"] = False
            for key in _ASSAY_COPY_ATTRS:
                if key in source_assay.attrs:
                    target_assay.attrs[key] = source_assay.attrs[key]
            feature_data = target_assay.create_group("featureData")
            source_feature_data = as_zarr_group(
                source_assay["featureData"],
                name="featureData",
            )
            # A mounted target is a newly created assay metadata store. Its
            # physical baseline must cover every feature row regardless of the
            # source's mutable ``I`` column.
            copy_zarr_group_tree(
                source_feature_data,
                feature_data,
                exclude_members={"I"},
                profile=profile,
            )
            source_feature_ids = as_zarr_array(
                source_feature_data["ids"],
                name="ids",
            )
            create_metadata_column(
                feature_data,
                "I",
                data=np.ones(int(source_feature_ids.shape[0]), dtype=bool),
                dtype=bool,
                chunkSize=100_000,
                profile=profile,
            )

            publish_preparation(
                target_assay,
                cell_data,
                source_matrices[assay_name],
                require_transpose=assay_name in required_transposes,
                expected_fingerprint=str(
                    assay_manifest[assay_name]["datasetFingerprint"]
                ),
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

    from .identity import REBUILD_REQUIRED, count_fingerprint, validate_preparation

    source_zw = _workspace_group(source_root, workspace)
    source_cells = as_zarr_group(source_zw["cellData"], name="cellData")
    if MATRIX_SOURCE_ATTR in source_root.attrs:
        raise ValueError(
            "The mounted source must own its count matrices; create a fresh target"
        )
    for assay_name, expected in entries:
        if set(expected) != {
            "datasetFingerprint",
            "countsFingerprint",
            "requiresTranspose",
        } or not isinstance(expected["requiresTranspose"], bool):
            raise ValueError(
                f"Existing mount has an unsupported identity contract; create a fresh target. {REBUILD_REQUIRED}"
            )
        source_assay = as_zarr_group(source_zw[assay_name], name=assay_name)
        path = assay_name if workspace is None else f"matrices/{assay_name}"
        matrix = as_zarr_group(source_root[path], name=path)
        fingerprint = validate_preparation(
            source_assay,
            source_cells,
            matrix,
            require_transpose=expected["requiresTranspose"],
        )
        counts = as_zarr_array(matrix["counts"], name="counts")
        if (
            fingerprint != expected["datasetFingerprint"]
            or count_fingerprint(counts) != expected["countsFingerprint"]
        ):
            raise ValueError(
                f"Matrix source assay {assay_name!r} no longer matches the mounted identity"
            )
    return source_root, workspace
