from typing import Any

import zarr
from zarr.abc.store import Store

from .types import ZarrMode
from .profiles import (
    ZarrLocation,
    _maybe_auto_cloud_profile,
    configure_zarr_io_for_profile,
    is_remote_zarr_location,
)

type ZARRLOC = str | Store


def zarr_group_root(group: zarr.Group, mode: ZarrMode = "r+") -> zarr.Group:
    """Open the root Zarr group sharing the same store as ``group``."""
    return zarr.open_group(store=group.store, mode=mode)


def zarr_root_path(group: zarr.Group) -> str | None:
    """Return the filesystem path for a Zarr group when available."""
    store = group.store
    root = getattr(store, "root", None)
    if root is not None:
        return str(root)
    storePath = getattr(group, "store_path", None)
    if storePath and str(storePath).startswith("file://"):
        return str(storePath)[7:]
    return None


def is_remote_datastore(
    zarr_loc: ZarrLocation | None,
    group: zarr.Group,
) -> bool:
    """Return whether a datastore uses a remote or object backend."""
    if isinstance(zarr_loc, str) and zarr_loc:
        return is_remote_zarr_location(zarr_loc)
    if zarr_root_path(group) is not None:
        return False
    store_name = type(group.store).__name__
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
            try:
                from obstore.store import from_url as obstore_from_url
                from zarr.storage import ObjectStore
            except ImportError as exc:
                raise ImportError("Remote Zarr stores require obstore.") from exc
            _maybe_auto_cloud_profile(location)
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
    configure_zarr_io_for_profile()
    if isinstance(store, str):
        return zarr.open_group(store, mode=mode)
    return zarr.open_group(store=store, mode=mode)


def load_zarr(
    zarr_loc: ZARRLOC,
    mode: ZarrMode,
    synchronizer: Any = None,
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Open a local or remote Zarr group."""
    if synchronizer is not None:
        from ..utils.logging import logger

        logger.debug("ThreadSynchronizer is ignored under Zarr v3")
    store = make_store(
        zarr_loc,
        storage_options=storage_options,
        read_only=(mode == "r"),
    )
    configure_zarr_io_for_profile()
    if isinstance(store, str):
        return zarr.open_group(store, mode=mode)
    return zarr.open_group(store=store, mode=mode)
