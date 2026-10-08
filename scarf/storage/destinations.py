"""The checks that writers run before they create a store at a destination.

A writer creates its store at an empty destination. With ``overwrite=True`` it
also replaces a Scarf store that no ``DataStore`` has opened: a root group that
holds no prepared assay, no ``matrixSource``, and only members that Scarf
writes. Every source that a writer reads is prepared, so no writer replaces
its source. A local destination inside another Zarr store and a destination
whose root group is an assay group are refused.

:func:`refuse_pending_assays` refuses a source store that holds a derived
assay left pending, which no writer copies.
"""

import os
import re
from pathlib import Path
from typing import Any

import zarr
from zarr.abc.store import Store
from zarr.core.sync import sync
from zarr.errors import ContainsArrayError
from zarr.storage import LocalStore, WrapperStore

from . import stores
from .profiles import ZarrLocation
from .schema import PENDING_ASSAY_ATTR, pending_assay_message, pending_assays

# Documents that make a directory a node of a Zarr hierarchy.
_NODE_DOCUMENTS = ("zarr.json", ".zgroup", ".zarray")
# Keys of a group's own metadata, which are not members of the group.
_METADATA_DOCUMENTS = frozenset({"zarr.json", ".zgroup", ".zattrs", ".zmetadata"})
# An array directory holds its metadata and its chunks: the "c" directory of
# Zarr v3, or the numbered chunks of Zarr v2.
_ARRAY_DOCUMENTS = frozenset({"zarr.json", ".zarray", ".zattrs"})
_CHUNK_ENTRY = re.compile(r"c|\d+(\.\d+)*")
# Groups that Scarf writes besides assays, at the root or in a workspace.
_SCARF_GROUPS = frozenset(
    {"cellData", "matrices", "artifacts", "pipeline", "agent_results"}
)


def _enclosing_store(path: Path) -> Path | None:
    """Return the outermost directory above ``path`` that is a Zarr node."""
    for parent in reversed(path.resolve().parents):
        if any((parent / name).is_file() for name in _NODE_DOCUMENTS):
            return parent
    return None


def _refuse_enclosed(location: ZarrLocation, path: Path) -> None:
    """Raise when the local directory ``path`` lies inside another Zarr store."""
    enclosing = _enclosing_store(path)
    if enclosing is not None:
        raise ValueError(
            f"Destination {location} lies inside the Zarr store at {enclosing}. "
            "A writer creates a new store and never writes into another store; "
            "choose a location outside every Zarr store."
        )


def _read_only_store(
    location: ZarrLocation, storage_options: dict[str, Any] | None
) -> Store | None:
    """Return the destination's store to read, or None for an absent local path."""
    store = stores.make_store(location, storage_options=storage_options, read_only=True)
    if not isinstance(store, str):
        backend = store
        while isinstance(backend, WrapperStore):
            backend = backend._store
        if isinstance(backend, LocalStore):
            _refuse_enclosed(location, Path(backend.root))
        return store
    # A local path stays a Path, which Zarr reads as it is written.
    path = Path(store)
    _refuse_enclosed(location, path)
    if not os.path.lexists(path):
        return None
    if not path.is_dir():
        raise FileExistsError(
            f"Destination {location} is a file, not a store. Choose another location."
        )
    return LocalStore(path, read_only=True)


async def _list_dir(store: Store, prefix: str) -> list[str]:
    return [name async for name in store.list_dir(prefix)]


def _members(group: zarr.Group) -> list[str]:
    """Return the names of the keys directly below ``group``, Zarr nodes or not."""
    names = sync(_list_dir(group.store, group.path))
    return [name for name in names if name not in _METADATA_DOCUMENTS]


def _scarf_member(name: str, group: zarr.Group, path: str, prepared: list[str]) -> bool:
    """Return whether Scarf writes ``group`` as ``name``; record a prepared assay."""
    if "is_assay" in group.attrs:
        if group.attrs.get("prepared") is True:
            prepared.append(path)
        return True
    return name in _SCARF_GROUPS or PENDING_ASSAY_ATTR in group.attrs


def _foreign_files(group: zarr.Group, path: str) -> list[str]:
    """Return the keys below ``group`` that are not Zarr metadata or chunks."""
    found: list[str] = []
    for name in _members(group):
        node = group.get(name)
        if node is None:
            found.append(f"{path}/{name}")
        elif isinstance(node, zarr.Group):
            found.extend(_foreign_files(node, f"{path}/{name}"))
        else:
            found.extend(
                f"{path}/{name}/{entry}"
                for entry in sync(_list_dir(node.store, node.path))
                if entry not in _ARRAY_DOCUMENTS and not _CHUNK_ENTRY.fullmatch(entry)
            )
    return found


def _replacement_refusal(root: zarr.Group) -> str | None:
    """Return why ``overwrite=True`` may not replace the store at ``root``, or None."""
    if stores.MATRIX_SOURCE_ATTR in root.attrs:
        return f"records a {stores.MATRIX_SOURCE_ATTR}, so it is a mounted store"
    prepared: list[str] = []
    foreign: list[str] = []
    for name in _members(root):
        node = root.get(name)
        if not isinstance(node, zarr.Group):
            foreign.append(name)
        elif _scarf_member(name, node, name, prepared):
            # A Scarf group holds only Zarr nodes, so a file such as notes
            # or .DS_Store inside one is someone else's.
            foreign.extend(_foreign_files(node, name))
        else:
            # Any other group is a workspace, which holds what a root holds.
            for member in _members(node):
                child = node.get(member)
                path = f"{name}/{member}"
                if not isinstance(child, zarr.Group) or not _scarf_member(
                    member, child, path, prepared
                ):
                    foreign.append(path)
                else:
                    foreign.extend(_foreign_files(child, path))
    if prepared:
        return f"holds the prepared assays {sorted(prepared)}"
    if foreign:
        return f"holds {foreign[0]!r}, which is not part of a Scarf store"
    return None


def _check(
    location: ZarrLocation,
    *,
    overwrite: bool,
    storage_options: dict[str, Any] | None,
) -> bool:
    """Run :func:`check_destination` and return whether the destination holds keys."""
    store = _read_only_store(location, storage_options)
    if store is None or sync(store.is_empty("")):
        return False
    root: zarr.Group | None
    try:
        # Each node's own metadata is read, never consolidated metadata.
        root = zarr.open_group(store=store, mode="r", use_consolidated=False)
    except (FileNotFoundError, ContainsArrayError):
        root = None
    if root is not None and "is_assay" in root.attrs:
        raise ValueError(
            f"Destination {location} is an assay group of a Zarr store, not a store "
            "of its own. Choose a location outside every Zarr store."
        )
    if not overwrite:
        raise FileExistsError(
            f"Destination {location} is not empty. Pass overwrite=True to replace a "
            "Scarf store that no DataStore has opened, or choose another location."
        )
    reason = (
        "holds content without a Zarr root group"
        if root is None
        else _replacement_refusal(root)
    )
    if reason is not None:
        raise FileExistsError(
            f"Destination {location} {reason}. overwrite=True replaces only a Scarf "
            "store that no DataStore has opened; delete it yourself or choose "
            "another location."
        )
    return True


def check_destination(
    location: ZarrLocation,
    *,
    overwrite: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Raise unless a writer may create its store at ``location``; write nothing.

    Args:
        location: Destination path, URI, or Zarr store.
        overwrite: Accept a Scarf store that no ``DataStore`` has opened.
        storage_options: Backend options used to open a URI.

    Raises:
        ValueError: If the destination lies inside a Zarr store or is an assay group.
        FileExistsError: If the destination is not empty and may not be replaced.
    """
    _check(location, overwrite=overwrite, storage_options=storage_options)


def create_destination(
    location: ZarrLocation,
    *,
    overwrite: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Check a destination with :func:`check_destination` and create its store.

    Args:
        location: Destination path, URI, or Zarr store.
        overwrite: Replace a Scarf store that no ``DataStore`` has opened.
        storage_options: Backend options used to open a URI.

    Returns:
        The new, empty root group, open for writing.
    """
    replace = _check(location, overwrite=overwrite, storage_options=storage_options)
    # Mode "w-" refuses keys that appeared since the check, and mode "w"
    # deletes the store that the check accepted.
    return stores.load_zarr(
        location, mode="w" if replace else "w-", storage_options=storage_options
    )


def refuse_pending_assays(
    root: zarr.Group,
    *,
    operation: str,
    subject: str = "The source store",
) -> None:
    """Raise unless a source store holds no derived assay left pending.

    Args:
        root: Root group of the source store, as the writer reads it.
        operation: What the store cannot be, such as ``"subset"`` or ``"repacked"``.
        subject: How the message names the store.

    Raises:
        ValueError: If the store holds a pending derived assay.
    """
    pending = pending_assays(root)
    if pending:
        raise ValueError(
            f"{subject} holds a pending derived assay and cannot be {operation}. "
            + pending_assay_message(*pending[0])
        )
