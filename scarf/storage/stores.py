import asyncio
import os
import posixpath
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterable,
    Sequence,
)
from typing import Any, Self, cast

import numpy as np
import zarr
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.common import ZARR_JSON
from zarr.storage import WrapperStore

from .types import ZarrMode, as_zarr_array, as_zarr_group
from .profiles import (
    StorageProfile,
    ZarrLocation,
    is_remote_zarr_location,
    local_zarr_path,
    resolve_storage_profile,
)

MATRIX_SOURCE_ATTR = "matrixSource"
_MATRIX_SOURCE_KEYS = frozenset({"location", "workspace", "assays"})
_MATRIX_SOURCE_ASSAY_KEYS = frozenset(
    {"datasetFingerprint", "countsFingerprint", "requiresTranspose"}
)
# Object stores stop after 10 retries by default, which is about nine seconds
# of backoff. Keep retrying transient errors for up to three minutes instead.
_REMOTE_RETRY_CONFIG = {"max_retries": 100, "retry_timeout": timedelta(minutes=3)}
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
    return None


def _is_local_store(store: Store) -> bool:
    return (
        getattr(store, "root", None) is not None
        or str(store).startswith("file://")
        or type(store).__name__ in ("MemoryStore", "LocalStore")
    )


def is_remote_datastore(
    zarr_loc: ZarrLocation | None,
    node: zarr.Group | zarr.Array,
) -> bool:
    """Return whether ``node`` is read from a remote or object backend.

    In a mounted artifact namespace the stores that serve ``node`` decide: a
    node inside an artifact group is read from the store that holds the group,
    and any other node, such as the root, is remote when the target or the
    source is. Otherwise a non-empty path or URI ``zarr_loc`` decides by its
    scheme, and ``node``'s store decides when ``zarr_loc`` is None, empty, or a
    ``Store``.
    """
    store = node.store
    if isinstance(store, MountedArtifactStore):
        from zarr.core.sync import sync

        return not all(map(_is_local_store, sync(store._node_stores(node.path))))
    if isinstance(zarr_loc, str) and zarr_loc:
        return is_remote_zarr_location(zarr_loc)
    return not _is_local_store(store)


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
            options = {"retry_config": _REMOTE_RETRY_CONFIG, **(storage_options or {})}
            obstore = obstore_from_url(location, **options)
            return ObjectStore(store=obstore, read_only=read_only)  # type: ignore[type-var]
        return local_zarr_path(location)

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
        # Zarr reads a string as a URL, which ends a local path at '#', '?',
        # or ';', so the path that the destination checks read is passed as a
        # Path, which Zarr opens as it is written.
        return zarr.open_group(Path(store), mode=mode)
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
    store = make_store(location, storage_options=storage_options, read_only=True)
    # Only a local path, including a file:// URI, resolves to a string.
    if isinstance(store, str):
        path = store[7:] if store.startswith("file://") else store
        return os.path.lexists(path)

    from zarr.core.sync import sync

    return not bool(sync(store.is_empty("")))


def _persistable_location(source: str) -> str:
    """Return a location that resolves identically from any working directory."""
    if "://" in source:
        return source
    return os.path.abspath(source)


def discard_mount_target(target: zarr.Group, at: ZarrLocation) -> None:
    """Delete a mount target that the mount being made created."""
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
    from .destinations import refuse_pending_assays

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
    refuse_pending_assays(source_root, operation="mounted")
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
        discard_mount_target(target, at)
        raise
    return target


def resolve_matrix_source(
    root: zarr.Group,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[zarr.Group, str | None] | None:
    """Open and validate a mounted matrix source, if present.

    The manifest holds exactly ``location``, ``workspace``, and ``assays``, and
    each assay entry exactly its recorded identity. Any other shape, including
    a field added by a later release, is rejected rather than ignored.
    """
    raw = root.attrs.get(MATRIX_SOURCE_ATTR)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("matrixSource attribute must be a mapping")
    if set(raw) != _MATRIX_SOURCE_KEYS:
        raise ValueError(
            "Existing mount has an unsupported matrix source contract; create a "
            "fresh target with mount_datastore"
        )
    location = raw.get("location")
    if not isinstance(location, str) or not location:
        raise ValueError("matrixSource.location must be a non-empty string")
    workspace = raw.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        raise ValueError("matrixSource.workspace must be a string or null")
    assays = raw.get("assays")
    if not isinstance(assays, dict) or not assays:
        raise ValueError("matrixSource.assays must be a non-empty mapping")

    entries: list[tuple[str, dict[str, Any]]] = []
    for assay_name, expected in assays.items():
        if not isinstance(expected, dict):
            raise ValueError(
                f"matrixSource assay entry for {assay_name!r} must be a mapping"
            )
        if set(expected) != _MATRIX_SOURCE_ASSAY_KEYS or not isinstance(
            expected["requiresTranspose"], bool
        ):
            raise ValueError(
                f"Existing mount has an unsupported identity contract for assay "
                f"{assay_name!r}; create a fresh target with mount_datastore"
            )
        entries.append((assay_name, expected))

    from .identity import count_fingerprint, validate_preparation

    source_root = load_zarr(
        location,
        mode="r",
        storage_options=storage_options,
    )
    source_zw = _workspace_group(source_root, workspace)
    source_cells = as_zarr_group(source_zw["cellData"], name="cellData")
    if MATRIX_SOURCE_ATTR in source_root.attrs:
        raise ValueError(
            "The mounted source must own its count matrices; create a fresh target"
        )
    for assay_name, expected in entries:
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


_NO_SYNC_IO = (
    "A mounted artifact namespace routes keys only through asynchronous store IO"
)


class MountedArtifactStore(WrapperStore[Store]):
    """A mounted target that resolves its matrix source's artifacts read only.

    Artifacts are immutable and their IDs are random 256-bit tokens, so a group
    the source holds is the artifact that a copy in the target would be. Keys
    below the artifact roots, ``[workspace/]artifacts`` and
    ``[workspace/]{assay}/artifacts`` for every mounted assay, are routed by the
    group ``{root}/{kind}/{id}`` that holds them: a group whose ``zarr.json``
    the source holds and the target does not is read from the source, and
    every other key goes to the target. Listings of a root and of its kind
    directories are the union of both stores, a listing of the group above a
    root names the root when only the source holds it, and a directory
    document the target lacks is read from the source. Keys outside the roots,
    such as cell and feature tables and ``pipeline/``, belong to the target.

    Every write goes to the target. Writing or deleting inside a source group
    raises ``PermissionError``. The store offers no synchronous IO, so Zarr's
    synchronous fast paths take the routed asynchronous ones. Closing it closes
    the target only; the source store belongs to the root that opened it.
    """

    def __init__(self, store: Store, source: Store, roots: Iterable[str]) -> None:
        super().__init__(store)
        self._source = source
        self._roots = tuple(roots)
        # Whether each artifact group is read from the source. A group found in
        # neither store is not cached, so one the target creates later is
        # resolved again. Copies made by with_read_only share the cache. It
        # holds plain values, never loop-bound futures, because storage
        # coroutines run on several event loops.
        self._in_source: dict[str, bool] = {}

    def _with_store(self, store: Store) -> Self:
        clone = type(self)(store, self._source, self._roots)
        clone._in_source = self._in_source
        return clone

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, MountedArtifactStore)
            and self._store == value._store
            and self._source == value._source
            and self._roots == value._roots
        )

    def __str__(self) -> str:
        return str(self._store)

    def __repr__(self) -> str:
        return f"MountedArtifactStore({self._store!r}, source={self._source!r})"

    @property
    def root(self) -> Any:
        """The target's filesystem root, so location checks see the mount itself."""
        return getattr(self._store, "root", None)

    @property
    def _supports_sync_io(self) -> bool:
        return False

    def _in_tree(self, key: str) -> bool:
        return any(key == root or key.startswith(f"{root}/") for root in self._roots)

    def _group(self, key: str) -> str | None:
        """Return the artifact group that holds ``key``, if any."""
        for root in self._roots:
            if key.startswith(f"{root}/"):
                parts = key[len(root) + 1 :].split("/", 2)
                if len(parts) > 1 and parts[1] not in ("", ZARR_JSON):
                    return f"{root}/{parts[0]}/{parts[1]}"
                return None
        return None

    async def _reads_source(self, group: str) -> bool:
        cached = self._in_source.get(group)
        if cached is not None:
            return cached
        document = f"{group}/{ZARR_JSON}"
        if await self._store.exists(document):
            in_source = False
        elif await self._source.exists(document):
            in_source = True
        else:
            return False
        self._in_source[group] = in_source
        return in_source

    async def _owner(self, key: str) -> Store:
        """Return the store that holds ``key`` in the namespace."""
        group = self._group(key)
        if group is not None:
            return self._source if await self._reads_source(group) else self._store
        if (
            self._in_tree(key)
            and not await self._store.exists(key)
            and await self._source.exists(key)
        ):
            return self._source
        return self._store

    async def _node_stores(self, path: str) -> tuple[Store, ...]:
        """Return the stores that serve the node at ``path``."""
        group = self._group(posixpath.join(path, ZARR_JSON))
        if group is None:
            return self._store, self._source
        return (self._source if await self._reads_source(group) else self._store,)

    async def _refuse_source_write(self, key: str) -> None:
        group = self._group(key)
        if group is not None and await self._reads_source(group):
            raise PermissionError(
                f"Artifact {group!r} belongs to the read-only matrix source of "
                "this mount"
            )

    def _forget(self, path: str) -> None:
        """Drop the cached origins of groups that a delete of ``path`` removes."""
        path = path.rstrip("/").removesuffix(f"/{ZARR_JSON}")
        # Storage coroutines run on several event-loop threads, so iterate a copy.
        for group in tuple(self._in_source):
            if not path or group == path or group.startswith(f"{path}/"):
                self._in_source.pop(group, None)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        group = self._group(key)
        if (group is None and self._in_tree(key)) or (
            group is not None
            and group not in self._in_source
            and key == f"{group}/{ZARR_JSON}"
        ):
            # A directory document, or a group document whose read decides the
            # group's origin: the target's copy wins over the source's.
            for in_source, store in ((False, self._store), (True, self._source)):
                value = await store.get(key, prototype, byte_range)
                if value is not None:
                    if group is not None:
                        self._in_source[group] = in_source
                    return value
            return None
        return await (await self._owner(key)).get(key, prototype, byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # WrapperStore would forward the batch to the target unrouted.
        return await asyncio.gather(
            *(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)
        )

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        # WrapperStore would forward the batch to the target unrouted; the base
        # implementation reads each key through get.
        async for item in Store._get_many(self, requests):
            yield item

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int | None = None,
        max_gap_bytes: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        # Sharded reads fetch inner chunks here, so a source shard must be read
        # from the source with that store's own coalescing.
        options = {
            name: value
            for name, value in (
                ("max_concurrency", max_concurrency),
                ("max_gap_bytes", max_gap_bytes),
                ("max_coalesced_bytes", max_coalesced_bytes),
            )
            if value is not None
        }
        store = await self._owner(key)
        async for group in store.get_ranges(
            key, byte_ranges, prototype=prototype, **options
        ):
            yield group

    async def exists(self, key: str) -> bool:
        return await (await self._owner(key)).exists(key)

    async def getsize(self, key: str) -> int:
        return await (await self._owner(key)).getsize(key)

    async def is_empty(self, prefix: str) -> bool:
        return await Store.is_empty(self, prefix)

    def list(self) -> AsyncIterator[str]:
        return self.list_prefix("")

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        seen: set[str] = set()
        async for key in self._store.list_prefix(prefix):
            if self._group(key) is None or await self._owner(key) is self._store:
                seen.add(key)
                yield key
        if self._in_tree(prefix.rstrip("/")):
            source_prefixes = [prefix]
        else:
            source_prefixes = [
                f"{root}/" for root in self._roots if root.startswith(prefix)
            ]
        for source_prefix in source_prefixes:
            async for key in self._source.list_prefix(source_prefix):
                if key not in seen and await self._owner(key) is self._source:
                    yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        path = prefix.rstrip("/")
        group = self._group(path)
        if group is not None:
            owner = self._source if await self._reads_source(group) else self._store
            async for name in owner.list_dir(prefix):
                yield name
            return
        names: set[str] = set()
        async for name in self._store.list_dir(prefix):
            names.add(name)
            yield name
        if self._in_tree(path):
            async for name in self._source.list_dir(prefix):
                if name not in names:
                    names.add(name)
                    yield name
            return
        for root in self._roots:
            parent, _, name = root.rpartition("/")
            if (
                parent == path
                and name not in names
                and await self._source.exists(f"{root}/{ZARR_JSON}")
            ):
                names.add(name)
                yield name

    async def set(self, key: str, value: Buffer) -> None:
        await self._refuse_source_write(key)
        await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        await self._refuse_source_write(key)
        await self._store.set_if_not_exists(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        items = list(values)
        for key, _ in items:
            await self._refuse_source_write(key)
        await self._store._set_many(items)

    async def delete(self, key: str) -> None:
        await self._refuse_source_write(key)
        await self._store.delete(key)
        self._forget(key)

    async def delete_dir(self, prefix: str) -> None:
        await self._refuse_source_write(prefix.rstrip("/"))
        await self._store.delete_dir(prefix)
        self._forget(prefix)

    async def clear(self) -> None:
        await self._store.clear()
        self._in_source.clear()

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        raise TypeError(_NO_SYNC_IO)

    def set_sync(self, key: str, value: Buffer) -> None:
        raise TypeError(_NO_SYNC_IO)

    def delete_sync(self, key: str) -> None:
        raise TypeError(_NO_SYNC_IO)


def mount_artifact_namespace(
    target_root: zarr.Group,
    source_root: zarr.Group,
    workspace: str | None,
) -> zarr.Group:
    """Reopen a mounted target so that it resolves its source's artifacts.

    ``target_root`` is the root of a target whose manifest
    :func:`resolve_matrix_source` validated, and ``source_root`` and
    ``workspace`` are what it returned. The namespace keeps the target's access
    mode and reuses the open source store, so it needs no second connection or
    credentials.
    """
    prefix = "" if workspace is None else f"{workspace}/"
    manifest = cast(dict[str, Any], target_root.attrs[MATRIX_SOURCE_ATTR])
    roots = [
        f"{prefix}artifacts",
        *(f"{prefix}{assay}/artifacts" for assay in manifest["assays"]),
    ]
    store = MountedArtifactStore(target_root.store, source_root.store, roots)
    return open_store(store, mode="r" if target_root.read_only else "r+")
