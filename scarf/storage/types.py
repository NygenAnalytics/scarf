from typing import Literal, overload

import zarr
from zarr.errors import GroupNotFoundError


type ZarrMode = Literal["r", "r+", "a", "w", "w-"]


def as_zarr_array(
    node: zarr.Array | zarr.Group,
    *,
    name: str = "",
) -> zarr.Array:
    if isinstance(node, zarr.Array):
        return node
    label = f" at {name!r}" if name else ""
    raise TypeError(f"Expected Zarr array{label}, got {type(node).__name__}")


def writable(array: zarr.Array) -> zarr.Array:
    """Return ``array`` configured to write every chunk it receives.

    Zarr otherwise compares each chunk against the fill value before encoding
    it. Scarf matrices rarely contain an all-fill chunk, and Zstd stores one in
    a few bytes, so that full scan costs more than it saves.
    """
    return array.with_config({"write_empty_chunks": True})


def as_zarr_group(
    node: zarr.Array | zarr.Group,
    *,
    name: str = "",
) -> zarr.Group:
    if isinstance(node, zarr.Group):
        return node
    label = f" at {name!r}" if name else ""
    raise TypeError(f"Expected Zarr group{label}, got {type(node).__name__}")


def array_metadata_shards(array: zarr.Array) -> tuple[int, ...] | None:
    return getattr(array.metadata, "shards", None)


@overload
def read_fresh_group(
    parent: zarr.Group,
    path: str = "",
    *,
    mode: Literal["r", "r+"] | None = None,
    missing_ok: Literal[False] = False,
) -> zarr.Group: ...


@overload
def read_fresh_group(
    parent: zarr.Group,
    path: str = "",
    *,
    mode: Literal["r", "r+"] | None = None,
    missing_ok: bool,
) -> zarr.Group | None: ...


def read_fresh_group(
    parent: zarr.Group,
    path: str = "",
    *,
    mode: Literal["r", "r+"] | None = None,
    missing_ok: bool = False,
) -> zarr.Group | None:
    """Open the group at ``path`` below ``parent`` from its stored record."""
    try:
        return zarr.open_group(
            store=parent.store,
            path=(parent.store_path / path).path,
            mode=("r" if parent.read_only else "r+") if mode is None else mode,
            zarr_format=parent.metadata.zarr_format,
        )
    except GroupNotFoundError:
        if missing_ok:
            return None
        raise
