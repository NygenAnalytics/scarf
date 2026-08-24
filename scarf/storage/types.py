from typing import Literal

import zarr


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
