"""Repack a Zarr store from v2 to v3 with optional sharding."""

import argparse
from pathlib import Path

import numpy as np
import zarr

from scarf.storage.layout import array_info, get_compressors, normalize_chunks
from scarf.storage.profiles import StorageProfile, get_storage_profile
from scarf.storage.sharding import finalize_sharded_counts
from scarf.storage.stores import open_store


def _copy_group(
    src: zarr.Group,
    dst: zarr.Group,
    profile: StorageProfile,
) -> None:
    for key in src.keys():
        node = src[key]
        if isinstance(node, zarr.Group):
            new_group = dst.create_group(key, overwrite=True)
            for attr_key, attr_val in node.attrs.items():
                new_group.attrs[attr_key] = attr_val
            _copy_group(node, new_group, profile)
            continue
        chunks = normalize_chunks(node.chunks, node.shape)
        dst_array = dst.create_array(
            key,
            data=np.asarray(node[...]),
            chunks=chunks,
            compressors=get_compressors(profile, zarrFormat=3),
            overwrite=True,
        )
        for attr_key, attr_val in node.attrs.items():
            dst_array.attrs[attr_key] = attr_val


def repack_store(
    input_path: str,
    output_path: str,
    profile: StorageProfile = "fast_local",
    shard_counts: bool = True,
    storage_options: dict | None = None,
) -> None:
    """Copy a Zarr store to a new path and optionally shard count arrays.

    Args:
        input_path: Source Zarr directory.
        output_path: Destination Zarr directory (created or overwritten).
        profile: Storage profile for compressors and shard sizes.
        shard_counts: Repack assay count arrays to sharded layout when True.
    """
    src = open_store(input_path, mode="r", storage_options=storage_options)
    dst = open_store(output_path, mode="w", storage_options=storage_options)
    _copy_group(src, dst, profile)
    if not shard_counts:
        return
    for key in dst.keys():
        node = dst[key]
        if not isinstance(node, zarr.Group):
            continue
        if node.attrs.get("is_assay") and "counts" in node:
            counts = finalize_sharded_counts(dst, key, workspace=None, profile=profile)
            print(f"  {key}/counts: {array_info(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack Zarr v2 stores to v3 with sharding"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        choices=["fast_local", "cloud"],
        default=get_storage_profile(),
    )
    parser.add_argument("--no-shard-counts", action="store_true")
    args = parser.parse_args()
    repack_store(
        str(args.input),
        str(args.output),
        profile=args.profile,
        shard_counts=not args.no_shard_counts,
    )
    print(f"Repacked {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
