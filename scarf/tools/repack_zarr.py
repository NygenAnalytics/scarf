"""Repack a Zarr store from v2 to v3 with optional sharding."""

import argparse
from pathlib import Path

import zarr

from scarf.storage.zarr_store import (
    StorageProfile,
    array_info,
    finalize_sharded_counts,
    get_compressors,
    get_storage_profile,
    normalize_chunks,
    open_store,
)


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
        dst.create_array(
            key,
            data=node[...],
            chunks=chunks,
            compressors=get_compressors(profile, zarrFormat=3),
            overwrite=True,
        )


def repack_store(
    input_path: str,
    output_path: str,
    profile: StorageProfile = "fast_local",
    shard_counts: bool = True,
) -> None:
    src = open_store(input_path, mode="r")
    dst = open_store(output_path, mode="w")
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
    parser = argparse.ArgumentParser(description="Repack Zarr v2 stores to v3 with sharding")
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
