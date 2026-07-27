"""Repack a Zarr store from v2 to v3 with optional sharding."""

import argparse
import posixpath
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import zarr

from scarf.storage.arrays import create_numeric_array
from scarf.storage.layout import (
    array_info,
    count_array_spec,
    get_compressors,
    normalize_chunks,
)
from scarf.storage.profiles import StorageProfile
from scarf.storage.sharding import write_counts_t, write_dense_in_shard_rows
from scarf.storage.stores import open_store
from scarf.storage.types import as_zarr_array, as_zarr_group


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


def _locations_overlap(first: str, second: str) -> bool:
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


def _count_assays(store: zarr.Group) -> list[tuple[str, str | None]]:
    assays: list[tuple[str, str | None]] = []
    for name in store.group_keys():
        group = as_zarr_group(store[name], name=name)
        if group.attrs.get("is_assay") and "counts" in group:
            assays.append((name, None))

    if "matrices" not in store:
        return assays
    matrices = as_zarr_group(store["matrices"], name="matrices")
    workspace_assays: set[str] = set()

    def visit(group: zarr.Group, path: str) -> None:
        for name in group.group_keys():
            if path == "" and name == "matrices":
                continue
            child_path = f"{path}/{name}" if path else name
            child = as_zarr_group(group[name], name=child_path)
            if child.attrs.get("is_assay"):
                if path == "" and "counts" in child:
                    continue
                if (
                    name not in workspace_assays
                    and name in matrices
                    and "counts" in as_zarr_group(matrices[name], name=name)
                ):
                    workspace_assays.add(name)
                    assays.append((name, path or "."))
                continue
            visit(child, child_path)

    visit(store, "")
    return assays


def _copy_group(
    src: zarr.Group,
    dst: zarr.Group,
    profile: StorageProfile,
    *,
    path: str = "",
    shardedCounts: frozenset[str] = frozenset(),
) -> None:
    for key in src.keys():
        node = src[key]
        child_path = f"{path}/{key}" if path else key
        if isinstance(node, zarr.Group):
            new_group = dst.create_group(key, overwrite=True)
            for attr_key, attr_val in node.attrs.items():
                new_group.attrs[attr_key] = attr_val
            _copy_group(
                node,
                new_group,
                profile,
                path=child_path,
                shardedCounts=shardedCounts,
            )
            continue
        if child_path in shardedCounts:
            spec = count_array_spec(
                int(node.shape[0]),
                int(node.shape[1]),
                dtype=node.dtype,
                profile=profile,
            )
            dst_array = create_numeric_array(dst, key, spec)
            write_dense_in_shard_rows(
                dst_array,
                lambda start, end: np.asarray(node[start:end, :]),
                msg=f"Repacking {child_path}",
            )
            dst.attrs["scarf:zarr_spec"] = {
                "profile": profile,
                "dtype": np.dtype(node.dtype).str,
                "chunks": list(dst_array.chunks),
                "shards": None if spec.shards is None else list(spec.shards),
                "zarr_format": 3,
            }
        else:
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
    if _locations_overlap(input_path, output_path):
        raise ValueError(
            "input_path and output_path must refer to different stores "
            "and must not overlap"
        )

    src = open_store(input_path, mode="r", storage_options=storage_options)
    dst = open_store(output_path, mode="w", storage_options=storage_options)
    assays = _count_assays(src)
    count_paths = frozenset(
        f"{assay_name}/counts" if workspace is None else f"matrices/{assay_name}/counts"
        for assay_name, workspace in assays
    )
    _copy_group(
        src,
        dst,
        profile,
        shardedCounts=count_paths if shard_counts else frozenset(),
    )
    if not shard_counts:
        return
    for assay_name, workspace in assays:
        counts_path = (
            f"{assay_name}/counts"
            if workspace is None
            else f"matrices/{assay_name}/counts"
        )
        group_path = assay_name if workspace is None else f"matrices/{assay_name}"
        counts = as_zarr_array(dst[counts_path], name=counts_path)
        write_counts_t(
            counts,
            as_zarr_group(dst[group_path], name=group_path),
            profile=profile,
        )
        print(f"  {counts_path}: {array_info(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack Zarr v2 stores to v3 with sharding"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        choices=["fast_local", "cloud"],
        default="fast_local",
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
