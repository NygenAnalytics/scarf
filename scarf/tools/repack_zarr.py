"""Repack a Zarr store to v3, sharding discovered assay counts."""

import argparse
import json
import posixpath
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import zarr

from scarf.storage.arrays import create_numeric_array
from scarf.storage.budget import ResourceBudget, resolve_budget
from scarf.storage.copy import _copy_metadata_array
from scarf.storage.layout import (
    ZarrArraySpec,
    array_info,
    count_array_spec,
    get_compressors,
    normalize_chunks,
)
from scarf.storage.profiles import StorageProfile
from scarf.storage.sharding import write_counts_t, write_dense_in_shard_rows
from scarf.storage.stores import open_store
from scarf.storage.types import array_metadata_shards, as_zarr_array, as_zarr_group


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


def _counts_t_path(counts_path: str) -> str:
    if counts_path == "counts" or counts_path.endswith("/counts"):
        return f"{counts_path[: -len('counts')]}countsT"
    raise ValueError(f"Not a counts path: {counts_path!r}")


def _copy_array_attrs(src: zarr.Array, dst: zarr.Array) -> None:
    for attr_key, attr_val in src.attrs.items():
        dst.attrs[attr_key] = attr_val


def _row_block_producer(source: zarr.Array) -> Callable[[int, int], np.ndarray]:
    def produce(start: int, end: int) -> np.ndarray:
        return np.asarray(source[start:end, :])

    return produce


def _is_string_like(dtype: np.dtype) -> bool:
    return dtype.kind in {"O", "S", "U"} or dtype.hasobject


def _default_fill_value(dtype: np.dtype) -> object:
    if dtype.kind == "b":
        return False
    if dtype.kind in {"i", "u", "f", "c"}:
        return 0
    return 0


def _resolve_fill_value(array: zarr.Array, *, numeric_1d: bool) -> object | None:
    fill_value: object | None = getattr(array, "fill_value", None)
    if fill_value is not None:
        return fill_value
    if numeric_1d:
        return _default_fill_value(np.dtype(array.dtype))
    return None


def _realign_shards(
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if shards is None or len(shards) != len(chunks):
        return None
    aligned: list[int] = []
    for chunk, shard in zip(chunks, shards, strict=True):
        chunk_size = int(chunk)
        shard_size = int(shard)
        if shard_size % chunk_size == 0 and shard_size >= chunk_size:
            aligned.append(shard_size)
        elif shard_size >= chunk_size:
            aligned.append(chunk_size * max(1, shard_size // chunk_size))
        else:
            aligned.append(chunk_size)
    return tuple(aligned)


def _copy_numeric_1d(
    array: zarr.Array,
    dst: zarr.Group,
    key: str,
    profile: StorageProfile,
) -> zarr.Array:
    shape = (int(array.shape[0]),)
    chunks = normalize_chunks(array.chunks, shape)
    spec = ZarrArraySpec(
        shape=shape,
        chunks=chunks,
        dtype=array.dtype,
        compressors=get_compressors(profile, zarrFormat=3),
        shards=None,
        fillValue=_resolve_fill_value(array, numeric_1d=True),
    )
    dst_array = create_numeric_array(dst, key, spec)
    n_rows = shape[0]
    if n_rows == 0:
        return dst_array
    step = int(dst_array.chunks[0])
    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        dst_array[start:stop] = np.asarray(array[start:stop])
    return dst_array


def _copy_numeric_2d(
    array: zarr.Array,
    dst: zarr.Group,
    key: str,
    profile: StorageProfile,
    *,
    resources: ResourceBudget,
    path: str,
) -> zarr.Array:
    shape = (int(array.shape[0]), int(array.shape[1]))
    chunks = normalize_chunks(array.chunks, shape)
    shards = _realign_shards(chunks, array_metadata_shards(array))
    spec = ZarrArraySpec(
        shape=shape,
        chunks=chunks,
        dtype=array.dtype,
        compressors=get_compressors(profile, zarrFormat=3),
        shards=shards,
        fillValue=_resolve_fill_value(array, numeric_1d=False),
    )
    dst_array = create_numeric_array(dst, key, spec)
    write_dense_in_shard_rows(
        dst_array,
        _row_block_producer(array),
        msg=f"Repacking {path}",
        resources=resources,
    )
    return dst_array


def _copy_group(
    src: zarr.Group,
    dst: zarr.Group,
    profile: StorageProfile,
    *,
    resources: ResourceBudget,
    path: str = "",
    shardedCounts: frozenset[str] = frozenset(),
    skipPaths: frozenset[str] = frozenset(),
) -> None:
    for key in src.keys():
        node = src[key]
        child_path = f"{path}/{key}" if path else key
        if child_path in skipPaths:
            continue
        if isinstance(node, zarr.Group):
            new_group = dst.create_group(key, overwrite=True)
            for attr_key, attr_val in node.attrs.items():
                new_group.attrs[attr_key] = attr_val
            _copy_group(
                node,
                new_group,
                profile,
                resources=resources,
                path=child_path,
                shardedCounts=shardedCounts,
                skipPaths=skipPaths,
            )
            continue

        array = as_zarr_array(node, name=child_path)
        if child_path in shardedCounts:
            spec = count_array_spec(
                int(array.shape[0]),
                int(array.shape[1]),
                dtype=array.dtype,
                profile=profile,
            )
            dst_array = create_numeric_array(dst, key, spec)
            write_dense_in_shard_rows(
                dst_array,
                _row_block_producer(array),
                msg=f"Repacking {child_path}",
                resources=resources,
            )
            stored_shards = array_metadata_shards(dst_array)
            dst.attrs["scarf:zarr_spec"] = {
                "profile": profile,
                "dtype": np.dtype(array.dtype).str,
                "chunks": list(dst_array.chunks),
                "shards": None if stored_shards is None else list(stored_shards),
                "zarr_format": 3,
            }
            _copy_array_attrs(array, dst_array)
            continue

        if array.ndim == 1:
            if _is_string_like(np.dtype(array.dtype)):
                _copy_metadata_array(
                    array,
                    dst,
                    key,
                    overwrite=True,
                    profile=profile,
                )
                dst_array = as_zarr_array(dst[key], name=child_path)
            else:
                dst_array = _copy_numeric_1d(array, dst, key, profile)
            _copy_array_attrs(array, dst_array)
            continue

        if array.ndim == 2:
            dst_array = _copy_numeric_2d(
                array,
                dst,
                key,
                profile,
                resources=resources,
                path=child_path,
            )
            _copy_array_attrs(array, dst_array)
            continue

        chunks = normalize_chunks(array.chunks, array.shape)
        dst_array = dst.create_array(
            key,
            data=np.asarray(array[...]),
            chunks=chunks,
            compressors=get_compressors(profile, zarrFormat=3),
            overwrite=True,
        )
        _copy_array_attrs(array, dst_array)


def repack_store(
    input_path: str,
    output_path: str,
    profile: StorageProfile = "fast_local",
    storage_options: dict | None = None,
    mem_budget: int | str | None = None,
    nthreads: int | None = None,
) -> None:
    """Copy a Zarr store to v3 and shard discovered assay count matrices.

    Args:
        input_path: Source Zarr directory or URI.
        output_path: Destination Zarr directory or URI (created or overwritten).
        profile: Storage profile for compressors and count shard sizes.
        storage_options: Backend options for remote stores (for example credentials).
        mem_budget: Memory budget for streaming writers (bytes, size string, or None).
        nthreads: Worker count for streaming writers (or None for auto-detect).
    """
    if _locations_overlap(input_path, output_path):
        raise ValueError(
            "input_path and output_path must refer to different stores "
            "and must not overlap"
        )

    resources = resolve_budget(mem_budget, nthreads)
    src = open_store(input_path, mode="r", storage_options=storage_options)
    dst = open_store(output_path, mode="w", storage_options=storage_options)
    for attr_key, attr_val in src.attrs.items():
        dst.attrs[attr_key] = attr_val

    assays = _count_assays(src)
    count_paths = frozenset(
        f"{assay_name}/counts" if workspace is None else f"matrices/{assay_name}/counts"
        for assay_name, workspace in assays
    )
    skip_paths = frozenset(_counts_t_path(path) for path in count_paths)
    _copy_group(
        src,
        dst,
        profile,
        resources=resources,
        shardedCounts=count_paths,
        skipPaths=skip_paths,
    )
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
            resources=resources,
        )
        print(f"  {counts_path}: {array_info(counts)}")


def _parse_storage_options(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--storage-options must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--storage-options must be a JSON object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack Zarr stores to v3 with sharded assay counts"
    )
    parser.add_argument(
        "input",
        help="Source Zarr path or URI (for example s3://bucket/store.zarr)",
    )
    parser.add_argument(
        "output",
        help="Destination Zarr path or URI",
    )
    parser.add_argument(
        "--profile",
        choices=["fast_local", "cloud"],
        default="fast_local",
    )
    parser.add_argument(
        "--mem-budget",
        default=None,
        help="Memory budget for streaming writers (for example 8G)",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Worker count for streaming writers",
    )
    parser.add_argument(
        "--storage-options",
        default=None,
        help=(
            "JSON object of backend options, for example "
            "'{\"skip_signature\": true}' for public S3/GCS"
        ),
    )
    args = parser.parse_args()
    repack_store(
        args.input,
        args.output,
        profile=args.profile,
        storage_options=_parse_storage_options(args.storage_options),
        mem_budget=args.mem_budget,
        nthreads=args.nthreads,
    )
    print(f"Repacked {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
