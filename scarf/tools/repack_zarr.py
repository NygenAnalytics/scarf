"""Repack a Zarr store to v3, sharding discovered assay counts."""

import argparse
import json
from collections.abc import Callable

import numpy as np
import zarr

from scarf.storage.arrays import create_numeric_array
from scarf.storage.budget import ResourceBudget, resolve_budget
from scarf.storage.copy import (
    _copy_metadata_array,
    copy_zarr_group_tree,
    validate_metadata_dependencies,
)
from scarf.storage.identity import (
    GENERATED_FEATURE_COLUMNS,
    CountSummary,
    finalize_counts,
    generated_cell_columns,
    publish_preparation,
    validate_preparation,
)
from scarf.storage.count_matrix import (
    COUNT_MATRIX_LAYOUT_KEY,
    create_product_counts_array,
)
from scarf.storage.layout import (
    ZarrArraySpec,
    array_info,
    get_compressors,
    normalize_chunks,
)
from scarf.storage.pipeline_runs import _copy_pipeline_label_claims
from scarf.storage.profiles import StorageProfile
from scarf.storage.schema import pending_assay_message, pending_assays
from scarf.storage.sharding import write_counts_t, write_dense_in_shard_rows
from scarf.storage.stores import (
    open_store,
    resolve_matrix_source,
    locations_overlap,
    MATRIX_SOURCE_ATTR,
)
from scarf.storage.types import array_metadata_shards, as_zarr_array, as_zarr_group


def _count_assays(store: zarr.Group) -> list[tuple[str, str | None]]:
    assays: list[tuple[str, str | None]] = []
    for name, group in store.groups():
        if group.attrs.get("is_assay") is True:
            assays.append((name, None))
        elif name not in {"matrices", "artifacts", "pipeline"}:
            assays.extend(
                (assay_name, name)
                for assay_name, assay in group.groups()
                if assay.attrs.get("is_assay") is True
            )
    return sorted(assays, key=lambda item: (item[1] or "", item[0]))


def _retired_assay_state_paths(store: zarr.Group) -> frozenset[str]:
    paths: set[str] = set()

    def visit(group: zarr.Group, path: str) -> None:
        for name in group.group_keys():
            child_path = f"{path}/{name}" if path else name
            child = as_zarr_group(group[name], name=child_path)
            if child.attrs.get("is_assay") and "state" in child:
                paths.add(f"{child_path}/state")
            visit(child, child_path)

    visit(store, "")
    return frozenset(paths)


def _copy_array_attrs(
    src: zarr.Array,
    dst: zarr.Array,
    *,
    strip_keys: frozenset[str] = frozenset(),
) -> None:
    for attr_key, attr_val in src.attrs.items():
        if attr_key in strip_keys:
            continue
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
        producerBytes=int(np.prod(array.chunks)) * array.dtype.itemsize,
    )
    return dst_array


def _copy_group(
    src: zarr.Group,
    dst: zarr.Group,
    profile: StorageProfile,
    *,
    resources: ResourceBudget,
    path: str = "",
    keepPaths: frozenset[str] | None = None,
    skipPaths: frozenset[str] = frozenset(),
) -> None:
    for key in src.keys():
        node = src[key]
        child_path = f"{path}/{key}" if path else key
        if child_path in skipPaths:
            continue
        if keepPaths is not None and not any(
            candidate == child_path
            or candidate.startswith(child_path + "/")
            or child_path.startswith(candidate + "/")
            for candidate in keepPaths
        ):
            continue
        if isinstance(node, zarr.Group):
            new_group = dst.create_group(key, overwrite=True)
            is_dataset = (
                node.attrs.get("is_assay") is True
                or f"{child_path}/counts" in skipPaths
            )
            for attr_key, attr_val in node.attrs.items():
                if is_dataset and attr_key in {
                    COUNT_MATRIX_LAYOUT_KEY,
                    "prepared",
                    "dataset_fingerprint",
                    "counts_fingerprint",
                    MATRIX_SOURCE_ATTR,
                }:
                    continue
                if keepPaths is not None and attr_key not in {
                    "is_assay",
                    "size_factor",
                    "defaultAssay",
                    "assayTypes",
                }:
                    continue
                new_group.attrs[attr_key] = attr_val
            if node.attrs.get("is_assay") is True:
                new_group.attrs["prepared"] = False
            if key in {"cellData", "featureData"}:
                excluded = {
                    member
                    for member in node.keys()
                    if f"{child_path}/{member}" in skipPaths
                }
                copy_zarr_group_tree(
                    node, new_group, exclude_members=excluded, profile=profile
                )
                continue
            _copy_group(
                node,
                new_group,
                profile,
                resources=resources,
                path=child_path,
                keepPaths=keepPaths,
                skipPaths=skipPaths,
            )
            continue

        array = as_zarr_array(node, name=child_path)
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
    *,
    data_only: bool = False,
) -> None:
    """Copy prepared data, or rebuild raw data, into a fresh Zarr v3 destination."""
    from ..assay import RNAassay, preset_assay_types
    from ..assay.classification import (
        DEFAULT_PERCENT_PATTERNS,
        is_rna_assay_type,
        lookup_persisted_assay_type,
    )
    from ..metadata import MetaData
    from ..assay.classification import default_feature_sets

    if locations_overlap(input_path, output_path):
        raise ValueError("input_path and output_path must not overlap")
    resources = resolve_budget(mem_budget, nthreads)
    src = open_store(input_path, mode="r", storage_options=storage_options)
    manifest = src.attrs.get(MATRIX_SOURCE_ATTR)
    if manifest is not None:
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("location"), str
        ):
            raise ValueError("Malformed matrix source location")
        if locations_overlap(manifest["location"], output_path):
            raise ValueError("The destination overlaps the mounted count owner")
    resolved = resolve_matrix_source(src, storage_options=storage_options)
    mounted_owner = None if resolved is None else resolved[0]
    pending = pending_assays(src)
    if pending:
        raise ValueError(
            "An interrupted derived assay cannot be repacked. "
            + pending_assay_message(*pending[0])
        )
    assays = _count_assays(src)
    if not assays:
        raise ValueError("No logical assays found in source")

    counts_to_copy: dict[str, zarr.Array] = {}
    feature_tables: dict[str, str] = {}
    required_transposes: set[str] = set()
    source_fingerprints: dict[tuple[str, str | None], str] = {}
    assay_types: dict[tuple[str, str | None], str] = {}
    skip_paths: set[str] = set(_retired_assay_state_paths(src))
    keep_paths: set[str] = set()
    for name, workspace in assays:
        attr_root = (
            src if workspace is None else as_zarr_group(src[workspace], name=workspace)
        )
        if attr_root.attrs.get("scarf:import_complete") is False:
            raise ValueError("An incomplete import cannot be repacked as complete data")
        assay = as_zarr_group(attr_root[name], name=name)
        prefix = "" if workspace is None else f"{workspace}/"
        matrix_path = name if workspace is None else f"matrices/{name}"
        owner = (
            mounted_owner
            if manifest is not None and name in manifest["assays"]
            else src
        )
        assert owner is not None
        matrix = as_zarr_group(owner[matrix_path], name=matrix_path)
        counts = as_zarr_array(matrix["counts"], name="counts")
        if counts.attrs.get("complete") is False:
            raise ValueError(
                "An incomplete count matrix cannot be repacked as complete data"
            )
        path = f"{matrix_path}/counts"
        counts_to_copy[path] = counts
        feature_tables[path] = f"{prefix}{name}"
        skip_paths.update({path, f"{matrix_path}/countsT"})
        raw_types = attr_root.attrs.get("assayTypes", {})
        type_name = lookup_persisted_assay_type(
            name, raw_types if isinstance(raw_types, dict) else None
        )
        assay_types[name, workspace] = type_name
        required = is_rna_assay_type(type_name)
        if required:
            required_transposes.add(path)
        cells = as_zarr_group(attr_root["cellData"], name="cellData")
        features = as_zarr_group(assay["featureData"], name="featureData")
        if not data_only:
            validate_metadata_dependencies(cells)
            validate_metadata_dependencies(features)
            fingerprint = validate_preparation(
                assay, cells, matrix, require_transpose=required
            )
            assert fingerprint is not None
            source_fingerprints[name, workspace] = fingerprint
        else:
            skip_paths.update(
                f"{prefix}cellData/{column}"
                for column in generated_cell_columns(
                    name, assay.attrs.get("percentFeatures")
                )
            )
            skip_paths.update(
                f"{prefix}{name}/featureData/{column}"
                for column in GENERATED_FEATURE_COLUMNS
            )
            keep_paths.update(
                {f"{prefix}cellData", f"{prefix}{name}/featureData", path}
            )

    for name, workspace in assays:
        prefix = "" if workspace is None else f"{workspace}/"
        for table in (f"{prefix}cellData", f"{prefix}{name}/featureData"):
            excluded = {
                path[len(table) + 1 :]
                for path in skip_paths
                if path.startswith(f"{table}/")
            }
            validate_metadata_dependencies(
                as_zarr_group(src[table], name=table), exclude_members=excluded
            )

    dst = open_store(output_path, mode="w-", storage_options=storage_options)
    for key, value in src.attrs.items():
        if key == MATRIX_SOURCE_ATTR or (
            data_only and key not in {"defaultAssay", "assayTypes"}
        ):
            continue
        dst.attrs[key] = value
    _copy_group(
        src,
        dst,
        profile,
        resources=resources,
        skipPaths=frozenset(skip_paths),
        keepPaths=frozenset(keep_paths) if data_only else None,
    )
    for path, source_counts in counts_to_copy.items():
        group_path = path.rsplit("/", 1)[0]
        group = dst.require_group(group_path)
        counts = create_product_counts_array(
            group,
            source_counts.shape[0],
            source_counts.shape[1],
            source_counts.dtype,
            profile=profile,
        )
        summary = CountSummary(counts)
        write_dense_in_shard_rows(
            counts,
            _row_block_producer(source_counts),
            resources=resources,
            msg=f"Repacking {path}",
            producerBytes=int(np.prod(source_counts.chunks))
            * source_counts.dtype.itemsize,
            residentBytes=summary.nbytes,
            countSummary=summary,
        )
        finalize_counts(counts, summary=summary)
        if path in required_transposes:
            write_counts_t(
                counts,
                group,
                resources=resources,
                profile=profile,
                featureSets=default_feature_sets(
                    as_zarr_group(dst[feature_tables[path]], name=feature_tables[path])
                ),
            )
        print(f"  {path}: {array_info(counts)}")
    for name, workspace in assays:
        attr_root = (
            dst if workspace is None else as_zarr_group(dst[workspace], name=workspace)
        )
        assay_group = as_zarr_group(attr_root[name], name=name)
        cells = as_zarr_group(attr_root["cellData"], name="cellData")
        matrix_path = name if workspace is None else f"matrices/{name}"
        matrix = as_zarr_group(dst[matrix_path], name=matrix_path)
        type_name = assay_types[name, workspace]
        raw_types = attr_root.attrs.get("assayTypes", {})
        types = dict(raw_types) if isinstance(raw_types, dict) else {}
        types[name] = type_name
        attr_root.attrs["assayTypes"] = types
        if data_only:
            assay = preset_assay_types()[type_name](
                z=dst,
                workspace=workspace,
                name=name,
                cell_data=MetaData(cells),
                nthreads=resources.workers,
                resources=resources,
            )
            patterns = (
                {
                    f"{name}_{suffix}": pattern
                    for suffix, pattern in DEFAULT_PERCENT_PATTERNS.items()
                }
                if isinstance(assay, RNAassay)
                else {}
            )
            assay.prepare(patterns)
        else:
            publish_preparation(
                assay_group,
                cells,
                matrix,
                require_transpose=is_rna_assay_type(type_name),
                expected_fingerprint=source_fingerprints[name, workspace],
            )
    if not data_only:
        _copy_pipeline_label_claims(src, dst)


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
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Rebuild raw data without saved analyses or generated summaries",
    )
    args = parser.parse_args()
    repack_store(
        args.input,
        args.output,
        profile=args.profile,
        storage_options=_parse_storage_options(args.storage_options),
        mem_budget=args.mem_budget,
        nthreads=args.nthreads,
        data_only=args.data_only,
    )
    print(f"Repacked {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
