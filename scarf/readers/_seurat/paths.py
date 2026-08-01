import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import h5py
import numpy as np

from .errors import MatrixSourceError, ResourceLimitError, UnsafeSidecarError
from .sources import DEFAULT_LIMITS, SourceLimits


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _path_text(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if not isinstance(text, str):
        raise TypeError("sidecar paths must be text paths")
    if "\x00" in text:
        raise UnsafeSidecarError("sidecar path contains a NUL character")
    return text


def _serialized_absolute_path(text: str) -> PurePath | None:
    posix = PurePosixPath(text)
    if posix.is_absolute():
        return posix
    windows = PureWindowsPath(text)
    if windows.is_absolute():
        return windows
    return None


class SidecarPathResolver:
    def __init__(
        self,
        rds_path: str | os.PathLike[str],
        *,
        absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
        | None = None,
        require_exists: bool = True,
    ) -> None:
        rds = Path(_path_text(rds_path)).expanduser().resolve(strict=False)
        self.rdsPath = rds
        self.anchor = rds.parent
        self.requireExists = bool(require_exists)
        remaps: list[tuple[PurePath, Path]] = []
        for source, destination in (absolute_prefix_remaps or {}).items():
            source_text = _path_text(source)
            source_path = _serialized_absolute_path(source_text)
            destination_path = Path(_path_text(destination)).expanduser()
            if source_path is None or not destination_path.is_absolute():
                raise ValueError("absolute path remaps require absolute prefixes")
            destination_root = destination_path.resolve(strict=False)
            remaps.append((source_path, destination_root))
        self.absolutePrefixRemaps = tuple(
            sorted(remaps, key=lambda item: len(item[0].parts), reverse=True)
        )

    def resolve(
        self,
        sidecar_path: str | os.PathLike[str],
        *,
        expect: str = "any",
    ) -> Path:
        text = _path_text(sidecar_path)
        serialized_absolute = _serialized_absolute_path(text)
        if serialized_absolute is not None:
            candidate, root = self._remap_absolute(serialized_absolute)
        else:
            windows_relative = PureWindowsPath(text)
            if windows_relative.drive:
                raise UnsafeSidecarError(
                    f"drive-relative sidecar path {text!r} is rejected"
                )
            raw = (
                Path(*windows_relative.parts)
                if "\\" in text
                else Path(text).expanduser()
            )
            candidate = self.anchor / raw
            root = self.anchor
        resolved = candidate.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
        if not _is_within(resolved, resolved_root):
            raise UnsafeSidecarError(
                f"sidecar path {text!r} escapes allowed root {str(resolved_root)!r}"
            )
        if self.requireExists and not resolved.exists():
            raise FileNotFoundError(resolved)
        if expect == "file" and resolved.exists() and not resolved.is_file():
            raise UnsafeSidecarError(f"sidecar path {resolved} is not a regular file")
        if expect == "directory" and resolved.exists() and not resolved.is_dir():
            raise UnsafeSidecarError(f"sidecar path {resolved} is not a directory")
        if expect not in {"any", "file", "directory"}:
            raise ValueError("expect must be 'any', 'file', or 'directory'")
        return resolved

    def _remap_absolute(self, path: PurePath) -> tuple[Path, Path]:
        for source_prefix, destination_prefix in self.absolutePrefixRemaps:
            if type(path) is not type(source_prefix):
                continue
            if path == source_prefix or path.is_relative_to(source_prefix):
                relative = path.relative_to(source_prefix)
                return (
                    destination_prefix.joinpath(*relative.parts),
                    destination_prefix,
                )
        raise UnsafeSidecarError(
            f"absolute sidecar path {str(path)!r} has no explicit prefix remap"
        )


def resolve_sidecar_path(
    sidecar_path: str | os.PathLike[str],
    rds_path: str | os.PathLike[str],
    *,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None = None,
    require_exists: bool = True,
    expect: str = "any",
) -> Path:
    return SidecarPathResolver(
        rds_path,
        absolute_prefix_remaps=absolute_prefix_remaps,
        require_exists=require_exists,
    ).resolve(sidecar_path, expect=expect)


def require_filesystem_path(value: Any, description: str = "HDF5 source") -> Path:
    if isinstance(value, h5py.File | h5py.Group | h5py.Dataset):
        raise UnsafeSidecarError(
            f"{description} must be a path; live HDF5 handles are rejected"
        )
    if not isinstance(value, str | os.PathLike):
        raise TypeError(f"{description} must be a filesystem path")
    path = Path(_path_text(value)).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise UnsafeSidecarError(f"{description} path {path} is not a regular file")
    return path


def _has_reference_dtype(dtype: np.dtype[Any]) -> bool:
    if dtype.fields is not None:
        return any(_has_reference_dtype(field[0]) for field in dtype.fields.values())
    if dtype.subdtype is not None:
        return _has_reference_dtype(dtype.subdtype[0])
    try:
        return h5py.check_dtype(ref=dtype) is not None
    except TypeError:
        return False


def _validate_attribute_value(
    value: Any,
    object_path: str,
    limits: SourceLimits,
) -> None:
    if isinstance(value, h5py.Reference | h5py.RegionReference):
        raise UnsafeSidecarError(f"HDF5 references are rejected at {object_path}")
    array = np.asarray(value)
    payload_bytes = int(array.nbytes)
    if array.dtype.kind in {"O", "S", "U"}:
        for item in array.reshape(-1):
            if isinstance(item, bytes | np.bytes_):
                payload_bytes += len(bytes(item))
            elif isinstance(item, str | np.str_):
                payload_bytes += len(str(item).encode("utf-8"))
    if payload_bytes > limits.maxMetadataBytes:
        raise ResourceLimitError(
            f"HDF5 attribute {object_path} exceeds "
            f"maxMetadataBytes={limits.maxMetadataBytes}"
        )
    if _has_reference_dtype(array.dtype):
        raise UnsafeSidecarError(f"HDF5 reference dtype is rejected at {object_path}")


def _validate_filters(dataset: h5py.Dataset) -> None:
    property_list = dataset.id.get_create_plist()
    optional_flag = int(getattr(h5py.h5z, "FLAG_OPTIONAL", 1))
    decode_flag = int(getattr(h5py.h5z, "FILTER_CONFIG_DECODE_ENABLED", 2))
    for index in range(property_list.get_nfilters()):
        filter_id, flags, _values, name = property_list.get_filter(index)
        available = bool(h5py.h5z.filter_avail(filter_id))
        decodable = False
        if available:
            try:
                decodable = bool(h5py.h5z.get_filter_info(filter_id) & decode_flag)
            except RuntimeError:
                decodable = False
        if not available or not decodable:
            optional = bool(flags & optional_flag)
            if not optional:
                display = (
                    name.decode("utf-8", errors="replace")
                    if isinstance(name, bytes)
                    else str(name)
                )
                raise UnsafeSidecarError(
                    f"HDF5 dataset {dataset.name!r} requires unavailable "
                    f"filter {filter_id} ({display})"
                )


def _validate_dataset(dataset: h5py.Dataset) -> None:
    if dataset.is_virtual:
        raise UnsafeSidecarError(f"HDF5 virtual dataset {dataset.name!r} is rejected")
    property_list = dataset.id.get_create_plist()
    if property_list.get_external_count():
        raise UnsafeSidecarError(
            f"HDF5 external storage at {dataset.name!r} is rejected"
        )
    if _has_reference_dtype(dataset.dtype):
        raise UnsafeSidecarError(f"HDF5 reference dataset {dataset.name!r} is rejected")
    _validate_filters(dataset)


def _validate_group_links(
    group: h5py.Group,
    visited: set[int],
) -> None:
    object_id = hash(group.id)
    if object_id in visited:
        return
    visited.add(object_id)
    for key in group.keys():
        link = group.get(key, getlink=True)
        link_path = f"{group.name.rstrip('/')}/{key}"
        if isinstance(link, h5py.ExternalLink):
            raise UnsafeSidecarError(f"HDF5 external link at {link_path!r} is rejected")
        if not isinstance(link, h5py.HardLink):
            continue
        node = group.get(key, getlink=False)
        if isinstance(node, h5py.Group):
            _validate_group_links(node, visited)


def validate_hdf5_file(
    path: str | os.PathLike[str] | Any,
    *,
    limits: SourceLimits = DEFAULT_LIMITS,
) -> Path:
    resolved = require_filesystem_path(path)
    try:
        with h5py.File(resolved, mode="r") as handle:
            _validate_group_links(handle, set())
            metadata_bytes = 0

            def inspect_attributes(node: h5py.Group | h5py.Dataset) -> None:
                nonlocal metadata_bytes
                for key in node.attrs:
                    attribute_id = node.attrs.get_id(key)
                    try:
                        storage_size = int(attribute_id.get_storage_size())
                    finally:
                        attribute_id.close()
                    metadata_bytes += storage_size + len(key.encode("utf-8")) + 8
                    if metadata_bytes > limits.maxMetadataBytes:
                        raise ResourceLimitError(
                            "HDF5 attributes exceed "
                            f"maxMetadataBytes={limits.maxMetadataBytes}"
                        )
                    _validate_attribute_value(
                        node.attrs[key],
                        f"{node.name}@{key}",
                        limits,
                    )

            def inspect(name: str, node: h5py.Group | h5py.Dataset) -> None:
                nonlocal metadata_bytes
                metadata_bytes += len(name.encode("utf-8")) + 8
                if metadata_bytes > limits.maxMetadataBytes:
                    raise ResourceLimitError(
                        "HDF5 object metadata exceeds "
                        f"maxMetadataBytes={limits.maxMetadataBytes}"
                    )
                inspect_attributes(node)
                if isinstance(node, h5py.Dataset):
                    _validate_dataset(node)

            handle.visititems(inspect)
            inspect_attributes(handle)
    except OSError as error:
        raise MatrixSourceError(f"cannot open HDF5 sidecar {resolved}") from error
    return resolved


def _decode_hdf5_text(value: Any, object_path: str) -> str:
    if isinstance(value, bytes | np.bytes_):
        try:
            result = bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MatrixSourceError(
                f"HDF5 text at {object_path} is not valid UTF-8"
            ) from error
    elif isinstance(value, str | np.str_):
        result = str(value)
    else:
        raise MatrixSourceError(f"HDF5 metadata at {object_path} must contain strings")
    if "\x00" in result:
        raise MatrixSourceError(f"HDF5 text at {object_path} contains NUL")
    return result


def read_hdf5_names(
    handle: h5py.File,
    dataset_path: str | None,
    expected_length: int,
    *,
    required: bool = False,
    limits: SourceLimits = DEFAULT_LIMITS,
) -> tuple[str, ...] | None:
    if dataset_path is None:
        return None
    normalized_path = "/" + dataset_path.strip("/")
    if normalized_path not in handle:
        if required:
            raise MatrixSourceError(
                f"HDF5 names dataset {normalized_path!r} is missing"
            )
        return None
    node = handle[normalized_path]
    if not isinstance(node, h5py.Dataset):
        raise MatrixSourceError(f"HDF5 names path {normalized_path!r} is not a dataset")
    if node.ndim != 1 or int(node.shape[0]) != expected_length:
        raise MatrixSourceError(
            f"HDF5 names dataset {normalized_path!r} has shape {node.shape}; "
            f"expected ({expected_length},)"
        )
    if _has_reference_dtype(node.dtype):
        raise UnsafeSidecarError(
            f"HDF5 reference names at {normalized_path!r} are rejected"
        )
    if expected_length * 8 > limits.maxMetadataBytes:
        raise ResourceLimitError(
            f"HDF5 names exceed maxMetadataBytes={limits.maxMetadataBytes}"
        )
    string_info = h5py.check_string_dtype(node.dtype)
    if string_info is None:
        raise MatrixSourceError(
            f"HDF5 names dataset {normalized_path!r} must contain strings"
        )
    if string_info.length is not None:
        declared_bytes = expected_length * (string_info.length + 8)
        if declared_bytes > limits.maxMetadataBytes:
            raise ResourceLimitError(
                f"HDF5 names exceed maxMetadataBytes={limits.maxMetadataBytes}"
            )
    values: list[str] = []
    size = 0
    chunk = 4096
    for start in range(0, expected_length, chunk):
        stop = min(expected_length, start + chunk)
        for value in np.asarray(node[start:stop]).reshape(-1):
            decoded = _decode_hdf5_text(value, normalized_path)
            size += len(decoded.encode("utf-8")) + 8
            if size > limits.maxMetadataBytes:
                raise ResourceLimitError(
                    f"HDF5 names exceed maxMetadataBytes={limits.maxMetadataBytes}"
                )
            values.append(decoded)
    return tuple(values)


def read_hdf5_shape(
    value: Any,
    object_path: str,
) -> tuple[int, int]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != 2:
        raise MatrixSourceError(
            f"HDF5 shape at {object_path} must contain two integers"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"HDF5 shape at {object_path} must contain integers")
    shape = (int(array[0]), int(array[1]))
    if shape[0] < 0 or shape[1] < 0:
        raise MatrixSourceError(f"HDF5 shape at {object_path} cannot contain negatives")
    return shape


def require_hdf5_group(
    handle: h5py.File,
    group_path: str,
) -> h5py.Group:
    normalized = "/" + group_path.strip("/")
    if normalized not in handle:
        raise MatrixSourceError(f"HDF5 group {normalized!r} is missing")
    node = handle[normalized]
    if not isinstance(node, h5py.Group):
        raise MatrixSourceError(f"HDF5 path {normalized!r} is not a group")
    return node


def require_hdf5_datasets(
    group: h5py.Group,
    names: Sequence[str],
) -> dict[str, h5py.Dataset]:
    result: dict[str, h5py.Dataset] = {}
    for name in names:
        if name not in group:
            raise MatrixSourceError(
                f"HDF5 dataset {group.name.rstrip('/')}/{name} is missing"
            )
        node = group[name]
        if not isinstance(node, h5py.Dataset):
            raise MatrixSourceError(
                f"HDF5 path {group.name.rstrip('/')}/{name} is not a dataset"
            )
        result[name] = node
    return result
