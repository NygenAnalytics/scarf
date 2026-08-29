import hashlib
import inspect
import json
import math
import secrets
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .arrays import _decode_metadata_values
from .geometry import array_geometry
from .partition import row_band
from .refs import (
    ARTIFACT_KINDS as ARTIFACT_KINDS,
    ArtifactRef as ArtifactRef,
    ArtifactScope as ArtifactScope,
    ExternalArtifactRef as ExternalArtifactRef,
    _validate_artifact_kind,
    _validate_name,
    artifact_path as artifact_path,
    parse_artifact_path as parse_artifact_path,
)
from .types import as_zarr_array, as_zarr_group


def new_artifact_id() -> str:
    return secrets.token_hex(32)


def group_at(root: zarr.Group, path: str) -> zarr.Group:
    return as_zarr_group(root[path], name=path)


def artifact_group(root: zarr.Group, ref: ArtifactRef) -> zarr.Group:
    return group_at(root, artifact_path(ref))


def _canonical_node(value: Any) -> Any:
    if isinstance(value, ArtifactRef | ExternalArtifactRef):
        return _canonical_node(value.to_dict())
    if isinstance(value, np.generic):
        return _canonical_node(value.item())
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Provenance cannot contain non-finite floats")
        return ["float", struct.pack(">d", value).hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Path):
        return _canonical_node(str(value))
    if isinstance(value, np.ndarray):
        raise TypeError("Arrays must be represented by a value_fingerprint")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Provenance mappings require string keys")
        return [
            "mapping",
            [[key, _canonical_node(value[key])] for key in sorted(value)],
        ]
    if isinstance(value, set | frozenset):
        items = [_canonical_node(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, separators=(",", ":")))
        return ["set", items]
    if isinstance(value, Sequence):
        return ["sequence", [_canonical_node(item) for item in value]]
    raise TypeError(f"Unsupported provenance value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_node(value),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return np.dtype((_canonical_dtype(base), shape))
    if dtype.fields is not None:
        assert dtype.names is not None
        return np.dtype(
            [(name, _canonical_dtype(dtype.fields[name][0])) for name in dtype.names],
            align=False,
        )
    return dtype.newbyteorder("<")


def _dtype_descriptor(dtype: np.dtype[Any]) -> dict[str, Any]:
    normalized = _canonical_dtype(dtype)
    if normalized.fields is not None:
        assert normalized.names is not None
        return {
            "fields": [
                [name, _dtype_descriptor(normalized.fields[name][0])]
                for name in normalized.names
            ]
        }
    if normalized.subdtype is not None:
        base, shape = normalized.subdtype
        return {
            "base": _dtype_descriptor(base),
            "shape": list(shape),
        }
    return {"str": normalized.str}


def _canonical_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("Object arrays require an explicit stable encoding")
    dtype = _canonical_dtype(array.dtype)
    if array.dtype.fields is None:
        return np.ascontiguousarray(array.astype(dtype, copy=False))
    packed = np.empty(array.shape, dtype=dtype)
    assert array.dtype.names is not None
    for name in array.dtype.names:
        packed[name] = array[name]
    return np.ascontiguousarray(packed)


class ValueFingerprintBuilder:
    def __init__(self) -> None:
        self._digest = hashlib.blake2b(digest_size=32, person=b"scarf-values")
        self._active_array: tuple[str, np.dtype[Any], tuple[int, ...], int] | None = (
            None
        )

    def update_bytes(self, name: str, payload: bytes) -> None:
        if self._active_array is not None:
            raise RuntimeError("Finish the active array before adding another value")
        encoded_name = name.encode("utf-8")
        self._digest.update(len(encoded_name).to_bytes(8, "big"))
        self._digest.update(encoded_name)
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def update_array(self, name: str, values: np.ndarray) -> None:
        array = np.asarray(values)
        self.begin_array(name, array.shape, array.dtype)
        if array.shape[0]:
            self.update_array_block(name, (0,) * array.ndim, array)
        self.end_array(name)

    def begin_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | str,
    ) -> None:
        if self._active_array is not None:
            raise RuntimeError("Finish the active array before starting another")
        normalized_dtype = np.dtype(dtype)
        if normalized_dtype.hasobject:
            raise TypeError("Object arrays require an explicit stable encoding")
        normalized_dtype = _canonical_dtype(normalized_dtype)
        normalized_shape = tuple(int(size) for size in shape)
        if not normalized_shape or any(size < 0 for size in normalized_shape):
            raise ValueError("Array shape must contain non-negative dimensions")
        self.update_bytes(
            f"{name}:metadata",
            canonical_bytes(
                {
                    "dtype": _dtype_descriptor(normalized_dtype),
                    "shape": list(normalized_shape),
                }
            ),
        )
        self._active_array = (name, normalized_dtype, normalized_shape, 0)

    def update_array_block(
        self,
        name: str,
        offset: tuple[int, ...],
        values: np.ndarray,
    ) -> None:
        if self._active_array is None:
            raise RuntimeError("begin_array must be called before writing blocks")
        active_name, dtype, shape, next_row = self._active_array
        if name != active_name:
            raise ValueError(f"Expected block for {active_name!r}, got {name!r}")
        raw_array = np.asarray(values)
        if _canonical_dtype(raw_array.dtype) != dtype:
            raise TypeError(f"Expected dtype {dtype}, got {raw_array.dtype}")
        array = _canonical_array(raw_array)
        expected_offset = (next_row,) + (0,) * (len(shape) - 1)
        if offset != expected_offset:
            raise ValueError(f"Expected block offset {expected_offset}, got {offset}")
        if array.ndim != len(shape) or array.shape[1:] != shape[1:]:
            raise ValueError(
                f"Block shape {array.shape} is incompatible with array shape {shape}"
            )
        stop = next_row + array.shape[0]
        if stop > shape[0]:
            raise ValueError("Array block exceeds declared shape")
        self._digest.update(array.view(np.uint8).tobytes())
        self._active_array = (active_name, dtype, shape, stop)

    def end_array(self, name: str) -> None:
        if self._active_array is None:
            raise RuntimeError("No active array to finish")
        active_name, _dtype, shape, next_row = self._active_array
        if name != active_name:
            raise ValueError(f"Expected to finish {active_name!r}, got {name!r}")
        if next_row != shape[0]:
            raise ValueError(
                f"Array {name!r} is incomplete: wrote {next_row} of {shape[0]} rows"
            )
        self._active_array = None

    def hexdigest(self) -> str:
        if self._active_array is not None:
            raise RuntimeError("Cannot finish fingerprint while an array is incomplete")
        return self._digest.hexdigest()


def fingerprint_array(values: np.ndarray) -> str:
    builder = ValueFingerprintBuilder()
    builder.update_array("values", values)
    return builder.hexdigest()


def _stored_array_chunk_rows(array: zarr.Array) -> int:
    return row_band(array_geometry(array), unit="chunk", fallback=1)


def fingerprint_stored_arrays(
    group: zarr.Group,
    names: Sequence[str],
) -> str:
    builder = ValueFingerprintBuilder()
    for name in names:
        array = as_zarr_array(group[name], name=name)
        builder.begin_array(name, array.shape, array.dtype)
        chunk_rows = _stored_array_chunk_rows(array)
        for start in range(0, array.shape[0], chunk_rows):
            stop = min(start + chunk_rows, array.shape[0])
            block = np.asarray(array[start:stop])
            builder.update_array_block(
                name,
                (start,) + (0,) * (array.ndim - 1),
                block,
            )
        builder.end_array(name)
    return builder.hexdigest()


def fingerprint_stored_strings(array: zarr.Array) -> str:
    """Fingerprint a stored string column without loading it in full."""
    if array.ndim != 1:
        raise ValueError("Stored string fingerprints require a one-dimensional array")

    chunk_rows = _stored_array_chunk_rows(array)
    source_dtype = np.dtype(array.dtype)
    if source_dtype.hasobject:
        max_length = 1
        for start in range(0, array.shape[0], chunk_rows):
            stop = min(start + chunk_rows, array.shape[0])
            values = _decode_metadata_values(array[start:stop])
            if values.size:
                max_length = max(
                    max_length,
                    max(len(str(value)) for value in values),
                )
        string_dtype = np.dtype(f"U{max_length}")
    else:
        string_dtype = np.empty(0, dtype=source_dtype).astype(str).dtype

    builder = ValueFingerprintBuilder()
    builder.begin_array("values", array.shape, string_dtype)
    for start in range(0, array.shape[0], chunk_rows):
        stop = min(start + chunk_rows, array.shape[0])
        block = np.asarray(
            _decode_metadata_values(array[start:stop]),
        ).astype(string_dtype)
        builder.update_array_block("values", (start,), block)
    builder.end_array("values")
    return builder.hexdigest()


def fingerprint_strings(values: np.ndarray) -> str:
    strings = np.asarray(values).astype(str)
    return fingerprint_array(strings)


def fingerprint_string_blocks(
    blocks: Iterable[tuple[int, np.ndarray]],
    *,
    length: int,
    max_length: int,
) -> str:
    """Fingerprint ordered strings without collecting the full column."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    dtype = np.dtype(f"U{max_length}")
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (length,), dtype)
    next_row = 0
    for start, raw_values in blocks:
        if int(start) != next_row:
            raise ValueError(
                f"String blocks must be contiguous; expected {next_row}, "
                f"received {start}"
            )
        values = np.asarray(raw_values).astype(dtype)
        if values.ndim != 1:
            raise ValueError("String blocks must be one-dimensional")
        builder.update_array_block("values", (next_row,), values)
        next_row += len(values)
    builder.end_array("values")
    return builder.hexdigest()


def callable_identity(value: Any) -> dict[str, str]:
    explicit = getattr(value, "artifact_identity", None)
    if explicit is not None:
        return {"identity": str(explicit)}
    qualname = str(getattr(value, "__qualname__", type(value).__qualname__))
    if (
        (not inspect.isfunction(value) and not inspect.isbuiltin(value))
        or "<locals>" in qualname
        or "<lambda>" in qualname
        or bool(getattr(value, "__closure__", None))
    ):
        raise ValueError("Dynamic or stateful callables must define artifact_identity")
    return {
        "module": str(getattr(value, "__module__", type(value).__module__)),
        "qualname": qualname,
    }


def serialize_artifact_value(value: Any) -> Any:
    if isinstance(value, ArtifactRef | ExternalArtifactRef):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return {"value_fingerprint": fingerprint_array(value)}
    if isinstance(value, np.generic):
        return serialize_artifact_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"special_float": "nan"}
        return {"special_float": "inf" if value > 0 else "-inf"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if callable(value):
        return {"external_hook": True, **callable_identity(value)}
    if isinstance(value, Mapping):
        return {str(key): serialize_artifact_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [serialize_artifact_value(item) for item in value]
    if isinstance(value, set | frozenset):
        values = [serialize_artifact_value(item) for item in value]
        return sorted(values, key=canonical_bytes)
    return value


def make_provenance(
    *,
    operation: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_name(operation, "operation")
    provenance = {
        "operation": operation,
        "parameters": serialize_artifact_value(parameters),
        "inputs": serialize_artifact_value(inputs),
    }
    canonical_bytes(provenance)
    return provenance


def provenance_hash(provenance: Mapping[str, Any]) -> str:
    digest = hashlib.blake2b(digest_size=32, person=b"scarf-provenance")
    digest.update(canonical_bytes(provenance))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    ref: ArtifactRef
    path: str
    exists: bool
    complete: bool
    provenance: dict[str, Any] | None = None
    execution_options: dict[str, Any] | None = None
    created_at_ns: int | None = None
    scarf_version: str | None = None

    @property
    def operation(self) -> str | None:
        if self.provenance is None:
            return None
        value = self.provenance.get("operation")
        return value if isinstance(value, str) else None

    @property
    def parameters(self) -> dict[str, Any] | None:
        if self.provenance is None:
            return None
        value = self.provenance.get("parameters")
        return dict(value) if isinstance(value, Mapping) else None

    @property
    def inputs(self) -> dict[str, Any] | None:
        if self.provenance is None:
            return None
        value = self.provenance.get("inputs")
        return dict(value) if isinstance(value, Mapping) else None


def _mapping_attr(group: zarr.Group, name: str) -> dict[str, Any] | None:
    if name not in group.attrs:
        return None
    value = group.attrs[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"Artifact attr {name!r} must be a mapping")
    return dict(value)


def require_complete_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
) -> ArtifactStatus:
    status = inspect_artifact(root, ref)
    if not status.exists:
        raise KeyError(f"Artifact does not exist: {status.path}")
    if not status.complete:
        raise RuntimeError(f"Artifact is incomplete: {status.path}")
    return status


def inspect_artifact(root: zarr.Group, ref: ArtifactRef) -> ArtifactStatus:
    path = artifact_path(ref)
    if path not in root:
        return ArtifactStatus(ref=ref, path=path, exists=False, complete=False)
    group = group_at(root, path)
    stored_id = group.attrs.get("artifact_id")
    stored_kind = group.attrs.get("kind")
    if stored_id is not None and stored_id != ref.artifact_id:
        raise ValueError(f"Artifact at {path} has mismatched artifact_id")
    if stored_kind is not None and stored_kind != ref.kind:
        raise ValueError(f"Artifact at {path} has mismatched kind")
    raw_complete = group.attrs.get("complete", False)
    if not isinstance(raw_complete, bool):
        raise TypeError(f"Artifact complete attr at {path} must be boolean")
    complete = raw_complete
    if complete:
        required = {
            "artifact_id",
            "kind",
            "provenance",
            "execution_options",
            "complete",
        }
        missing = required - set(group.attrs)
        if missing:
            raise KeyError(
                f"Completed artifact at {path} is missing attrs: "
                f"{', '.join(sorted(missing))}"
            )
    provenance = _mapping_attr(group, "provenance")
    execution_options = _mapping_attr(group, "execution_options")
    raw_created_at_ns = group.attrs.get("created_at_ns")
    if raw_created_at_ns is not None and (
        isinstance(raw_created_at_ns, bool)
        or not isinstance(raw_created_at_ns, int | np.integer)
        or int(raw_created_at_ns) <= 0
    ):
        raise TypeError(f"Artifact created_at_ns at {path} must be a positive integer")
    created_at_ns = None if raw_created_at_ns is None else int(raw_created_at_ns)
    raw_scarf_version = group.attrs.get("scarf_version")
    if raw_scarf_version is not None and (
        not isinstance(raw_scarf_version, str) or not raw_scarf_version
    ):
        raise TypeError(f"Artifact scarf_version at {path} must be a non-empty string")
    if complete:
        if provenance is None or execution_options is None:
            raise KeyError(f"Completed artifact at {path} has an incomplete record")
        operation = provenance.get("operation")
        parameters = provenance.get("parameters")
        inputs = provenance.get("inputs")
        if (
            not isinstance(operation, str)
            or not isinstance(parameters, Mapping)
            or not isinstance(inputs, Mapping)
        ):
            raise TypeError(f"Artifact provenance at {path} is malformed")
        _validate_name(operation, "operation")
        canonical_bytes(provenance)
    return ArtifactStatus(
        ref=ref,
        path=path,
        exists=True,
        complete=complete,
        provenance=provenance,
        execution_options=execution_options,
        created_at_ns=created_at_ns,
        scarf_version=raw_scarf_version,
    )


def artifact_exists(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    require_complete: bool = True,
) -> bool:
    status = inspect_artifact(root, ref)
    return status.exists and (status.complete or not require_complete)


def list_artifacts(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    assay: str | None = None,
    kind: str | None = None,
    complete_only: bool = False,
) -> list[ArtifactRef]:
    if scope not in {"assay", "datastore"}:
        raise ValueError(f"Invalid artifact scope: {scope!r}")
    if scope == "assay":
        if assay is None or not assay or "/" in assay:
            raise ValueError("assay is required for assay-scoped artifact listing")
        base_path = f"{assay}/artifacts"
    else:
        if assay is not None:
            raise ValueError("assay cannot be set for datastore-scoped listing")
        base_path = "artifacts"
    if kind is not None:
        _validate_artifact_kind(kind)
    if base_path not in root:
        return []
    base = as_zarr_group(root[base_path], name=base_path)
    kinds = [kind] if kind is not None else sorted(base.group_keys())
    refs = []
    for artifact_kind in kinds:
        if artifact_kind not in base:
            continue
        _validate_artifact_kind(artifact_kind)
        kind_group = as_zarr_group(base[artifact_kind], name=artifact_kind)
        for artifact_id in sorted(kind_group.group_keys()):
            try:
                ref = ArtifactRef(
                    scope=scope,
                    assay=assay,
                    kind=artifact_kind,
                    artifact_id=artifact_id,
                )
            except ValueError:
                continue
            if complete_only and not artifact_exists(root, ref):
                continue
            refs.append(ref)
    return refs


def find_reusable_artifacts(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    provenance: Mapping[str, Any],
    assay: str | None = None,
    invalidate_cache: bool = False,
) -> list[ArtifactRef]:
    if invalidate_cache:
        return []
    requested = make_provenance(
        operation=str(provenance["operation"]),
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )
    requested_hash = provenance_hash(requested)
    requested_bytes = canonical_bytes(requested)
    reusable: list[tuple[int, ArtifactRef]] = []
    for ref in list_artifacts(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
    ):
        try:
            status = inspect_artifact(root, ref)
        except (KeyError, TypeError, ValueError):
            continue
        if not status.complete or status.provenance is None:
            continue
        if provenance_hash(status.provenance) != requested_hash:
            continue
        if canonical_bytes(status.provenance) == requested_bytes:
            group = group_at(root, status.path)
            raw_created = group.attrs.get("created_at_ns", 0)
            created_at_ns = (
                int(raw_created)
                if not isinstance(raw_created, bool)
                and isinstance(raw_created, (int, np.integer))
                else 0
            )
            reusable.append((created_at_ns, ref))
    reusable.sort(
        key=lambda item: (item[0], item[1].artifact_id),
        reverse=True,
    )
    return [ref for _, ref in reusable]
