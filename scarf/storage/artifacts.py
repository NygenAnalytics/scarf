import hashlib
import inspect
import json
import math
import secrets
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from ..utils.logging import logger
from .arrays import _decode_metadata_values, text_dtype
from .errors import ArtifactErrorContextValue, ArtifactResolutionError
from .geometry import array_geometry
from .operation_revisions import (
    OperationRevision,
    applicable_revisions,
    effective_revision,
)
from .partition import scan_band
from .refs import (
    ARTIFACT_KINDS as ARTIFACT_KINDS,
    ArtifactRef as ArtifactRef,
    ArtifactScope as ArtifactScope,
    ExternalArtifactRef as ExternalArtifactRef,
    _validate_artifact_kind,
    _validate_name,
    artifact_path as artifact_path,
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
        # A byte view of the contiguous block hashes without copying it.
        self._digest.update(array.view(np.uint8))
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


def _stored_array_chunk_rows(array: Any) -> int:
    return scan_band(array_geometry(array), fallback=max(1, int(array.shape[0])))


def fingerprint_stored_arrays(
    group: zarr.Group,
    names: Sequence[str],
    *,
    arrays: Mapping[str, zarr.Array] | None = None,
) -> str:
    """Fingerprint stored arrays; pass ``arrays`` already opened to skip reopening."""
    builder = ValueFingerprintBuilder()
    for name in names:
        array = as_zarr_array(
            group[name] if arrays is None else arrays[name], name=name
        )
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


def fingerprint_text_blocks(
    n_values: int,
    source_dtype: Any,
    blocks: Callable[[], Iterable[Any]],
) -> str:
    """Fingerprint consecutive blocks of values as decoded fixed-width text.

    ``blocks`` returns a fresh iterator on each call; object and variable-width
    values are read twice, once to measure the text width.
    """
    dtype = text_dtype(
        source_dtype,
        lambda: (_decode_metadata_values(block) for block in blocks()),
    )
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (n_values,), dtype)
    offset = 0
    for block in blocks():
        # Values that already hold the text dtype are hashed as read, so a
        # band of row identifiers is held once.
        values = np.asarray(_decode_metadata_values(block)).astype(dtype, copy=False)
        del block
        if values.size:
            builder.update_array_block("values", (offset,), values)
            offset += len(values)
    builder.end_array("values")
    return builder.hexdigest()


def fingerprint_stored_strings(array: Any) -> str:
    """Fingerprint a stored or in-memory string column in bounded bands."""
    if array.ndim != 1:
        raise ValueError("Stored string fingerprints require a one-dimensional array")
    rows = _stored_array_chunk_rows(array)
    n_rows = int(array.shape[0])
    return fingerprint_text_blocks(
        n_rows,
        array.dtype,
        lambda: (array[start : start + rows] for start in range(0, n_rows, rows)),
    )


def fingerprint_strings(values: np.ndarray) -> str:
    """Fingerprint in-memory values as decoded text, keeping their shape."""
    array = np.asarray(values)
    decoded = _decode_metadata_values(array)
    return fingerprint_array(
        decoded.astype(text_dtype(array.dtype, lambda: (decoded,)))
    )


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


def _validate_revision(revision: Any) -> int:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError(f"revision must be a positive integer, got {revision!r}")
    return revision


def make_provenance(
    *,
    operation: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    revision: int = 1,
) -> dict[str, Any]:
    """Return canonical provenance: operation, parameters, inputs, and revision."""
    _validate_name(operation, "operation")
    _validate_revision(revision)
    provenance = {
        "operation": operation,
        "parameters": serialize_artifact_value(parameters),
        "inputs": serialize_artifact_value(inputs),
    }
    if revision > 1:
        provenance["revision"] = revision
    canonical_bytes(provenance)
    return provenance


def provenance_hash(provenance: Mapping[str, Any]) -> str:
    digest = hashlib.blake2b(digest_size=32, person=b"scarf-provenance")
    digest.update(canonical_bytes(provenance))
    return digest.hexdigest()


def parse_artifact_ref(
    raw: Any,
    name: str,
    *,
    owner: ArtifactRef | None = None,
) -> ArtifactRef:
    """Parse a persisted reference that holds exactly the ``ArtifactRef`` fields.

    Args:
        raw: The recorded value.
        name: What the value is, such as the input name an artifact records.
        owner: The artifact that records the value, when there is one.

    Raises:
        ArtifactResolutionError: With code ``corrupt_payload`` when ``raw`` is
            missing or is not an exact artifact reference.
    """
    context: dict[str, ArtifactErrorContextValue] = {"input_name": name}
    if owner is None:
        missing = f"{name} is missing"
        malformed = f"{name} is not a valid artifact reference"
    else:
        context.update(
            {
                "assay": owner.assay,
                "artifact_id": owner.artifact_id,
                "actual_kind": owner.kind,
            }
        )
        missing = f"{owner.kind} artifact has no {name!r} input"
        malformed = f"{owner.kind} artifact has a malformed {name!r} input"
    if raw is None:
        raise ArtifactResolutionError(missing, code="corrupt_payload", context=context)
    try:
        return ArtifactRef.from_dict(raw)
    except (TypeError, ValueError) as error:
        raise ArtifactResolutionError(
            malformed,
            code="corrupt_payload",
            context=context,
        ) from error


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

    @property
    def revision(self) -> int | None:
        """The operation revision the artifact records, or None if unknown."""
        if self.provenance is None:
            return None
        if "revision" not in self.provenance:
            return 1
        value = self.provenance["revision"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            return None
        return value

    @property
    def current_revision(self) -> int | None:
        """The revision this Scarf release records for the same provenance."""
        operation, parameters, inputs = self.operation, self.parameters, self.inputs
        if operation is None or parameters is None or inputs is None:
            return None
        return effective_revision(operation, self.ref.kind, parameters, inputs)

    @property
    def is_current(self) -> bool:
        """Whether the artifact records the revision this release records.

        Only the artifact's own revision is judged, never its inputs.
        """
        revision = self.revision
        return revision is not None and revision == self.current_revision

    @property
    def superseded_by(self) -> tuple[OperationRevision, ...]:
        """Released revisions newer than the recorded one that apply, oldest first."""
        revision = self.revision
        operation, parameters, inputs = self.operation, self.parameters, self.inputs
        if (
            revision is None
            or operation is None
            or parameters is None
            or inputs is None
        ):
            return ()
        return tuple(
            entry
            for entry in applicable_revisions(
                operation, self.ref.kind, parameters, inputs
            )
            if entry.revision > revision
        )

    def input_ref(self, name: str) -> ArtifactRef:
        """Return the exact artifact reference recorded as input ``name``.

        Raises:
            ArtifactResolutionError: With code ``corrupt_payload`` when the
                input is missing or is not an exact artifact reference.
        """
        return parse_artifact_ref((self.inputs or {}).get(name), name, owner=self.ref)


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
    return open_artifact(root, ref)[0]


def open_artifact(
    root: zarr.Group, ref: ArtifactRef
) -> tuple[ArtifactStatus, zarr.Group | None]:
    """Inspect an artifact and return its group from the same metadata read."""
    path = artifact_path(ref)
    try:
        group = group_at(root, path)
    except KeyError:
        return ArtifactStatus(ref=ref, path=path, exists=False, complete=False), None
    return _artifact_status(group, ref, path), group


def _artifact_status(group: zarr.Group, ref: ArtifactRef, path: str) -> ArtifactStatus:
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


def list_artifacts(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    assay: str | None = None,
    kind: str | None = None,
    complete_only: bool = False,
    operation: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> list[ArtifactRef]:
    if operation is not None:
        _validate_name(operation, "operation")
    requested_parameters = (
        None if parameters is None else serialize_artifact_value(parameters)
    )
    requested_inputs = None if inputs is None else serialize_artifact_value(inputs)
    if requested_parameters is not None and not isinstance(
        requested_parameters, Mapping
    ):
        raise TypeError("parameters must be a mapping")
    if requested_inputs is not None and not isinstance(requested_inputs, Mapping):
        raise TypeError("inputs must be a mapping")
    filter_provenance = any(
        value is not None for value in (operation, parameters, inputs)
    )

    def matches_mapping(
        stored: Mapping[str, Any] | None,
        requested: Mapping[str, Any] | None,
    ) -> bool:
        if requested is None:
            return True
        stored = {} if stored is None else stored
        if not requested:
            return not stored
        return all(
            key in stored
            and canonical_bytes(stored[key]) == canonical_bytes(requested_value)
            for key, requested_value in requested.items()
        )

    refs = []
    for ref, group in _artifact_groups(root, scope=scope, assay=assay, kind=kind):
        if complete_only or filter_provenance:
            status = _artifact_status(group, ref, artifact_path(ref))
            if not status.complete:
                continue
            if operation is not None and status.operation != operation:
                continue
            if not matches_mapping(status.parameters, requested_parameters):
                continue
            if not matches_mapping(status.inputs, requested_inputs):
                continue
        refs.append(ref)
    return refs


def _artifact_groups(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    assay: str | None,
    kind: str | None,
) -> list[tuple[ArtifactRef, zarr.Group]]:
    """Return every artifact group in a scope, sorted by kind and ID.

    ``groups()`` reads every child's metadata concurrently, so callers inspect
    artifact attributes without reopening each group.
    """
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
    try:
        base = group_at(root, base_path)
        kind_groups = (
            sorted(dict(base.groups()).items())
            if kind is None
            else [(kind, group_at(base, kind))]
        )
    except KeyError:
        return []
    found = []
    for artifact_kind, kind_group in kind_groups:
        _validate_artifact_kind(artifact_kind)
        # Object-store listings can repeat a group, so keep one entry per ID.
        for artifact_id, group in sorted(dict(kind_group.groups()).items()):
            try:
                ref = ArtifactRef(
                    scope=scope,
                    assay=assay,
                    kind=artifact_kind,
                    artifact_id=artifact_id,
                )
            except ValueError:
                continue
            found.append((ref, group))
    return found


def _without_revision(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key != "revision"}


def reusable_artifact_groups(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    provenance: Mapping[str, Any],
    assay: str | None = None,
    invalidate_cache: bool = False,
) -> list[tuple[ArtifactRef, zarr.Group]]:
    """Complete artifacts with exactly this provenance, newest first, with groups.

    Without one, the newest artifact that differs only in its revision is logged.
    """
    if invalidate_cache:
        return []
    requested = make_provenance(
        operation=str(provenance["operation"]),
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
        revision=provenance.get("revision", 1),
    )
    revision = requested.pop("revision", 1)
    requested_hash = provenance_hash(requested)
    requested_bytes = canonical_bytes(requested)
    reusable: list[tuple[int, ArtifactRef, zarr.Group]] = []
    superseded: list[tuple[int, ArtifactRef, ArtifactStatus]] = []
    for ref, group in _artifact_groups(root, scope=scope, assay=assay, kind=kind):
        try:
            status = _artifact_status(group, ref, artifact_path(ref))
        except (KeyError, TypeError, ValueError):
            continue
        if not status.complete or status.provenance is None:
            continue
        stored = _without_revision(status.provenance)
        if provenance_hash(stored) != requested_hash:
            continue
        if canonical_bytes(stored) != requested_bytes:
            continue
        if status.revision == revision:
            reusable.append((status.created_at_ns or 0, ref, group))
        else:
            superseded.append((status.created_at_ns or 0, ref, status))
    reusable.sort(
        key=lambda item: (item[0], item[1].artifact_id),
        reverse=True,
    )
    if superseded and not reusable:
        _, ref, status = max(
            superseded, key=lambda item: (item[0], item[1].artifact_id)
        )
        changes = "; ".join(entry.change for entry in status.superseded_by)
        logger.info(
            f"Recomputing {requested['operation']}: artifact {ref.artifact_id[:12]} "
            f"is revision {status.revision}, current {revision}"
            + (f": {changes}" if changes else "")
        )
    return [(ref, group) for _, ref, group in reusable]
