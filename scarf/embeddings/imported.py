from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import zarr

from ..graph.state import (
    ArtifactRef,
    ArtifactSelectionError,
    AssayState,
    ImportedArtifactStorage,
    fingerprint_selected_stored_strings,
    read_assay_state,
    validate_cell_selection_artifact,
    validate_imported_coordinates_artifact,
    write_assay_state,
)

type ImportedBlockSource = (
    np.ndarray | zarr.Array | Iterable[np.ndarray] | Callable[[], Iterable[np.ndarray]]
)
type ImportedStringSource = Sequence[str | bytes] | np.ndarray | zarr.Array
type ImportedFeatureIdSource = (
    ImportedStringSource | Iterable[np.ndarray] | Callable[[], Iterable[np.ndarray]]
)


@dataclass(slots=True)
class _ArraySource:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    produce: Callable[[], Iterator[np.ndarray]]
    reusable: bool


def _positive_block_rows(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError("block_rows must be a positive integer")
    if int(value) < 1:
        raise ValueError("block_rows must be greater than zero")
    return int(value)


def _resolve_source(
    values: ImportedBlockSource,
    *,
    name: str,
    shape: tuple[int, ...] | None,
    dtype: Any | None,
    block_rows: int,
) -> _ArraySource:
    if isinstance(values, np.ndarray | zarr.Array):
        resolved_shape = tuple(int(size) for size in values.shape)
        resolved_dtype = np.dtype(values.dtype)

        def produce() -> Iterator[np.ndarray]:
            for start in range(0, resolved_shape[0], block_rows):
                stop = min(start + block_rows, resolved_shape[0])
                yield np.asarray(values[start:stop])

        return _ArraySource(resolved_shape, resolved_dtype, produce, True)

    if shape is None or dtype is None:
        raise ValueError(
            f"{name}_shape and {name}_dtype are required for streamed blocks"
        )
    resolved_shape = tuple(int(size) for size in shape)
    resolved_dtype = np.dtype(dtype)
    if callable(values):

        def produce() -> Iterator[np.ndarray]:
            return iter(values())

        reusable = True
    else:
        used = False

        def produce() -> Iterator[np.ndarray]:
            nonlocal used
            if used:
                raise RuntimeError(f"{name} block source can only be consumed once")
            used = True
            return iter(values)

        reusable = False

    return _ArraySource(resolved_shape, resolved_dtype, produce, reusable)


def _validate_numeric_source(source: _ArraySource, name: str, ndim: int) -> None:
    if len(source.shape) != ndim or any(size < 1 for size in source.shape):
        raise ValueError(f"{name} must have {ndim} non-empty dimensions")
    if source.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating-point dtype")


def _validate_fingerprint(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be a 64-character lowercase hex fingerprint")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a hexadecimal fingerprint") from exc
    return value


def _required_payload_fingerprints(
    values: Mapping[str, str],
    required: set[str],
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("payload_fingerprints must be a mapping")
    fingerprints = {
        str(name): _validate_fingerprint(fingerprint, f"{name} fingerprint")
        for name, fingerprint in values.items()
    }
    missing = required - set(fingerprints)
    if missing:
        raise ValueError("Missing payload fingerprints: " + ", ".join(sorted(missing)))
    unexpected = set(fingerprints) - required
    if unexpected:
        raise ValueError(
            "Unexpected payload fingerprints: " + ", ".join(sorted(unexpected))
        )
    return fingerprints


def _validate_source_digest(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise TypeError("source_digest must contain exactly 32 bytes")
    return value


def _string_source_length(values: ImportedStringSource) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError("source_cell_ids must be one-dimensional")
        return int(shape[0])
    return len(cast(Sequence[str | bytes], values))


def _string_block(
    values: ImportedStringSource,
    start: int,
    stop: int,
    *,
    name: str = "source_cell_ids",
) -> np.ndarray:
    raw = values[start:stop]
    array = np.asarray(raw)
    if array.ndim != 1 or len(array) != stop - start:
        raise ValueError(f"{name} do not support bounded one-dimensional reads")
    normalized: list[str] = []
    for value in array:
        if isinstance(value, bytes | np.bytes_):
            try:
                text = bytes(value).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("source_cell_ids contain invalid UTF-8") from exc
        elif isinstance(value, str | np.str_):
            text = str(value)
        else:
            raise TypeError(f"{name} must contain strings")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{name} contain invalid Unicode") from exc
        if not text or "\x00" in text:
            raise ValueError(f"{name} contain an invalid identifier")
        normalized.append(text)
    return np.asarray(normalized)


def _resolve_feature_id_source(
    values: ImportedFeatureIdSource,
    *,
    shape: tuple[int] | None,
    dtype: Any | None,
    block_rows: int,
) -> _ArraySource:
    if (
        not callable(values)
        and not isinstance(values, np.ndarray | zarr.Array)
        and shape is None
        and dtype is None
        and hasattr(values, "__len__")
        and hasattr(values, "__getitem__")
    ):
        strings = cast(ImportedStringSource, values)
        resolved_shape = (_string_source_length(strings),)

        def produce() -> Iterator[np.ndarray]:
            for start in range(0, resolved_shape[0], block_rows):
                stop = min(start + block_rows, resolved_shape[0])
                yield _string_block(
                    strings,
                    start,
                    stop,
                    name="feature_ids",
                )

        return _ArraySource(resolved_shape, np.dtype(object), produce, True)
    return _resolve_source(
        cast(ImportedBlockSource, values),
        name="feature_ids",
        shape=shape,
        dtype=dtype,
        block_rows=block_rows,
    )


def _selection_alignment(
    root: zarr.Group,
    *,
    cell_selection: ArtifactRef,
    cell_key: str,
    source_cell_ids: ImportedStringSource,
    coordinate_rows: int,
    block_rows: int,
) -> tuple[zarr.Array, int, str]:
    storage = ImportedArtifactStorage(root)
    validate_cell_selection_artifact(root, cell_selection, cell_key)
    selection_group = storage.artifact_group(cell_selection)
    selection = storage.as_array(selection_group["values"], "values")
    cell_data = storage.as_group(root["cellData"], "cellData")
    stored_ids = storage.as_array(cell_data["ids"], "ids")
    source_count = _string_source_length(source_cell_ids)
    context: dict[str, str | int | float | bool | None] = {
        "cell_key": cell_key,
        "coordinate_rows": int(coordinate_rows),
        "source_cell_count": source_count,
        "artifact_id": cell_selection.artifact_id,
    }
    if source_count != coordinate_rows:
        raise ArtifactSelectionError(
            "Imported coordinates do not match the source cell count",
            code="dimreduc_row_count_mismatch",
            context=context,
        )
    source_start = 0
    selected_count = 0
    for start in range(0, int(selection.shape[0]), block_rows):
        stop = min(start + block_rows, int(selection.shape[0]))
        mask = np.asarray(selection[start:stop], dtype=bool)
        selected = int(np.count_nonzero(mask))
        source_stop = source_start + selected
        if source_stop > source_count:
            context["selected_count"] = source_stop
            raise ArtifactSelectionError(
                "Imported coordinates do not match the selected cell count",
                code="dimreduc_row_count_mismatch",
                context=context,
            )
        expected = _string_block(source_cell_ids, source_start, source_stop)
        actual = _string_block(
            np.asarray(stored_ids[start:stop])[mask],
            0,
            selected,
        )
        if not np.array_equal(expected, actual):
            context["selected_count"] = selected_count + selected
            raise ArtifactSelectionError(
                "Imported cell IDs do not match the selected cell order",
                code="dimreduc_cell_identity_mismatch",
                context=context,
            )
        source_start = source_stop
        selected_count += selected
    context["selected_count"] = selected_count
    if selected_count != coordinate_rows or source_start != source_count:
        raise ArtifactSelectionError(
            "Imported coordinates do not match the selected cell count",
            code="dimreduc_row_count_mismatch",
            context=context,
        )
    fingerprint, fingerprint_count = fingerprint_selected_stored_strings(
        stored_ids,
        selection,
    )
    if fingerprint_count != selected_count:
        raise RuntimeError("Selected cell count changed during alignment")
    return selection, selected_count, fingerprint


def _create_numeric_destination(
    storage: ImportedArtifactStorage,
    group: zarr.Group,
    name: str,
    source: _ArraySource,
    block_rows: int,
) -> zarr.Array:
    return storage.create_numeric(
        group,
        name,
        shape=source.shape,
        dtype=source.dtype,
        block_rows=block_rows,
    )


def _write_numeric_payload(
    storage: ImportedArtifactStorage,
    group: zarr.Group,
    name: str,
    source: _ArraySource,
    expected_fingerprint: str,
    block_rows: int,
) -> zarr.Array:
    destination = _create_numeric_destination(
        storage,
        group,
        name,
        source,
        block_rows,
    )
    builder = storage.fingerprint_builder()
    builder.begin_array("values", source.shape, source.dtype)
    start = 0
    blocks = source.produce()
    try:
        for raw_block in blocks:
            block = np.asarray(raw_block)
            if (
                block.ndim != len(source.shape)
                or block.shape[1:] != source.shape[1:]
                or np.dtype(block.dtype) != source.dtype
            ):
                raise ValueError(f"{name} block has an invalid shape or dtype")
            if block.shape[0] == 0:
                continue
            if not np.all(np.isfinite(block)):
                raise ValueError(f"{name} contains non-finite values")
            stop = start + int(block.shape[0])
            if stop > source.shape[0]:
                raise ValueError(f"{name} stream exceeds its declared row count")
            offset = (start,) + (0,) * (block.ndim - 1)
            builder.update_array_block("values", offset, block)
            index = (slice(start, stop),) + (slice(None),) * (block.ndim - 1)
            destination[index] = block
            start = stop
    finally:
        close = getattr(blocks, "close", None)
        if callable(close):
            close()
    if start != source.shape[0]:
        raise ValueError(
            f"{name} stream contains {start} rows, expected {source.shape[0]}"
        )
    builder.end_array("values")
    if builder.hexdigest() != expected_fingerprint:
        raise ValueError(f"{name} payload fingerprint does not match its source")
    return destination


def _write_feature_ids(
    storage: ImportedArtifactStorage,
    group: zarr.Group,
    source: _ArraySource,
    expected_fingerprint: str,
    block_rows: int,
) -> zarr.Array:
    if len(source.shape) != 1:
        raise ValueError("feature_ids must be one-dimensional")
    if source.dtype.hasobject:
        if not source.reusable:
            raise ValueError(
                "Object-typed feature_ids blocks require a callable source "
                "that can be read twice"
            )
        maximum = 1
        for raw_block in source.produce():
            raw = np.asarray(raw_block)
            block = _string_block(raw, 0, len(raw), name="feature_ids")
            if block.size:
                maximum = max(maximum, max(len(value) for value in block))
        string_dtype = np.dtype(f"U{maximum}")
    else:
        string_dtype = np.empty(0, dtype=source.dtype).astype(str).dtype
    destination = storage.create_metadata(
        group,
        "feature_ids",
        dtype=string_dtype,
        shape=source.shape[0],
        block_rows=block_rows,
    )
    builder = storage.fingerprint_builder()
    builder.begin_array("values", source.shape, string_dtype)
    start = 0
    for raw_block in source.produce():
        raw = np.asarray(raw_block)
        block = _string_block(raw, 0, len(raw), name="feature_ids").astype(string_dtype)
        stop = start + int(block.shape[0])
        if block.ndim != 1 or stop > source.shape[0]:
            raise ValueError("feature_ids block has an invalid shape")
        builder.update_array_block("values", (start,), block)
        destination[start:stop] = block
        start = stop
    if start != source.shape[0]:
        raise ValueError(
            f"feature_ids stream contains {start} rows, expected {source.shape[0]}"
        )
    builder.end_array("values")
    if builder.hexdigest() != expected_fingerprint:
        raise ValueError("feature_ids payload fingerprint does not match its source")
    return destination


def _stored_numeric_fingerprint(
    storage: ImportedArtifactStorage,
    array: zarr.Array,
) -> str:
    builder = storage.fingerprint_builder()
    builder.begin_array("values", array.shape, array.dtype)
    block_rows = storage.block_rows(array)
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        block = np.asarray(array[start:stop])
        builder.update_array_block(
            "values",
            (start,) + (0,) * (array.ndim - 1),
            block,
        )
    builder.end_array("values")
    return builder.hexdigest()


def _stored_payload_fingerprint(
    storage: ImportedArtifactStorage,
    group: zarr.Group,
    name: str,
) -> str:
    array = storage.as_array(group[name], name)
    if name == "feature_ids":
        return storage.fingerprint_stored_strings(array)
    return _stored_numeric_fingerprint(storage, array)


def _payloads_match(
    storage: ImportedArtifactStorage,
    group: zarr.Group,
    *,
    shapes: Mapping[str, tuple[int, ...]],
    fingerprints: Mapping[str, str],
) -> bool:
    try:
        for name, shape in shapes.items():
            if name not in group:
                return False
            array = storage.as_array(group[name], name)
            if tuple(array.shape) != tuple(shape):
                return False
            if _stored_payload_fingerprint(storage, group, name) != fingerprints[name]:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _register_named_result(
    root: zarr.Group,
    ref: ArtifactRef,
    name: str,
    *,
    cell_key: str,
    feat_key: str | None,
) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("named_result must be a non-empty string")
    if ref.assay is None:
        raise ValueError("Named imported results must be assay-scoped")
    state = read_assay_state(root, ref.assay)
    if state is None:
        if feat_key is None:
            raise ValueError(
                "feat_key is required to register a named result without AssayState"
            )
        state = AssayState(
            assay=ref.assay,
            cell_key=cell_key,
            feat_key=feat_key,
        )
    named_results = dict(state.named_results)
    named_results[name] = ref
    write_assay_state(
        root,
        replace(state, named_results=named_results),
    )


def write_imported_coordinates(
    root: zarr.Group,
    *,
    assay: str,
    dimreduc_key: str,
    role: str,
    coordinates: ImportedBlockSource,
    source_digest: bytes,
    payload_fingerprints: Mapping[str, str],
    source_cell_ids: ImportedStringSource,
    cell_selection: ArtifactRef,
    cell_key: str,
    coordinate_shape: tuple[int, int] | None = None,
    coordinate_dtype: Any | None = None,
    loadings: ImportedBlockSource | None = None,
    loadings_shape: tuple[int, int] | None = None,
    loadings_dtype: Any | None = None,
    feature_ids: ImportedFeatureIdSource | None = None,
    feature_id_shape: tuple[int] | None = None,
    feature_id_dtype: Any | None = None,
    stdev: ImportedBlockSource | None = None,
    stdev_shape: tuple[int] | None = None,
    stdev_dtype: Any | None = None,
    named_result: str | None = None,
    feat_key: str | None = None,
    block_rows: int = 100_000,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    storage = ImportedArtifactStorage(root)
    source_digest = _validate_source_digest(source_digest)
    if not isinstance(dimreduc_key, str) or not dimreduc_key:
        raise ValueError("dimreduc_key must be a non-empty string")
    if not isinstance(role, str) or not role or role.lower() in {"umap", "tsne"}:
        raise ValueError("Imported graph coordinates require a non-layout role")
    block_rows = _positive_block_rows(block_rows)
    data_source = _resolve_source(
        coordinates,
        name="coordinate",
        shape=coordinate_shape,
        dtype=coordinate_dtype,
        block_rows=block_rows,
    )
    _validate_numeric_source(data_source, "coordinates", 2)
    dims = int(data_source.shape[1])
    _selection, selected_count, ordered_cell_ids_fingerprint = _selection_alignment(
        root,
        cell_selection=cell_selection,
        cell_key=cell_key,
        source_cell_ids=source_cell_ids,
        coordinate_rows=data_source.shape[0],
        block_rows=block_rows,
    )

    if (loadings is None) != (feature_ids is None):
        raise ValueError("loadings and feature_ids must be provided together")
    loading_source = (
        _resolve_source(
            loadings,
            name="loadings",
            shape=loadings_shape,
            dtype=loadings_dtype,
            block_rows=block_rows,
        )
        if loadings is not None
        else None
    )
    if loading_source is not None:
        _validate_numeric_source(loading_source, "loadings", 2)
        if loading_source.shape[1] != dims:
            raise ValueError("loadings dimensions must match coordinates")
        assert feature_ids is not None
        feature_source = _resolve_feature_id_source(
            feature_ids,
            shape=feature_id_shape,
            dtype=feature_id_dtype,
            block_rows=block_rows,
        )
        if feature_source.shape != (loading_source.shape[0],):
            raise ValueError("feature_ids must align with loadings rows")
    else:
        feature_source = None
    stdev_source = (
        _resolve_source(
            stdev,
            name="stdev",
            shape=stdev_shape,
            dtype=stdev_dtype,
            block_rows=block_rows,
        )
        if stdev is not None
        else None
    )
    if stdev_source is not None:
        _validate_numeric_source(stdev_source, "stdev", 1)
        if stdev_source.shape != (dims,):
            raise ValueError("stdev length must match coordinate dimensions")

    required_payloads = {"data"}
    if loading_source is not None:
        required_payloads.update({"loadings", "feature_ids"})
    if stdev_source is not None:
        required_payloads.add("stdev")
    fingerprints = _required_payload_fingerprints(
        payload_fingerprints,
        required_payloads,
    )
    sources = {"data": data_source}
    if loading_source is not None:
        sources["loadings"] = loading_source
        assert feature_source is not None
        sources["feature_ids"] = feature_source
    if stdev_source is not None:
        sources["stdev"] = stdev_source
    shapes = {name: source.shape for name, source in sources.items()}
    requirements = tuple(
        storage.requirement(
            name,
            shape=shape,
            dtype_kind=None if name == "feature_ids" else "f",
        )
        for name, shape in shapes.items()
    )
    planned = storage.plan(
        assay=assay,
        kind="imported_coordinates",
        parameters={
            "dimreduc_key": dimreduc_key,
            "role": role.lower(),
            "dims": dims,
            "loadings_stored": loading_source is not None,
            "feature_ids_stored": feature_source is not None,
            "stdev_stored": stdev_source is not None,
        },
        inputs={
            "source_digest": source_digest,
            "payload_fingerprints": fingerprints,
            "ordered_cell_ids_fingerprint": ordered_cell_ids_fingerprint,
            "cell_selection": cell_selection,
        },
        execution_options={
            "cell_key": cell_key,
            "block_rows": block_rows,
        },
        invalidate_cache=invalidate_cache,
        required_arrays=requirements,
        reuse_validator=lambda _ref, group: _payloads_match(
            storage,
            group,
            shapes=shapes,
            fingerprints=fingerprints,
        ),
    )
    if not planned.reused:
        group = storage.start(planned)
        _write_numeric_payload(
            storage,
            group,
            "data",
            data_source,
            fingerprints["data"],
            block_rows,
        )
        if loading_source is not None:
            _write_numeric_payload(
                storage,
                group,
                "loadings",
                loading_source,
                fingerprints["loadings"],
                block_rows,
            )
            assert feature_source is not None
            _write_feature_ids(
                storage,
                group,
                feature_source,
                fingerprints["feature_ids"],
                block_rows,
            )
        if stdev_source is not None:
            _write_numeric_payload(
                storage,
                group,
                "stdev",
                stdev_source,
                fingerprints["stdev"],
                block_rows,
            )
        storage.finish(group, planned)
    validate_imported_coordinates_artifact(
        root,
        planned.ref,
        cell_key=cell_key,
    )
    if named_result is not None:
        _register_named_result(
            root,
            planned.ref,
            named_result,
            cell_key=cell_key,
            feat_key=feat_key,
        )
    if selected_count != data_source.shape[0]:
        raise RuntimeError("Imported coordinate selection changed during writing")
    return planned.ref


def validate_imported_embedding_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    cell_key: str | None = None,
) -> None:
    storage = ImportedArtifactStorage(root)
    if ref.kind != "embedding" or ref.scope != "assay" or ref.assay is None:
        raise ValueError("Imported embedding must be an assay-scoped embedding")
    status = storage.require_complete(ref)
    if status.operation != "import_dimreduc":
        raise ValueError("Imported embedding operation must be 'import_dimreduc'")
    execution = status.execution_options or {}
    stored_cell_key = execution.get("cell_key")
    if not isinstance(stored_cell_key, str) or not stored_cell_key:
        raise ValueError("Imported embedding has no cell selection key")
    if cell_key is not None and cell_key != stored_cell_key:
        raise ValueError("cell_key does not match the imported embedding")
    block_rows = execution.get("block_rows")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int | np.integer)
        or int(block_rows) < 1
    ):
        raise ValueError("Imported embedding block_rows is invalid")
    inputs = status.inputs or {}
    raw_selection = inputs.get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ValueError("Imported embedding has no cell selection input")
    selection = ArtifactRef.from_dict(raw_selection)
    validate_cell_selection_artifact(root, selection, stored_cell_key)
    group = storage.artifact_group(ref)
    if "values" not in group:
        raise ValueError("Imported embedding has no values array")
    values = storage.as_array(group["values"], "values")
    parameters = status.parameters or {}
    dimreduc_key = parameters.get("dimreduc_key")
    if not isinstance(dimreduc_key, str) or not dimreduc_key:
        raise ValueError("Imported embedding source key is missing")
    dims = parameters.get("dims")
    role = parameters.get("role")
    if (
        values.ndim != 2
        or int(values.shape[0]) < 1
        or np.dtype(values.dtype).kind != "f"
        or isinstance(dims, bool)
        or not isinstance(dims, int | np.integer)
        or int(dims) < 1
        or tuple(values.shape)[1] != int(dims)
        or role not in {"umap", "tsne"}
    ):
        raise ValueError("Imported embedding payload is malformed")
    selection_values = storage.as_array(
        storage.artifact_group(selection)["values"],
        "values",
    )
    cell_data = storage.as_group(root["cellData"], "cellData")
    ids = storage.as_array(cell_data["ids"], "ids")
    selected_fingerprint, selected_count = fingerprint_selected_stored_strings(
        ids,
        selection_values,
    )
    if int(values.shape[0]) != selected_count:
        raise ValueError("Imported embedding rows do not match its cell selection")
    if inputs.get("ordered_cell_ids_fingerprint") != selected_fingerprint:
        raise ValueError("Imported embedding cell IDs are out of order")
    source_digest = inputs.get("source_digest")
    if (
        not isinstance(source_digest, Mapping)
        or set(source_digest) != {"bytes_hex"}
        or not isinstance(source_digest.get("bytes_hex"), str)
        or len(source_digest["bytes_hex"]) != 64
        or source_digest["bytes_hex"].lower() != source_digest["bytes_hex"]
    ):
        raise ValueError("Imported embedding source digest is missing")
    try:
        bytes.fromhex(source_digest["bytes_hex"])
    except ValueError as exc:
        raise ValueError("Imported embedding source digest is not hexadecimal") from exc
    fingerprints = inputs.get("payload_fingerprints")
    if (
        not isinstance(fingerprints, Mapping)
        or set(fingerprints) != {"values"}
        or fingerprints.get("values") != _stored_numeric_fingerprint(storage, values)
    ):
        raise ValueError("Imported embedding payload fingerprint does not match")


def _default_embedding_columns(
    assay: str,
    cell_key: str,
    role: str,
    dims: int,
) -> tuple[str, ...]:
    label = "UMAP" if role == "umap" else "tSNE"
    prefix = f"{assay}_{label}" if cell_key == "I" else f"{assay}_{cell_key}_{label}"
    return tuple(f"{prefix}{index + 1}" for index in range(dims))


def _publish_embedding_columns(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    selection: zarr.Array,
    columns: Sequence[str],
    cell_data: zarr.Group,
    block_rows: int,
) -> None:
    storage = ImportedArtifactStorage(root)
    group = storage.artifact_group(ref)
    values = storage.as_array(group["values"], "values")
    if len(columns) != int(values.shape[1]):
        raise ValueError(
            "metadata_columns must contain one name per embedding dimension"
        )
    if len(set(columns)) != len(columns) or any(
        not isinstance(column, str) or not column or column in {"I", "ids", "names"}
        for column in columns
    ):
        raise ValueError("metadata_columns contains invalid or duplicate names")
    n_rows = int(selection.shape[0])
    outputs = [
        storage.create_metadata(
            cell_data,
            column,
            dtype=values.dtype,
            shape=n_rows,
            block_rows=block_rows,
        )
        for column in columns
    ]
    for output in outputs:
        output[:] = np.nan
    source_row = 0
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        block_mask = np.asarray(selection[start:stop], dtype=bool)
        selected = int(block_mask.sum())
        source_stop = source_row + selected
        source_values = np.asarray(values[source_row:source_stop])
        if source_values.shape != (selected, len(columns)):
            raise ValueError("Imported embedding values ended before cell metadata")
        if selected:
            for index, output in enumerate(outputs):
                block = np.full(stop - start, np.nan, dtype=values.dtype)
                block[block_mask] = source_values[:, index]
                output[start:stop] = block
        source_row = source_stop
    if source_row != int(values.shape[0]):
        raise ValueError("Imported embedding values exceed the cell selection")
    for index, output in enumerate(outputs):
        output.attrs["source_artifact"] = ref.to_dict()
        output.attrs["source_value"] = "values"
        output.attrs["value_index"] = index


def write_imported_embedding(
    root: zarr.Group,
    *,
    assay: str,
    dimreduc_key: str,
    role: str,
    coordinates: ImportedBlockSource,
    source_digest: bytes,
    payload_fingerprints: Mapping[str, str],
    source_cell_ids: ImportedStringSource,
    cell_selection: ArtifactRef,
    cell_key: str,
    coordinate_shape: tuple[int, int] | None = None,
    coordinate_dtype: Any | None = None,
    metadata_columns: Sequence[str] | None = None,
    cell_data: zarr.Group | None = None,
    named_result: str | None = None,
    feat_key: str | None = None,
    block_rows: int = 100_000,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    storage = ImportedArtifactStorage(root)
    if not isinstance(role, str):
        raise TypeError("role must be a string")
    normalized_role = role.lower()
    if normalized_role not in {"umap", "tsne"}:
        raise ValueError("Imported embeddings require role 'umap' or 'tsne'")
    source_digest = _validate_source_digest(source_digest)
    if not isinstance(dimreduc_key, str) or not dimreduc_key:
        raise ValueError("dimreduc_key must be a non-empty string")
    block_rows = _positive_block_rows(block_rows)
    source = _resolve_source(
        coordinates,
        name="coordinate",
        shape=coordinate_shape,
        dtype=coordinate_dtype,
        block_rows=block_rows,
    )
    _validate_numeric_source(source, "coordinates", 2)
    selection, _selected_count, ordered_cell_ids_fingerprint = _selection_alignment(
        root,
        cell_selection=cell_selection,
        cell_key=cell_key,
        source_cell_ids=source_cell_ids,
        coordinate_rows=source.shape[0],
        block_rows=block_rows,
    )
    fingerprints = _required_payload_fingerprints(
        payload_fingerprints,
        {"values"},
    )
    planned = storage.plan(
        assay=assay,
        kind="embedding",
        parameters={
            "dimreduc_key": dimreduc_key,
            "role": normalized_role,
            "dims": int(source.shape[1]),
        },
        inputs={
            "source_digest": source_digest,
            "payload_fingerprints": fingerprints,
            "ordered_cell_ids_fingerprint": ordered_cell_ids_fingerprint,
            "cell_selection": cell_selection,
        },
        execution_options={
            "cell_key": cell_key,
            "block_rows": block_rows,
        },
        invalidate_cache=invalidate_cache,
        required_arrays=(
            storage.requirement("values", shape=source.shape, dtype_kind="f"),
        ),
        reuse_validator=lambda _ref, group: _payloads_match(
            storage,
            group,
            shapes={"values": source.shape},
            fingerprints=fingerprints,
        ),
    )
    if not planned.reused:
        group = storage.start(planned)
        _write_numeric_payload(
            storage,
            group,
            "values",
            source,
            fingerprints["values"],
            block_rows,
        )
        storage.finish(group, planned)
    validate_imported_embedding_artifact(
        root,
        planned.ref,
        cell_key=cell_key,
    )
    resolved_cell_data = cell_data
    if resolved_cell_data is None and "cellData" in root:
        resolved_cell_data = storage.as_group(root["cellData"], "cellData")
    if resolved_cell_data is not None:
        columns = (
            tuple(metadata_columns)
            if metadata_columns is not None
            else _default_embedding_columns(
                assay,
                cell_key,
                normalized_role,
                int(source.shape[1]),
            )
        )
        _publish_embedding_columns(
            root,
            planned.ref,
            selection=selection,
            columns=columns,
            cell_data=resolved_cell_data,
            block_rows=block_rows,
        )
    elif metadata_columns is not None:
        raise ValueError("cell_data is required when metadata_columns are provided")
    if named_result is not None:
        _register_named_result(
            root,
            planned.ref,
            named_result,
            cell_key=cell_key,
            feat_key=feat_key,
        )
    return planned.ref
