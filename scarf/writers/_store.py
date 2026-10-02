from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import zarr

from ..storage.arrays import (
    MISSING_MASK_PREFIX,
    create_zarr_dataset as _create_zarr_dataset,
    create_zarr_obj_array as _create_zarr_obj_array,
)
from ..storage.schema import (
    create_zarr_count_assay as _create_zarr_count_assay,
)
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.metadata_keys import (
    is_reserved_metadata_name,
    metadata_column_key,
    validate_metadata_column_name,
)
from ..storage.profiles import StorageProfile
from ..storage.refs import ArtifactRef
from ..utils.logging import logger

DEFAULT_IMPORT_BLOCK_ROWS = 65_536


def keyed_metadata_columns[T](
    columns: Iterable[tuple[str, T]],
    keys: Mapping[str, str],
    axis: str,
) -> Iterator[tuple[str, T]]:
    """Yield source metadata columns under the keys planned for them.

    Plan ``keys`` with :func:`~scarf.storage.metadata_keys.metadata_column_keys`
    over the source column names. Columns stream one at a time, so a caller
    holds a single column payload in memory. A source name that the plan
    leaves out is skipped with a warning: it is reserved by Scarf, cannot name
    a Zarr array, or is already a column of the destination. A repeated source
    name is skipped after its first column. Scarf keeps no record of a renamed
    column's source name; the warning is the only report of it.

    Args:
        columns: Pairs of source column name and payload, in source order.
        keys: Destination key of each source name to write.
        axis: Table description used in warnings, such as ``cell``.

    Yields:
        Pairs of destination key and payload.
    """
    seen: set[str] = set()
    for name, payload in columns:
        if name in seen:
            logger.warning(
                f"Skipped source {axis} metadata column {name!r} because the "
                "source repeats that column name"
            )
            continue
        seen.add(name)
        key = keys.get(name)
        if key is None:
            logger.warning(_skipped_column_reason(name, axis))
            continue
        if key != name:
            base = metadata_column_key(name)
            clash = f", and {base!r} is already used" if key != base else ""
            logger.warning(
                f"Stored source {axis} metadata column {name!r} as {key!r} "
                f"because Zarr reads '/' and '\\' as path separators{clash}"
            )
        yield key, payload


def _skipped_column_reason(name: str, axis: str) -> str:
    if is_reserved_metadata_name(metadata_column_key(name)):
        return (
            f"Skipped source {axis} metadata column {name!r} because Scarf "
            "reserves the column names 'I', 'ids', and 'names' and the "
            f"prefix {MISSING_MASK_PREFIX!r}"
        )
    if name in {"", ".", ".."}:
        return (
            f"Skipped source {axis} metadata column {name!r} because that name "
            "cannot name a Zarr array"
        )
    return (
        f"Skipped source {axis} metadata column {name!r} because the "
        f"destination already has a column named {name!r}"
    )


def decode_text(value: Any) -> str:
    """Return a string value, decoding byte strings as UTF-8."""
    if isinstance(value, bytes | np.bytes_):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text values must be valid UTF-8") from exc
    if isinstance(value, str | np.str_):
        return str(value)
    raise TypeError(f"Expected text, found {type(value).__name__}")


def _is_missing_value(value: Any) -> bool:
    return value is None or (
        isinstance(value, float | np.floating) and bool(np.isnan(value))
    )


def _stored_values(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Return a typed column whose masked rows hold a placeholder."""
    if values.dtype.kind == "O":
        present = values[~missing]
        if present.size and all(isinstance(v, bool | np.bool_) for v in present):
            dtype: Any = np.dtype(bool)
        elif present.size and all(
            isinstance(v, int | np.integer) and not isinstance(v, bool | np.bool_)
            for v in present
        ):
            dtype = np.dtype(np.int64)
        elif present.size and all(
            isinstance(v, int | float | np.number)
            and not isinstance(v, bool | np.bool_)
            for v in present
        ):
            dtype = np.dtype(np.float64)
        else:
            text = [
                ""
                if absent
                else (
                    decode_text(value)
                    if isinstance(value, bytes | np.bytes_ | str | np.str_)
                    else str(value)
                )
                for value, absent in zip(values, missing, strict=True)
            ]
            width = max((len(value) for value in text), default=1)
            return np.asarray(text, dtype=f"U{max(width, 1)}")
        stored = np.zeros(values.shape, dtype=dtype)
        stored[~missing] = np.asarray(present.tolist(), dtype=dtype)
        return stored
    if values.dtype.kind in "SU":
        text = [
            "" if absent else decode_text(value)
            for value, absent in zip(values, missing, strict=True)
        ]
        width = max((len(value) for value in text), default=1)
        return np.asarray(text, dtype=f"U{max(width, 1)}")
    stored = values.copy()
    stored[missing] = np.nan if values.dtype.kind in "fc" else 0
    return stored


def write_metadata_column(
    group: zarr.Group,
    name: str,
    values: Any,
    missing: Any = None,
    *,
    profile: StorageProfile | None = None,
) -> None:
    """Write one metadata column and link a missing-value mask when needed.

    Args:
        group: Destination metadata group.
        name: Column name.
        values: One value per row.
        missing: Rows to flag as missing. When None, ``None`` and NaN entries
            of an object array are missing.
        profile: Zarr encoding profile. When None, chosen from the store.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` cannot name one Zarr array, or the values are
            not one-dimensional or the mask does not align with them.
    """
    from ..storage.arrays import MetadataBlock, create_streamed_metadata_column

    validate_metadata_column_name(name)
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"Metadata column {name!r} must hold one value per row; "
            f"found shape {array.shape}"
        )
    if missing is not None:
        mask = np.asarray(missing, dtype=bool)
    elif array.dtype.kind == "O":
        mask = np.fromiter(
            (_is_missing_value(value) for value in array),
            dtype=bool,
            count=array.size,
        )
    else:
        mask = np.zeros(array.shape, dtype=bool)
    if mask.shape != array.shape:
        raise ValueError(f"Metadata column {name!r} has a misaligned missing mask")
    has_missing = bool(mask.any())
    stored = (
        _stored_values(array, mask) if has_missing or array.dtype.kind == "O" else array
    )
    if not has_missing:
        _create_zarr_obj_array(group, name, stored, stored.dtype, profile=profile)
        return
    create_streamed_metadata_column(
        group,
        name,
        shape=int(array.size),
        dtype=stored.dtype,
        blocks=(MetadataBlock(0, stored, mask),),
        chunkSize=min(100_000, max(1, int(array.size))),
        hasMissing=True,
        profile=profile,
    )


def floating_payload_dtype(dtype: Any, label: str) -> np.dtype[Any]:
    """Return the floating dtype that stores an imported numeric payload."""
    source: np.dtype[Any] = np.dtype(dtype)
    if source.kind == "f":
        return np.dtype(source.str)
    if source.kind in "biu":
        return np.dtype(np.float64)
    raise TypeError(f"{label} uses unsupported dtype {source}")


def bounded_block_rows(
    requested: int | None,
    *,
    row_bytes: int,
    memory_bytes: int,
) -> int:
    """Return rows per import block within an eighth of the memory budget."""
    bytes_per_row = max(1, int(row_bytes))
    memory_rows = max(1, int(memory_bytes) // (8 * bytes_per_row))
    preferred = DEFAULT_IMPORT_BLOCK_ROWS if requested is None else requested
    return int(max(1, min(int(preferred), memory_rows)))


def fingerprint_row_blocks(
    blocks: Iterable[np.ndarray],
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    label: str,
) -> str:
    """Fingerprint row blocks of one payload, rejecting non-finite numbers."""
    from ..storage.artifacts import ValueFingerprintBuilder

    builder = ValueFingerprintBuilder()
    builder.begin_array("values", shape, dtype)
    start = 0
    for block in blocks:
        if block.dtype.kind in "fc" and not bool(np.isfinite(block).all()):
            raise ValueError(f"{label} contains non-finite values")
        builder.update_array_block(
            "values",
            (start, *(0,) * (block.ndim - 1)),
            block,
        )
        start += int(block.shape[0])
    builder.end_array("values")
    return str(builder.hexdigest())


def resolve_import_cell_selection(
    root: zarr.Group,
    *,
    source: str,
    inputs: Mapping[str, Any],
) -> ArtifactRef:
    """Return the artifact that selects every imported cell."""
    from ..storage.selections import resolve_stored_selection_artifact

    return resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="import_cell_selection",
        parameters={"source": source},
        inputs=dict(inputs),
    )


def create_zarr_dataset(
    g: zarr.Group,
    name: str,
    chunks: tuple[int, ...] | int,
    dtype: Any,
    shape: tuple[int, ...],
    overwrite: bool = True,
) -> zarr.Array:
    """Creates and returns a Zarr array.

    Args:
        g: Parent Zarr group.
        name: Array name within the group.
        chunks: Chunk shape, or a single integer applied to every axis.
        dtype: NumPy dtype for the array.
        shape: Array shape.
        overwrite: If True, replace an existing array of the same name.

    Returns:
        A Zarr Array.
    """
    return _create_zarr_dataset(g, name, chunks, dtype, shape, overwrite)


def create_zarr_obj_array(
    g: zarr.Group,
    name: str,
    data: Any,
    dtype: str | Any = None,
    overwrite: bool = True,
    chunk_size: int = 100000,
    shape: int | None = None,
) -> zarr.Array:
    """Creates and returns a metadata column array.

    Args:
        g: Parent Zarr group.
        name: Array name within the group.
        data: Values to write, or None to create an empty array.
        dtype: Optional dtype. Inferred from ``data`` when omitted.
        overwrite: If True, replace an existing array of the same name.
        chunk_size: Chunk length along the first axis.
        shape: Explicit length when ``data`` is None.

    Returns:
        A Zarr Array.
    """
    return _create_zarr_obj_array(
        g,
        name,
        data,
        dtype,
        overwrite,
        chunk_size,
        shape,
    )


def create_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: Any,
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
) -> zarr.Array:
    """Creates and returns a Zarr array with name 'counts'.

    Args:
        z: Root Zarr group.
        assay_name: Assay group that will own the counts array.
        workspace: Workspace name. None uses the legacy layout.
        n_cells: Number of cells (rows).
        feat_ids: Feature identifiers written to feature metadata.
        feat_names: Feature display names written to feature metadata.
        dtype: Storage dtype for counts. Import writers store the dtype that
               :func:`~scarf.storage.count_dtype.count_storage_dtype` resolves
               from the canonical values.
        profile: Zarr encoding profile. When None, chosen from the store.
        policy: Count-matrix geometry policy. When None, the default plan
                is used.

    Returns:
        The created ``counts`` array.
    """
    return _create_zarr_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        feat_ids,
        feat_names,
        dtype,
        profile=profile,
        policy=policy,
    )
