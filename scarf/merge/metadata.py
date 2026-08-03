from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import zarr

from ..metadata.rows import (
    array_row_selection_parts,
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ..storage.arrays import (
    MetadataBlock,
    create_streamed_metadata_column,
)
from ..storage.budget import ResourceBudget, admitted_worker_split
from ..storage.layout import PROFILE_METADATA_CHUNK, _encoded_chunk_bound
from ..storage.partition import affordable_width
from ..storage.profiles import StorageProfile
from ..storage.types import as_zarr_array, as_zarr_group
from .row_plan import (
    RowPlan,
    RowPlanSegment,
    iter_row_plan_segments,
    max_row_plan_block_rows,
    verify_merged_cell_ids,
)


_PROTECTED = frozenset({"ids", "I", "names"})


@dataclass(frozen=True, slots=True)
class MetadataColumnSpec:
    name: str
    dtype: np.dtype[Any]
    hasMissing: bool
    role: str | None = None
    assay: str | None = None
    sourceReadFixedBytes: int = 0
    sourceReadBytesPerRow: int = 0
    maskReadFixedBytes: int = 0
    maskReadBytesPerRow: int = 0


def _chunk_resident_bytes(dtype: np.dtype[Any], chunk_rows: int) -> int:
    raw = max(1, int(chunk_rows)) * max(1, int(dtype.itemsize))
    return int(2 * raw + 2 * _encoded_chunk_bound(raw))


def _column_working_bytes(
    spec: MetadataColumnSpec,
    rows: int,
) -> int:
    width = max(1, int(rows))
    itemsize = max(1, int(spec.dtype.itemsize))
    mask_bytes = width * np.dtype(bool).itemsize if spec.hasMissing else 0
    string_bytes = width * itemsize if spec.dtype.kind in {"U", "S", "O"} else 0
    staging = width * 3 * itemsize + mask_bytes + string_bytes
    source_read = max(0, int(spec.sourceReadFixedBytes)) + width * max(
        0, int(spec.sourceReadBytesPerRow)
    )
    mask_read = (
        width * itemsize
        + max(0, int(spec.maskReadFixedBytes))
        + width * max(0, int(spec.maskReadBytesPerRow))
    )
    peaks = [staging, source_read]
    if spec.maskReadFixedBytes or spec.maskReadBytesPerRow:
        peaks.append(mask_read)
    return max(peaks, default=1)


@dataclass(frozen=True, slots=True)
class CellMetadataPlan:
    columns: tuple[MetadataColumnSpec, ...]
    blockRows: int

    def peak_block_bytes_at(self, rows: int) -> int:
        return max(
            (_column_working_bytes(spec, rows) for spec in self.columns),
            default=1,
        )

    def peak_write_bytes_at(self, rows: int, *, chunk_rows: int) -> int:
        width = max(1, int(rows))
        destination_peak = max(
            (
                width * max(1, int(spec.dtype.itemsize))
                + (width * np.dtype(bool).itemsize if spec.hasMissing else 0)
                + max(
                    _chunk_resident_bytes(spec.dtype, chunk_rows),
                    (
                        _chunk_resident_bytes(np.dtype(bool), chunk_rows)
                        if spec.hasMissing
                        else 0
                    ),
                )
                for spec in self.columns
            ),
            default=1,
        )
        return max(self.peak_block_bytes_at(width), destination_peak)


def metadata_chunk_rows(row_plan: RowPlan) -> int:
    """Return the deterministic chunk width for merged cell metadata."""
    return max(
        1,
        min(
            PROFILE_METADATA_CHUNK,
            max(1, max_row_plan_block_rows(row_plan)),
            max(1, int(row_plan.nCells)),
        ),
    )


def effective_metadata_segment_rows(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
) -> int:
    return max(
        1,
        min(int(metadata_plan.blockRows), metadata_chunk_rows(row_plan)),
    )


def resolve_metadata_segment_rows(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> int:
    """Return the largest metadata write width that fits the budget."""
    preferred = max(
        1,
        min(int(metadata_plan.blockRows), metadata_chunk_rows(row_plan)),
    )
    chunk_rows = metadata_chunk_rows(row_plan)

    def fits(width: int) -> bool:
        try:
            admitted_worker_split(
                resources,
                nTasks=1,
                residentBytes=max(0, int(resident_bytes)),
                taskBytes=lambda _: metadata_plan.peak_write_bytes_at(
                    width,
                    chunk_rows=chunk_rows,
                ),
                requested=1,
            )
        except MemoryError:
            return False
        return True

    rows = affordable_width(fits, preferred)
    if rows < 1:
        raise MemoryError(
            "Merged cell metadata cannot fit one row within the operation memory budget"
        )
    return int(rows)


def admit_cell_metadata_plan(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> CellMetadataPlan:
    """Resolve a budget-admitted metadata width and keep schema specs stable."""
    admitted = resolve_metadata_segment_rows(
        metadata_plan,
        row_plan,
        resources,
        resident_bytes=resident_bytes,
    )
    return replace(metadata_plan, blockRows=admitted)


def resolve_identity_validation_rows(
    metadata_plan: CellMetadataPlan,
    row_plan: RowPlan,
    stored_ids: Any,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> int:
    """Return the largest admitted cell-identity validation width."""
    preferred = min(
        metadata_chunk_rows(row_plan),
        max(1, max_row_plan_block_rows(row_plan)),
    )
    ids_spec = next(spec for spec in metadata_plan.columns if spec.name == "ids")
    stored_fixed, stored_per_row = array_row_selection_parts(stored_ids)
    expected_per_row = max(1, int(ids_spec.dtype.itemsize))
    index_per_row = np.dtype(np.int64).itemsize

    def fits(width: int) -> bool:
        source_phase = _column_working_bytes(ids_spec, width)
        destination_phase = (
            stored_fixed
            + width * stored_per_row
            + width * (expected_per_row + index_per_row)
        )
        try:
            admitted_worker_split(
                resources,
                nTasks=1,
                residentBytes=max(0, int(resident_bytes)),
                taskBytes=lambda _: max(source_phase, destination_phase),
                requested=1,
            )
        except MemoryError:
            return False
        return True

    rows = affordable_width(fits, preferred)
    if rows < 1:
        raise MemoryError(
            "Merged cell identity validation cannot fit one row within the "
            "operation memory budget"
        )
    return int(rows)


def _public_column_name(
    column: str,
    prepend_text: str | None,
) -> str:
    if column in _PROTECTED:
        return column
    if prepend_text is None or prepend_text == "":
        return column
    return f"{prepend_text}_{column}"


def _max_text_width(
    table: Any,
    column: str,
    *,
    block_rows: int,
) -> int:
    width = 1
    for values in iter_metadata_column_blocks(
        table,
        column,
        block_rows=block_rows,
    ):
        if np.asarray(values).dtype.kind == "O":
            if len(values):
                width = max(width, max(len(str(value)) for value in values))
            continue
        strings = np.asarray(values, dtype=str)
        if strings.size:
            width = max(width, int(np.char.str_len(strings).max()))
    return width


def _string_itemsize_bound(dtype: np.dtype[Any]) -> int:
    if dtype.kind == "U":
        return max(1, int(dtype.itemsize))
    if dtype.kind == "S":
        return max(1, 4 * int(dtype.itemsize))
    if dtype.kind == "b":
        return 4 * len("False")
    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        return 4 * max(len(str(info.min)), len(str(info.max)))
    if dtype.kind == "f":
        return 4 * 64
    if dtype.kind == "c":
        return 4 * 128
    if dtype.kind in {"M", "m"}:
        return 4 * 64
    return max(1, int(dtype.itemsize))


def resolve_metadata_schema_scan_rows(
    source_cell_tables: list[Any],
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
    preferred_rows: int,
) -> int:
    """Return the largest admitted metadata schema-scan width."""
    arrays: list[Any] = []
    for table in source_cell_tables:
        for column in table.columns:
            dtype = np.dtype(table.get_dtype(column))
            if column in {"ids", "names"} or dtype.kind in {"U", "S", "O"}:
                arrays.append(table._get_array(column))
    preferred = max(1, int(preferred_rows))

    def fits(width: int) -> bool:
        task_bytes: list[int] = []
        for array in arrays:
            dtype = np.dtype(array.dtype)
            itemsize = max(1, int(dtype.itemsize))
            text_itemsize = _string_itemsize_bound(dtype)
            selection = array_row_selection_parts(array)
            selection_peak = selection[0] + width * selection[1]
            conversion_peak = width * (
                itemsize + text_itemsize + np.dtype(np.int64).itemsize
            )
            task_bytes.append(max(selection_peak, conversion_peak))
        try:
            admitted_worker_split(
                resources,
                nTasks=1,
                residentBytes=max(0, int(resident_bytes)),
                taskBytes=lambda _: max(task_bytes, default=1),
                requested=1,
            )
        except MemoryError:
            return False
        return True

    rows = affordable_width(fits, preferred)
    if rows < 1:
        raise MemoryError(
            "Merged cell metadata schema discovery cannot fit one source row "
            "within the operation memory budget"
        )
    return int(rows)


def _max_selection_parts(arrays: Iterable[Any]) -> tuple[int, int]:
    fixed = 0
    per_row = 0
    for array in arrays:
        array_fixed, array_per_row = array_row_selection_parts(array)
        fixed = max(fixed, array_fixed)
        per_row = max(per_row, array_per_row)
    return fixed, per_row


def _metadata_column_spec(
    name: str,
    dtype: np.dtype[Any],
    has_missing: bool,
    *,
    value_arrays: Iterable[Any] = (),
    mask_arrays: Iterable[Any] = (),
    role: str | None = None,
    assay: str | None = None,
) -> MetadataColumnSpec:
    source_fixed, source_per_row = _max_selection_parts(value_arrays)
    mask_fixed, mask_per_row = _max_selection_parts(mask_arrays)
    return MetadataColumnSpec(
        name,
        dtype,
        has_missing,
        role=role,
        assay=assay,
        sourceReadFixedBytes=source_fixed,
        sourceReadBytesPerRow=source_per_row,
        maskReadFixedBytes=mask_fixed,
        maskReadBytesPerRow=mask_per_row,
    )


def plan_cell_metadata(
    source_cell_tables: list[Any],
    source_names: list[str],
    *,
    prepend_text: str | None,
    reset_cell_filter: bool,
    source_column: str | None,
    membership_assays: list[str] | None = None,
    block_rows: int = 100_000,
    scan_rows: int | None = None,
) -> CellMetadataPlan:
    """Resolve the destination cell-metadata schema without writing."""
    block_rows = max(1, int(block_rows))
    scan_rows = block_rows if scan_rows is None else max(1, int(scan_rows))
    if source_column is not None and (
        not isinstance(source_column, str)
        or not source_column.strip()
        or source_column in _PROTECTED
    ):
        raise ValueError(
            "source_column must be a non-empty string that is not ids, I, or names"
        )
    if prepend_text == "":
        prepend_text = None

    # Collect public columns per source.
    per_source: list[dict[str, str]] = []
    for table in source_cell_tables:
        mapping: dict[str, str] = {}
        for column in table.columns:
            public = _public_column_name(column, prepend_text)
            mapping[public] = column
        per_source.append(mapping)

    all_public: set[str] = set()
    for mapping in per_source:
        all_public.update(mapping)
    all_public.update(_PROTECTED)
    membership = membership_assays or []
    membership_columns = {f"{assay_name}_I" for assay_name in membership}
    all_public.update(membership_columns)
    if source_column is not None:
        if source_column in all_public:
            raise ValueError(
                f"source_column {source_column!r} conflicts with merged metadata"
            )
        all_public.add(source_column)

    columns: list[MetadataColumnSpec] = []
    # Stable order: ids, names, I, source, membership, then remaining sorted.
    ordered = ["ids", "names", "I"]
    if source_column is not None:
        ordered.append(source_column)
    for assay_name in membership:
        ordered.append(f"{assay_name}_I")
    remaining = sorted(name for name in all_public if name not in ordered)
    ordered.extend(remaining)

    for public in ordered:
        if public == "ids":
            max_len = 1
            for table, name in zip(source_cell_tables, source_names, strict=True):
                max_len = max(
                    max_len,
                    len(name)
                    + 2
                    + _max_text_width(
                        table,
                        "ids",
                        block_rows=scan_rows,
                    ),
                )
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(f"U{max_len}"),
                    False,
                    value_arrays=(
                        table._get_array("ids") for table in source_cell_tables
                    ),
                )
            )
            continue
        if public == "names":
            max_len = 1
            for table in source_cell_tables:
                max_len = max(
                    max_len,
                    _max_text_width(
                        table,
                        "names",
                        block_rows=scan_rows,
                    ),
                )
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(f"U{max_len}"),
                    False,
                    value_arrays=(
                        table._get_array("names") for table in source_cell_tables
                    ),
                )
            )
            continue
        if public == "I":
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(bool),
                    False,
                    value_arrays=(
                        ()
                        if reset_cell_filter
                        else (table._get_array("I") for table in source_cell_tables)
                    ),
                )
            )
            continue
        if public == source_column:
            max_len = max(1, max(len(name) for name in source_names))
            columns.append(
                _metadata_column_spec(public, np.dtype(f"U{max_len}"), False)
            )
            continue
        if public.endswith("_I") and public[:-2] in membership:
            columns.append(
                _metadata_column_spec(
                    public,
                    np.dtype(bool),
                    False,
                    role="assay_membership",
                    assay=public[:-2],
                )
            )
            continue

        present_dtypes: list[np.dtype[Any]] = []
        present_count = 0
        for table, mapping in zip(source_cell_tables, per_source, strict=True):
            source_col = mapping.get(public)
            if source_col is None:
                continue
            present_count += 1
            present_dtypes.append(np.dtype(table.get_dtype(source_col)))
        has_missing = present_count < len(source_cell_tables) or any(
            source_col is not None
            and metadata_missing_mask(table, source_col) is not None
            for table, mapping in zip(source_cell_tables, per_source, strict=True)
            if (source_col := mapping.get(public)) is not None
        )
        dtype = _promote_dtypes(present_dtypes)
        if dtype.kind in {"U", "S", "O"}:
            max_len = 1
            for table, mapping in zip(source_cell_tables, per_source, strict=True):
                source_col = mapping.get(public)
                if source_col is None:
                    continue
                max_len = max(
                    max_len,
                    _max_text_width(
                        table,
                        source_col,
                        block_rows=scan_rows,
                    ),
                )
            dtype = np.dtype(f"U{max_len}")
        value_arrays: list[Any] = []
        mask_arrays: list[Any] = []
        for table, mapping in zip(source_cell_tables, per_source, strict=True):
            source_col = mapping.get(public)
            if source_col is None:
                continue
            value_arrays.append(table._get_array(source_col))
            mask = metadata_missing_mask(table, source_col)
            if mask is not None:
                mask_arrays.append(mask)
        columns.append(
            _metadata_column_spec(
                public,
                dtype,
                has_missing,
                value_arrays=value_arrays,
                mask_arrays=mask_arrays,
            )
        )

    # reset_cell_filter only affects values, not schema.
    _ = reset_cell_filter
    return CellMetadataPlan(tuple(columns), block_rows)


def _promote_dtypes(dtypes: list[np.dtype[Any]]) -> np.dtype[Any]:
    if not dtypes:
        return np.dtype(bool)
    if all(dtype == dtypes[0] for dtype in dtypes):
        return dtypes[0]
    if all(dtype.kind in "biu" for dtype in dtypes):
        return np.result_type(*dtypes)
    if all(dtype.kind in "bif" for dtype in dtypes):
        return np.dtype(np.float64)
    if any(dtype.kind in {"U", "S", "O"} for dtype in dtypes):
        return np.dtype("U1")
    return np.dtype(np.float64)


def _fill_value(dtype: np.dtype[Any]) -> Any:
    if dtype.kind == "b":
        return False
    if dtype.kind in "iu":
        return 0
    if dtype.kind == "f":
        return np.nan
    return ""


def _iter_destination_chunk_segments(
    row_plan: RowPlan,
    *,
    segment_rows: int,
    chunk_rows: int,
) -> Iterator[RowPlanSegment]:
    for segment in iter_row_plan_segments(row_plan):
        offset = 0
        while offset < segment.localRows.size:
            dest_start = segment.destStart + offset
            rows_to_boundary = chunk_rows - (dest_start % chunk_rows)
            width = min(
                segment_rows,
                rows_to_boundary,
                int(segment.localRows.size) - offset,
            )
            yield RowPlanSegment(
                sourceIdx=segment.sourceIdx,
                blockIdx=segment.blockIdx,
                destStart=dest_start,
                localRows=segment.localRows[offset : offset + width],
            )
            offset += width


def _iter_column_blocks(
    spec: MetadataColumnSpec,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    prepend_text: str | None,
    reset_cell_filter: bool,
    source_column: str | None,
    membership_by_source: dict[str, set[str]] | None,
    segment_rows: int,
    chunk_rows: int,
) -> Iterator[MetadataBlock]:
    if prepend_text == "":
        prepend_text = None
    membership_by_source = membership_by_source or {}
    per_source_map: list[dict[str, str]] = []
    for table in source_cell_tables:
        mapping: dict[str, str] = {}
        for column in table.columns:
            public = _public_column_name(column, prepend_text)
            mapping[public] = column
        per_source_map.append(mapping)

    for segment in _iter_destination_chunk_segments(
        row_plan,
        segment_rows=segment_rows,
        chunk_rows=chunk_rows,
    ):
        source_idx = segment.sourceIdx
        local_rows = segment.localRows
        name = row_plan.sourceNames[source_idx]
        table = source_cell_tables[source_idx]
        n = int(local_rows.size)
        missing: np.ndarray | None = None

        if spec.name == "ids":
            source_values = read_metadata_rows_chunkwise(table, "ids", local_rows)
            values = np.empty(n, dtype=spec.dtype)
            for index, value in enumerate(source_values):
                values[index] = f"{name}__{value}"
            del source_values
        elif spec.name == "names":
            source_values = read_metadata_rows_chunkwise(table, "names", local_rows)
            values = np.asarray(
                source_values,
                dtype=spec.dtype,
            )
            del source_values
        elif spec.name == "I":
            if reset_cell_filter:
                values = np.ones(n, dtype=bool)
            else:
                values = np.asarray(
                    read_metadata_rows_chunkwise(table, "I", local_rows),
                    dtype=bool,
                )
        elif spec.name == source_column:
            values = np.full(n, name, dtype=spec.dtype)
        elif spec.role == "assay_membership":
            present = spec.assay in membership_by_source.get(name, set())
            values = np.full(n, present, dtype=bool)
        else:
            source_col = per_source_map[source_idx].get(spec.name)
            if source_col is None:
                values = np.full(n, _fill_value(spec.dtype), dtype=spec.dtype)
                if spec.hasMissing:
                    missing = np.ones(n, dtype=bool)
            else:
                raw = read_metadata_rows_chunkwise(table, source_col, local_rows)
                values = raw.astype(spec.dtype, copy=False)
                del raw
                if spec.hasMissing:
                    source_missing = read_metadata_missing_rows_chunkwise(
                        table,
                        source_col,
                        local_rows,
                    )
                    missing = (
                        np.zeros(n, dtype=bool)
                        if source_missing is None
                        else source_missing
                    )

        yield MetadataBlock(
            segment.destStart,
            values,
            missing,
        )


def write_cell_metadata(
    root: zarr.Group,
    workspace: str | None,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    metadata_plan: CellMetadataPlan,
    *,
    profile: StorageProfile,
    prepend_text: str | None,
    reset_cell_filter: bool,
    source_column: str | None,
    membership_by_source: dict[str, set[str]] | None = None,
    overwrite: bool = True,
) -> zarr.Group:
    """Stream cell metadata columns in merged row order."""
    cell_slot = "cellData" if workspace is None else f"{workspace}/cellData"
    if cell_slot in root and overwrite:
        del root[cell_slot]
    group = root.create_group(cell_slot)
    segment_rows = effective_metadata_segment_rows(metadata_plan, row_plan)
    chunk_size = metadata_chunk_rows(row_plan)
    for spec in metadata_plan.columns:
        blocks = _iter_column_blocks(
            spec,
            row_plan,
            source_cell_tables,
            prepend_text=prepend_text,
            reset_cell_filter=reset_cell_filter,
            source_column=source_column,
            membership_by_source=membership_by_source,
            segment_rows=segment_rows,
            chunk_rows=chunk_size,
        )
        array = create_streamed_metadata_column(
            group,
            spec.name,
            shape=row_plan.nCells,
            dtype=spec.dtype,
            blocks=blocks,
            overwrite=True,
            chunkSize=chunk_size,
            hasMissing=spec.hasMissing,
            profile=profile,
        )
        if spec.role is not None:
            array.attrs["role"] = spec.role
        if spec.assay is not None:
            array.attrs["assay"] = spec.assay
    group.attrs["complete"] = True
    return group


def validate_cell_metadata(
    root: zarr.Group,
    workspace: str | None,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    metadata_plan: CellMetadataPlan,
    *,
    resources: ResourceBudget,
    resident_bytes: int,
) -> str | None:
    """Return why a completed cellData component cannot be reused."""
    cell_path = "cellData" if workspace is None else f"{workspace}/cellData"
    if cell_path not in root:
        return f"cell metadata group {cell_path!r} is missing"
    group = as_zarr_group(root[cell_path], name=cell_path)
    if group.attrs.get("complete") is not True:
        return f"cell metadata group {cell_path!r} is not complete"

    expected_chunks = (metadata_chunk_rows(row_plan),)
    for spec in metadata_plan.columns:
        if spec.name not in group:
            return f"cell metadata column {spec.name!r} is missing"
        array = as_zarr_array(group[spec.name], name=f"{cell_path}/{spec.name}")
        if tuple(int(value) for value in array.shape) != (row_plan.nCells,):
            return f"cell metadata column {spec.name!r} has the wrong shape"
        if np.dtype(array.dtype) != spec.dtype:
            return (
                f"cell metadata column {spec.name!r} has dtype "
                f"{np.dtype(array.dtype)}, expected {spec.dtype}"
            )
        if tuple(int(value) for value in array.chunks) != expected_chunks:
            return f"cell metadata column {spec.name!r} has the wrong chunks"
        if spec.role is not None and array.attrs.get("role") != spec.role:
            return f"cell metadata column {spec.name!r} has the wrong role"
        if spec.assay is not None and array.attrs.get("assay") != spec.assay:
            return f"cell metadata column {spec.name!r} has the wrong assay"
        missing_name = f"__scarf_missing__{spec.name}"
        if spec.hasMissing:
            if array.attrs.get("missing_mask") != missing_name:
                return f"cell metadata column {spec.name!r} has no missing mask"
            if missing_name not in group:
                return f"missing mask for cell metadata column {spec.name!r} is missing"
            missing = as_zarr_array(
                group[missing_name],
                name=f"{cell_path}/{missing_name}",
            )
            if tuple(int(value) for value in missing.shape) != (row_plan.nCells,):
                return f"missing mask for cell metadata column {spec.name!r} has the wrong shape"
            if np.dtype(missing.dtype) != np.dtype(bool):
                return f"missing mask for cell metadata column {spec.name!r} has the wrong dtype"
            if tuple(int(value) for value in missing.chunks) != expected_chunks:
                return f"missing mask for cell metadata column {spec.name!r} has the wrong chunks"

    identity_rows = resolve_identity_validation_rows(
        metadata_plan,
        row_plan,
        group["ids"],
        resources,
        resident_bytes=resident_bytes,
    )
    try:
        verify_merged_cell_ids(
            as_zarr_array(group["ids"], name=f"{cell_path}/ids"),
            row_plan,
            source_cell_tables,
            block_rows=identity_rows,
        )
    except ValueError as error:
        return str(error)
    return None
