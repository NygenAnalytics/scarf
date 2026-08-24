from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metadata.rows import (
    read_array_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ..utils.arrays import permute_into_chunks


@dataclass(frozen=True, slots=True)
class RowPlan:
    """Shared merged cell order across every assay in a DataStoreMerge."""

    permutationsRows: dict[int, dict[int, np.ndarray]]
    coordinatesPermutations: np.ndarray
    cellOrder: dict[int, dict[int, np.ndarray]]
    nCells: int
    sourceNames: tuple[str, ...]

    def resident_bytes(self) -> int:
        arrays = [
            self.coordinatesPermutations,
            *(
                rows
                for chunks in self.permutationsRows.values()
                for rows in chunks.values()
            ),
            *(rows for chunks in self.cellOrder.values() for rows in chunks.values()),
        ]
        return sum(
            array.nbytes for array in {id(array): array for array in arrays}.values()
        )


@dataclass(frozen=True, slots=True)
class RowPlanSegment:
    sourceIdx: int
    blockIdx: int
    destStart: int
    localRows: np.ndarray


def build_row_plan(
    n_cells_per_source: list[int],
    row_chunk_sizes: list[int],
    source_names: list[str],
    seed: int | None = 42,
) -> RowPlan:
    """Build a deterministic shared row order for concatenated sources."""
    if len(n_cells_per_source) != len(row_chunk_sizes):
        raise ValueError("Row chunk sizes must match the number of sources")
    if len(n_cells_per_source) != len(source_names):
        raise ValueError("Source names must match the number of sources")
    if any(rows <= 0 for rows in row_chunk_sizes):
        raise ValueError("Row chunk sizes must be positive")
    if len(source_names) != len(set(source_names)):
        raise ValueError("A unique name must be provided for each source DataStore")

    rng = np.random.default_rng(seed=seed)
    chunk_size = np.asarray(row_chunk_sizes, dtype=int)
    n_cells = np.asarray(n_cells_per_source, dtype=int)
    # Within-block permutation keeps the historical fixed seed used by
    # permute_into_chunks; the caller seed only reorders source blocks.
    permutations = {
        i: permute_into_chunks(int(n_cells[i]), int(chunk_size[i]))
        for i in range(len(n_cells_per_source))
    }
    permutations_rows = {
        key: {i: x for i, x in enumerate(arrays)}
        for key, arrays in permutations.items()
    }

    permutations_rows_offset: dict[int, dict[int, np.ndarray]] = {}
    offset = 0
    for key, val_dict in permutations_rows.items():
        permutations_rows_offset[key] = {
            in_key: arrs + offset for in_key, arrs in val_dict.items()
        }
        offset += int(n_cells[key])

    coordinates: list[list[int]] = []
    extra: list[list[int]] = []
    for i in range(len(n_cells_per_source)):
        for j in range(len(permutations[i])):
            if j == len(permutations[i]) - 1:
                extra.append([i, j])
                continue
            coordinates.append([i, j])
    coordinates_permutations = rng.permutation(coordinates)
    if len(coordinates_permutations) > 0:
        coordinates_permutations = np.concatenate(
            [coordinates_permutations, extra],
            axis=0,
        )
    else:
        coordinates_permutations = np.array(extra, dtype=np.int64)

    if permutations_rows_offset:
        first = permutations_rows_offset[0][0]
        if int(first.min()) != 0:
            raise AssertionError(
                "ERROR: Randomization of rows failed. The first row should be at 0. "
                "Please report this issue."
            )
        last_source = max(permutations_rows_offset)
        last_block = max(permutations_rows_offset[last_source])
        if int(permutations_rows_offset[last_source][last_block].max()) != int(
            n_cells.sum() - 1
        ):
            raise AssertionError(
                "ERROR: Randomization of rows failed. The last row should be at "
                "the end of the dataset. Please report this issue."
            )

    cell_order: dict[int, dict[int, np.ndarray]] = {
        i: {} for i in range(len(n_cells_per_source))
    }
    offset = 0
    for x, y in coordinates_permutations:
        size = permutations_rows[int(x)][int(y)].size
        cell_order[int(x)][int(y)] = np.arange(offset, offset + size, dtype=np.int64)
        offset += size

    return RowPlan(
        permutationsRows=permutations_rows,
        coordinatesPermutations=np.asarray(coordinates_permutations, dtype=np.int64),
        cellOrder=cell_order,
        nCells=int(n_cells.sum()),
        sourceNames=tuple(source_names),
    )


def max_row_plan_block_rows(row_plan: RowPlan) -> int:
    return max(
        (
            int(rows.size)
            for blocks in row_plan.permutationsRows.values()
            for rows in blocks.values()
        ),
        default=0,
    )


def iter_row_plan_segments(
    row_plan: RowPlan,
    *,
    segment_rows: int | None = None,
) -> Iterator[RowPlanSegment]:
    """Yield destination-ordered source row segments from a row plan."""
    if segment_rows is not None and int(segment_rows) < 1:
        raise ValueError("segment_rows must be positive")

    expected_start = 0
    for source_value, block_value in row_plan.coordinatesPermutations:
        source_idx = int(source_value)
        block_idx = int(block_value)
        local_rows = row_plan.permutationsRows[source_idx][block_idx]
        destination_rows = row_plan.cellOrder[source_idx][block_idx]
        if local_rows.size != destination_rows.size:
            raise AssertionError(
                "Merged row plan source and destination blocks have different sizes"
            )
        if local_rows.size:
            dest_start = int(destination_rows[0])
            if dest_start != expected_start:
                raise AssertionError(
                    "Merged row plan destination segments are not contiguous"
                )
        else:
            dest_start = expected_start

        if local_rows.size:
            width = local_rows.size if segment_rows is None else int(segment_rows)
            for offset in range(0, local_rows.size, width):
                yield RowPlanSegment(
                    sourceIdx=source_idx,
                    blockIdx=block_idx,
                    destStart=dest_start + offset,
                    localRows=local_rows[offset : offset + width],
                )
        expected_start += int(local_rows.size)

    if expected_start != row_plan.nCells:
        raise AssertionError("Merged row plan does not cover every planned cell")


def iter_merged_cell_ids(
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    dtype: Any,
    block_rows: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield contiguous merged cell-id blocks in destination row order.

    Each yield is ``(start, ids)`` where ``ids`` are already prefixed with
    ``{source_name}__``. Blocks follow ``coordinatesPermutations`` so a single
    generator can both write and verify cell identity.
    """
    if len(source_cell_tables) != len(row_plan.sourceNames):
        raise ValueError("Source cell tables must match the row plan")
    for segment in iter_row_plan_segments(row_plan, segment_rows=block_rows):
        source_idx = segment.sourceIdx
        source_ids = read_metadata_rows_chunkwise(
            source_cell_tables[source_idx],
            "ids",
            segment.localRows,
        )
        name = row_plan.sourceNames[source_idx]
        merged = np.empty(source_ids.size, dtype=np.dtype(dtype))
        for index, value in enumerate(source_ids):
            merged[index] = f"{name}__{value}"
        del source_ids
        yield segment.destStart, merged


def verify_merged_cell_ids(
    stored_ids: Any,
    row_plan: RowPlan,
    source_cell_tables: list[Any],
    *,
    block_rows: int = 100_000,
) -> None:
    """Compare stored cell ids against the row-plan identity generator."""
    if int(stored_ids.shape[0]) != row_plan.nCells:
        raise ValueError(
            "ERROR: order of cells does not match the one in existing file"
        )
    for start, expected in iter_merged_cell_ids(
        row_plan,
        source_cell_tables,
        dtype=stored_ids.dtype,
        block_rows=block_rows,
    ):
        stop = start + expected.size
        destination_rows = np.arange(start, stop, dtype=np.int64)
        actual = read_array_rows_chunkwise(stored_ids, destination_rows)
        if not np.array_equal(actual, expected):
            raise ValueError(
                "ERROR: order of cells does not match the one in existing file"
            )
