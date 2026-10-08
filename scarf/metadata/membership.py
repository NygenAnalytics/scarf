"""Cell columns that record which cells an assay measures.

A store whose assay measures only some of its cells marks the measured cells
in the boolean cell column ``<assay>_I`` (see
:func:`~scarf.storage.metadata_keys.assay_membership_column`). The column has
no missing-value mask, and its attributes include those
:func:`membership_attributes` returns; other attributes are allowed. A store
without that column measures every cell with the assay, and a cell outside an
assay holds none of its counts.

Only imports, merges, and derived assays, which take the membership of the
assay that they derive from, write the column, directly in storage.
:class:`~scarf.metadata.MetaData` refuses to write the membership column of
any assay of its store or workspace, whether or not it exists, and to drop one
that carries the membership role (see
:func:`~scarf.storage.identity.protect_metadata_column`). A subset or a merge
that leaves an assay out leaves its membership column out too.

An AnnData that Scarf exports holds the counts of one assay. It keeps that
assay's membership column as an ordinary ``obs`` column and declares it in
``uns["scarf"]["assayMembership"]``, which maps the assay to its membership
column, so an import restores the column as the membership of that assay
instead of guessing it from the name. The membership columns of other assays
describe assays that the file does not hold, so the export leaves them out
(see :func:`exported_membership`).

An operation that reads an assay's values over a cell selection refuses cells
that the assay did not measure, whose zero counts are no measurement, with
:class:`UnmeasuredCellsError` (see :func:`require_measured_cells`), and
:func:`measured_selection` keeps the measured cells of a selection.
Display reads show those cells as missing (see :func:`measured_rows`).

Membership is read in bounded blocks. A selection of a few cells reads only
the chunks that hold them; a selection of at least one cell in
``_DENSE_SELECTION`` streams the column once.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np

from ..storage.artifacts import ValueFingerprintBuilder
from ..storage.metadata_keys import (
    ASSAY_MEMBERSHIP_ROLE,
    assay_membership_attributes,
    assay_membership_column,
)
from .rows import (
    iter_metadata_column_blocks,
    metadata_missing_mask,
    read_metadata_rows_chunkwise,
)

# A selection of at least one cell in this many streams the membership column
# once, since it touches nearly every chunk; a smaller one reads only the
# chunks that hold its rows.
_DENSE_SELECTION = 16

type MeasuredCellsRemedy = Literal[
    "labels", "graph", "cell_key", "cell_selection", "export"
]
"""How a refused caller selects measured cells, by the input it takes.

``"labels"``: the cells come from a label artifact, which
``snapshot_cluster_labels`` narrows. ``"graph"``: the cells come from the
lineage of a graph, which must be built again over measured cells.
``"cell_key"``: the cells are those of a boolean cell column, so the caller
takes another column, True only for the cells of that column that the assay
measured. ``"cell_selection"``: the caller's ``cell_selection`` keyword
narrows the cells that it reads, such as those of its labels or of the live
``I`` column. ``"export"``: a raw export that cannot declare which cells the
assay measured exports only measured cells, or an export that declares them
is used. Without one, the caller takes a cell selection, which
``select_measured_cells`` narrows.
"""

SCARF_UNS_KEY = "scarf"
"""Key of Scarf's namespace in the ``uns`` mapping of an exported AnnData."""

MEMBERSHIP_UNS_KEY = "assayMembership"
"""Key, in Scarf's ``uns`` namespace, of each assay's membership column."""


def membership_attributes(assay: str) -> dict[str, str]:
    """Return the attributes that mark a cell column as ``assay``'s membership."""
    return assay_membership_attributes(assay)


def _has_column(table: Any, name: str) -> bool:
    """Return whether ``table`` has a column ``name``.

    A ``MetaData`` table looks the one column up, so an object store does not
    list every column; another table answers from its column list.
    """
    has_column = getattr(table, "_has_column", None)
    if callable(has_column):
        return bool(has_column(name))
    return name in table.columns


def resolve_assay_membership(table: Any, assay: str) -> str | None:
    """Return ``assay``'s membership column, or None if the cell table has none."""
    name = assay_membership_column(assay)
    if not _has_column(table, name):
        return None
    dtype = np.dtype(table.get_dtype(name))
    expected = membership_attributes(assay)
    attributes = getattr(table._get_array(name), "attrs", {})
    problem: str | None = None
    if dtype.kind != "b":
        problem = f"has dtype {dtype}, not bool"
    elif metadata_missing_mask(table, name) is not None:
        problem = "has a missing-value mask"
    else:
        found = {key: attributes.get(key) for key in expected}
        if found != expected:
            problem = f"has attributes {found}, not {expected}"
    if problem is None:
        return name
    if attributes.get("role") == ASSAY_MEMBERSHIP_ROLE:
        remedy = "Import the data again to write a valid column."
    else:
        remedy = (
            "Without the membership role it is a plain column, such as one that "
            "an earlier release imported as ordinary metadata: drop it from the "
            f"store's cells with cells.drop({name!r}), after which the assay "
            "counts every cell as measured, or import the data again to "
            "restore the membership."
        )
    raise ValueError(
        f"Cell column {name!r} is reserved for the membership of assay "
        f"{assay!r}, but it {problem}. A membership column is a boolean "
        f"column without missing values whose role and assay attributes are "
        f"these. {remedy}"
    )


def membership_columns(table: Any) -> dict[str, str]:
    """Return the membership column of each assay that a cell table records."""
    found: dict[str, str] = {}
    for name in table.columns:
        attributes = getattr(table._get_array(name), "attrs", {})
        if attributes.get("role") != ASSAY_MEMBERSHIP_ROLE:
            continue
        assay = attributes.get("assay")
        if isinstance(assay, str) and assay_membership_column(assay) == name:
            column = resolve_assay_membership(table, assay)
            assert column is not None
            found[assay] = column
    return found


def exported_membership(
    table: Any, assay: str
) -> tuple[dict[str, str], frozenset[str]]:
    """Return the membership that an export of ``assay`` writes and leaves out."""
    columns = membership_columns(table)
    declared = {assay: columns[assay]} if assay in columns else {}
    omitted = frozenset(column for name, column in columns.items() if name != assay)
    return declared, omitted


def membership_declaration(
    declared: Mapping[str, str],
) -> dict[str, dict[str, dict[str, str]]]:
    """Return the AnnData ``uns`` entries that declare assay membership."""
    if not declared:
        return {}
    return {SCARF_UNS_KEY: {MEMBERSHIP_UNS_KEY: dict(declared)}}


def parse_membership_declaration(declared: Mapping[Any, Any]) -> dict[str, str]:
    """Check and return the assay membership that an AnnData ``uns`` declares."""
    parsed: dict[str, str] = {}
    for assay, column in declared.items():
        if not isinstance(assay, str) or column != assay_membership_column(assay):
            raise ValueError(
                f"uns['{SCARF_UNS_KEY}']['{MEMBERSHIP_UNS_KEY}'] maps {assay!r} to "
                f"{column!r}; Scarf declares each assay's membership column as "
                "'<assay>_I'. Remove the entry from the file or correct it."
            )
        parsed[assay] = column
    return parsed


def checked_membership_values(
    values: np.ndarray,
    missing: np.ndarray,
    *,
    assay: str,
) -> np.ndarray:
    """Return declared membership values as booleans after checking them."""
    column = assay_membership_column(assay)
    array = np.asarray(values)
    if array.dtype.kind != "b":
        raise ValueError(
            f"Column {column!r} is declared as the membership of assay {assay!r}, "
            f"but it has dtype {array.dtype}, not bool"
        )
    if np.asarray(missing, dtype=bool).any():
        raise ValueError(
            f"Column {column!r} is declared as the membership of assay {assay!r}, "
            "but it has missing values"
        )
    return array.astype(bool, copy=False)


def reserved_membership_columns(assays: Iterable[str]) -> dict[str, str]:
    """Return the membership column name of each assay, mapped to its assay."""
    return {assay_membership_column(assay): assay for assay in assays}


def count_unmeasured_cells_with_counts(table: Any, assay: str, column: str) -> int:
    """Return how many cells outside ``assay``'s membership have its counts."""
    totals_name, positives_name = f"{assay}_nCounts", f"{assay}_nFeatures"
    for name in (totals_name, positives_name):
        if name not in table.columns:
            raise ValueError(
                f"The cell table has no {name!r} column to check membership "
                f"column {column!r} against. Open the store as a DataStore with "
                "zarr_mode='r+' once so that its assays are prepared."
            )
    totals = table._get_array(totals_name)
    # A row whose counts cancel to a zero total still has a positive entry.
    positives = table._get_array(positives_name)
    count = 0
    start = 0
    for members in iter_metadata_column_blocks(table, column):
        stop = start + len(members)
        outside = ~np.asarray(members, dtype=bool)
        if outside.any():
            counted = (np.asarray(totals[start:stop]) != 0) | (
                np.asarray(positives[start:stop]) != 0
            )
            count += int(np.count_nonzero(counted[outside]))
        start = stop
    return count


def _measured_cells_remedy(
    remedy: MeasuredCellsRemedy | None,
    assay: str,
    column: str,
    cell_key: str | None,
) -> str:
    """Return the sentence that tells a refused caller how to select cells."""
    measured = f"select_measured_cells({assay!r}, cell_selection=...)"
    if remedy is None:
        return f"Pass a selection of measured cells from `{measured}`."
    if remedy == "labels":
        return (
            "Narrow the labels to measured cells with "
            f"`snapshot_cluster_labels(labels, cell_selection={measured})`."
        )
    if remedy == "graph":
        return f"Build the graph over a selection of measured cells from `{measured}`."
    if remedy == "cell_selection":
        return (
            f"Pass `cell_selection={measured}` to read only the cells that it measured."
        )
    if remedy == "cell_key":
        # The membership column alone would also select cells outside the
        # caller's column, such as cells that a filter removed from I.
        key = "I" if cell_key is None else cell_key
        name = f"{assay}_measured"
        return (
            "Pass as cell_key a boolean cell column that is True only for the "
            f"cells of {key!r} that it measured, such as {name!r} after "
            f"`ds.cells.insert({name!r}, ds.cells.fetch_all({key!r}) & "
            f"ds.cells.fetch_all({column!r}))`."
        )
    if remedy == "export":
        return (
            "Export only the cells that it measured, such as those of a subset "
            f"that `SubsetZarr(..., cell_key={column!r})` writes, or export "
            f"assay {assay!r} with `to_h5ad` or with `to_anndata(from_assay="
            f"{assay!r})` without layers, which declare which cells it measured."
        )
    raise ValueError(f"Unknown measured-cells remedy {remedy!r}")


class UnmeasuredCellsError(ValueError):
    """An operation would read an assay over cells that the assay did not measure.

    Attributes:
        operation: The refusing operation.
        assay: The assay that the operation reads.
        column: The assay's membership column, ``<assay>_I``.
        unmeasured: How many of the selected cells the assay did not measure.
        selected: How many cells the operation would read.
        remedy: The input that the message tells the caller to narrow.
        cell_key: The boolean cell column whose cells the operation reads, or None.
    """

    def __init__(
        self,
        operation: str,
        assay: str,
        column: str,
        unmeasured: int,
        selected: int,
        remedy: MeasuredCellsRemedy | None = None,
        cell_key: str | None = None,
    ) -> None:
        self.operation = operation
        self.assay = assay
        self.column = column
        self.unmeasured = unmeasured
        self.selected = selected
        self.remedy = remedy
        self.cell_key = cell_key
        super().__init__(
            f"`{operation}` reads assay {assay!r} over {unmeasured} of {selected} "
            f"selected cells that it did not measure: cell column {column!r} is "
            f"False for them. "
            f"{_measured_cells_remedy(remedy, assay, column, cell_key)}"
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.operation,
                self.assay,
                self.column,
                self.unmeasured,
                self.selected,
                self.remedy,
                self.cell_key,
            ),
        )


def _strictly_increasing(rows: np.ndarray) -> bool:
    return rows.size < 2 or bool(np.all(rows[1:] > rows[:-1]))


def _cell_rows(rows: Any, n_cells: int) -> np.ndarray:
    """Return integer cell rows as int64 after checking them.

    Raises:
        ValueError: If ``rows`` is not one-dimensional.
        TypeError: If ``rows`` does not hold integers.
        IndexError: If a row is negative or not below ``n_cells``.
    """
    indices = np.asarray(rows)
    if indices.ndim != 1:
        raise ValueError("Cell rows must be one-dimensional")
    if indices.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Cell rows must be integers or a boolean mask")
    indices = indices.astype(np.int64, copy=False)
    if int(indices.min()) < 0 or int(indices.max()) >= n_cells:
        raise IndexError(f"Cell rows are out of bounds for a table of {n_cells} cells")
    return indices


def _checked_cells(table: Any, cells: Any) -> Any:
    """Return ``cells`` as a boolean mask over every cell, or as checked rows.

    The name of a boolean cell column gives the column's array. A boolean
    mask, such as the stored values of a cell-selection artifact, must have
    one entry per cell; it is read later in bounded blocks.
    """
    if isinstance(cells, str):
        return table._bool_array(cells)
    if not hasattr(cells, "dtype"):
        cells = np.asarray(cells)
    if np.dtype(cells.dtype).kind != "b":
        return _cell_rows(cells, table.N)
    if tuple(cells.shape) != (table.N,):
        raise ValueError(
            f"A cell mask needs one entry per cell ({table.N}); got shape "
            f"{tuple(cells.shape)}"
        )
    return cells


def _dense(rows: np.ndarray, n_cells: int) -> bool:
    return rows.size * _DENSE_SELECTION >= n_cells


def _membership_at(table: Any, column: str, rows: np.ndarray) -> np.ndarray:
    """Read the membership of each of the checked ``rows``, in their order.

    A dense selection streams the column once into a copy of one byte per
    cell. Otherwise only the chunks that hold the rows are read, and rows
    that are not strictly increasing are read once each.
    """
    if rows.size == 0:
        return np.zeros(0, dtype=bool)
    if _dense(rows, table.N):
        members = np.empty(table.N, dtype=bool)
        start = 0
        for block in iter_metadata_column_blocks(table, column):
            members[start : start + len(block)] = block
            start += len(block)
        gathered: np.ndarray = members[rows]
        return gathered
    if _strictly_increasing(rows):
        return np.asarray(read_metadata_rows_chunkwise(table, column, rows), dtype=bool)
    distinct, positions = np.unique(rows, return_inverse=True)
    values = np.asarray(
        read_metadata_rows_chunkwise(table, column, distinct), dtype=bool
    )
    return values[positions]


def measured_rows(table: Any, assay: str, rows: Any) -> np.ndarray | None:
    """Return whether ``assay`` measured each cell of ``rows``.

    Returns None when it measured all of them.
    """
    column = resolve_assay_membership(table, assay)
    if column is None:
        return None
    indices = _cell_rows(rows, table.N)
    if indices.size == 0:
        return None
    members = _membership_at(table, column, indices)
    return None if bool(members.all()) else members


def _count_unmeasured(table: Any, column: str, cells: Any) -> tuple[int, int]:
    """Return how many distinct cells of ``cells`` ``column`` marks False.

    ``cells`` is checked by :func:`_checked_cells`. Returns the number of
    unmeasured cells and the number of distinct cells.
    """
    if np.dtype(cells.dtype).kind != "b":
        rows = cells
        if not _strictly_increasing(rows):
            if _dense(rows, table.N):
                # A mask of the rows counts each cell once without sorting.
                chosen = np.zeros(table.N, dtype=bool)
                chosen[rows] = True
                return _count_unmeasured(table, column, chosen)
            rows = np.unique(rows)
        members = _membership_at(table, column, rows)
        return int(np.count_nonzero(~members)), int(members.size)
    unmeasured = 0
    selected = 0
    start = 0
    # The membership column bounds each read; the mask is read at its rows.
    for members in iter_metadata_column_blocks(table, column):
        stop = start + len(members)
        chosen = np.asarray(cells[start:stop], dtype=bool)
        selected += int(np.count_nonzero(chosen))
        unmeasured += int(np.count_nonzero(chosen & ~np.asarray(members, dtype=bool)))
        start = stop
    return unmeasured, selected


def require_measured_cells(
    table: Any,
    assay: str,
    cells: Any,
    *,
    operation: str,
    remedy: MeasuredCellsRemedy | None = None,
) -> None:
    """Raise ``UnmeasuredCellsError`` if ``assay`` did not measure one of ``cells``.

    ``cells`` is a boolean cell column name, a mask with one entry per cell, or
    integer rows.
    """
    column = resolve_assay_membership(table, assay)
    if column is None:
        return
    checked = _checked_cells(table, cells)
    unmeasured, selected = _count_unmeasured(table, column, checked)
    if unmeasured:
        raise UnmeasuredCellsError(
            operation,
            assay,
            column,
            unmeasured,
            selected,
            remedy,
            cells if isinstance(cells, str) else None,
        )


def measured_selection(
    table: Any, assay: str, selected: np.ndarray
) -> tuple[str, np.ndarray, str] | None:
    """Return the cells of ``selected`` that ``assay`` measured, as a mask.

    Returns the membership column, the mask, and the column's fingerprint, or
    None when ``assay`` has no membership column.
    """
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (table.N,):
        raise ValueError(
            f"A cell mask needs one entry per cell ({table.N}); got shape {mask.shape}"
        )
    column = resolve_assay_membership(table, assay)
    if column is None:
        return None
    measured = np.zeros(table.N, dtype=bool)
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (table.N,), np.dtype(bool))
    start = 0
    for block in iter_metadata_column_blocks(table, column):
        members = np.asarray(block, dtype=bool)
        stop = start + len(members)
        builder.update_array_block("values", (start,), members)
        measured[start:stop] = mask[start:stop] & members
        start = stop
    builder.end_array("values")
    return column, measured, builder.hexdigest()
