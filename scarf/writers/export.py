"""Export assays and completed pipeline runs to H5AD and Matrix Market files.

Every H5AD export is an :class:`H5adExportPlan`: the matrices, columns, and
embeddings that the file holds, each read only while it is written.
:func:`live_h5ad_plan` plans the export of a complete assay with its live
metadata, and a ``DataStore`` plans the export of a completed pipeline run.
:func:`write_h5ad_plan` writes either one with h5py, row block by row block,
so it never holds a complete matrix and needs no AnnData.
:func:`materialize_h5ad_matrix` and :func:`h5ad_frame` build the in-memory
form of the same plan, which ``DataStore.to_anndata`` returns.
"""

import os
import re
import secrets
import stat
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack

from ..metadata.rows import apply_missing_mask, metadata_missing_mask
from ..utils.compute import compute_with_progress
from ..utils.logging import logger

if TYPE_CHECKING:
    import pandas as pd

type H5adText = Literal["string", "categorical"]

# Element storage of a streamed CSR matrix.
_CSR_CHUNK = 65_536
# Value kinds that hold text: strings, bytes, objects, and NumPy strings.
_TEXT_KINDS = frozenset("USOT")


@dataclass(frozen=True, slots=True)
class H5adMatrix:
    """A cells-by-features matrix that an H5AD export reads in row blocks."""

    shape: tuple[int, int]
    dtype: np.dtype[Any]
    blocks: Callable[[], Iterator[csr_matrix | np.ndarray]]
    reserve: Callable[[], int] | None = None


@dataclass(frozen=True, slots=True)
class H5adColumn:
    """One ``obs`` or ``var`` column of an H5AD export, or the index of either."""

    name: str
    read: Callable[[], tuple[np.ndarray, np.ndarray | None]]
    text: H5adText = "string"


@dataclass(frozen=True, slots=True)
class H5adExportPlan:
    """Everything that one H5AD file holds, read while it is written."""

    x: H5adMatrix
    obs_index: H5adColumn
    obs: tuple[H5adColumn, ...]
    var_index: H5adColumn
    var: tuple[H5adColumn, ...]
    obsm: Mapping[str, Callable[[], np.ndarray]] = field(default_factory=dict)
    layers: Mapping[str, H5adMatrix] = field(default_factory=dict)
    membership: Mapping[str, str] = field(default_factory=dict)


# Dense values that one conversion step turns into CSR. A dense row block is
# converted a step of rows at a time, and each step is written before the
# next is converted, so the conversion holds a bounded number of bytes
# whatever the size of the block.
_CSR_STEP_VALUES = 1 << 20


def _csr_step_rows(n_cols: int) -> int:
    """Return the rows of one conversion step: a step of values, or one row."""
    return max(1, _CSR_STEP_VALUES // max(1, int(n_cols)))


def _csr_index_dtype(n_cols: int, step_values: int) -> np.dtype[Any]:
    """Return the dtype of a piece's column indices and row offsets.

    SciPy keeps int32 indexes that it can hold as they are, and would copy
    int64 indexes that fit into int32.
    """
    limit = int(np.iinfo(np.int32).max)
    return np.dtype(
        np.int32 if max(int(n_cols), int(step_values)) <= limit else np.int64
    )


def h5ad_conversion_bytes(dtype: Any, n_cols: int, block_rows: int) -> int:
    """Return the bytes held while converting dense row blocks to CSR."""
    rows = min(_csr_step_rows(n_cols), int(block_rows))
    if rows <= 0:
        return 0
    values = rows * max(1, int(n_cols))
    itemsize = int(np.dtype(dtype).itemsize)
    index_bytes = _csr_index_dtype(n_cols, values).itemsize
    int64 = np.dtype(np.int64).itemsize
    per_value = 2 * itemsize + int64 + index_bytes
    per_row = 2 * int64 + index_bytes
    return values * per_value + (rows + 1) * per_row


def largest_block_rows(matrix: Any) -> int:
    """Return the rows of the largest block that ``matrix`` streams."""
    rows = int(matrix.shape[0])
    chunks = getattr(matrix, "chunksize", None)
    return min(rows, int(chunks[0])) if rows and chunks is not None else rows


def _dense_csr_pieces(block: np.ndarray) -> Iterator[csr_matrix]:
    """Yield a dense row block as CSR pieces of one conversion step each.

    Only nonzero values are stored, in row-major order, as
    ``csr_matrix(block)`` stores them, and each piece holds what
    :func:`h5ad_conversion_bytes` bounds. A block without rows yields
    nothing.
    """
    n_rows, n_cols = block.shape
    step = _csr_step_rows(n_cols)
    index_dtype = _csr_index_dtype(n_cols, step * max(1, n_cols))
    for start in range(0, n_rows, step):
        rows = block[start : start + step]
        n_step = int(rows.shape[0])
        # A view of C-contiguous rows; a copy of the step otherwise.
        flat = np.ravel(rows)
        positions = np.flatnonzero(flat)
        data = flat[positions]
        del flat
        row_starts = np.arange(1, n_step + 1, dtype=np.int64)
        row_starts *= n_cols
        indptr = np.empty(n_step + 1, dtype=index_dtype)
        indptr[0] = 0
        indptr[1:] = np.searchsorted(positions, row_starts)
        del row_starts
        np.remainder(positions, max(1, n_cols), out=positions)
        indices = positions.astype(index_dtype, copy=False)
        del positions
        yield csr_matrix(
            (data, indices, indptr), shape=(n_step, n_cols), dtype=block.dtype
        )
        del data, indices, indptr


def iter_h5ad_blocks(
    matrix: H5adMatrix,
    label: str = "The X matrix",
) -> Iterator[csr_matrix]:
    """Yield the rows of ``matrix`` as CSR pieces, checked against it."""
    n_rows, n_cols = matrix.shape
    rows = 0
    for values in matrix.blocks():
        dimensions = int(np.ndim(values))
        if dimensions != 2:
            plural = "" if dimensions == 1 else "s"
            raise ValueError(
                f"{label} has a row block with {dimensions} dimension{plural}; "
                "blocks are two-dimensional"
            )
        block = values if issparse(values) else np.asarray(values)
        if block.shape[1] != n_cols:
            raise ValueError(
                f"{label} has a row block of {block.shape[1]} columns; "
                f"the export declares {n_cols}"
            )
        if block.dtype != matrix.dtype:
            raise ValueError(
                f"{label} has a row block of dtype {block.dtype}; "
                f"the export declares {matrix.dtype}"
            )
        rows += block.shape[0]
        if rows > n_rows:
            raise ValueError(
                f"{label} has more than the {n_rows} rows that the export declares"
            )
        if issparse(block):
            yield csr_matrix(block)
        else:
            yield from _dense_csr_pieces(block)
        # The stream reads the next block while this one is held no longer.
        del values, block
    if rows != n_rows:
        raise ValueError(f"{label} has {rows} rows; the export declares {n_rows}")


def materialize_h5ad_matrix(
    matrix: H5adMatrix,
    label: str = "The X matrix",
) -> csr_matrix:
    """Return ``matrix`` as one CSR matrix, built from its checked row blocks."""
    blocks = list(iter_h5ad_blocks(matrix, label))
    if not blocks:
        return csr_matrix(matrix.shape, dtype=matrix.dtype)
    return cast(csr_matrix, vstack(blocks, format="csr"))


_DIGIT_RUNS = re.compile(r"(\d+)")


def _natural_key(text: str) -> tuple[str | int, ...]:
    """Return the key of the natural order of a run export's categories.

    The canonical decomposition (NFD) of ``text`` splits into runs of decimal
    digits, which compare as integers, and the text between them, which
    compares as strings, so ``d2`` sorts before ``d10``. A key that would
    start with a number starts with an empty string, so a number never meets
    a string at one position. A digit that is not a decimal digit, such as a
    superscript or a circled digit, is text. This order is Scarf's own: the
    run plan applies it, and AnnData keeps it when it writes the categorical
    that ``to_anndata`` returns, although natsort, which AnnData sorts the
    categories of text columns with, reads such digits as numbers.
    """
    parts: list[str | int] = [
        int(part) if index % 2 else part
        for index, part in enumerate(
            _DIGIT_RUNS.split(unicodedata.normalize("NFD", text))
        )
        if part
    ]
    if parts and isinstance(parts[0], int):
        parts.insert(0, "")
    return tuple(parts)


def _categorical_column(
    values: np.ndarray,
    missing: np.ndarray | None,
) -> "pd.Categorical | None":
    """Return a ``"categorical"`` column as the categorical it is, or None.

    This is the one rule of run exports, which both the file and the object
    that ``DataStore.to_anndata(run=...)`` returns follow: text whose values
    repeat or are missing, so that it holds fewer distinct values than rows,
    is a categorical whose categories are its distinct values as text in
    natural order (:func:`_natural_key`), with no category for a missing row
    and none at all for text that is missing in every row. A column that is
    neither numeric nor boolean and has missing rows is such text too.
    Numbers, booleans, and other text are not categoricals, and return None.
    """
    import pandas as pd

    if values.dtype.kind in "biuf":
        return None
    masked = missing if missing is not None and bool(missing.any()) else None
    if masked is None and values.dtype.kind not in _TEXT_KINDS:
        return None
    text = values.astype(str).astype(object)
    if masked is not None:
        text[masked] = None
    categorical = pd.Categorical(text)
    if len(categorical.categories) >= len(categorical):
        return None
    natural = sorted(categorical.categories, key=_natural_key)
    return categorical.reorder_categories(natural)


def _encoded(node: Any, encoding: str, version: str) -> Any:
    """Record the AnnData element encoding on an HDF5 dataset or group."""
    node.attrs["encoding-type"] = encoding
    node.attrs["encoding-version"] = version
    return node


def _write_array(group: Any, name: str, values: np.ndarray) -> None:
    """Write a numeric, boolean, or text array as an AnnData element."""
    import h5py

    if values.dtype.kind in "biufc":
        _encoded(group.create_dataset(name, data=values), "array", "0.2.0")
        return
    text = values.astype(str).astype(object)
    _encoded(
        group.create_dataset(name, data=text, dtype=h5py.string_dtype()),
        "string-array",
        "0.2.0",
    )


def _write_categorical(
    group: Any,
    name: str,
    categories: np.ndarray,
    codes: np.ndarray,
) -> None:
    """Write one unordered categorical whose code -1 marks a missing row."""
    column = group.create_group(name)
    _encoded(column, "categorical", "0.2.0")
    column.attrs["ordered"] = False
    _write_array(column, "categories", categories)
    _write_array(column, "codes", codes)


def _write_masked_column(
    group: Any,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
    text: H5adText,
) -> None:
    """Write one H5AD column whose masked rows are AnnData missing values.

    Numeric columns become float64 with NaN, boolean columns use the nullable
    boolean encoding, and other columns become categoricals whose masked rows
    have code -1. With ``text`` ``"categorical"``, a masked boolean row stores
    False, as AnnData stores it, and other columns are the categoricals of
    :func:`_categorical_column`, which :func:`_write_column` writes. With
    ``"string"``, categories are sorted, codes are int32, and a masked
    boolean row keeps its stored placeholder.
    """
    if values.dtype.kind in {"f", "i", "u"}:
        _write_array(group, name, apply_missing_mask(values, missing))
        return
    if values.dtype.kind == "b":
        column = group.create_group(name)
        _encoded(column, "nullable-boolean", "0.1.0")
        if text == "categorical":
            values = values & ~missing
        _write_array(column, "values", values)
        _write_array(column, "mask", missing)
        return
    categories, observed_codes = np.unique(
        values[~missing].astype(str),
        return_inverse=True,
    )
    codes = np.full(len(values), -1, dtype=np.int32)
    codes[~missing] = observed_codes
    _write_categorical(group, name, categories, codes)


def _read_column(
    column: H5adColumn,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one column and check that it and its missing mask have ``n_rows``."""
    raw_values, raw_missing = column.read()
    values = np.asarray(raw_values)
    if values.ndim != 1:
        raise ValueError(
            f"Column {column.name!r} has {values.ndim} dimensions; "
            "columns are one-dimensional"
        )
    if values.shape[0] != n_rows:
        raise ValueError(
            f"Column {column.name!r} has {values.shape[0]} rows; "
            f"the export declares {n_rows}"
        )
    if raw_missing is None:
        return values, None
    missing = np.asarray(raw_missing, dtype=bool)
    if missing.shape != values.shape:
        raise ValueError(
            f"Column {column.name!r} has a missing mask of shape {missing.shape}; "
            f"its values have {n_rows} rows"
        )
    return values, missing


def _write_column(group: Any, column: H5adColumn, n_rows: int) -> bool:
    """Write one column; skip one of unsupported dtype with a warning."""
    values, missing = _read_column(column, n_rows)
    if column.text == "categorical":
        categorical = _categorical_column(values, missing)
        if categorical is not None:
            _write_categorical(
                group,
                column.name,
                np.asarray(categorical.categories, dtype=object),
                np.asarray(categorical.codes),
            )
            return True
    if missing is not None and missing.any():
        _write_masked_column(group, column.name, values, missing, column.text)
        return True
    try:
        _write_array(group, column.name, values)
    except TypeError:
        logger.warning(
            f"Skipping metadata column {column.name!r} with unsupported dtype "
            f"{values.dtype}"
        )
        return False
    return True


def h5ad_frame(
    index: H5adColumn,
    columns: tuple[H5adColumn, ...],
    n_rows: int,
) -> "pd.DataFrame":
    """Return an ``obs`` or ``var`` table as an in-memory export holds it."""
    import pandas as pd

    from ..metadata.queries import missing_frame_values

    index_values, index_missing = _read_column(index, n_rows)
    if index_missing is not None and index_missing.any():
        raise ValueError(f"Index {index.name!r} holds missing values")
    data: dict[str, Any] = {}
    for column in columns:
        values, missing = _read_column(column, n_rows)
        categorical = (
            _categorical_column(values, missing)
            if column.text == "categorical"
            else None
        )
        if categorical is not None:
            data[column.name] = categorical
        elif column.text == "categorical" and values.dtype.kind in _TEXT_KINDS:
            # Text without missing rows; the file stores it as a string array.
            data[column.name] = values.astype(str)
        else:
            data[column.name] = missing_frame_values(values, missing)
    return pd.DataFrame(data, index=pd.Index(index_values, name=index.name))


def _write_frame(
    group: Any,
    index: H5adColumn,
    columns: tuple[H5adColumn, ...],
    n_rows: int,
) -> set[str]:
    """Write an ``obs`` or ``var`` dataframe; return the columns it holds."""
    values, missing = _read_column(index, n_rows)
    if missing is not None and missing.any():
        raise ValueError(f"Index {index.name!r} holds missing values")
    _write_array(group, index.name, values)
    # column-order names only written columns: AnnData reads every column it
    # names, so a skipped one would make the file unreadable.
    written = [
        column.name for column in columns if _write_column(group, column, n_rows)
    ]
    group.attrs["_index"] = index.name
    # HDF5 cannot store an empty object array; AnnData writes an empty list.
    group.attrs["column-order"] = np.array(written, dtype=object) if written else []
    _encoded(group, "dataframe", "0.2.0")
    return set(written)


def _write_csr(group: Any, matrix: H5adMatrix, label: str) -> None:
    """Append ``matrix`` to a CSR group row block by row block."""
    n_rows, n_cols = matrix.shape
    capacity = 0 if matrix.reserve is None else int(matrix.reserve())
    # The blocks define the row boundaries, even when stored QC is stale.
    indptr = group.create_dataset(
        "indptr",
        (n_rows + 1,),
        chunks=True,
        compression="gzip",
        dtype="int64",
    )
    data = group.create_dataset(
        "data",
        (capacity,),
        maxshape=(None,),
        chunks=(_CSR_CHUNK,),
        compression="gzip",
        dtype=matrix.dtype,
    )
    indices = group.create_dataset(
        "indices",
        (capacity,),
        maxshape=(None,),
        chunks=(_CSR_CHUNK,),
        compression="gzip",
        dtype="int64",
    )
    row = offset = 0
    indptr[0] = 0
    for block in iter_h5ad_blocks(matrix, label):
        end_row = row + block.shape[0]
        end = offset + block.nnz
        if end > data.shape[0]:
            data.resize((end,))
            indices.resize((end,))
        data[offset:end] = block.data
        indices[offset:end] = block.indices
        indptr[row + 1 : end_row + 1] = block.indptr[1:].astype(np.int64) + offset
        row, offset = end_row, end
        # Each piece is released before the next one is converted.
        del block
    data.resize((offset,))
    indices.resize((offset,))
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = np.array([n_rows, n_cols])


def _check_plan(plan: H5adExportPlan) -> None:
    """Reject a plan that no file can hold, before anything is written."""
    for table, index, columns in (
        ("obs", plan.obs_index, plan.obs),
        ("var", plan.var_index, plan.var),
    ):
        names = [index.name, *(column.name for column in columns)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Column {repeated[0]!r} is written twice in {table}")
    for name, layer in plan.layers.items():
        if layer.shape != plan.x.shape:
            raise ValueError(
                f"Layer {name!r} has shape {layer.shape}; X has shape {plan.x.shape}"
            )


def _write_plan(h5: Any, plan: H5adExportPlan) -> None:
    import h5py

    from ..metadata.membership import MEMBERSHIP_UNS_KEY, SCARF_UNS_KEY

    n_obs, n_vars = plan.x.shape
    _encoded(h5, "anndata", "0.1.0")
    for name in ("X", "obs", "var"):
        h5.create_group(name)
    obsm = _encoded(h5.create_group("obsm"), "dict", "0.1.0")
    _write_csr(h5["X"], plan.x, "The X matrix")
    obs_columns = _write_frame(h5["obs"], plan.obs_index, plan.obs, n_obs)
    _write_frame(h5["var"], plan.var_index, plan.var, n_vars)
    for name, read in plan.obsm.items():
        values = np.asarray(read())
        if values.ndim != 2:
            raise ValueError(
                f"obsm {name!r} has {values.ndim} dimensions; "
                "obsm arrays are two-dimensional"
            )
        if values.shape[0] != n_obs:
            raise ValueError(
                f"obsm {name!r} has {values.shape[0]} rows; the export declares {n_obs}"
            )
        _write_array(obsm, name, values)
    if plan.layers:
        layers = _encoded(h5.create_group("layers"), "dict", "0.1.0")
        for name, layer in plan.layers.items():
            _write_csr(layers.create_group(name), layer, f"Layer {name!r}")
    declared = {
        assay_name: column
        for assay_name, column in plan.membership.items()
        if column in obs_columns
    }
    if declared:
        uns = _encoded(h5.create_group("uns"), "dict", "0.1.0")
        scarf_uns = _encoded(uns.create_group(SCARF_UNS_KEY), "dict", "0.1.0")
        entries = _encoded(scarf_uns.create_group(MEMBERSHIP_UNS_KEY), "dict", "0.1.0")
        for assay_name, column in declared.items():
            _encoded(
                entries.create_dataset(
                    assay_name, data=column, dtype=h5py.string_dtype()
                ),
                "string",
                "0.2.0",
            )


def write_h5ad_plan(plan: H5adExportPlan, path: str | os.PathLike[str]) -> None:
    """Stream ``plan`` to an H5AD file at ``path``, replacing it only once complete."""
    import h5py

    _check_plan(plan)
    directory = os.path.dirname(os.path.abspath(path))
    temporary = os.path.join(directory, f".{secrets.token_hex(8)}.h5ad.tmp")
    try:
        mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None
    # A replaced file keeps its permissions, also while its data is written.
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    os.close(os.open(temporary, flags, 0o666 if mode is None else 0o600))
    try:
        with h5py.File(temporary, "w") as h5:
            _write_plan(h5, plan)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.remove(temporary)
        raise


def _embedding_groups(
    columns: list[str],
    assay_name: str,
    prefixes: list[str],
) -> dict[str, list[str]]:
    """Group live embedding columns by prefix in numeric component order."""
    components: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for column in columns:
        match = re.fullmatch(r"(.+?)(\d+)", column)
        if match is None:
            continue
        prefix, component = match.groups()
        if any(prefix.startswith(f"{assay_name}_{name}") for name in prefixes):
            components[prefix].append((int(component), column))
    return {
        prefix: [column for _component, column in sorted(items)]
        for prefix, items in components.items()
    }


def _live_column(table: Any, column: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one live metadata column and its linked missing mask."""
    mask = metadata_missing_mask(table, column)
    missing = None if mask is None else np.asarray(mask[:], dtype=bool)
    return np.asarray(table.fetch_all(column)), missing


def _live_embedding(cells: Any, columns: list[str]) -> np.ndarray:
    """Stack live embedding columns into a cells-by-components array."""
    return np.array([cells.fetch_all(column) for column in columns]).T


def _live_count_blocks(assay: Any, nthreads: int) -> Iterator[np.ndarray]:
    """Stream every row of an assay's raw counts.

    The writer's conversion of each block to CSR is charged as resident.
    """
    counts = assay.rawData
    blocks: Iterator[np.ndarray] = counts._stream_blocks(
        nthreads=nthreads,
        msg="Writing raw counts",
        prefetch=None,
        row_mask=None,
        resident_bytes=h5ad_conversion_bytes(
            counts.dtype, counts.shape[1], largest_block_rows(counts)
        ),
    )
    return blocks


def _live_count_nonzero(assay: Any, nthreads: int) -> int:
    """Count the nonzero raw counts of an assay."""
    return int(
        compute_with_progress(
            assay.rawData.count_nonzero(),
            msg="Counting nonzero entries",
            nthreads=nthreads,
        )
    )


def live_h5ad_plan(
    assay: Any,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    nthreads: int = 4,
) -> H5adExportPlan:
    """Plan the H5AD export of a complete assay with its live metadata."""
    from ..metadata.membership import exported_membership

    membership, omitted = exported_membership(assay.cells, assay.name)
    cells, feats = assay.cells, assay.feats
    if embeddings_cols is None:
        embeddings_cols = ["UMAP", "tSNE"]
    embeddings = _embedding_groups(list(cells.columns), assay.name, embeddings_cols)
    embedding_columns = {
        column for columns in embeddings.values() for column in columns
    }
    obsm: dict[str, Callable[[], np.ndarray]] = {}
    for prefix, columns in embeddings.items():
        name = prefix.lower().replace(f"{assay.name.lower()}_", "X_")
        if name in obsm:
            raise ValueError(
                f"Embedding columns of two prefixes both export as obsm[{name!r}]"
            )
        obsm[name] = partial(_live_embedding, cells, columns)

    def column(table: Any, key: str, name: str | None = None) -> H5adColumn:
        return H5adColumn(
            key if name is None else name, partial(_live_column, table, key)
        )

    return H5adExportPlan(
        x=H5adMatrix(
            shape=(int(cells.N), int(feats.N)),
            dtype=np.dtype(assay.rawData.dtype),
            blocks=partial(_live_count_blocks, assay, nthreads),
            # A preliminary pass counts the stored values to allocate.
            reserve=(
                None
                if skip_recalc_nfeats
                else partial(_live_count_nonzero, assay, nthreads)
            ),
        ),
        obs_index=column(cells, "ids", "_index"),
        obs=tuple(
            column(cells, key)
            for key in cells.columns
            if key != "ids" and key not in omitted and key not in embedding_columns
        ),
        var_index=column(feats, "ids", "_index"),
        var=tuple(
            column(feats, key, "gene_short_name" if key == "names" else None)
            for key in feats.columns
            if key != "ids"
        ),
        obsm=obsm,
        membership=membership,
    )


def to_h5ad(
    assay: Any,
    h5ad_filename: str,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    nthreads: int = 4,
    *,
    run: object | None = None,
    matrix: Literal["raw", "normed"] = "raw",
) -> None:
    """Save an assay or a completed pipeline run as an H5AD file.

    Rows that a nullable metadata column's linked missing mask flags are
    written as missing values: NaN in a float64 column for numeric columns, a
    nullable boolean for boolean columns, and a missing category for other
    columns.

    Args:
        assay: Assay to save in H5ad format
        h5ad_filename: Name for the H5ad file to be created.
        embeddings_cols: Cell-metadata column prefixes treated as embeddings
                         (for example UMAP, tSNE). When None, uses
                         ``["UMAP", "tSNE"]``. Pass an empty list to skip
                         embeddings.
        skip_recalc_nfeats: Skip a preliminary nonzero-count pass. (Default value: True)
        nthreads: Number of processing threads to use (Default value: 4)
        run: Completed pipeline run opened from the datastore that owns
             ``assay``. The frozen run selections and fields are exported.
             Consecutive frozen UMAP fields are already in ``obsm['X_umap']``
             as in ``to_anndata(run=...)``; cluster labels remain in
             ``obs['clusters']``. Live embedding columns and feature-count
             recalculation options do not apply to run export.
        matrix: ``"raw"`` counts, or ``"normed"`` values, which require ``run``.

    Returns:
        None
    """
    if matrix not in ("raw", "normed"):
        raise ValueError("matrix must be either 'raw' or 'normed'")
    if run is None:
        if matrix != "raw":
            raise ValueError(
                "matrix='normed' requires run: only a pipeline run freezes "
                "normalized values"
            )
        write_h5ad_plan(
            live_h5ad_plan(assay, embeddings_cols, skip_recalc_nfeats, nthreads),
            h5ad_filename,
        )
        logger.info(
            f"Exported {assay.cells.N} cells and {assay.feats.N} features "
            f"to {h5ad_filename}"
        )
        return None

    from ..storage.pipeline_runs import PipelineRunRecord

    if not isinstance(getattr(run, "_record", None), PipelineRunRecord):
        raise TypeError("run must be a PipelineRun")
    pipeline_run = cast(Any, run)
    pipeline_run._require_completed("H5AD export")
    owner = pipeline_run._owner
    get_assay = getattr(owner, "_get_assay", None)
    # Writers may not import the datastore, so the run's owner plans it.
    run_plan = getattr(owner, "_h5ad_run_plan", None)
    if not callable(get_assay) or not callable(run_plan):
        raise TypeError("run must be opened from a DataStore")
    if get_assay(pipeline_run.assay) is not assay:
        raise ValueError("assay must be the exact run assay owned by the run datastore")
    if embeddings_cols is not None:
        raise ValueError(
            "Run-aware export uses frozen embedding fields; "
            "embeddings_cols cannot be provided"
        )
    if skip_recalc_nfeats is not True:
        raise ValueError(
            "Run-aware export uses the frozen selection; "
            "skip_recalc_nfeats cannot be disabled"
        )
    if nthreads != 4:
        raise ValueError(
            "Run-aware export uses the datastore execution settings; "
            "nthreads cannot be overridden"
        )

    plan = run_plan(pipeline_run, matrix=matrix)
    write_h5ad_plan(plan, h5ad_filename)
    n_cells, n_features = plan.x.shape
    logger.info(
        f"Exported pipeline run {pipeline_run.run_id} with {n_cells} cells and "
        f"{n_features} features to {h5ad_filename}"
    )
    return None


def to_mtx(assay: Any, mtx_directory: str, compress: bool = False) -> None:
    """Save an assay as a Matrix Market directory.

    Args:
        assay: Scarf assay. For example: `ds.RNA`
        mtx_directory: Out directory where MTX file will be saved along with barcodes and features file
        compress: If True, then the files are compressed and saved with .gz extension, using Cell Ranger 3
                  names; ``features.tsv.gz`` then also holds a feature-type column. (Default value: False).

    Returns:
        None

    Raises:
        UnmeasuredCellsError: If the assay did not measure a cell.
    """
    import gzip

    import pandas as pd
    from scipy.sparse import coo_matrix

    from ..assay.classification import is_rna_assay_type
    from ..metadata.membership import require_measured_cells

    require_measured_cells(
        assay.cells,
        assay.name,
        np.ones(assay.cells.N, dtype=bool),
        operation="to_mtx",
        remedy="export",
    )
    if os.path.isdir(mtx_directory) is False:
        os.mkdir(mtx_directory)

    tot_counts = int(
        compute_with_progress(
            assay.rawData.count_nonzero(),
            msg="Counting nonzero entries",
            nthreads=assay.nthreads,
        )
    )
    if compress:
        barcodes_fn = "barcodes.tsv.gz"
        features_fn = "features.tsv.gz"
        matrix_path = os.path.join(mtx_directory, "matrix.mtx.gz")
    else:
        barcodes_fn = "barcodes.tsv"
        features_fn = "genes.tsv"
        matrix_path = os.path.join(mtx_directory, "matrix.mtx")
    numeric_type = (
        "integer" if np.issubdtype(assay.rawData.dtype, np.integer) else "real"
    )
    with gzip.open(matrix_path, "wt") if compress else open(matrix_path, "w") as handle:
        handle.write(
            f"%%MatrixMarket matrix coordinate {numeric_type} general\n"
            "% Generated by Scarf\n"
        )
        handle.write(f"{assay.feats.N} {assay.cells.N} {tot_counts}\n")
        s = 0
        for values in assay.rawData.stream_blocks(
            nthreads=assay.nthreads,
            msg="Writing Matrix Market counts",
        ):
            block = coo_matrix(values)
            df = pd.DataFrame(
                {
                    "col": block.col + 1,
                    "row": block.row + s + 1,
                    "d": block.data,
                }
            )
            df.to_csv(
                handle,
                sep=" ",
                header=False,
                index=False,
                mode="a",
                lineterminator="\n",
            )
            s += block.shape[0]
    assay.cells.to_pandas_dataframe(["ids"]).to_csv(
        os.path.join(mtx_directory, barcodes_fn), sep="\t", header=False, index=False
    )

    features = assay.feats.to_pandas_dataframe(["ids", "names"])
    if compress:
        # Cell Ranger 3 feature files carry a third feature-type column.
        if "feature_type" in assay.feats.columns:
            features["feature_type"] = assay.feats.fetch_all("feature_type")
        else:
            features["feature_type"] = (
                "Gene Expression" if is_rna_assay_type(assay) else assay.name
            )
    features.to_csv(
        os.path.join(mtx_directory, features_fn), sep="\t", header=False, index=False
    )
    logger.info(
        f"Exported {assay.cells.N} cells and {assay.feats.N} features "
        f"to {mtx_directory}"
    )
