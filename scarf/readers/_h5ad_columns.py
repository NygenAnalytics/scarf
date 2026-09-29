from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from ._text import as_text

# AnnData writes a dataframe column as a group when it needs more than one
# array: a categorical needs codes with categories, and a pandas nullable
# column (integer, boolean, or string) needs values with a missingness mask.
CATEGORICAL_KEYS = frozenset({"codes", "categories"})
NULLABLE_KEYS = frozenset({"values", "mask"})
SPARSE_KEYS = frozenset({"data", "indices", "indptr"})

type H5adNode = h5py.Group | h5py.Dataset


def column_encoding(node: H5adNode) -> str:
    """Return the AnnData ``encoding-type`` of a node, or ``unknown``."""
    encoding = node.attrs.get("encoding-type")
    return "unknown" if encoding is None else as_text(encoding)


def is_categorical(node: Any) -> bool:
    return isinstance(node, h5py.Group) and CATEGORICAL_KEYS.issubset(node.keys())


def is_nullable(node: Any) -> bool:
    return isinstance(node, h5py.Group) and NULLABLE_KEYS.issubset(node.keys())


def is_column(node: Any) -> bool:
    """Return whether a dataframe child holds values that can be decoded."""
    return isinstance(node, h5py.Dataset) or is_categorical(node) or is_nullable(node)


@dataclass(frozen=True, slots=True)
class H5adTableMembers:
    """Children of an H5AD dataframe group, found without reading values.

    Attributes:
        members: Source name and node of each child. Names listed in
            ``column-order`` come first, in that order, followed by direct
            children the attribute does not list, such as the index.
        unresolved: Names listed in ``column-order`` that name no child.
    """

    members: tuple[tuple[str, H5adNode], ...]
    unresolved: tuple[str, ...]


def column_order(table: h5py.Group) -> tuple[str, ...] | None:
    """Return the column names an AnnData dataframe lists in ``column-order``.

    Reading the attribute touches no dataset values. AnnData stores an empty
    table's attribute as an empty array without a string dtype.

    Returns:
        The listed names without repeats, or None when the attribute is absent.

    Raises:
        ValueError: If the attribute holds values that are not text.
    """
    raw = table.attrs.get("column-order")
    if raw is None:
        return None
    values = np.asarray(raw).reshape(-1)
    names = []
    for value in values.tolist():
        if not isinstance(value, str | bytes):
            raise ValueError(
                f"H5AD table {table.name!r} has a column-order attribute that "
                "does not list column names"
            )
        names.append(as_text(value))
    return tuple(dict.fromkeys(names))


def _is_member_path(name: str) -> bool:
    # h5py resolves a leading "/" from the file root and "." as the group itself.
    return not name.startswith("/") and all(
        part not in {"", ".", ".."} for part in name.split("/")
    )


def table_members(table: h5py.Group) -> H5adTableMembers:
    """Resolve the children of an AnnData dataframe group by source name.

    Old AnnData versions wrote a column whose name contains ``/`` as nested
    HDF5 groups, while ``column-order`` kept the full name. Resolving each
    listed name as an HDF5 path finds such a column. The index, which
    ``column-order`` leaves out, is resolved the same way through the
    ``_index`` attribute, with or without ``column-order``. Other direct
    children follow; a group that only holds the nested levels of a resolved
    name is not reported. This reads one attribute and each member's object
    header, never column values.

    Args:
        table: The dataframe group, such as ``obs`` or ``var``.

    Returns:
        The resolved members and the listed names that resolve to nothing.
    """
    order = column_order(table)
    listed: tuple[str, ...] = () if order is None else order
    members: list[tuple[str, H5adNode]] = []
    unresolved: list[str] = []
    for name in listed:
        node = table.get(name) if _is_member_path(name) else None
        if isinstance(node, h5py.Group | h5py.Dataset):
            members.append((name, node))
        else:
            unresolved.append(name)
    index = index_key(table)
    if index is not None and "/" in index and index not in listed:
        node = table.get(index) if _is_member_path(index) else None
        if isinstance(node, h5py.Group | h5py.Dataset):
            members.append((index, node))
            listed = (*listed, index)
    resolved = set(listed)
    levels = {
        name.rsplit("/", depth)[0]
        for name in resolved
        for depth in range(1, name.count("/") + 1)
    }
    for name in table.keys():
        if name in resolved:
            continue
        node = table[name]
        # A decodable column keeps its place even when a listed path runs
        # through it. A group of nested levels is dropped only when resolved
        # names cover everything in it; otherwise the reader reports it.
        if name in levels and not is_column(node):
            if _holds_only(node, name, resolved, levels):
                continue
        members.append((name, node))
    return H5adTableMembers(tuple(members), tuple(unresolved))


def _holds_only(
    group: h5py.Group,
    path: str,
    resolved: set[str],
    levels: set[str],
) -> bool:
    """Return whether resolved names cover every member under ``group``."""
    for child in group.keys():
        child_path = f"{path}/{child}"
        if child_path in resolved:
            continue
        node = group[child]
        if child_path in levels and isinstance(node, h5py.Group):
            if not is_column(node) and _holds_only(node, child_path, resolved, levels):
                continue
        return False
    return True


def is_table_column(node: Any) -> bool:
    """Return whether a dataframe member imports as one metadata column.

    A dataset holding more than one dimension decodes but is not imported.
    """
    if isinstance(node, h5py.Dataset):
        return int(node.ndim) == 1
    return is_column(node)


def table_column_names(table: Any) -> list[str]:
    """Return the source names of a dataframe's decodable columns.

    A compound dataset, as AnnData 0.6 wrote, lists its fields. A group lists
    the members that :func:`is_table_column` accepts, in :func:`table_members`
    order.
    """
    if isinstance(table, h5py.Dataset):
        return list(table.dtype.names or ())
    if not isinstance(table, h5py.Group):
        return []
    return [
        name for name, node in table_members(table).members if is_table_column(node)
    ]


def column_length(node: Any) -> int | None:
    """Return the number of rows of a decodable dataframe column."""
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.shape else None
    if is_categorical(node):
        return int(node["codes"].shape[0])
    if is_nullable(node):
        return int(node["values"].shape[0])
    return None


def index_key(table: Any) -> str | None:
    """Return the dataframe index name that AnnData records in ``_index``."""
    if not isinstance(table, h5py.Group):
        return None
    value = table.attrs.get("_index")
    return None if value is None else as_text(value)


def decode_categories(
    codes: np.ndarray,
    categories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map category codes to labels; negative or unknown codes are missing."""
    codes = np.asarray(codes)
    categories = np.asarray(categories)
    valid = (codes >= 0) & (codes < len(categories))
    values = np.empty(codes.shape, dtype=object)
    values[valid] = categories[codes[valid]]
    values[~valid] = None
    return values, ~valid


def read_column(
    node: H5adNode,
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one dataframe column slice and its missing-value mask.

    Categorical labels come back as an object array holding ``None`` in
    missing rows. Pandas nullable columns keep their stored values and dtype;
    the mask flags the rows that are missing.

    Raises:
        TypeError: If the node uses an encoding that cannot be decoded.
        ValueError: If a nullable column's mask does not align with its values.
    """
    selection = slice(start, stop)
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[selection])
        return values, np.zeros(values.shape, dtype=bool)
    if is_categorical(node):
        return decode_categories(node["codes"][selection], node["categories"][:])
    if is_nullable(node):
        values = np.asarray(node["values"][selection])
        missing = np.asarray(node["mask"][selection], dtype=bool)
        if missing.shape != values.shape:
            raise ValueError(
                f"Column {node.name!r} has a missing-value mask that does not "
                "align with its values"
            )
        return values, missing
    raise TypeError(
        f"Column {node.name!r} uses the unsupported H5AD encoding "
        f"{column_encoding(node)!r}"
    )


def legacy_categories(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
) -> np.ndarray | None:
    """Return categories that old AnnData files store apart from their codes.

    AnnData 0.7 kept them in a ``__categories`` group beside the columns and
    AnnData 0.6 in ``uns/<key>_categories``.
    """
    if isinstance(table, h5py.Group):
        for name in category_groups:
            group = table.get(name)
            if isinstance(group, h5py.Group) and key in group:
                return np.asarray(group[key][:])
    uns = h5.get("uns")
    if isinstance(uns, h5py.Group) and f"{key}_categories" in uns:
        return np.asarray(uns[f"{key}_categories"][:])
    return None


def read_table_column(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one column of a dataframe group or of an old compound dataset."""
    if isinstance(table, h5py.Dataset):
        fields = table.dtype.fields
        if fields is None or key not in fields:
            raise KeyError(f"Column {key!r} was not found in {table.name!r}")
        values = np.asarray(table.fields(key)[start:stop])
        missing = np.zeros(values.shape, dtype=bool)
    else:
        node = table[key]
        values, missing = read_column(node, start, stop)
        if not isinstance(node, h5py.Dataset):
            return values, missing
    categories = legacy_categories(h5, table, key, category_groups)
    if categories is not None and values.dtype.kind in "iu":
        return decode_categories(values, categories)
    return values, missing


def table_column_dtype(
    h5: h5py.File,
    table: H5adNode,
    key: str,
    category_groups: tuple[str, ...],
) -> np.dtype[Any]:
    """Return the dtype of the labels or values a table column decodes to."""
    source: np.dtype[Any]
    if isinstance(table, h5py.Dataset):
        fields = table.dtype.fields
        if fields is None or key not in fields:
            raise KeyError(f"Column {key!r} was not found in {table.name!r}")
        source = np.dtype(fields[key][0])
    else:
        node = table[key]
        if is_categorical(node):
            source = np.dtype(node["categories"].dtype)
            return source
        if is_nullable(node):
            source = np.dtype(node["values"].dtype)
            return source
        source = np.dtype(node.dtype)
    categories = legacy_categories(h5, table, key, category_groups)
    if categories is not None and source.kind in "iu":
        source = np.dtype(categories.dtype)
    return source


def present_column(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Show missing rows as NaN in numeric columns and as None otherwise."""
    if not bool(missing.any()):
        return values
    if values.dtype.kind in "iuf":
        decoded = values.astype(np.float64)
        decoded[missing] = np.nan
        return decoded
    decoded = values.astype(object)
    decoded[missing] = None
    return decoded


def sparse_encoding(group: h5py.Group) -> str | None:
    """Return ``csr`` or ``csc`` for a sparse matrix group, or None.

    AnnData records the layout in ``encoding-type``; files written by the
    older h5sparse layout record it in ``h5sparse_format``.
    """
    encoding = group.attrs.get("encoding-type")
    if encoding is None:
        encoding = group.attrs.get("h5sparse_format")
    if encoding is None:
        return None
    normalized = as_text(encoding).lower()
    if normalized in {"csr", "csr_matrix"}:
        return "csr"
    if normalized in {"csc", "csc_matrix"}:
        return "csc"
    return None


def sparse_shape(group: h5py.Group) -> tuple[int, int] | None:
    """Return the stored ``shape`` (or h5sparse ``h5sparse_shape``) attribute."""
    shape: Any = group.attrs.get("shape")
    if shape is None:
        shape = group.attrs.get("h5sparse_shape")
    if shape is None:
        return None
    values = np.asarray(shape).reshape(-1)
    if values.size != 2:
        return None
    return int(values[0]), int(values[1])
