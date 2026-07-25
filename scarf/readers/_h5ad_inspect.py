from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from ..utils.logging import logger
from ._assay_names import auto_name_feat_table, make_feat_table_from_types


_FEATURE_ID_KEYS = (
    "_index",
    "gene_ids",
    "gene_id",
    "ensembl_id",
    "feature_ids",
    "feature_id",
    "id",
    "index",
)
_FEATURE_NAME_KEYS = (
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "feature_name",
    "gene_short_name",
    "name",
    "index",
)
_CELL_ID_KEYS = (
    "_index",
    "index",
    "cell_id",
    "cell_ids",
    "barcode",
    "barcodes",
)
_MATRIX_COMPONENTS = frozenset({"data", "indices", "indptr"})
_NON_MATRIX_PREFIXES = (
    "obs/",
    "var/",
    "obsm/",
    "varm/",
    "obsp/",
    "varp/",
    "uns/",
    "raw/var/",
    "raw/varm/",
)


@dataclass(frozen=True)
class H5adInspectResult:
    h5adFn: str
    matrixKey: str
    matrixCandidates: tuple[str, ...]
    matrixEncoding: str
    cellAttrsKey: str
    cellIdsKey: str
    featureAttrsKey: str
    featureIdsKey: str
    featureNameKey: str
    categoryNamesKey: str
    assaySplitKey: str | None
    suggestedAssays: dict[str, int]
    layers: tuple[str, ...]
    title: str | None
    description: str | None
    nCells: int
    nFeatures: int

    def to_reader_kwargs(self) -> dict[str, Any]:
        return {
            "h5ad_fn": self.h5adFn,
            "cell_attrs_key": self.cellAttrsKey,
            "cell_ids_key": self.cellIdsKey,
            "feature_attrs_key": self.featureAttrsKey,
            "feature_ids_key": self.featureIdsKey,
            "feature_name_key": self.featureNameKey,
            "matrix_key": self.matrixKey,
            "category_names_key": self.categoryNamesKey,
        }


@dataclass(frozen=True)
class _MatrixCandidate:
    key: str
    encoding: str
    shape: tuple[int, int]
    integerLike: bool

    @property
    def isSparse(self) -> bool:
        return self.encoding in {"csr", "csc"}


def _as_text(value: Any) -> str:
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8")
    return str(value)


def _node_length(node: h5py.Group | h5py.Dataset | None) -> int | None:
    if node is None:
        return None
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.shape else None

    index_key = node.attrs.get("_index")
    if index_key is not None:
        index_name = _as_text(index_key)
        if index_name in node:
            values = node[index_name]
            if isinstance(values, h5py.Dataset) and values.shape:
                return int(values.shape[0])
            if isinstance(values, h5py.Group) and "codes" in values:
                return int(values["codes"].shape[0])

    for key in ("_index", "index"):
        if key in node:
            values = node[key]
            if isinstance(values, h5py.Dataset) and values.shape:
                return int(values.shape[0])
            if isinstance(values, h5py.Group) and "codes" in values:
                return int(values["codes"].shape[0])

    for values in node.values():
        if isinstance(values, h5py.Dataset) and values.shape:
            return int(values.shape[0])
        if (
            isinstance(values, h5py.Group)
            and "codes" in values
            and isinstance(values["codes"], h5py.Dataset)
        ):
            return int(values["codes"].shape[0])
    return None


def _sparse_encoding(group: h5py.Group) -> str | None:
    encoding = group.attrs.get("encoding-type")
    if encoding is None:
        encoding = group.attrs.get("h5sparse_format")
    if encoding is None:
        return "csr"

    normalized = _as_text(encoding).lower()
    if normalized in {"csr", "csr_matrix"}:
        return "csr"
    if normalized in {"csc", "csc_matrix"}:
        return "csc"
    return None


def _stored_shape(group: h5py.Group) -> tuple[int, int] | None:
    shape: Any = group.attrs.get("shape")
    if shape is None:
        shape = group.attrs.get("h5sparse_shape")
    if shape is None and "shape" in group and isinstance(group["shape"], h5py.Dataset):
        shape = group["shape"][:]
    if shape is None:
        return None
    values = np.asarray(shape).reshape(-1)
    if values.size != 2:
        return None
    return int(values[0]), int(values[1])


def _infer_sparse_shape(
    h5: h5py.File,
    key: str,
    group: h5py.Group,
    encoding: str,
) -> tuple[int, int]:
    stored = _stored_shape(group)
    if stored is not None:
        return stored

    compressed_axis = int(group["indptr"].shape[0] - 1)
    indices = group["indices"]
    observed_axis = int(np.max(indices[:])) + 1 if indices.shape[0] else 0
    obs_length = _node_length(h5.get("obs"))
    feature_group = h5.get("raw/var" if key.startswith("raw/") else "var")
    feature_length = _node_length(feature_group)
    if encoding == "csr":
        return compressed_axis, feature_length or observed_axis
    return obs_length or observed_axis, compressed_axis


def _is_integer_like(dataset: h5py.Dataset) -> bool:
    if np.issubdtype(dataset.dtype, np.integer):
        return True
    sample = np.asarray(dataset[: min(101, dataset.shape[0])])
    if sample.size == 0 or not np.issubdtype(sample.dtype, np.number):
        return False
    return bool(
        np.all(np.isfinite(sample))
        and np.allclose(sample, np.round(sample), rtol=0, atol=1e-8)
    )


def _dense_is_integer_like(dataset: h5py.Dataset) -> bool:
    if np.issubdtype(dataset.dtype, np.integer):
        return True
    rows = min(10, dataset.shape[0])
    columns = min(100, dataset.shape[1])
    sample = np.asarray(dataset[:rows, :columns])
    if sample.size == 0 or not np.issubdtype(sample.dtype, np.number):
        return False
    return bool(
        np.all(np.isfinite(sample))
        and np.allclose(sample, np.round(sample), rtol=0, atol=1e-8)
    )


def _is_matrix_path(key: str) -> bool:
    return not key.startswith(_NON_MATRIX_PREFIXES)


def _matrix_candidates(h5: h5py.File) -> list[_MatrixCandidate]:
    candidates: list[_MatrixCandidate] = []

    def visit(key: str, node: h5py.Group | h5py.Dataset) -> None:
        if not _is_matrix_path(key):
            return
        if isinstance(node, h5py.Group) and _MATRIX_COMPONENTS.issubset(node.keys()):
            encoding = _sparse_encoding(node)
            if encoding is None:
                logger.warning(
                    f"Ignoring sparse matrix candidate with unknown encoding: {key}"
                )
                return
            candidates.append(
                _MatrixCandidate(
                    key=key,
                    encoding=encoding,
                    shape=_infer_sparse_shape(h5, key, node, encoding),
                    integerLike=_is_integer_like(node["data"]),
                )
            )
        elif isinstance(node, h5py.Dataset) and len(node.shape) == 2:
            if not np.issubdtype(node.dtype, np.number):
                return
            candidates.append(
                _MatrixCandidate(
                    key=key,
                    encoding="dense",
                    shape=(int(node.shape[0]), int(node.shape[1])),
                    integerLike=_dense_is_integer_like(node),
                )
            )

    h5.visititems(visit)

    def rank(candidate: _MatrixCandidate) -> tuple[int, int, int, int, str]:
        # Integer-like values signal raw counts and take priority over storage
        # layout, so a dense count matrix outranks a sparse transformed layer.
        integer = 0 if candidate.integerLike else 1
        raw = 0 if candidate.key.startswith("raw/") else 1
        canonical = 0 if candidate.key == "X" else 1
        sparse = 0 if candidate.isSparse else 1
        return integer, raw, canonical, sparse, candidate.key

    return sorted(candidates, key=rank)


def _read_column(
    node: h5py.Group | h5py.Dataset,
    key: str,
) -> np.ndarray | None:
    if isinstance(node, h5py.Dataset):
        if node.dtype.names is None or key not in node.dtype.names:
            return None
        return np.asarray(node[key])
    if key not in node:
        return None

    values = node[key]
    if isinstance(values, h5py.Dataset):
        raw = np.asarray(values[:])
        for category_group_name in ("__categories", "categories"):
            if category_group_name not in node:
                continue
            category_group = node[category_group_name]
            if isinstance(category_group, h5py.Group) and key in category_group:
                if not np.issubdtype(raw.dtype, np.integer):
                    return raw
                categories = np.asarray(category_group[key][:])
                valid = (raw >= 0) & (raw < len(categories))
                decoded = np.empty(raw.shape, dtype=object)
                decoded[valid] = categories[raw[valid]]
                decoded[~valid] = None
                return decoded
        return raw

    if isinstance(values, h5py.Group) and {"codes", "categories"}.issubset(
        values.keys()
    ):
        codes = np.asarray(values["codes"][:])
        categories = np.asarray(values["categories"][:])
        valid = (codes >= 0) & (codes < len(categories))
        decoded = np.empty(codes.shape, dtype=object)
        decoded[valid] = categories[codes[valid]]
        decoded[~valid] = None
        return decoded
    return None


def _column_names(node: h5py.Group | h5py.Dataset) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return list(node.dtype.names or ())
    return [
        key
        for key in node.keys()
        if key not in {"__categories", "categories"}
        and (
            isinstance(node[key], h5py.Dataset)
            or (
                isinstance(node[key], h5py.Group)
                and {"codes", "categories"}.issubset(node[key].keys())
            )
        )
    ]


def _index_key(node: h5py.Group | h5py.Dataset) -> str | None:
    """Return the dataframe index dataset name recorded by AnnData."""
    if not isinstance(node, h5py.Group):
        return None
    index_attr = node.attrs.get("_index")
    if index_attr is None:
        return None
    return _as_text(index_attr)


def _matching_key(names: list[str], preferences: tuple[str, ...]) -> str | None:
    normalized = {name.lower(): name for name in names}
    for preferred in preferences:
        if preferred in normalized:
            return normalized[preferred]
    return None


def _is_string_column(values: np.ndarray) -> bool:
    if values.dtype.kind in {"S", "U"}:
        return True
    if values.dtype.kind != "O":
        return False
    return all(
        value is None or isinstance(value, str | bytes | np.str_ | np.bytes_)
        for value in values[:100]
    )


def _is_unique(values: np.ndarray, expected_length: int) -> bool:
    if values.ndim != 1 or len(values) != expected_length:
        return False
    normalized = np.asarray(
        [None if value is None else _as_text(value) for value in values],
        dtype=object,
    )
    return len(set(normalized.tolist())) == expected_length


def _find_cell_ids(
    node: h5py.Group | h5py.Dataset | None,
    n_cells: int,
) -> str:
    if node is None:
        return "_index"
    names = _column_names(node)
    index_key = _index_key(node)
    if index_key is not None and index_key in names:
        values = _read_column(node, index_key)
        if values is not None and _is_unique(values, n_cells):
            return index_key
    preferred = _matching_key(names, _CELL_ID_KEYS)
    if preferred is not None:
        values = _read_column(node, preferred)
        if values is not None and _is_unique(values, n_cells):
            return preferred

    candidates: list[tuple[str, bool]] = []
    for name in names:
        values = _read_column(node, name)
        if values is None or not _is_unique(values, n_cells):
            continue
        candidates.append((name, _is_string_column(values)))
    if candidates:
        candidates.sort(key=lambda item: (not item[1], item[0]))
        return candidates[0][0]

    logger.warning("No unique cell ID column found; generated IDs will be used")
    return "_index"


def _mean_text_length(values: np.ndarray) -> float:
    sample = [
        _as_text(value)
        for value in values[: min(100, len(values))]
        if value is not None
    ]
    if not sample:
        return 0
    return float(np.mean([len(value) for value in sample]))


def _find_features(
    node: h5py.Group | h5py.Dataset,
    n_features: int,
) -> tuple[str, str]:
    names = _column_names(node)
    index_key = _index_key(node)
    id_key: str | None = None
    if index_key is not None and index_key in names:
        values = _read_column(node, index_key)
        if values is not None and _is_unique(values, n_features):
            id_key = index_key
    if id_key is None:
        id_key = _matching_key(names, _FEATURE_ID_KEYS)
        if id_key is not None:
            values = _read_column(node, id_key)
            if values is None or not _is_unique(values, n_features):
                id_key = None

    name_key = _matching_key(names, _FEATURE_NAME_KEYS)
    if name_key is not None:
        values = _read_column(node, name_key)
        if (
            values is None
            or values.ndim != 1
            or len(values) != n_features
            or not _is_string_column(values)
        ):
            name_key = None

    string_columns: list[tuple[str, float, bool]] = []
    for name in names:
        values = _read_column(node, name)
        if (
            values is None
            or values.ndim != 1
            or len(values) != n_features
            or not _is_string_column(values)
        ):
            continue
        string_columns.append(
            (name, _mean_text_length(values), _is_unique(values, n_features))
        )

    if id_key is None:
        unique_columns = [column for column in string_columns if column[2]]
        if unique_columns:
            id_key = max(unique_columns, key=lambda column: (column[1], column[0]))[0]
    if name_key is None and string_columns:
        alternatives = [
            column for column in string_columns if column[0] != id_key
        ] or string_columns
        name_key = min(alternatives, key=lambda column: (column[1], column[0]))[0]

    if id_key is None and name_key is None:
        logger.warning("No feature ID or name column found; generated IDs will be used")
        return "_index", "_index"
    if id_key is None:
        id_key = name_key
    if name_key is None:
        name_key = id_key
    assert id_key is not None
    assert name_key is not None
    return id_key, name_key


def _category_names_key(
    *nodes: h5py.Group | h5py.Dataset | None,
) -> str:
    for node in nodes:
        if isinstance(node, h5py.Group) and "categories" in node:
            if isinstance(node["categories"], h5py.Group):
                return "categories"
    return "__categories"


def _read_text_scalar(
    h5: h5py.File,
    key: str,
    max_length: int,
) -> str | None:
    if key not in h5:
        return None
    node = h5[key]
    if not isinstance(node, h5py.Dataset):
        return None
    value = node[()]
    text = _as_text(value)
    return text[:max_length]


def _feature_group_for(key: str) -> str:
    return "raw/var" if key.startswith("raw/") else "var"


def _select_matrix(
    h5: h5py.File,
    candidates: list[_MatrixCandidate],
) -> tuple[_MatrixCandidate, str]:
    obs_length = _node_length(h5.get("obs"))
    lengths = {
        "raw/var": _node_length(h5.get("raw/var")),
        "var": _node_length(h5.get("var")),
    }

    for candidate in candidates:
        n_cells, n_features = candidate.shape
        if obs_length is not None and obs_length != n_cells:
            logger.warning(
                f"Ignoring matrix candidate {candidate.key}: "
                f"{n_cells} rows do not match obs length {obs_length}"
            )
            continue

        own_key = _feature_group_for(candidate.key)
        own_length = lengths[own_key]
        if own_length == n_features:
            return candidate, own_key
        if own_length is not None:
            logger.warning(
                f"Ignoring matrix candidate {candidate.key}: feature metadata "
                f"group `{own_key}` length {own_length} does not match feature "
                f"count {n_features}"
            )
            continue

        # The conventional feature group for this matrix is absent. Fall back to
        # a dimension-matched group, otherwise keep the conventional key so the
        # reader generates feature IDs rather than borrowing an unrelated table.
        other_key = "var" if own_key == "raw/var" else "raw/var"
        if lengths[other_key] == n_features:
            return candidate, other_key
        if lengths[other_key] is None:
            return candidate, own_key
        logger.warning(
            f"Ignoring matrix candidate {candidate.key}: no feature metadata "
            f"group has {n_features} rows"
        )

    raise ValueError("No matrix candidate matches the obs and var dimensions")


def inspect_h5ad(h5ad_fn: str) -> H5adInspectResult:
    """Report the matrix and metadata layout of an H5AD file.

    Args:
        h5ad_fn: Path to the H5AD file.

    Returns:
        Keys, shape, and column names needed to configure
        :class:`~scarf.readers.H5adReader`.
    """
    with h5py.File(h5ad_fn, mode="r") as h5:
        candidates = _matrix_candidates(h5)
        if not candidates:
            raise ValueError("No sparse or numeric 2D matrix found in the H5AD file")

        matrix, feature_attrs_key = _select_matrix(h5, candidates)
        n_cells, n_features = matrix.shape
        cell_node = h5.get("obs")
        feature_node = h5.get(feature_attrs_key)
        if feature_node is None or not isinstance(
            feature_node, h5py.Group | h5py.Dataset
        ):
            feature_ids_key = "_index"
            feature_name_key = "_index"
        else:
            feature_ids_key, feature_name_key = _find_features(feature_node, n_features)

        cell_ids_key = _find_cell_ids(
            cell_node if isinstance(cell_node, h5py.Group | h5py.Dataset) else None,
            n_cells,
        )
        category_names_key = _category_names_key(
            cell_node if isinstance(cell_node, h5py.Group | h5py.Dataset) else None,
            feature_node
            if isinstance(feature_node, h5py.Group | h5py.Dataset)
            else None,
        )

        assay_split_key = None
        suggested_assays: dict[str, int] = {}
        if isinstance(feature_node, h5py.Group | h5py.Dataset):
            feature_columns = _column_names(feature_node)
            assay_split_key = _matching_key(
                feature_columns, ("feature_types", "feature_type")
            )
            if assay_split_key is not None:
                values = _read_column(feature_node, assay_split_key)
                if values is None or len(values) != n_features:
                    assay_split_key = None
                else:
                    feature_types = [_as_text(value) for value in values]
                    assay_table = auto_name_feat_table(
                        make_feat_table_from_types(feature_types)
                    )
                    for assay_name in dict.fromkeys(assay_table.columns):
                        selected = assay_table[assay_name]
                        count = (
                            int(selected.loc["nFeatures"].sum())
                            if selected.ndim == 2
                            else int(selected.loc["nFeatures"])
                        )
                        suggested_assays[str(assay_name)] = count

        layers_node = h5.get("layers")
        layers = (
            tuple(sorted(layers_node.keys()))
            if isinstance(layers_node, h5py.Group)
            else ()
        )
        title = _read_text_scalar(h5, "uns/title", 500)
        description = _read_text_scalar(h5, "uns/description", 10_000)
        if description is None:
            description = _read_text_scalar(h5, "uns/citation", 10_000)

    return H5adInspectResult(
        h5adFn=h5ad_fn,
        matrixKey=matrix.key,
        matrixCandidates=tuple(candidate.key for candidate in candidates),
        matrixEncoding=matrix.encoding,
        cellAttrsKey="obs",
        cellIdsKey=cell_ids_key,
        featureAttrsKey=feature_attrs_key,
        featureIdsKey=feature_ids_key,
        featureNameKey=feature_name_key,
        categoryNamesKey=category_names_key,
        assaySplitKey=assay_split_key,
        suggestedAssays=suggested_assays,
        layers=layers,
        title=title,
        description=description,
        nCells=n_cells,
        nFeatures=n_features,
    )
