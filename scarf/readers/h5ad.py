from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._assay_names import auto_name_feat_table, make_feat_table_from_types
from ._h5ad_inspect import H5adInspectResult, _as_text, inspect_h5ad as inspect_h5ad

# AnnData writes a column as a group when it needs more than one array: a
# categorical needs codes with categories, a pandas nullable dtype needs values
# with a missingness mask.
_CATEGORICAL_KEYS = frozenset({"codes", "categories"})
_NULLABLE_KEYS = frozenset({"values", "mask"})


def _column_encoding(node: h5py.Group) -> str:
    encoding = node.attrs.get("encoding-type")
    return "unknown" if encoding is None else _as_text(encoding)


def _is_decodable_column(node: h5py.Group) -> bool:
    keys = set(node.keys())
    return _CATEGORICAL_KEYS.issubset(keys) or _NULLABLE_KEYS.issubset(keys)


@dataclass(frozen=True)
class _H5adAssayFeatures:
    featureIndexes: np.ndarray
    featureIds: np.ndarray
    featureNames: np.ndarray


class H5adReader:
    """A class to read in data from a H5ad file (h5 file with AnnData
    information).

    Args:
        h5ad_fn: Path to H5AD file
        cell_attrs_key: H5 group under which cell attributes are saved.(Default value: 'obs')
        feature_attrs_key: H5 group under which feature attributes are saved.(Default value: 'var')
        cell_ids_key: Key in `obs` group that contains unique cell IDs. By default the index will be used.
        feature_ids_key: Key in `var` group that contains unique feature IDs. By default the index will be used.
        feature_name_key: Key in `var` group that contains feature names. (Default: gene_short_name)
        matrix_key: Group where in the sparse matrix resides (default: 'X')
        category_names_key: Looks up this group and replaces the values in `var` and 'obs' child datasets with the
                            corresponding index value within this group.
        dtype: Numpy dtype of the matrix data. This dtype is enforced when streaming the data through `consume`
               method. (Default value: Automatically determined)

    Attributes:
        h5: A File object from the h5py package.
        matrixKey: Group where in the sparse matrix resides (default: 'X')
        cellAttrsKey: Group wherein the cell attributes are present
        featureAttrsKey: Group wherein the feature attributes are present
        groupCodes: Used to ensure compatibility with different AnnData versions.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        cellIdsKey: Key in `obs` group that contains unique cell IDs. By default the index will be used.
        featIdsKey: Key in `var` group that contains unique feature IDs. By default the index will be used.
        featNamesKey: Key in `var` group that contains feature names. (Default: gene_short_name)
        catNamesKey: Looks up this group and replaces the values in `var` and 'obs' child datasets with the
                     corresponding index value within this group.
        matrixDtype: dtype of the matrix containing the data (as indicated by matrix_key)
    """

    def __init__(
        self,
        h5ad_fn: str,
        cell_attrs_key: str = "obs",
        cell_ids_key: str = "_index",
        feature_attrs_key: str = "var",
        feature_ids_key: str = "_index",
        feature_name_key: str = "gene_short_name",
        matrix_key: str = "X",
        obsm_attrs_key: str = "obsm",
        category_names_key: str = "__categories",
        dtype: str | None = None,
    ) -> None:
        self.h5adFn = h5ad_fn
        self.h5: h5py.File = h5py.File(h5ad_fn, mode="r")
        self.matrixKey = matrix_key
        self.cellAttrsKey, self.featureAttrsKey, self.obsmAttrsKey = (
            cell_attrs_key,
            feature_attrs_key,
            obsm_attrs_key,
        )
        self.groupCodes: dict[str, int] = {
            self.cellAttrsKey: self._validate_group(self.cellAttrsKey),
            self.featureAttrsKey: self._validate_group(self.featureAttrsKey),
            self.obsmAttrsKey: self._validate_group(self.obsmAttrsKey),
            self.matrixKey: self._validate_group(self.matrixKey),
        }
        self.matrixOrientation = self._validate_sparse_matrix()
        self._convertedCsr: csr_matrix | None = None
        self._indptrCache: np.ndarray | None = None
        self._cumulativeRowNnz: np.ndarray | None = None
        self.nCells, self.nFeatures = (
            self._get_n(self.cellAttrsKey),
            self._get_n(self.featureAttrsKey),
        )
        self.cellIdsKey = self._fix_name_key(self.cellAttrsKey, cell_ids_key)
        self.featIdsKey = self._fix_name_key(self.featureAttrsKey, feature_ids_key)
        self.featNamesKey = feature_name_key
        self.catNamesKey = category_names_key
        self.sourceMatrixDtype: Any = self._get_matrix_dtype()
        self.matrixDtype: Any = self.sourceMatrixDtype if dtype is None else dtype
        self.storageDtype: Any = self.matrixDtype
        self._dtypeOverridden = dtype is not None

    def _clone_kwargs(self) -> dict[str, Any]:
        return {
            "h5ad_fn": self.h5adFn,
            "cell_attrs_key": self.cellAttrsKey,
            "cell_ids_key": self.cellIdsKey,
            "feature_attrs_key": self.featureAttrsKey,
            "feature_ids_key": self.featIdsKey,
            "feature_name_key": self.featNamesKey,
            "matrix_key": self.matrixKey,
            "obsm_attrs_key": self.obsmAttrsKey,
            "category_names_key": self.catNamesKey,
            "dtype": self.matrixDtype if self._dtypeOverridden else None,
        }

    def open_clone(self) -> "H5adReader":
        """Open an independent h5py handle on the same file."""
        clone = type(self)(**self._clone_kwargs())
        clone.storageDtype = self.storageDtype
        clone.matrixDtype = self.matrixDtype
        clone.sourceMatrixDtype = self.sourceMatrixDtype
        if self._convertedCsr is not None:
            clone._convertedCsr = self._convertedCsr
        if self._indptrCache is not None:
            clone._indptrCache = self._indptrCache
        if self._cumulativeRowNnz is not None:
            clone._cumulativeRowNnz = self._cumulativeRowNnz
        return clone

    @classmethod
    def from_inspect(
        cls,
        inspection: H5adInspectResult,
        **overrides: Any,
    ) -> "H5adReader":
        reader_kwargs = inspection.to_reader_kwargs()
        reader_kwargs.update(overrides)
        return cls(**reader_kwargs)

    def _validate_sparse_matrix(self) -> str:
        if self.groupCodes[self.matrixKey] != 2:
            return "dense"

        group = self.h5[self.matrixKey]
        if not isinstance(group, h5py.Group):
            return "dense"

        required = {"data", "indices", "indptr"}
        missing = required.difference(group.keys())
        if missing:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` is missing: "
                f"{', '.join(sorted(missing))}"
            )

        encoding = group.attrs.get("encoding-type")
        if encoding is None:
            encoding = group.attrs.get("h5sparse_format")
        if encoding is None:
            logger.warning(
                f"Sparse matrix group `{self.matrixKey}` has no sparse encoding; "
                "assuming legacy CSR encoding"
            )
            return "csr"
        if isinstance(encoding, bytes | np.bytes_):
            encoding = encoding.decode("utf-8")
        normalized = str(encoding).lower()
        if normalized in {"csr", "csr_matrix"}:
            return "csr"
        if normalized in {"csc", "csc_matrix"}:
            return "csc"
        raise ValueError(
            f"ERROR: Sparse matrix encoding `{encoding}` is not supported. "
            "H5adReader supports CSR and CSC encoding."
        )

    def _validate_group(self, group: str) -> int:
        if group not in self.h5:
            logger.warning(f"`{group}` group not found in the H5ad file")
            ret_val = 0
        elif isinstance(self.h5[group], h5py.Dataset):
            ret_val = 1
        elif isinstance(self.h5[group], h5py.Group):
            ret_val = 2
        else:
            logger.warning(
                f"`{group}` slot in H5ad file is not of Dataset or Group type. "
                f"Due to this, no information in `{group}` can be used"
            )
            ret_val = 0
        if ret_val == 2:
            if len(self.h5[group].keys()) == 0:
                logger.warning(f"`{group}` slot in H5ad file is empty.")
                ret_val = 0
            elif (
                len(
                    set(
                        [
                            self.h5[group][x].shape[0]
                            for x in self.h5[group].keys()
                            if isinstance(self.h5[group][x], h5py.Dataset)
                        ]
                    )
                )
                > 1
            ):
                if sorted(self.h5[group].keys()) != ["data", "indices", "indptr"]:
                    logger.warning(
                        f"`{group}` slot in H5ad file has unequal sized child groups"
                    )
        return ret_val

    def _get_matrix_dtype(self) -> Any:
        if self.groupCodes[self.matrixKey] == 1:
            return self.h5[self.matrixKey].dtype
        elif self.groupCodes[self.matrixKey] == 2:
            return self.h5[self.matrixKey]["data"].dtype
        else:
            raise ValueError(
                f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
            )

    def _matrix_shape(self) -> tuple[int, int]:
        matrix = self.h5[self.matrixKey]
        if isinstance(matrix, h5py.Dataset):
            return int(matrix.shape[0]), int(matrix.shape[1])

        shape: Any = matrix.attrs.get("shape")
        if shape is None:
            shape = matrix.attrs.get("h5sparse_shape")
        if shape is None and "shape" in matrix:
            shape_node = matrix["shape"]
            if isinstance(shape_node, h5py.Dataset):
                shape = shape_node[:]
        if shape is not None:
            values = np.asarray(shape).reshape(-1)
            if values.size == 2:
                return int(values[0]), int(values[1])

        compressed_axis = int(matrix["indptr"].shape[0] - 1)
        indices = matrix["indices"]
        observed_axis = int(np.max(indices[:])) + 1 if indices.shape[0] else 0
        if self.matrixOrientation == "csr":
            return compressed_axis, observed_axis
        return observed_axis, compressed_axis

    def _check_exists(self, group: str, key: str) -> bool:
        if group in self.groupCodes:
            group_code = self.groupCodes[group]
        else:
            group_code = self._validate_group(group)
            self.groupCodes[group] = group_code
        if group_code == 1:
            if key in list(self.h5[group].dtype.names):
                return True
        if group_code == 2:
            if key in self.h5[group].keys():
                return True
        return False

    def _fix_name_key(self, group: str, key: str) -> str:
        if self._check_exists(group, key) is False:
            if key.startswith("_"):
                temp_key = key[1:]
                if self._check_exists(group, temp_key):
                    return temp_key
        return key

    def _get_n(self, group: str) -> int:
        if self.groupCodes[group] == 0:
            matrix_shape = self._matrix_shape()
            return matrix_shape[0 if group == self.cellAttrsKey else 1]
        elif self.groupCodes[group] == 1:
            return int(self.h5[group].shape[0])
        else:
            for i in self.h5[group].keys():
                node = self.h5[group][i]
                if isinstance(node, h5py.Dataset):
                    return int(node.shape[0])
                if not isinstance(node, h5py.Group):
                    continue
                # Group encoded columns carry the axis length in their codes
                # (categorical) or values (pandas nullable) array.
                if "codes" in node:
                    return int(node["codes"].shape[0])
                if _NULLABLE_KEYS.issubset(node.keys()):
                    return int(node["values"].shape[0])
            raise KeyError(
                f"ERROR: `{group}` key doesn't contain any child node of Dataset type."
                f"Aborting because unexpected H5ad format."
            )

    def cell_ids(self) -> np.ndarray:
        """Returns a list of cell IDs."""
        if self._check_exists(self.cellAttrsKey, self.cellIdsKey):
            values = self.h5[self.cellAttrsKey][self.cellIdsKey]
            return self._replace_category_values(
                values, self.cellIdsKey, self.cellAttrsKey
            ).astype(object)
        logger.warning(
            f"Cell ID key {self.cellIdsKey!r} was not found in H5AD obs; "
            "generated IDs will be used"
        )
        return np.array([f"cell_{x}" for x in range(self.nCells)])

    # noinspection DuplicatedCode
    def feat_ids(self) -> np.ndarray:
        """Returns a list of feature IDs."""
        if self._check_exists(self.featureAttrsKey, self.featIdsKey):
            values = self.h5[self.featureAttrsKey][self.featIdsKey]
            return self._replace_category_values(
                values, self.featIdsKey, self.featureAttrsKey
            ).astype(object)
        logger.warning(
            f"Feature ID key {self.featIdsKey!r} was not found in "
            f"{self.featureAttrsKey}; generated IDs will be used"
        )
        return np.array([f"feature_{x}" for x in range(self.nFeatures)])

    # noinspection DuplicatedCode
    def feat_names(self) -> np.ndarray:
        """Returns a list of feature names."""
        if self._check_exists(self.featureAttrsKey, self.featNamesKey):
            values = self.h5[self.featureAttrsKey][self.featNamesKey]
            return self._replace_category_values(
                values, self.featNamesKey, self.featureAttrsKey
            ).astype(object)
        logger.warning(
            f"Feature name key {self.featNamesKey!r} was not found in "
            f"{self.featureAttrsKey}; feature IDs will be used"
        )
        return self.feat_ids()

    def _replace_category_values(
        self, v: np.ndarray | h5py.Group | h5py.Dataset, key: str, group: str
    ) -> np.ndarray:
        if isinstance(v, h5py.Group):
            if _CATEGORICAL_KEYS.issubset(v.keys()):
                codes = v["codes"][:]
                categories = v["categories"][:]
                valid = (codes >= 0) & (codes < len(categories))
                decoded = np.empty(codes.shape, dtype=object)
                decoded[valid] = categories[codes[valid]]
                decoded[~valid] = None
                return decoded
            if _NULLABLE_KEYS.issubset(v.keys()):
                return self._decode_nullable(v, key)
            logger.warning(
                f"Column {key!r} in {group} uses the H5AD encoding "
                f"{_column_encoding(v)!r}, which cannot be decoded"
            )
            return np.array([], dtype=object)

        # if v is a Dataset
        if isinstance(v, h5py.Dataset):
            v = v[:]

        if self.catNamesKey is not None:
            if self._check_exists(group, self.catNamesKey):
                cat_g = self.h5[group][self.catNamesKey]
                if isinstance(cat_g, h5py.Group):
                    if key in cat_g:
                        return self._decode_legacy_categories(v, cat_g[key][:])
        if "uns" in self.h5:
            if key + "_categories" in self.h5["uns"]:
                categories = self.h5["uns"][key + "_categories"][:]
                return self._decode_legacy_categories(v, categories)
        return np.asarray(v)

    @staticmethod
    def _decode_nullable(v: h5py.Group, key: str) -> np.ndarray:
        """Decode a pandas nullable column without losing numeric semantics."""
        values = np.asarray(v["values"][:])
        mask = np.asarray(v["mask"][:])
        if mask.shape != values.shape:
            logger.warning(
                f"Column {key!r} has a missingness mask of shape {mask.shape} "
                f"for {values.shape} values; the mask will be ignored"
            )
            return values
        mask = mask.astype(bool, copy=False)
        if not mask.any():
            # Nothing is missing, so the native dtype survives the round trip.
            return values
        if values.dtype.kind in "iuf":
            decoded = values.astype(np.float64)
            decoded[mask] = np.nan
            return decoded
        decoded = np.empty(values.shape, dtype=object)
        decoded[~mask] = values[~mask]
        decoded[mask] = None
        return decoded

    @staticmethod
    def _decode_legacy_categories(
        codes: np.ndarray, categories: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(codes)
        if not np.issubdtype(values.dtype, np.integer):
            return values
        try:
            # Negative codes mark missing values in legacy AnnData categoricals;
            # they must decode to None rather than wrap to the final category.
            return np.array([None if code < 0 else categories[code] for code in values])
        except (IndexError, TypeError):
            return values

    def _get_col_data(
        self, group: str, ignore_keys: list[str]
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        if self.groupCodes[group] == 1:
            for i in iter_progress(
                self.h5[group].dtype.names,
                desc=f"Reading attributes from group {group}",
            ):
                if i in ignore_keys:
                    continue
                yield i, self._replace_category_values(self.h5[group][i][:], i, group)
        if self.groupCodes[group] == 2:
            for i in iter_progress(
                self.h5[group].keys(), desc=f"Reading attributes from group {group}"
            ):
                if i in ignore_keys:
                    continue
                values = self.h5[group][i]
                if not isinstance(values, h5py.Dataset | h5py.Group):
                    continue
                if isinstance(values, h5py.Group) and not _is_decodable_column(values):
                    logger.warning(
                        f"Skipping {group} column {i!r} because its H5AD encoding "
                        f"{_column_encoding(values)!r} is not supported"
                    )
                    continue
                yield (
                    i,
                    self._replace_category_values(values, i, group),
                )

    def _get_obsm_data(
        self, group: str
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        if self.groupCodes[group] == 2:
            for i in iter_progress(
                self.h5[group].keys(), desc=f"Reading attributes from group {group}"
            ):
                g = self.h5[group][i]
                if not isinstance(g, h5py.Dataset):
                    logger.warning(
                        f"Skipping H5AD slot {i!r} because only dense "
                        f"{group} arrays can be imported"
                    )
                    continue
                if g.shape[0] != self.nCells:
                    logger.warning(
                        f"Skipping H5AD slot {i!r} with unexpected shape {g.shape}"
                    )
                    continue
                for j in range(g.shape[1]):
                    yield f"{i}{j + 1}", g[:, j]
        else:
            logger.warning(
                "H5AD obsm is missing or has an unsupported format; "
                "embeddings will be skipped"
            )

    def get_cell_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the cell columns."""
        for i, j in self._get_col_data(
            self.cellAttrsKey, [self.cellIdsKey, self.catNamesKey]
        ):
            yield i, j
        for i, j in self._get_obsm_data(self.obsmAttrsKey):
            yield i, j

    def get_feat_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the feature columns."""
        for i, j in self._get_col_data(
            self.featureAttrsKey,
            [self.featIdsKey, self.featNamesKey, self.catNamesKey],
        ):
            yield i, j

    def feature_types(self, key: str) -> list[str]:
        """Return decoded feature types from a var column."""
        if not self._check_exists(self.featureAttrsKey, key):
            raise KeyError(
                f"Feature type key `{key}` was not found in {self.featureAttrsKey}"
            )
        values = self._replace_category_values(
            self.h5[self.featureAttrsKey][key],
            key,
            self.featureAttrsKey,
        )
        if values.ndim != 1 or len(values) != self.nFeatures:
            raise ValueError(
                f"Feature type key `{key}` has {len(values)} values; "
                f"expected {self.nFeatures}"
            )
        return [
            value.decode("utf-8")
            if isinstance(value, bytes | np.bytes_)
            else str(value)
            for value in values
        ]

    def assay_feature_slices(
        self,
        key: str,
        name_map: Mapping[str, str] | None = None,
    ) -> dict[str, _H5adAssayFeatures]:
        """Resolve feature ranges and metadata for each assay."""
        assay_table = auto_name_feat_table(
            make_feat_table_from_types(self.feature_types(key)),
            name_map,
        )
        feature_ids = self.feat_ids()
        feature_names = self.feat_names()
        assays: dict[str, _H5adAssayFeatures] = {}
        for assay_name in dict.fromkeys(assay_table.columns):
            selected = assay_table[assay_name]
            ranges: tuple[tuple[int, int], ...]
            if selected.ndim == 1:
                ranges = ((int(selected.loc["start"]), int(selected.loc["end"])),)
            else:
                ranges = tuple(
                    (int(start), int(end))
                    for start, end in zip(
                        selected.loc["start"],
                        selected.loc["end"],
                        strict=True,
                    )
                )
            indexes = np.concatenate(
                [np.arange(start, end, dtype=np.int64) for start, end in ranges]
            )
            assays[str(assay_name)] = _H5adAssayFeatures(
                featureIndexes=indexes,
                featureIds=feature_ids[indexes],
                featureNames=feature_names[indexes],
            )
        return assays

    # noinspection DuplicatedCode
    def consume_dataset(
        self,
        batch_size: int = 1000,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        dset = self.h5[self.matrixKey]
        start = max(0, int(row_start))
        stop = int(dset.shape[0] if row_end is None else row_end)
        if stop < start or stop > int(dset.shape[0]):
            raise ValueError("consume row range is outside the matrix")
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            yield coo_matrix(dset[offset:end])

    def _sparse_indices_are_strictly_sorted(self, maxValues: int) -> bool:
        group = self.h5[self.matrixKey]
        if not isinstance(group, h5py.Group) or maxValues < 1:
            return False
        indptr_node = group["indptr"]
        indices_node = group["indices"]
        compressed_size = int(indptr_node.size) - 1
        start = 0
        while start < compressed_size:
            pointer_end = min(compressed_size, start + maxValues)
            pointers = np.asarray(indptr_node[start : pointer_end + 1])
            base = int(pointers[0])
            relative = pointers - base
            vectors = int(np.searchsorted(relative, maxValues, side="right") - 1)
            if vectors < 1:
                return False
            pointers = pointers[: vectors + 1]
            end = start + vectors
            indices = np.asarray(indices_node[base : int(pointers[-1])])
            offsets = pointers - base
            for left, right in zip(offsets[:-1], offsets[1:], strict=True):
                vector = indices[int(left) : int(right)]
                if vector.size > 1 and np.any(vector[1:] <= vector[:-1]):
                    return False
            start = end
        return True

    def infer_storage_dtype(self, maxScanBytes: int = 64 * 1024 * 1024) -> Any:
        """Resolve the smallest lossless storage dtype."""
        if (
            self._dtypeOverridden
            or self.groupCodes[self.matrixKey] != 2
            or np.dtype(self.matrixDtype).kind != "f"
        ):
            return self.storageDtype
        group = self.h5[self.matrixKey]
        if not isinstance(group, h5py.Group):
            return self.storageDtype

        data_node = group["data"]
        indices_node = group["indices"]
        bytes_per_value = max(
            64,
            3 * int(data_node.dtype.itemsize)
            + 3 * int(indices_node.dtype.itemsize)
            + int(group["indptr"].dtype.itemsize),
        )
        check_values = min(
            1024 * 1024,
            max(0, int(maxScanBytes)) // bytes_per_value,
        )
        if not self._sparse_indices_are_strictly_sorted(check_values):
            logger.debug(
                "Keeping the H5AD source dtype because sparse coordinates are "
                "not canonical within the dtype-scan memory limit"
            )
            return self.storageDtype

        finite = True
        integral = True
        minimum = np.inf
        maximum = -np.inf
        for start in range(0, data_node.size, check_values):
            values = np.asarray(data_node[start : start + check_values])
            if not values.size:
                continue
            finite = finite and bool(np.isfinite(values).all())
            integral = integral and bool(np.equal(values, np.trunc(values)).all())
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))

        source_dtype = np.dtype(self.matrixDtype)
        storage_dtype = source_dtype
        if finite and integral and minimum >= 0:
            for candidate in (
                np.dtype("uint8"),
                np.dtype("uint16"),
                np.dtype("uint32"),
            ):
                if (
                    maximum <= np.iinfo(candidate).max
                    and candidate.itemsize < source_dtype.itemsize
                ):
                    storage_dtype = candidate
                    break

        self.storageDtype = storage_dtype
        logger.debug(f"Resolved H5AD storage dtype={storage_dtype}")
        return storage_dtype

    def csc_conversion_peak_bytes(self) -> int:
        """Return a conservative peak estimate for one CSC to CSR conversion."""
        if self.matrixOrientation != "csc":
            return 0
        group = self.h5[self.matrixKey]
        if not isinstance(group, h5py.Group):
            raise TypeError("CSC matrix slot must be an HDF5 group")
        data_node = group["data"]
        indices_node = group["indices"]
        indptr_node = group["indptr"]
        source = sum(
            int(dataset.size) * int(dataset.dtype.itemsize)
            for dataset in (data_node, indices_node, indptr_node)
        )
        index_itemsize = int(indices_node.dtype.itemsize)
        if (
            max(self.nCells, self.nFeatures, int(data_node.size))
            <= np.iinfo(np.int32).max
        ):
            index_itemsize = np.dtype("int32").itemsize
        normalized = (
            int(data_node.size) * np.dtype(self.storageDtype).itemsize
            + int(indices_node.size) * index_itemsize
            + int(indptr_node.size) * index_itemsize
        )
        canonical_value_itemsize = max(
            int(data_node.dtype.itemsize),
            np.dtype(np.int64).itemsize,
        )
        canonicalization = int(data_node.size) * (
            4 * canonical_value_itemsize + 6 * np.dtype(np.int64).itemsize + 4
        )
        destination = (
            int(data_node.size)
            * (np.dtype(self.storageDtype).itemsize + index_itemsize)
            + (self.nCells + 1) * index_itemsize
        )
        return int(source + normalized + canonicalization + destination)

    def materialized_csr_bytes(self) -> int:
        """Return bytes retained by the materialized CSC-to-CSR conversion."""
        if self._convertedCsr is None:
            return 0
        return int(
            self._convertedCsr.data.nbytes
            + self._convertedCsr.indices.nbytes
            + self._convertedCsr.indptr.nbytes
        )

    def _csr_indptr(self) -> np.ndarray | None:
        if self.matrixOrientation == "dense":
            return None
        if self._convertedCsr is not None:
            return np.asarray(self._convertedCsr.indptr)
        if self.matrixOrientation != "csr":
            return None
        if self._indptrCache is None:
            self._indptrCache = np.asarray(self.h5[self.matrixKey]["indptr"][:])
        return self._indptrCache

    def _row_nnz_cumulative(self) -> np.ndarray | None:
        indptr = self._csr_indptr()
        if indptr is None:
            return None
        if self._cumulativeRowNnz is None:
            cumulative = np.empty(self.nCells + 1, dtype=np.int64)
            cumulative[0] = 0
            np.cumsum(np.diff(indptr), dtype=np.int64, out=cumulative[1:])
            self._cumulativeRowNnz = cumulative
        return self._cumulativeRowNnz

    def _prepare_sparse_import(self) -> None:
        self._row_nnz_cumulative()

    def _sparse_import_resident_bytes(self) -> int:
        total = (
            0 if self._cumulativeRowNnz is None else int(self._cumulativeRowNnz.nbytes)
        )
        if self._convertedCsr is None and self._indptrCache is not None:
            total += int(self._indptrCache.nbytes)
        return total

    def max_batch_nnz(self, batch_size: int) -> int:
        """Return the largest contiguous row-window nnz without loading values."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch_rows = min(batch_size, self.nCells)
        if self.matrixOrientation == "dense":
            return int(batch_rows * self.nFeatures)
        cumulative = self._row_nnz_cumulative()
        if cumulative is None:
            return int(batch_rows * self.nFeatures)

        if self.nCells == 0:
            return 0
        return int(np.max(cumulative[batch_rows:] - cumulative[:-batch_rows]))

    def max_batch_nnz_peak_bytes(self) -> int:
        """Bound temporary row-pointer arrays used to plan sparse batches."""
        if self.matrixOrientation == "dense":
            return 0
        self._prepare_sparse_import()
        return self._sparse_import_resident_bytes()

    def producer_batch_staging_bytes(self, batch_size: int) -> int:
        """Bound sparse row pointers retained while one batch is produced."""
        rows = min(max(1, int(batch_size)), self.nCells)
        if rows == 0 or self.matrixOrientation == "dense":
            return 0
        if self._convertedCsr is not None:
            itemsize = np.asarray(self._convertedCsr.indptr).dtype.itemsize
        else:
            itemsize = self.h5[self.matrixKey]["indptr"].dtype.itemsize
        normalized_itemsize = np.dtype(np.int32).itemsize
        return int((rows + 1) * (2 * itemsize + normalized_itemsize))

    def materialize_csc(self) -> None:
        """Convert the complete CSC source to CSR once."""
        from ..utils.arrays import canonicalize_sparse

        if self.matrixOrientation != "csc" or self._convertedCsr is not None:
            return
        group = self.h5[self.matrixKey]
        if not isinstance(group, h5py.Group):
            raise TypeError("CSC matrix slot must be an HDF5 group")
        data_node = group["data"]
        data = np.asarray(data_node[:])
        indices = np.asarray(group["indices"][:])
        indptr = np.asarray(group["indptr"][:])
        if max(self.nCells, self.nFeatures, data.size) <= np.iinfo(np.int32).max:
            indices = indices.astype(np.int32, copy=False)
            indptr = indptr.astype(np.int32, copy=False)

        source = csc_matrix(
            (data, indices, indptr),
            shape=(self.nCells, self.nFeatures),
        )
        canonical = canonicalize_sparse(
            source.tocoo(copy=False),
            self.storageDtype,
        )
        self._convertedCsr = canonical.tocsr()
        logger.debug(
            f"Materialized H5AD CSR matrix for conversion with "
            f"dtype={self.storageDtype}"
        )

    def consume_group(
        self,
        batch_size: int,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        start = max(0, int(row_start))
        stop = int(self.nCells if row_end is None else row_end)
        if stop < start or stop > self.nCells:
            raise ValueError("consume row range is outside the matrix")

        if self._convertedCsr is not None or self.matrixOrientation == "csc":
            yield from self._consume_converted_csr(batch_size, start, stop)
            return

        grp = self.h5[self.matrixKey]
        source_indptr = self._csr_indptr()
        if source_indptr is None:
            raise RuntimeError("CSR row pointers are unavailable")
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            indptr = source_indptr[offset : end + 1]
            data_start = int(indptr[0])
            data_end = int(indptr[-1])
            local_indptr = indptr - data_start
            n_rows = end - offset
            batch = csr_matrix(
                (
                    np.asarray(grp["data"][data_start:data_end]),
                    np.asarray(grp["indices"][data_start:data_end]),
                    local_indptr,
                ),
                shape=(n_rows, self.nFeatures),
            )
            yield batch.tocoo(copy=False)

    def _consume_converted_csr(
        self,
        batch_size: int,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> Generator[coo_matrix, None, None]:
        """Convert the complete CSC matrix once before yielding row batches."""
        if self._convertedCsr is None:
            self.materialize_csc()
        if self._convertedCsr is None:
            raise RuntimeError("CSC materialization did not produce a CSR matrix")
        start = max(0, int(row_start))
        stop = int(self.nCells if row_end is None else row_end)
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            yield self._convertedCsr[offset:end].tocoo(copy=False)

    def consume_row_range(
        self,
        batch_size: int,
        row_start: int,
        row_end: int,
    ) -> Generator[coo_matrix, None, None]:
        """Yield source batches covering ``[row_start, row_end)``."""
        if self.groupCodes[self.matrixKey] == 1:
            return self.consume_dataset(batch_size, row_start, row_end)
        if self.groupCodes[self.matrixKey] == 2:
            return self.consume_group(batch_size, row_start, row_end)
        raise ValueError(
            f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
        )

    def consume(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        return self.consume_row_range(batch_size, 0, self.nCells)
