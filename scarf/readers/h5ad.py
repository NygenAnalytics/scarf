from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arrays import assay_feature_ranges, cumulative_nnz, max_window_nnz
from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._assay_names import auto_name_feat_table, make_feat_table_from_types
from ._h5ad_columns import (
    SPARSE_KEYS,
    column_encoding,
    column_length,
    index_key,
    is_column,
    is_nullable,
    present_column,
    read_table_column,
    sparse_encoding,
    sparse_shape,
    table_column_dtype,
)
from ._h5ad_inspect import H5adInspectResult, inspect_h5ad as inspect_h5ad
from ._sparse import SparseRowStore
from ._text import as_text, require_unique_identifiers

type H5adEmbeddingRole = Literal["umap", "tsne"]


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
               method. (Default value: Automatically determined). float16 is not a storage dtype, so float16
               source values are read as float32.
        temp_dir: Parent directory for temporary CSC row storage. None uses the system temporary directory.

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
        embedding_roles: Exact ``obsm`` keys to import as immutable UMAP or
                         t-SNE artifacts.
        cluster_keys: Exact ``obs`` keys to import as immutable cluster-label
                      artifacts instead of raw cell metadata.
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
        embedding_roles: Mapping[str, H5adEmbeddingRole] | None = None,
        cluster_keys: Sequence[str] = (),
        *,
        temp_dir: str | Path | None = None,
    ) -> None:
        self.h5adFn = h5ad_fn
        self._tempDir = temp_dir
        self.h5: h5py.File = h5py.File(h5ad_fn, mode="r")
        try:
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
            self._convertedCsr: SparseRowStore | None = None
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
            self.embeddingRoles = self._validate_embedding_roles(embedding_roles)
            self.clusterKeys = self._validate_cluster_keys(cluster_keys)
        except BaseException:
            self.h5.close()
            raise

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
            "embedding_roles": dict(self.embeddingRoles),
            "cluster_keys": self.clusterKeys,
            "temp_dir": self._tempDir,
        }

    def close(self) -> None:
        self.h5.close()
        if self._convertedCsr is not None:
            self._convertedCsr.close()
        self._convertedCsr = None
        self._indptrCache = None
        self._cumulativeRowNnz = None

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

        missing = SPARSE_KEYS.difference(group.keys())
        if missing:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` is missing: "
                f"{', '.join(sorted(missing))}"
            )

        encoding = sparse_encoding(group)
        if encoding is None:
            declared = group.attrs.get(
                "encoding-type", group.attrs.get("h5sparse_format")
            )
            raise ValueError(
                f"ERROR: Sparse matrix encoding `{declared}` of `{self.matrixKey}` "
                "is not supported. H5adReader supports CSR and CSC encoding."
            )
        return encoding

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
        """Return the dtype in which the reader yields matrix values.

        float16 is not a count storage dtype, and SciPy sparse matrices
        cannot hold it, so float16 values are read as float32.
        """
        if self.groupCodes[self.matrixKey] == 1:
            dtype = self.h5[self.matrixKey].dtype
        elif self.groupCodes[self.matrixKey] == 2:
            dtype = self.h5[self.matrixKey]["data"].dtype
        else:
            raise ValueError(
                f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
            )
        return np.dtype(np.float32) if dtype.newbyteorder("=") == np.float16 else dtype

    def _matrix_values(self, node: h5py.Dataset, selection: slice) -> np.ndarray:
        """Read matrix values in ``sourceMatrixDtype``."""
        if node.dtype == self.sourceMatrixDtype:
            return np.asarray(node[selection])
        return np.asarray(node.astype(self.sourceMatrixDtype)[selection])

    def _matrix_shape(self) -> tuple[int, int]:
        matrix = self.h5[self.matrixKey]
        if isinstance(matrix, h5py.Dataset):
            return int(matrix.shape[0]), int(matrix.shape[1])
        shape = sparse_shape(matrix)
        if shape is None:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` has no shape attribute"
            )
        return shape

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
        if self._check_exists(group, key):
            return key
        if key == "_index" and self.groupCodes.get(group) == 2:
            # AnnData stores a named dataframe index under its name and records
            # that name in the ``_index`` attribute.
            recorded = index_key(self.h5[group])
            if recorded is not None and self._check_exists(group, recorded):
                return recorded
        if key.startswith("_"):
            temp_key = key[1:]
            if self._check_exists(group, temp_key):
                return temp_key
        return key

    @property
    def _categoryGroups(self) -> tuple[str, ...]:
        return (self.catNamesKey,)

    def _read_column(
        self,
        group: str,
        key: str,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return read_table_column(
            self.h5,
            self.h5[group],
            key,
            self._categoryGroups,
            start,
            stop,
        )

    def _validate_embedding_roles(
        self,
        roles: Mapping[str, H5adEmbeddingRole] | None,
    ) -> dict[str, H5adEmbeddingRole]:
        if roles is None:
            return {}
        if not isinstance(roles, Mapping):
            raise TypeError("embedding_roles must be a mapping")
        resolved: dict[str, H5adEmbeddingRole] = {}
        for raw_key, raw_role in roles.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError("embedding_roles keys must be non-empty strings")
            if raw_role not in {"umap", "tsne"}:
                raise ValueError("embedding_roles values must be 'umap' or 'tsne'")
            if not self._check_exists(self.obsmAttrsKey, raw_key):
                raise KeyError(
                    f"Embedding key {raw_key!r} was not found in {self.obsmAttrsKey}"
                )
            node = self.h5[self.obsmAttrsKey][raw_key]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f"Embedding key {raw_key!r} must be a dense H5AD array")
            if node.ndim != 2 or node.shape[0] != self.nCells or node.shape[1] < 1:
                raise ValueError(
                    f"Embedding key {raw_key!r} has incompatible shape {node.shape}"
                )
            if np.dtype(node.dtype).kind not in "biuf":
                raise TypeError(
                    f"Embedding key {raw_key!r} must contain numeric values"
                )
            resolved[raw_key] = raw_role
        return resolved

    def _validate_cluster_keys(self, keys: Sequence[str]) -> tuple[str, ...]:
        if isinstance(keys, str | bytes) or not isinstance(keys, Sequence):
            raise TypeError("cluster_keys must be a sequence of column names")
        resolved = tuple(keys)
        if len(set(resolved)) != len(resolved):
            raise ValueError("cluster_keys must be unique")
        for key in resolved:
            if not isinstance(key, str) or not key:
                raise ValueError("cluster_keys must contain non-empty strings")
            if key in {self.cellIdsKey, self.catNamesKey}:
                raise ValueError(f"Cluster key {key!r} is reserved H5AD metadata")
            if not self._check_exists(self.cellAttrsKey, key):
                raise KeyError(
                    f"Cluster key {key!r} was not found in {self.cellAttrsKey}"
                )
            cell_attrs = self.h5[self.cellAttrsKey]
            if isinstance(cell_attrs, h5py.Dataset):
                if cell_attrs.ndim != 1:
                    raise TypeError(
                        f"Cluster key {key!r} must contain one scalar value per cell"
                    )
                fields = cell_attrs.dtype.fields
                if fields is None or key not in fields:
                    raise TypeError(f"Cluster key {key!r} is not an H5AD column")
                field_dtype = np.dtype(fields[key][0])
                if field_dtype.subdtype is not None:
                    raise TypeError(
                        f"Cluster key {key!r} must contain one scalar value per cell; "
                        f"found vector dtype {field_dtype}"
                    )
                length = int(cell_attrs.shape[0])
            else:
                node = cell_attrs[key]
                if not is_column(node):
                    raise TypeError(
                        f"Cluster key {key!r} uses unsupported H5AD encoding "
                        f"{column_encoding(node)!r}"
                    )
                value_node = (
                    node
                    if isinstance(node, h5py.Dataset)
                    else node["codes" if "codes" in node else "values"]
                )
                if value_node.ndim != 1:
                    raise TypeError(
                        f"Cluster key {key!r} must contain one scalar value per cell"
                    )
                if is_nullable(node) and node["mask"].shape != node["values"].shape:
                    raise ValueError(
                        f"Cluster key {key!r} has a misaligned missingness mask"
                    )
                length = int(value_node.shape[0])
            if length != self.nCells:
                raise ValueError(
                    f"Cluster key {key!r} has {length} rows; expected {self.nCells}"
                )
            dtype = self._cell_column_value_dtype(key)
            if dtype.kind not in "biufOSU":
                raise TypeError(f"Cluster key {key!r} uses unsupported dtype {dtype}")
        return resolved

    def _get_n(self, group: str) -> int:
        if self.groupCodes[group] == 0:
            matrix_shape = self._matrix_shape()
            return matrix_shape[0 if group == self.cellAttrsKey else 1]
        if self.groupCodes[group] == 1:
            return int(self.h5[group].shape[0])
        table = self.h5[group]
        for key in (index_key(table), *table.keys()):
            if key is None or key not in table:
                continue
            length = column_length(table[key])
            if length is not None:
                return length
        raise KeyError(
            f"ERROR: `{group}` key doesn't contain any child node of Dataset type."
            f"Aborting because unexpected H5ad format."
        )

    def _identifiers(self, group: str, key: str, generated: str) -> np.ndarray:
        if self._check_exists(group, key):
            values = present_column(*self._read_column(group, key)).astype(object)
        else:
            n_rows = self.nCells if group == self.cellAttrsKey else self.nFeatures
            logger.warning(
                f"ID key {key!r} was not found in H5AD {group}; generated IDs "
                "will be used"
            )
            values = np.array([f"{generated}_{x}" for x in range(n_rows)])
        return values

    def cell_ids(self) -> np.ndarray:
        """Returns a list of cell IDs."""
        values = self._identifiers(self.cellAttrsKey, self.cellIdsKey, "cell")
        require_unique_identifiers(values, "H5AD cell IDs")
        return values

    def feat_ids(self) -> np.ndarray:
        """Returns a list of feature IDs."""
        values = self._identifiers(self.featureAttrsKey, self.featIdsKey, "feature")
        require_unique_identifiers(values, "H5AD feature IDs")
        return values

    def feat_names(self) -> np.ndarray:
        """Returns a list of feature names."""
        if self._check_exists(self.featureAttrsKey, self.featNamesKey):
            return present_column(
                *self._read_column(self.featureAttrsKey, self.featNamesKey)
            ).astype(object)
        logger.warning(
            f"Feature name key {self.featNamesKey!r} was not found in "
            f"{self.featureAttrsKey}; feature IDs will be used"
        )
        return self.feat_ids()

    def _table_columns(
        self, group: str, ignore_keys: Sequence[str]
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        """Yield each decodable column with its stored values and missing mask."""
        code = self.groupCodes[group]
        if code not in {1, 2}:
            return
        table = self.h5[group]
        names = table.dtype.names if code == 1 else tuple(table.keys())
        for name in iter_progress(names, desc=f"Reading attributes from group {group}"):
            if name in ignore_keys:
                continue
            if code == 2 and not is_column(table[name]):
                if isinstance(table[name], h5py.Group):
                    logger.warning(
                        f"Skipping {group} column {name!r} because its H5AD encoding "
                        f"{column_encoding(table[name])!r} is not supported"
                    )
                continue
            try:
                values, missing = self._read_column(group, name)
            except ValueError as error:
                logger.warning(f"Skipping {group} column {name!r}: {error}")
                continue
            if values.ndim != 1:
                logger.warning(
                    f"Skipping {group} column {name!r} because it holds "
                    f"{values.ndim}-dimensional values"
                )
                continue
            yield name, values, missing

    def _cell_column_value_dtype(self, key: str) -> np.dtype[Any]:
        return table_column_dtype(
            self.h5,
            self.h5[self.cellAttrsKey],
            key,
            self._categoryGroups,
        )

    def _cell_column_block(
        self,
        key: str,
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if start < 0 or stop < start or stop > self.nCells:
            raise ValueError("H5AD cell-column block is outside the cell axis")
        values, missing = self._read_column(self.cellAttrsKey, key, start, stop)
        if values.ndim != 1:
            raise TypeError(
                f"Cell column {key!r} must contain one scalar value per cell"
            )
        if values.dtype.kind in "fc":
            missing = missing | ~np.isfinite(values)
        return values, missing

    def _cell_ids_block(self, start: int, stop: int) -> np.ndarray:
        if self._check_exists(self.cellAttrsKey, self.cellIdsKey):
            values, missing = self._cell_column_block(
                self.cellIdsKey,
                start,
                stop,
            )
            if bool(missing.any()):
                raise ValueError("H5AD cell IDs contain missing values")
            return np.asarray(values)
        return np.asarray([f"cell_{index}" for index in range(start, stop)])

    def _obsm_array(self, key: str) -> h5py.Dataset:
        node = self.h5[self.obsmAttrsKey][key]
        if not isinstance(node, h5py.Dataset):
            raise TypeError(f"Embedding key {key!r} is not a dense H5AD array")
        return node

    def _iter_obsm_blocks(
        self,
        key: str,
        block_rows: int,
        dtype: np.dtype[Any],
    ) -> Iterator[np.ndarray]:
        node = self._obsm_array(key)
        for start in range(0, self.nCells, block_rows):
            stop = min(start + block_rows, self.nCells)
            yield np.asarray(node[start:stop], dtype=dtype)

    def _cell_columns(self) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        return self._table_columns(
            self.cellAttrsKey,
            [self.cellIdsKey, self.catNamesKey, *self.clusterKeys],
        )

    def _feature_columns(self) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        return self._table_columns(
            self.featureAttrsKey,
            [self.featIdsKey, self.featNamesKey, self.catNamesKey],
        )

    def get_cell_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yield raw ``obs`` metadata, excluding selected cluster artifacts.

        Missing categorical and text values are ``None``; missing numeric
        values are NaN.
        """
        for name, values, missing in self._cell_columns():
            yield name, present_column(values, missing)

    def get_feat_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the feature columns."""
        for name, values, missing in self._feature_columns():
            yield name, present_column(values, missing)

    def feature_types(self, key: str) -> list[str]:
        """Return decoded feature types from a var column."""
        if not self._check_exists(self.featureAttrsKey, key):
            raise KeyError(
                f"Feature type key `{key}` was not found in {self.featureAttrsKey}"
            )
        values = present_column(*self._read_column(self.featureAttrsKey, key))
        if values.ndim != 1 or len(values) != self.nFeatures:
            raise ValueError(
                f"Feature type key `{key}` has {len(values)} values; "
                f"expected {self.nFeatures}"
            )
        return [as_text(value) for value in values]

    def assay_feature_slices(
        self,
        key: str,
        name_map: Mapping[str, str] | None = None,
    ) -> dict[str, _H5adAssayFeatures]:
        """Resolve feature ranges and metadata for each assay."""
        ranges = assay_feature_ranges(
            auto_name_feat_table(
                make_feat_table_from_types(self.feature_types(key)),
                name_map,
            )
        )
        feature_ids = self.feat_ids()
        feature_names = self.feat_names()
        assays: dict[str, _H5adAssayFeatures] = {}
        for assay_name, spans in ranges.items():
            indexes = np.concatenate(
                [np.arange(start, end, dtype=np.int64) for start, end in spans]
            )
            assays[assay_name] = _H5adAssayFeatures(
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
            yield coo_matrix(self._matrix_values(dset, slice(offset, end)))

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
        """Resolve the smallest lossless storage dtype.

        float16 values are read as float32, so a float16 source is stored as
        float32 unless an unsigned integer dtype holds its values.
        """
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

    def materialized_csr_bytes(self) -> int:
        """Return bytes retained by the materialized CSC-to-CSR conversion."""
        if self._convertedCsr is None:
            return 0
        return int(self._convertedCsr.indptr.nbytes)

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
            self._cumulativeRowNnz = cumulative_nnz(np.diff(indptr))
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
        return max_window_nnz(cumulative, batch_size)

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

    def materialize_csc(self, maxBytes: int = 64 * 1024 * 1024) -> None:
        """Convert CSC into temporary row storage in bounded blocks."""
        if self.matrixOrientation != "csc" or self._convertedCsr is not None:
            return
        group = self.h5[self.matrixKey]
        if not isinstance(group, h5py.Group):
            raise TypeError("CSC matrix slot must be an HDF5 group")
        data_node = group["data"]
        metadata = (self.nCells + 1) * 32 + (self.nFeatures + 1) * 8
        if maxBytes < metadata + 384:
            raise MemoryError("CSC row conversion exceeds the memory limit")
        chunk_nnz = min(1024 * 1024, (maxBytes - metadata) // 384)
        indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        shape = (self.nCells, self.nFeatures)

        def chunks() -> Iterator[coo_matrix]:
            for start in range(0, int(data_node.size), chunk_nnz):
                stop = min(int(data_node.size), start + chunk_nnz)
                columns = (
                    np.searchsorted(
                        indptr, np.arange(start, stop, dtype=np.int64), side="right"
                    )
                    - 1
                )
                yield coo_matrix(
                    (
                        self._matrix_values(data_node, slice(start, stop)),
                        (np.asarray(group["indices"][start:stop]), columns),
                    ),
                    shape=shape,
                )

        self._convertedCsr = SparseRowStore(
            chunks,
            shape,
            self.storageDtype,
            source_dtype=self.sourceMatrixDtype,
            max_bytes=maxBytes - indptr.nbytes,
            temp_dir=self._tempDir,
        )
        logger.debug(
            f"Prepared H5AD row storage for conversion with dtype={self.storageDtype}"
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
                    self._matrix_values(grp["data"], slice(data_start, data_end)),
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
        """Yield row batches from the temporary CSC conversion."""
        if self._convertedCsr is None:
            self.materialize_csc()
        if self._convertedCsr is None:
            raise RuntimeError("CSC materialization did not produce a CSR matrix")
        start = max(0, int(row_start))
        stop = int(self.nCells if row_end is None else row_end)
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            yield self._convertedCsr.read(offset, end).tocoo(copy=False)

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
