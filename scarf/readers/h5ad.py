from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arrays import assay_feature_ranges, max_window_nnz
from ..utils.count_values import (
    CountValueRange,
    compressed_count_ranges,
    dense_count_ranges,
)
from ..utils.logging import logger
from ..utils.progress import iter_progress
from ._assay_names import auto_name_feat_table, make_feat_table_from_types
from ._h5ad_columns import (
    SPARSE_KEYS,
    column_encoding,
    column_length,
    column_order,
    index_key,
    is_column,
    is_nullable,
    present_column,
    read_table_column,
    sparse_encoding,
    sparse_shape,
    table_column_dtype,
    table_column_names,
    table_members,
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
        sourceMatrixDtype: dtype in which the source holds the matrix values, in native byte order.
                           float16 is not a count storage dtype, so float16 source values are read
                           as float32.
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
            # A malformed column-order must fail here, before a writer opens
            # its destination.
            for group in (self.cellAttrsKey, self.featureAttrsKey):
                if self.groupCodes[group] == 2:
                    column_order(self.h5[group])
            self.matrixOrientation = self._validate_sparse_matrix()
            self.sourceMatrixDtype: Any = self._get_matrix_dtype()
            self._convertedCsr: SparseRowStore | None = None
            self._indptrCache: np.ndarray | None = None
            matrix_shape = self._matrix_shape()
            self.nCells, self.nFeatures = (
                self._get_n(self.cellAttrsKey, matrix_shape[0]),
                self._get_n(self.featureAttrsKey, matrix_shape[1]),
            )
            if (self.nCells, self.nFeatures) != matrix_shape:
                raise ValueError(
                    f"ERROR: Matrix `{self.matrixKey}` has shape {matrix_shape}, "
                    f"but `{self.cellAttrsKey}` has {self.nCells} rows and "
                    f"`{self.featureAttrsKey}` has {self.nFeatures} rows"
                )
            self.cellIdsKey = self._fix_name_key(self.cellAttrsKey, cell_ids_key)
            self.featIdsKey = self._fix_name_key(self.featureAttrsKey, feature_ids_key)
            self.featNamesKey = feature_name_key
            self.catNamesKey = category_names_key
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
        """Return the stored dtype in native byte order, with float16 as float32.

        SciPy sparse matrices hold neither float16 nor another byte order, and
        float16 is not a count storage dtype, so the reader yields matrix
        values in this dtype.
        """
        if self.groupCodes[self.matrixKey] == 1:
            dtype = self.h5[self.matrixKey].dtype
        elif self.groupCodes[self.matrixKey] == 2:
            dtype = self.h5[self.matrixKey]["data"].dtype
        else:
            raise ValueError(
                f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
            )
        dtype = dtype.newbyteorder("=")
        return np.dtype(np.float32) if dtype == np.float16 else dtype

    def _matrix_values(self, node: h5py.Dataset, selection: slice) -> np.ndarray:
        """Read matrix values in ``sourceMatrixDtype``."""
        if node.dtype == self.sourceMatrixDtype:
            return np.asarray(node[selection])
        return np.asarray(node.astype(self.sourceMatrixDtype)[selection])

    def _matrix_shape(self) -> tuple[int, int]:
        """Return the stored matrix shape after checking that it is consistent.

        A sparse group must record its shape in an attribute, and its indptr
        must hold one pointer per row (CSR) or column (CSC) of that shape.
        """
        matrix = self.h5[self.matrixKey]
        if self.matrixOrientation == "dense":
            if matrix.ndim != 2:
                raise ValueError(
                    f"ERROR: Dense matrix `{self.matrixKey}` must be two-dimensional"
                )
            return int(matrix.shape[0]), int(matrix.shape[1])
        shape = sparse_shape(matrix)
        if shape is None:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` has no shape attribute"
            )
        expected = ((shape[0] if self.matrixOrientation == "csr" else shape[1]) + 1,)
        if matrix["indptr"].shape != expected:
            raise ValueError(
                f"ERROR: Sparse matrix group `{self.matrixKey}` of shape {shape} "
                f"needs an indptr of shape {expected}; found {matrix['indptr'].shape}"
            )
        return shape

    def _check_exists(self, group: str, key: str) -> bool:
        if group in self.groupCodes:
            group_code = self.groupCodes[group]
        else:
            group_code = self._validate_group(group)
            self.groupCodes[group] = group_code
        if group_code == 1:
            # A plain dataset has no fields, so it holds no named column.
            if key in (self.h5[group].dtype.names or ()):
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
                field_dtype = np.dtype(cell_attrs.dtype.fields[key][0])
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

    def _get_n(self, group: str, matrix_length: int) -> int:
        """Return the row count of a table; an absent table takes the matrix's."""
        if self.groupCodes[group] == 0:
            return matrix_length
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
        """Yield each decodable column with its stored values and missing mask.

        Columns come by source name. A dataframe group lists them in
        ``column-order``, which resolves a name that old AnnData versions
        nested into groups because it contains ``/``.
        """
        code = self.groupCodes[group]
        if code not in {1, 2}:
            return
        table = self.h5[group]
        nodes: dict[str, Any] = {}
        if code == 1:
            names: tuple[str, ...] = tuple(table.dtype.names or ())
        else:
            members = table_members(table)
            for name in members.unresolved:
                logger.warning(
                    f"Skipping {group} column {name!r} because column-order lists "
                    "it but the file does not contain it"
                )
            nodes = dict(members.members)
            names = tuple(nodes)
        for name in iter_progress(names, desc=f"Reading attributes from group {group}"):
            if name in ignore_keys:
                continue
            if code == 2 and not is_column(nodes[name]):
                if isinstance(nodes[name], h5py.Group):
                    logger.warning(
                        f"Skipping {group} column {name!r} because its H5AD encoding "
                        f"{column_encoding(nodes[name])!r} is not supported"
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

    def _source_column_names(self, group: str) -> list[str]:
        """Return the source names of the decodable columns of a table.

        The names include the ID, name, and cluster columns that metadata
        import leaves out, so a caller can plan storage keys over the whole
        table.
        """
        if self.groupCodes.get(group) not in {1, 2}:
            return []
        return table_column_names(self.h5[group])

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
        """Return rows of a cluster column with non-finite values marked missing.

        The constructor validated each cluster column as one scalar per cell.
        """
        values, missing = self._read_column(self.cellAttrsKey, key, start, stop)
        if values.dtype.kind in "fc":
            missing = missing | ~np.isfinite(values)
        return values, missing

    def _cell_ids_block(self, start: int, stop: int) -> np.ndarray:
        """Return cell IDs ``[start, stop)`` as the text an import stores.

        An import validates every ID with :meth:`cell_ids` before it reads
        blocks.
        """
        if not self._check_exists(self.cellAttrsKey, self.cellIdsKey):
            return np.asarray([f"cell_{index}" for index in range(start, stop)])
        values, _ = self._read_column(self.cellAttrsKey, self.cellIdsKey, start, stop)
        return np.asarray(
            [as_text(value) for value in values.astype(object)], dtype=str
        )

    def _obsm_array(self, key: str) -> h5py.Dataset:
        # The constructor validated each embedding key as a dense array.
        return self.h5[self.obsmAttrsKey][key]

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
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dset = self.h5[self.matrixKey]
        start = max(0, int(row_start))
        stop = int(dset.shape[0] if row_end is None else row_end)
        if stop < start or stop > int(dset.shape[0]):
            raise ValueError("consume row range is outside the matrix")
        for offset in range(start, stop, batch_size):
            end = min(offset + batch_size, stop)
            yield coo_matrix(self._matrix_values(dset, slice(offset, end)))

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        """Return the range of the canonical values of each group of features.

        Duplicate coordinates of a sparse matrix are summed, as the import
        stores them, so the ranges depend neither on the order of distinct
        coordinates, nor on orientation or encoding. Pass a range to
        :func:`~scarf.storage.count_dtype.count_storage_dtype` to resolve the
        storage dtype of its group's counts.

        Args:
            maxBytes: Memory available to the scan.
            featureGroups: Group, numbered from zero, of each feature, such as
                the assay that stores it. None puts every feature in one
                group.

        Raises:
            MemoryError: If one compressed vector or dense row does not fit in
                ``maxBytes``.
            ValueError: If the matrix holds NaN or an infinite value.
        """
        node = self.h5[self.matrixKey]
        if self.matrixOrientation == "dense":
            return dense_count_ranges(
                lambda start, stop: self._matrix_values(node, slice(start, stop)),
                int(node.shape[0]),
                int(node.shape[1]),
                maxBytes=maxBytes,
                groups=featureGroups,
            )
        csr = self.matrixOrientation == "csr"
        return compressed_count_ranges(
            node["indptr"],
            node["indices"],
            node["data"],
            minorSize=self.nFeatures if csr else self.nCells,
            maxBytes=maxBytes,
            groups=featureGroups,
            # Features are the minor axis of CSR and the vectors of CSC.
            groupAxis=1 if csr else 0,
        )

    @property
    def consumeDtype(self) -> np.dtype[Any]:
        """Return the dtype in which ``consume`` yields the matrix values.

        It is ``sourceMatrixDtype``, except that the converted rows of a CSC
        integer source yield their duplicate-summed values in int64, or uint64
        for unsigned and boolean sources.
        """
        source = (
            self.sourceMatrixDtype
            if self._convertedCsr is None
            else self._convertedCsr.dtype
        )
        dtype: np.dtype[Any] = np.dtype(source)
        return dtype

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

    def _prepare_sparse_import(self) -> None:
        self._csr_indptr()

    def _sparse_import_resident_bytes(self) -> int:
        """Return the bytes of cached CSR row pointers.

        Converted CSC rows count under :meth:`materialized_csr_bytes`.
        """
        if self._convertedCsr is None and self._indptrCache is not None:
            return int(self._indptrCache.nbytes)
        return 0

    def max_batch_nnz(self, batch_size: int) -> int:
        """Return the largest contiguous row-window nnz without loading values."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch_rows = min(batch_size, self.nCells)
        indptr = self._csr_indptr()
        if indptr is None:
            return int(batch_rows * self.nFeatures)
        return max_window_nnz(indptr, batch_size)

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
        """Convert CSC into temporary row storage in bounded blocks.

        The rows hold the source values with duplicate coordinates summed.
        Floating-point values keep ``sourceMatrixDtype``; integers are held in
        the 64-bit integer dtype in which their duplicates are summed, so a sum
        past the range of a narrow source dtype is kept, as a CSR import keeps
        it.
        """
        if self.matrixOrientation != "csc" or self._convertedCsr is not None:
            return
        group = self.h5[self.matrixKey]
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

        source = np.dtype(self.sourceMatrixDtype)
        self._convertedCsr = SparseRowStore(
            chunks,
            shape,
            (
                np.dtype(np.uint64 if source.kind in "bu" else np.int64)
                if source.kind in "biu"
                else source
            ),
            max_bytes=maxBytes - indptr.nbytes,
            source_dtype=source,
            temp_dir=self._tempDir,
        )
        logger.debug("Prepared H5AD row storage for the CSC conversion")

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
        assert source_indptr is not None
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
        start: int,
        stop: int,
    ) -> Generator[coo_matrix, None, None]:
        """Yield row batches from the temporary CSC conversion."""
        if self._convertedCsr is None:
            self.materialize_csc()
        assert self._convertedCsr is not None
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
        # The constructor rejects a matrix slot that is neither a dataset nor
        # a group.
        if self.groupCodes[self.matrixKey] == 1:
            return self.consume_dataset(batch_size, row_start, row_end)
        return self.consume_group(batch_size, row_start, row_end)

    def consume(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        return self.consume_row_range(batch_size, 0, self.nCells)
