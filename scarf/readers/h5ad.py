from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from ..utils.logging import logger
from ..utils.progress import tqdmbar
from ._assay_names import auto_name_feat_table, make_feat_table_from_types
from ._h5ad_inspect import H5adInspectResult, inspect_h5ad as inspect_h5ad


@dataclass(frozen=True)
class _H5adAssayFeatures:
    ranges: tuple[tuple[int, int], ...]
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
        self.nCells, self.nFeatures = (
            self._get_n(self.cellAttrsKey),
            self._get_n(self.featureAttrsKey),
        )
        self.cellIdsKey = self._fix_name_key(self.cellAttrsKey, cell_ids_key)
        self.featIdsKey = self._fix_name_key(self.featureAttrsKey, feature_ids_key)
        self.featNamesKey = feature_name_key
        self.catNamesKey = category_names_key
        self.matrixDtype: Any = self._get_matrix_dtype() if dtype is None else dtype

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
                    logger.info(
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
                if isinstance(self.h5[group][i], h5py.Dataset):
                    return int(self.h5[group][i].shape[0])
                if (
                    isinstance(self.h5[group][i], h5py.Group)
                    and "codes" in self.h5[group][i]
                ):
                    return int(self.h5[group][i]["codes"].shape[0])
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
        logger.warning(f"Could not find cells ids key: {self.cellIdsKey} in `obs`.")
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
            f"Could not find feature ids key: {self.featIdsKey} in {self.featureAttrsKey}."
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
            f"Could not find feature names key: {self.featNamesKey} in self.featureAttrsKey."
        )
        return self.feat_ids()

    def _replace_category_values(
        self, v: np.ndarray | h5py.Group | h5py.Dataset, key: str, group: str
    ) -> np.ndarray:
        # check if v is a Group with codes + categories structure
        if isinstance(v, h5py.Group):
            if "codes" in v and "categories" in v:
                codes = v["codes"][:]
                categories = v["categories"][:]
                valid = (codes >= 0) & (codes < len(categories))
                decoded = np.empty(codes.shape, dtype=object)
                decoded[valid] = categories[codes[valid]]
                decoded[~valid] = None
                return decoded
            logger.warning(f"{key} is a Group but is not a categorical column")
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
            for i in tqdmbar(
                self.h5[group].dtype.names,
                desc=f"Reading attributes from group {group}",
            ):
                if i in ignore_keys:
                    continue
                yield i, self._replace_category_values(self.h5[group][i][:], i, group)
        if self.groupCodes[group] == 2:
            for i in tqdmbar(
                self.h5[group].keys(), desc=f"Reading attributes from group {group}"
            ):
                if i in ignore_keys:
                    continue
                values = self.h5[group][i]
                if isinstance(values, h5py.Dataset | h5py.Group):
                    if isinstance(values, h5py.Group) and not {
                        "codes",
                        "categories",
                    }.issubset(values.keys()):
                        continue
                    yield (
                        i,
                        self._replace_category_values(values, i, group),
                    )

    def _get_obsm_data(
        self, group: str
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        if self.groupCodes[group] == 2:
            for i in tqdmbar(
                self.h5[group].keys(), desc=f"Reading attributes from group {group}"
            ):
                g = self.h5[group][i]
                if g.shape[0] != self.nCells:
                    logger.error(
                        f"Dimension of {i}({g.shape}) is not correct."
                        f" Will not save this specific slot into Zarr."
                    )
                    continue
                if isinstance(g, h5py.Dataset):
                    for j in range(g.shape[1]):
                        yield f"{i}{j + 1}", g[:, j]
        else:
            logger.warning(
                f"Reading of obsm failed because it either does not exist or is not in expected format"  # noqa: F541
            )

    def get_cell_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the cell columns."""
        for i, j in self._get_col_data(self.cellAttrsKey, [self.cellIdsKey]):
            yield i, j
        for i, j in self._get_obsm_data(self.obsmAttrsKey):
            yield i, j

    def get_feat_columns(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the feature columns."""
        for i, j in self._get_col_data(
            self.featureAttrsKey, [self.featIdsKey, self.featNamesKey]
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
                ranges=ranges,
                featureIndexes=indexes,
                featureIds=feature_ids[indexes],
                featureNames=feature_names[indexes],
            )
        return assays

    # noinspection DuplicatedCode
    def consume_dataset(
        self, batch_size: int = 1000
    ) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        dset = self.h5[self.matrixKey]
        s = 0
        for e in range(batch_size, dset.shape[0] + batch_size, batch_size):
            if e > dset.shape[0]:
                e = dset.shape[0]
            yield coo_matrix(dset[s:e])
            s = e

    def consume_group(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.matrixOrientation == "csc":
            yield from self._consume_converted_csr(batch_size)
            return

        grp = self.h5[self.matrixKey]
        for row_start in range(0, self.nCells, batch_size):
            row_end = min(row_start + batch_size, self.nCells)
            indptr = np.asarray(grp["indptr"][row_start : row_end + 1])
            data_start = int(indptr[0])
            data_end = int(indptr[-1])
            local_indptr = indptr - data_start
            n_rows = row_end - row_start
            row_indices = np.repeat(
                np.arange(n_rows),
                np.diff(local_indptr),
            )
            yield coo_matrix(
                (
                    grp["data"][data_start:data_end],
                    (row_indices, grp["indices"][data_start:data_end]),
                ),
                shape=(n_rows, self.nFeatures),
            )

    def _consume_converted_csr(
        self,
        batch_size: int,
    ) -> Generator[coo_matrix, None, None]:
        """Convert the complete CSC matrix once before yielding row batches."""
        if self._convertedCsr is None:
            group = self.h5[self.matrixKey]
            if not isinstance(group, h5py.Group):
                raise TypeError("CSC matrix slot must be an HDF5 group")
            converted = csc_matrix(
                (
                    np.asarray(group["data"][:]),
                    np.asarray(group["indices"][:]),
                    np.asarray(group["indptr"][:]),
                ),
                shape=(self.nCells, self.nFeatures),
            ).tocsr()
            self._convertedCsr = converted.astype(self.matrixDtype, copy=False)

        for row_start in range(0, self.nCells, batch_size):
            row_end = min(row_start + batch_size, self.nCells)
            yield self._convertedCsr[row_start:row_end].tocoo(copy=False)

    def consume(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        if self.groupCodes[self.matrixKey] == 1:
            return self.consume_dataset(batch_size)
        elif self.groupCodes[self.matrixKey] == 2:
            return self.consume_group(batch_size)
        raise ValueError(
            f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
        )
