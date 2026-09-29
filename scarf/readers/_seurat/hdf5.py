import os
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import csr_matrix

from .._h5ad_columns import index_key, sparse_encoding
from .errors import MatrixSourceError
from .paths import (
    read_hdf5_names,
    read_hdf5_shape,
    require_hdf5_datasets,
    require_hdf5_group,
    validate_hdf5_file,
)
from .sources import (
    DEFAULT_LIMITS,
    BaseMatrixSource,
    CompressedMatrixSource,
    MatrixBlock,
    MatrixSource,
    MemoryEstimate,
    SourceLimits,
    _validate_shape,
    validate_compressed_pointers,
    validate_minor_indexes,
)


def _shape_from_group(group: h5py.Group) -> tuple[int, int] | None:
    for key in ("shape", "h5sparse_shape", "dim"):
        if key in group.attrs:
            return read_hdf5_shape(group.attrs[key], f"{group.name}@{key}")
        if key in group and isinstance(group[key], h5py.Dataset):
            return read_hdf5_shape(group[key][:], f"{group.name}/{key}")
    return None


def _h5ad_index_path(handle: h5py.File, group_path: str) -> str | None:
    """Return the dataframe index that AnnData names in ``_index``, if any."""
    key = index_key(handle.get(group_path))
    return None if key is None else f"{group_path}/{key}"


class HDF5DenseMatrixSource(BaseMatrixSource):
    """Dense HDF5 dataset written from R; its dimensions are reversed."""

    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        dataset: str,
        *,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        dtype: DTypeLike | None = None,
        as_sparse: bool = False,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.path = validate_hdf5_file(path, limits=limits)
        self.dataset = "/" + dataset.strip("/")
        with h5py.File(self.path, mode="r") as handle:
            if self.dataset not in handle:
                raise MatrixSourceError(f"HDF5 dataset {self.dataset!r} is missing")
            node = handle[self.dataset]
            if not isinstance(node, h5py.Dataset):
                raise MatrixSourceError(f"HDF5 path {self.dataset!r} is not a dataset")
            if node.ndim != 2:
                raise MatrixSourceError(
                    f"HDF5 dense dataset {self.dataset!r} must be two-dimensional"
                )
            if node.dtype.kind not in "biufc" or node.dtype.hasobject:
                raise TypeError(
                    f"HDF5 dense dataset {self.dataset!r} has nonnumeric dtype "
                    f"{node.dtype}"
                )
            logical_shape = _validate_shape(
                (int(node.shape[1]), int(node.shape[0])), limits
            )
            source_dtype = node.dtype if dtype is None else np.dtype(dtype)
        super().__init__(
            logical_shape,
            source_dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=as_sparse,
            limits=limits,
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        dense = (stop - start) * self.n_features * self.dtype.itemsize
        output = self._output_bytes(stop - start, (stop - start) * self.n_features)
        return MemoryEstimate(self.resident_bytes, dense + output, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        with h5py.File(self.path, mode="r") as handle:
            node = handle[self.dataset]
            assert isinstance(node, h5py.Dataset)
            values = np.ascontiguousarray(
                np.asarray(node[start:stop, :], dtype=self.dtype)
            )
        if self.is_sparse:
            return csr_matrix(values)
        return values


class ReshapedHDF5ArrayMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        dataset: str,
        shape: Sequence[int],
        *,
        dtype: DTypeLike | None = None,
        as_sparse: bool = False,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.path = validate_hdf5_file(path, limits=limits)
        self.dataset = "/" + dataset.strip("/")
        logical_shape = _validate_shape(shape, limits)
        with h5py.File(self.path, mode="r") as handle:
            if self.dataset not in handle:
                raise MatrixSourceError(f"HDF5 dataset {self.dataset!r} is missing")
            node = handle[self.dataset]
            if not isinstance(node, h5py.Dataset):
                raise MatrixSourceError(f"HDF5 path {self.dataset!r} is not a dataset")
            if node.ndim == 0:
                raise MatrixSourceError(
                    f"HDF5 dataset {self.dataset!r} cannot be reshaped from a scalar"
                )
            if node.dtype.kind not in "biufc" or node.dtype.hasobject:
                raise TypeError(
                    f"HDF5 dataset {self.dataset!r} has nonnumeric dtype {node.dtype}"
                )
            if int(node.size) != logical_shape[0] * logical_shape[1]:
                raise MatrixSourceError(
                    f"HDF5 dataset {self.dataset!r} has {node.size} values; "
                    f"reshaped matrix requires {logical_shape[0] * logical_shape[1]}"
                )
            self.physicalShape = tuple(int(value) for value in node.shape)
            source_dtype = node.dtype if dtype is None else np.dtype(dtype)
        super().__init__(
            logical_shape,
            source_dtype,
            is_sparse=as_sparse,
            limits=limits,
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        dense = (stop - start) * self.n_features * self.dtype.itemsize
        output = self._output_bytes(stop - start, (stop - start) * self.n_features)
        return MemoryEstimate(self.resident_bytes, dense + output, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        self._admit(self.estimate_read_memory(start, stop))
        flat_start = start * self.n_features
        flat_stop = stop * self.n_features
        output = np.empty(flat_stop - flat_start, dtype=self.dtype)
        with h5py.File(self.path, mode="r") as handle:
            node = handle[self.dataset]
            assert isinstance(node, h5py.Dataset)
            position = flat_start
            output_position = 0
            while position < flat_stop:
                coordinates = tuple(
                    int(value)
                    for value in np.unravel_index(
                        position,
                        self.physicalShape,
                        order="C",
                    )
                )
                run = int(
                    min(
                        flat_stop - position,
                        self.physicalShape[-1] - coordinates[-1],
                    )
                )
                selection: tuple[Any, ...] = coordinates[:-1] + (
                    slice(coordinates[-1], coordinates[-1] + run),
                )
                output[output_position : output_position + run] = np.asarray(
                    node[selection],
                    dtype=self.dtype,
                ).reshape(-1)
                position += run
                output_position += run
        values = output.reshape(stop - start, self.n_features)
        return csr_matrix(values) if self.is_sparse else values


class HDF5CompressedMatrixSource(CompressedMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        group: str,
        *,
        physical_shape: Sequence[int],
        physical_layout: str,
        physical_order: str,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        dtype: DTypeLike | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.path = validate_hdf5_file(path, limits=limits)
        self.group = "/" + group.strip("/")
        layout = physical_layout.lower()
        if layout not in {"csr", "csc"}:
            raise MatrixSourceError("physical_layout must be 'csr' or 'csc'")
        if physical_order not in {"cell_by_feature", "feature_by_cell"}:
            raise MatrixSourceError(
                "physical_order must be 'cell_by_feature' or 'feature_by_cell'"
            )
        if len(physical_shape) != 2:
            raise MatrixSourceError("physical sparse shape must have length two")
        physical = (int(physical_shape[0]), int(physical_shape[1]))
        if min(physical) < 0:
            raise MatrixSourceError("physical sparse shape cannot be negative")
        cell_by_feature = physical_order == "cell_by_feature"
        logical_shape = _validate_shape(
            physical[::-1] if cell_by_feature else physical, limits
        )
        compressed_axis = physical[0] if layout == "csr" else physical[1]
        minor_axis = physical[1] if layout == "csr" else physical[0]
        with h5py.File(self.path, mode="r") as handle:
            sparse_group = require_hdf5_group(handle, self.group)
            arrays = require_hdf5_datasets(sparse_group, ("data", "indices", "indptr"))
            data = arrays["data"]
            indices = arrays["indices"]
            indptr = arrays["indptr"]
            if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
                raise MatrixSourceError(
                    f"HDF5 sparse arrays under {self.group!r} must be one-dimensional"
                )
            if data.dtype.kind not in "biufc" or data.dtype.hasobject:
                raise TypeError("HDF5 sparse data must have a numeric dtype")
            if indptr.shape != (compressed_axis + 1,):
                raise MatrixSourceError(
                    f"HDF5 sparse indptr has shape {indptr.shape}; "
                    f"expected ({compressed_axis + 1},)"
                )
            nnz = validate_compressed_pointers(
                lambda start, stop: np.asarray(indptr[start:stop]),
                compressed_axis + 1,
                limits,
                label="HDF5 sparse indptr",
            )
            if int(data.size) != nnz or int(indices.size) != nnz:
                raise MatrixSourceError(
                    "HDF5 sparse data, indices, and indptr lengths are inconsistent"
                )
            validate_minor_indexes(
                lambda start, stop: np.asarray(indices[start:stop]),
                nnz,
                minor_axis,
                limits,
                label="HDF5 sparse indices",
            )
            source_dtype = data.dtype if dtype is None else np.dtype(dtype)
        super().__init__(
            logical_shape,
            source_dtype,
            cells_compressed=(layout == "csr") == cell_by_feature,
            nnz=nnz,
            row_names=row_names,
            column_names=column_names,
            limits=limits,
        )

    def _read_pointers(self, start: int, stop: int) -> NDArray[np.int64]:
        with h5py.File(self.path, mode="r") as handle:
            node = require_hdf5_group(handle, self.group)["indptr"]
            assert isinstance(node, h5py.Dataset)
            return np.asarray(node[start:stop], dtype=np.int64)

    def _read_entries(
        self, start: int, stop: int
    ) -> tuple[NDArray[Any], NDArray[np.int64]]:
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            data_node = group["data"]
            index_node = group["indices"]
            assert isinstance(data_node, h5py.Dataset)
            assert isinstance(index_node, h5py.Dataset)
            return (
                np.asarray(data_node[start:stop], dtype=self.dtype),
                np.asarray(index_node[start:stop], dtype=np.int64),
            )


class H5SparseMatrixSource(HDF5CompressedMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        group: str,
        *,
        shape: Sequence[int] | None = None,
        sparse_layout: str | None = None,
        dtype: DTypeLike | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        resolved = validate_hdf5_file(path, limits=limits)
        group_path = "/" + group.strip("/")
        with h5py.File(resolved, mode="r") as handle:
            sparse_group = require_hdf5_group(handle, group_path)
            stored_shape = _shape_from_group(sparse_group)
            if shape is None:
                if stored_shape is None:
                    raise MatrixSourceError(
                        f"HDF5 sparse group {group_path!r} has no shape metadata"
                    )
                physical_shape = stored_shape
            else:
                if len(shape) != 2:
                    raise MatrixSourceError(
                        "H5 sparse logical shape must have length two"
                    )
                physical_shape = (int(shape[1]), int(shape[0]))
                if stored_shape is not None and stored_shape != physical_shape:
                    raise MatrixSourceError(
                        f"HDF5 sparse stored shape {stored_shape} conflicts with "
                        f"logical shape {physical_shape[::-1]}"
                    )
            if sparse_layout is None:
                physical_layout = sparse_encoding(sparse_group)
                if physical_layout is None:
                    pointer = sparse_group.get("indptr")
                    if not isinstance(pointer, h5py.Dataset):
                        raise MatrixSourceError(
                            f"HDF5 sparse group {group_path!r} has no indptr"
                        )
                    csr = int(pointer.size) == physical_shape[0] + 1
                    csc = int(pointer.size) == physical_shape[1] + 1
                    if csr == csc:
                        raise MatrixSourceError(
                            "cannot infer HDF5 sparse physical layout"
                        )
                    physical_layout = "csr" if csr else "csc"
            else:
                logical_layout = sparse_layout.lower()
                if logical_layout not in {"csr", "csc"}:
                    raise MatrixSourceError(
                        "sparse_layout must describe logical CSR or CSC storage"
                    )
                physical_layout = "csc" if logical_layout == "csr" else "csr"
        super().__init__(
            resolved,
            group_path,
            physical_shape=physical_shape,
            physical_layout=physical_layout,
            physical_order="cell_by_feature",
            dtype=dtype,
            limits=limits,
        )


class _DelegatingMatrixSource:
    _delegate: MatrixSource

    @property
    def shape(self) -> tuple[int, int]:
        return self._delegate.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._delegate.dtype

    @property
    def row_names(self) -> tuple[str, ...] | None:
        return self._delegate.row_names

    @property
    def column_names(self) -> tuple[str, ...] | None:
        return self._delegate.column_names

    @property
    def is_sparse(self) -> bool:
        return self._delegate.is_sparse

    @property
    def zero_preserving(self) -> bool:
        return self._delegate.zero_preserving

    @property
    def resident_bytes(self) -> int:
        return self._delegate.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        return self._delegate.estimate_read_memory(start, stop)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        return self._delegate.read_cells(start, stop)


class H5ADMatrixSource(_DelegatingMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        *,
        layer: str | None = None,
        matrix_path: str | None = None,
        dtype: DTypeLike | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        resolved = validate_hdf5_file(path, limits=limits)
        if layer is not None and matrix_path is not None:
            raise ValueError("provide layer or matrix_path, not both")
        resolved_matrix_path = (
            "/" + matrix_path.strip("/")
            if matrix_path is not None
            else ("/X" if layer is None else f"/layers/{layer.strip('/')}")
        )
        with h5py.File(resolved, mode="r") as handle:
            if resolved_matrix_path not in handle:
                raise MatrixSourceError(
                    f"H5AD matrix path {resolved_matrix_path!r} is missing"
                )
            matrix = handle[resolved_matrix_path]
            if isinstance(matrix, h5py.Dataset):
                if matrix.ndim != 2:
                    raise MatrixSourceError(
                        f"H5AD dense matrix {resolved_matrix_path!r} "
                        "must be two-dimensional"
                    )
                physical_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
                layout = None
            elif isinstance(matrix, h5py.Group):
                group_shape = _shape_from_group(matrix)
                if group_shape is None:
                    raise MatrixSourceError(
                        f"H5AD sparse matrix {resolved_matrix_path!r} "
                        "has no shape metadata"
                    )
                physical_shape = group_shape
                layout = sparse_encoding(matrix)
                if layout is None:
                    raise MatrixSourceError(
                        f"H5AD sparse matrix {resolved_matrix_path!r} "
                        "has no CSR/CSC encoding"
                    )
            else:
                raise MatrixSourceError(
                    f"H5AD matrix path {resolved_matrix_path!r} "
                    "has an unsupported node type"
                )
            row_names = read_hdf5_names(
                handle,
                _h5ad_index_path(handle, "/var"),
                physical_shape[1],
                limits=limits,
            )
            column_names = read_hdf5_names(
                handle,
                _h5ad_index_path(handle, "/obs"),
                physical_shape[0],
                limits=limits,
            )
        if layout is None:
            self._delegate = HDF5DenseMatrixSource(
                resolved,
                resolved_matrix_path,
                row_names=row_names,
                column_names=column_names,
                dtype=dtype,
                limits=limits,
            )
        else:
            self._delegate = HDF5CompressedMatrixSource(
                resolved,
                resolved_matrix_path,
                physical_shape=physical_shape,
                physical_layout=layout,
                physical_order="cell_by_feature",
                row_names=row_names,
                column_names=column_names,
                dtype=dtype,
                limits=limits,
            )


class TenXMatrixSource(HDF5CompressedMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        *,
        group: str = "matrix",
        dtype: DTypeLike | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        resolved = validate_hdf5_file(path, limits=limits)
        group_path = "/" + group.strip("/")
        with h5py.File(resolved, mode="r") as handle:
            matrix_group = require_hdf5_group(handle, group_path)
            if "shape" not in matrix_group:
                raise MatrixSourceError(
                    f"10x group {group_path!r} has no shape dataset"
                )
            shape_node = matrix_group["shape"]
            if not isinstance(shape_node, h5py.Dataset):
                raise MatrixSourceError("10x shape path must be a dataset")
            physical_shape = read_hdf5_shape(shape_node[:], f"{group_path}/shape")
            feature_candidates = (
                f"{group_path}/features/name",
                f"{group_path}/features/id",
                f"{group_path}/gene_names",
                f"{group_path}/genes",
            )
            feature_path = next(
                (candidate for candidate in feature_candidates if candidate in handle),
                feature_candidates[0],
            )
            row_names = read_hdf5_names(
                handle,
                feature_path,
                physical_shape[0],
                required=True,
                limits=limits,
            )
            column_names = read_hdf5_names(
                handle,
                f"{group_path}/barcodes",
                physical_shape[1],
                required=True,
                limits=limits,
            )
        super().__init__(
            resolved,
            group_path,
            physical_shape=physical_shape,
            physical_layout="csc",
            physical_order="feature_by_cell",
            row_names=row_names,
            column_names=column_names,
            dtype=dtype,
            limits=limits,
        )
