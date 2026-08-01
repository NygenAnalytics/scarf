import os
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .errors import MatrixSourceError, ResourceLimitError
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
    MatrixBlock,
    MatrixSource,
    MemoryEstimate,
    SourceLimits,
    _validate_shape,
)


def _text_attribute(value: Any, object_path: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise MatrixSourceError(f"HDF5 attribute {object_path} must be scalar")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes | np.bytes_):
        try:
            return bytes(scalar).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MatrixSourceError(
                f"HDF5 attribute {object_path} is not valid UTF-8"
            ) from error
    if isinstance(scalar, str | np.str_):
        return str(scalar)
    raise MatrixSourceError(f"HDF5 attribute {object_path} must be text")


def _shape_from_group(group: h5py.Group) -> tuple[int, int] | None:
    for key in ("shape", "h5sparse_shape", "dim"):
        if key in group.attrs:
            return read_hdf5_shape(group.attrs[key], f"{group.name}@{key}")
        if key in group and isinstance(group[key], h5py.Dataset):
            return read_hdf5_shape(group[key][:], f"{group.name}/{key}")
    return None


def _physical_sparse_layout(group: h5py.Group) -> str | None:
    for key in ("encoding-type", "h5sparse_format", "sparse_layout", "layout"):
        if key not in group.attrs:
            continue
        value = _text_attribute(group.attrs[key], f"{group.name}@{key}").lower()
        if value in {"csr", "csr_matrix", "csr-matrix"}:
            return "csr"
        if value in {"csc", "csc_matrix", "csc-matrix"}:
            return "csc"
    return None


def _h5ad_index_path(
    handle: h5py.File,
    group_path: str,
) -> str | None:
    normalized = "/" + group_path.strip("/")
    if normalized not in handle:
        return None
    node = handle[normalized]
    if not isinstance(node, h5py.Group):
        return None
    key = "_index"
    if "_index" in node.attrs:
        key = _text_attribute(node.attrs["_index"], f"{normalized}@_index")
    if key not in node:
        return None
    return f"{normalized}/{key}"


class HDF5DenseMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        dataset: str,
        *,
        r_transposed: bool = True,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        row_names_path: str | None = None,
        column_names_path: str | None = None,
        dtype: DTypeLike | None = None,
        as_sparse: bool = False,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.path = validate_hdf5_file(path, limits=limits)
        self.dataset = "/" + dataset.strip("/")
        self.rTransposed = bool(r_transposed)
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
            physical_shape = (int(node.shape[0]), int(node.shape[1]))
            logical_shape = (
                (physical_shape[1], physical_shape[0])
                if self.rTransposed
                else physical_shape
            )
            logical_shape = _validate_shape(logical_shape, limits)
            source_dtype = node.dtype if dtype is None else np.dtype(dtype)
            resolved_rows = (
                row_names
                if row_names is not None
                else read_hdf5_names(
                    handle,
                    row_names_path,
                    logical_shape[0],
                    limits=limits,
                )
            )
            resolved_columns = (
                column_names
                if column_names is not None
                else read_hdf5_names(
                    handle,
                    column_names_path,
                    logical_shape[1],
                    limits=limits,
                )
            )
        super().__init__(
            logical_shape,
            source_dtype,
            row_names=resolved_rows,
            column_names=resolved_columns,
            is_sparse=as_sparse,
            limits=limits,
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        if self.is_sparse:
            output += (stop - start + 1 + (stop - start) * self.n_features) * np.dtype(
                np.int64
            ).itemsize
        return MemoryEstimate(self.resident_bytes, output, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        with h5py.File(self.path, mode="r") as handle:
            node = handle[self.dataset]
            assert isinstance(node, h5py.Dataset)
            if self.rTransposed:
                values = np.asarray(node[start:stop, :], dtype=self.dtype)
            else:
                values = np.asarray(node[:, start:stop], dtype=self.dtype).T
        values = np.ascontiguousarray(values)
        if self.is_sparse:
            return csr_matrix(values)
        return values


HDF5ArrayMatrixSource = HDF5DenseMatrixSource
HDF5ArraySource = HDF5DenseMatrixSource


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
        output = (stop - start) * self.n_features * self.dtype.itemsize
        sparse_extra = 0
        if self.is_sparse:
            sparse_extra = (
                stop - start + 1 + (stop - start) * self.n_features
            ) * np.dtype(np.int64).itemsize
        return MemoryEstimate(self.resident_bytes, output + sparse_extra, output)

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


class HDF5CompressedMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        group: str,
        *,
        physical_shape: Sequence[int],
        physical_layout: str,
        physical_order: str,
        data_name: str = "data",
        indices_name: str = "indices",
        indptr_name: str = "indptr",
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
        self.physicalShape = (int(physical_shape[0]), int(physical_shape[1]))
        if min(self.physicalShape) < 0:
            raise MatrixSourceError("physical sparse shape cannot be negative")
        self.physicalLayout = layout
        self.physicalOrder = physical_order
        self.dataName = data_name
        self.indicesName = indices_name
        self.indptrName = indptr_name
        logical_shape = (
            (self.physicalShape[1], self.physicalShape[0])
            if physical_order == "cell_by_feature"
            else self.physicalShape
        )
        logical_shape = _validate_shape(logical_shape, limits)
        with h5py.File(self.path, mode="r") as handle:
            sparse_group = require_hdf5_group(handle, self.group)
            arrays = require_hdf5_datasets(
                sparse_group, (data_name, indices_name, indptr_name)
            )
            data = arrays[data_name]
            indices = arrays[indices_name]
            indptr = arrays[indptr_name]
            if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
                raise MatrixSourceError(
                    f"HDF5 sparse arrays under {self.group!r} must be one-dimensional"
                )
            if data.dtype.kind not in "biufc" or data.dtype.hasobject:
                raise TypeError("HDF5 sparse data must have a numeric dtype")
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("HDF5 sparse indices must contain integers")
            if not np.issubdtype(indptr.dtype, np.integer):
                raise TypeError("HDF5 sparse indptr must contain integers")
            compressed_axis = (
                self.physicalShape[0] if layout == "csr" else self.physicalShape[1]
            )
            if indptr.shape != (compressed_axis + 1,):
                raise MatrixSourceError(
                    f"HDF5 sparse indptr has shape {indptr.shape}; "
                    f"expected ({compressed_axis + 1},)"
                )
            self._nnz = self._validate_structure(data, indices, indptr, limits)
            source_dtype = data.dtype if dtype is None else np.dtype(dtype)
        self._direct = (layout == "csr" and physical_order == "cell_by_feature") or (
            layout == "csc" and physical_order == "feature_by_cell"
        )
        super().__init__(
            logical_shape,
            source_dtype,
            row_names=row_names,
            column_names=column_names,
            is_sparse=True,
            limits=limits,
        )

    def _validate_structure(
        self,
        data: h5py.Dataset,
        indices: h5py.Dataset,
        indptr: h5py.Dataset,
        limits: SourceLimits,
    ) -> int:
        previous: int | None = None
        chunk = max(1, min(limits.compressedChunkNnz, int(indptr.size)))
        final = 0
        for start in range(0, int(indptr.size), chunk):
            stop = min(int(indptr.size), start + chunk)
            pointers = np.asarray(indptr[start:stop])
            if pointers.size > 1 and np.any(pointers[1:] < pointers[:-1]):
                raise MatrixSourceError("HDF5 sparse indptr must be nondecreasing")
            if previous is not None and pointers.size and int(pointers[0]) < previous:
                raise MatrixSourceError("HDF5 sparse indptr must be nondecreasing")
            if start == 0 and (not pointers.size or int(pointers[0]) != 0):
                raise MatrixSourceError("HDF5 sparse indptr must start at zero")
            if pointers.size:
                previous = int(pointers[-1])
                final = previous
        if final < 0:
            raise MatrixSourceError("HDF5 sparse indptr contains negative offsets")
        if final > limits.maxNnz:
            raise ResourceLimitError(
                f"HDF5 sparse nnz {final} exceeds maxNnz={limits.maxNnz}"
            )
        if int(data.size) != final or int(indices.size) != final:
            raise MatrixSourceError(
                "HDF5 sparse data, indices, and indptr lengths are inconsistent"
            )
        minor_axis = (
            self.physicalShape[1]
            if self.physicalLayout == "csr"
            else self.physicalShape[0]
        )
        for start in range(0, final, limits.compressedChunkNnz):
            stop = min(final, start + limits.compressedChunkNnz)
            values = np.asarray(indices[start:stop])
            if values.size and (np.any(values < 0) or np.any(values >= minor_axis)):
                raise MatrixSourceError(
                    "HDF5 sparse indices contain an out-of-range value"
                )
        return final

    @property
    def nnz(self) -> int:
        return self._nnz

    def _direct_bounds(
        self,
        start: int,
        stop: int,
    ) -> tuple[NDArray[np.int64], int, int]:
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            node = group[self.indptrName]
            assert isinstance(node, h5py.Dataset)
            pointers = np.asarray(node[start : stop + 1], dtype=np.int64)
        data_start = int(pointers[0])
        data_stop = int(pointers[-1])
        return pointers - data_start, data_start, data_stop

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        index_size = np.dtype(np.int64).itemsize
        if self._direct:
            pointers, data_start, data_stop = self._direct_bounds(start, stop)
            nnz = data_stop - data_start
            output = nnz * (self.dtype.itemsize + index_size) + pointers.nbytes
            return MemoryEstimate(self.resident_bytes, output, output)
        max_output_nnz = min(self.nnz, (stop - start) * self.n_features)
        output = max_output_nnz * (self.dtype.itemsize + 2 * index_size)
        output += (stop - start + 1) * index_size
        working_nnz = min(self.nnz, self._limits.compressedChunkNnz)
        working = working_nnz * (self.dtype.itemsize + 2 * index_size)
        return MemoryEstimate(self.resident_bytes, working, output)

    def read_cells(self, start: int, stop: int) -> csr_matrix:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        if self._direct:
            return self._read_direct(start, stop)
        return self._read_scanned(start, stop)

    def _read_direct(self, start: int, stop: int) -> csr_matrix:
        pointers, data_start, data_stop = self._direct_bounds(start, stop)
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            data_node = group[self.dataName]
            index_node = group[self.indicesName]
            assert isinstance(data_node, h5py.Dataset)
            assert isinstance(index_node, h5py.Dataset)
            data = np.asarray(data_node[data_start:data_stop], dtype=self.dtype)
            indices = np.asarray(index_node[data_start:data_stop], dtype=np.int64)
        return csr_matrix(
            (data, indices, pointers),
            shape=(stop - start, self.n_features),
            dtype=self.dtype,
        )

    def _read_scanned(self, start: int, stop: int) -> csr_matrix:
        if start == stop:
            return csr_matrix((0, self.n_features), dtype=self.dtype)
        data_parts: list[NDArray[Any]] = []
        row_parts: list[NDArray[np.int64]] = []
        column_parts: list[NDArray[np.int64]] = []
        retained_nnz = 0
        with h5py.File(self.path, mode="r") as handle:
            group = require_hdf5_group(handle, self.group)
            data_node = group[self.dataName]
            index_node = group[self.indicesName]
            pointer_node = group[self.indptrName]
            assert isinstance(data_node, h5py.Dataset)
            assert isinstance(index_node, h5py.Dataset)
            assert isinstance(pointer_node, h5py.Dataset)
            for feature in range(self.n_features):
                bounds = np.asarray(pointer_node[feature : feature + 2], dtype=np.int64)
                vector_start = int(bounds[0])
                vector_stop = int(bounds[1])
                for chunk_start in range(
                    vector_start,
                    vector_stop,
                    self._limits.compressedChunkNnz,
                ):
                    chunk_stop = min(
                        vector_stop,
                        chunk_start + self._limits.compressedChunkNnz,
                    )
                    cell_indexes = np.asarray(
                        index_node[chunk_start:chunk_stop], dtype=np.int64
                    )
                    keep = (cell_indexes >= start) & (cell_indexes < stop)
                    count = int(np.count_nonzero(keep))
                    if count == 0:
                        continue
                    retained_nnz += count
                    required = retained_nnz * (
                        self.dtype.itemsize + 2 * np.dtype(np.int64).itemsize
                    )
                    if required > self._limits.maxBlockBytes:
                        raise ResourceLimitError(
                            "sparse block exceeds "
                            f"maxBlockBytes={self._limits.maxBlockBytes}"
                        )
                    data_parts.append(
                        np.asarray(
                            data_node[chunk_start:chunk_stop],
                            dtype=self.dtype,
                        )[keep]
                    )
                    row_parts.append(cell_indexes[keep] - start)
                    column_parts.append(np.full(count, feature, dtype=np.int64))
        if not data_parts:
            return csr_matrix((stop - start, self.n_features), dtype=self.dtype)
        return coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(stop - start, self.n_features),
            dtype=self.dtype,
        ).tocsr()


class H5SparseMatrixSource(HDF5CompressedMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        group: str,
        *,
        shape: Sequence[int] | None = None,
        sparse_layout: str | None = None,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        row_names_path: str | None = None,
        column_names_path: str | None = None,
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
                logical_shape = (stored_shape[1], stored_shape[0])
            else:
                if len(shape) != 2:
                    raise MatrixSourceError(
                        "H5 sparse logical shape must have length two"
                    )
                logical_shape = (int(shape[0]), int(shape[1]))
                physical_shape = (logical_shape[1], logical_shape[0])
                if stored_shape is not None and stored_shape != physical_shape:
                    raise MatrixSourceError(
                        f"HDF5 sparse stored shape {stored_shape} conflicts with "
                        f"logical shape {logical_shape}"
                    )
            if sparse_layout is None:
                physical_layout = _physical_sparse_layout(sparse_group)
                if physical_layout is None:
                    pointer = sparse_group.get("indptr")
                    if not isinstance(pointer, h5py.Dataset):
                        raise MatrixSourceError(
                            f"HDF5 sparse group {group_path!r} has no indptr"
                        )
                    if int(pointer.size) == physical_shape[0] + 1:
                        physical_layout = "csr"
                    elif int(pointer.size) == physical_shape[1] + 1:
                        physical_layout = "csc"
                    else:
                        raise MatrixSourceError(
                            "cannot infer HDF5 sparse physical layout"
                        )
            else:
                logical_layout = sparse_layout.lower()
                if logical_layout not in {"csr", "csc"}:
                    raise MatrixSourceError(
                        "sparse_layout must describe logical CSR or CSC storage"
                    )
                physical_layout = "csc" if logical_layout == "csr" else "csr"
            resolved_rows = (
                row_names
                if row_names is not None
                else read_hdf5_names(
                    handle,
                    row_names_path,
                    logical_shape[0],
                    limits=limits,
                )
            )
            resolved_columns = (
                column_names
                if column_names is not None
                else read_hdf5_names(
                    handle,
                    column_names_path,
                    logical_shape[1],
                    limits=limits,
                )
            )
        super().__init__(
            resolved,
            group_path,
            physical_shape=physical_shape,
            physical_layout=physical_layout,
            physical_order="cell_by_feature",
            row_names=resolved_rows,
            column_names=resolved_columns,
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
    def rowNames(self) -> tuple[str, ...] | None:
        return self.row_names

    @property
    def columnNames(self) -> tuple[str, ...] | None:
        return self.column_names

    @property
    def n_features(self) -> int:
        return self.shape[0]

    @property
    def n_cells(self) -> int:
        return self.shape[1]

    @property
    def is_sparse(self) -> bool:
        return self._delegate.is_sparse

    @property
    def sparse(self) -> bool:
        return self.is_sparse

    @property
    def zero_preserving(self) -> bool:
        return self._delegate.zero_preserving

    @property
    def zeroPreserving(self) -> bool:
        return self.zero_preserving

    @property
    def resident_bytes(self) -> int:
        return self._delegate.resident_bytes

    @property
    def residentBytes(self) -> int:
        return self.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        return self._delegate.estimate_read_memory(start, stop)

    def memory_estimate(self, start: int, stop: int) -> MemoryEstimate:
        return self.estimate_read_memory(start, stop)

    def estimate_read_bytes(self, start: int, stop: int) -> int:
        return self.estimate_read_memory(start, stop).peakBytes

    def estimate_memory(self, start: int, stop: int) -> MemoryEstimate:
        return self.estimate_read_memory(start, stop)

    def estimated_peak_bytes(self, start: int, stop: int) -> int:
        return self.estimate_read_bytes(start, stop)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        return self._delegate.read_cells(start, stop)


class H5ADMatrixSource(_DelegatingMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        *,
        layer: str | None = None,
        matrix_path: str | None = None,
        row_names_path: str | None = None,
        column_names_path: str | None = None,
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
                layout = _physical_sparse_layout(matrix)
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
            logical_shape = (physical_shape[1], physical_shape[0])
            feature_names_path = (
                row_names_path
                if row_names_path is not None
                else _h5ad_index_path(handle, "/var")
            )
            cell_names_path = (
                column_names_path
                if column_names_path is not None
                else _h5ad_index_path(handle, "/obs")
            )
            row_names = read_hdf5_names(
                handle,
                feature_names_path,
                logical_shape[0],
                limits=limits,
            )
            column_names = read_hdf5_names(
                handle,
                cell_names_path,
                logical_shape[1],
                limits=limits,
            )
        if layout is None:
            self._delegate = HDF5DenseMatrixSource(
                resolved,
                resolved_matrix_path,
                r_transposed=True,
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


H5adMatrixSource = H5ADMatrixSource


class TenXMatrixSource(HDF5CompressedMatrixSource):
    def __init__(
        self,
        path: str | os.PathLike[str] | Any,
        *,
        group: str = "matrix",
        feature_names: str = "name",
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
                f"{group_path}/features/{feature_names}",
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


TENxMatrixSource = TenXMatrixSource
TenxMatrixSource = TenXMatrixSource
