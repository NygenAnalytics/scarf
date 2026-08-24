from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import spmatrix

from .errors import (
    MatrixSourceError,
    ResourceLimitError,
    UnsupportedMatrixOperation,
)
from .fragments import build_fragment_matrix_source
from .sources import (
    DEFAULT_LIMITS,
    BaseMatrixSource,
    CellBindMatrixSource,
    DtypeMatrixSource,
    FeatureBindMatrixSource,
    MappedMatrixSource,
    MatrixBlock,
    MatrixSource,
    MemoryEstimate,
    RenamedMatrixSource,
    SourceLimits,
    TransposeMatrixSource,
    _block_to_csr,
    _block_to_dense,
    _normalize_indexes,
    _read_selected_cells,
)


type NumericScalar = int | float | complex | np.number[Any]


def _is_numeric_scalar(value: object) -> TypeGuard[NumericScalar]:
    return isinstance(value, int | float | complex | np.number) and not isinstance(
        value, bool | np.bool_
    )


class MatrixOperation(str, Enum):
    SUBSET = "subset"
    TRANSPOSE = "transpose"
    APERM = "aperm"
    ROW_BIND = "rbind"
    COLUMN_BIND = "cbind"
    RENAME = "dimnames"
    DTYPE = "dtype"
    UNARY = "unary"
    BINARY = "binary"
    MASK = "mask"
    SUBASSIGNMENT = "subassignment"
    RANK = "rank"
    MULTIPLY = "multiply"
    FRAGMENT = "fragment-derived"


@dataclass(frozen=True)
class OperationCapability:
    operation: MatrixOperation
    aliases: tuple[str, ...]
    local: bool


@dataclass(frozen=True)
class _UnaryKernel:
    function: Any
    zeroPreserving: bool


def _round(values: NDArray[Any]) -> NDArray[Any]:
    return np.round(values)


_UNARY_KERNELS: dict[str, _UnaryKernel] = {
    "identity": _UnaryKernel(np.positive, True),
    "positive": _UnaryKernel(np.positive, True),
    "+": _UnaryKernel(np.positive, True),
    "abs": _UnaryKernel(np.abs, True),
    "absolute": _UnaryKernel(np.abs, True),
    "negative": _UnaryKernel(np.negative, True),
    "neg": _UnaryKernel(np.negative, True),
    "sqrt": _UnaryKernel(np.sqrt, True),
    "square": _UnaryKernel(np.square, True),
    "sign": _UnaryKernel(np.sign, True),
    "log1p": _UnaryKernel(np.log1p, True),
    "expm1": _UnaryKernel(np.expm1, True),
    "sin": _UnaryKernel(np.sin, True),
    "sinh": _UnaryKernel(np.sinh, True),
    "asin": _UnaryKernel(np.arcsin, True),
    "asinh": _UnaryKernel(np.arcsinh, True),
    "tan": _UnaryKernel(np.tan, True),
    "tanh": _UnaryKernel(np.tanh, True),
    "atan": _UnaryKernel(np.arctan, True),
    "atanh": _UnaryKernel(np.arctanh, True),
    "floor": _UnaryKernel(np.floor, True),
    "ceil": _UnaryKernel(np.ceil, True),
    "ceiling": _UnaryKernel(np.ceil, True),
    "trunc": _UnaryKernel(np.trunc, True),
    "round": _UnaryKernel(_round, True),
    "log": _UnaryKernel(np.log, False),
    "log2": _UnaryKernel(np.log2, False),
    "log10": _UnaryKernel(np.log10, False),
    "exp": _UnaryKernel(np.exp, False),
    "cos": _UnaryKernel(np.cos, False),
    "cosh": _UnaryKernel(np.cosh, False),
    "acos": _UnaryKernel(np.arccos, False),
    "acosh": _UnaryKernel(np.arccosh, False),
    "!": _UnaryKernel(np.logical_not, False),
    "is.na": _UnaryKernel(np.isnan, True),
    "is.nan": _UnaryKernel(np.isnan, True),
    "is.finite": _UnaryKernel(np.isfinite, False),
    "is.infinite": _UnaryKernel(np.isinf, True),
    "reciprocal": _UnaryKernel(np.reciprocal, False),
}


_BINARY_KERNELS: dict[str, Any] = {
    "add": np.add,
    "+": np.add,
    "subtract": np.subtract,
    "sub": np.subtract,
    "-": np.subtract,
    "multiply": np.multiply,
    "mul": np.multiply,
    "*": np.multiply,
    "divide": np.divide,
    "true_divide": np.divide,
    "/": np.divide,
    "floor_divide": np.floor_divide,
    "%/%": np.floor_divide,
    "remainder": np.remainder,
    "%%": np.remainder,
    "power": np.power,
    "pow": np.power,
    "^": np.power,
    "minimum": np.minimum,
    "pmin": np.minimum,
    "maximum": np.maximum,
    "pmax": np.maximum,
    "greater": np.greater,
    ">": np.greater,
    "greater_equal": np.greater_equal,
    ">=": np.greater_equal,
    "less": np.less,
    "<": np.less,
    "less_equal": np.less_equal,
    "<=": np.less_equal,
    "equal": np.equal,
    "==": np.equal,
    "not_equal": np.not_equal,
    "!=": np.not_equal,
    "logical_and": np.logical_and,
    "&": np.logical_and,
    "logical_or": np.logical_or,
    "|": np.logical_or,
}


def _result_dtype(
    function: Any, left: np.dtype[Any], right: Any = None
) -> np.dtype[Any]:
    with np.errstate(all="ignore"):
        left_value = np.zeros(1, dtype=left)
        result = (
            function(left_value)
            if right is None
            else function(left_value, np.asarray([right]))
        )
    dtype = np.asarray(result).dtype
    if dtype.kind not in "biufc":
        raise TypeError(f"operation produces unsupported dtype {dtype}")
    return dtype


def _zero_result(function: Any, right: Any = None) -> bool:
    with np.errstate(all="ignore"):
        result = (
            function(np.asarray([0.0]))
            if right is None
            else function(np.asarray([0.0]), np.asarray([right]))
        )
    value = np.asarray(result).reshape(-1)[0]
    return bool(np.isfinite(value) and value == 0)


class UnaryTransformMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        operation: str,
        *,
        dtype: DTypeLike | None = None,
        parameter: int | None = None,
        object_path: str = "$",
        class_name: str | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if not isinstance(operation, str):
            raise UnsupportedMatrixOperation(
                object_path,
                repr(operation),
                class_name,
                "custom functions are not executed",
            )
        normalized = operation.lower()
        if normalized not in _UNARY_KERNELS:
            raise UnsupportedMatrixOperation(
                object_path, operation, class_name, "unknown unary function"
            )
        self.source = source
        self.operation = normalized
        self.kernel = (
            _UnaryKernel(
                lambda values: np.round(values, decimals=parameter),
                True,
            )
            if normalized == "round" and parameter is not None
            else _UNARY_KERNELS[normalized]
        )
        result_dtype = (
            _result_dtype(self.kernel.function, source.dtype)
            if dtype is None
            else np.dtype(dtype)
        )
        zero_preserving = source.zero_preserving and self.kernel.zeroPreserving
        super().__init__(
            source.shape,
            result_dtype,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=source.is_sparse and zero_preserving,
            zero_preserving=zero_preserving,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + self.source.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child = self.source.estimate_read_memory(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child.peakBytes, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        block = self.source.read_cells(start, stop)
        with np.errstate(all="ignore"):
            if self.is_sparse:
                result = _block_to_csr(block).astype(self.dtype, copy=True)
                result.data = np.asarray(
                    self.kernel.function(result.data), dtype=self.dtype
                )
                result.eliminate_zeros()
                return result
            values = _block_to_dense(block)
            return np.asarray(self.kernel.function(values), dtype=self.dtype)


UnaryMatrixSource = UnaryTransformMatrixSource
LocalUnaryMatrixSource = UnaryTransformMatrixSource


class BinaryTransformMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        left: MatrixSource,
        right: MatrixSource | NumericScalar,
        operation: str,
        *,
        dtype: DTypeLike | None = None,
        reverse: bool = False,
        object_path: str = "$",
        class_name: str | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if not isinstance(operation, str):
            raise UnsupportedMatrixOperation(
                object_path,
                repr(operation),
                class_name,
                "custom functions are not executed",
            )
        normalized = operation.lower()
        if normalized not in _BINARY_KERNELS:
            raise UnsupportedMatrixOperation(
                object_path, operation, class_name, "unknown binary function"
            )
        if not isinstance(right, MatrixSource) and not _is_numeric_scalar(right):
            raise TypeError(
                "binary right operand must be a MatrixSource or numeric scalar"
            )
        self.left = left
        self.right = right
        self.operation = normalized
        self.kernel = _BINARY_KERNELS[normalized]
        self.reverse = bool(reverse)
        if self.reverse and isinstance(right, MatrixSource):
            raise MatrixSourceError(
                "reversed binary operations require a scalar operand"
            )
        if isinstance(right, MatrixSource):
            if right.shape != left.shape:
                raise MatrixSourceError(
                    f"binary source shapes differ: {left.shape} and {right.shape}"
                )
            if (
                left.row_names is not None
                and right.row_names is not None
                and left.row_names != right.row_names
            ):
                raise MatrixSourceError("binary source row names conflict")
            if (
                left.column_names is not None
                and right.column_names is not None
                and left.column_names != right.column_names
            ):
                raise MatrixSourceError("binary source column names conflict")
            right_sample: Any = np.zeros(1, dtype=right.dtype)
            with np.errstate(all="ignore"):
                zero_value = self.kernel(np.asarray([0.0]), np.asarray([0.0]))[0]
            zero_preserving = bool(np.isfinite(zero_value) and zero_value == 0)
            sparse = left.is_sparse and right.is_sparse and zero_preserving
        else:
            right_sample = right
            zero_preserving = (
                _zero_result(self.kernel, right)
                if not self.reverse
                else _zero_result(lambda values: self.kernel(right, values))
            )
            sparse = left.is_sparse and zero_preserving
        if dtype is None:
            with np.errstate(all="ignore"):
                left_sample = np.zeros(1, dtype=left.dtype)
                result_dtype = np.asarray(
                    self.kernel(right_sample, left_sample)
                    if self.reverse
                    else self.kernel(left_sample, right_sample)
                ).dtype
        else:
            result_dtype = np.dtype(dtype)
        super().__init__(
            left.shape,
            result_dtype,
            row_names=left.row_names,
            column_names=left.column_names,
            is_sparse=sparse,
            zero_preserving=(
                left.zero_preserving
                and (right.zero_preserving if isinstance(right, MatrixSource) else True)
                and zero_preserving
            ),
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        right_bytes = (
            self.right.resident_bytes if isinstance(self.right, MatrixSource) else 0
        )
        return super().resident_bytes + self.left.resident_bytes + right_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child = self.left.estimate_read_memory(start, stop).peakBytes
        if isinstance(self.right, MatrixSource):
            child += self.right.estimate_read_memory(start, stop).peakBytes
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        left_block = self.left.read_cells(start, stop)
        right_block = (
            self.right.read_cells(start, stop)
            if isinstance(self.right, MatrixSource)
            else self.right
        )
        with np.errstate(all="ignore"):
            if self.is_sparse:
                left_sparse = _block_to_csr(left_block, dtype=self.dtype)
                if isinstance(self.right, MatrixSource):
                    right_sparse = _block_to_csr(right_block, dtype=self.dtype)
                    if self.operation in {"add", "+"}:
                        result = left_sparse + right_sparse
                    elif self.operation in {"subtract", "sub", "-"}:
                        result = left_sparse - right_sparse
                    elif self.operation in {"multiply", "mul", "*"}:
                        result = left_sparse.multiply(right_sparse)
                    elif self.operation in {"minimum", "pmin"}:
                        result = left_sparse.minimum(right_sparse)
                    elif self.operation in {"maximum", "pmax"}:
                        result = left_sparse.maximum(right_sparse)
                    else:
                        dense = self.kernel(
                            left_sparse.toarray(), right_sparse.toarray()
                        )
                        return _block_to_csr(
                            np.asarray(dense, dtype=self.dtype),
                            dtype=self.dtype,
                        )
                    result = result.tocsr().astype(self.dtype, copy=False)
                    result.eliminate_zeros()
                    return result
                result = left_sparse.copy()
                result.data = np.asarray(
                    (
                        self.kernel(self.right, result.data)
                        if self.reverse
                        else self.kernel(result.data, self.right)
                    ),
                    dtype=self.dtype,
                )
                result.eliminate_zeros()
                return result
            left_dense = _block_to_dense(left_block)
            right_dense = (
                _block_to_dense(right_block)
                if isinstance(self.right, MatrixSource)
                else self.right
            )
            return np.asarray(
                (
                    self.kernel(right_dense, left_dense)
                    if self.reverse
                    else self.kernel(left_dense, right_dense)
                ),
                dtype=self.dtype,
            )


BinaryMatrixSource = BinaryTransformMatrixSource
LocalBinaryMatrixSource = BinaryTransformMatrixSource


class MaskMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        mask: MatrixSource,
        *,
        fill_value: int | float | complex = 0,
        keep_nonzero: bool = True,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if source.shape != mask.shape:
            raise MatrixSourceError(
                f"mask shape {mask.shape} does not match source {source.shape}"
            )
        if not np.isscalar(fill_value):
            raise TypeError("mask fill_value must be numeric scalar")
        self.source = source
        self.mask = mask
        self.fillValue = fill_value
        self.keepNonzero = bool(keep_nonzero)
        dtype = np.result_type(source.dtype, type(fill_value))
        sparse = source.is_sparse and fill_value == 0
        super().__init__(
            source.shape,
            dtype,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=sparse,
            zero_preserving=source.zero_preserving and fill_value == 0,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return (
            super().resident_bytes
            + self.source.resident_bytes
            + self.mask.resident_bytes
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child = self.source.estimate_read_memory(start, stop).peakBytes
        child += self.mask.estimate_read_memory(start, stop).peakBytes
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child, output)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        values = self.source.read_cells(start, stop)
        mask_block = self.mask.read_cells(start, stop)
        if self.is_sparse:
            source_sparse = _block_to_csr(values, dtype=self.dtype)
            if self.keepNonzero:
                selected_mask: Any = _block_to_csr(mask_block, dtype=bool)
            else:
                selected_mask = np.logical_not(_block_to_dense(mask_block, dtype=bool))
            result = source_sparse.multiply(selected_mask).tocsr()
            result.eliminate_zeros()
            return result
        dense = _block_to_dense(values, dtype=self.dtype)
        mask_dense = _block_to_dense(mask_block, dtype=bool)
        if not self.keepNonzero:
            mask_dense = np.logical_not(mask_dense)
        return np.where(mask_dense, dense, self.fillValue).astype(
            self.dtype, copy=False
        )


@dataclass(frozen=True, init=False)
class Subassignment:
    featureIndices: Sequence[int] | NDArray[Any]
    cellIndices: Sequence[int] | NDArray[Any]
    value: MatrixSource | NDArray[Any] | NumericScalar

    def __init__(
        self,
        feature_indices: Sequence[int] | NDArray[Any] | None = None,
        cell_indices: Sequence[int] | NDArray[Any] | None = None,
        value: MatrixSource | NDArray[Any] | NumericScalar | None = None,
        *,
        featureIndices: Sequence[int] | NDArray[Any] | None = None,
        cellIndices: Sequence[int] | NDArray[Any] | None = None,
    ) -> None:
        if feature_indices is not None and featureIndices is not None:
            raise TypeError("provide only one feature index spelling")
        if cell_indices is not None and cellIndices is not None:
            raise TypeError("provide only one cell index spelling")
        resolved_features = (
            feature_indices if featureIndices is None else featureIndices
        )
        resolved_cells = cell_indices if cellIndices is None else cellIndices
        if resolved_features is None or resolved_cells is None or value is None:
            raise TypeError(
                "subassignment requires feature indexes, cell indexes, and value"
            )
        object.__setattr__(self, "featureIndices", resolved_features)
        object.__setattr__(self, "cellIndices", resolved_cells)
        object.__setattr__(self, "value", value)

    @property
    def feature_indices(self) -> Sequence[int] | NDArray[Any]:
        return self.featureIndices

    @property
    def cell_indices(self) -> Sequence[int] | NDArray[Any]:
        return self.cellIndices


@dataclass(frozen=True)
class _ResolvedSubassignment:
    featureIndices: NDArray[np.int64]
    cellIndices: NDArray[np.int64]
    value: MatrixSource | NDArray[Any] | NumericScalar


class DelayedSubassignmentMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        assignments: Sequence[Subassignment],
        *,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if not assignments:
            raise ValueError("delayed subassignment requires at least one assignment")
        self.source = source
        resolved: list[_ResolvedSubassignment] = []
        dtypes: list[np.dtype[Any]] = [source.dtype]
        for index, assignment in enumerate(assignments):
            features = _normalize_indexes(
                assignment.featureIndices, source.shape[0], "feature"
            )
            cells = _normalize_indexes(assignment.cellIndices, source.shape[1], "cell")
            value = assignment.value
            if callable(value):
                raise TypeError("subassignment values cannot be callable")
            if isinstance(value, MatrixSource):
                expected = (features.size, cells.size)
                if value.shape != expected:
                    raise MatrixSourceError(
                        f"subassignment {index} source has shape {value.shape}; "
                        f"expected {expected}"
                    )
                dtypes.append(value.dtype)
            elif _is_numeric_scalar(value):
                dtypes.append(np.asarray(value).dtype)
            else:
                array = np.asarray(value)
                if array.shape != (features.size, cells.size):
                    raise MatrixSourceError(
                        f"subassignment {index} array has shape {array.shape}; "
                        f"expected {(features.size, cells.size)}"
                    )
                value = array
                dtypes.append(array.dtype)
            resolved.append(_ResolvedSubassignment(features, cells, value))
        self.assignments = tuple(resolved)
        super().__init__(
            source.shape,
            np.result_type(*dtypes),
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=False,
            zero_preserving=False,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        mappings = sum(
            assignment.featureIndices.nbytes + assignment.cellIndices.nbytes
            for assignment in self.assignments
        )
        arrays = sum(
            assignment.value.nbytes
            for assignment in self.assignments
            if isinstance(assignment.value, np.ndarray)
        )
        source_values = sum(
            assignment.value.resident_bytes
            for assignment in self.assignments
            if isinstance(assignment.value, MatrixSource)
        )
        return int(
            super().resident_bytes
            + self.source.resident_bytes
            + mappings
            + arrays
            + source_values
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        child = self.source.estimate_read_memory(start, stop).peakBytes
        for assignment in self.assignments:
            if not isinstance(assignment.value, MatrixSource):
                continue
            selected = np.flatnonzero(
                (assignment.cellIndices >= start) & (assignment.cellIndices < stop)
            )
            child += sum(
                assignment.value.estimate_read_memory(
                    int(position), int(position) + 1
                ).peakBytes
                for position in selected
            )
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(self.resident_bytes, child, output)

    def read_cells(self, start: int, stop: int) -> NDArray[Any]:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        output = _block_to_dense(
            self.source.read_cells(start, stop), dtype=self.dtype
        ).copy()
        for assignment in self.assignments:
            selected_positions: NDArray[np.int64] = np.flatnonzero(
                (assignment.cellIndices >= start) & (assignment.cellIndices < stop)
            ).astype(np.int64, copy=False)
            if selected_positions.size == 0 or assignment.featureIndices.size == 0:
                continue
            output_rows = (assignment.cellIndices[selected_positions] - start).astype(
                np.int64, copy=False
            )
            value = assignment.value
            if isinstance(value, MatrixSource):
                assignment_values = _block_to_dense(
                    _read_selected_cells(value, selected_positions),
                    dtype=self.dtype,
                )
            elif _is_numeric_scalar(value):
                assignment_values = np.asarray(value, dtype=self.dtype)
            else:
                assert isinstance(value, np.ndarray)
                assignment_values = np.asarray(
                    value[:, selected_positions], dtype=self.dtype
                ).T
            output[np.ix_(output_rows, assignment.featureIndices)] = assignment_values
        return output


SubassignmentMatrixSource = DelayedSubassignmentMatrixSource


def _parameter_vector(
    values: Any,
    length: int,
    *,
    name: str,
    limits: SourceLimits,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (length,):
        raise MatrixSourceError(f"{name} has length {result.size}; expected {length}")
    if result.nbytes > limits.maxMetadataBytes:
        raise ResourceLimitError(
            f"{name} exceeds maxMetadataBytes={limits.maxMetadataBytes}"
        )
    return result


class AxisMinimumMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        parameters: Any,
        *,
        axis: str,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if axis not in {"feature", "cell"}:
            raise MatrixSourceError("minimum axis must be feature or cell")
        self.source = source
        self.axis = axis
        self.parameters = _parameter_vector(
            parameters,
            source.shape[0] if axis == "feature" else source.shape[1],
            name=f"{axis} minimum parameters",
            limits=limits,
        )
        if np.any(~np.isfinite(self.parameters)) or np.any(self.parameters <= 0):
            raise MatrixSourceError("minimum parameters must be finite and positive")
        super().__init__(
            source.shape,
            np.float64,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=source.is_sparse,
            zero_preserving=True,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return (
            super().resident_bytes + self.source.resident_bytes + self.parameters.nbytes
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        source = self.source.estimate_read_memory(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(
            self.resident_bytes,
            source.workingBytes + source.outputBytes + output,
            output,
        )

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        self._admit(self.estimate_read_memory(start, stop))
        block = self.source.read_cells(start, stop)
        if isinstance(block, spmatrix):
            result = block.tocsr(copy=True).astype(np.float64)
            if self.axis == "feature":
                result.data = np.minimum(
                    result.data,
                    self.parameters[result.indices],
                )
            else:
                for row in range(result.shape[0]):
                    row_start = int(result.indptr[row])
                    row_stop = int(result.indptr[row + 1])
                    result.data[row_start:row_stop] = np.minimum(
                        result.data[row_start:row_stop],
                        self.parameters[start + row],
                    )
            result.eliminate_zeros()
            return result
        values = np.asarray(block, dtype=np.float64)
        parameters = (
            self.parameters[np.newaxis, :]
            if self.axis == "feature"
            else self.parameters[start:stop, np.newaxis]
        )
        return np.minimum(values, parameters)


class ScaleShiftMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        feature_scale: Any | None = None,
        cell_scale: Any | None = None,
        global_scale: float = 1.0,
        feature_shift: Any | None = None,
        cell_shift: Any | None = None,
        global_shift: float = 0.0,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self.featureScale = (
            None
            if feature_scale is None
            else _parameter_vector(
                feature_scale,
                source.shape[0],
                name="feature scale parameters",
                limits=limits,
            )
        )
        self.cellScale = (
            None
            if cell_scale is None
            else _parameter_vector(
                cell_scale,
                source.shape[1],
                name="cell scale parameters",
                limits=limits,
            )
        )
        self.featureShift = (
            None
            if feature_shift is None
            else _parameter_vector(
                feature_shift,
                source.shape[0],
                name="feature shift parameters",
                limits=limits,
            )
        )
        self.cellShift = (
            None
            if cell_shift is None
            else _parameter_vector(
                cell_shift,
                source.shape[1],
                name="cell shift parameters",
                limits=limits,
            )
        )
        self.globalScale = float(global_scale)
        self.globalShift = float(global_shift)
        shifts_zero = self.globalShift == 0 and all(
            values is None or not np.any(values)
            for values in (self.featureShift, self.cellShift)
        )
        super().__init__(
            source.shape,
            np.float64,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=source.is_sparse and shifts_zero,
            zero_preserving=shifts_zero,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        parameters = sum(
            0 if values is None else values.nbytes
            for values in (
                self.featureScale,
                self.cellScale,
                self.featureShift,
                self.cellShift,
            )
        )
        return super().resident_bytes + self.source.resident_bytes + parameters

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        source = self.source.estimate_read_memory(start, stop)
        dense_output = (stop - start) * self.n_features * self.dtype.itemsize
        output = source.outputBytes if self.is_sparse else dense_output
        return MemoryEstimate(
            self.resident_bytes,
            source.workingBytes + source.outputBytes + dense_output,
            output,
        )

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        self._admit(self.estimate_read_memory(start, stop))
        block = self.source.read_cells(start, stop)
        feature_scale: float | NDArray[np.float64] = self.globalScale
        if self.featureScale is not None:
            feature_scale = self.featureScale * feature_scale
        if self.is_sparse and isinstance(block, spmatrix):
            result = block.tocsr(copy=True).astype(np.float64)
            result.data *= (
                feature_scale
                if isinstance(feature_scale, float)
                else feature_scale[result.indices]
            )
            if self.cellScale is not None:
                for row in range(result.shape[0]):
                    row_start = int(result.indptr[row])
                    row_stop = int(result.indptr[row + 1])
                    result.data[row_start:row_stop] *= self.cellScale[start + row]
            result.eliminate_zeros()
            return result
        result = _block_to_dense(block, dtype=np.float64)
        result *= feature_scale
        if self.cellScale is not None:
            result *= self.cellScale[start:stop, np.newaxis]
        if self.featureShift is not None:
            result += self.featureShift
        if self.cellShift is not None:
            result += self.cellShift[start:stop, np.newaxis]
        result += self.globalShift
        return result


class PearsonResidualMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        theta_inverse: Any,
        gene_beta: Any,
        cell_read_counts: Any,
        global_parameters: Any,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self.thetaInverse = _parameter_vector(
            theta_inverse,
            source.shape[0],
            name="SCTransform inverse theta",
            limits=limits,
        )
        self.geneBeta = _parameter_vector(
            gene_beta,
            source.shape[0],
            name="SCTransform gene beta",
            limits=limits,
        )
        self.cellReadCounts = _parameter_vector(
            cell_read_counts,
            source.shape[1],
            name="SCTransform cell read counts",
            limits=limits,
        )
        global_values = np.asarray(global_parameters, dtype=np.float64).reshape(-1)
        if global_values.shape != (3,):
            raise MatrixSourceError(
                "SCTransform global parameters must contain three values"
            )
        self.sdInverseMax = float(global_values[0])
        self.clipMinimum = float(global_values[1])
        self.clipMaximum = float(global_values[2])
        if self.clipMinimum > self.clipMaximum:
            raise MatrixSourceError("SCTransform clip bounds are reversed")
        super().__init__(
            source.shape,
            np.float64,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=False,
            zero_preserving=False,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return (
            super().resident_bytes
            + self.source.resident_bytes
            + self.thetaInverse.nbytes
            + self.geneBeta.nbytes
            + self.cellReadCounts.nbytes
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        source = self.source.estimate_read_memory(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(
            self.resident_bytes,
            source.workingBytes + source.outputBytes + 3 * output,
            output,
        )

    def read_cells(self, start: int, stop: int) -> NDArray[np.float64]:
        start, stop = self._window(start, stop)
        self._admit(self.estimate_read_memory(start, stop))
        values = _block_to_dense(
            self.source.read_cells(start, stop),
            dtype=np.float64,
        )
        mu = self.cellReadCounts[start:stop, np.newaxis] * self.geneBeta[np.newaxis, :]
        variance = mu + mu * mu * self.thetaInverse[np.newaxis, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse_sd = np.minimum(self.sdInverseMax, 1.0 / np.sqrt(variance))
            result = (values - mu) * inverse_sd
        return np.clip(result, self.clipMinimum, self.clipMaximum)


class LinearResidualMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        feature_parameters: Any,
        cell_parameters: Any,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        self.source = source
        self.featureParameters = np.asarray(
            feature_parameters,
            dtype=np.float64,
        )
        self.cellParameters = np.asarray(cell_parameters, dtype=np.float64)
        if (
            self.featureParameters.ndim != 2
            or self.cellParameters.ndim != 2
            or self.featureParameters.shape[1] != source.shape[0]
            or self.cellParameters.shape[1] != source.shape[1]
            or self.featureParameters.shape[0] != self.cellParameters.shape[0]
        ):
            raise MatrixSourceError(
                "linear residual parameters do not match the matrix shape"
            )
        parameter_bytes = self.featureParameters.nbytes + self.cellParameters.nbytes
        if parameter_bytes > limits.maxMetadataBytes:
            raise ResourceLimitError(
                "linear residual parameters exceed "
                f"maxMetadataBytes={limits.maxMetadataBytes}"
            )
        super().__init__(
            source.shape,
            np.float64,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=False,
            zero_preserving=False,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return (
            super().resident_bytes
            + self.source.resident_bytes
            + self.featureParameters.nbytes
            + self.cellParameters.nbytes
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        source = self.source.estimate_read_memory(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        return MemoryEstimate(
            self.resident_bytes,
            source.workingBytes + source.outputBytes + 2 * output,
            output,
        )

    def read_cells(self, start: int, stop: int) -> NDArray[np.float64]:
        start, stop = self._window(start, stop)
        self._admit(self.estimate_read_memory(start, stop))
        values = _block_to_dense(
            self.source.read_cells(start, stop),
            dtype=np.float64,
        )
        prediction = self.cellParameters[:, start:stop].T @ self.featureParameters
        return values - prediction


class UnsupportedExecutionMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource | None,
        operation: str,
        *,
        object_path: str,
        class_name: str | None,
        reason: str,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if source is None and (shape is None or dtype is None):
            raise TypeError("unsupported source requires a source or shape and dtype")
        resolved_shape = source.shape if shape is None and source is not None else shape
        resolved_dtype = source.dtype if dtype is None and source is not None else dtype
        if resolved_shape is None or resolved_dtype is None:
            raise TypeError("unsupported source metadata is incomplete")
        self.source = source
        self.operation = operation
        self.objectPath = object_path
        self.className = class_name
        self.reason = reason
        super().__init__(
            resolved_shape,
            resolved_dtype,
            row_names=(
                source.row_names
                if row_names is None and source is not None
                else row_names
            ),
            column_names=(
                source.column_names
                if column_names is None and source is not None
                else column_names
            ),
            is_sparse=source.is_sparse if source is not None else True,
            zero_preserving=(source.zero_preserving if source is not None else True),
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        source_bytes = 0 if self.source is None else self.source.resident_bytes
        return super().resident_bytes + source_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        self._window(start, stop)
        if self.source is None:
            return MemoryEstimate(self.resident_bytes)
        return self.source.estimate_read_memory(start, stop)

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        self._window(start, stop)
        raise UnsupportedMatrixOperation(
            self.objectPath,
            self.operation,
            self.className,
            self.reason,
        )


def _rank_values(values: NDArray[Any], total_size: int) -> NDArray[np.float64]:
    numeric = np.asarray(values)
    if numeric.ndim != 1 or numeric.size > total_size:
        raise MatrixSourceError("rank input has an invalid shape")
    if not np.all(np.isfinite(numeric)):
        raise MatrixSourceError("rank input contains non-finite values")
    implicit_zeros = total_size - int(numeric.size)
    explicit_zeros = int(np.count_nonzero(numeric == 0))
    negative_count = int(np.count_nonzero(numeric < 0))
    zero_count = implicit_zeros + explicit_zeros
    zero_rank = negative_count + (1 + zero_count) / 2.0 if zero_count else 0.0
    order = np.argsort(numeric, kind="stable")
    sorted_values = numeric[order]
    sorted_ranks = np.empty(numeric.size, dtype=np.float64)
    start = 0
    while start < numeric.size:
        stop = start + 1
        while stop < numeric.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        value = sorted_values[start]
        if value == 0:
            rank = 0.0
        else:
            rank = (start + 1 + stop) / 2.0 - zero_rank
            if value > 0:
                rank += implicit_zeros
        sorted_ranks[start:stop] = rank
        start = stop
    result = np.empty(numeric.size, dtype=np.float64)
    result[order] = sorted_ranks
    return result


class RankMatrixSource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        axis: str = "column",
        object_path: str = "$",
        class_name: str | None = "RankMatrix",
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if axis not in {"column", "row"}:
            raise MatrixSourceError("rank axis must be 'column' or 'row'")
        self.source = source
        self.axis = axis
        self.operation = "rank"
        self.objectPath = object_path
        self.className = class_name
        super().__init__(
            source.shape,
            np.float64,
            row_names=source.row_names,
            column_names=source.column_names,
            is_sparse=source.is_sparse,
            zero_preserving=True,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return super().resident_bytes + self.source.resident_bytes

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        source = self.source.estimate_read_memory(start, stop)
        output = (stop - start) * self.n_features * self.dtype.itemsize
        if self.axis == "row":
            scan_peak = 0
            for scan_start in range(0, self.n_cells, self._limits.tileCells):
                scan_stop = min(scan_start + self._limits.tileCells, self.n_cells)
                scan = self.source.estimate_read_memory(
                    scan_start,
                    scan_stop,
                )
                scan_peak = max(
                    scan_peak,
                    scan.workingBytes + scan.outputBytes,
                )
            counters = 2 * output
            sparse_conversion = output if self.source.is_sparse else 0
            return MemoryEstimate(
                self.resident_bytes,
                source.workingBytes
                + source.outputBytes
                + scan_peak
                + counters
                + sparse_conversion,
                output,
            )
        return MemoryEstimate(
            self.resident_bytes,
            source.workingBytes + source.outputBytes + output,
            output,
        )

    def read_cells(self, start: int, stop: int) -> MatrixBlock:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        block = self.source.read_cells(start, stop)
        if self.axis == "row":
            target = _block_to_dense(block)
            if not np.all(np.isfinite(target)):
                raise MatrixSourceError("rank input contains non-finite values")
            less = np.zeros(target.shape, dtype=np.int64)
            equal = np.zeros(target.shape, dtype=np.int64)
            negative_count = np.zeros(self.n_features, dtype=np.int64)
            zero_count = np.zeros(self.n_features, dtype=np.int64)
            for scan_start in range(0, self.n_cells, self._limits.tileCells):
                scan_stop = min(scan_start + self._limits.tileCells, self.n_cells)
                scan_block = self.source.read_cells(scan_start, scan_stop)
                if isinstance(scan_block, spmatrix):
                    sparse_block = scan_block.tocsc(copy=False)
                    for feature in range(self.n_features):
                        data_start = int(sparse_block.indptr[feature])
                        data_stop = int(sparse_block.indptr[feature + 1])
                        values = np.asarray(sparse_block.data[data_start:data_stop])
                        if not np.all(np.isfinite(values)):
                            raise MatrixSourceError(
                                "rank input contains non-finite values"
                            )
                        ordered = np.sort(values)
                        implicit_zeros = (scan_stop - scan_start) - len(values)
                        targets = target[:, feature]
                        left = np.searchsorted(ordered, targets, side="left")
                        right = np.searchsorted(ordered, targets, side="right")
                        less[:, feature] += left
                        less[:, feature] += implicit_zeros * (targets > 0)
                        equal[:, feature] += right - left
                        equal[:, feature] += implicit_zeros * (targets == 0)
                        negative_count[feature] += int(
                            np.searchsorted(ordered, 0, side="left")
                        )
                        zero_count[feature] += int(
                            np.searchsorted(ordered, 0, side="right")
                            - np.searchsorted(ordered, 0, side="left")
                            + implicit_zeros
                        )
                else:
                    dense_block = np.asarray(scan_block)
                    if not np.all(np.isfinite(dense_block)):
                        raise MatrixSourceError("rank input contains non-finite values")
                    for feature in range(self.n_features):
                        ordered = np.sort(dense_block[:, feature])
                        targets = target[:, feature]
                        left = np.searchsorted(ordered, targets, side="left")
                        right = np.searchsorted(ordered, targets, side="right")
                        less[:, feature] += left
                        equal[:, feature] += right - left
                        negative_count[feature] += int(
                            np.searchsorted(ordered, 0, side="left")
                        )
                        zero_count[feature] += int(
                            np.searchsorted(ordered, 0, side="right")
                            - np.searchsorted(ordered, 0, side="left")
                        )
            standard_rank = less + (equal + 1) / 2.0
            zero_rank = np.where(
                zero_count > 0,
                negative_count + (zero_count + 1) / 2.0,
                0.0,
            )
            result = np.where(
                target == 0,
                0.0,
                standard_rank - zero_rank[np.newaxis, :],
            )
            return _block_to_csr(result) if self.source.is_sparse else result
        if isinstance(block, spmatrix):
            result = block.tocsr(copy=True).astype(np.float64)
            for row in range(result.shape[0]):
                row_start = int(result.indptr[row])
                row_stop = int(result.indptr[row + 1])
                result.data[row_start:row_stop] = _rank_values(
                    result.data[row_start:row_stop],
                    self.n_features,
                )
            result.eliminate_zeros()
            return result
        values = np.asarray(block)
        output = np.empty(values.shape, dtype=np.float64)
        for row in range(values.shape[0]):
            output[row] = _rank_values(values[row], self.n_features)
        return output


class MatrixMultiplySource(BaseMatrixSource):
    def __init__(
        self,
        source: MatrixSource,
        *,
        right: MatrixSource | None = None,
        shape: Sequence[int] | None = None,
        row_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        column_names: Sequence[str | bytes] | NDArray[Any] | None = None,
        object_path: str = "$",
        class_name: str | None = "MatrixMultiply",
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> None:
        if right is None:
            raise MatrixSourceError(
                f"matrix multiplication at {object_path} requires a right operand"
            )
        self.right = right
        self.source = source
        self.operation = "multiply"
        self.objectPath = object_path
        self.className = class_name
        if source.shape[1] != right.shape[0]:
            raise MatrixSourceError(
                "matrix multiplication inner dimensions do not match"
            )
        inferred_shape = (source.shape[0], right.shape[1])
        if shape is not None and tuple(int(value) for value in shape) != inferred_shape:
            raise MatrixSourceError(
                f"matrix multiplication shape {tuple(shape)} does not match "
                f"inferred shape {inferred_shape}"
            )
        super().__init__(
            inferred_shape,
            np.result_type(source.dtype, right.dtype),
            row_names=source.row_names if row_names is None else row_names,
            column_names=right.column_names if column_names is None else column_names,
            is_sparse=False,
            zero_preserving=True,
            limits=limits,
        )

    @property
    def resident_bytes(self) -> int:
        return (
            super().resident_bytes
            + self.source.resident_bytes
            + self.right.resident_bytes
        )

    def estimate_read_memory(self, start: int, stop: int) -> MemoryEstimate:
        start, stop = self._window(start, stop)
        rows = stop - start
        inner = self.source.shape[1]
        output = rows * self.n_features * self.dtype.itemsize
        right_output = rows * inner * self.dtype.itemsize
        right_estimate = self.right.estimate_read_memory(start, stop)
        left_peak = 0
        for inner_start in range(0, inner, self._limits.tileCells):
            inner_stop = min(inner_start + self._limits.tileCells, inner)
            estimate = self.source.estimate_read_memory(inner_start, inner_stop)
            left_output = (
                (inner_stop - inner_start) * self.n_features * self.dtype.itemsize
            )
            left_peak = max(
                left_peak,
                estimate.workingBytes + max(estimate.outputBytes, left_output),
            )
        working = (
            right_estimate.workingBytes
            + max(right_estimate.outputBytes, right_output)
            + left_peak
            + output
        )
        return MemoryEstimate(self.resident_bytes, working, output)

    def read_cells(self, start: int, stop: int) -> NDArray[Any]:
        start, stop = self._window(start, stop)
        estimate = self.estimate_read_memory(start, stop)
        self._admit(estimate)
        right_values = _block_to_dense(
            self.right.read_cells(start, stop),
            dtype=self.dtype,
        )
        output = np.zeros((stop - start, self.n_features), dtype=self.dtype)
        for inner_start in range(0, self.source.shape[1], self._limits.tileCells):
            inner_stop = min(
                inner_start + self._limits.tileCells,
                self.source.shape[1],
            )
            left_values = _block_to_dense(
                self.source.read_cells(inner_start, inner_stop),
                dtype=self.dtype,
            )
            output += right_values[:, inner_start:inner_stop] @ left_values
        return output


MultiplyMatrixSource = MatrixMultiplySource


_CAPABILITIES = (
    OperationCapability(MatrixOperation.SUBSET, ("subset", "[", "extract"), True),
    OperationCapability(MatrixOperation.TRANSPOSE, ("transpose", "t"), True),
    OperationCapability(MatrixOperation.APERM, ("aperm",), True),
    OperationCapability(
        MatrixOperation.ROW_BIND, ("rbind", "row_bind", "feature_bind"), True
    ),
    OperationCapability(
        MatrixOperation.COLUMN_BIND, ("cbind", "column_bind", "cell_bind"), True
    ),
    OperationCapability(
        MatrixOperation.RENAME, ("dimnames", "rename", "set_dimnames"), True
    ),
    OperationCapability(MatrixOperation.DTYPE, ("dtype", "cast", "convert_type"), True),
    OperationCapability(MatrixOperation.UNARY, ("unary", "unary_transform"), True),
    OperationCapability(MatrixOperation.BINARY, ("binary", "binary_transform"), True),
    OperationCapability(MatrixOperation.MASK, ("mask",), True),
    OperationCapability(
        MatrixOperation.SUBASSIGNMENT,
        ("subassignment", "subassign", "[<-"),
        True,
    ),
    OperationCapability(MatrixOperation.RANK, ("rank",), False),
    OperationCapability(
        MatrixOperation.MULTIPLY,
        ("multiply", "matrix_multiply", "matmul"),
        False,
    ),
    OperationCapability(
        MatrixOperation.FRAGMENT,
        ("fragment-derived", "fragment_matrix"),
        False,
    ),
)


_KNOWN_CLASSES = frozenset(
    {
        "DelayedArray",
        "DelayedMatrix",
        "HDF5Array",
        "HDF5Matrix",
        "H5SparseMatrix",
        "H5ADMatrix",
        "TENxMatrix",
        "DelayedOp",
        "DelayedUnaryOp",
        "DelayedUnaryIsoOp",
        "DelayedNaryOp",
        "DelayedSubset",
        "DelayedAperm",
        "SeedDimPicker",
        "DelayedAbind",
        "SeedBinder",
        "DelayedSetDimnames",
        "DelayedUnaryIsoOpStack",
        "DelayedUnaryIsoOpWithArgs",
        "DelayedNaryIsoOp",
        "DelayedSubassign",
        "IterableMatrix",
        "TransformedMatrix",
        "TransformLog1p",
        "TransformLog1pSlow",
        "TransformExpm1",
        "TransformExpm1Slow",
        "TransformAbs",
        "TransformNegate",
        "TransformSign",
        "TransformSqrt",
        "TransformSquare",
        "TransformPow",
        "TransformMin",
        "TransformMinByRow",
        "TransformMinByCol",
        "TransformBinarize",
        "TransformRound",
        "TransformScaleShift",
        "SCTransformPearson",
        "SCTransformPearsonSlow",
        "SCTransformPearsonTranspose",
        "SCTransformPearsonTransposeSlow",
        "TransformLinearResidual",
        "MatrixAddition",
        "MatrixMask",
        "MatrixRankTransform",
        "MatrixSubset",
        "RenameDims",
        "RowBindMatrices",
        "ColBindMatrices",
        "ConvertMatrixType",
        "PeakMatrix",
        "TileMatrix",
        "SubsetMatrix",
        "TransposeMatrix",
        "ApermMatrix",
        "RowBindMatrix",
        "ColumnBindMatrix",
        "RenameMatrix",
        "DtypeMatrix",
        "UnaryMatrix",
        "BinaryMatrix",
        "MaskMatrix",
        "SubassignmentMatrix",
        "RankMatrix",
        "MatrixMultiply",
        "FragmentMatrix",
        "BPCellsMatrix",
    }
)


class MatrixOperationRegistry:
    def __init__(self) -> None:
        aliases: dict[str, MatrixOperation] = {}
        for capability in _CAPABILITIES:
            for alias in capability.aliases:
                aliases[alias.lower()] = capability.operation
        self._aliases = aliases
        self.capabilities = {
            capability.operation: capability for capability in _CAPABILITIES
        }

    def resolve(
        self,
        operation: Any,
        *,
        object_path: str,
        class_name: str | None,
    ) -> MatrixOperation:
        if not isinstance(operation, str):
            raise UnsupportedMatrixOperation(
                object_path,
                repr(operation),
                class_name,
                "custom functions are not executed",
            )
        normalized = operation.lower()
        if normalized not in self._aliases:
            raise UnsupportedMatrixOperation(
                object_path, operation, class_name, "unknown operation"
            )
        return self._aliases[normalized]

    def build(
        self,
        specification: Mapping[str, Any],
        *,
        object_path: str = "$",
        source: MatrixSource | None = None,
        limits: SourceLimits = DEFAULT_LIMITS,
    ) -> MatrixSource:
        if not isinstance(specification, Mapping):
            raise TypeError("matrix operation specification must be a mapping")
        operation_value = specification.get("operation", specification.get("op"))
        class_value = specification.get("className", specification.get("class"))
        class_names: tuple[str, ...]
        if class_value is None:
            class_names = ()
        elif isinstance(class_value, str):
            class_names = (class_value,)
        elif isinstance(class_value, Sequence) and not isinstance(class_value, bytes):
            class_names = tuple(str(value) for value in class_value)
            if not class_names:
                raise UnsupportedMatrixOperation(
                    object_path,
                    str(operation_value),
                    None,
                    "empty class vector",
                )
        else:
            class_names = (str(class_value),)
        class_name = class_names[0] if class_names else None
        unknown_classes = [
            value for value in class_names if value not in _KNOWN_CLASSES
        ]
        if unknown_classes:
            raise UnsupportedMatrixOperation(
                object_path,
                str(operation_value),
                unknown_classes[0],
                "unknown or custom class",
            )
        operation = self.resolve(
            operation_value,
            object_path=object_path,
            class_name=class_name,
        )
        if operation in {MatrixOperation.ROW_BIND, MatrixOperation.COLUMN_BIND}:
            inputs = specification.get("sources")
            if not isinstance(inputs, Sequence) or isinstance(inputs, str | bytes):
                raise MatrixSourceError(
                    f"bind operation at {object_path} requires a source sequence"
                )
            sources = tuple(inputs)
            if not all(isinstance(item, MatrixSource) for item in sources):
                raise TypeError("bind sources must implement MatrixSource")
            return (
                FeatureBindMatrixSource(sources, limits=limits)
                if operation == MatrixOperation.ROW_BIND
                else CellBindMatrixSource(sources, limits=limits)
            )
        primary = specification.get("source", source)
        if operation == MatrixOperation.FRAGMENT:
            return build_fragment_matrix_source(
                specification,
                object_path=object_path,
                limits=limits,
            )
        if not isinstance(primary, MatrixSource):
            raise MatrixSourceError(
                f"matrix operation at {object_path} has no MatrixSource input"
            )
        if operation == MatrixOperation.SUBSET:
            return MappedMatrixSource(
                primary,
                feature_indices=specification.get(
                    "featureIndices",
                    specification.get(
                        "featureIndexes",
                        specification.get(
                            "feature_indices",
                            specification.get(
                                "feature_indexes", specification.get("rows")
                            ),
                        ),
                    ),
                ),
                cell_indices=specification.get(
                    "cellIndices",
                    specification.get(
                        "cellIndexes",
                        specification.get(
                            "cell_indices",
                            specification.get(
                                "cell_indexes", specification.get("columns")
                            ),
                        ),
                    ),
                ),
                limits=limits,
            )
        if operation in {MatrixOperation.TRANSPOSE, MatrixOperation.APERM}:
            permutation = specification.get("permutation", (1, 0))
            normalized_permutation = tuple(int(value) for value in permutation)
            if normalized_permutation in {(0, 1), (1, 2)}:
                return primary
            if normalized_permutation not in {(1, 0), (2, 1)}:
                raise UnsupportedMatrixOperation(
                    object_path,
                    operation.value,
                    class_name,
                    f"2D permutation {normalized_permutation!r} is invalid",
                )
            return TransposeMatrixSource(primary, limits=limits)
        if operation == MatrixOperation.RENAME:
            return RenamedMatrixSource(
                primary,
                row_names=specification.get("rowNames", specification.get("row_names")),
                column_names=specification.get(
                    "columnNames", specification.get("column_names")
                ),
                limits=limits,
            )
        if operation == MatrixOperation.DTYPE:
            if "dtype" not in specification:
                raise MatrixSourceError(
                    f"dtype operation at {object_path} has no dtype"
                )
            return DtypeMatrixSource(primary, specification["dtype"], limits=limits)
        if operation == MatrixOperation.UNARY:
            function = specification.get("function", specification.get("name"))
            return UnaryTransformMatrixSource(
                primary,
                function,
                dtype=specification.get("dtype"),
                parameter=specification.get("parameter"),
                object_path=object_path,
                class_name=class_name,
                limits=limits,
            )
        if operation == MatrixOperation.BINARY:
            if "right" not in specification:
                raise MatrixSourceError(
                    f"binary operation at {object_path} has no right operand"
                )
            function = specification.get("function", specification.get("name"))
            return BinaryTransformMatrixSource(
                primary,
                specification["right"],
                function,
                dtype=specification.get("dtype"),
                reverse=bool(specification.get("reverse", False)),
                object_path=object_path,
                class_name=class_name,
                limits=limits,
            )
        if operation == MatrixOperation.MASK:
            mask = specification.get("mask")
            if not isinstance(mask, MatrixSource):
                raise MatrixSourceError(
                    f"mask operation at {object_path} has no MatrixSource mask"
                )
            return MaskMatrixSource(
                primary,
                mask,
                fill_value=specification.get(
                    "fillValue", specification.get("fill_value", 0)
                ),
                keep_nonzero=specification.get(
                    "keepNonzero",
                    specification.get(
                        "keep_nonzero",
                        (
                            bool(specification.get("invert", False))
                            if class_name == "MatrixMask"
                            else True
                        ),
                    ),
                ),
                limits=limits,
            )
        if operation == MatrixOperation.SUBASSIGNMENT:
            raw_assignments = specification.get("assignments")
            if not isinstance(raw_assignments, Sequence) or isinstance(
                raw_assignments, str | bytes
            ):
                raise MatrixSourceError(
                    f"subassignment at {object_path} requires assignments"
                )
            assignments: list[Subassignment] = []
            for index, assignment in enumerate(raw_assignments):
                if isinstance(assignment, Subassignment):
                    assignments.append(assignment)
                    continue
                if not isinstance(assignment, Mapping):
                    raise TypeError(
                        f"subassignment {index} at {object_path} must be a mapping"
                    )
                try:
                    assignments.append(
                        Subassignment(
                            assignment.get(
                                "featureIndices", assignment.get("feature_indices")
                            ),
                            assignment.get(
                                "cellIndices", assignment.get("cell_indices")
                            ),
                            assignment.get("value"),
                        )
                    )
                except TypeError as error:
                    raise MatrixSourceError(
                        f"subassignment {index} at {object_path} is incomplete"
                    ) from error
            return DelayedSubassignmentMatrixSource(primary, assignments, limits=limits)
        if operation == MatrixOperation.RANK:
            return RankMatrixSource(
                primary,
                axis=str(specification.get("axis", "column")),
                object_path=object_path,
                class_name=class_name,
                limits=limits,
            )
        if operation == MatrixOperation.MULTIPLY:
            right = specification.get("right")
            if right is not None and not isinstance(right, MatrixSource):
                raise TypeError(
                    "matrix multiplication right operand must implement MatrixSource"
                )
            return MatrixMultiplySource(
                primary,
                right=right,
                shape=specification.get(
                    "shape",
                    specification.get("Dim", specification.get("dim")),
                ),
                row_names=specification.get("rowNames", specification.get("row_names")),
                column_names=specification.get(
                    "columnNames", specification.get("column_names")
                ),
                object_path=object_path,
                class_name=class_name,
                limits=limits,
            )
        raise AssertionError(f"unhandled matrix operation {operation}")


DEFAULT_OPERATION_REGISTRY = MatrixOperationRegistry()


def build_matrix_operation(
    specification: Mapping[str, Any],
    *,
    object_path: str = "$",
    source: MatrixSource | None = None,
    limits: SourceLimits = DEFAULT_LIMITS,
) -> MatrixSource:
    return DEFAULT_OPERATION_REGISTRY.build(
        specification,
        object_path=object_path,
        source=source,
        limits=limits,
    )
