import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bpcells import (
    BPCellsDirectoryMatrixSource,
    BPCellsHDF5MatrixSource,
    BPCellsMemoryMatrixSource,
)
from .errors import (
    MatrixSourceError,
    UnsafeSidecarError,
    UnsupportedMatrixOperation,
)
from .fragments import (
    FRAGMENT_CAPABILITY_REGISTRY,
    FragmentSource,
    fragment_source_from_slots,
)
from .hdf5 import (
    H5ADMatrixSource,
    H5SparseMatrixSource,
    HDF5DenseMatrixSource,
    ReshapedHDF5ArrayMatrixSource,
    TenXMatrixSource,
)
from .operations import (
    AxisMinimumMatrixSource,
    LinearResidualMatrixSource,
    PearsonResidualMatrixSource,
    ScaleShiftMatrixSource,
    Subassignment,
    build_matrix_operation,
)
from .paths import SidecarPathResolver
from .sources import (
    DEFAULT_LIMITS,
    CscMatrixSource,
    DenseMatrixSource,
    MatrixSource,
    RenamedMatrixSource,
    SourceLimits,
    TransposeMatrixSource,
)
from .values import (
    class_names,
    decode_text,
    logical_scalar,
    read_bounded,
    scalar_value,
    shape_value,
    vector_length,
)


_DENSE_CLASSES = frozenset(
    {
        "matrix",
        "array",
        "denseMatrix",
        "dgeMatrix",
        "lgeMatrix",
        "ngeMatrix",
        "igeMatrix",
    }
)
_CSC_CLASSES = frozenset(
    {
        "CsparseMatrix",
        "dgCMatrix",
        "lgCMatrix",
        "ngCMatrix",
        "igCMatrix",
    }
)
_BPCELLS_MEMORY_CLASSES: dict[str, tuple[str, str]] = {
    "PackedMatrixMem_uint32_t": ("packed", "uint"),
    "PackedMatrixMem_float": ("packed", "float"),
    "PackedMatrixMem_double": ("packed", "double"),
    "UnpackedMatrixMem_uint32_t": ("unpacked", "uint"),
    "UnpackedMatrixMem_float": ("unpacked", "float"),
    "UnpackedMatrixMem_double": ("unpacked", "double"),
}
_HDF5_DENSE_CLASSES = frozenset({"HDF5ArraySeed", "Dense_H5ADArraySeed"})
_H5_SPARSE_CLASSES = frozenset(
    {
        "H5SparseMatrixSeed",
        "CSC_H5SparseMatrixSeed",
        "CSR_H5SparseMatrixSeed",
    }
)
_H5AD_CLASSES = frozenset(
    {
        "H5ADMatrixSeed",
        "Dense_H5ADMatrixSeed",
        "CSC_H5ADMatrixSeed",
        "CSR_H5ADMatrixSeed",
        "AnnDataMatrixH5",
    }
)
_TENX_CLASSES = frozenset({"TENxMatrixSeed", "10xMatrixH5"})
_WRAPPER_CLASSES = frozenset(
    {
        "DelayedArray",
        "DelayedMatrix",
        "HDF5Array",
        "HDF5Matrix",
        "H5SparseMatrix",
        "H5ADMatrix",
        "TENxMatrix",
    }
)
_RENAME_CLASSES = frozenset({"DelayedSetDimnames", "RenameDims"})
_INHERIT_DIMNAMES = -1


def _first_value(slots: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in slots:
            return slots[name]
    raise MatrixSourceError(f"matrix slots are missing one of {names!r}")


def _optional_value(
    slots: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if name in slots:
            return slots[name]
    return default


def _bounded_array(
    value: Any,
    *,
    object_path: str,
    limits: SourceLimits,
    dtype: Any = None,
) -> NDArray[Any]:
    """Materialize a serialized vector, rejecting lengths beyond the budget."""
    if isinstance(value, np.ndarray | list | tuple):
        values = np.asarray(value)
        return values if dtype is None else values.astype(dtype, copy=False)
    return read_bounded(
        value,
        max_length=max(1, limits.maxMetadataBytes // 8),
        object_path=object_path,
        dtype=dtype,
    )


def _slot_text(value: Any, *, slot_name: str, object_path: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray) or vector_length(value, object_path) != 1:
        raise MatrixSourceError(f"{slot_name} at {object_path} must contain one string")
    try:
        return decode_text(
            scalar_value(value, object_path), f"{slot_name} at {object_path}"
        )
    except TypeError as error:
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must contain one string"
        ) from error


def _slot_flag(slots: Mapping[str, Any], name: str, object_path: str) -> bool:
    value = slots.get(name)
    return False if value is None else logical_scalar(value, f"{name} at {object_path}")


def _parameter_matrix(
    slots: Mapping[str, Any],
    name: str,
    *,
    object_path: str,
    limits: SourceLimits,
) -> NDArray[Any]:
    value = slots.get(name)
    if value is None:
        return np.empty((0, 0), dtype=np.float64)
    values = _bounded_array(value, object_path=f"{object_path}@{name}", limits=limits)
    if values.ndim == 1:
        if values.size == 0:
            return np.empty((0, 0), dtype=values.dtype)
        return values.reshape(1, -1)
    if values.ndim != 2:
        raise MatrixSourceError(
            f"{name} at {object_path} must be a two-dimensional matrix"
        )
    return values


def _parameter_vector(
    slots: Mapping[str, Any],
    name: str,
    *,
    object_path: str,
    max_length: int,
) -> NDArray[Any]:
    value = slots.get(name)
    if value is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(value, np.ndarray | list | tuple):
        values = np.asarray(value).reshape(-1)
        if values.size > max_length:
            raise MatrixSourceError(
                f"{name} at {object_path} has more than {max_length} values"
            )
        return values
    return read_bounded(
        value, max_length=max_length, object_path=f"{name} at {object_path}"
    )


def _operation_arguments(
    value: Any,
    *,
    object_path: str,
) -> tuple[int | float | complex, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw_values = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        raw_values = tuple(value)
    else:
        raw_values = (value,)
    output: list[int | float | complex] = []
    for index, raw in enumerate(raw_values):
        try:
            scalar = scalar_value(raw, f"{object_path}[{index}]")
        except MatrixSourceError as error:
            raise UnsupportedMatrixOperation(
                f"{object_path}[{index}]",
                "delayed-function-argument",
                None,
                "only scalar numeric arguments are supported",
            ) from error
        if isinstance(scalar, bool | np.bool_) or not isinstance(
            scalar,
            int | float | complex | np.number,
        ):
            raise UnsupportedMatrixOperation(
                f"{object_path}[{index}]",
                "delayed-function-argument",
                None,
                "only scalar numeric arguments are supported",
            )
        output.append(scalar.item() if isinstance(scalar, np.generic) else scalar)
    return tuple(output)


def _dimnames(
    slots: Mapping[str, Any],
    object_path: str,
) -> tuple[Any, Any]:
    """Return a node's row and column names, each a vector, ``None``, or a marker.

    A rename node marks an inherited axis with -1, which ``_renamed_axis``
    resolves. A mapping is rejected here, where the names enter.
    """
    values = _optional_value(slots, "Dimnames", "dimnames")
    if values is None:
        return None, None
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or len(values) != 2
    ):
        raise MatrixSourceError("dimnames must contain row and column names")
    for axis, names in zip(("row", "column"), values, strict=True):
        if isinstance(names, Mapping):
            raise MatrixSourceError(
                f"{axis} names in dimnames at {object_path} must be a vector, "
                "not a mapping"
            )
    return values[0], values[1]


def _renamed_axis(value: Any, inherited: tuple[str, ...] | None) -> Any:
    """Return a rename node's names; DelayedArray marks inherited axes with -1."""
    if isinstance(value, int | np.integer) and int(value) == _INHERIT_DIMNAMES:
        return inherited
    if (
        value is not None
        and not isinstance(value, str | bytes)
        and getattr(value, "dtype", np.dtype(object)).kind in "iu"
        and vector_length(value, "dimnames") == 1
        and int(scalar_value(value, "dimnames")) == _INHERIT_DIMNAMES
    ):
        return inherited
    return value


def _r_indexes(
    value: Any,
    *,
    slot_name: str,
    object_path: str,
    limits: SourceLimits,
) -> NDArray[np.int64] | None:
    """Convert a serialized one-based R index vector to zero-based indexes."""
    if value is None:
        return None
    indexes = _bounded_array(
        value, object_path=f"{slot_name} at {object_path}", limits=limits
    )
    if indexes.size == 0:
        return np.empty(0, dtype=np.int64)
    if indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer):
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must be a one-dimensional integer vector"
        )
    indexes = indexes.astype(np.int64, copy=False)
    if np.any(indexes <= 0):
        raise MatrixSourceError(
            f"{slot_name} at {object_path} contains a missing or nonpositive R index"
        )
    return indexes - 1


def _r_scalar_integer(value: Any, *, slot_name: str, object_path: str) -> int:
    try:
        scalar = scalar_value(value, object_path)
    except MatrixSourceError as error:
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must contain one integer"
        ) from error
    if isinstance(scalar, bool | np.bool_) or not isinstance(scalar, int | np.integer):
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must contain one integer"
        )
    return int(scalar)


def _sidecar_path(
    value: Any,
    *,
    sidecar_root: str | os.PathLike[str] | None,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None,
    expect: str,
    object_path: str,
) -> Path:
    if sidecar_root is None:
        raise UnsafeSidecarError(
            f"sidecar path at {object_path} needs an anchor directory"
        )
    path = (
        value
        if isinstance(value, os.PathLike)
        else _slot_text(value, slot_name="sidecar path", object_path=object_path)
    )
    return SidecarPathResolver(
        sidecar_root,
        absolute_prefix_remaps=absolute_prefix_remaps,
    ).resolve(path, expect=expect)


def _finalize_slot_source(
    source: MatrixSource,
    slots: Mapping[str, Any],
    row_names: Any,
    column_names: Any,
    limits: SourceLimits,
    *,
    stored_row_major: bool | None = None,
) -> MatrixSource:
    """Orient a node's source and apply its dimensions and names.

    ``stored_row_major`` describes how a stored leaf lays out its values. A leaf
    is transposed when the node's ``transpose`` flag disagrees with it. Every
    other source is already in the node's logical orientation.
    """
    transpose_value = slots.get("transpose")
    transpose = (
        False
        if transpose_value is None
        else logical_scalar(transpose_value, "transpose slot")
    )
    if stored_row_major is not None and transpose != stored_row_major:
        source = TransposeMatrixSource(source, limits=limits)
    dimensions = _optional_value(slots, "Dim", "dim")
    if dimensions is not None:
        expected_shape = shape_value(dimensions, "dim slot")
        if source.shape != expected_shape:
            raise MatrixSourceError(
                f"matrix source shape {source.shape} does not match dim slot "
                f"{expected_shape}"
            )
    if row_names is not None or column_names is not None:
        source = RenamedMatrixSource(
            source,
            row_names=source.row_names if row_names is None else row_names,
            column_names=source.column_names if column_names is None else column_names,
            limits=limits,
        )
    return source


_UNARY_TRANSFORM_CLASSES = {
    "TransformAbs": "abs",
    "TransformExpm1": "expm1",
    "TransformExpm1Slow": "expm1",
    "TransformLog1p": "log1p",
    "TransformLog1pSlow": "log1p",
    "TransformNegate": "negative",
    "TransformRound": "round",
    "TransformSign": "sign",
    "TransformSqrt": "sqrt",
    "TransformSquare": "square",
}


def _inferred_operation_spec(
    primary_class: str,
    classes: tuple[str, ...],
    slots: Mapping[str, Any],
    *,
    object_path: str,
    resolve_source: Callable[[Any, str], MatrixSource],
    resolve_fragment: Callable[[Any, str], FragmentSource],
    limits: SourceLimits,
) -> dict[str, Any] | None:
    base: dict[str, Any] = {"className": classes}
    transposed = _slot_flag(slots, "transpose", object_path)
    if primary_class == "DelayedSubset":
        source = resolve_source(
            _first_value(slots, "seed"),
            f"{object_path}@seed",
        )
        index = _first_value(slots, "index")
        if not isinstance(index, Sequence) or isinstance(
            index, (str, bytes, bytearray)
        ):
            raise TypeError(f"index at {object_path} must be a sequence")
        if len(index) != 2:
            raise UnsupportedMatrixOperation(
                object_path,
                "subset",
                primary_class,
                "only two-dimensional DelayedSubset nodes are supported",
            )
        return {
            **base,
            "operation": "subset",
            "source": source,
            "featureIndices": _r_indexes(
                index[0],
                slot_name="index[[1]]",
                object_path=object_path,
                limits=limits,
            ),
            "cellIndices": _r_indexes(
                index[1],
                slot_name="index[[2]]",
                object_path=object_path,
                limits=limits,
            ),
        }
    if primary_class in {"DelayedAperm", "SeedDimPicker"}:
        permutation = _bounded_array(
            _first_value(slots, "perm", "dim_combination"),
            object_path=f"perm at {object_path}",
            limits=limits,
        )
        if (
            permutation.ndim != 1
            or permutation.size != 2
            or not np.issubdtype(permutation.dtype, np.integer)
            or np.any(permutation <= 0)
        ):
            raise UnsupportedMatrixOperation(
                object_path,
                "aperm",
                primary_class,
                "only two-dimensional permutations without missing axes are supported",
            )
        return {
            **base,
            "operation": "aperm",
            "source": resolve_source(
                _first_value(slots, "seed"),
                f"{object_path}@seed",
            ),
            "permutation": tuple(int(value) for value in permutation),
        }
    if primary_class in {"DelayedAbind", "SeedBinder"}:
        values = _first_value(slots, "seeds")
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise TypeError(f"seeds at {object_path} must be a sequence")
        sources = [
            resolve_source(value, f"{object_path}@seeds[{index}]")
            for index, value in enumerate(values)
        ]
        along = _r_scalar_integer(
            _first_value(slots, "along"),
            slot_name="along",
            object_path=object_path,
        )
        if along not in {1, 2}:
            raise UnsupportedMatrixOperation(
                object_path,
                "abind",
                primary_class,
                "only two-dimensional row or column binding is supported",
            )
        return {
            **base,
            "operation": "feature_bind" if along == 1 else "cell_bind",
            "sources": sources,
        }
    if primary_class in _RENAME_CLASSES:
        source = resolve_source(
            _first_value(
                slots, "seed" if primary_class == "DelayedSetDimnames" else "matrix"
            ),
            f"{object_path}@seed",
        )
        rows, columns = _dimnames(slots, object_path)
        return {
            **base,
            "operation": "rename",
            "source": source,
            "rowNames": _renamed_axis(rows, source.row_names),
            "columnNames": _renamed_axis(columns, source.column_names),
        }
    if primary_class == "DelayedSubassign":
        index = _first_value(slots, "Lindex")
        if not isinstance(index, Sequence) or isinstance(
            index, (str, bytes, bytearray)
        ):
            raise TypeError(f"Lindex at {object_path} must be a sequence")
        if len(index) != 2:
            raise UnsupportedMatrixOperation(
                object_path,
                "subassignment",
                primary_class,
                "only two-dimensional DelayedSubassign nodes are supported",
            )
        source = resolve_source(
            _first_value(slots, "seed"),
            f"{object_path}@seed",
        )
        feature_indices = _r_indexes(
            index[0],
            slot_name="Lindex[[1]]",
            object_path=object_path,
            limits=limits,
        )
        cell_indices = _r_indexes(
            index[1],
            slot_name="Lindex[[2]]",
            object_path=object_path,
            limits=limits,
        )
        if feature_indices is None:
            feature_indices = np.arange(source.shape[0], dtype=np.int64)
        if cell_indices is None:
            cell_indices = np.arange(source.shape[1], dtype=np.int64)
        replacement = _first_value(slots, "Rvalue")
        if not isinstance(replacement, MatrixSource):
            try:
                replacement = scalar_value(replacement, f"{object_path}@Rvalue")
            except MatrixSourceError as error:
                raise UnsupportedMatrixOperation(
                    object_path,
                    "subassignment",
                    primary_class,
                    "ordinary replacement values must contain one numeric scalar",
                ) from error
        return {
            **base,
            "operation": "subassignment",
            "source": source,
            "assignments": (Subassignment(feature_indices, cell_indices, replacement),),
        }
    if primary_class == "MatrixSubset":
        source = resolve_source(
            _first_value(slots, "matrix"),
            f"{object_path}@matrix",
        )
        zero_dims = _bounded_array(
            slots.get("zero_dims", (False, False)),
            object_path=f"zero_dims at {object_path}",
            limits=limits,
            dtype=bool,
        ).reshape(-1)
        if zero_dims.size != 2:
            raise MatrixSourceError(f"zero_dims at {object_path} must have two values")
        rows = _r_indexes(
            slots.get("row_selection", ()),
            slot_name="row_selection",
            object_path=object_path,
            limits=limits,
        )
        columns = _r_indexes(
            slots.get("col_selection", ()),
            slot_name="col_selection",
            object_path=object_path,
            limits=limits,
        )
        selections = [
            None
            if selection is None or (selection.size == 0 and not zero)
            else selection
            for selection, zero in zip((rows, columns), zero_dims.tolist(), strict=True)
        ]
        # BPCells stores selections in storage orientation; a transposed node
        # selects logical features through col_selection.
        features, cells = selections[::-1] if transposed else selections
        return {
            **base,
            "operation": "subset",
            "source": source,
            "featureIndices": features,
            "cellIndices": cells,
        }
    if primary_class in {"RowBindMatrices", "ColBindMatrices"}:
        values = _first_value(slots, "matrix_list")
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise TypeError(f"matrix list at {object_path} must be a sequence")
        sources = [
            resolve_source(value, f"{object_path}@matrix_list[{index}]")
            for index, value in enumerate(values)
        ]
        # A transposed bind binds storage rows, which are logical columns.
        binds_features = (primary_class == "RowBindMatrices") != transposed
        return {
            **base,
            "operation": "feature_bind" if binds_features else "cell_bind",
            "sources": sources,
        }
    if primary_class == "ConvertMatrixType":
        dtype_value = _slot_text(
            _first_value(slots, "type"),
            slot_name="type",
            object_path=object_path,
        )
        dtype_aliases = {
            "uint32_t": np.dtype(np.uint32),
            "float": np.dtype(np.float32),
            "double": np.dtype(np.float64),
        }
        dtype = dtype_aliases.get(dtype_value)
        if dtype is None:
            raise UnsupportedMatrixOperation(
                object_path,
                "dtype",
                primary_class,
                f"unknown BPCells matrix type {dtype_value!r}",
            )
        return {
            **base,
            "operation": "dtype",
            "source": resolve_source(
                _first_value(slots, "matrix"),
                f"{object_path}@matrix",
            ),
            "dtype": dtype,
        }
    if primary_class in _UNARY_TRANSFORM_CLASSES:
        parameter: int | None = None
        if primary_class == "TransformRound":
            parameters = _parameter_vector(
                {"global_params": slots.get("global_params", (0,))},
                "global_params",
                object_path=object_path,
                max_length=1,
            )
            if (
                parameters.size != 1
                or not np.isfinite(parameters[0])
                or parameters[0] != np.floor(parameters[0])
            ):
                raise MatrixSourceError(
                    f"{primary_class} at {object_path} requires one integer digit"
                )
            parameter = int(parameters[0])
        result = {
            **base,
            "operation": "unary",
            "source": resolve_source(
                _first_value(slots, "matrix"),
                f"{object_path}@matrix",
            ),
            "function": _UNARY_TRANSFORM_CLASSES[primary_class],
        }
        if parameter is not None:
            result["parameter"] = parameter
        return result
    if primary_class in {"TransformPow", "TransformMin"}:
        parameters = _parameter_vector(
            slots, "global_params", object_path=object_path, max_length=2
        )
        if parameters.size != 1:
            raise MatrixSourceError(
                f"{primary_class} at {object_path} requires one parameter"
            )
        return {
            **base,
            "operation": "binary",
            "source": resolve_source(
                _first_value(slots, "matrix"),
                f"{object_path}@matrix",
            ),
            "right": parameters[0].item(),
            "function": ("power" if primary_class == "TransformPow" else "minimum"),
        }
    if primary_class == "TransformBinarize":
        parameters = _parameter_vector(
            slots, "global_params", object_path=object_path, max_length=3
        )
        if parameters.size != 2:
            raise MatrixSourceError(
                f"{primary_class} at {object_path} requires two parameters"
            )
        return {
            **base,
            "operation": "binary",
            "source": resolve_source(
                _first_value(slots, "matrix"),
                f"{object_path}@matrix",
            ),
            "right": parameters[0].item(),
            "function": "greater" if bool(parameters[1]) else "greater_equal",
        }
    if primary_class == "MatrixAddition":
        return {
            **base,
            "operation": "binary",
            "source": resolve_source(
                _first_value(slots, "left"), f"{object_path}@left"
            ),
            "right": resolve_source(
                _first_value(slots, "right"), f"{object_path}@right"
            ),
            "function": "add",
        }
    if primary_class == "MatrixMask":
        return {
            **base,
            "operation": "mask",
            "source": resolve_source(
                _first_value(slots, "matrix"),
                f"{object_path}@matrix",
            ),
            "mask": resolve_source(_first_value(slots, "mask"), f"{object_path}@mask"),
            "keepNonzero": _slot_flag(slots, "invert", object_path),
        }
    if primary_class == "MatrixRankTransform":
        return {
            **base,
            "operation": "rank",
            "source": resolve_source(
                _first_value(slots, "matrix"),
                f"{object_path}@matrix",
            ),
            "axis": "row" if transposed else "column",
        }
    if primary_class == "MatrixMultiply":
        left = resolve_source(_first_value(slots, "left"), f"{object_path}@left")
        right = resolve_source(_first_value(slots, "right"), f"{object_path}@right")
        # BPCells multiplies transposed operands as t(t(y) %*% t(x)), which stores
        # x %*% y with its operands swapped.
        if transposed:
            left, right = right, left
        return {
            **base,
            "operation": "multiply",
            "source": left,
            "right": right,
        }
    if primary_class in {"PeakMatrix", "TileMatrix"}:
        result = {
            **base,
            "operation": "fragment-derived",
            "matrixType": primary_class,
            "fragments": resolve_fragment(
                _first_value(slots, "fragments"),
                f"{object_path}@fragments",
            ),
            "chrId": _first_value(slots, "chr_id"),
            "start": _first_value(slots, "start"),
            "end": _first_value(slots, "end"),
            "chrLevels": _first_value(slots, "chr_levels"),
            "mode": _first_value(slots, "mode"),
            "transpose": _optional_value(slots, "transpose", default=True),
            "shape": _first_value(slots, "dim"),
        }
        if primary_class == "TileMatrix":
            result["tileWidths"] = _first_value(slots, "tile_width")
        return result
    return None


def matrix_source_from_slots(
    specification: Mapping[str, Any],
    *,
    object_path: str = "$",
    sidecar_root: str | os.PathLike[str] | None = None,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None = None,
    limits: SourceLimits = DEFAULT_LIMITS,
) -> MatrixSource:
    """Build a matrix source from serialized R class and slot values.

    Sidecar paths resolve inside ``sidecar_root``; without it, sidecar-backed
    classes are rejected.
    """
    if not isinstance(specification, Mapping):
        raise TypeError("matrix source specification must be a mapping")
    nested = specification.get("slots")
    if nested is None:
        slots = specification
    elif isinstance(nested, Mapping):
        slots = nested
    else:
        raise TypeError(f"matrix slots at {object_path} must be a mapping")
    classes = class_names(specification.get("class", slots.get("class")))
    primary_class = classes[0] if classes else None

    def resolve_source(value: Any, path: str) -> MatrixSource:
        if isinstance(value, MatrixSource):
            return value
        if isinstance(value, Mapping):
            return matrix_source_from_slots(
                value,
                object_path=path,
                sidecar_root=sidecar_root,
                absolute_prefix_remaps=absolute_prefix_remaps,
                limits=limits,
            )
        raise TypeError(f"matrix input at {path} must be a source or mapping")

    def resolve_fragment(value: Any, path: str) -> FragmentSource:
        if isinstance(value, FragmentSource):
            return value
        if isinstance(value, Mapping):
            return fragment_source_from_slots(
                value,
                object_path=path,
                sidecar_root=sidecar_root,
                absolute_prefix_remaps=absolute_prefix_remaps,
                limits=limits,
            )
        raise TypeError(f"fragment input at {path} must be a source or mapping")

    def sidecar(names: tuple[str, ...], expect: str) -> Path:
        return _sidecar_path(
            _first_value(slots, *names),
            sidecar_root=sidecar_root,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect=expect,
            object_path=object_path,
        )

    if primary_class is None:
        raise UnsupportedMatrixOperation(
            object_path, "leaf", None, "matrix class is missing"
        )
    if FRAGMENT_CAPABILITY_REGISTRY.recognizes(primary_class):
        FRAGMENT_CAPABILITY_REGISTRY.resolve(classes, object_path=object_path)
        raise UnsupportedMatrixOperation(
            object_path,
            "leaf",
            primary_class,
            "a fragment source cannot be used as a matrix",
        )
    row_names, column_names = _dimnames(slots, object_path)
    if primary_class in _WRAPPER_CLASSES:
        return _finalize_slot_source(
            resolve_source(_first_value(slots, "seed"), f"{object_path}@seed"),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in {
        "DelayedUnaryIsoOpStack",
        "DelayedUnaryIsoOpWithArgs",
        "DelayedNaryIsoOp",
    }:
        if primary_class == "DelayedUnaryIsoOpStack":
            operation_value = _first_value(slots, "OPS")
            if not isinstance(operation_value, Sequence) or isinstance(
                operation_value,
                str | bytes | bytearray,
            ):
                raise UnsupportedMatrixOperation(
                    object_path,
                    "unary",
                    primary_class,
                    "OPS must be a sequence of recognized primitives",
                )
            operations = tuple(operation_value)
            source = resolve_source(
                _first_value(slots, "seed"),
                f"{object_path}@seed",
            )
            transformed = source
            for index, operation_name in enumerate(operations):
                transformed = build_matrix_operation(
                    {
                        "operation": "unary",
                        "className": classes,
                        "source": transformed,
                        "function": operation_name,
                    },
                    object_path=f"{object_path}@OPS[{index}]",
                    limits=limits,
                )
        elif primary_class == "DelayedNaryIsoOp":
            operation_value = _first_value(slots, "OP")
            if not isinstance(operation_value, str):
                raise UnsupportedMatrixOperation(
                    object_path,
                    "binary",
                    primary_class,
                    "OP must be a recognized primitive",
                )
            values = _first_value(slots, "seeds")
            if not isinstance(values, Sequence) or isinstance(
                values,
                str | bytes | bytearray,
            ):
                raise TypeError(f"seeds at {object_path} must be a sequence")
            sources = [
                resolve_source(value, f"{object_path}@seeds[{index}]")
                for index, value in enumerate(values)
            ]
            if not sources:
                raise MatrixSourceError(
                    f"DelayedNaryIsoOp at {object_path} has no seeds"
                )
            transformed = sources[0]
            for index, right in enumerate(sources[1:], start=1):
                transformed = build_matrix_operation(
                    {
                        "operation": "binary",
                        "className": classes,
                        "source": transformed,
                        "right": right,
                        "function": operation_value,
                    },
                    object_path=f"{object_path}@seeds[{index}]",
                    limits=limits,
                )
            for index, scalar_right in enumerate(
                _operation_arguments(
                    slots.get("Rargs"),
                    object_path=f"{object_path}@Rargs",
                )
            ):
                transformed = build_matrix_operation(
                    {
                        "operation": "binary",
                        "className": classes,
                        "source": transformed,
                        "right": scalar_right,
                        "function": operation_value,
                    },
                    object_path=f"{object_path}@Rargs[{index}]",
                    limits=limits,
                )
        else:
            operation_value = _first_value(slots, "OP")
            if not isinstance(operation_value, str):
                raise UnsupportedMatrixOperation(
                    object_path,
                    "unary",
                    primary_class,
                    "OP must be a recognized primitive",
                )
            source = resolve_source(
                _first_value(slots, "seed"),
                f"{object_path}@seed",
            )
            left_arguments = _operation_arguments(
                slots.get("Largs"),
                object_path=f"{object_path}@Largs",
            )
            right_arguments = _operation_arguments(
                slots.get("Rargs"),
                object_path=f"{object_path}@Rargs",
            )
            single_right = right_arguments[0] if len(right_arguments) == 1 else None
            if len(left_arguments) + len(right_arguments) == 0:
                transformed = build_matrix_operation(
                    {
                        "operation": "unary",
                        "className": classes,
                        "source": source,
                        "function": operation_value,
                    },
                    object_path=object_path,
                    limits=limits,
                )
            elif (
                operation_value == "round"
                and not left_arguments
                and isinstance(single_right, int | float)
                and float(single_right).is_integer()
            ):
                transformed = build_matrix_operation(
                    {
                        "operation": "unary",
                        "className": classes,
                        "source": source,
                        "function": "round",
                        "parameter": int(single_right),
                    },
                    object_path=object_path,
                    limits=limits,
                )
            elif (
                operation_value == "log"
                and not left_arguments
                and isinstance(single_right, int | float)
                and float(single_right) > 0
                and float(single_right) != 1
            ):
                logged = build_matrix_operation(
                    {
                        "operation": "unary",
                        "className": classes,
                        "source": source,
                        "function": "log",
                    },
                    object_path=object_path,
                    limits=limits,
                )
                transformed = build_matrix_operation(
                    {
                        "operation": "binary",
                        "className": classes,
                        "source": logged,
                        "right": float(np.log(float(single_right))),
                        "function": "divide",
                    },
                    object_path=object_path,
                    limits=limits,
                )
            elif len(left_arguments) == 1 and not right_arguments:
                transformed = build_matrix_operation(
                    {
                        "operation": "binary",
                        "className": classes,
                        "source": source,
                        "right": left_arguments[0],
                        "function": operation_value,
                        "reverse": True,
                    },
                    object_path=object_path,
                    limits=limits,
                )
            elif not left_arguments and len(right_arguments) == 1:
                transformed = build_matrix_operation(
                    {
                        "operation": "binary",
                        "className": classes,
                        "source": source,
                        "right": right_arguments[0],
                        "function": operation_value,
                    },
                    object_path=object_path,
                    limits=limits,
                )
            else:
                raise UnsupportedMatrixOperation(
                    object_path,
                    operation_value,
                    primary_class,
                    "only one scalar left or right argument is supported",
                )
        return _finalize_slot_source(
            transformed,
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in {"TransformMinByRow", "TransformMinByCol"}:
        source = resolve_source(
            _first_value(slots, "matrix"),
            f"{object_path}@matrix",
        )
        transposed = _slot_flag(slots, "transpose", object_path)
        parameter_name = (
            "row_params" if primary_class == "TransformMinByRow" else "col_params"
        )
        parameters = _parameter_matrix(
            slots,
            parameter_name,
            object_path=object_path,
            limits=limits,
        )
        axis = (
            "cell"
            if (primary_class == "TransformMinByRow") == transposed
            else "feature"
        )
        return _finalize_slot_source(
            AxisMinimumMatrixSource(
                source,
                parameters.reshape(-1),
                axis=axis,
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class == "TransformScaleShift":
        source = resolve_source(
            _first_value(slots, "matrix"),
            f"{object_path}@matrix",
        )
        transposed = _slot_flag(slots, "transpose", object_path)
        active = _bounded_array(
            _first_value(slots, "active_transforms"),
            object_path=f"active_transforms at {object_path}",
            limits=limits,
            dtype=bool,
        )
        if active.ndim == 1 and active.size == 6:
            active = active.reshape((3, 2), order="F")
        if active.shape != (3, 2):
            raise MatrixSourceError(
                f"active_transforms at {object_path} must have shape (3, 2)"
            )
        row_parameters = _parameter_matrix(
            slots,
            "row_params",
            object_path=object_path,
            limits=limits,
        )
        column_parameters = _parameter_matrix(
            slots,
            "col_params",
            object_path=object_path,
            limits=limits,
        )
        global_parameters = _parameter_vector(
            slots, "global_params", object_path=object_path, max_length=2
        ).astype(np.float64)

        def active_parameters(
            values: NDArray[Any],
            parameter_row: int,
            active_row: int,
        ) -> NDArray[Any] | None:
            if not active[active_row, parameter_row]:
                return None
            if values.ndim != 2 or values.shape[0] <= parameter_row:
                raise MatrixSourceError(
                    f"TransformScaleShift parameters at {object_path} are incomplete"
                )
            return np.asarray(values[parameter_row])

        row_scale = active_parameters(row_parameters, 0, 0)
        column_scale = active_parameters(column_parameters, 0, 1)
        row_shift = active_parameters(row_parameters, 1, 0)
        column_shift = active_parameters(column_parameters, 1, 1)
        if np.any(active[2]) and global_parameters.size < 2:
            raise MatrixSourceError(
                f"global_params at {object_path} must contain scale and shift"
            )
        feature_scale, cell_scale = (
            (column_scale, row_scale) if transposed else (row_scale, column_scale)
        )
        feature_shift, cell_shift = (
            (column_shift, row_shift) if transposed else (row_shift, column_shift)
        )
        return _finalize_slot_source(
            ScaleShiftMatrixSource(
                source,
                feature_scale=feature_scale,
                cell_scale=cell_scale,
                global_scale=(float(global_parameters[0]) if active[2, 0] else 1.0),
                feature_shift=feature_shift,
                cell_shift=cell_shift,
                global_shift=(float(global_parameters[1]) if active[2, 1] else 0.0),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in {
        "SCTransformPearson",
        "SCTransformPearsonSlow",
        "SCTransformPearsonTranspose",
        "SCTransformPearsonTransposeSlow",
    }:
        source = resolve_source(
            _first_value(slots, "matrix"),
            f"{object_path}@matrix",
        )
        row_parameters = _parameter_matrix(
            slots,
            "row_params",
            object_path=object_path,
            limits=limits,
        )
        column_parameters = _parameter_matrix(
            slots,
            "col_params",
            object_path=object_path,
            limits=limits,
        )
        transposed_kernel = "Transpose" in primary_class
        feature_parameters, cell_parameters = (
            (column_parameters, row_parameters)
            if transposed_kernel
            else (row_parameters, column_parameters)
        )
        if feature_parameters.shape[0] != 2 or cell_parameters.shape[0] != 1:
            raise MatrixSourceError(
                f"{primary_class} parameters at {object_path} have invalid shapes"
            )
        return _finalize_slot_source(
            PearsonResidualMatrixSource(
                source,
                theta_inverse=feature_parameters[0],
                gene_beta=feature_parameters[1],
                cell_read_counts=cell_parameters[0],
                global_parameters=_parameter_vector(
                    slots, "global_params", object_path=object_path, max_length=3
                ),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class == "TransformLinearResidual":
        source = resolve_source(
            _first_value(slots, "matrix"),
            f"{object_path}@matrix",
        )
        row_parameters = _parameter_matrix(
            slots,
            "row_params",
            object_path=object_path,
            limits=limits,
        )
        column_parameters = _parameter_matrix(
            slots,
            "col_params",
            object_path=object_path,
            limits=limits,
        )
        transposed = _slot_flag(slots, "transpose", object_path)
        if row_parameters.size == 0 or column_parameters.size == 0:
            residual_source: MatrixSource = source
        else:
            feature_parameters, cell_parameters = (
                (column_parameters, row_parameters)
                if transposed
                else (row_parameters, column_parameters)
            )
            residual_source = LinearResidualMatrixSource(
                source,
                feature_parameters=feature_parameters,
                cell_parameters=cell_parameters,
                limits=limits,
            )
        return _finalize_slot_source(
            residual_source,
            slots,
            row_names,
            column_names,
            limits,
        )
    inferred = _inferred_operation_spec(
        primary_class,
        classes,
        slots,
        object_path=object_path,
        resolve_source=resolve_source,
        resolve_fragment=resolve_fragment,
        limits=limits,
    )
    if inferred is not None:
        renamed = primary_class in _RENAME_CLASSES
        return _finalize_slot_source(
            build_matrix_operation(
                inferred,
                object_path=object_path,
                limits=limits,
            ),
            slots,
            None if renamed else row_names,
            None if renamed else column_names,
            limits,
        )
    if primary_class in _DENSE_CLASSES:
        values = _first_value(slots, ".Data", "x")
        shape = _first_value(slots, "dim", "Dim")
        return _finalize_slot_source(
            DenseMatrixSource(
                values,
                shape_value(shape, f"dim slot at {object_path}"),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _CSC_CLASSES:
        return _finalize_slot_source(
            CscMatrixSource(
                _optional_value(slots, "x"),
                _first_value(slots, "i"),
                _first_value(slots, "p"),
                shape_value(_first_value(slots, "Dim"), f"dim slot at {object_path}"),
                class_name=primary_class,
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class == "Iterable_dgCMatrix_wrapper":
        return _finalize_slot_source(
            resolve_source(_first_value(slots, "mat"), f"{object_path}@mat"),
            slots,
            row_names,
            column_names,
            limits,
            stored_row_major=False,
        )
    if primary_class in _BPCELLS_MEMORY_CLASSES:
        compression, datatype = _BPCELLS_MEMORY_CLASSES[primary_class]
        version = _slot_text(
            _first_value(slots, "version"),
            slot_name="version",
            object_path=object_path,
        )
        expected_prefix = f"{compression}-{datatype}-matrix-v"
        if not version.startswith(expected_prefix):
            raise MatrixSourceError(
                f"BPCells class {primary_class!r} conflicts with format {version!r}"
            )
        shape = shape_value(_first_value(slots, "dim"), f"dim slot at {object_path}")
        array_names = {
            "idxptr",
            "index",
            "index_data",
            "index_starts",
            "index_idx",
            "index_idx_offsets",
            "val",
            "val_data",
            "val_idx",
            "val_idx_offsets",
        }
        arrays = {
            name: slots[name]
            for name in array_names
            if name in slots and slots[name] is not None
        }
        # A memory matrix stores its arrays in the orientation its own
        # transpose flag describes, so it is already logical.
        return _finalize_slot_source(
            BPCellsMemoryMatrixSource(
                version,
                arrays,
                shape=shape,
                storage_order=(
                    "row" if _slot_flag(slots, "transpose", object_path) else "col"
                ),
                row_names=row_names,
                column_names=column_names,
                float_bit_arrays=(
                    frozenset({"val"}) if datatype == "float" else frozenset()
                ),
                limits=limits,
            ),
            slots,
            None,
            None,
            limits,
        )
    if primary_class == "MatrixDir":
        directory_source = BPCellsDirectoryMatrixSource(
            sidecar(("dir",), "directory"), limits=limits
        )
        return _finalize_slot_source(
            directory_source,
            slots,
            row_names,
            column_names,
            limits,
            stored_row_major=directory_source.storageOrder == "row",
        )
    if primary_class == "MatrixH5":
        hdf5_source = BPCellsHDF5MatrixSource(
            sidecar(("path",), "file"),
            group=_slot_text(
                _first_value(slots, "group"),
                slot_name="group",
                object_path=object_path,
            ),
            limits=limits,
        )
        return _finalize_slot_source(
            hdf5_source,
            slots,
            row_names,
            column_names,
            limits,
            stored_row_major=hdf5_source.storageOrder == "row",
        )
    if primary_class == "ReshapedHDF5ArraySeed":
        reshaped = shape_value(
            _first_value(slots, "reshaped_dim"),
            f"reshaped_dim at {object_path}",
        )
        return _finalize_slot_source(
            ReshapedHDF5ArrayMatrixSource(
                sidecar(("filepath",), "file"),
                _slot_text(
                    _first_value(slots, "name"),
                    slot_name="name",
                    object_path=object_path,
                ),
                reshaped,
                as_sparse=_slot_flag(slots, "as_sparse", object_path),
                limits=limits,
            ),
            {**slots, "dim": reshaped},
            row_names,
            column_names,
            limits,
        )
    if primary_class in _HDF5_DENSE_CLASSES:
        return _finalize_slot_source(
            HDF5DenseMatrixSource(
                sidecar(("filepath",), "file"),
                _slot_text(
                    _first_value(slots, "name"),
                    slot_name="name",
                    object_path=object_path,
                ),
                as_sparse=_slot_flag(slots, "as_sparse", object_path),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _H5_SPARSE_CLASSES:
        return _finalize_slot_source(
            H5SparseMatrixSource(
                sidecar(("filepath",), "file"),
                _slot_text(
                    _first_value(slots, "group"),
                    slot_name="group",
                    object_path=object_path,
                ),
                sparse_layout=(
                    primary_class[:3]
                    if primary_class.startswith(("CSC_", "CSR_"))
                    else None
                ),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _H5AD_CLASSES:
        anndata = primary_class == "AnnDataMatrixH5"
        layer = _optional_value(slots, "layer")
        return _finalize_slot_source(
            H5ADMatrixSource(
                sidecar(("path",) if anndata else ("filepath",), "file"),
                layer=(
                    None
                    if anndata or layer is None
                    else _slot_text(layer, slot_name="layer", object_path=object_path)
                ),
                matrix_path=(
                    _slot_text(
                        _optional_value(slots, "group", default="X"),
                        slot_name="group",
                        object_path=object_path,
                    )
                    if anndata
                    else None
                ),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _TENX_CLASSES:
        return _finalize_slot_source(
            TenXMatrixSource(
                sidecar(
                    ("path",) if primary_class == "10xMatrixH5" else ("filepath",),
                    "file",
                ),
                group=_slot_text(
                    _optional_value(slots, "group", default="matrix"),
                    slot_name="group",
                    object_path=object_path,
                ),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    raise UnsupportedMatrixOperation(
        object_path,
        "leaf",
        primary_class,
        "unknown or custom matrix class",
    )
