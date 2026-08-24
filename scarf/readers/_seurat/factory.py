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
    HDF5ArrayMatrixSource,
    ReshapedHDF5ArrayMatrixSource,
    TENxMatrixSource,
)
from .operations import (
    AxisMinimumMatrixSource,
    LinearResidualMatrixSource,
    PearsonResidualMatrixSource,
    ScaleShiftMatrixSource,
    build_matrix_operation,
)
from .paths import SidecarPathResolver, require_filesystem_path
from .sources import (
    DEFAULT_LIMITS,
    CscMatrixSource,
    DenseMatrixSource,
    MatrixSource,
    RenamedMatrixSource,
    SourceLimits,
    TransposeMatrixSource,
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
        "CSC",
    }
)
_BPCELLS_DIRECTORY_CLASSES = frozenset({"MatrixDir", "BPCellsMatrixDir"})
_BPCELLS_HDF5_CLASSES = frozenset({"MatrixH5", "BPCellsMatrixH5"})
_BPCELLS_MEMORY_CLASSES: dict[str, tuple[str, str]] = {
    "PackedMatrixMem_uint32_t": ("packed", "uint"),
    "PackedMatrixMem_float": ("packed", "float"),
    "PackedMatrixMem_double": ("packed", "double"),
    "UnpackedMatrixMem_uint32_t": ("unpacked", "uint"),
    "UnpackedMatrixMem_float": ("unpacked", "float"),
    "UnpackedMatrixMem_double": ("unpacked", "double"),
}
_HDF5_DENSE_CLASSES = frozenset({"HDF5ArraySeed", "Dense_H5ADArraySeed"})
_HDF5_RESHAPED_CLASSES = frozenset({"ReshapedHDF5ArraySeed"})
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
_HDF5_WRAPPER_CLASSES = frozenset(
    {
        "HDF5Array",
        "HDF5Matrix",
        "H5SparseMatrix",
        "H5ADMatrix",
        "TENxMatrix",
    }
)


def _class_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        result = tuple(str(item) for item in value)
        if not result:
            raise MatrixSourceError("matrix class vector cannot be empty")
        return result
    return (str(value),)


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


def _slot_array(value: Any, *, dtype: Any = None) -> NDArray[Any]:
    read_block = getattr(value, "read_block", None)
    raw = read_block(0, len(value)) if callable(read_block) else value
    return np.asarray(raw, dtype=dtype)


def _slot_shape(value: Any, *, object_path: str) -> tuple[int, int]:
    values = _slot_array(value)
    if (
        values.ndim != 1
        or values.size != 2
        or not np.issubdtype(values.dtype, np.number)
        or np.any(~np.isfinite(values))
        or np.any(values != np.floor(values))
    ):
        raise MatrixSourceError(f"dim slot at {object_path} must contain two integers")
    shape = (int(values[0]), int(values[1]))
    if min(shape) < 0:
        raise MatrixSourceError(f"dim slot at {object_path} cannot be negative")
    return shape


def _slot_text(value: Any, *, slot_name: str, object_path: str) -> str:
    if isinstance(value, str):
        return value
    values = value.read_block(0, len(value)) if hasattr(value, "read_block") else value
    if (
        not isinstance(values, Sequence | np.ndarray)
        or isinstance(values, bytes | bytearray)
        or len(values) != 1
    ):
        raise MatrixSourceError(f"{slot_name} at {object_path} must contain one string")
    item = values[0]
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MatrixSourceError(
                f"{slot_name} at {object_path} is not valid UTF-8"
            ) from error
    if not isinstance(item, str):
        raise MatrixSourceError(f"{slot_name} at {object_path} must contain one string")
    return item


def _slot_boolean(
    value: Any,
    *,
    slot_name: str,
    object_path: str,
) -> bool:
    values = _slot_array(value).reshape(-1)
    if values.size != 1:
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must contain one logical value"
        )
    return bool(values[0])


def _parameter_matrix(
    slots: Mapping[str, Any],
    name: str,
    *,
    object_path: str,
) -> NDArray[Any]:
    value = slots.get(name)
    if value is None:
        return np.empty((0, 0), dtype=np.float64)
    values = _slot_array(value)
    if values.ndim == 1:
        if values.size == 0:
            return np.empty((0, 0), dtype=values.dtype)
        return values.reshape(1, -1)
    if values.ndim != 2:
        raise MatrixSourceError(
            f"{name} at {object_path} must be a two-dimensional matrix"
        )
    return values


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
        values = _slot_array(raw).reshape(-1)
        if values.size != 1:
            raise UnsupportedMatrixOperation(
                f"{object_path}[{index}]",
                "delayed-function-argument",
                None,
                "only scalar numeric arguments are supported",
            )
        scalar = values[0]
        if isinstance(scalar, np.generic):
            scalar = scalar.item()
        if isinstance(scalar, bool) or not isinstance(
            scalar,
            int | float | complex,
        ):
            raise UnsupportedMatrixOperation(
                f"{object_path}[{index}]",
                "delayed-function-argument",
                None,
                "only scalar numeric arguments are supported",
            )
        output.append(scalar)
    return tuple(output)


def _dimnames(
    slots: Mapping[str, Any],
) -> tuple[Sequence[str | bytes] | np.ndarray[Any, Any] | None, ...]:
    row_names = _optional_value(slots, "rowNames", "row_names")
    column_names = _optional_value(slots, "columnNames", "column_names")
    if row_names is not None or column_names is not None:
        return row_names, column_names
    values = _optional_value(slots, "Dimnames", "dimnames")
    if values is None:
        return None, None
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or len(values) != 2
    ):
        raise MatrixSourceError("dimnames must contain row and column names")
    return values[0], values[1]


def _sidecar_path(
    value: Any,
    *,
    rds_path: str | os.PathLike[str] | None,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None,
    expect: str,
) -> Path:
    if rds_path is None:
        if expect == "file":
            return require_filesystem_path(value)
        if not isinstance(value, str | os.PathLike):
            raise TypeError("sidecar directory must be a filesystem path")
        path = Path(value).expanduser().resolve(strict=False)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_dir():
            raise UnsafeSidecarError(f"sidecar path {path} is not a directory")
        return path
    return SidecarPathResolver(
        rds_path,
        absolute_prefix_remaps=absolute_prefix_remaps,
    ).resolve(value, expect=expect)


def _finalize_slot_source(
    source: MatrixSource,
    slots: Mapping[str, Any],
    row_names: Sequence[str | bytes] | np.ndarray[Any, Any] | None,
    column_names: Sequence[str | bytes] | np.ndarray[Any, Any] | None,
    limits: SourceLimits,
) -> MatrixSource:
    transpose_value = _optional_value(slots, "transpose", default=False)
    transpose_array = (
        np.asarray(transpose_value.read_block(0, len(transpose_value)))
        if hasattr(transpose_value, "read_block")
        else np.asarray(transpose_value)
    )
    if transpose_array.size != 1:
        raise MatrixSourceError("transpose slot must be scalar")
    expected_shape_value = _optional_value(slots, "Dim", "dim")
    expected_shape: tuple[int, int] | None = None
    if expected_shape_value is not None:
        expected_array = (
            np.asarray(expected_shape_value.read_block(0, len(expected_shape_value)))
            if hasattr(expected_shape_value, "read_block")
            else np.asarray(expected_shape_value)
        )
        if (
            expected_array.ndim != 1
            or expected_array.size != 2
            or not np.issubdtype(expected_array.dtype, np.number)
            or np.any(~np.isfinite(expected_array))
            or np.any(expected_array != np.floor(expected_array))
        ):
            raise MatrixSourceError("dim slot must contain two integers")
        expected_shape = (int(expected_array[0]), int(expected_array[1]))
        if min(expected_shape) < 0:
            raise MatrixSourceError("dim slot cannot contain negative values")
    should_transpose = bool(transpose_array.reshape(-1)[0])
    if expected_shape is not None:
        if source.shape == expected_shape:
            should_transpose = False
        elif source.shape[::-1] == expected_shape:
            should_transpose = True
        else:
            raise MatrixSourceError(
                f"matrix source shape {source.shape} does not match dim slot "
                f"{expected_shape}"
            )
    if should_transpose:
        source = TransposeMatrixSource(source, limits=limits)
    if row_names is not None or column_names is not None:
        source = RenamedMatrixSource(
            source,
            row_names=row_names,
            column_names=column_names,
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


def _r_selection(
    slots: Mapping[str, Any],
    name: str,
    zero_dims: NDArray[np.bool_],
    axis: int,
) -> NDArray[np.int64] | None:
    raw = _slot_array(slots.get(name, ()), dtype=np.int64).reshape(-1)
    if raw.size:
        if np.any(raw <= 0):
            raise MatrixSourceError(f"{name} must contain positive R indexes")
        return raw - 1
    if zero_dims.size > axis and bool(zero_dims[axis]):
        return np.empty(0, dtype=np.int64)
    return None


def _r_delayed_selection(
    value: Any,
    *,
    slot_name: str,
    object_path: str,
) -> NDArray[np.int64] | None:
    if value is None:
        return None
    indexes = _slot_array(value)
    if indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer):
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must be a one-dimensional integer vector"
        )
    indexes = indexes.astype(np.int64, copy=False)
    if indexes.size and np.any(indexes <= 0):
        raise MatrixSourceError(
            f"{slot_name} at {object_path} contains a missing or nonpositive R index"
        )
    return indexes - 1


def _r_scalar_integer(value: Any, *, slot_name: str, object_path: str) -> int:
    values = _slot_array(value)
    if (
        values.size != 1
        or not np.issubdtype(values.dtype, np.integer)
        or isinstance(values.reshape(-1)[0], np.bool_)
    ):
        raise MatrixSourceError(
            f"{slot_name} at {object_path} must contain one integer"
        )
    return int(values.reshape(-1)[0])


def _inferred_operation_spec(
    primary_class: str,
    classes: tuple[str, ...],
    slots: Mapping[str, Any],
    *,
    object_path: str,
    resolve_source: Callable[[Any, str], MatrixSource],
    resolve_fragment: Callable[[Any, str], FragmentSource],
) -> dict[str, Any] | None:
    base: dict[str, Any] = {"className": classes}
    if primary_class in {"DelayedArray", "DelayedMatrix"} | _HDF5_WRAPPER_CLASSES:
        return {
            **base,
            "operation": "rename",
            "source": resolve_source(
                _first_value(slots, "seed"),
                f"{object_path}@seed",
            ),
        }
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
            "featureIndices": _r_delayed_selection(
                index[0],
                slot_name="index[[1]]",
                object_path=object_path,
            ),
            "cellIndices": _r_delayed_selection(
                index[1],
                slot_name="index[[2]]",
                object_path=object_path,
            ),
        }
    if primary_class in {"DelayedAperm", "SeedDimPicker"}:
        permutation = _slot_array(_first_value(slots, "perm", "dim_combination"))
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
    if primary_class == "DelayedSetDimnames":
        return {
            **base,
            "operation": "rename",
            "source": resolve_source(
                _first_value(slots, "seed"),
                f"{object_path}@seed",
            ),
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
        feature_indices = _r_delayed_selection(
            index[0],
            slot_name="Lindex[[1]]",
            object_path=object_path,
        )
        cell_indices = _r_delayed_selection(
            index[1],
            slot_name="Lindex[[2]]",
            object_path=object_path,
        )
        if feature_indices is None:
            feature_indices = np.arange(source.shape[0], dtype=np.int64)
        if cell_indices is None:
            cell_indices = np.arange(source.shape[1], dtype=np.int64)
        replacement = _first_value(slots, "Rvalue")
        if not isinstance(replacement, MatrixSource):
            replacement_values = _slot_array(replacement)
            if replacement_values.size != 1:
                raise UnsupportedMatrixOperation(
                    object_path,
                    "subassignment",
                    primary_class,
                    "ordinary replacement values must contain one numeric scalar",
                )
            replacement = replacement_values.reshape(-1)[0].item()
        return {
            **base,
            "operation": "subassignment",
            "source": source,
            "assignments": (
                {
                    "featureIndices": feature_indices,
                    "cellIndices": cell_indices,
                    "value": replacement,
                },
            ),
        }
    if primary_class == "MatrixSubset":
        source = resolve_source(
            _first_value(slots, "matrix", "source", "seed"),
            f"{object_path}@matrix",
        )
        zero_dims = _slot_array(
            slots.get("zero_dims", slots.get("zeroDims", (False, False))),
            dtype=bool,
        ).reshape(-1)
        return {
            **base,
            "operation": "subset",
            "source": source,
            "featureIndices": _r_selection(slots, "row_selection", zero_dims, 0),
            "cellIndices": _r_selection(slots, "col_selection", zero_dims, 1),
        }
    if primary_class in {"RowBindMatrices", "ColBindMatrices"}:
        values = _first_value(slots, "matrix_list", "sources", "matrices")
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise TypeError(f"matrix list at {object_path} must be a sequence")
        sources = [
            resolve_source(value, f"{object_path}@matrix_list[{index}]")
            for index, value in enumerate(values)
        ]
        return {
            **base,
            "operation": (
                "feature_bind" if primary_class == "RowBindMatrices" else "cell_bind"
            ),
            "sources": sources,
        }
    if primary_class == "RenameDims":
        return {
            **base,
            "operation": "rename",
            "source": resolve_source(
                _first_value(slots, "matrix", "source", "seed"),
                f"{object_path}@matrix",
            ),
        }
    if primary_class == "ConvertMatrixType":
        dtype_value = _first_value(slots, "type", "matrix_type", "matrixType", "dtype")
        dtype_aliases = {
            "uint32_t": np.dtype(np.uint32),
            "float": np.dtype(np.float32),
            "double": np.dtype(np.float64),
        }
        dtype = dtype_aliases.get(str(dtype_value))
        if dtype is None:
            dtype = np.dtype(dtype_value)
        return {
            **base,
            "operation": "dtype",
            "source": resolve_source(
                _first_value(slots, "matrix", "source", "seed"),
                f"{object_path}@matrix",
            ),
            "dtype": dtype,
        }
    if primary_class in _UNARY_TRANSFORM_CLASSES:
        parameter: int | None = None
        if primary_class == "TransformRound":
            parameters = _slot_array(
                slots.get("global_params", slots.get("globalParams", (0,)))
            ).reshape(-1)
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
                _first_value(slots, "matrix", "source", "seed"),
                f"{object_path}@matrix",
            ),
            "function": _UNARY_TRANSFORM_CLASSES[primary_class],
        }
        if parameter is not None:
            result["parameter"] = parameter
        return result
    if primary_class in {"TransformPow", "TransformMin"}:
        parameters = _slot_array(
            _first_value(slots, "global_params", "globalParams")
        ).reshape(-1)
        if parameters.size != 1:
            raise MatrixSourceError(
                f"{primary_class} at {object_path} requires one parameter"
            )
        return {
            **base,
            "operation": "binary",
            "source": resolve_source(
                _first_value(slots, "matrix", "source", "seed"),
                f"{object_path}@matrix",
            ),
            "right": parameters[0].item(),
            "function": ("power" if primary_class == "TransformPow" else "minimum"),
        }
    if primary_class == "TransformBinarize":
        parameters = _slot_array(
            _first_value(slots, "global_params", "globalParams")
        ).reshape(-1)
        if parameters.size != 2:
            raise MatrixSourceError(
                f"{primary_class} at {object_path} requires two parameters"
            )
        return {
            **base,
            "operation": "binary",
            "source": resolve_source(
                _first_value(slots, "matrix", "source", "seed"),
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
                _first_value(slots, "matrix", "source"),
                f"{object_path}@matrix",
            ),
            "mask": resolve_source(_first_value(slots, "mask"), f"{object_path}@mask"),
            "invert": bool(slots.get("invert", False)),
        }
    if primary_class == "MatrixRankTransform":
        return {
            **base,
            "operation": "rank",
            "source": resolve_source(
                _first_value(slots, "matrix", "source"),
                f"{object_path}@matrix",
            ),
            "axis": "row" if bool(slots.get("transpose", False)) else "column",
        }
    if primary_class == "MatrixMultiply":
        result = {
            **base,
            "operation": "multiply",
            "source": resolve_source(
                _first_value(slots, "left"), f"{object_path}@left"
            ),
            "right": resolve_source(
                _first_value(slots, "right"), f"{object_path}@right"
            ),
        }
        if "dim" in slots:
            result["dim"] = slots["dim"]
        elif "Dim" in slots:
            result["Dim"] = slots["Dim"]
        return result
    if primary_class in {"PeakMatrix", "TileMatrix"}:
        result = {
            **base,
            "operation": "fragment-derived",
            "matrixType": primary_class,
            "fragments": resolve_fragment(
                _first_value(slots, "fragments"),
                f"{object_path}@fragments",
            ),
            "chrId": _first_value(slots, "chr_id", "chrId"),
            "start": _first_value(slots, "start"),
            "end": _first_value(slots, "end"),
            "chrLevels": _first_value(slots, "chr_levels", "chrLevels"),
            "mode": _first_value(slots, "mode"),
            "transpose": _optional_value(slots, "transpose", default=True),
            "shape": _first_value(slots, "dim", "Dim", "shape"),
        }
        if primary_class == "TileMatrix":
            result["tileWidths"] = _first_value(
                slots,
                "tile_width",
                "tileWidths",
            )
        return result
    return None


def matrix_source_from_slots(
    specification: Mapping[str, Any],
    *,
    object_path: str = "$",
    class_name: str | None = None,
    rds_path: str | os.PathLike[str] | None = None,
    absolute_prefix_remaps: Mapping[str | os.PathLike[str], str | os.PathLike[str]]
    | None = None,
    limits: SourceLimits = DEFAULT_LIMITS,
) -> MatrixSource:
    if not isinstance(specification, Mapping):
        raise TypeError("matrix source specification must be a mapping")
    nested = specification.get("slots")
    if nested is None:
        slots = specification
    elif isinstance(nested, Mapping):
        slots = nested
    else:
        raise TypeError(f"matrix slots at {object_path} must be a mapping")
    classes = _class_names(
        class_name
        if class_name is not None
        else specification.get(
            "className",
            specification.get(
                "class",
                slots.get("className", slots.get("class")),
            ),
        )
    )
    primary_class = classes[0] if classes else None

    def resolve_source(value: Any, path: str) -> MatrixSource:
        if isinstance(value, MatrixSource):
            return value
        if isinstance(value, Mapping):
            return matrix_source_from_slots(
                value,
                object_path=path,
                rds_path=rds_path,
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
                rds_path=rds_path,
                absolute_prefix_remaps=absolute_prefix_remaps,
                limits=limits,
            )
        raise TypeError(f"fragment input at {path} must be a source or mapping")

    operation = specification.get("operation", specification.get("op"))
    if operation is not None:
        operation_spec = dict(specification)
        operation_spec.update(slots)
        for key in ("source", "matrix", "seed", "left", "right", "mask"):
            value = operation_spec.get(key)
            if isinstance(value, Mapping):
                operation_spec[key] = resolve_source(value, f"{object_path}@{key}")
        fragment_value = operation_spec.get("fragments")
        if isinstance(fragment_value, Mapping):
            operation_spec["fragments"] = resolve_fragment(
                fragment_value,
                f"{object_path}@fragments",
            )
        for key in ("sources", "matrices", "matrix_list"):
            values = operation_spec.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray)
            ):
                operation_spec[key] = [
                    resolve_source(value, f"{object_path}@{key}[{index}]")
                    for index, value in enumerate(values)
                ]
        if primary_class is not None:
            operation_spec["className"] = classes
        return build_matrix_operation(
            operation_spec,
            object_path=object_path,
            limits=limits,
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
    row_names, column_names = _dimnames(slots)
    if primary_class in {
        "DelayedUnaryIsoOpStack",
        "DelayedUnaryIsoOpWithArgs",
        "DelayedNaryIsoOp",
    }:
        operation_value = _first_value(slots, "OP", "op", "function", "OPS")
        if primary_class == "DelayedUnaryIsoOpStack":
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
            _first_value(slots, "matrix", "source", "seed"),
            f"{object_path}@matrix",
        )
        transposed = _slot_boolean(
            _optional_value(slots, "transpose", default=False),
            slot_name="transpose",
            object_path=object_path,
        )
        parameter_name = (
            "row_params" if primary_class == "TransformMinByRow" else "col_params"
        )
        parameters = _parameter_matrix(
            slots,
            parameter_name,
            object_path=object_path,
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
            _first_value(slots, "matrix", "source", "seed"),
            f"{object_path}@matrix",
        )
        transposed = _slot_boolean(
            _optional_value(slots, "transpose", default=False),
            slot_name="transpose",
            object_path=object_path,
        )
        active = _slot_array(
            _first_value(slots, "active_transforms"),
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
        )
        column_parameters = _parameter_matrix(
            slots,
            "col_params",
            object_path=object_path,
        )
        global_parameters = _slot_array(
            slots.get("global_params", ()),
            dtype=np.float64,
        ).reshape(-1)

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
            _first_value(slots, "matrix", "source", "seed"),
            f"{object_path}@matrix",
        )
        row_parameters = _parameter_matrix(
            slots,
            "row_params",
            object_path=object_path,
        )
        column_parameters = _parameter_matrix(
            slots,
            "col_params",
            object_path=object_path,
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
                global_parameters=_slot_array(
                    _first_value(slots, "global_params"),
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
            _first_value(slots, "matrix", "source", "seed"),
            f"{object_path}@matrix",
        )
        row_parameters = _parameter_matrix(
            slots,
            "row_params",
            object_path=object_path,
        )
        column_parameters = _parameter_matrix(
            slots,
            "col_params",
            object_path=object_path,
        )
        transposed = _slot_boolean(
            _optional_value(slots, "transpose", default=False),
            slot_name="transpose",
            object_path=object_path,
        )
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
    )
    if inferred is not None:
        return _finalize_slot_source(
            build_matrix_operation(
                inferred,
                object_path=object_path,
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _DENSE_CLASSES:
        values = _first_value(slots, ".Data", "data", "values", "x")
        shape = _first_value(slots, "dim", "shape")
        return _finalize_slot_source(
            DenseMatrixSource(
                values,
                shape,
                dtype=_optional_value(slots, "dtype"),
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
                _first_value(slots, "Dim", "dim", "shape"),
                dtype=_optional_value(slots, "dtype"),
                class_name=primary_class,
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class == "Iterable_dgCMatrix_wrapper":
        wrapped = _first_value(slots, "mat")
        if not isinstance(wrapped, MatrixSource):
            wrapped = resolve_source(wrapped, f"{object_path}@mat")
        return _finalize_slot_source(
            wrapped,
            slots,
            row_names,
            column_names,
            limits,
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
        shape = _slot_shape(
            _first_value(slots, "dim", "shape"),
            object_path=object_path,
        )
        transpose = _slot_array(
            _optional_value(slots, "transpose", default=False)
        ).reshape(-1)
        if transpose.size != 1:
            raise MatrixSourceError("transpose slot must be scalar")
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
        return _finalize_slot_source(
            BPCellsMemoryMatrixSource(
                version,
                arrays,
                shape=shape,
                storage_order="row" if bool(transpose[0]) else "col",
                row_names=row_names,
                column_names=column_names,
                float_bit_arrays=(
                    frozenset({"val"}) if datatype == "float" else frozenset()
                ),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _BPCELLS_DIRECTORY_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "dir", "directory", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="directory",
        )
        return _finalize_slot_source(
            BPCellsDirectoryMatrixSource(path, limits=limits),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _BPCELLS_HDF5_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "filepath", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        return _finalize_slot_source(
            BPCellsHDF5MatrixSource(
                path,
                group=str(_first_value(slots, "group")),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _HDF5_RESHAPED_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "filepath", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        reshaped = _slot_shape(
            _first_value(slots, "reshaped_dim", "reshapedDim"),
            object_path=object_path,
        )
        return _finalize_slot_source(
            ReshapedHDF5ArrayMatrixSource(
                path,
                str(_first_value(slots, "name", "dataset")),
                reshaped,
                dtype=_optional_value(slots, "dtype"),
                as_sparse=bool(
                    _optional_value(slots, "asSparse", "as_sparse", default=False)
                ),
                limits=limits,
            ),
            {**slots, "dim": reshaped},
            row_names,
            column_names,
            limits,
        )
    if primary_class in _HDF5_DENSE_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "filepath", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        return _finalize_slot_source(
            HDF5ArrayMatrixSource(
                path,
                str(_first_value(slots, "name", "dataset")),
                dtype=_optional_value(slots, "dtype"),
                as_sparse=bool(
                    _optional_value(slots, "asSparse", "as_sparse", default=False)
                ),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _H5_SPARSE_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "filepath", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        sparse_layout = _optional_value(slots, "sparseLayout", "sparse_layout")
        if sparse_layout is None and primary_class.startswith(("CSC_", "CSR_")):
            sparse_layout = primary_class[:3]
        return _finalize_slot_source(
            H5SparseMatrixSource(
                path,
                str(_first_value(slots, "group")),
                shape=_optional_value(slots, "dim", "shape"),
                sparse_layout=sparse_layout,
                dtype=_optional_value(slots, "dtype"),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _H5AD_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "filepath", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        return _finalize_slot_source(
            H5ADMatrixSource(
                path,
                layer=(
                    _optional_value(slots, "layer")
                    if primary_class != "AnnDataMatrixH5"
                    else None
                ),
                matrix_path=(
                    _optional_value(slots, "group", default="X")
                    if primary_class == "AnnDataMatrixH5"
                    else None
                ),
                dtype=_optional_value(slots, "dtype"),
                limits=limits,
            ),
            slots,
            row_names,
            column_names,
            limits,
        )
    if primary_class in _TENX_CLASSES:
        path = _sidecar_path(
            _first_value(slots, "filepath", "path"),
            rds_path=rds_path,
            absolute_prefix_remaps=absolute_prefix_remaps,
            expect="file",
        )
        return _finalize_slot_source(
            TENxMatrixSource(
                path,
                group=str(_optional_value(slots, "group", default="matrix")),
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


source_from_slot_mapping = matrix_source_from_slots
matrix_source_from_mapping = matrix_source_from_slots
