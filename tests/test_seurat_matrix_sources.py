from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix

from scarf.readers._seurat import (
    AxisMinimumMatrixSource,
    BPCellsDirectoryMatrixSource,
    BPCellsHDF5MatrixSource,
    BPCellsMemoryMatrixSource,
    BinaryTransformMatrixSource,
    CellBindMatrixSource,
    CscMatrixSource,
    DelayedSubassignmentMatrixSource,
    DenseMatrixSource,
    DtypeMatrixSource,
    FeatureBindMatrixSource,
    FRAGMENT_CAPABILITY_REGISTRY,
    FragmentDerivedMatrixSource,
    H5ADMatrixSource,
    H5SparseMatrixSource,
    HDF5ArrayMatrixSource,
    HDF5CompressedMatrixSource,
    LayerPlacement,
    LayerStitchMatrixSource,
    LinearResidualMatrixSource,
    MappedMatrixSource,
    MaskMatrixSource,
    MatrixMultiplySource,
    MatrixSource,
    MatrixSourceError,
    PearsonResidualMatrixSource,
    RankMatrixSource,
    RenamedMatrixSource,
    ReshapedHDF5ArrayMatrixSource,
    ResourceLimitError,
    ScaleShiftMatrixSource,
    SidecarPathResolver,
    SourceLimits,
    Subassignment,
    TENxMatrixSource,
    TransposeMatrixSource,
    UnaryTransformMatrixSource,
    UnsafeSidecarError,
    UnsupportedMatrixOperation,
    build_matrix_operation,
    decode_bp128,
    decode_bp128_d1z,
    decode_bp128_m1,
    fragment_source_from_slots,
    matrix_source_from_slots,
    read_hdf5_names,
    require_filesystem_path,
    resolve_sidecar_path,
    validate_hdf5_file,
)
from scarf.readers._seurat.paths import (
    read_hdf5_shape,
    require_hdf5_datasets,
    require_hdf5_group,
)


_HEADERS = {
    ("u", 4): b"UINT32v1",
    ("u", 8): b"UINT64v1",
    ("f", 4): b"FLOATSv1",
    ("f", 8): b"DOUBLEv1",
}


class _TrackedVector:
    def __init__(self, values: Sequence[int] | NDArray[Any]) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.reads: list[tuple[int | None, int | None]] = []

    def __getitem__(self, key: slice) -> NDArray[Any]:
        assert isinstance(key, slice)
        self.reads.append((key.start, key.stop))
        return self.values[key]


def _pack_bp128_block(
    values: Sequence[int] | NDArray[Any],
) -> NDArray[np.uint32]:
    values = np.asarray(values, dtype=np.uint32)
    assert values.shape == (128,)
    bits = max((int(value).bit_length() for value in values), default=0)
    if bits == 0:
        return np.empty(0, dtype=np.uint32)
    packed: NDArray[np.uint64] = np.zeros((bits, 4), dtype=np.uint64)
    mask = (1 << bits) - 1 if bits < 32 else 0xFFFFFFFF
    for vector_index in range(32):
        lanes = values[vector_index * 4 : vector_index * 4 + 4].astype(np.uint64)
        lanes &= np.uint64(mask)
        bit_position = vector_index * bits
        word_index = bit_position // 32
        shift = bit_position & 31
        packed[word_index] |= (lanes << np.uint64(shift)) & np.uint64(0xFFFFFFFF)
        if shift + bits > 32:
            packed[word_index + 1] |= lanes >> np.uint64(32 - shift)
    return packed.astype(np.uint32).reshape(-1)


def _encode_bp128(
    values: Sequence[int] | NDArray[Any],
    transform: str,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint32],
    NDArray[np.uint64],
    NDArray[np.uint32],
]:
    values = np.asarray(values, dtype=np.uint32)
    data_parts: list[NDArray[np.uint32]] = []
    indexes: list[int] = [0]
    starts: list[int] = []
    for block_start in range(0, values.size, 128):
        block = values[block_start : block_start + 128]
        if block.size < 128:
            pad_value = int(block[-1]) if block.size else 0
            block = np.pad(
                block,
                (0, 128 - block.size),
                constant_values=pad_value,
            )
        if transform == "m1":
            assert np.all(block > 0)
            encoded = block - np.uint32(1)
        elif transform == "d1z":
            starts.append(int(block[0]))
            signed = block.astype(np.int64)
            differences = np.diff(signed, prepend=signed[0])
            encoded = np.where(
                differences >= 0,
                differences * 2,
                -differences * 2 - 1,
            ).astype(np.uint32)
        elif transform == "d1":
            starts.append(int(block[0]))
            previous = np.concatenate((block[:1], block[:-1]))
            encoded = block - previous
        elif transform == "plain":
            encoded = block
        else:
            raise AssertionError(transform)
        packed = _pack_bp128_block(encoded)
        data_parts.append(packed)
        indexes.append(indexes[-1] + packed.size)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.uint32)
    return (
        data,
        np.asarray(indexes, dtype=np.uint32),
        np.asarray([0, len(indexes)], dtype=np.uint64),
        np.asarray(starts, dtype=np.uint32),
    )


def _write_numeric_file(directory: Path, name: str, values: Any) -> None:
    values = np.asarray(values)
    key = (values.dtype.kind, values.dtype.itemsize)
    assert key in _HEADERS
    little_endian = values.astype(values.dtype.newbyteorder("<"), copy=False)
    (directory / name).write_bytes(_HEADERS[key] + little_endian.tobytes(order="C"))


def _matrix_arrays(
    values: NDArray[Any],
    storage_order: str,
) -> tuple[NDArray[Any], NDArray[np.uint32], NDArray[Any]]:
    matrix = csc_matrix(values) if storage_order == "col" else csr_matrix(values)
    return (
        np.asarray(matrix.data),
        matrix.indices.astype(np.uint32),
        matrix.indptr,
    )


def _bpcells_payload(
    values: NDArray[Any],
    *,
    packed: bool,
    version: int,
    storage_order: str,
) -> dict[str, Any]:
    data, indexes, pointers = _matrix_arrays(values, storage_order)
    datatype = {
        np.dtype(np.uint32): "uint",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }[values.dtype]
    payload = {
        "version": f"{'packed' if packed else 'unpacked'}-{datatype}-matrix-v{version}",
        "shape": np.asarray(values.shape, dtype=np.uint32),
        "idxptr": pointers.astype(np.uint32 if version == 1 else np.uint64),
        "storage_order": storage_order,
        "row_names": tuple(f"f{index}" for index in range(values.shape[0])),
        "col_names": tuple(f"c{index}" for index in range(values.shape[1])),
    }
    if not packed:
        payload["index"] = indexes
        payload["val"] = data
        return payload
    (
        payload["index_data"],
        payload["index_idx"],
        payload["index_idx_offsets"],
        payload["index_starts"],
    ) = _encode_bp128(indexes, "d1z")
    if datatype == "uint":
        (
            payload["val_data"],
            payload["val_idx"],
            payload["val_idx_offsets"],
            _,
        ) = _encode_bp128(data, "m1")
    else:
        payload["val"] = data
    return payload


def _write_bpcells_directory(
    path: Path,
    payload: dict[str, Any],
    *,
    version: int,
) -> None:
    path.mkdir()
    (path / "version").write_text(payload["version"] + "\n")
    (path / "storage_order").write_text(payload["storage_order"] + "\n")
    (path / "row_names").write_text("\n".join(payload["row_names"]) + "\n")
    (path / "col_names").write_text("\n".join(payload["col_names"]) + "\n")
    for name, values in payload.items():
        if name in {
            "version",
            "storage_order",
            "row_names",
            "col_names",
        }:
            continue
        if version == 1 and name.endswith("_idx_offsets"):
            continue
        _write_numeric_file(path, name, values)


def _write_bpcells_hdf5(
    path: Path,
    payload: dict[str, Any],
    *,
    version: int,
) -> None:
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("matrix")
        group.attrs["version"] = payload["version"]
        group.create_dataset(
            "storage_order",
            data=np.asarray(payload["storage_order"], dtype=string_dtype),
        )
        group.create_dataset(
            "row_names",
            data=np.asarray(payload["row_names"], dtype=string_dtype),
        )
        group.create_dataset(
            "col_names",
            data=np.asarray(payload["col_names"], dtype=string_dtype),
        )
        for name, values in payload.items():
            if name in {
                "version",
                "storage_order",
                "row_names",
                "col_names",
            }:
                continue
            if version == 1 and name.endswith("_idx_offsets"):
                continue
            group.create_dataset(name, data=values)


def _memory_matrix_value(value: Any) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype == np.dtype(np.uint32):
        return value.view(np.int32)
    if value.dtype == np.dtype(np.uint64):
        return value.astype(np.float64)
    if value.dtype == np.dtype(np.float32):
        return value.view(np.int32)
    return value


def _memory_matrix_spec(
    payload: dict[str, Any],
    *,
    packed: bool,
    datatype: str,
    version: int,
) -> dict[str, Any]:
    class_datatype = "uint32_t" if datatype == "uint" else datatype
    slots = {
        name: _memory_matrix_value(value)
        for name, value in payload.items()
        if name
        not in {
            "shape",
            "storage_order",
            "row_names",
            "col_names",
        }
        and not (version == 1 and name.endswith("_idx_offsets"))
    }
    slots.update(
        {
            "version": payload["version"],
            "dim": payload["shape"].astype(np.float64),
            "transpose": payload["storage_order"] == "row",
            "dimnames": [payload["row_names"], payload["col_names"]],
        }
    )
    return {
        "class": [
            f"{'Packed' if packed else 'Unpacked'}MatrixMem_{class_datatype}",
            "IterableMatrix",
        ],
        "slots": slots,
    }


def _fragment_payload(*, packed: bool, version: int) -> dict[str, Any]:
    cells = np.asarray([0, 1, 0, 2, 1, 2, 0], dtype=np.uint32)
    starts = np.asarray([0, 5, 10, 12, 20, 1, 4], dtype=np.uint32)
    ends = np.asarray([10, 15, 20, 18, 30, 9, 12], dtype=np.uint32)
    payload: dict[str, Any] = {
        "version": f"{'packed' if packed else 'unpacked'}-fragments-v{version}",
        "end_max": np.asarray([30], dtype=np.uint32),
        "chr_ptr": np.asarray(
            [0, 5, 5, 7],
            dtype=np.uint32 if version == 1 else np.uint64,
        ),
        "chr_names": ("chr1", "chr2"),
        "cell_names": ("c1", "c2", "c3"),
    }
    if not packed:
        payload.update({"cell": cells, "start": starts, "end": ends})
        return payload
    for name, values, transform in (
        ("cell", cells, "plain"),
        ("start", starts, "d1"),
        ("end", ends - starts, "plain"),
    ):
        data, indexes, offsets, block_starts = _encode_bp128(values, transform)
        payload[f"{name}_data"] = data
        payload[f"{name}_idx"] = indexes
        payload[f"{name}_idx_offsets"] = offsets
        if name == "start":
            payload["start_starts"] = block_starts
    return payload


def _write_fragment_directory(
    path: Path,
    payload: dict[str, Any],
    *,
    version: int,
) -> None:
    path.mkdir()
    (path / "version").write_text(payload["version"] + "\n")
    (path / "chr_names").write_text("\n".join(payload["chr_names"]) + "\n")
    (path / "cell_names").write_text("\n".join(payload["cell_names"]) + "\n")
    for name, values in payload.items():
        if name in {"version", "chr_names", "cell_names"}:
            continue
        if version == 1 and name.endswith("_idx_offsets"):
            continue
        _write_numeric_file(path, name, values)


def _write_fragment_hdf5(
    path: Path,
    payload: dict[str, Any],
    *,
    version: int,
) -> None:
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("fragments")
        group.attrs["version"] = payload["version"]
        group.create_dataset(
            "chr_names",
            data=np.asarray(payload["chr_names"], dtype=string_dtype),
        )
        group.create_dataset(
            "cell_names",
            data=np.asarray(payload["cell_names"], dtype=string_dtype),
        )
        for name, values in payload.items():
            if name in {"version", "chr_names", "cell_names"}:
                continue
            if version == 1 and name.endswith("_idx_offsets"):
                continue
            group.create_dataset(name, data=values)


def _memory_fragment_value(value: Any) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype == np.dtype(np.uint32):
        return value.view(np.int32)
    if value.dtype == np.dtype(np.uint64):
        return value.astype(np.float64)
    return value


def _fragment_leaf_spec(
    tmp_path: Path,
    *,
    backend: str,
    packed: bool,
    version: int,
) -> dict[str, Any]:
    payload = _fragment_payload(packed=packed, version=version)
    if backend == "memory":
        slots = {name: _memory_fragment_value(value) for name, value in payload.items()}
        slots["version"] = [payload["version"]]
        return {
            "class": [
                "PackedMemFragments" if packed else "UnpackedMemFragments",
                "IterableFragments",
            ],
            "slots": slots,
        }
    if backend == "directory":
        path = tmp_path / f"fragments-{packed}-{version}"
        _write_fragment_directory(path, payload, version=version)
        return {
            "class": ["FragmentsDir", "IterableFragments"],
            "slots": {
                "dir": str(path),
                "compressed": packed,
                "buffer_size": 128,
                "chr_names": payload["chr_names"],
                "cell_names": payload["cell_names"],
            },
        }
    if backend == "hdf5":
        path = tmp_path / f"fragments-{packed}-{version}.h5"
        _write_fragment_hdf5(path, payload, version=version)
        return {
            "class": ["FragmentsHDF5", "IterableFragments"],
            "slots": {
                "path": str(path),
                "group": "fragments",
                "compressed": packed,
                "buffer_size": 128,
                "chr_names": payload["chr_names"],
                "cell_names": payload["cell_names"],
            },
        }
    raise AssertionError(backend)


def _peak_matrix_spec(
    fragments: dict[str, Any],
    mode: str,
    *,
    transpose: bool = True,
) -> dict[str, Any]:
    feature_names = ("p0", "p_span", "p1", "p2", "p_chr2")
    cell_names = ("c1", "c2", "c3")
    return {
        "class": ["PeakMatrix", "IterableMatrix"],
        "slots": {
            "fragments": fragments,
            "chr_id": np.asarray([0, 0, 0, 0, 1], dtype=np.int32),
            "start": np.asarray([0, 11, 10, 5, 0], dtype=np.int32),
            "end": np.asarray([10, 14, 20, 25, 10], dtype=np.int32),
            "chr_levels": ("chr1", "chr2"),
            "mode": [mode],
            "transpose": transpose,
            "dim": (5, 3) if transpose else (3, 5),
            "dimnames": (
                (feature_names, cell_names)
                if transpose
                else (cell_names, feature_names)
            ),
        },
    }


def _tile_matrix_spec(
    fragments: dict[str, Any],
    mode: str,
    *,
    transpose: bool = True,
) -> dict[str, Any]:
    feature_names = ("t0", "t1", "t2", "t3", "t4")
    cell_names = ("c1", "c2", "c3")
    return {
        "class": ["TileMatrix", "IterableMatrix"],
        "slots": {
            "fragments": fragments,
            "chr_id": np.asarray([0, 1], dtype=np.int32),
            "start": np.asarray([0, 0], dtype=np.int32),
            "end": np.asarray([30, 15], dtype=np.int32),
            "tile_width": np.asarray([10, 10], dtype=np.int32),
            "chr_levels": ("chr1", "chr2"),
            "mode": [mode],
            "transpose": transpose,
            "dim": (5, 3) if transpose else (3, 5),
            "dimnames": (
                (feature_names, cell_names)
                if transpose
                else (cell_names, feature_names)
            ),
        },
    }


def _write_h5_sparse_group(
    parent: h5py.Group | h5py.File,
    name: str,
    values: NDArray[Any],
    layout: str,
) -> h5py.Group:
    matrix = csr_matrix(values) if layout == "csr" else csc_matrix(values)
    group = parent.create_group(name)
    group.attrs["shape"] = matrix.shape
    group.attrs["encoding-type"] = f"{layout}_matrix"
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices.astype(np.int64))
    group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
    return group


def _write_axis_names(
    handle: h5py.File,
    features: Sequence[str],
    cells: Sequence[str],
) -> None:
    obs = handle.create_group("obs")
    obs.create_dataset("_index", data=np.asarray(cells, dtype="S"))
    var = handle.create_group("var")
    var.create_dataset("_index", data=np.asarray(features, dtype="S"))


def test_dense_r_column_major_source_is_bounded_and_oriented() -> None:
    feature_by_cell = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int16, order="F")
    tracked = _TrackedVector(feature_by_cell.ravel(order="F"))
    source = DenseMatrixSource(
        tracked,
        feature_by_cell.shape,
        row_names=["g1", "g2"],
        column_names=["c1", "c2", "c3"],
    )

    assert isinstance(source, MatrixSource)
    assert source.shape == (2, 3)
    assert source.dtype == np.dtype(np.int16)
    assert source.row_names == ("g1", "g2")
    assert source.column_names == ("c1", "c2", "c3")
    np.testing.assert_array_equal(source.read_cells(1, 3), [[2, 5], [3, 6]])
    assert tracked.reads == [(2, 6)]
    assert source.estimate_read_memory(1, 3).peakBytes > 0


def test_dense_source_enforces_block_memory_limit() -> None:
    source = DenseMatrixSource(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        limits=SourceLimits(maxBlockBytes=32),
    )
    with pytest.raises(ResourceLimitError, match="maxBlockBytes=32"):
        source.read_cells(0, 2)


def test_matrix_source_public_accessors_bounds_and_dtype_validation() -> None:
    source = DenseMatrixSource(
        np.arange(6, dtype=np.int16).reshape(2, 3),
        row_names=["f1", "f2"],
        column_names=["c1", "c2", "c3"],
    )

    assert source.rowNames == source.row_names == ("f1", "f2")
    assert source.columnNames == source.column_names == ("c1", "c2", "c3")
    assert source.n_features == 2
    assert source.n_cells == 3
    assert source.sparse is source.is_sparse is False
    assert source.zeroPreserving is source.zero_preserving is True
    assert source.residentBytes == source.resident_bytes

    estimate = source.memory_estimate(0, 1)
    assert estimate.resident_bytes == estimate.residentBytes
    assert estimate.working_bytes == estimate.workingBytes
    assert estimate.output_bytes == estimate.outputBytes
    assert estimate.peak_bytes == estimate.peakBytes
    assert source.estimate_memory(0, 1) == estimate
    assert source.estimate_read_bytes(0, 1) == estimate.peakBytes
    assert source.estimated_peak_bytes(0, 1) == estimate.peakBytes

    with pytest.raises(TypeError, match="cell bounds must be integers"):
        source.read_cells(True, 1)
    with pytest.raises(IndexError, match=r"cell window \[-1, 1\)"):
        source.read_cells(-1, 1)
    with pytest.raises(MatrixSourceError, match="exactly two dimensions"):
        DenseMatrixSource(np.arange(2), shape=(1, 1, 2))
    with pytest.raises(MatrixSourceError, match="cannot be negative"):
        DenseMatrixSource(np.empty((1, 1)), shape=(-1, 1))
    with pytest.raises(TypeError, match="not numeric"):
        DenseMatrixSource(np.asarray([["text"]]))
    with pytest.raises(TypeError, match="not numeric"):
        DenseMatrixSource(np.asarray([[object()]], dtype=object))
    with pytest.raises(TypeError, match="not numeric"):
        DtypeMatrixSource(source, np.dtype("datetime64[D]"))

    cast = DtypeMatrixSource(source, np.float32)
    values = cast.read_cells(0, 2)
    assert values.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(values, [[0, 3], [1, 4]])


def test_csc_source_preserves_matrix_classes_and_orientation() -> None:
    feature_by_cell = np.asarray([[0, 2, 0], [1, 0, 3], [4, 5, 0]], dtype=np.float64)
    matrix = csc_matrix(feature_by_cell)
    source = CscMatrixSource(
        matrix.data,
        matrix.indices,
        matrix.indptr,
        matrix.shape,
        class_name="dgCMatrix",
    )

    assert source.is_sparse
    assert source.nnz == matrix.nnz
    np.testing.assert_array_equal(
        source.read_cells(1, 3).toarray(),
        feature_by_cell[:, 1:3].T,
    )

    pattern = CscMatrixSource(
        None,
        matrix.indices,
        matrix.indptr,
        matrix.shape,
        class_name="ngCMatrix",
    )
    np.testing.assert_array_equal(
        pattern.read_cells(0, 3).toarray(),
        (feature_by_cell != 0).T,
    )


def test_csc_source_validates_offsets_indexes_and_limits() -> None:
    with pytest.raises(MatrixSourceError, match="nondecreasing"):
        CscMatrixSource([1], [0], [0, 1, 0], (2, 2))
    with pytest.raises(MatrixSourceError, match="out-of-range"):
        CscMatrixSource([1], [2], [0, 1], (2, 1))
    with pytest.raises(ResourceLimitError, match="maxNnz=1"):
        CscMatrixSource(
            [1, 2],
            [0, 1],
            [0, 2],
            (2, 1),
            limits=SourceLimits(maxNnz=1),
        )


def test_mapping_transpose_and_bind_sources() -> None:
    first_values = np.asarray([[1, 2, 3], [4, 5, 6]])
    first = DenseMatrixSource(
        first_values,
        row_names=["f1", "f2"],
        column_names=["c1", "c2", "c3"],
    )
    mapped = MappedMatrixSource(
        first,
        feature_indices=[1, 0, 1],
        cell_indices=[2, 0],
    )
    np.testing.assert_array_equal(
        mapped.read_cells(0, 2),
        [[6, 3, 6], [4, 1, 4]],
    )

    transposed = TransposeMatrixSource(first, tile_cells=1)
    assert transposed.shape == (3, 2)
    np.testing.assert_array_equal(
        transposed.read_cells(0, 2),
        first_values,
    )

    second_features = DenseMatrixSource(
        np.asarray([[7, 8, 9]]),
        row_names=["f3"],
        column_names=["c1", "c2", "c3"],
    )
    feature_bound = FeatureBindMatrixSource([first, second_features])
    np.testing.assert_array_equal(
        feature_bound.read_cells(0, 3),
        np.vstack([first_values, [[7, 8, 9]]]).T,
    )

    second_cells = DenseMatrixSource(
        np.asarray([[7], [8]]),
        row_names=["f1", "f2"],
        column_names=["c4"],
    )
    cell_bound = CellBindMatrixSource([first, second_cells])
    np.testing.assert_array_equal(
        cell_bound.read_cells(2, 4),
        [[3, 6], [7, 8]],
    )


def test_layer_stitching_maps_global_axes_and_rejects_conflicts() -> None:
    layer_one = DenseMatrixSource(
        np.asarray([[1, 2], [3, 4]]),
        row_names=["f1", "f2"],
        column_names=["c1", "c3"],
    )
    layer_two = DenseMatrixSource(
        np.asarray([[5], [6]]),
        row_names=["f2", "f3"],
        column_names=["c2"],
    )
    stitched = LayerStitchMatrixSource(
        [layer_one, layer_two],
        row_names=["f1", "f2", "f3"],
        column_names=["c1", "c2", "c3"],
    )
    np.testing.assert_array_equal(
        stitched.read_cells(0, 3).toarray(),
        [[1, 3, 0], [0, 5, 6], [2, 4, 0]],
    )

    conflicting = DenseMatrixSource(
        np.asarray([[9]]),
        row_names=["f2"],
        column_names=["c1"],
    )
    with pytest.raises(MatrixSourceError, match="coordinate conflict"):
        LayerStitchMatrixSource(
            [
                LayerPlacement(layer_one, name="counts.1"),
                LayerPlacement(conflicting, name="counts.2"),
            ],
            row_names=["f1", "f2", "f3"],
            column_names=["c1", "c2", "c3"],
        )


def test_local_transforms_mask_and_subassignment() -> None:
    values = np.asarray([[0, 2], [3, 0]], dtype=np.float64)
    sparse_matrix = csc_matrix(values)
    source = CscMatrixSource(
        sparse_matrix.data,
        sparse_matrix.indices,
        sparse_matrix.indptr,
        sparse_matrix.shape,
    )
    log_source = UnaryTransformMatrixSource(source, "log1p")
    assert log_source.is_sparse
    np.testing.assert_allclose(
        log_source.read_cells(0, 2).toarray(),
        np.log1p(values.T),
    )

    exp_source = UnaryTransformMatrixSource(source, "exp")
    assert not exp_source.is_sparse
    np.testing.assert_allclose(
        exp_source.read_cells(0, 2),
        np.exp(values.T),
    )

    doubled = BinaryTransformMatrixSource(source, 2, "multiply")
    np.testing.assert_array_equal(doubled.read_cells(0, 2).toarray(), values.T * 2)
    added = BinaryTransformMatrixSource(source, 1, "add")
    assert not added.is_sparse
    np.testing.assert_array_equal(added.read_cells(0, 2), values.T + 1)

    mask = DenseMatrixSource(np.asarray([[True, False], [False, True]]))
    masked = MaskMatrixSource(source, mask)
    np.testing.assert_array_equal(
        masked.read_cells(0, 2).toarray(),
        [[0, 0], [0, 0]],
    )

    assigned = DelayedSubassignmentMatrixSource(
        source,
        [
            Subassignment([0], [1], 7),
            Subassignment([1], [0], np.asarray([[8]])),
        ],
    )
    np.testing.assert_array_equal(assigned.read_cells(0, 2), [[0, 8], [7, 0]])


def test_binary_matrix_operations_cover_sparse_and_reverse_paths() -> None:
    left_values = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.int32)
    right_values = np.asarray([[0, 5, 2], [7, 0, 1]], dtype=np.int32)

    def sparse_source(values: NDArray[Any]) -> CscMatrixSource:
        matrix = csc_matrix(values)
        return CscMatrixSource(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            matrix.shape,
        )

    left = sparse_source(left_values)
    right = sparse_source(right_values)
    operations = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "logical_and": np.logical_and,
    }
    for operation, function in operations.items():
        transformed = BinaryTransformMatrixSource(left, right, operation)
        assert transformed.is_sparse
        np.testing.assert_array_equal(
            transformed.read_cells(0, 3).toarray(),
            function(left_values.T, right_values.T),
        )

    reverse_sparse = BinaryTransformMatrixSource(
        left,
        2,
        "multiply",
        reverse=True,
    )
    np.testing.assert_array_equal(
        reverse_sparse.read_cells(0, 3).toarray(),
        left_values.T * 2,
    )
    reverse_dense = BinaryTransformMatrixSource(
        left,
        10,
        "subtract",
        reverse=True,
    )
    assert not reverse_dense.is_sparse
    np.testing.assert_array_equal(
        reverse_dense.read_cells(0, 3),
        10 - left_values.T,
    )

    with pytest.raises(TypeError, match="numeric scalar"):
        BinaryTransformMatrixSource(left, object(), "add")
    with pytest.raises(MatrixSourceError, match="reversed binary operations"):
        BinaryTransformMatrixSource(left, right, "add", reverse=True)
    with pytest.raises(MatrixSourceError, match="source shapes differ"):
        BinaryTransformMatrixSource(
            left,
            DenseMatrixSource(np.ones((1, 1))),
            "add",
        )


def test_sparse_transpose_bind_and_cast_operations() -> None:
    values = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.int16)
    matrix = csc_matrix(values)
    source = CscMatrixSource(
        matrix.data,
        matrix.indices,
        matrix.indptr,
        matrix.shape,
    )

    transposed = TransposeMatrixSource(source, tile_cells=1)
    np.testing.assert_array_equal(
        transposed.read_cells(0, 2).toarray(),
        values,
    )
    assert transposed.read_cells(1, 1).shape == (0, 3)

    feature_bound = FeatureBindMatrixSource([source, source])
    np.testing.assert_array_equal(
        feature_bound.read_cells(0, 3).toarray(),
        np.hstack([values.T, values.T]),
    )
    cell_bound = CellBindMatrixSource([source, source])
    np.testing.assert_array_equal(
        cell_bound.read_cells(0, 6).toarray(),
        np.vstack([values.T, values.T]),
    )
    assert cell_bound.read_cells(2, 2).shape == (0, 2)

    cast = DtypeMatrixSource(source, np.float32)
    cast_block = cast.read_cells(0, 3)
    assert cast_block.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(cast_block.toarray(), values.T)


def test_operation_registry_builds_structural_nodes() -> None:
    source = DenseMatrixSource(
        np.asarray([[1, 2], [3, 4]]),
        row_names=["f1", "f2"],
        column_names=["c1", "c2"],
    )
    subset = build_matrix_operation(
        {
            "operation": "subset",
            "className": "DelayedSubset",
            "featureIndices": [1],
            "cellIndices": [1, 0],
        },
        source=source,
        object_path="assays/RNA/counts",
    )
    np.testing.assert_array_equal(subset.read_cells(0, 2), [[4], [3]])

    renamed = build_matrix_operation(
        {
            "operation": "dimnames",
            "className": "DelayedSetDimnames",
            "rowNames": ["a", "b"],
            "columnNames": ["x", "y"],
        },
        source=source,
    )
    assert isinstance(renamed, RenamedMatrixSource)
    assert renamed.row_names == ("a", "b")

    cast = build_matrix_operation(
        {
            "operation": "dtype",
            "className": "DtypeMatrix",
            "dtype": "float32",
        },
        source=source,
    )
    assert isinstance(cast, DtypeMatrixSource)
    assert cast.dtype == np.dtype(np.float32)

    bound = build_matrix_operation(
        {
            "operation": "rbind",
            "class": ["DelayedAbind", "DelayedMatrix"],
            "sources": [source, source],
        }
    )
    np.testing.assert_array_equal(
        bound.read_cells(0, 2),
        np.hstack([source.read_cells(0, 2), source.read_cells(0, 2)]),
    )

    transposed = build_matrix_operation(
        {
            "operation": "aperm",
            "className": "DelayedAperm",
            "permutation": [2, 1],
        },
        source=source,
    )
    np.testing.assert_array_equal(
        transposed.read_cells(0, 2),
        np.asarray([[1, 2], [3, 4]]),
    )

    mask = DenseMatrixSource(np.asarray([[1, 0], [0, 1]]))
    dropped = build_matrix_operation(
        {
            "operation": "mask",
            "className": "MatrixMask",
            "mask": mask,
            "invert": False,
        },
        source=source,
    )
    np.testing.assert_array_equal(dropped.read_cells(0, 2), [[0, 3], [2, 0]])


def test_plain_slot_mapping_factory_has_no_rds_parser_dependency(
    tmp_path: Path,
) -> None:
    dense = matrix_source_from_slots(
        {
            "class": ["matrix", "array"],
            "slots": {
                ".Data": [1, 3, 2, 4],
                "dim": [2, 2],
                "dimnames": [["f1", "f2"], ["c1", "c2"]],
            },
        },
        object_path="assays/RNA/counts",
    )
    np.testing.assert_array_equal(dense.read_cells(0, 2), [[1, 3], [2, 4]])

    sparse = csc_matrix(np.asarray([[1, 0], [0, 2]]))
    csc_source = matrix_source_from_slots(
        {
            "className": "dgCMatrix",
            "x": sparse.data,
            "i": sparse.indices,
            "p": sparse.indptr,
            "Dim": sparse.shape,
        }
    )
    np.testing.assert_array_equal(
        csc_source.read_cells(0, 2).toarray(), [[1, 0], [0, 2]]
    )
    wrapped = matrix_source_from_slots(
        {
            "class": ["Iterable_dgCMatrix_wrapper", "IterableMatrix"],
            "slots": {
                "mat": {
                    "className": "dgCMatrix",
                    "x": sparse.data,
                    "i": sparse.indices,
                    "p": sparse.indptr,
                    "Dim": sparse.shape,
                },
                "dim": sparse.shape,
                "transpose": False,
            },
        }
    )
    np.testing.assert_array_equal(wrapped.read_cells(0, 2).toarray(), [[1, 0], [0, 2]])

    rds = tmp_path / "object.rds"
    rds.write_bytes(b"rds")
    sidecar = tmp_path / "counts.h5"
    with h5py.File(sidecar, mode="w") as handle:
        handle.create_dataset("counts", data=np.asarray([[1, 2], [3, 4]]))
    hdf5_source = matrix_source_from_slots(
        {
            "class": "HDF5ArraySeed",
            "filepath": "counts.h5",
            "name": "counts",
        },
        rds_path=rds,
    )
    np.testing.assert_array_equal(hdf5_source.read_cells(0, 2), [[1, 2], [3, 4]])


def test_explicit_factory_operations_resolve_nested_sources_and_overrides() -> None:
    leaf = {
        "class": ["matrix", "array"],
        "slots": {
            ".Data": [1, 3, 2, 4],
            "dim": [2, 2],
        },
    }
    cast = matrix_source_from_slots(
        {
            "operation": "dtype",
            "className": "DtypeMatrix",
            "source": leaf,
            "dtype": "float32",
        },
        object_path="assays/RNA/layers/cast",
    )
    assert cast.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(cast.read_cells(0, 2), [[1, 3], [2, 4]])

    bound = matrix_source_from_slots(
        {
            "operation": "cbind",
            "className": "ColumnBindMatrix",
            "sources": [leaf, leaf],
        },
        object_path="assays/RNA/layers/bound",
    )
    np.testing.assert_array_equal(
        bound.read_cells(0, 4),
        [[1, 3], [2, 4], [1, 3], [2, 4]],
    )

    overridden = matrix_source_from_slots(
        {
            "class": "CustomDenseMatrix",
            "data": [1, 2],
            "dim": [1, 2],
            "dtype": "float64",
        },
        class_name="matrix",
        object_path="assays/RNA/layers/overridden",
    )
    assert overridden.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(overridden.read_cells(0, 2), [[1], [2]])


def test_factory_missing_structural_inputs_have_meaningful_paths() -> None:
    with pytest.raises(MatrixSourceError, match="missing one of.*seed"):
        matrix_source_from_slots(
            {
                "class": ["DelayedMatrix", "DelayedArray"],
                "slots": {},
            },
            object_path="assays/RNA/layers/counts",
        )
    with pytest.raises(MatrixSourceError, match="has no MatrixSource input"):
        matrix_source_from_slots(
            {
                "operation": "dtype",
                "className": "DtypeMatrix",
                "dtype": "float32",
            },
            object_path="assays/RNA/layers/cast",
        )
    with pytest.raises(MatrixSourceError, match="has no dtype"):
        build_matrix_operation(
            {
                "operation": "dtype",
                "className": "DtypeMatrix",
            },
            source=DenseMatrixSource(np.eye(2)),
            object_path="assays/RNA/layers/cast",
        )


def test_factory_executes_structural_delayedarray_nodes() -> None:
    leaf = {
        "class": ["matrix", "array"],
        "slots": {
            ".Data": [1, 2, 3, 4, 5, 6],
            "dim": [3, 2],
        },
    }
    delayed = {
        "class": ["DelayedMatrix", "DelayedArray"],
        "slots": {"seed": leaf},
    }
    source = matrix_source_from_slots(delayed, object_path="assays/RNA/counts")
    np.testing.assert_array_equal(
        source.read_cells(0, 2),
        [[1, 2, 3], [4, 5, 6]],
    )

    subset = matrix_source_from_slots(
        {
            "class": ["DelayedSubset", "DelayedUnaryOp"],
            "slots": {
                "seed": delayed,
                "index": [[3, 1], [2]],
            },
        },
        object_path="assays/RNA/subset",
    )
    np.testing.assert_array_equal(subset.read_cells(0, 1), [[6, 4]])

    transposed = matrix_source_from_slots(
        {
            "class": ["DelayedAperm", "DelayedUnaryOp"],
            "slots": {
                "seed": delayed,
                "perm": [2, 1],
            },
        },
        object_path="assays/RNA/aperm",
    )
    np.testing.assert_array_equal(
        transposed.read_cells(0, 3),
        [[1, 4], [2, 5], [3, 6]],
    )

    bound = matrix_source_from_slots(
        {
            "class": ["DelayedAbind", "DelayedNaryOp"],
            "slots": {
                "seeds": [delayed, delayed],
                "along": [2],
            },
        },
        object_path="assays/RNA/abind",
    )
    np.testing.assert_array_equal(
        bound.read_cells(0, 4),
        [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]],
    )

    assigned = matrix_source_from_slots(
        {
            "class": ["DelayedSubassign", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": delayed,
                "Lindex": [[2], [1, 2]],
                "Rvalue": [9],
            },
        },
        object_path="assays/RNA/subassign",
    )
    np.testing.assert_array_equal(
        assigned.read_cells(0, 2),
        [[1, 9, 3], [4, 9, 6]],
    )


def test_factory_executes_allowlisted_delayedarray_primitives() -> None:
    source = DenseMatrixSource(np.asarray([[1, 2], [3, 4]], dtype=np.float64))
    right_added = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": source,
                "OP": "+",
                "Largs": [],
                "Rargs": [2],
            },
        }
    )
    np.testing.assert_array_equal(right_added.read_cells(0, 2), [[3, 5], [4, 6]])

    left_subtracted = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": source,
                "OP": "-",
                "Largs": [10],
                "Rargs": [],
            },
        }
    )
    np.testing.assert_array_equal(
        left_subtracted.read_cells(0, 2),
        [[9, 7], [8, 6]],
    )

    stacked = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpStack", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": DenseMatrixSource(-np.square(np.asarray([[1, 2], [3, 4]]))),
                "OPS": ["abs", "sqrt"],
            },
        }
    )
    np.testing.assert_array_equal(stacked.read_cells(0, 2), [[1, 3], [2, 4]])

    nary = matrix_source_from_slots(
        {
            "class": ["DelayedNaryIsoOp", "DelayedNaryOp"],
            "slots": {
                "seeds": [source, source],
                "OP": "+",
                "Rargs": [1],
            },
        }
    )
    np.testing.assert_array_equal(nary.read_cells(0, 2), [[3, 7], [5, 9]])


def test_factory_resolves_nested_bpcells_operations_and_fragments(
    tmp_path: Path,
) -> None:
    nested = matrix_source_from_slots(
        {
            "class": ["MatrixSubset", "IterableMatrix"],
            "slots": {
                "matrix": {
                    "class": [
                        "TransformLog1p",
                        "TransformedMatrix",
                        "IterableMatrix",
                    ],
                    "slots": {
                        "matrix": {
                            "class": ["matrix", "array"],
                            "slots": {
                                ".Data": [1, 2, 3, 4, 5, 6],
                                "dim": [3, 2],
                            },
                        },
                        "dim": [3, 2],
                    },
                },
                "row_selection": [3, 1],
                "col_selection": [2],
                "zero_dims": [False, False],
                "dim": [2, 1],
                "dimnames": [["f3", "f1"], ["c2"]],
            },
        },
        object_path="assays/RNA/data",
    )
    np.testing.assert_allclose(nested.read_cells(0, 1), np.log1p([[6, 4]]))
    assert nested.row_names == ("f3", "f1")
    assert nested.column_names == ("c2",)

    fragment = matrix_source_from_slots(
        _peak_matrix_spec(
            _fragment_leaf_spec(
                tmp_path,
                backend="memory",
                packed=False,
                version=2,
            ),
            "fragments",
        ),
        object_path="assays/ATAC/counts",
    )
    assert isinstance(fragment, RenamedMatrixSource)
    assert isinstance(fragment.source, FragmentDerivedMatrixSource)
    np.testing.assert_array_equal(
        fragment.read_cells(1, 3).toarray(),
        [
            [1, 0, 1, 2, 0],
            [0, 1, 1, 1, 1],
        ],
    )


def test_factory_uses_dimensions_to_separate_storage_order_from_transpose(
    tmp_path: Path,
) -> None:
    values = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.uint32)
    payload = _bpcells_payload(
        values,
        packed=True,
        version=2,
        storage_order="row",
    )
    matrix_dir = tmp_path / "row-matrix"
    _write_bpcells_directory(matrix_dir, payload, version=2)

    source = matrix_source_from_slots(
        {
            "class": ["MatrixDir", "IterableMatrix"],
            "slots": {
                "dir": matrix_dir,
                "dim": list(values.shape),
                "dimnames": [payload["row_names"], payload["col_names"]],
                "transpose": True,
            },
        }
    )
    np.testing.assert_array_equal(source.read_cells(0, 3).toarray(), values.T)

    transposed = matrix_source_from_slots(
        {
            "class": ["MatrixDir", "IterableMatrix"],
            "slots": {
                "dir": matrix_dir,
                "dim": list(values.shape[::-1]),
                "dimnames": [payload["col_names"], payload["row_names"]],
                "transpose": False,
            },
        }
    )
    np.testing.assert_array_equal(transposed.read_cells(0, 2).toarray(), values)


def test_unknown_registry_function_and_class_report_object_path() -> None:
    source = DenseMatrixSource(np.eye(2))
    with pytest.raises(UnsupportedMatrixOperation) as function_error:
        build_matrix_operation(
            {
                "operation": "unary",
                "className": "DelayedUnaryIsoOpStack",
                "function": "custom_package_function",
            },
            source=source,
            object_path="assays/RNA/layers/custom",
        )
    message = str(function_error.value)
    assert "assays/RNA/layers/custom" in message
    assert "custom_package_function" in message
    assert "DelayedUnaryIsoOpStack" in message

    with pytest.raises(UnsupportedMatrixOperation) as class_error:
        build_matrix_operation(
            {"operation": "subset", "className": "CustomSeed"},
            source=source,
            object_path="assays/RNA/counts",
        )
    message = str(class_error.value)
    assert "assays/RNA/counts" in message
    assert "CustomSeed" in message
    assert "subset" in message


def test_fragment_wrapper_reports_nested_custom_source_path() -> None:
    with pytest.raises(UnsupportedMatrixOperation) as error:
        matrix_source_from_slots(
            {
                "class": ["PeakMatrix", "IterableMatrix"],
                "slots": {
                    "fragments": {
                        "class": ["ShiftFragments", "IterableFragments"],
                        "slots": {
                            "fragments": {
                                "class": ["CustomFragments", "IterableFragments"],
                                "slots": {},
                            },
                            "shift_start": [4],
                            "shift_end": [-4],
                        },
                    },
                    "chr_id": [0],
                    "start": [0],
                    "end": [10],
                    "chr_levels": ["chr1"],
                    "mode": ["insertions"],
                    "transpose": True,
                    "dim": [1, 1],
                },
            },
            object_path="assays/ATAC/counts",
        )
    assert error.value.objectPath == "assays/ATAC/counts@fragments@fragments"
    assert error.value.className == "CustomFragments"
    assert error.value.reason == "unknown or custom fragment class"


def test_custom_fragment_graph_rejection_has_exact_object_path() -> None:
    with pytest.raises(UnsupportedMatrixOperation) as error:
        matrix_source_from_slots(
            {
                "class": ["PeakMatrix", "IterableMatrix"],
                "slots": {
                    "fragments": {
                        "class": ["CustomFragments", "IterableFragments"],
                        "slots": {"loader": "custom::open_fragments"},
                    },
                    "chr_id": [0],
                    "start": [0],
                    "end": [10],
                    "chr_levels": ["chr1"],
                    "mode": ["insertions"],
                    "transpose": True,
                    "dim": [1, 1],
                },
            },
            object_path="assays/ATAC/counts",
        )

    assert error.value.objectPath == "assays/ATAC/counts@fragments"
    assert error.value.className == "CustomFragments"
    assert error.value.reason == "unknown or custom fragment class"


def test_fragment_capability_registry_has_exact_supported_profile() -> None:
    assert FRAGMENT_CAPABILITY_REGISTRY.acceptedClasses == (
        "UnpackedMemFragments",
        "PackedMemFragments",
        "FragmentsDir",
        "FragmentsHDF5",
        "ShiftFragments",
        "SelectLength",
        "ChrSelectName",
        "ChrSelectIndex",
        "CellSelectName",
        "CellSelectIndex",
        "CellMerge",
        "ChrRename",
        "CellRename",
        "CellPrefix",
        "RegionSelect",
        "MergeFragments",
    )


@pytest.mark.parametrize("backend", ["memory", "directory", "hdf5"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_fragment_leaves_and_all_formal_layouts_execute_through_factory(
    tmp_path: Path,
    backend: str,
    packed: bool,
    version: int,
) -> None:
    source = matrix_source_from_slots(
        _peak_matrix_spec(
            _fragment_leaf_spec(
                tmp_path,
                backend=backend,
                packed=packed,
                version=version,
            ),
            "overlaps",
        ),
        object_path="assays/ATAC/counts",
    )

    assert source.shape == (5, 3)
    assert source.dtype == np.dtype(np.uint32)
    assert source.row_names == ("p0", "p_span", "p1", "p2", "p_chr2")
    assert source.column_names == ("c1", "c2", "c3")
    np.testing.assert_array_equal(
        source.read_cells(1, 3).toarray(),
        [
            [1, 1, 1, 2, 0],
            [0, 1, 1, 1, 1],
        ],
    )


def test_fragment_records_are_read_in_bounded_chromosome_blocks() -> None:
    count = 130
    cells = np.ones(count, dtype=np.int32)
    cells[0] = 0
    starts = np.arange(count, dtype=np.int32)
    ends = starts + 1
    tracked_cells = _TrackedVector(cells)
    tracked_starts = _TrackedVector(starts)
    tracked_ends = _TrackedVector(ends)
    source = matrix_source_from_slots(
        {
            "class": ["PeakMatrix", "IterableMatrix"],
            "slots": {
                "fragments": {
                    "class": ["UnpackedMemFragments", "IterableFragments"],
                    "slots": {
                        "version": ["unpacked-fragments-v2"],
                        "cell": tracked_cells,
                        "start": tracked_starts,
                        "end": tracked_ends,
                        "end_max": np.asarray([128, 130], dtype=np.int32),
                        "chr_ptr": np.asarray([0, count], dtype=np.float64),
                        "chr_names": ("chr1",),
                        "cell_names": ("c1", "c2"),
                    },
                },
                "chr_id": np.asarray([0], dtype=np.int32),
                "start": np.asarray([0], dtype=np.int32),
                "end": np.asarray([1], dtype=np.int32),
                "chr_levels": ("chr1",),
                "mode": ["insertions"],
                "transpose": True,
                "dim": (1, 2),
                "dimnames": (("p1",), ("c1", "c2")),
            },
        },
        limits=SourceLimits(maxBlockBytes=16_384),
    )

    np.testing.assert_array_equal(source.read_cells(0, 1).toarray(), [[2]])
    for tracked in (tracked_cells, tracked_starts, tracked_ends):
        assert len(tracked.reads) > 1
        assert all(
            read_start is not None
            and read_stop is not None
            and read_stop - read_start <= 128
            for read_start, read_stop in tracked.reads
        )


def test_fragment_end_max_matches_bpcells_at_aligned_chromosome_boundary() -> None:
    first_starts = np.arange(128, dtype=np.uint32)
    starts = np.concatenate((first_starts, np.asarray([0, 1], dtype=np.uint32)))
    ends = np.concatenate(
        (first_starts + np.uint32(1000), np.asarray([1, 2], dtype=np.uint32))
    )
    source = matrix_source_from_slots(
        {
            "class": ["PeakMatrix", "IterableMatrix"],
            "slots": {
                "fragments": {
                    "class": ["UnpackedMemFragments", "IterableFragments"],
                    "slots": {
                        "version": ["unpacked-fragments-v2"],
                        "cell": np.zeros(130, dtype=np.int32),
                        "start": starts.view(np.int32),
                        "end": ends.view(np.int32),
                        "end_max": np.asarray([1127, 1127], dtype=np.int32),
                        "chr_ptr": np.asarray(
                            [0, 128, 128, 130],
                            dtype=np.float64,
                        ),
                        "chr_names": ("chr1", "chr2"),
                        "cell_names": ("c1",),
                    },
                },
                "chr_id": np.asarray([1], dtype=np.int32),
                "start": np.asarray([0], dtype=np.int32),
                "end": np.asarray([2], dtype=np.int32),
                "chr_levels": ("chr1", "chr2"),
                "mode": ["insertions"],
                "transpose": True,
                "dim": (1, 1),
                "dimnames": (("p1",), ("c1",)),
            },
        }
    )

    np.testing.assert_array_equal(source.read_cells(0, 1).toarray(), [[4]])


@pytest.mark.parametrize(
    "mode,expected",
    [
        (
            "insertions",
            [
                [2, 0, 2, 3, 1],
                [1, 0, 1, 3, 0],
                [0, 1, 2, 2, 2],
            ],
        ),
        (
            "fragments",
            [
                [1, 0, 1, 2, 1],
                [1, 0, 1, 2, 0],
                [0, 1, 1, 1, 1],
            ],
        ),
        (
            "overlaps",
            [
                [1, 1, 1, 2, 1],
                [1, 1, 1, 2, 0],
                [0, 1, 1, 1, 1],
            ],
        ),
    ],
)
def test_peak_matrix_modes_match_bpcells_endpoint_semantics(
    tmp_path: Path,
    mode: str,
    expected: list[list[int]],
) -> None:
    source = matrix_source_from_slots(
        _peak_matrix_spec(
            _fragment_leaf_spec(
                tmp_path,
                backend="memory",
                packed=True,
                version=2,
            ),
            mode,
        )
    )
    np.testing.assert_array_equal(source.read_cells(0, 3).toarray(), expected)


@pytest.mark.parametrize(
    "mode,expected",
    [
        (
            "insertions",
            [
                [2, 2, 0, 1, 1],
                [1, 1, 2, 0, 0],
                [0, 2, 0, 2, 0],
            ],
        ),
        (
            "fragments",
            [
                [1, 1, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
            ],
        ),
    ],
)
def test_tile_matrix_modes_match_bpcells_duplicate_endpoint_handling(
    tmp_path: Path,
    mode: str,
    expected: list[list[int]],
) -> None:
    source = matrix_source_from_slots(
        _tile_matrix_spec(
            _fragment_leaf_spec(
                tmp_path,
                backend="memory",
                packed=False,
                version=1,
            ),
            mode,
        )
    )
    np.testing.assert_array_equal(source.read_cells(0, 3).toarray(), expected)


def test_fragment_matrix_honors_native_storage_orientation(tmp_path: Path) -> None:
    source = matrix_source_from_slots(
        _peak_matrix_spec(
            _fragment_leaf_spec(
                tmp_path,
                backend="memory",
                packed=False,
                version=2,
            ),
            "fragments",
            transpose=False,
        )
    )

    assert source.shape == (3, 5)
    assert source.row_names == ("c1", "c2", "c3")
    assert source.column_names == ("p0", "p_span", "p1", "p2", "p_chr2")
    np.testing.assert_array_equal(
        source.read_cells(1, 4).toarray(),
        [
            [0, 0, 1],
            [1, 1, 1],
            [2, 2, 1],
        ],
    )


def test_fragment_metadata_is_validated_before_reads(tmp_path: Path) -> None:
    malformed = _fragment_leaf_spec(
        tmp_path,
        backend="memory",
        packed=False,
        version=2,
    )
    malformed["slots"]["end_max"] = np.asarray([29], dtype=np.int32)
    with pytest.raises(MatrixSourceError, match="end_max is inconsistent at block 0"):
        matrix_source_from_slots(
            _peak_matrix_spec(malformed, "insertions"),
            object_path="assays/ATAC/counts",
        )

    unsorted = _fragment_leaf_spec(
        tmp_path,
        backend="memory",
        packed=False,
        version=2,
    )
    unsorted["slots"]["start"] = np.asarray(
        [0, 5, 10, 12, 20, 4, 1],
        dtype=np.int32,
    )
    with pytest.raises(MatrixSourceError, match="not sorted on chromosome 1"):
        matrix_source_from_slots(_peak_matrix_spec(unsorted, "insertions"))

    mismatched_levels = _peak_matrix_spec(
        _fragment_leaf_spec(
            tmp_path,
            backend="memory",
            packed=False,
            version=2,
        ),
        "insertions",
    )
    mismatched_levels["slots"]["chr_levels"] = ("chr2", "chr1")
    with pytest.raises(MatrixSourceError, match="do not match the fragment source"):
        matrix_source_from_slots(mismatched_levels)

    invalid_cell = _fragment_leaf_spec(
        tmp_path,
        backend="memory",
        packed=False,
        version=2,
    )
    invalid_cell["slots"]["cell"] = np.asarray(
        [3, 0, 1, 0, 2, 2, 1],
        dtype=np.int32,
    )
    with pytest.raises(MatrixSourceError, match="cell ID is out of range at record 0"):
        matrix_source_from_slots(_peak_matrix_spec(invalid_cell, "insertions"))

    invalid_coordinates = _fragment_leaf_spec(
        tmp_path,
        backend="memory",
        packed=False,
        version=2,
    )
    invalid_coordinates["slots"]["end"] = np.asarray(
        [3, 4, 7, 10, 10, 4, 12],
        dtype=np.int32,
    )
    with pytest.raises(MatrixSourceError, match="end precedes start at record 1"):
        matrix_source_from_slots(_peak_matrix_spec(invalid_coordinates, "insertions"))

    invalid_pointers = _fragment_leaf_spec(
        tmp_path,
        backend="memory",
        packed=False,
        version=2,
    )
    invalid_pointers["slots"]["chr_ptr"] = np.asarray(
        [0, 4, 5, 7],
        dtype=np.float64,
    )
    with pytest.raises(MatrixSourceError, match="chromosome 1 starts at 5; expected 4"):
        matrix_source_from_slots(_peak_matrix_spec(invalid_pointers, "insertions"))

    invalid_chromosome = _peak_matrix_spec(
        _fragment_leaf_spec(
            tmp_path,
            backend="memory",
            packed=False,
            version=2,
        ),
        "insertions",
    )
    invalid_chromosome["slots"]["chr_id"] = np.asarray(
        [0, 0, 0, 0, 2],
        dtype=np.int32,
    )
    with pytest.raises(MatrixSourceError, match="out-of-range chromosome ID"):
        matrix_source_from_slots(invalid_chromosome)

    invalid_dimensions = _peak_matrix_spec(
        _fragment_leaf_spec(
            tmp_path,
            backend="memory",
            packed=False,
            version=2,
        ),
        "insertions",
    )
    invalid_dimensions["slots"]["dim"] = (4, 3)
    with pytest.raises(MatrixSourceError, match="does not match derived shape"):
        matrix_source_from_slots(invalid_dimensions)

    invalid_compression = _fragment_leaf_spec(
        tmp_path,
        backend="directory",
        packed=False,
        version=2,
    )
    invalid_compression["slots"]["compressed"] = True
    with pytest.raises(MatrixSourceError, match="compressed slot.*disagrees"):
        matrix_source_from_slots(_peak_matrix_spec(invalid_compression, "insertions"))


@pytest.mark.parametrize(
    "limits,match",
    [
        (SourceLimits(maxFeatures=4), "maxFeatures=4"),
        (SourceLimits(maxCells=2), "maxCells=2"),
        (SourceLimits(maxNnz=6), "maxNnz=6"),
        (SourceLimits(maxMetadataBytes=64), "maxMetadataBytes=64"),
        (SourceLimits(maxBlockBytes=1024), "maxBlockBytes=1024"),
    ],
)
def test_fragment_sources_enforce_resource_limits_at_inspection(
    tmp_path: Path,
    limits: SourceLimits,
    match: str,
) -> None:
    with pytest.raises(ResourceLimitError, match=match):
        matrix_source_from_slots(
            _peak_matrix_spec(
                _fragment_leaf_spec(
                    tmp_path,
                    backend="memory",
                    packed=True,
                    version=2,
                ),
                "overlaps",
            ),
            limits=limits,
        )


def test_fragment_output_enforces_nnz_limit(tmp_path: Path) -> None:
    source = matrix_source_from_slots(
        _peak_matrix_spec(
            _fragment_leaf_spec(
                tmp_path,
                backend="memory",
                packed=False,
                version=2,
            ),
            "overlaps",
        ),
        limits=SourceLimits(maxNnz=7),
    )
    with pytest.raises(ResourceLimitError, match="maxNnz=7"):
        source.read_cells(0, 3)


def test_documented_fragment_wrappers_execute(tmp_path: Path) -> None:
    base = _fragment_leaf_spec(
        tmp_path,
        backend="memory",
        packed=False,
        version=2,
    )

    def wrap(class_name: str, **slots: Any):
        return fragment_source_from_slots(
            {
                "class": [class_name, "IterableFragments"],
                "slots": {"fragments": base, **slots},
            }
        )

    shifted = wrap("ShiftFragments", shift_start=1, shift_end=2)
    first_original = next(fragment_source_from_slots(base).iter_chromosome(0))
    first_shifted = next(shifted.iter_chromosome(0))
    np.testing.assert_array_equal(first_shifted.starts, first_original.starts + 1)
    np.testing.assert_array_equal(first_shifted.ends, first_original.ends + 2)

    length_selected = wrap("SelectLength", min_len=10, max_len=12)
    assert all(
        np.all((block.ends - block.starts >= 10) & (block.ends - block.starts <= 12))
        for chromosome in range(len(length_selected.chromosomeNames))
        for block in length_selected.iter_chromosome(chromosome)
    )

    chromosome_by_name = wrap("ChrSelectName", chr_names=["chr2"])
    chromosome_by_index = wrap("ChrSelectIndex", chr_index_selection=[2])
    assert chromosome_by_name.chromosomeNames == ("chr2",)
    assert chromosome_by_index.chromosomeNames == ("chr2",)

    cells_by_name = wrap("CellSelectName", cell_names=["c3", "c1"])
    cells_by_index = wrap("CellSelectIndex", cell_index_selection=[3, 1])
    assert cells_by_name.cellNames == ("c3", "c1")
    assert cells_by_index.cellNames == ("c3", "c1")

    merged_cells = wrap(
        "CellMerge",
        group_ids=[0, 0, 1],
        group_names=["ab", "c"],
    )
    assert merged_cells.cellNames == ("ab", "c")

    renamed_chromosomes = wrap(
        "ChrRename",
        chr_names=["one", "two"],
    )
    renamed_cells = wrap(
        "CellRename",
        cell_names=["one", "two", "three"],
    )
    prefixed_cells = wrap("CellPrefix", prefix="sample_")
    assert renamed_chromosomes.chromosomeNames == ("one", "two")
    assert renamed_cells.cellNames == ("one", "two", "three")
    assert prefixed_cells.cellNames == ("sample_c1", "sample_c2", "sample_c3")

    selected_regions = wrap(
        "RegionSelect",
        chr_id=[0],
        start=[0],
        end=[10],
        chr_levels=["chr1", "chr2"],
        invert_selection=False,
    )
    assert all(
        np.all((block.starts <= 10) & (block.ends >= 0))
        for block in selected_regions.iter_chromosome(0)
    )

    merged = fragment_source_from_slots(
        {
            "class": ["MergeFragments", "IterableFragments"],
            "slots": {"fragments_list": [base, base]},
        }
    )
    assert merged.chromosomeNames == ("chr1", "chr2")
    assert merged.cellNames == ("c1", "c2", "c3", "c1", "c2", "c3")
    assert all(
        int(block.cellIds.max(initial=0)) < len(merged.cellNames)
        for chromosome in range(len(merged.chromosomeNames))
        for block in merged.iter_chromosome(chromosome)
    )


@pytest.mark.parametrize("class_name", ["FragmentsTsv", "IterableFragments"])
def test_unsupported_fragment_sources_are_typed_rejections(class_name: str) -> None:
    with pytest.raises(UnsupportedMatrixOperation) as error:
        fragment_source_from_slots(
            {
                "class": [class_name, "IterableFragments"],
                "slots": {},
            },
            object_path="assays/ATAC/counts@fragments",
        )
    assert error.value.objectPath == "assays/ATAC/counts@fragments"
    assert error.value.className == class_name
    assert error.value.operation == "fragment-source"


def test_rank_transform_matches_bpcells_zero_offset_and_ties() -> None:
    dense = DenseMatrixSource(
        np.array(
            [
                [0, -1, 2],
                [2, 0, 1],
                [2, 0, 0],
                [5, 3, 0],
            ],
            dtype=np.float64,
        )
    )
    ranked = RankMatrixSource(dense)
    np.testing.assert_array_equal(
        ranked.read_cells(0, 3),
        np.array(
            [
                [0, 1.5, 1.5, 3],
                [-1.5, 0, 0, 1.5],
                [2.5, 1.5, 0, 0],
            ],
            dtype=np.float64,
        ),
    )

    sparse = CscMatrixSource(
        np.array([2.0, -1.0, 2.0, 3.0]),
        np.array([1, 3, 2, 3]),
        np.array([0, 2, 4]),
        (4, 2),
    )
    sparse_ranked = RankMatrixSource(sparse)
    assert sparse_ranked.is_sparse
    np.testing.assert_array_equal(
        sparse_ranked.read_cells(0, 2).toarray(),
        np.array(
            [
                [0, 1.5, 0, -1.5],
                [0, 0, 1.5, 2.5],
            ],
            dtype=np.float64,
        ),
    )


def test_rank_transform_factory_and_nonlocal_axis_contract() -> None:
    source = DenseMatrixSource(np.eye(3))
    ranked = matrix_source_from_slots(
        {
            "class": "MatrixRankTransform",
            "matrix": source,
            "transpose": False,
            "dim": (3, 3),
        },
        object_path="assays/RNA/counts",
    )
    assert isinstance(ranked, RankMatrixSource)
    np.testing.assert_array_equal(ranked.read_cells(0, 3), np.eye(3) * 1.5)

    row_ranked = matrix_source_from_slots(
        {
            "class": "MatrixRankTransform",
            "matrix": source,
            "transpose": True,
        },
        object_path="assays/RNA/counts",
        limits=SourceLimits(tileCells=1),
    )
    np.testing.assert_array_equal(row_ranked.read_cells(1, 3), np.eye(3)[1:] * 1.5)


def test_row_rank_streams_dense_and_sparse_sources() -> None:
    dense = DenseMatrixSource(
        np.array(
            [
                [0, -1, 2],
                [2, 0, 1],
                [2, 0, 0],
                [5, 3, 0],
            ],
            dtype=np.float64,
        )
    )
    dense_ranked = RankMatrixSource(
        dense,
        axis="row",
        limits=SourceLimits(tileCells=1),
    )
    np.testing.assert_array_equal(
        dense_ranked.read_cells(1, 3),
        [[-1, 0, 0, 1], [1, 1, 0, 0]],
    )

    sparse = CscMatrixSource(
        np.array([2.0, -1.0, 2.0, 3.0]),
        np.array([1, 3, 2, 3]),
        np.array([0, 2, 4]),
        (4, 2),
    )
    sparse_ranked = RankMatrixSource(
        sparse,
        axis="row",
        limits=SourceLimits(tileCells=1),
    )
    np.testing.assert_array_equal(
        sparse_ranked.read_cells(0, 2).toarray(),
        [[0, 1, 0, 1], [0, 0, 1, 2]],
    )


def test_bpcells_parameterized_numeric_transforms_execute_through_factory() -> None:
    values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    source = DenseMatrixSource(values)

    minimum = matrix_source_from_slots(
        {
            "class": ["TransformMinByRow", "TransformedMatrix"],
            "slots": {
                "matrix": source,
                "row_params": [[2, 5]],
                "dim": source.shape,
                "transpose": False,
            },
        }
    )
    np.testing.assert_array_equal(
        minimum.read_cells(0, 3),
        [[1, 4], [2, 5], [2, 5]],
    )
    column_minimum = matrix_source_from_slots(
        {
            "class": ["TransformMinByCol", "TransformedMatrix"],
            "slots": {
                "matrix": source,
                "col_params": [[2, 4, 10]],
                "dim": source.shape,
                "transpose": False,
            },
        }
    )
    np.testing.assert_array_equal(
        column_minimum.read_cells(0, 3),
        [[1, 2], [2, 4], [3, 6]],
    )

    active = np.ones((3, 2), dtype=bool)
    row_parameters = np.asarray([[2, 3], [1, 2]], dtype=np.float64)
    column_parameters = np.asarray(
        [[1, 10, 100], [0, 5, 10]],
        dtype=np.float64,
    )
    scaled = matrix_source_from_slots(
        {
            "class": ["TransformScaleShift", "TransformedMatrix"],
            "slots": {
                "matrix": source,
                "row_params": row_parameters,
                "col_params": column_parameters,
                "global_params": [0.5, -1],
                "active_transforms": active,
                "dim": source.shape,
                "transpose": False,
            },
        }
    )
    expected_scaled = values.T
    expected_scaled = (
        expected_scaled * row_parameters[0] * column_parameters[0, :, np.newaxis] * 0.5
        + row_parameters[1]
        + column_parameters[1, :, np.newaxis]
        - 1
    )
    np.testing.assert_array_equal(scaled.read_cells(0, 3), expected_scaled)

    theta_inverse = np.asarray([0.1, 0.2])
    gene_beta = np.asarray([0.5, 1.0])
    cell_reads = np.asarray([2.0, 3.0, 4.0])
    mu = cell_reads[:, np.newaxis] * gene_beta
    expected_pearson = (values.T - mu) / np.sqrt(mu + mu * mu * theta_inverse)
    for class_name in ("SCTransformPearson", "SCTransformPearsonSlow"):
        pearson = matrix_source_from_slots(
            {
                "class": [class_name, "TransformedMatrix"],
                "slots": {
                    "matrix": source,
                    "row_params": np.vstack([theta_inverse, gene_beta]),
                    "col_params": cell_reads[np.newaxis, :],
                    "global_params": [np.inf, -10, 10],
                    "dim": source.shape,
                    "transpose": False,
                },
            }
        )
        np.testing.assert_allclose(pearson.read_cells(0, 3), expected_pearson)
    for class_name in (
        "SCTransformPearsonTranspose",
        "SCTransformPearsonTransposeSlow",
    ):
        pearson = matrix_source_from_slots(
            {
                "class": [class_name, "TransformedMatrix"],
                "slots": {
                    "matrix": source,
                    "row_params": cell_reads[np.newaxis, :],
                    "col_params": np.vstack([theta_inverse, gene_beta]),
                    "global_params": [np.inf, -10, 10],
                    "dim": source.shape,
                    "transpose": True,
                },
            }
        )
        np.testing.assert_allclose(pearson.read_cells(0, 3), expected_pearson)

    feature_parameters = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    cell_parameters = np.asarray(
        [[1, 0, 2], [0, 1, 1]],
        dtype=np.float64,
    )
    residual = matrix_source_from_slots(
        {
            "class": ["TransformLinearResidual", "TransformedMatrix"],
            "slots": {
                "matrix": source,
                "row_params": feature_parameters,
                "col_params": cell_parameters,
                "dim": source.shape,
                "transpose": False,
            },
        }
    )
    np.testing.assert_array_equal(
        residual.read_cells(1, 3),
        values.T[1:3] - cell_parameters[:, 1:3].T @ feature_parameters,
    )

    rounded = matrix_source_from_slots(
        {
            "class": ["TransformRound", "TransformedMatrix"],
            "slots": {
                "matrix": DenseMatrixSource(np.asarray([[1.24, 1.26], [2.55, 2.54]])),
                "global_params": [1],
                "dim": (2, 2),
            },
        }
    )
    np.testing.assert_array_equal(
        rounded.read_cells(0, 2),
        [[1.2, 2.6], [1.3, 2.5]],
    )


def test_matrix_multiply_streams_output_cells_and_inner_tiles() -> None:
    left = DenseMatrixSource(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        row_names=("g1", "g2"),
        column_names=("k1", "k2", "k3"),
    )
    right = DenseMatrixSource(
        np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
        row_names=("k1", "k2", "k3"),
        column_names=("c1", "c2"),
    )
    source = MatrixMultiplySource(
        left,
        right=right,
        limits=SourceLimits(tileCells=2),
    )

    assert source.shape == (2, 2)
    assert source.row_names == ("g1", "g2")
    assert source.column_names == ("c1", "c2")
    np.testing.assert_array_equal(
        source.read_cells(0, 2),
        np.array([[58, 139], [64, 154]], dtype=np.float32),
    )
    np.testing.assert_array_equal(source.read_cells(1, 2), [[64, 154]])

    built = build_matrix_operation(
        {
            "operation": "multiply",
            "className": "MatrixMultiply",
            "right": right,
        },
        source=left,
    )
    assert isinstance(built, MatrixMultiplySource)
    np.testing.assert_array_equal(built.read_cells(0, 1), [[58, 139]])


def test_matrix_multiply_validates_contract_and_memory_limit() -> None:
    left = DenseMatrixSource(np.ones((3, 4), dtype=np.float64))
    right = DenseMatrixSource(np.ones((5, 2), dtype=np.float64))
    with pytest.raises(MatrixSourceError, match="inner dimensions"):
        MatrixMultiplySource(left, right=right)
    with pytest.raises(MatrixSourceError, match="requires a right operand"):
        MatrixMultiplySource(left)

    compatible = DenseMatrixSource(np.ones((4, 2), dtype=np.float64))
    limited = MatrixMultiplySource(
        left,
        right=compatible,
        limits=SourceLimits(maxBlockBytes=64, tileCells=1),
    )
    with pytest.raises(ResourceLimitError, match="maxBlockBytes"):
        limited.read_cells(0, 2)


def test_bp128_reference_decode_crosses_blocks_and_supports_windows() -> None:
    values = np.asarray([(index * 17) % 257 for index in range(259)], dtype=np.uint32)
    data, indexes, offsets, _ = _encode_bp128(values, "plain")
    decoded = decode_bp128(
        data,
        indexes,
        values.size,
        index_offsets=offsets,
    )
    np.testing.assert_array_equal(decoded, values)
    np.testing.assert_array_equal(
        decode_bp128(
            data,
            indexes,
            values.size,
            index_offsets=offsets,
            start=121,
            stop=143,
        ),
        values[121:143],
    )

    positive = values + 1
    m1_data, m1_indexes, m1_offsets, _ = _encode_bp128(positive, "m1")
    np.testing.assert_array_equal(
        decode_bp128_m1(
            m1_data,
            m1_indexes,
            positive.size,
            index_offsets=m1_offsets,
        ),
        positive,
    )

    d1_values = np.asarray(
        [100 + ((index * 13) % 31) for index in range(259)],
        dtype=np.uint32,
    )
    d1_data, d1_indexes, d1_offsets, starts = _encode_bp128(d1_values, "d1z")
    np.testing.assert_array_equal(
        decode_bp128_d1z(
            d1_data,
            d1_indexes,
            starts,
            d1_values.size,
            index_offsets=d1_offsets,
        ),
        d1_values,
    )


@pytest.mark.parametrize("backend", ["directory", "hdf5"])
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("datatype", ["uint", "float", "double"])
@pytest.mark.parametrize("storage_order", ["col", "row"])
def test_bpcells_all_documented_matrix_leaves(
    tmp_path: Path,
    backend: str,
    version: int,
    packed: bool,
    datatype: str,
    storage_order: str,
) -> None:
    base = np.asarray(
        [[1, 0, 2, 0], [3, 4, 0, 5], [0, 6, 7, 8]],
        dtype={
            "uint": np.uint32,
            "float": np.float32,
            "double": np.float64,
        }[datatype],
    )
    if datatype != "uint":
        base = base / 2
    payload = _bpcells_payload(
        base,
        packed=packed,
        version=version,
        storage_order=storage_order,
    )
    source: MatrixSource
    if backend == "directory":
        path = tmp_path / "matrix"
        _write_bpcells_directory(path, payload, version=version)
        source = BPCellsDirectoryMatrixSource(path)
    else:
        path = tmp_path / "matrix.h5"
        _write_bpcells_hdf5(path, payload, version=version)
        source = BPCellsHDF5MatrixSource(path, group="matrix")

    assert source.shape == base.shape
    assert source.storageOrder == storage_order
    assert source.row_names == ("f0", "f1", "f2")
    assert source.column_names == ("c0", "c1", "c2", "c3")
    np.testing.assert_array_equal(source.read_cells(0, 4).toarray(), base.T)
    np.testing.assert_array_equal(source.read_cells(1, 3).toarray(), base[:, 1:3].T)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("datatype", ["uint", "float", "double"])
@pytest.mark.parametrize("storage_order", ["col", "row"])
def test_bpcells_in_r_memory_leaves_execute_through_factory(
    version: int,
    packed: bool,
    datatype: str,
    storage_order: str,
) -> None:
    expected = np.asarray(
        [[1, 0, 2, 0], [3, 4, 0, 5], [0, 6, 7, 8]],
        dtype={
            "uint": np.uint32,
            "float": np.float32,
            "double": np.float64,
        }[datatype],
    )
    if datatype != "uint":
        expected = expected / 2
    payload = _bpcells_payload(
        expected,
        packed=packed,
        version=version,
        storage_order=storage_order,
    )
    source = matrix_source_from_slots(
        _memory_matrix_spec(
            payload,
            packed=packed,
            datatype=datatype,
            version=version,
        )
    )
    np.testing.assert_array_equal(source.read_cells(0, 4).toarray(), expected.T)


def test_bpcells_directory_validates_numeric_header(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "version").write_text("unpacked-uint-matrix-v1\n")
    (directory / "shape").write_bytes(b"NOTATYPE" + b"\x00" * 8)
    with pytest.raises(MatrixSourceError, match="unknown 8-byte header"):
        BPCellsDirectoryMatrixSource(directory)


def test_sidecar_sources_enforce_shape_and_nnz_limits(tmp_path: Path) -> None:
    values = np.asarray([[1, 2], [3, 4]], dtype=np.uint32)
    payload = _bpcells_payload(
        values,
        packed=False,
        version=2,
        storage_order="col",
    )
    directory = tmp_path / "limited-matrix"
    _write_bpcells_directory(directory, payload, version=2)
    with pytest.raises(ResourceLimitError, match="maxNnz=2"):
        BPCellsDirectoryMatrixSource(
            directory,
            limits=SourceLimits(maxNnz=2),
        )

    path = tmp_path / "limited-dense.h5"
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("counts", data=values.T)
    with pytest.raises(ResourceLimitError, match="maxFeatures=1"):
        HDF5ArrayMatrixSource(
            path,
            "counts",
            limits=SourceLimits(maxFeatures=1),
        )

    metadata_path = tmp_path / "limited-metadata.h5"
    with h5py.File(metadata_path, mode="w") as handle:
        handle.attrs["large"] = "x" * 100
    with pytest.raises(ResourceLimitError, match="maxMetadataBytes=32"):
        validate_hdf5_file(
            metadata_path,
            limits=SourceLimits(maxMetadataBytes=32),
        )


def test_hdf5array_dense_orientation_and_metadata(tmp_path: Path) -> None:
    feature_by_cell = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    path = tmp_path / "dense.h5"
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("counts", data=feature_by_cell.T)
        handle.create_dataset("features", data=np.asarray(["f1", "f2"], dtype="S"))
        handle.create_dataset("cells", data=np.asarray(["c1", "c2", "c3"], dtype="S"))
    source = HDF5ArrayMatrixSource(
        path,
        "counts",
        row_names_path="/features",
        column_names_path="/cells",
    )
    assert source.shape == (2, 3)
    np.testing.assert_array_equal(source.read_cells(1, 3), [[2, 5], [3, 6]])


def test_hdf5_sources_reject_missing_nodes_and_nonnumeric_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-structure.h5"
    with h5py.File(path, mode="w") as handle:
        counts = handle.create_group("counts")
        counts.attrs["shape"] = np.asarray([2, 2], dtype=np.int64)
        counts.attrs["encoding-type"] = "csr_matrix"
        handle.create_group("matrix")
        handle.create_dataset("labels", data=np.asarray([["a"]], dtype="S1"))

    with pytest.raises(MatrixSourceError, match="'/counts' is not a dataset"):
        HDF5ArrayMatrixSource(path, "counts")
    with pytest.raises(MatrixSourceError, match="/counts/data is missing"):
        H5SparseMatrixSource(path, "counts")
    with pytest.raises(MatrixSourceError, match="H5AD matrix path '/X' is missing"):
        H5ADMatrixSource(path)
    with pytest.raises(MatrixSourceError, match="has no shape dataset"):
        TENxMatrixSource(path)
    with pytest.raises(TypeError, match="nonnumeric dtype"):
        HDF5ArrayMatrixSource(path, "labels")


def test_reshaped_hdf5array_and_facade_execute_through_factory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reshaped.h5"
    physical = np.arange(12, dtype=np.int16).reshape(2, 3, 2)
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("counts", data=physical)

    seed = {
        "class": "ReshapedHDF5ArraySeed",
        "slots": {
            "filepath": path,
            "name": "counts",
            "dim": [2, 3, 2],
            "reshaped_dim": [3, 4],
        },
    }
    source = matrix_source_from_slots(seed)
    np.testing.assert_array_equal(
        source.read_cells(1, 3),
        physical.reshape(-1)[3:9].reshape(2, 3),
    )

    facade = matrix_source_from_slots(
        {
            "class": ["HDF5Matrix", "DelayedMatrix", "DelayedArray"],
            "slots": {
                "seed": seed,
                "dim": [3, 4],
            },
        }
    )
    np.testing.assert_array_equal(
        facade.read_cells(0, 4),
        physical.reshape(-1).reshape(4, 3),
    )


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_h5_sparse_matrix_physical_transpose(
    layout: str,
    tmp_path: Path,
) -> None:
    logical = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.float32)
    physical = logical.T
    path = tmp_path / f"sparse-{layout}.h5"
    with h5py.File(path, mode="w") as handle:
        _write_h5_sparse_group(handle, "counts", physical, layout)
    source = H5SparseMatrixSource(
        path,
        "counts",
        row_names=["f1", "f2"],
        column_names=["c1", "c2", "c3"],
    )
    np.testing.assert_array_equal(source.read_cells(0, 3).toarray(), logical.T)


@pytest.mark.parametrize("layout", ["dense", "csr", "csc"])
def test_h5ad_dense_and_sparse_sources(layout: str, tmp_path: Path) -> None:
    logical = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.float64)
    physical = logical.T
    path = tmp_path / f"matrix-{layout}.h5ad"
    with h5py.File(path, mode="w") as handle:
        if layout == "dense":
            handle.create_dataset("X", data=physical)
        else:
            _write_h5_sparse_group(handle, "X", physical, layout)
        _write_axis_names(handle, ["f1", "f2"], ["c1", "c2", "c3"])
    source = H5ADMatrixSource(path)
    assert source.row_names == ("f1", "f2")
    assert source.column_names == ("c1", "c2", "c3")
    np.testing.assert_array_equal(
        source.read_cells(0, 3).toarray()
        if source.is_sparse
        else source.read_cells(0, 3),
        logical.T,
    )


def test_tenx_source_orientation_and_names(tmp_path: Path) -> None:
    logical = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.uint32)
    matrix = csc_matrix(logical)
    path = tmp_path / "tenx.h5"
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("shape", data=np.asarray(logical.shape, dtype=np.uint64))
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset("barcodes", data=np.asarray(["c1", "c2", "c3"], dtype="S"))
        features = group.create_group("features")
        features.create_dataset("name", data=np.asarray(["f1", "f2"], dtype="S"))
    source = TENxMatrixSource(path)
    np.testing.assert_array_equal(source.read_cells(0, 3).toarray(), logical.T)
    assert source.row_names == ("f1", "f2")
    assert source.column_names == ("c1", "c2", "c3")


def test_sidecar_path_resolution_anchors_remaps_and_contains(
    tmp_path: Path,
) -> None:
    rds = tmp_path / "object.rds"
    rds.write_bytes(b"rds")
    relative = tmp_path / "sidecar.h5"
    relative.write_bytes(b"h5")
    resolver = SidecarPathResolver(rds)
    assert resolver.resolve("sidecar.h5", expect="file") == relative.resolve()

    outside = tmp_path.parent / "outside-sidecar.h5"
    outside.write_bytes(b"h5")
    with pytest.raises(UnsafeSidecarError, match="escapes allowed root"):
        resolver.resolve("../outside-sidecar.h5")
    with pytest.raises(UnsafeSidecarError, match="escapes allowed root"):
        resolver.resolve(r"..\outside-sidecar.h5")
    with pytest.raises(UnsafeSidecarError, match="no explicit prefix remap"):
        resolver.resolve(outside)

    remapped_root = tmp_path / "remapped"
    remapped_root.mkdir()
    remapped = remapped_root / "counts.h5"
    remapped.write_bytes(b"h5")
    remapping = SidecarPathResolver(
        rds,
        absolute_prefix_remaps={"/original/data": remapped_root},
    )
    assert remapping.resolve("/original/data/counts.h5") == remapped.resolve()

    windows_remapping = SidecarPathResolver(
        rds,
        absolute_prefix_remaps={r"C:\original\data": remapped_root},
    )
    assert (
        windows_remapping.resolve(r"C:\original\data\counts.h5") == remapped.resolve()
    )


def test_path_resolution_rejects_symlink_escape(tmp_path: Path) -> None:
    rds = tmp_path / "object.rds"
    rds.write_bytes(b"rds")
    outside = tmp_path.parent / "outside-target.h5"
    outside.write_bytes(b"h5")
    link = tmp_path / "linked.h5"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available")
    with pytest.raises(UnsafeSidecarError, match="escapes allowed root"):
        SidecarPathResolver(rds).resolve("linked.h5")


def test_path_resolution_rejects_nul_and_invalid_absolute_remaps(
    tmp_path: Path,
) -> None:
    rds = tmp_path / "object.rds"
    rds.write_bytes(b"rds")

    with pytest.raises(UnsafeSidecarError, match="NUL character"):
        SidecarPathResolver(rds).resolve("bad\x00name.h5")
    with pytest.raises(
        ValueError, match="absolute path remaps require absolute prefixes"
    ):
        SidecarPathResolver(
            rds,
            absolute_prefix_remaps={"relative/prefix": tmp_path / "dest"},
        )
    with pytest.raises(
        ValueError, match="absolute path remaps require absolute prefixes"
    ):
        SidecarPathResolver(
            rds,
            absolute_prefix_remaps={"/absolute/source": "relative/dest"},
        )


def test_hdf5_rejects_external_links_and_live_handles(tmp_path: Path) -> None:
    target = tmp_path / "target.h5"
    with h5py.File(target, mode="w") as handle:
        handle.create_dataset("data", data=np.arange(3))
    linked = tmp_path / "linked.h5"
    with h5py.File(linked, mode="w") as handle:
        handle["external"] = h5py.ExternalLink(str(target), "/data")

    with pytest.raises(UnsafeSidecarError, match="external link"):
        validate_hdf5_file(linked)

    with h5py.File(target, mode="r") as handle:
        with pytest.raises(UnsafeSidecarError, match="live HDF5 handles"):
            HDF5ArrayMatrixSource(handle, "data")


def test_hdf5_rejects_external_storage_virtual_and_reference_data(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "external.raw"
    external = tmp_path / "external-storage.h5"
    with h5py.File(external, mode="w") as handle:
        handle.create_dataset(
            "data",
            shape=(2,),
            dtype=np.int32,
            external=[(str(raw), 0, h5py.h5f.UNLIMITED)],
        )
    with pytest.raises(UnsafeSidecarError, match="external storage"):
        validate_hdf5_file(external)

    source_path = tmp_path / "vds-source.h5"
    with h5py.File(source_path, mode="w") as handle:
        handle.create_dataset("data", data=np.arange(3))
    virtual_path = tmp_path / "virtual.h5"
    layout = h5py.VirtualLayout(shape=(3,), dtype=np.int64)
    layout[:] = h5py.VirtualSource(str(source_path), "data", shape=(3,))
    with h5py.File(virtual_path, mode="w") as handle:
        handle.create_virtual_dataset("data", layout)
    with pytest.raises(UnsafeSidecarError, match="virtual dataset"):
        validate_hdf5_file(virtual_path)

    references = tmp_path / "references.h5"
    with h5py.File(references, mode="w") as handle:
        target = handle.create_dataset("target", data=np.arange(1))
        dataset = handle.create_dataset("refs", shape=(1,), dtype=h5py.ref_dtype)
        dataset[0] = target.ref
    with pytest.raises(UnsafeSidecarError, match="reference dataset"):
        validate_hdf5_file(references)


def test_hdf5_rejects_unknown_required_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unknown-filter.h5"
    filter_id = h5py.h5z.FILTER_DEFLATE
    with h5py.File(path, mode="w") as handle:
        space = h5py.h5s.create_simple((1,))
        properties = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        properties.set_chunk((1,))
        properties.set_filter(filter_id, 0, (1,))
        h5py.h5d.create(
            handle.id,
            b"data",
            h5py.h5t.NATIVE_INT32,
            space,
            dcpl=properties,
        )
    available = h5py.h5z.filter_avail
    monkeypatch.setattr(
        h5py.h5z,
        "filter_avail",
        lambda candidate: False if candidate == filter_id else available(candidate),
    )
    with pytest.raises(UnsafeSidecarError, match="requires unavailable filter"):
        validate_hdf5_file(path)


def test_bp128_decode_validates_windows_metadata_and_numeric_range() -> None:
    empty = np.empty(0, dtype=np.uint32)
    zero_index = np.asarray([0, 0], dtype=np.uint32)
    np.testing.assert_array_equal(decode_bp128(empty, zero_index, 1), [0])
    assert decode_bp128(empty, zero_index, 1, start=0, stop=0).size == 0

    with pytest.raises(ValueError, match="count cannot be negative"):
        decode_bp128(empty, np.asarray([0], dtype=np.uint32), -1)
    with pytest.raises(IndexError, match=r"window \[-1, 0\)"):
        decode_bp128(empty, zero_index, 1, start=-1, stop=0)
    with pytest.raises(TypeError, match="idx must be a one-dimensional integer"):
        decode_bp128(empty, np.asarray([[0, 0]], dtype=np.uint32), 1)
    with pytest.raises(TypeError, match="idx_offsets must be"):
        decode_bp128(
            empty,
            zero_index,
            1,
            index_offsets=np.asarray([0.0, 2.0]),
        )
    with pytest.raises(MatrixSourceError, match="partition the complete idx"):
        decode_bp128(
            empty,
            zero_index,
            1,
            index_offsets=np.asarray([0, 1], dtype=np.uint64),
        )
    with pytest.raises(MatrixSourceError, match="idx must start at zero"):
        decode_bp128(
            np.zeros(4, dtype=np.uint32),
            np.asarray([1, 4], dtype=np.uint32),
            1,
        )
    with pytest.raises(MatrixSourceError, match="must be nondecreasing"):
        decode_bp128(
            empty,
            np.asarray([0, 4, 3], dtype=np.uint32),
            129,
        )
    with pytest.raises(MatrixSourceError, match="idx has length"):
        decode_bp128(empty, np.asarray([0], dtype=np.uint32), 1)
    with pytest.raises(TypeError, match="data must be a one-dimensional integer"):
        decode_bp128(np.empty((0, 1), dtype=np.uint32), zero_index, 1)
    with pytest.raises(MatrixSourceError, match="data has 0 words"):
        decode_bp128(
            empty,
            np.asarray([0, 4], dtype=np.uint32),
            1,
        )
    with pytest.raises(ValueError, match="unknown BP128 transform"):
        decode_bp128(empty, zero_index, 1, transform="delta")
    with pytest.raises(MatrixSourceError, match="requires starts"):
        decode_bp128(empty, zero_index, 1, transform="d1")
    with pytest.raises(MatrixSourceError, match="starts must have length 1"):
        decode_bp128(
            empty,
            zero_index,
            1,
            transform="d1",
            starts=np.asarray([0, 1], dtype=np.uint32),
        )
    with pytest.raises(MatrixSourceError, match="not divisible by four"):
        decode_bp128(
            np.zeros(1, dtype=np.uint32),
            np.asarray([0, 1], dtype=np.uint32),
            1,
        )
    with pytest.raises(MatrixSourceError, match="bit width 33"):
        decode_bp128(
            np.zeros(132, dtype=np.uint32),
            np.asarray([0, 132], dtype=np.uint32),
            1,
        )

    maximum_block = _pack_bp128_block(
        np.full(128, np.iinfo(np.uint32).max, dtype=np.uint32)
    )
    with pytest.raises(MatrixSourceError, match="m1 decode overflows"):
        decode_bp128_m1(
            maximum_block,
            np.asarray([0, maximum_block.size], dtype=np.uint32),
            1,
        )

    delta_values = np.arange(10, 140, dtype=np.uint32)
    data, indexes, offsets, starts = _encode_bp128(delta_values, "d1")
    np.testing.assert_array_equal(
        decode_bp128(
            data,
            indexes,
            delta_values.size,
            index_offsets=offsets,
            transform="d1",
            starts=starts,
        ),
        delta_values,
    )
    overflow_block = _pack_bp128_block(
        np.concatenate(
            (
                np.asarray([1], dtype=np.uint32),
                np.zeros(127, dtype=np.uint32),
            )
        )
    )
    with pytest.raises(MatrixSourceError, match="d1 decode leaves uint32 range"):
        decode_bp128(
            overflow_block,
            np.asarray([0, overflow_block.size], dtype=np.uint32),
            1,
            transform="d1",
            starts=np.asarray([np.iinfo(np.uint32).max], dtype=np.uint32),
        )


def test_bpcells_memory_recipes_validate_arrays_names_and_storage() -> None:
    values = np.asarray([[1, 0], [0, 2]], dtype=np.uint32)
    payload = _bpcells_payload(
        values,
        packed=False,
        version=2,
        storage_order="col",
    )
    arrays = {
        name: value
        for name, value in payload.items()
        if name not in {"version", "storage_order", "row_names", "col_names"}
    }

    with pytest.raises(MatrixSourceError, match="unsupported BPCells matrix format"):
        BPCellsMemoryMatrixSource(
            "unpacked-uint-matrix-v3",
            arrays,
            shape=values.shape,
            storage_order="col",
        )
    with pytest.raises(MatrixSourceError, match="storage order must be row or col"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            arrays,
            shape=values.shape,
            storage_order="diagonal",
        )
    with pytest.raises(MatrixSourceError, match="missing arrays"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            {},
            shape=values.shape,
            storage_order="col",
        )

    invalid_pointers = {
        **arrays,
        "idxptr": np.asarray([0.0, np.nan, 2.0]),
    }
    with pytest.raises(MatrixSourceError, match="contains invalid integer values"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            invalid_pointers,
            shape=values.shape,
            storage_order="col",
        )
    invalid_indexes = {
        **arrays,
        "index": np.asarray(["zero", "one"]),
    }
    with pytest.raises(TypeError, match="must be numeric"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            invalid_indexes,
            shape=values.shape,
            storage_order="col",
        )
    with pytest.raises(MatrixSourceError, match="not valid UTF-8"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            arrays,
            shape=values.shape,
            storage_order="col",
            row_names=[b"\xff", b"f2"],
        )
    with pytest.raises(TypeError, match="must contain strings"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            arrays,
            shape=values.shape,
            storage_order="col",
            row_names=[1, 2],
        )
    with pytest.raises(ResourceLimitError, match="maxMetadataBytes=12"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            arrays,
            shape=values.shape,
            storage_order="col",
            row_names=["longer", "names"],
            limits=SourceLimits(maxMetadataBytes=12),
        )

    unnamed = BPCellsMemoryMatrixSource(
        payload["version"],
        arrays,
        shape=values.shape,
        storage_order="col",
        row_names=[],
    )
    assert unnamed.row_names is None
    with pytest.raises(MatrixSourceError, match="row_names has length 1"):
        BPCellsMemoryMatrixSource(
            payload["version"],
            arrays,
            shape=values.shape,
            storage_order="col",
            row_names=["only-one"],
        )

    float_values = np.asarray([[1.5]], dtype=np.float32)
    float_payload = _bpcells_payload(
        float_values,
        packed=False,
        version=2,
        storage_order="col",
    )
    float_arrays = {
        name: value
        for name, value in float_payload.items()
        if name not in {"version", "storage_order", "row_names", "col_names"}
    }
    malformed_float = BPCellsMemoryMatrixSource(
        float_payload["version"],
        float_arrays,
        shape=float_values.shape,
        storage_order="col",
        float_bit_arrays=frozenset({"val"}),
    )
    with pytest.raises(TypeError, match="floating-point bit patterns"):
        malformed_float.read_cells(0, 1)


def test_bpcells_directory_recipes_reject_malformed_local_layouts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        BPCellsDirectoryMatrixSource(missing)

    regular_file = tmp_path / "matrix-file"
    regular_file.write_bytes(b"not-a-directory")
    with pytest.raises(UnsafeSidecarError, match="is not a directory"):
        BPCellsDirectoryMatrixSource(regular_file)

    repeated_version = tmp_path / "repeated-version"
    repeated_version.mkdir()
    (repeated_version / "version").write_text(
        "unpacked-uint-matrix-v2\nunpacked-uint-matrix-v2\n"
    )
    with pytest.raises(MatrixSourceError, match="exactly one line"):
        BPCellsDirectoryMatrixSource(repeated_version)

    missing_shape = tmp_path / "missing-shape"
    missing_shape.mkdir()
    (missing_shape / "version").write_text("unpacked-uint-matrix-v2\n")
    with pytest.raises(MatrixSourceError, match="numeric array 'shape' is missing"):
        BPCellsDirectoryMatrixSource(missing_shape)

    shape_directory = tmp_path / "shape-directory"
    shape_directory.mkdir()
    (shape_directory / "version").write_text("unpacked-uint-matrix-v2\n")
    (shape_directory / "shape").mkdir()
    with pytest.raises(UnsafeSidecarError, match="'shape' is not a regular file"):
        BPCellsDirectoryMatrixSource(shape_directory)

    truncated_shape = tmp_path / "truncated-shape"
    truncated_shape.mkdir()
    (truncated_shape / "version").write_text("unpacked-uint-matrix-v2\n")
    (truncated_shape / "shape").write_bytes(b"UINT32v1\x00")
    with pytest.raises(MatrixSourceError, match="truncated payload"):
        BPCellsDirectoryMatrixSource(truncated_shape)

    values = np.asarray([[1, 0], [0, 2]], dtype=np.uint32)
    payload = _bpcells_payload(
        values,
        packed=False,
        version=2,
        storage_order="col",
    )

    missing_order = tmp_path / "missing-order"
    _write_bpcells_directory(missing_order, payload, version=2)
    (missing_order / "storage_order").unlink()
    with pytest.raises(
        MatrixSourceError, match="text array 'storage_order' is missing"
    ):
        BPCellsDirectoryMatrixSource(missing_order)

    order_directory = tmp_path / "order-directory"
    _write_bpcells_directory(order_directory, payload, version=2)
    (order_directory / "storage_order").unlink()
    (order_directory / "storage_order").mkdir()
    with pytest.raises(
        UnsafeSidecarError, match="'storage_order' is not a regular file"
    ):
        BPCellsDirectoryMatrixSource(order_directory)

    oversized_order = tmp_path / "oversized-order"
    _write_bpcells_directory(oversized_order, payload, version=2)
    (oversized_order / "storage_order").write_text("x" * 100)
    with pytest.raises(ResourceLimitError, match="maxMetadataBytes=40"):
        BPCellsDirectoryMatrixSource(
            oversized_order,
            limits=SourceLimits(maxMetadataBytes=40),
        )

    invalid_utf8 = tmp_path / "invalid-utf8"
    _write_bpcells_directory(invalid_utf8, payload, version=2)
    (invalid_utf8 / "storage_order").write_bytes(b"\xff")
    with pytest.raises(MatrixSourceError, match="not valid UTF-8"):
        BPCellsDirectoryMatrixSource(invalid_utf8)


def test_hdf5_dense_reshape_delegation_and_handle_ownership(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dense-recipes.h5"
    physical = np.arange(12, dtype=np.int16).reshape(2, 3, 2)
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("matrix", data=np.arange(6).reshape(2, 3))
        handle.create_dataset("vector", data=np.arange(3))
        handle.create_dataset("physical", data=physical)
        handle.create_dataset("scalar", data=np.asarray(1))
        handle.create_dataset("labels", data=np.asarray(["a", "b"], dtype="S"))
        handle.create_dataset("wrong-size", data=np.arange(5))
        handle.create_group("group")

    open_files_before = h5py.h5f.get_obj_count(
        h5py.h5f.OBJ_ALL,
        h5py.h5f.OBJ_FILE,
    )
    dense = HDF5ArrayMatrixSource(
        path,
        "matrix",
        r_transposed=False,
        dtype=np.float32,
        as_sparse=True,
    )
    assert dense.shape == (2, 3)
    assert dense.read_cells(1, 3).dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        dense.read_cells(1, 3).toarray(),
        [[1, 4], [2, 5]],
    )

    reshaped = ReshapedHDF5ArrayMatrixSource(
        path,
        "physical",
        (3, 4),
        dtype=np.float32,
        as_sparse=True,
    )
    np.testing.assert_array_equal(
        reshaped.read_cells(1, 3).toarray(),
        physical.reshape(-1)[3:9].reshape(2, 3),
    )
    assert (
        h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE) == open_files_before
    )

    with pytest.raises(MatrixSourceError, match="'/missing' is missing"):
        HDF5ArrayMatrixSource(path, "missing")
    with pytest.raises(MatrixSourceError, match="must be two-dimensional"):
        HDF5ArrayMatrixSource(path, "vector")
    with pytest.raises(MatrixSourceError, match="cannot be reshaped from a scalar"):
        ReshapedHDF5ArrayMatrixSource(path, "scalar", (1, 1))
    with pytest.raises(TypeError, match="nonnumeric dtype"):
        ReshapedHDF5ArrayMatrixSource(path, "labels", (1, 2))
    with pytest.raises(MatrixSourceError, match="reshaped matrix requires 6"):
        ReshapedHDF5ArrayMatrixSource(path, "wrong-size", (2, 3))
    with pytest.raises(MatrixSourceError, match="'/missing' is missing"):
        ReshapedHDF5ArrayMatrixSource(path, "missing", (1, 1))
    with pytest.raises(MatrixSourceError, match="'/group' is not a dataset"):
        ReshapedHDF5ArrayMatrixSource(path, "group", (1, 1))

    h5ad_path = tmp_path / "delegating.h5ad"
    logical = np.asarray([[1, 0, 2], [0, 3, 4]], dtype=np.float64)
    with h5py.File(h5ad_path, mode="w") as handle:
        handle.create_dataset("X", data=logical.T)
        _write_axis_names(handle, ["f1", "f2"], ["c1", "c2", "c3"])
    delegated = H5ADMatrixSource(h5ad_path)
    assert delegated.shape == (2, 3)
    assert delegated.dtype == np.dtype(np.float64)
    assert delegated.rowNames == delegated.row_names == ("f1", "f2")
    assert delegated.columnNames == delegated.column_names == ("c1", "c2", "c3")
    assert delegated.n_features == 2
    assert delegated.n_cells == 3
    assert delegated.sparse is delegated.is_sparse is False
    assert delegated.zeroPreserving is delegated.zero_preserving is True
    assert delegated.residentBytes == delegated.resident_bytes
    estimate = delegated.memory_estimate(0, 1)
    assert delegated.estimate_memory(0, 1) == estimate
    assert delegated.estimate_read_bytes(0, 1) == estimate.peakBytes
    assert delegated.estimated_peak_bytes(0, 1) == estimate.peakBytes


def test_hdf5_compressed_sources_validate_structure_and_infer_layout(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid-sparse.h5"
    with h5py.File(valid_path, mode="w") as handle:
        _write_h5_sparse_group(
            handle,
            "matrix",
            np.asarray([[1, 0], [0, 2]]),
            "csr",
        )

    with pytest.raises(MatrixSourceError, match="physical_layout"):
        HDF5CompressedMatrixSource(
            valid_path,
            "matrix",
            physical_shape=(2, 2),
            physical_layout="coo",
            physical_order="cell_by_feature",
        )
    with pytest.raises(MatrixSourceError, match="physical_order"):
        HDF5CompressedMatrixSource(
            valid_path,
            "matrix",
            physical_shape=(2, 2),
            physical_layout="csr",
            physical_order="row-major",
        )
    with pytest.raises(MatrixSourceError, match="shape must have length two"):
        HDF5CompressedMatrixSource(
            valid_path,
            "matrix",
            physical_shape=(2,),
            physical_layout="csr",
            physical_order="cell_by_feature",
        )
    with pytest.raises(MatrixSourceError, match="shape cannot be negative"):
        HDF5CompressedMatrixSource(
            valid_path,
            "matrix",
            physical_shape=(-1, 2),
            physical_layout="csr",
            physical_order="cell_by_feature",
        )

    def write_arrays(
        name: str,
        data: Any,
        indices: Any,
        indptr: Any,
    ) -> Path:
        target = tmp_path / f"{name}.h5"
        with h5py.File(target, mode="w") as handle:
            group = handle.create_group("matrix")
            group.create_dataset("data", data=data)
            group.create_dataset("indices", data=indices)
            group.create_dataset("indptr", data=indptr)
        return target

    malformed_cases = (
        (
            "two-dimensional",
            np.asarray([[1]]),
            np.asarray([0]),
            np.asarray([0, 1, 1]),
            MatrixSourceError,
            "one-dimensional",
            SourceLimits(),
        ),
        (
            "text-data",
            np.asarray(["x"], dtype="S"),
            np.asarray([0]),
            np.asarray([0, 1, 1]),
            TypeError,
            "numeric dtype",
            SourceLimits(),
        ),
        (
            "float-indices",
            np.asarray([1]),
            np.asarray([0.0]),
            np.asarray([0, 1, 1]),
            TypeError,
            "indices must contain integers",
            SourceLimits(),
        ),
        (
            "float-indptr",
            np.asarray([1]),
            np.asarray([0]),
            np.asarray([0.0, 1.0, 1.0]),
            TypeError,
            "indptr must contain integers",
            SourceLimits(),
        ),
        (
            "wrong-indptr-shape",
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([0, 0]),
            MatrixSourceError,
            "indptr has shape",
            SourceLimits(),
        ),
        (
            "decreasing-indptr",
            np.asarray([1]),
            np.asarray([0]),
            np.asarray([0, 1, 0]),
            MatrixSourceError,
            "must be nondecreasing",
            SourceLimits(),
        ),
        (
            "nonzero-start",
            np.asarray([1]),
            np.asarray([0]),
            np.asarray([1, 1, 1]),
            MatrixSourceError,
            "must start at zero",
            SourceLimits(),
        ),
        (
            "nnz-limit",
            np.asarray([1, 2]),
            np.asarray([0, 1]),
            np.asarray([0, 1, 2]),
            ResourceLimitError,
            "maxNnz=1",
            SourceLimits(maxNnz=1),
        ),
        (
            "length-mismatch",
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([0, 1, 1]),
            MatrixSourceError,
            "lengths are inconsistent",
            SourceLimits(),
        ),
        (
            "index-out-of-range",
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([0, 1, 1]),
            MatrixSourceError,
            "out-of-range value",
            SourceLimits(),
        ),
    )
    for (
        name,
        data,
        indices,
        indptr,
        error_type,
        match,
        limits,
    ) in malformed_cases:
        malformed_path = write_arrays(name, data, indices, indptr)
        with pytest.raises(error_type, match=match):
            HDF5CompressedMatrixSource(
                malformed_path,
                "matrix",
                physical_shape=(2, 2),
                physical_layout="csr",
                physical_order="cell_by_feature",
                limits=limits,
            )

    def write_inferred(
        name: str,
        shape: tuple[int, int],
        data: Sequence[int],
        indices: Sequence[int],
        indptr: Sequence[int],
    ) -> Path:
        target = tmp_path / f"{name}.h5"
        with h5py.File(target, mode="w") as handle:
            group = handle.create_group("matrix")
            group.attrs["shape"] = shape
            group.create_dataset("data", data=np.asarray(data))
            group.create_dataset("indices", data=np.asarray(indices))
            group.create_dataset("indptr", data=np.asarray(indptr))
        return target

    csr_path = write_inferred(
        "inferred-csr",
        (3, 2),
        [1, 2],
        [0, 1],
        [0, 1, 1, 2],
    )
    inferred_csr = H5SparseMatrixSource(csr_path, "matrix")
    assert inferred_csr.shape == (2, 3)

    csc_path = write_inferred(
        "inferred-csc",
        (3, 2),
        [1, 2],
        [0, 0],
        [0, 1, 2],
    )
    inferred_csc = H5SparseMatrixSource(csc_path, "matrix")
    assert inferred_csc.read_cells(1, 1).shape == (0, 2)
    assert inferred_csc.read_cells(1, 2).nnz == 0

    ambiguous_path = write_inferred(
        "ambiguous",
        (3, 2),
        [],
        [],
        [0, 0],
    )
    with pytest.raises(MatrixSourceError, match="cannot infer"):
        H5SparseMatrixSource(ambiguous_path, "matrix")
    with pytest.raises(MatrixSourceError, match="logical shape must have length two"):
        H5SparseMatrixSource(csr_path, "matrix", shape=(2,))
    with pytest.raises(MatrixSourceError, match="stored shape.*conflicts"):
        H5SparseMatrixSource(csr_path, "matrix", shape=(2, 2))
    with pytest.raises(MatrixSourceError, match="sparse_layout must describe"):
        H5SparseMatrixSource(csr_path, "matrix", sparse_layout="coo")
    explicitly_csc = H5SparseMatrixSource(csc_path, "matrix", sparse_layout="csr")
    assert explicitly_csc.shape == (2, 3)


def test_paths_validate_local_resolution_hdf5_names_and_shapes(
    tmp_path: Path,
) -> None:
    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"bytes-path"

    with pytest.raises(TypeError, match="paths must be text"):
        SidecarPathResolver(BytesPath())

    rds = tmp_path / "object.rds"
    rds.write_bytes(b"rds")
    resolver = SidecarPathResolver(rds, require_exists=False)
    with pytest.raises(UnsafeSidecarError, match="drive-relative"):
        resolver.resolve(r"C:relative\counts.h5")

    directory = tmp_path / "directory"
    directory.mkdir()
    regular_file = tmp_path / "regular.h5"
    regular_file.write_bytes(b"data")
    with pytest.raises(UnsafeSidecarError, match="not a regular file"):
        resolver.resolve("directory", expect="file")
    with pytest.raises(UnsafeSidecarError, match="not a directory"):
        resolver.resolve("regular.h5", expect="directory")
    with pytest.raises(ValueError, match="expect must be"):
        resolver.resolve("regular.h5", expect="socket")

    unresolved = resolve_sidecar_path(
        "future.h5",
        rds,
        require_exists=False,
    )
    assert unresolved == (tmp_path / "future.h5").resolve()
    with pytest.raises(TypeError, match="must be a filesystem path"):
        require_filesystem_path(object())
    with pytest.raises(FileNotFoundError):
        require_filesystem_path(tmp_path / "absent.h5")
    with pytest.raises(UnsafeSidecarError, match="not a regular file"):
        require_filesystem_path(directory)

    mismatched_remap = SidecarPathResolver(
        rds,
        absolute_prefix_remaps={r"C:\source": tmp_path},
        require_exists=False,
    )
    with pytest.raises(UnsafeSidecarError, match="no explicit prefix remap"):
        mismatched_remap.resolve("/source/counts.h5")

    invalid_hdf5 = tmp_path / "invalid.h5"
    invalid_hdf5.write_bytes(b"plain text")
    with pytest.raises(MatrixSourceError, match="cannot open HDF5 sidecar"):
        validate_hdf5_file(invalid_hdf5)

    metadata_path = tmp_path / "metadata.h5"
    with h5py.File(metadata_path, mode="w") as handle:
        handle.create_group("name-group")
        handle.create_dataset("wrong-shape", data=np.asarray([["a"]], dtype="S"))
        handle.create_dataset("numbers", data=np.asarray([1]))
        handle.create_dataset("invalid-utf8", data=np.asarray([b"\xff"], dtype="S1"))
        handle.create_dataset("nul", data=np.asarray([b"a\x00b"], dtype="S3"))
        handle.create_dataset("fixed-long", data=np.asarray([b"a"], dtype="S64"))
        string_dtype = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "variable-long",
            data=np.asarray(["x" * 64], dtype=string_dtype),
        )
        target = handle.create_dataset("target", data=np.asarray([1]))
        references = handle.create_dataset(
            "references", shape=(1,), dtype=h5py.ref_dtype
        )
        references[0] = target.ref

    with h5py.File(metadata_path, mode="r") as handle:
        assert read_hdf5_names(handle, None, 1) is None
        assert read_hdf5_names(handle, "missing", 1) is None
        with pytest.raises(
            MatrixSourceError, match="names dataset '/missing' is missing"
        ):
            read_hdf5_names(handle, "missing", 1, required=True)
        with pytest.raises(MatrixSourceError, match="is not a dataset"):
            read_hdf5_names(handle, "name-group", 1)
        with pytest.raises(MatrixSourceError, match="has shape"):
            read_hdf5_names(handle, "wrong-shape", 1)
        with pytest.raises(MatrixSourceError, match="must contain strings"):
            read_hdf5_names(handle, "numbers", 1)
        with pytest.raises(MatrixSourceError, match="not valid UTF-8"):
            read_hdf5_names(handle, "invalid-utf8", 1)
        with pytest.raises(MatrixSourceError, match="contains NUL"):
            read_hdf5_names(handle, "nul", 1)
        with pytest.raises(ResourceLimitError, match="maxMetadataBytes=32"):
            read_hdf5_names(
                handle,
                "fixed-long",
                1,
                limits=SourceLimits(maxMetadataBytes=32),
            )
        with pytest.raises(ResourceLimitError, match="maxMetadataBytes=32"):
            read_hdf5_names(
                handle,
                "variable-long",
                1,
                limits=SourceLimits(maxMetadataBytes=32),
            )
        with pytest.raises(UnsafeSidecarError, match="reference names"):
            read_hdf5_names(handle, "references", 1)
        with pytest.raises(MatrixSourceError, match="group '/missing' is missing"):
            require_hdf5_group(handle, "missing")
        with pytest.raises(MatrixSourceError, match="is not a group"):
            require_hdf5_group(handle, "target")
        group = require_hdf5_group(handle, "name-group")
        with pytest.raises(MatrixSourceError, match="/required is missing"):
            require_hdf5_datasets(group, ["required"])

    with pytest.raises(MatrixSourceError, match="must contain two integers"):
        read_hdf5_shape(np.asarray([1]), "/shape")
    with pytest.raises(TypeError, match="must contain integers"):
        read_hdf5_shape(np.asarray([1.0, 2.0]), "/shape")
    with pytest.raises(MatrixSourceError, match="cannot contain negatives"):
        read_hdf5_shape(np.asarray([-1, 2]), "/shape")

    reference_attribute = tmp_path / "reference-attribute.h5"
    with h5py.File(reference_attribute, mode="w") as handle:
        target = handle.create_dataset("target", data=np.asarray([1]))
        handle.attrs["reference"] = target.ref
    with pytest.raises(UnsafeSidecarError, match="references are rejected"):
        validate_hdf5_file(reference_attribute)

    cyclic_links = tmp_path / "cyclic-links.h5"
    with h5py.File(cyclic_links, mode="w") as handle:
        group = handle.create_group("group")
        group["self"] = group
        handle["soft"] = h5py.SoftLink("/group")
    assert validate_hdf5_file(cyclic_links) == cyclic_links.resolve()


def test_operations_validate_recipes_and_execute_sparse_local_paths() -> None:
    values = np.asarray([[1, 0, 3], [0, 4, 2]], dtype=np.float64)
    matrix = csc_matrix(values)
    sparse = CscMatrixSource(
        matrix.data,
        matrix.indices,
        matrix.indptr,
        matrix.shape,
        row_names=["f1", "f2"],
        column_names=["c1", "c2", "c3"],
    )

    rounded = UnaryTransformMatrixSource(sparse, "round")
    np.testing.assert_array_equal(rounded.read_cells(0, 3).toarray(), values.T)
    with pytest.raises(UnsupportedMatrixOperation, match="custom functions"):
        UnaryTransformMatrixSource(sparse, object())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedMatrixOperation, match="unknown unary function"):
        UnaryTransformMatrixSource(sparse, "custom")
    with pytest.raises(UnsupportedMatrixOperation, match="custom functions"):
        BinaryTransformMatrixSource(sparse, 1, object())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedMatrixOperation, match="unknown binary function"):
        BinaryTransformMatrixSource(sparse, 1, "custom")

    conflicting_rows = DenseMatrixSource(
        values,
        row_names=["other", "rows"],
        column_names=["c1", "c2", "c3"],
    )
    with pytest.raises(MatrixSourceError, match="row names conflict"):
        BinaryTransformMatrixSource(sparse, conflicting_rows, "add")
    conflicting_columns = DenseMatrixSource(
        values,
        row_names=["f1", "f2"],
        column_names=["x", "y", "z"],
    )
    with pytest.raises(MatrixSourceError, match="column names conflict"):
        BinaryTransformMatrixSource(sparse, conflicting_columns, "add")
    explicit_dtype = BinaryTransformMatrixSource(
        sparse,
        2,
        "multiply",
        dtype=np.float32,
    )
    assert explicit_dtype.dtype == np.dtype(np.float32)

    mask = DenseMatrixSource(np.asarray([[1, 0, 1], [0, 1, 0]], dtype=bool))
    inverted = MaskMatrixSource(sparse, mask, keep_nonzero=False)
    np.testing.assert_array_equal(
        inverted.read_cells(0, 3).toarray(),
        [[0, 0], [0, 0], [0, 2]],
    )
    with pytest.raises(MatrixSourceError, match="mask shape"):
        MaskMatrixSource(sparse, DenseMatrixSource(np.ones((1, 1))))
    with pytest.raises(TypeError, match="fill_value"):
        MaskMatrixSource(sparse, mask, fill_value=[1])  # type: ignore[arg-type]

    feature_minimum = AxisMinimumMatrixSource(
        sparse,
        [2, 3],
        axis="feature",
    )
    np.testing.assert_array_equal(
        feature_minimum.read_cells(0, 3).toarray(),
        np.minimum(values.T, np.asarray([2, 3])),
    )
    cell_minimum = AxisMinimumMatrixSource(
        sparse,
        [1, 2, 3],
        axis="cell",
    )
    np.testing.assert_array_equal(
        cell_minimum.read_cells(0, 3).toarray(),
        np.minimum(values.T, np.asarray([1, 2, 3])[:, np.newaxis]),
    )
    with pytest.raises(MatrixSourceError, match="minimum axis"):
        AxisMinimumMatrixSource(sparse, [1, 2], axis="row")
    with pytest.raises(MatrixSourceError, match="has length 1"):
        AxisMinimumMatrixSource(sparse, [1], axis="feature")
    with pytest.raises(ResourceLimitError, match="maxMetadataBytes=8"):
        AxisMinimumMatrixSource(
            sparse,
            [1, 2],
            axis="feature",
            limits=SourceLimits(maxMetadataBytes=8),
        )
    with pytest.raises(MatrixSourceError, match="finite and positive"):
        AxisMinimumMatrixSource(sparse, [1, 0], axis="feature")

    scaled = ScaleShiftMatrixSource(
        sparse,
        feature_scale=[2, 3],
        cell_scale=[1, 2, 4],
        global_scale=0.5,
    )
    assert scaled.is_sparse
    np.testing.assert_array_equal(
        scaled.read_cells(0, 3).toarray(),
        values.T * np.asarray([2, 3]) * 0.5 * np.asarray([1, 2, 4])[:, np.newaxis],
    )


def test_subassignment_residual_and_rank_validation_paths() -> None:
    source = DenseMatrixSource(np.arange(6, dtype=np.float64).reshape(2, 3))

    with pytest.raises(TypeError, match="feature index spelling"):
        Subassignment([], [], 1, featureIndices=[])
    with pytest.raises(TypeError, match="cell index spelling"):
        Subassignment([], [], 1, cellIndices=[])
    with pytest.raises(TypeError, match="requires feature indexes"):
        Subassignment([], [], None)
    assignment = Subassignment(featureIndices=[0], cellIndices=[1], value=9)
    assert assignment.feature_indices == [0]
    assert assignment.cell_indices == [1]

    with pytest.raises(ValueError, match="at least one assignment"):
        DelayedSubassignmentMatrixSource(source, [])
    with pytest.raises(TypeError, match="cannot be callable"):
        DelayedSubassignmentMatrixSource(
            source,
            [Subassignment([0], [0], lambda: 1)],
        )
    with pytest.raises(MatrixSourceError, match="source has shape"):
        DelayedSubassignmentMatrixSource(
            source,
            [Subassignment([0, 1], [0], DenseMatrixSource(np.ones((1, 1))))],
        )
    with pytest.raises(MatrixSourceError, match="array has shape"):
        DelayedSubassignmentMatrixSource(
            source,
            [Subassignment([0, 1], [0], np.ones((1, 2)))],
        )

    replacement = DenseMatrixSource(np.asarray([[9, 8], [7, 6]]))
    assigned = DelayedSubassignmentMatrixSource(
        source,
        [
            Subassignment([0, 1], [0, 2], replacement),
            Subassignment(np.asarray([], dtype=np.int64), [1], 5),
        ],
    )
    assert assigned.estimate_read_memory(0, 3).workingBytes > 0
    np.testing.assert_array_equal(
        assigned.read_cells(0, 3),
        [[9, 7], [1, 4], [8, 6]],
    )
    np.testing.assert_array_equal(assigned.read_cells(1, 2), [[1, 4]])

    with pytest.raises(MatrixSourceError, match="three values"):
        PearsonResidualMatrixSource(
            source,
            theta_inverse=[0.1, 0.2],
            gene_beta=[1, 1],
            cell_read_counts=[1, 1, 1],
            global_parameters=[1, -1],
        )
    with pytest.raises(MatrixSourceError, match="clip bounds are reversed"):
        PearsonResidualMatrixSource(
            source,
            theta_inverse=[0.1, 0.2],
            gene_beta=[1, 1],
            cell_read_counts=[1, 1, 1],
            global_parameters=[1, 2, -2],
        )
    with pytest.raises(MatrixSourceError, match="do not match"):
        LinearResidualMatrixSource(
            source,
            feature_parameters=np.ones((1, 1)),
            cell_parameters=np.ones((1, 3)),
        )
    with pytest.raises(ResourceLimitError, match="maxMetadataBytes=16"):
        LinearResidualMatrixSource(
            source,
            feature_parameters=np.ones((1, 2)),
            cell_parameters=np.ones((1, 3)),
            limits=SourceLimits(maxMetadataBytes=16),
        )

    with pytest.raises(MatrixSourceError, match="rank axis"):
        RankMatrixSource(source, axis="feature")
    with pytest.raises(MatrixSourceError, match="non-finite"):
        RankMatrixSource(DenseMatrixSource(np.asarray([[np.inf]]))).read_cells(0, 1)

    dense_scan = DenseMatrixSource(np.asarray([[1.0, np.inf]]))
    with pytest.raises(MatrixSourceError, match="non-finite"):
        RankMatrixSource(
            dense_scan,
            axis="row",
            limits=SourceLimits(tileCells=1),
        ).read_cells(0, 1)

    sparse_values = csc_matrix(np.asarray([[1.0, np.inf]]))
    sparse_scan = CscMatrixSource(
        sparse_values.data,
        sparse_values.indices,
        sparse_values.indptr,
        sparse_values.shape,
    )
    with pytest.raises(MatrixSourceError, match="non-finite"):
        RankMatrixSource(
            sparse_scan,
            axis="row",
            limits=SourceLimits(tileCells=1),
        ).read_cells(0, 1)


def test_operation_registry_rejects_malformed_recipes() -> None:
    source = DenseMatrixSource(np.eye(2))
    with pytest.raises(TypeError, match="specification must be a mapping"):
        build_matrix_operation(1)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedMatrixOperation, match="custom functions"):
        build_matrix_operation({"operation": 1}, source=source)
    with pytest.raises(UnsupportedMatrixOperation, match="unknown operation"):
        build_matrix_operation({"operation": "custom"}, source=source)
    with pytest.raises(UnsupportedMatrixOperation, match="empty class vector"):
        build_matrix_operation(
            {"operation": "subset", "class": []},
            source=source,
        )
    with pytest.raises(UnsupportedMatrixOperation, match="unknown or custom class"):
        build_matrix_operation(
            {"operation": "subset", "class": 3},
            source=source,
        )
    with pytest.raises(MatrixSourceError, match="requires a source sequence"):
        build_matrix_operation(
            {"operation": "rbind", "className": "RowBindMatrix"},
        )
    with pytest.raises(TypeError, match="bind sources"):
        build_matrix_operation(
            {
                "operation": "rbind",
                "className": "RowBindMatrix",
                "sources": [source, object()],
            },
        )
    assert (
        build_matrix_operation(
            {
                "operation": "aperm",
                "className": "DelayedAperm",
                "permutation": [1, 2],
            },
            source=source,
        )
        is source
    )
    with pytest.raises(UnsupportedMatrixOperation, match="permutation"):
        build_matrix_operation(
            {
                "operation": "aperm",
                "className": "DelayedAperm",
                "permutation": [3, 1],
            },
            source=source,
        )
    with pytest.raises(MatrixSourceError, match="has no right operand"):
        build_matrix_operation(
            {"operation": "binary", "className": "BinaryMatrix"},
            source=source,
        )
    with pytest.raises(MatrixSourceError, match="has no MatrixSource mask"):
        build_matrix_operation(
            {"operation": "mask", "className": "MaskMatrix"},
            source=source,
        )
    with pytest.raises(MatrixSourceError, match="requires assignments"):
        build_matrix_operation(
            {"operation": "subassignment", "className": "SubassignmentMatrix"},
            source=source,
        )

    complete = Subassignment([0], [0], 1)
    built = build_matrix_operation(
        {
            "operation": "subassignment",
            "className": "SubassignmentMatrix",
            "assignments": [complete],
        },
        source=source,
    )
    np.testing.assert_array_equal(built.read_cells(0, 1), [[1, 0]])
    with pytest.raises(TypeError, match="must be a mapping"):
        build_matrix_operation(
            {
                "operation": "subassignment",
                "className": "SubassignmentMatrix",
                "assignments": [object()],
            },
            source=source,
        )
    with pytest.raises(MatrixSourceError, match="is incomplete"):
        build_matrix_operation(
            {
                "operation": "subassignment",
                "className": "SubassignmentMatrix",
                "assignments": [{}],
            },
            source=source,
        )
    with pytest.raises(TypeError, match="right operand"):
        build_matrix_operation(
            {
                "operation": "multiply",
                "className": "MatrixMultiply",
                "right": 2,
            },
            source=source,
        )


def test_core_sources_validate_names_shapes_indexes_and_bind_contracts() -> None:
    limits = SourceLimits(
        maxFeatures=4,
        maxCells=5,
        maxNnz=6,
        maxBlockBytes=7,
        maxMetadataBytes=8,
        tileCells=9,
        compressedChunkNnz=10,
    )
    assert limits.max_features == 4
    assert limits.max_cells == 5
    assert limits.max_nnz == 6
    assert limits.max_block_bytes == 7
    assert limits.max_metadata_bytes == 8
    assert limits.tile_cells == 9
    assert limits.compressed_chunk_nnz == 10
    with pytest.raises(ValueError, match="maxFeatures must be positive"):
        SourceLimits(maxFeatures=0)

    values = np.asarray([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="row names must be a sequence"):
        DenseMatrixSource(values, row_names="rows")
    with pytest.raises(MatrixSourceError, match="invalid UTF-8"):
        DenseMatrixSource(values, row_names=[b"\xff", b"f2"])
    with pytest.raises(TypeError, match="only strings"):
        DenseMatrixSource(values, row_names=[1, 2])
    with pytest.raises(MatrixSourceError, match="NUL"):
        DenseMatrixSource(values, row_names=["bad\x00name", "f2"])
    with pytest.raises(ResourceLimitError, match="maxMetadataBytes=12"):
        DenseMatrixSource(
            values,
            row_names=["longer", "names"],
            limits=SourceLimits(maxMetadataBytes=12),
        )
    with pytest.raises(MatrixSourceError, match="length 1"):
        DenseMatrixSource(values, row_names=["f1"])
    with pytest.raises(MatrixSourceError, match="shape is required"):
        DenseMatrixSource(np.arange(4))
    with pytest.raises(MatrixSourceError, match="flat dense source has 3 values"):
        DenseMatrixSource(np.arange(3), shape=(2, 2))
    with pytest.raises(MatrixSourceError, match="does not match"):
        DenseMatrixSource(values, shape=(1, 4))
    with pytest.raises(MatrixSourceError, match="one or two-dimensional"):
        DenseMatrixSource(np.zeros((1, 1, 1)))

    matrix = csc_matrix(values)
    with pytest.raises(MatrixSourceError, match="unsupported Matrix class"):
        CscMatrixSource(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            matrix.shape,
            class_name="custom",
        )
    with pytest.raises(MatrixSourceError, match="p slot has shape"):
        CscMatrixSource([1], [0], [0, 1], (2, 2))
    with pytest.raises(MatrixSourceError, match="i slot must be one-dimensional"):
        CscMatrixSource([1], np.asarray([[0]]), [0, 1], (2, 1))
    with pytest.raises(MatrixSourceError, match="x slot has shape"):
        CscMatrixSource([1, 2], [0], [0, 1], (2, 1))
    with pytest.raises(TypeError, match="p slot must contain integers"):
        CscMatrixSource([1], [0], [0.0, 1.0], (2, 1))
    with pytest.raises(MatrixSourceError, match="must start at zero"):
        CscMatrixSource([1], [0], [1, 1], (2, 1))
    with pytest.raises(MatrixSourceError, match="i slot has length"):
        CscMatrixSource([], [0], [0, 0], (2, 1))
    with pytest.raises(TypeError, match="i slot must contain integers"):
        CscMatrixSource([1], [0.0], [0, 1], (2, 1))

    dense = DenseMatrixSource(
        values,
        row_names=["f1", "f2"],
        column_names=["c1", "c2"],
    )
    with pytest.raises(MatrixSourceError, match="indexes must be one-dimensional"):
        MappedMatrixSource(dense, feature_indices=[[0]])
    with pytest.raises(TypeError, match="indexes must contain integers"):
        MappedMatrixSource(dense, cell_indices=[0.0])
    with pytest.raises(IndexError, match="out-of-range"):
        MappedMatrixSource(dense, cell_indices=[2])
    passthrough = MappedMatrixSource(dense)
    np.testing.assert_array_equal(passthrough.read_cells(0, 2), values.T)
    empty = MappedMatrixSource(
        dense,
        cell_indices=np.asarray([], dtype=np.int64),
    )
    assert empty.read_cells(0, 0).shape == (0, 2)

    sparse_mapped = MappedMatrixSource(
        CscMatrixSource(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            matrix.shape,
        ),
        feature_indices=[1, 0],
    )
    np.testing.assert_array_equal(
        sparse_mapped.read_cells(0, 2).toarray(),
        values.T[:, [1, 0]],
    )
    with pytest.raises(ValueError, match="tile_cells must be positive"):
        TransposeMatrixSource(dense, tile_cells=0)

    with pytest.raises(ValueError, match="feature bind requires"):
        FeatureBindMatrixSource([])
    with pytest.raises(MatrixSourceError, match="equal cell counts"):
        FeatureBindMatrixSource([dense, DenseMatrixSource(np.ones((1, 1)))])
    with pytest.raises(MatrixSourceError, match="column names conflict"):
        FeatureBindMatrixSource(
            [
                dense,
                DenseMatrixSource(
                    np.ones((1, 2)),
                    row_names=["f3"],
                    column_names=["x", "y"],
                ),
            ]
        )
    with pytest.raises(ValueError, match="cell bind requires"):
        CellBindMatrixSource([])
    with pytest.raises(MatrixSourceError, match="equal feature counts"):
        CellBindMatrixSource([dense, DenseMatrixSource(np.ones((1, 1)))])
    with pytest.raises(MatrixSourceError, match="row names conflict"):
        CellBindMatrixSource(
            [
                dense,
                DenseMatrixSource(
                    np.ones((2, 1)),
                    row_names=["x", "y"],
                    column_names=["c3"],
                ),
            ]
        )


def test_layer_placements_validate_mappings_and_names() -> None:
    source = DenseMatrixSource(
        np.asarray([[1]]),
        row_names=["f1"],
        column_names=["c1"],
    )
    with pytest.raises(TypeError, match="feature index spelling"):
        LayerPlacement(source, [0], [0], featureIndices=[0])
    with pytest.raises(TypeError, match="cell index spelling"):
        LayerPlacement(source, [0], [0], cellIndices=[0])
    placement = LayerPlacement(source, featureIndices=[0], cellIndices=[0])
    assert placement.feature_indices == [0]
    assert placement.cell_indices == [0]

    with pytest.raises(ValueError, match="at least one layer"):
        LayerStitchMatrixSource([], row_names=[], column_names=[])
    with pytest.raises(MatrixSourceError, match="indexes are required"):
        LayerStitchMatrixSource(
            [DenseMatrixSource(np.asarray([[1]]))],
            row_names=["f1"],
            column_names=["c1"],
        )
    with pytest.raises(MatrixSourceError, match="global feature names must be unique"):
        LayerStitchMatrixSource(
            [source],
            row_names=["f1", "f1"],
            column_names=["c1"],
        )
    with pytest.raises(MatrixSourceError, match="is absent from the global axis"):
        LayerStitchMatrixSource(
            [source],
            row_names=["other"],
            column_names=["c1"],
        )
    with pytest.raises(MatrixSourceError, match="maps 2 features"):
        LayerStitchMatrixSource(
            [LayerPlacement(source, [0, 1], [0])],
            row_names=["f1", "f2"],
            column_names=["c1"],
        )
    with pytest.raises(MatrixSourceError, match="maps 2 cells"):
        LayerStitchMatrixSource(
            [LayerPlacement(source, [0], [0, 1])],
            row_names=["f1"],
            column_names=["c1", "c2"],
        )

    two_features = DenseMatrixSource(np.asarray([[1], [2]]))
    with pytest.raises(MatrixSourceError, match="repeats a global feature"):
        LayerStitchMatrixSource(
            [LayerPlacement(two_features, [0, 0], [0])],
            row_names=["f1"],
            column_names=["c1"],
        )
    two_cells = DenseMatrixSource(np.asarray([[1, 2]]))
    with pytest.raises(MatrixSourceError, match="repeats a global cell"):
        LayerStitchMatrixSource(
            [LayerPlacement(two_cells, [0], [0, 0])],
            row_names=["f1"],
            column_names=["c1"],
        )
    with pytest.raises(MatrixSourceError, match="feature names conflict"):
        LayerStitchMatrixSource(
            [LayerPlacement(source, [0], [0])],
            row_names=["other"],
            column_names=["c1"],
        )
    with pytest.raises(MatrixSourceError, match="cell names conflict"):
        LayerStitchMatrixSource(
            [LayerPlacement(source, [0], [0])],
            row_names=["f1"],
            column_names=["other"],
        )
