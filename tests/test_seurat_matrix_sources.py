from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix

from scarf.readers._seurat import (
    BPCellsDirectoryMatrixSource,
    BPCellsHDF5MatrixSource,
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
    LayerPlacement,
    LayerStitchMatrixSource,
    MappedMatrixSource,
    MaskMatrixSource,
    MatrixMultiplySource,
    MatrixSource,
    MatrixSourceError,
    RankMatrixSource,
    RenamedMatrixSource,
    ResourceLimitError,
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
    validate_hdf5_file,
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
