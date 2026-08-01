import gzip
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

import scarf.readers.seurat as seurat_module
from scarf.readers._rds import LazyAtomicVector, R_INT_NA, RdsClosedError, RType
from scarf.readers._seurat import SourceLimits
from scarf.readers.seurat import (
    SeuratImportError,
    SeuratReader,
    SeuratStringVector,
    inspect_seurat,
)


_FIXTURES = Path(__file__).resolve().parent / "datasets"
_V4_FIXTURE = _FIXTURES / "seurat_v4_1_3_pbmc_mye.rds"
_V5_FIXTURE = _FIXTURES / "seurat_assay5_synthetic.rds"


class _Wire:
    def integer(self, value: int) -> bytes:
        return struct.pack(">i", value)

    def real(self, value: float) -> bytes:
        return struct.pack(">d", value)

    @staticmethod
    def flags(
        r_type: RType,
        *,
        attributes: bool = False,
        tag: bool = False,
        object_: bool = False,
    ) -> int:
        return (
            int(r_type)
            | (int(object_) << 8)
            | (int(attributes) << 9)
            | (int(tag) << 10)
        )

    def header(self) -> bytes:
        return (
            b"X\n"
            + self.integer(3)
            + self.integer(4 * 65_536 + 4 * 256)
            + self.integer(3 * 65_536 + 5 * 256)
            + self.integer(5)
            + b"UTF-8"
        )

    def nil(self) -> bytes:
        return self.integer(RType.NIL_VALUE)

    def char(self, value: str | None) -> bytes:
        result = self.integer(self.flags(RType.CHAR))
        if value is None:
            return result + self.integer(-1)
        encoded = value.encode()
        return result + self.integer(len(encoded)) + encoded

    def symbol(self, value: str) -> bytes:
        return self.integer(RType.SYMBOL) + self.char(value)

    def builtin(self, value: str) -> bytes:
        encoded = value.encode()
        return (
            self.integer(self.flags(RType.BUILTIN))
            + self.integer(len(encoded))
            + encoded
        )

    def closure(self) -> bytes:
        return self.integer(self.flags(RType.CLOSURE)) + self.nil() + self.nil()

    def pair(self, car: bytes, cdr: bytes, *, tag: str) -> bytes:
        return (
            self.integer(self.flags(RType.PAIRLIST, tag=True))
            + self.symbol(tag)
            + car
            + cdr
        )

    def attributes(self, values: list[tuple[str, bytes]]) -> bytes:
        result = self.nil()
        for name, value in reversed(values):
            result = self.pair(value, result, tag=name)
        return result

    def string_vector(
        self,
        values: list[str | None],
        *,
        attributes: list[tuple[str, bytes]] | None = None,
    ) -> bytes:
        return (
            self.integer(self.flags(RType.STRING, attributes=attributes is not None))
            + self.integer(len(values))
            + b"".join(self.char(value) for value in values)
            + (self.attributes(attributes) if attributes is not None else b"")
        )

    def atomic_vector(
        self,
        r_type: RType,
        values: list[int] | list[float],
        *,
        attributes: list[tuple[str, bytes]] | None = None,
    ) -> bytes:
        encode = self.real if r_type is RType.REAL else self.integer
        return (
            self.integer(self.flags(r_type, attributes=attributes is not None))
            + self.integer(len(values))
            + b"".join(encode(value) for value in values)
            + (self.attributes(attributes) if attributes is not None else b"")
        )

    def integer_vector(
        self,
        values: list[int],
        *,
        attributes: list[tuple[str, bytes]] | None = None,
    ) -> bytes:
        return self.atomic_vector(RType.INTEGER, values, attributes=attributes)

    def logical_vector(
        self,
        values: list[int],
        *,
        attributes: list[tuple[str, bytes]] | None = None,
    ) -> bytes:
        return self.atomic_vector(RType.LOGICAL, values, attributes=attributes)

    def real_vector(
        self,
        values: list[float],
        *,
        attributes: list[tuple[str, bytes]] | None = None,
    ) -> bytes:
        return self.atomic_vector(RType.REAL, values, attributes=attributes)

    def vector(
        self,
        values: list[bytes],
        *,
        names: list[str] | None = None,
        attributes: list[tuple[str, bytes]] | None = None,
    ) -> bytes:
        attrs = list(attributes or [])
        if names is not None:
            attrs.insert(0, ("names", self.string_vector(names)))
        return (
            self.integer(self.flags(RType.VECTOR, attributes=bool(attrs)))
            + self.integer(len(values))
            + b"".join(values)
            + (self.attributes(attrs) if attrs else b"")
        )

    def s4(self, slots: list[tuple[str, bytes]]) -> bytes:
        return self.integer(
            self.flags(RType.S4, attributes=True, object_=True)
        ) + self.attributes(slots)

    def dimnames(
        self,
        rows: list[str] | None,
        columns: list[str] | None,
    ) -> bytes:
        return self.vector(
            [
                self.nil() if rows is None else self.string_vector(rows),
                self.nil() if columns is None else self.string_vector(columns),
            ]
        )

    def matrix(
        self,
        values: list[int] | list[float],
        shape: tuple[int, int],
        *,
        rows: list[str] | None = None,
        columns: list[str] | None = None,
        real: bool = False,
    ) -> bytes:
        attributes = [
            ("dim", self.integer_vector(list(shape))),
            ("dimnames", self.dimnames(rows, columns)),
            ("class", self.string_vector(["matrix", "array"])),
        ]
        if real:
            return self.real_vector(
                [float(value) for value in values],
                attributes=attributes,
            )
        return self.integer_vector(
            [int(value) for value in values],
            attributes=attributes,
        )

    def factor(
        self,
        values: list[int],
        levels: list[str],
        *,
        names: list[str] | None = None,
    ) -> bytes:
        attributes = [
            ("levels", self.string_vector(levels)),
            ("class", self.string_vector(["factor"])),
        ]
        if names is not None:
            attributes.append(("names", self.string_vector(names)))
        return self.integer_vector(values, attributes=attributes)

    def data_frame(
        self,
        columns: list[tuple[str, bytes]],
        row_names: list[str] | int,
        *,
        automatic_row_names: bool = True,
    ) -> bytes:
        encoded_rows = (
            self.string_vector(row_names)
            if isinstance(row_names, list)
            else self.integer_vector(
                [
                    R_INT_NA,
                    -row_names if automatic_row_names else row_names,
                ]
            )
        )
        return self.vector(
            [value for _, value in columns],
            attributes=[
                ("names", self.string_vector([name for name, _ in columns])),
                ("row.names", encoded_rows),
                ("class", self.string_vector(["data.frame"])),
            ],
        )

    def logmap(
        self,
        values: list[int],
        rows: list[str],
        layers: list[str],
    ) -> bytes:
        return self.logical_vector(
            values,
            attributes=[
                ("dim", self.integer_vector([len(rows), len(layers)])),
                ("class", self.string_vector(["LogMap"])),
                ("dimnames", self.dimnames(rows, layers)),
            ],
        )

    def document(self, root: bytes) -> bytes:
        return self.header() + root


def _legacy_assay(wire: _Wire) -> bytes:
    counts = wire.matrix(
        [1, 0, 0, 2, 3, 0],
        (2, 3),
        rows=["g1", "g2"],
        columns=["c1", "c2", "c3"],
    )
    feature_metadata = wire.data_frame(
        [("symbol", wire.string_vector(["G1", "G2"]))],
        ["g1", "g2"],
    )
    return wire.s4(
        [
            ("counts", counts),
            ("meta.features", feature_metadata),
            ("class", wire.string_vector(["Assay"])),
        ]
    )


def _assay5(
    wire: _Wire,
    *,
    invalid_membership: bool,
    overlap: bool,
) -> bytes:
    layer_names = ["counts.1", "counts.2", "data"]
    cell_values = [
        1,
        1,
        0,
        0,
        1 if overlap else 0,
        R_INT_NA if invalid_membership else 1,
        1,
        1,
        1,
    ]
    feature_values = [1, 1, 0, 0, 1, 1, 1, 1, 1]
    layers = wire.vector(
        [
            wire.matrix([1, 3, 2, 4], (2, 2)),
            wire.matrix(
                [5, 6, 7, 8] if overlap else [5, 6],
                (2, 2) if overlap else (2, 1),
            ),
            wire.matrix([0] * 9, (3, 3), real=True),
        ],
        names=layer_names,
    )
    metadata = wire.data_frame(
        [("kind", wire.string_vector(["a", "b", "c"]))],
        3,
        automatic_row_names=False,
    )
    return wire.s4(
        [
            ("layers", layers),
            (
                "cells",
                wire.logmap(cell_values, ["c1", "c2", "c3"], layer_names),
            ),
            (
                "features",
                wire.logmap(feature_values, ["p1", "p2", "p3"], layer_names),
            ),
            ("meta.data", metadata),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )


def _reduction(wire: _Wire) -> bytes:
    embeddings = wire.matrix(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (3, 2),
        rows=["c1", "c2", "c3"],
        columns=["PC_1", "PC_2"],
        real=True,
    )
    empty_loadings = wire.matrix([], (0, 0), real=True)
    return wire.s4(
        [
            ("cell.embeddings", embeddings),
            ("feature.loadings", empty_loadings),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([0])),
            ("stdev", wire.real_vector([2.0, 1.0])),
            ("key", wire.string_vector(["PC_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )


def _seurat_payload(
    *,
    invalid_membership: bool = False,
    overlap: bool = False,
) -> bytes:
    wire = _Wire()
    metadata = wire.data_frame(
        [
            ("logical", wire.logical_vector([1, 0, R_INT_NA])),
            ("integer", wire.integer_vector([1, R_INT_NA, 3])),
            ("real", wire.real_vector([1.5, float("nan"), 3.5])),
            ("character", wire.string_vector(["a", None, "c"])),
            ("group", wire.factor([1, 2, R_INT_NA], ["first", "second"])),
        ],
        ["c1", "c2", "c3"],
    )
    root = wire.s4(
        [
            (
                "assays",
                wire.vector(
                    [
                        _legacy_assay(wire),
                        _assay5(
                            wire,
                            invalid_membership=invalid_membership,
                            overlap=overlap,
                        ),
                    ],
                    names=["RNA", "ADT"],
                ),
            ),
            ("meta.data", metadata),
            ("active.assay", wire.string_vector(["RNA"])),
            (
                "active.ident",
                wire.factor(
                    [2, 1, R_INT_NA],
                    ["zero", "one"],
                    names=["c3", "c1", "c2"],
                ),
            ),
            (
                "reductions",
                wire.vector([_reduction(wire)], names=["pca"]),
            ),
            ("graphs", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    return wire.document(root)


def _write_fixture(
    path: Path,
    *,
    compressed: bool = False,
    invalid_membership: bool = False,
    overlap: bool = False,
) -> Path:
    payload = _seurat_payload(
        invalid_membership=invalid_membership,
        overlap=overlap,
    )
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    return path


def _write_cached_sidecar_fixture(
    path: Path,
    *,
    loader: str,
    matrix: np.ndarray | None = None,
    matrices: dict[str, np.ndarray] | None = None,
    dataset: str = "counts",
    package: str = "HDF5Array",
    source_class: str = "DelayedMatrix",
) -> Path:
    values = (
        np.asarray([[1, 0], [0, 2], [3, 0]], dtype=np.int32)
        if matrix is None
        else np.asarray(matrix)
    )
    sidecars = {"counts.h5": values} if matrices is None else matrices
    for filename, sidecar_values in sidecars.items():
        with h5py.File(path.with_name(filename), mode="w") as handle:
            handle.create_dataset(dataset, data=np.asarray(sidecar_values))

    wire = _Wire()
    assay = wire.s4(
        [
            ("layers", wire.vector([], names=[])),
            ("cells", wire.logmap([], ["c1", "c2", "c3"], [])),
            ("features", wire.logmap([], ["g1", "g2"], [])),
            (
                "meta.data",
                wire.data_frame(
                    [("symbol", wire.string_vector(["G1", "G2"]))],
                    2,
                ),
            ),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    cache = wire.data_frame(
        [
            ("layer", wire.string_vector(["counts"])),
            ("path", wire.string_vector([",".join(sidecars)])),
            ("class", wire.string_vector([source_class])),
            ("pkg", wire.string_vector([package])),
            ("fxn", wire.string_vector([loader])),
            ("assay", wire.string_vector(["RNA"])),
        ],
        1,
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    [("group", wire.string_vector(["a", "b", "a"]))],
                    ["c1", "c2", "c3"],
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            (
                "active.ident",
                wire.factor(
                    [1, 2, 1],
                    ["a", "b"],
                    names=["c1", "c2", "c3"],
                ),
            ),
            ("reductions", wire.vector([], names=[])),
            (
                "tools",
                wire.vector([cache], names=["SaveSeuratRds"]),
            ),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _write_delayed_hdf5array_fixture(
    path: Path,
    *,
    transformed: bool = False,
    delayed_primitive: bool = False,
    executable_function: bool = False,
) -> Path:
    with h5py.File(path.with_name("delayed-counts.h5"), mode="w") as handle:
        handle.create_dataset(
            "counts",
            data=np.asarray([[1, 0], [0, 2], [3, 0]], dtype=np.int32),
        )
    wire = _Wire()
    seed = wire.s4(
        [
            ("filepath", wire.string_vector(["delayed-counts.h5"])),
            ("name", wire.string_vector(["counts"])),
            ("class", wire.string_vector(["HDF5ArraySeed"])),
        ]
    )
    delayed = wire.s4(
        [
            ("seed", seed),
            ("class", wire.string_vector(["DelayedMatrix", "DelayedArray"])),
        ]
    )
    layer = delayed
    if transformed:
        layer = wire.s4(
            [
                ("matrix", delayed),
                (
                    "row_params",
                    wire.matrix([2.0, 1.0], (1, 2), real=True),
                ),
                ("dim", wire.real_vector([2.0, 3.0])),
                ("transpose", wire.logical_vector([0])),
                (
                    "class",
                    wire.string_vector(["TransformMinByRow", "TransformedMatrix"]),
                ),
            ]
        )
    if delayed_primitive:
        layer = wire.s4(
            [
                ("seed", delayed),
                ("OP", wire.builtin("+")),
                ("Largs", wire.vector([])),
                ("Rargs", wire.vector([wire.real_vector([2.0])])),
                (
                    "class",
                    wire.string_vector(
                        ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"]
                    ),
                ),
            ]
        )
    if executable_function:
        layer = wire.s4(
            [
                ("seed", delayed),
                ("OP", wire.closure()),
                ("Largs", wire.vector([])),
                ("Rargs", wire.vector([])),
                (
                    "class",
                    wire.string_vector(
                        ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"]
                    ),
                ),
            ]
        )
    assay = wire.s4(
        [
            ("layers", wire.vector([layer], names=["counts"])),
            ("cells", wire.logmap([1, 1, 1], ["c1", "c2", "c3"], ["counts"])),
            ("features", wire.logmap([1, 1], ["g1", "g2"], ["counts"])),
            (
                "meta.data",
                wire.data_frame(
                    [("symbol", wire.string_vector(["G1", "G2"]))],
                    2,
                ),
            ),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    [("group", wire.string_vector(["a", "b", "a"]))],
                    ["c1", "c2", "c3"],
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            (
                "active.ident",
                wire.factor(
                    [1, 2, 1],
                    ["a", "b"],
                    names=["c1", "c2", "c3"],
                ),
            ),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _write_bpcells_memory_fixture(path: Path) -> Path:
    wire = _Wire()
    layer = wire.s4(
        [
            ("idxptr", wire.real_vector([0.0, 1.0, 2.0, 4.0])),
            ("index", wire.integer_vector([0, 1, 0, 1])),
            ("val", wire.integer_vector([1, 2, 3, 4])),
            ("version", wire.string_vector(["unpacked-uint-matrix-v2"])),
            ("dim", wire.real_vector([2.0, 3.0])),
            ("transpose", wire.logical_vector([0])),
            ("dimnames", wire.dimnames(["g1", "g2"], ["c1", "c2", "c3"])),
            (
                "class",
                wire.string_vector(["UnpackedMatrixMem_uint32_t", "IterableMatrix"]),
            ),
        ]
    )
    assay = wire.s4(
        [
            ("layers", wire.vector([layer], names=["counts"])),
            ("cells", wire.logmap([1, 1, 1], ["c1", "c2", "c3"], ["counts"])),
            ("features", wire.logmap([1, 1], ["g1", "g2"], ["counts"])),
            (
                "meta.data",
                wire.data_frame(
                    [("symbol", wire.string_vector(["G1", "G2"]))],
                    2,
                ),
            ),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    [("group", wire.string_vector(["a", "b", "a"]))],
                    ["c1", "c2", "c3"],
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            (
                "active.ident",
                wire.factor(
                    [1, 2, 1],
                    ["a", "b"],
                    names=["c1", "c2", "c3"],
                ),
            ),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _write_fragment_matrix_fixture(path: Path) -> Path:
    wire = _Wire()
    fragments = wire.s4(
        [
            ("cell", wire.integer_vector([0, 1, 0, 2, 1, 2, 0])),
            ("start", wire.integer_vector([0, 5, 10, 12, 20, 1, 4])),
            ("end", wire.integer_vector([10, 15, 20, 18, 30, 9, 12])),
            ("end_max", wire.integer_vector([30])),
            ("chr_ptr", wire.real_vector([0.0, 5.0, 5.0, 7.0])),
            ("chr_names", wire.string_vector(["chr1", "chr2"])),
            ("cell_names", wire.string_vector(["c1", "c2", "c3"])),
            ("version", wire.string_vector(["unpacked-fragments-v2"])),
            (
                "class",
                wire.string_vector(["UnpackedMemFragments", "IterableFragments"]),
            ),
        ]
    )
    feature_ids = ["p0", "p_span", "p1", "p2", "p_chr2"]
    layer = wire.s4(
        [
            ("fragments", fragments),
            ("chr_id", wire.integer_vector([0, 0, 0, 0, 1])),
            ("start", wire.integer_vector([0, 11, 10, 5, 0])),
            ("end", wire.integer_vector([10, 14, 20, 25, 10])),
            ("chr_levels", wire.string_vector(["chr1", "chr2"])),
            ("mode", wire.string_vector(["insertions"])),
            ("transpose", wire.logical_vector([1])),
            ("dim", wire.real_vector([5.0, 3.0])),
            ("dimnames", wire.dimnames(feature_ids, ["c1", "c2", "c3"])),
            ("class", wire.string_vector(["PeakMatrix", "IterableMatrix"])),
        ]
    )
    assay = wire.s4(
        [
            ("layers", wire.vector([layer], names=["counts"])),
            ("cells", wire.logmap([1, 1, 1], ["c1", "c2", "c3"], ["counts"])),
            (
                "features",
                wire.logmap([1] * len(feature_ids), feature_ids, ["counts"]),
            ),
            (
                "meta.data",
                wire.data_frame(
                    [("symbol", wire.string_vector(feature_ids))],
                    len(feature_ids),
                ),
            ),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["ATAC"])),
            (
                "meta.data",
                wire.data_frame(
                    [("group", wire.string_vector(["a", "b", "a"]))],
                    ["c1", "c2", "c3"],
                ),
            ),
            ("active.assay", wire.string_vector(["ATAC"])),
            (
                "active.ident",
                wire.factor(
                    [1, 2, 1],
                    ["a", "b"],
                    names=["c1", "c2", "c3"],
                ),
            ),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def test_mixed_assay_dispatch_metadata_and_reduction(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "mixed.rds")

    with SeuratReader(path) as reader:
        assert reader.activeAssay == "RNA"
        assert reader.cellIds == ("c1", "c2", "c3")
        assert len(reader.inspection.sourceDigest) == 64
        assert reader.inspection.payloadDigest == reader.inspection.sourceDigest
        assert reader.inspection.compression == "none"
        rna_inspection = reader.inspection.assay("RNA")
        assert rna_inspection.sourceClass == "Assay"
        assert rna_inspection.dtype == "<i4"
        assert rna_inspection.backend == "DenseMatrixSource"
        assert rna_inspection.memoryEstimate is not None
        assert reader.inspection.assay("ADT").sourceClass == "Assay5"

        legacy = reader.get_assay("RNA")
        assert legacy.featureIds == ("g1", "g2")
        np.testing.assert_array_equal(
            legacy.counts.read_cells(0, 3),
            [[1, 0], [0, 2], [3, 0]],
        )

        assay5 = reader.get_assay("ADT")
        assert assay5.featureIds == ("p1", "p2", "p3")
        np.testing.assert_array_equal(
            assay5.counts.read_cells(0, 3).toarray(),
            [[1, 3, 0], [2, 4, 0], [0, 5, 6]],
        )
        assert any(
            notice.code == "ignored_normalized_layer"
            and notice.objectPath == "assays/ADT/layers/data"
            for notice in assay5.notices
        )

        assert reader.cellMetadata.column("logical").kind == "logical"
        integer = reader.cellMetadata.column("integer").read_block(0, 3)
        np.testing.assert_array_equal(integer.missing, [False, True, False])
        real = reader.cellMetadata.column("real").read_block(0, 3)
        np.testing.assert_array_equal(real.missing, [False, True, False])
        character = reader.cellMetadata.column("character").read_block(0, 3)
        assert character.values == ("a", None, "c")
        group = reader.cellMetadata.column("group")
        assert group.levels == ("first", "second")
        assert group.read_decoded_block(0, 3) == ("first", "second", None)
        assert reader.activeIdentity.levels == ("zero", "one")
        assert reader.activeIdentity.read_decoded_block(0, 3) == (
            "zero",
            None,
            "one",
        )

        pca = reader.get_reduction("pca")
        pca_inspection = reader.inspection.reduction("pca")
        assert pca_inspection.dtype == "<f8"
        assert pca_inspection.backend == "SeuratRMatrix"
        assert pca_inspection.memoryEstimate is not None
        assert pca.role == "graphCoordinates"
        assert pca.dimensions == (3, 2)
        assert pca.assayUsed == "RNA"
        assert pca.imported
        assert not pca.computedByScarf
        np.testing.assert_array_equal(
            pca.cellEmbeddings.read_rows(1, 3),
            [[2.0, 5.0], [3.0, 6.0]],
        )


def test_explicit_assay_layer_overrides(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "layer-overrides.rds")

    with SeuratReader(
        path,
        assays=["ADT"],
        assay_layers={"ADT": ["counts.1"]},
        reductions=[],
    ) as reader:
        assay = reader.get_assay("ADT")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 3, 0], [2, 4, 0], [0, 0, 0]],
        )
        assert any(
            notice.code == "ignored_unselected_count_layer"
            and notice.objectPath == "assays/ADT/layers/counts.2"
            for notice in assay.notices
        )

    with SeuratReader(
        path,
        assays=["ADT"],
        assay_layers={"ADT": ["data"]},
        reductions=[],
    ) as reader:
        diagnostic = reader.inspection.assay("ADT").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "invalid_layer_override"

    invalid_ignored = _write_fixture(
        tmp_path / "ignored-invalid-layer.rds",
        invalid_membership=True,
    )
    with SeuratReader(
        invalid_ignored,
        assays=["ADT"],
        assay_layers={"ADT": ["counts.1"]},
        reductions=[],
    ) as reader:
        assert reader.inspection.assay("ADT").importable

    with SeuratReader(
        path,
        assays=["RNA"],
        assay_layers={"RNA": ["counts.1"]},
        reductions=[],
    ) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "invalid_layer_override"


def test_save_seurat_rds_cache_restores_safe_hdf5array_layer(
    tmp_path: Path,
) -> None:
    source = _write_cached_sidecar_fixture(
        tmp_path / "cached.rds",
        loader=(
            "function(x) HDF5Array::HDF5Array(filepath = x, "
            "name = 'counts', as.sparse = FALSE)"
        ),
    )

    with SeuratReader(source, reductions=[]) as reader:
        inspection = reader.inspection.assay("RNA")
        assert inspection.importable
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )
        assert any(
            notice.code == "restored_sidecar_cache_layer"
            and notice.objectPath == "tools/SaveSeuratRds/0"
            for notice in assay.notices
        )
        assert any(
            notice.code == "used_save_seurat_rds_cache"
            for notice in reader.inspection.notices
        )


def test_assay5_reads_delayedmatrix_hdf5array_seed(tmp_path: Path) -> None:
    source = _write_delayed_hdf5array_fixture(tmp_path / "delayed.rds")

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )


def test_assay5_reads_serialized_bpcells_parameter_matrix(tmp_path: Path) -> None:
    source = _write_delayed_hdf5array_fixture(
        tmp_path / "transformed.rds",
        transformed=True,
    )

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 1], [2, 0]],
        )


def test_assay5_reads_allowlisted_delayedarray_primitive(tmp_path: Path) -> None:
    source = _write_delayed_hdf5array_fixture(
        tmp_path / "delayed-primitive.rds",
        delayed_primitive=True,
    )

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[3, 2], [2, 4], [5, 2]],
        )


def test_assay5_rejects_executable_delayedarray_function(tmp_path: Path) -> None:
    source = _write_delayed_hdf5array_fixture(
        tmp_path / "delayed-function.rds",
        executable_function=True,
    )

    with SeuratReader(source, reductions=[]) as reader:
        inspection = reader.inspection.assay("RNA")
        assert not inspection.importable
        assert inspection.blockingDiagnostic is not None
        assert inspection.blockingDiagnostic.code == "unsupported_matrix_function"
        assert inspection.blockingDiagnostic.objectPath.endswith("layers/counts/OP")
        with pytest.raises(SeuratImportError, match="executable R semantics"):
            reader.get_assay("RNA")


def test_assay5_reads_serialized_bpcells_memory_leaf(tmp_path: Path) -> None:
    source = _write_bpcells_memory_fixture(tmp_path / "memory.rds")

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 4]],
        )


def test_assay5_reads_serialized_fragment_matrix_graph(tmp_path: Path) -> None:
    source = _write_fragment_matrix_fixture(tmp_path / "fragments.rds")

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("ATAC")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [
                [2, 0, 2, 3, 1],
                [1, 0, 1, 3, 0],
                [0, 1, 2, 2, 2],
            ],
        )


def test_save_seurat_rds_cache_restores_generated_bpcells_cbind_recipe(
    tmp_path: Path,
) -> None:
    leaf = "function(x) BPCells::open_matrix_anndata_hdf5(path = x, group = 'X')"
    loader = (
        "function(x) { "
        "paths <- unlist(x = strsplit(x = x, split = ',')); "
        f'fxns <- list("{leaf}", "{leaf}"); '
        "mats <- vector(mode = 'list', length = length(x = paths)); "
        "for (i in seq_along(paths)) { "
        "fn <- eval(str2lang(fxns[[i]])); "
        "mats[[i]] <- fn(paths[i]); "
        "}; return(Reduce(cbind, mats)); }"
    )
    source = _write_cached_sidecar_fixture(
        tmp_path / "cached-bind.rds",
        loader=loader,
        matrices={
            "part-one.h5": np.asarray([[1, 0]], dtype=np.int32),
            "part-two.h5": np.asarray([[0, 2], [3, 0]], dtype=np.int32),
        },
        dataset="X",
        package="BPCells",
        source_class="IterableMatrix",
    )

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )


def test_save_seurat_rds_cache_rejects_code_and_irrecoverable_membership(
    tmp_path: Path,
) -> None:
    unsafe = _write_cached_sidecar_fixture(
        tmp_path / "unsafe.rds",
        loader="function(x) system(x)",
    )
    with SeuratReader(unsafe, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "unsupported_sidecar_cache_recipe"
        assert diagnostic.objectPath == "tools/SaveSeuratRds/0/fxn"

    irrecoverable = _write_cached_sidecar_fixture(
        tmp_path / "irrecoverable.rds",
        loader=(
            "function(x) HDF5Array::HDF5Array(filepath = x, "
            "name = 'counts', as.sparse = FALSE)"
        ),
        matrix=np.asarray([[1, 0], [0, 2]], dtype=np.int32),
    )
    with SeuratReader(irrecoverable, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "irrecoverable_sidecar_cache"
        assert diagnostic.objectPath == "assays/RNA/layers/counts/Dimnames/1"


def test_single_open_lifecycle_and_bounded_atomic_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_fixture(tmp_path / "single-open.rds")
    original_open = seurat_module.open_rds
    open_count = 0

    def tracked_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(seurat_module, "open_rds", tracked_open)
    reader = SeuratReader(path, assays=["RNA"], reductions=["pca"])
    assert open_count == 1

    assay = reader.get_assay("RNA")
    dense = assay.counts._source
    target_counts = dense._values
    reduction = reader.get_reduction("pca")
    target_embeddings = reduction.cellEmbeddings._values
    original_read = LazyAtomicVector.read_block
    reads: dict[int, list[tuple[int, int]]] = {
        id(target_counts): [],
        id(target_embeddings): [],
    }

    def tracked_read(self, start, stop):
        if id(self) in reads:
            reads[id(self)].append((start, stop))
        return original_read(self, start, stop)

    monkeypatch.setattr(LazyAtomicVector, "read_block", tracked_read)
    np.testing.assert_array_equal(assay.counts.read_cells(1, 2), [[0, 2]])
    assert reads[id(target_counts)] == [(2, 4)]
    reduction.cellEmbeddings.read_rows(1, 3)
    assert reads[id(target_embeddings)] == [(1, 3), (4, 6)]

    reader.close()
    reader.close()
    with pytest.raises(RdsClosedError, match="RDS document is closed"):
        _ = reader.inspection
    with pytest.raises(RdsClosedError, match="RDS document is closed"):
        assay.counts.read_cells(0, 1)
    with pytest.raises(RdsClosedError, match="RDS document is closed"):
        reduction.cellEmbeddings.read_rows(0, 1)


def test_identifier_axes_remain_block_readable_and_spill_indexes_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_fixture(tmp_path / "streamed-identifiers.rds")
    original = seurat_module._read_text_vector

    def reject_materialized_ids(node, *, object_path, **kwargs):
        if object_path.endswith("row.names") or "/Dimnames/" in object_path:
            raise AssertionError(f"identifier axis was materialized at {object_path}")
        return original(node, object_path=object_path, **kwargs)

    monkeypatch.setattr(seurat_module, "_read_text_vector", reject_materialized_ids)
    with SeuratReader(path, temp_dir=tmp_path) as reader:
        assert isinstance(reader.cellIds, SeuratStringVector)
        assert reader.cellIds.read_block(1, 3) == ("c2", "c3")
        assert isinstance(reader.get_assay("RNA").featureIds, SeuratStringVector)
        assert not tuple(tmp_path.glob("scarf-seurat-ids-*.sqlite3"))


def test_identifier_index_budget_is_enforced(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "identifier-budget.rds")
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(
            path,
            matrix_limits=SourceLimits(maxMetadataBytes=4_096),
        )
    assert error.value.code == "metadata_index_limit"
    assert error.value.objectPath == "meta.data/row.names"


def test_detached_inspection_cleans_compressed_scratch(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "compressed.rds", compressed=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    result = inspect_seurat(
        source,
        temp_dir=scratch,
        assays=["RNA"],
        reductions=["pca"],
    )

    assert result.activeAssay == "RNA"
    assert result.assay("RNA").dimensions == (2, 3)
    assert result.reduction("pca").dimensions == (3, 2)
    assert tuple(scratch.iterdir()) == ()


def test_logmap_failure_is_item_local_and_strict_extraction_raises(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path / "bad-logmap.rds",
        invalid_membership=True,
    )

    with SeuratReader(path) as reader:
        rna = reader.inspection.assay("RNA")
        adt = reader.inspection.assay("ADT")
        pca = reader.inspection.reduction("pca")
        assert rna.importable
        assert pca.importable
        assert not adt.importable
        assert adt.blockingDiagnostic is not None
        assert adt.blockingDiagnostic.code == "invalid_logmap_membership"
        assert adt.blockingDiagnostic.objectPath == "assays/ADT/cells/counts.2"

        with pytest.raises(SeuratImportError) as error:
            reader.get_assay("ADT")
        assert error.value.code == "invalid_logmap_membership"
        assert error.value.objectPath == "assays/ADT/cells/counts.2"
        assert error.value.context == {"row": 2, "value": R_INT_NA}


def test_logmap_coordinate_overlap_has_exact_diagnostic(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "overlap.rds", overlap=True)

    with SeuratReader(path, assays=["ADT"], reductions=[]) as reader:
        inspection = reader.inspection.assay("ADT")
        assert not inspection.importable
        assert inspection.blockingDiagnostic is not None
        assert inspection.blockingDiagnostic.code == "layer_stitch_conflict"
        assert inspection.blockingDiagnostic.objectPath == "assays/ADT/layers"
        assert "counts.1" in inspection.blockingDiagnostic.message
        assert "counts.2" in inspection.blockingDiagnostic.message


def test_real_seurat_v4_fixture() -> None:
    with SeuratReader(_V4_FIXTURE) as reader:
        assert reader.activeAssay == "RNA"
        assert reader.cellIds[:3] == (
            "sample1_GAGTCATGTACCCGCA-1",
            "sample1_TGGAGGAGTGTATACC-1",
            "sample1_CCCGGAAGTTGGCTAT-1",
        )
        assay = reader.get_assay("RNA")
        assert assay.sourceClass == "Assay"
        assert assay.dimensions == (17_195, 1_000)
        assert assay.featureIds[:3] == ("AL627309.1", "AL627309.5", "LINC01409")
        assert assay.counts.read_cells(7, 9).shape == (2, 17_195)
        assert reader.cellMetadata.columnNames[:3] == (
            "orig.ident",
            "nCount_RNA",
            "nFeature_RNA",
        )
        assert reader.activeIdentity.levels == ("DC", "Mono CD14", "Mono FCGR3A")
        assert reader.activeIdentity.read_decoded_block(0, 3) == (
            "Mono CD14",
            "Mono CD14",
            "Mono CD14",
        )
        assert reader.get_reduction("pca").dimensions == (1_000, 50)
        assert reader.get_reduction("pca").featureLoadings is not None
        assert reader.get_reduction("umap").dimensions == (1_000, 2)
        assert reader.get_reduction("umap").role == "displayEmbedding"


def test_real_seurat_v5_fixture() -> None:
    with SeuratReader(_V5_FIXTURE) as reader:
        assert reader.activeAssay == "RNA"
        assert reader.cellIds[:3] == ("Cell1", "Cell2", "Cell3")
        assay = reader.get_assay("RNA")
        assert assay.sourceClass == "Assay5"
        assert assay.dimensions == (500, 300)
        assert assay.featureIds[:3] == ("Gene1", "Gene2", "Gene3")
        assert assay.counts.read_cells(11, 13).shape == (2, 500)
        assert np.all(assay.cellMembership)
        assert reader.activeIdentity.read_decoded_block(0, 3) == ("1", "0", "2")
        pca = reader.get_reduction("pca")
        assert pca.dimensions == (300, 20)
        assert pca.featureLoadings is not None
        assert pca.featureLoadings.shape == (200, 20)
        assert reader.get_reduction("umap").dimensions == (300, 2)
        assert reader.get_reduction("umap").role == "displayEmbedding"
