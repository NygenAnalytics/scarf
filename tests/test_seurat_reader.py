import gzip
import hashlib
import io
import sqlite3
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.sparse import csc_matrix

import scarf.readers.seurat as seurat_module
from scarf.readers._rds import LazyAtomicVector, R_INT_NA, RdsClosedError, RType
from scarf.readers._seurat import (
    MatrixSourceError,
    MemoryEstimate,
    ResourceLimitError,
    SourceLimits,
    UnsupportedMatrixOperation,
    fragment_source_from_slots,
    matrix_source_from_slots,
)
from scarf.readers.seurat import (
    SeuratImportError,
    SeuratMembership,
    SeuratMetadataColumn,
    SeuratReader,
    SeuratStringVector,
    inspect_seurat,
)
from tests.test_seurat_matrix_sources import (
    _bpcells_payload,
    _write_bpcells_directory,
    _write_bpcells_hdf5,
)


_FIXTURES = Path(__file__).resolve().parent / "datasets"
_V4_FIXTURE = _FIXTURES / "seurat_v4_1_3_pbmc_mye.rds"
_V5_FIXTURE = _FIXTURES / "seurat_assay5_synthetic.rds"


def _assert_counts_match_seurat_totals(reader: SeuratReader) -> None:
    """Seurat's nCount_RNA and nFeature_RNA are the per-cell sum and nonzero count."""
    n_cells = len(reader.cellIds)
    counts = reader.get_assay("RNA").counts.read_cells(0, n_cells)
    totals = reader.cellMetadata.column("nCount_RNA").read_block(0, n_cells).values
    detected = reader.cellMetadata.column("nFeature_RNA").read_block(0, n_cells).values
    np.testing.assert_array_equal(np.asarray(counts.sum(axis=1)).ravel(), totals)
    np.testing.assert_array_equal(counts.getnnz(axis=1), detected)


def _assert_pca_matches_its_stdev_and_orthonormal_loadings(
    reader: SeuratReader, *, rtol: float
) -> None:
    """Seurat's PCA stdev is each component's sd; its loadings are orthonormal."""
    pca = reader.get_reduction("pca")
    assert pca.stdev is not None
    assert pca.featureLoadings is not None
    embeddings = pca.cellEmbeddings.read_rows(0, pca.dimensions[0])
    np.testing.assert_allclose(
        embeddings.std(axis=0, ddof=1),
        pca.stdev.read_block(0, pca.dimensions[1]),
        rtol=rtol,
    )
    loadings = pca.featureLoadings.read_rows(0, pca.featureLoadings.shape[0])
    np.testing.assert_allclose(
        loadings.T @ loadings, np.eye(pca.dimensions[1]), atol=1e-10
    )


def _decoded(
    column: SeuratMetadataColumn, start: int, stop: int
) -> tuple[str | bytes | None, ...]:
    """Return a factor or character column window as values, None when missing."""
    block = column.read_block(start, stop)
    if column.kind == "character":
        return tuple(block.values)
    return tuple(
        None if missing else column.levels[int(code) - 1]
        for code, missing in zip(block.values, block.missing, strict=True)
    )


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

    def untagged_pair(self, car: bytes, cdr: bytes) -> bytes:
        return self.integer(self.flags(RType.PAIRLIST)) + car + cdr

    def altrep(self, class_name: str, state: bytes) -> bytes:
        info = self.untagged_pair(
            self.symbol(class_name),
            self.untagged_pair(self.symbol("base"), self.nil()),
        )
        return self.integer(RType.ALTREP) + info + state + self.nil()

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


def _legacy_assay(
    wire: _Wire,
    *,
    source_class: str = "Assay",
    extra_slots: list[tuple[str, bytes]] | None = None,
) -> bytes:
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
    slots = [
        ("counts", counts),
        ("meta.features", feature_metadata),
        *(extra_slots or []),
        ("class", wire.string_vector([source_class])),
    ]
    return wire.s4(slots)


def _assay5(
    wire: _Wire,
    *,
    invalid_membership: bool,
    overlap: bool,
    source_class: str = "Assay5",
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
            ("class", wire.string_vector([source_class])),
        ]
    )


def _reduction(wire: _Wire, *, assay_used: str = "RNA") -> bytes:
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
            ("assay.used", wire.string_vector([assay_used])),
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
    unnamed_empty_reductions: bool = False,
    unnamed_nonempty_reductions: bool = False,
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
                (
                    wire.vector([])
                    if unnamed_empty_reductions
                    else wire.vector(
                        [_reduction(wire)],
                        names=None if unnamed_nonempty_reductions else ["pca"],
                    )
                ),
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
    unnamed_empty_reductions: bool = False,
    unnamed_nonempty_reductions: bool = False,
) -> Path:
    payload = _seurat_payload(
        invalid_membership=invalid_membership,
        overlap=overlap,
        unnamed_empty_reductions=unnamed_empty_reductions,
        unnamed_nonempty_reductions=unnamed_nonempty_reductions,
    )
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    return path


def _write_chromatin_fixture(path: Path) -> Path:
    wire = _Wire()
    metadata = wire.data_frame(
        [("well", wire.string_vector(["W3", "W3", "W3"]))],
        ["c1", "c2", "c3"],
    )
    chromatin = _legacy_assay(
        wire,
        source_class="ChromatinAssay",
        extra_slots=[
            (
                "ranges",
                wire.string_vector(["chr1:1-10", "chr2:5-20"]),
            )
        ],
    )
    root = wire.s4(
        [
            ("assays", wire.vector([chromatin], names=["ATAC"])),
            ("meta.data", metadata),
            ("active.assay", wire.string_vector(["ATAC"])),
            (
                "active.ident",
                wire.factor([1, 1, 1], ["cells"], names=["c1", "c2", "c3"]),
            ),
            (
                "reductions",
                wire.vector(
                    [_reduction(wire, assay_used="ATAC")],
                    names=["lsi"],
                ),
            ),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _write_single_assay_fixture(
    path: Path,
    *,
    wire: _Wire,
    assay: bytes,
    assay_name: str = "RNA",
) -> Path:
    metadata = wire.data_frame(
        [("group", wire.string_vector(["a", "b", "c"]))],
        ["c1", "c2", "c3"],
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=[assay_name])),
            ("meta.data", metadata),
            ("active.assay", wire.string_vector([assay_name])),
            (
                "active.ident",
                wire.factor([1, 1, 1], ["cells"], names=["c1", "c2", "c3"]),
            ),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _write_metadata_fixture(
    path: Path,
    *,
    wire: _Wire,
    columns: list[tuple[str, bytes]],
) -> Path:
    cells = ["c1", "c2", "c3"]
    root = wire.s4(
        [
            ("assays", wire.vector([_legacy_assay(wire)], names=["RNA"])),
            ("meta.data", wire.data_frame(columns, cells)),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1, 1, 1], ["cells"], names=cells)),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
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


def _composite_cache_loader(loader_list: str) -> str:
    return (
        "function(x) { "
        "paths <- unlist(x = strsplit(x = x, split = ',')); "
        f"fxns <- list({loader_list}); "
        "mats <- vector(mode = 'list', length = length(x = paths)); "
        "for (i in seq_along(paths)) { "
        "fn <- eval(str2lang(fxns[[i]])); "
        "mats[[i]] <- fn(paths[i]); "
        "}; return(Reduce(cbind, mats)); }"
    )


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


def _dense_factory_spec() -> dict[str, object]:
    return {
        "class": ["matrix", "array"],
        "slots": {
            ".Data": [1.0, 3.0, 2.0, 4.0],
            "dim": [2, 2],
        },
    }


def _memory_fragment_spec() -> dict[str, object]:
    return {
        "class": ["UnpackedMemFragments", "IterableFragments"],
        "slots": {
            "cell": np.asarray([0, 1, 0, 2, 1, 2, 0], dtype=np.int32),
            "start": np.asarray([0, 5, 10, 12, 20, 1, 4], dtype=np.int32),
            "end": np.asarray([10, 15, 20, 18, 30, 9, 12], dtype=np.int32),
            "end_max": np.asarray([30], dtype=np.int32),
            "chr_ptr": np.asarray([0, 5, 5, 7], dtype=np.float64),
            "chr_names": ["chr1", "chr2"],
            "cell_names": ["c1", "c2", "c3"],
            "version": ["unpacked-fragments-v2"],
        },
    }


def _fragment_matrix_spec(
    fragments: object,
    *,
    matrix_class: str = "PeakMatrix",
) -> dict[str, object]:
    slots: dict[str, object] = {
        "fragments": fragments,
        "chr_id": np.asarray([0], dtype=np.int32),
        "start": np.asarray([0], dtype=np.int32),
        "end": np.asarray([10], dtype=np.int32),
        "chr_levels": ["chr1", "chr2"],
        "mode": ["insertions"],
        "transpose": [1],
        "dim": [1, 3],
    }
    if matrix_class == "TileMatrix":
        slots["tile_width"] = np.asarray([5], dtype=np.int32)
        slots["dim"] = [2, 3]
    return {
        "class": [matrix_class, "IterableMatrix"],
        "slots": slots,
    }


def test_empty_unnamed_reductions_are_accepted(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "empty-unnamed-reductions.rds",
        unnamed_empty_reductions=True,
    )

    with SeuratReader(path) as reader:
        assert reader.inspection.reductions == ()
        assert reader.inspection.assay("RNA").importable


def test_nonempty_unnamed_reductions_remain_invalid(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path / "nonempty-unnamed-reductions.rds",
        unnamed_nonempty_reductions=True,
    )

    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)

    assert error.value.code == "invalid_named_list"
    assert error.value.objectPath == "reductions"


def test_chromatin_assay_uses_legacy_capabilities_and_reduction(
    tmp_path: Path,
) -> None:
    path = _write_chromatin_fixture(tmp_path / "chromatin.rds")

    with SeuratReader(path) as reader:
        inspection = reader.inspection.assay("ATAC")
        assert inspection.importable
        assert inspection.sourceClass == "ChromatinAssay"
        assay = reader.get_assay("ATAC")
        assert assay.sourceClass == "ChromatinAssay"
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3),
            [[1, 0], [0, 2], [3, 0]],
        )
        assert any(
            notice.code == "ignored_assay_slot"
            and notice.objectPath == "assays/ATAC/ranges"
            for notice in assay.notices
        )
        reduction = reader.get_reduction("lsi")
        assert reduction.assayUsed == "ATAC"
        assert reduction.dimensions == (3, 2)


def test_transposed_assay5_storage_is_rejected_explicitly(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    path = _write_single_assay_fixture(
        tmp_path / "assay5t.rds",
        wire=wire,
        assay=_assay5(
            wire,
            invalid_membership=False,
            overlap=False,
            source_class="Assay5T",
        ),
    )

    with SeuratReader(path, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "unsupported_assay_class"
        assert diagnostic.objectPath == "assays/RNA"


def test_malformed_assay5_and_legacy_layouts_keep_precise_diagnostics(
    tmp_path: Path,
) -> None:
    assay5_wire = _Wire()
    assay5_path = _write_single_assay_fixture(
        tmp_path / "malformed-assay5.rds",
        wire=assay5_wire,
        assay=assay5_wire.s4(
            [
                (
                    "layers",
                    assay5_wire.vector(
                        [
                            assay5_wire.matrix(
                                [1, 0, 0, 2, 3, 0],
                                (2, 3),
                            )
                        ],
                        names=["counts"],
                    ),
                ),
                ("class", assay5_wire.string_vector(["Assay5"])),
            ]
        ),
    )
    with SeuratReader(assay5_path, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "missing_slot"
        assert diagnostic.objectPath == "assays/RNA/cells"

    legacy_wire = _Wire()
    legacy_path = _write_single_assay_fixture(
        tmp_path / "malformed-legacy.rds",
        wire=legacy_wire,
        assay=legacy_wire.s4(
            [
                (
                    "counts",
                    legacy_wire.matrix(
                        [1, 0, 0, 2, 3, 0],
                        (2, 3),
                        rows=["g1", "g2"],
                        columns=["c1", "c2", "c3"],
                    ),
                ),
                ("class", legacy_wire.string_vector(["Assay"])),
            ]
        ),
    )
    with SeuratReader(legacy_path, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "missing_slot"
        assert diagnostic.objectPath == "assays/RNA/meta.features"


def test_mixed_assay_dispatch_metadata_and_reduction(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "mixed.rds")

    with SeuratReader(path) as reader:
        assert reader.activeAssay == "RNA"
        assert reader.cellIds == ("c1", "c2", "c3")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert reader.inspection.sourceDigest == digest
        assert reader.inspection.payloadDigest == digest
        assert reader.inspection.compression == "none"
        rna_inspection = reader.inspection.assay("RNA")
        assert rna_inspection.sourceClass == "Assay"
        assert rna_inspection.dtype == "<i4"
        assert rna_inspection.backend == "DenseMatrixSource"
        # The first cell's two int32 counts, read from the serialized vector.
        assert rna_inspection.memoryEstimate is not None
        assert rna_inspection.memoryEstimate.outputBytes == 8
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
        assert _decoded(group, 0, 3) == ("first", "second", None)
        assert reader.activeIdentity.levels == ("zero", "one")
        assert _decoded(reader.activeIdentity, 0, 3) == (
            "zero",
            None,
            "one",
        )

        pca = reader.get_reduction("pca")
        pca_inspection = reader.inspection.reduction("pca")
        assert pca_inspection.dtype == "<f8"
        assert pca_inspection.backend == "SeuratRMatrix"
        # One row of two float64 components.
        assert pca_inspection.memoryEstimate == MemoryEstimate(0, 16, 16)
        assert pca.role == "graphCoordinates"
        assert pca.dimensions == (3, 2)
        assert pca.assayUsed == "RNA"
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


@pytest.mark.parametrize(
    ("loader", "dataset"),
    [
        (
            "function(x) HDF5Array::H5ADMatrix(filepath = x)",
            "X",
        ),
        (
            "function(x) HDF5Array::H5ADMatrix(filepath = x, layer = 'counts')",
            "layers/counts",
        ),
    ],
)
def test_save_seurat_rds_cache_restores_h5ad_loaders(
    tmp_path: Path,
    loader: str,
    dataset: str,
) -> None:
    source = _write_cached_sidecar_fixture(
        tmp_path / "cached-h5ad.rds",
        loader=loader,
        dataset=dataset,
    )

    with SeuratReader(source, reductions=[]) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
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
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with SeuratReader(source, reductions=[], temp_dir=scratch) as reader:
        assay = reader.get_assay("ATAC")
        fragment = assay.counts._source.layers[0].source
        assert fragment._rowStore is None
        assert not list(scratch.glob("scarf-sparse-*"))
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [
                [2, 0, 2, 3, 1],
                [1, 0, 1, 3, 0],
                [0, 1, 2, 2, 2],
            ],
        )
        directory = Path(fragment._rowStore._directory.name)
        assert directory.is_relative_to(scratch)
        assert directory.is_dir()
    assert not directory.exists()
    reader.close()


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


def test_public_reader_containers_and_accessors_are_consistent(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path / "public-accessors.rds")
    with SeuratReader(
        path,
        assay_layers={"RNA": ["counts"]},
    ) as reader:
        document = reader.document
        assert not document.closed
        assert document.source.name == str(path)
        temp_paths = reader.document.temp_paths
        assert temp_paths
        assert all(Path(temp_path).exists() for temp_path in temp_paths)
        assert reader.assayNames == ("RNA", "ADT")
        assert tuple(item.name for item in reader.inspection.assays) == (
            "RNA",
            "ADT",
        )
        assert tuple(item.name for item in reader.inspection.reductions) == ("pca",)

        assert len(reader.cellIds) == 3
        assert reader.cellIds[-1] == "c3"
        assert reader.cellIds[::-1] == ("c3", "c2", "c1")
        assert tuple(reader.cellIds) == ("c1", "c2", "c3")
        assert reader.cellIds == ("c1", "c2", "c3")
        assert reader.cellMetadata.rowIds == reader.cellIds
        with pytest.raises(KeyError, match="missing"):
            reader.cellMetadata.column("missing")
        with pytest.raises(KeyError, match="missing"):
            reader.inspection.assay("missing")
        with pytest.raises(KeyError, match="missing"):
            reader.inspection.reduction("missing")

        assay = reader.get_assay()
        assert assay is reader.get_assay("RNA")
        assert assay.dimensions == (2, 3)
        assert assay.counts.shape == assay.dimensions
        assert assay.counts.dtype == np.dtype(np.int32)
        assert not assay.counts.is_sparse
        assert assay.counts.zero_preserving
        assert assay.counts.row_names is None
        assert assay.counts.column_names is None
        # One cell of two int32 counts, read straight from the serialized vector.
        assert assay.counts.estimate_read_memory(0, 1).outputBytes == 8

        reduction = reader.get_reduction("pca")
        assert reduction.stdev is not None
        assert len(reduction.stdev) == reduction.stdev.length == 2
        assert reduction.stdev.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(reduction.stdev.read_block(0, 2), [2.0, 1.0])
        np.testing.assert_array_equal(
            reduction.cellEmbeddings.read_rows(0, 1),
            [[1.0, 4.0]],
        )
        assert _decoded(reader.cellMetadata.column("character"), 0, 3) == (
            "a",
            None,
            "c",
        )
        integer_column = reader.cellMetadata.column("integer")

    assert document.closed
    assert all(not Path(temp_path).exists() for temp_path in temp_paths)
    with pytest.raises(RdsClosedError):
        _ = reader.inspection
    with pytest.raises(RdsClosedError):
        _ = reader.cellMetadata
    with pytest.raises(RdsClosedError):
        _ = reader.cellIds[0]
    with pytest.raises(
        RdsClosedError, match=r"^RDS document is closed at meta.data/integer$"
    ):
        integer_column.read_block(0, 1)
    with pytest.raises(RdsClosedError):
        reduction.stdev.read_block(0, 1)


def test_public_sequence_bounds_and_membership_validation(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "public-bounds.rds")
    with SeuratReader(path) as reader:
        with pytest.raises(IndexError, match="identifier window"):
            reader.cellIds.read_block(-1, 1)
        with pytest.raises(IndexError, match="identifier index out of range"):
            _ = reader.cellIds[3]

        integer = reader.cellMetadata.column("integer")
        with pytest.raises(TypeError, match="metadata bounds must be integers"):
            integer.read_block(True, 1)
        with pytest.raises(IndexError, match="metadata window"):
            integer.read_block(0, 4)

        matrix = reader.get_reduction("pca").cellEmbeddings
        with pytest.raises(TypeError, match="matrix bounds must be integers"):
            matrix.read_rows(False, 1)
        with pytest.raises(IndexError, match="matrix window"):
            matrix.read_rows(0, 4)

    membership = SeuratMembership(4, np.asarray([3, 1], dtype=np.int64))
    assert not membership.allIncluded
    assert len(membership) == 4
    np.testing.assert_array_equal(
        membership.read_block(0, 4),
        [False, True, False, True],
    )
    np.testing.assert_array_equal(membership.read_block(1, 3), [True, False])
    assert SeuratMembership(2, np.asarray([1, 0])).allIncluded
    with pytest.raises(IndexError, match="membership window"):
        membership.read_block(0, 5)
    with pytest.raises(ValueError, match="cannot be negative"):
        SeuratMembership(-1)
    with pytest.raises(ValueError, match="one-dimensional"):
        SeuratMembership(2, np.asarray([[0]]))
    with pytest.raises(ValueError, match="out of range"):
        SeuratMembership(2, np.asarray([2]))
    with pytest.raises(ValueError, match="duplicates"):
        SeuratMembership(2, np.asarray([1, 1]))


@pytest.mark.parametrize(
    ("loader_list", "message"),
    [
        ("unquoted", "invalid string list"),
        (r'"bad\q"', "unsupported escape"),
        ('"unterminated', "unterminated string"),
        ('"first" "second"', "string list is malformed"),
        ("", "no leaf recipes"),
    ],
)
def test_composite_cache_loader_rejects_malformed_string_lists(
    tmp_path: Path,
    loader_list: str,
    message: str,
) -> None:
    source = _write_cached_sidecar_fixture(
        tmp_path / "malformed-composite.rds",
        loader=_composite_cache_loader(loader_list),
        package="BPCells",
        source_class="IterableMatrix",
    )

    with SeuratReader(source, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "unsupported_sidecar_cache_recipe"
        assert diagnostic.objectPath == "tools/SaveSeuratRds/0/fxn"
        assert message in diagnostic.message


def test_composite_cache_loader_validates_package_and_path_cardinality(
    tmp_path: Path,
) -> None:
    leaf = "function(x) BPCells::open_matrix_anndata_hdf5(path = x, group = 'X')"
    wrong_package = _write_cached_sidecar_fixture(
        tmp_path / "wrong-package.rds",
        loader=_composite_cache_loader(f'"{leaf}"'),
        package="HDF5Array",
    )
    with SeuratReader(wrong_package, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "unsupported_sidecar_cache_recipe"
        assert diagnostic.objectPath == "tools/SaveSeuratRds/0/pkg"
        assert diagnostic.context == {"package": "HDF5Array"}

    mismatched_paths = _write_cached_sidecar_fixture(
        tmp_path / "mismatched-paths.rds",
        loader=_composite_cache_loader(f'"{leaf}", "{leaf}"'),
        package="BPCells",
        source_class="IterableMatrix",
    )
    with SeuratReader(mismatched_paths, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "invalid_sidecar_cache"
        assert diagnostic.objectPath == "tools/SaveSeuratRds/0/path"
        assert diagnostic.context == {"pathCount": 1, "loaderCount": 2}


@pytest.mark.parametrize("missing_slot", ["assays", "reductions"])
def test_root_requires_structural_named_lists(
    tmp_path: Path,
    missing_slot: str,
) -> None:
    wire = _Wire()
    slots: list[tuple[str, bytes]] = [
        (name, wire.vector([], names=[]))
        for name in ("assays", "reductions")
        if name != missing_slot
    ]
    slots.append(("class", wire.string_vector(["Seurat"])))
    path = tmp_path / f"missing-{missing_slot}.rds"
    path.write_bytes(wire.document(wire.s4(slots)))

    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)

    assert error.value.code == "missing_slot"
    assert error.value.objectPath == f"/{missing_slot}"
    assert error.value.context == {"slot": missing_slot}


def test_missing_delayed_seed_is_translated_at_the_layer_boundary(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    malformed_layer = wire.s4(
        [("class", wire.string_vector(["DelayedMatrix", "DelayedArray"]))]
    )
    assay = wire.s4(
        [
            ("layers", wire.vector([malformed_layer], names=["counts"])),
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
    source = _write_single_assay_fixture(
        tmp_path / "missing-delayed-seed.rds",
        wire=wire,
        assay=assay,
    )

    with SeuratReader(source, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "invalid_matrix"
        assert diagnostic.objectPath == "assays/RNA/layers/counts"
        assert diagnostic.context == {"causeType": "MatrixSourceError"}
        assert "missing one of ('seed',)" in diagnostic.message
        with pytest.raises(SeuratImportError) as error:
            reader.get_assay("RNA")
        assert error.value.code == diagnostic.code
        assert error.value.objectPath == diagnostic.objectPath


def test_missing_sidecar_error_keeps_layer_path_and_cause_type(tmp_path: Path) -> None:
    source = _write_cached_sidecar_fixture(
        tmp_path / "missing-sidecar.rds",
        loader=(
            "function(x) HDF5Array::HDF5Array(filepath = x, "
            "name = 'counts', as.sparse = FALSE)"
        ),
    )
    (tmp_path / "counts.h5").unlink()

    with SeuratReader(source, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "invalid_matrix"
        assert diagnostic.objectPath == "assays/RNA/layers/counts"
        assert diagnostic.context == {"causeType": "FileNotFoundError"}
        assert "counts.h5" in diagnostic.message


def test_missing_selections_are_item_local_and_strict_on_access(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path / "missing-selections.rds")
    with SeuratReader(
        path,
        assays=["missing"],
        reductions=["missing"],
    ) as reader:
        assay = reader.inspection.assay("missing")
        reduction = reader.inspection.reduction("missing")
        assert assay.blockingDiagnostic is not None
        assert assay.blockingDiagnostic.code == "assay_not_found"
        assert reduction.blockingDiagnostic is not None
        assert reduction.blockingDiagnostic.code == "reduction_not_found"

        with pytest.raises(SeuratImportError) as assay_error:
            reader.get_assay("missing")
        assert assay_error.value.code == "assay_not_found"
        with pytest.raises(SeuratImportError) as reduction_error:
            reader.get_reduction("missing")
        assert reduction_error.value.code == "reduction_not_found"
        with pytest.raises(SeuratImportError) as unselected_assay:
            reader.get_assay()
        assert unselected_assay.value.code == "assay_not_selected"
        with pytest.raises(SeuratImportError) as unselected_reduction:
            reader.get_reduction("pca")
        assert unselected_reduction.value.code == "reduction_not_selected"


@pytest.mark.parametrize(
    ("options", "error_type", "message"),
    [
        ({"assays": "RNA"}, TypeError, "selection must be a sequence"),
        ({"assays": ["RNA", "RNA"]}, ValueError, "duplicate names"),
        ({"assay_layers": []}, TypeError, "must map assay names"),
        (
            {"assay_layers": {"missing": ["counts"]}},
            ValueError,
            "unknown assay",
        ),
        (
            {"assay_layers": {"RNA": "counts"}},
            TypeError,
            "must be a sequence",
        ),
        ({"assay_layers": {"RNA": []}}, ValueError, "must not be empty"),
        (
            {"assay_layers": {"RNA": ["counts", "counts"]}},
            ValueError,
            "contains duplicates",
        ),
    ],
)
def test_selection_and_layer_override_arguments_are_validated(
    tmp_path: Path,
    options: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    path = _write_fixture(tmp_path / "invalid-selection.rds")
    with pytest.raises(error_type, match=message):
        SeuratReader(path, reductions=[], **options)


def test_factory_materializes_additional_structural_nodes() -> None:
    leaf = _dense_factory_spec()
    renamed = matrix_source_from_slots(
        {
            "class": ["DelayedSetDimnames", "DelayedUnaryOp"],
            "slots": {
                "seed": leaf,
                "dimnames": [["f1", "f2"], ["c1", "c2"]],
            },
        }
    )
    assert renamed.row_names == ("f1", "f2")
    assert renamed.column_names == ("c1", "c2")

    row_bound = matrix_source_from_slots(
        {
            "class": ["RowBindMatrices", "IterableMatrix"],
            "slots": {"matrix_list": [leaf, leaf]},
        }
    )
    column_bound = matrix_source_from_slots(
        {
            "class": ["ColBindMatrices", "IterableMatrix"],
            "slots": {"matrix_list": [leaf, leaf]},
        }
    )
    # The leaf holds features [1, 2] and [3, 4] over two cells.
    assert row_bound.shape == (4, 2)
    assert column_bound.shape == (2, 4)
    np.testing.assert_array_equal(
        row_bound.read_cells(0, 2), [[1, 3, 1, 3], [2, 4, 2, 4]]
    )
    np.testing.assert_array_equal(
        column_bound.read_cells(0, 4), [[1, 3], [2, 4], [1, 3], [2, 4]]
    )

    renamed_again = matrix_source_from_slots(
        {
            "class": ["RenameDims", "IterableMatrix"],
            "slots": {
                "matrix": leaf,
                "dimnames": [["a", "b"], ["x", "y"]],
            },
        }
    )
    assert renamed_again.row_names == ("a", "b")
    assert renamed_again.column_names == ("x", "y")
    converted = matrix_source_from_slots(
        {
            "class": ["ConvertMatrixType", "IterableMatrix"],
            "slots": {"matrix": leaf, "type": "float"},
        }
    )
    assert converted.dtype == np.dtype(np.float32)
    converted_values = converted.read_cells(0, 2)
    assert converted_values.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(converted_values, [[1, 3], [2, 4]])

    power = matrix_source_from_slots(
        {
            "class": ["TransformPow", "TransformedMatrix"],
            "slots": {"matrix": leaf, "global_params": [2.0]},
        }
    )
    np.testing.assert_array_equal(power.read_cells(0, 2), [[1, 9], [4, 16]])
    binarized = matrix_source_from_slots(
        {
            "class": ["TransformBinarize", "TransformedMatrix"],
            "slots": {"matrix": leaf, "global_params": [2.0, 1.0]},
        }
    )
    np.testing.assert_array_equal(binarized.read_cells(0, 2), [[0, 1], [0, 1]])

    added = matrix_source_from_slots(
        {
            "class": ["MatrixAddition", "IterableMatrix"],
            "slots": {"left": leaf, "right": leaf},
        }
    )
    np.testing.assert_array_equal(added.read_cells(0, 2), [[2, 6], [4, 8]])
    masked = matrix_source_from_slots(
        {
            "class": ["MatrixMask", "IterableMatrix"],
            "slots": {
                "matrix": leaf,
                "mask": {
                    "class": ["matrix", "array"],
                    "slots": {
                        ".Data": [1, 0, 0, 1],
                        "dim": [2, 2],
                    },
                },
            },
        }
    )
    np.testing.assert_array_equal(masked.read_cells(0, 2), [[0, 3], [2, 0]])
    multiplied = matrix_source_from_slots(
        {
            "class": ["MatrixMultiply", "IterableMatrix"],
            "slots": {
                "left": leaf,
                "right": leaf,
                "Dim": [2, 2],
            },
        }
    )
    # [[1, 2], [3, 4]] squared is [[7, 10], [15, 22]]; cells are its columns.
    np.testing.assert_array_equal(multiplied.read_cells(0, 2), [[7, 15], [10, 22]])

    delayed_subset = matrix_source_from_slots(
        {
            "class": ["DelayedSubset", "DelayedUnaryOp"],
            "slots": {"seed": leaf, "index": [None, [2]]},
        }
    )
    np.testing.assert_array_equal(delayed_subset.read_cells(0, 1), [[2, 4]])
    zero_subset = matrix_source_from_slots(
        {
            "class": ["MatrixSubset", "IterableMatrix"],
            "slots": {
                "matrix": leaf,
                "row_selection": [],
                "col_selection": [1],
                "zero_dims": [True, False],
                "dim": [0, 1],
            },
        }
    )
    assert zero_subset.shape == (0, 1)
    assert zero_subset.read_cells(0, 1).shape == (1, 0)
    assigned = matrix_source_from_slots(
        {
            "class": ["DelayedSubassign", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": leaf,
                "Lindex": [None, None],
                "Rvalue": [0.0],
            },
        }
    )
    np.testing.assert_array_equal(assigned.read_cells(0, 2), np.zeros((2, 2)))

    unary = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {"seed": leaf, "OP": "abs"},
        }
    )
    rounded = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": leaf,
                "OP": "round",
                "Largs": [],
                "Rargs": [1.0],
            },
        }
    )
    logged = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": leaf,
                "OP": "log",
                "Largs": [],
                "Rargs": [10.0],
            },
        }
    )
    left_argument = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {
                "seed": leaf,
                "OP": "-",
                "Largs": {"constant": [10.0]},
            },
        }
    )
    scalar_argument = matrix_source_from_slots(
        {
            "class": ["DelayedUnaryIsoOpWithArgs", "DelayedUnaryIsoOp"],
            "slots": {"seed": leaf, "OP": "+", "Rargs": 2.0},
        }
    )
    np.testing.assert_array_equal(unary.read_cells(0, 2), [[1, 3], [2, 4]])
    np.testing.assert_array_equal(rounded.read_cells(0, 2), [[1, 3], [2, 4]])
    np.testing.assert_allclose(
        logged.read_cells(0, 2),
        np.log10([[1, 3], [2, 4]]),
    )
    np.testing.assert_array_equal(left_argument.read_cells(0, 2), [[9, 7], [8, 6]])
    np.testing.assert_array_equal(scalar_argument.read_cells(0, 2), [[3, 5], [4, 6]])

    minimum = matrix_source_from_slots(
        {
            "class": ["TransformMinByRow", "TransformedMatrix"],
            "slots": {"matrix": leaf, "row_params": [2.0, 3.0]},
        }
    )
    inactive_scale_shift = matrix_source_from_slots(
        {
            "class": ["TransformScaleShift", "TransformedMatrix"],
            "slots": {
                "matrix": leaf,
                "active_transforms": [0, 0, 0, 0, 0, 0],
            },
        }
    )
    # Feature one is capped at 2 and feature two at 3.
    np.testing.assert_array_equal(minimum.read_cells(0, 2), [[1, 3], [2, 3]])
    np.testing.assert_array_equal(
        inactive_scale_shift.read_cells(0, 2),
        [[1, 3], [2, 4]],
    )

    # A residual without parameters leaves its input unchanged.
    for row_params, col_params in (([], [[1.0, 2.0]]), ([[1.0, 2.0]], [])):
        unadjusted = matrix_source_from_slots(
            {
                "class": ["TransformLinearResidual", "TransformedMatrix"],
                "slots": {
                    "matrix": leaf,
                    "row_params": row_params,
                    "col_params": col_params,
                },
            }
        )
        np.testing.assert_array_equal(unadjusted.read_cells(0, 2), [[1, 3], [2, 4]])

    # Serialized vectors are read through their bounded block reader.
    class _LazyExponent:
        def __init__(self) -> None:
            self.reads: list[tuple[int, int]] = []

        def __len__(self) -> int:
            return 1

        def read_block(self, start: int, stop: int) -> np.ndarray:
            self.reads.append((start, stop))
            return np.asarray([2.0])[start:stop]

    exponent = _LazyExponent()
    squared = matrix_source_from_slots(
        {
            "class": ["TransformPow", "TransformedMatrix"],
            "slots": {"matrix": leaf, "global_params": exponent},
        }
    )
    np.testing.assert_array_equal(squared.read_cells(0, 2), [[1, 9], [4, 16]])
    assert exponent.reads == [(0, 1)]

    # DelayedArray marks an axis whose names the seed keeps with -1.
    named_leaf = {
        **leaf,
        "slots": {**leaf["slots"], "dimnames": [["f1", "f2"], ["a", "b"]]},
    }
    for dimnames, expected in (
        ([-1, ["c1", "c2"]], (("f1", "f2"), ("c1", "c2"))),
        ([["r1", "r2"], np.asarray([-1], dtype=np.int32)], (("r1", "r2"), ("a", "b"))),
    ):
        inherited = matrix_source_from_slots(
            {
                "class": ["DelayedSetDimnames", "DelayedUnaryOp"],
                "slots": {"seed": named_leaf, "dimnames": dimnames},
            }
        )
        assert (inherited.row_names, inherited.column_names) == expected


@pytest.mark.parametrize(
    "case",
    [
        "not-mapping",
        "bad-slots",
        "empty-class",
        "missing-class",
        "bad-dimnames",
        "bad-transpose",
        "bad-dim",
        "negative-dim",
        "mismatched-dim",
        "bad-subset-index",
        "short-subset-index",
        "float-subset-index",
        "zero-subset-index",
        "bad-permutation",
        "bad-abind-sources",
        "bad-abind-axis",
        "bad-subassign-index",
        "short-subassign-index",
        "bad-subassign-value",
        "bad-bind-sources",
        "bad-stack",
        "bad-nary-op",
        "bad-nary-sources",
        "empty-nary",
        "bad-unary-op",
        "many-unary-args",
        "fragment-as-matrix",
        "unknown-class",
        "scalar-seed",
        "scalar-fragments",
        "short-zero-dims",
        "unknown-matrix-type",
        "long-binary-parameters",
    ],
)
def test_factory_rejects_invalid_structural_nodes(case: str) -> None:
    leaf = _dense_factory_spec()
    cases: dict[str, tuple[object, type[Exception], str]] = {
        "not-mapping": ([], TypeError, "must be a mapping"),
        "bad-slots": (
            {"class": "matrix", "slots": []},
            TypeError,
            "slots.*must be a mapping",
        ),
        "empty-class": (
            {"class": []},
            MatrixSourceError,
            "class vector cannot be empty",
        ),
        "missing-class": (
            {"slots": {}},
            UnsupportedMatrixOperation,
            "matrix class is missing",
        ),
        "bad-dimnames": (
            {
                "class": "matrix",
                ".Data": [1, 2, 3, 4],
                "dim": [2, 2],
                "dimnames": [["f1", "f2"]],
            },
            MatrixSourceError,
            "dimnames must contain row and column names",
        ),
        "bad-transpose": (
            {
                "class": "matrix",
                ".Data": [1, 2, 3, 4],
                "dim": [2, 2],
                "transpose": [0, 1],
            },
            MatrixSourceError,
            "transpose at .* must contain one logical value",
        ),
        "bad-dim": (
            {
                "class": "matrix",
                ".Data": [1, 2, 3, 4],
                "dim": [2, 2],
                "Dim": [2],
            },
            MatrixSourceError,
            "dim slot must contain two integers",
        ),
        "negative-dim": (
            {
                "class": "matrix",
                ".Data": [1, 2, 3, 4],
                "dim": [2, 2],
                "Dim": [-1, 2],
            },
            MatrixSourceError,
            "dim slot cannot contain negative values",
        ),
        "mismatched-dim": (
            {
                "class": "matrix",
                ".Data": [1, 2, 3, 4],
                "dim": [2, 2],
                "Dim": [3, 3],
            },
            MatrixSourceError,
            "does not match dim slot",
        ),
        "bad-subset-index": (
            {
                "class": "DelayedSubset",
                "slots": {"seed": leaf, "index": 1},
            },
            TypeError,
            "index.*must be a sequence",
        ),
        "short-subset-index": (
            {
                "class": "DelayedSubset",
                "slots": {"seed": leaf, "index": [[1]]},
            },
            UnsupportedMatrixOperation,
            "two-dimensional DelayedSubset",
        ),
        "float-subset-index": (
            {
                "class": "DelayedSubset",
                "slots": {
                    "seed": leaf,
                    "index": [np.asarray([1.5]), None],
                },
            },
            MatrixSourceError,
            "one-dimensional integer vector",
        ),
        "zero-subset-index": (
            {
                "class": "DelayedSubset",
                "slots": {"seed": leaf, "index": [[0], None]},
            },
            MatrixSourceError,
            "nonpositive R index",
        ),
        "bad-permutation": (
            {
                "class": "DelayedAperm",
                "slots": {"seed": leaf, "perm": [1]},
            },
            UnsupportedMatrixOperation,
            "two-dimensional permutations",
        ),
        "bad-abind-sources": (
            {
                "class": "DelayedAbind",
                "slots": {"seeds": leaf, "along": [1]},
            },
            TypeError,
            "seeds.*must be a sequence",
        ),
        "bad-abind-axis": (
            {
                "class": "DelayedAbind",
                "slots": {"seeds": [leaf], "along": [3]},
            },
            UnsupportedMatrixOperation,
            "row or column binding",
        ),
        "bad-subassign-index": (
            {
                "class": "DelayedSubassign",
                "slots": {
                    "seed": leaf,
                    "Lindex": "all",
                    "Rvalue": [0],
                },
            },
            TypeError,
            "Lindex.*must be a sequence",
        ),
        "short-subassign-index": (
            {
                "class": "DelayedSubassign",
                "slots": {
                    "seed": leaf,
                    "Lindex": [None],
                    "Rvalue": [0],
                },
            },
            UnsupportedMatrixOperation,
            "two-dimensional DelayedSubassign",
        ),
        "bad-subassign-value": (
            {
                "class": "DelayedSubassign",
                "slots": {
                    "seed": leaf,
                    "Lindex": [None, None],
                    "Rvalue": [1, 2],
                },
            },
            UnsupportedMatrixOperation,
            "one numeric scalar",
        ),
        "bad-bind-sources": (
            {
                "class": "RowBindMatrices",
                "slots": {"matrix_list": leaf},
            },
            TypeError,
            "matrix list.*must be a sequence",
        ),
        "bad-stack": (
            {
                "class": "DelayedUnaryIsoOpStack",
                "slots": {"seed": leaf, "OPS": "abs"},
            },
            UnsupportedMatrixOperation,
            "OPS must be a sequence",
        ),
        "bad-nary-op": (
            {
                "class": "DelayedNaryIsoOp",
                "slots": {"seeds": [leaf], "OP": ["+"]},
            },
            UnsupportedMatrixOperation,
            "OP must be a recognized primitive",
        ),
        "bad-nary-sources": (
            {
                "class": "DelayedNaryIsoOp",
                "slots": {"seeds": leaf, "OP": "+"},
            },
            TypeError,
            "seeds.*must be a sequence",
        ),
        "empty-nary": (
            {
                "class": "DelayedNaryIsoOp",
                "slots": {"seeds": [], "OP": "+"},
            },
            MatrixSourceError,
            "has no seeds",
        ),
        "bad-unary-op": (
            {
                "class": "DelayedUnaryIsoOpWithArgs",
                "slots": {"seed": leaf, "OP": ["+"]},
            },
            UnsupportedMatrixOperation,
            "OP must be a recognized primitive",
        ),
        "many-unary-args": (
            {
                "class": "DelayedUnaryIsoOpWithArgs",
                "slots": {
                    "seed": leaf,
                    "OP": "+",
                    "Largs": [1, 2],
                    "Rargs": [],
                },
            },
            UnsupportedMatrixOperation,
            "only one scalar left or right argument",
        ),
        "fragment-as-matrix": (
            _memory_fragment_spec(),
            UnsupportedMatrixOperation,
            "fragment source cannot be used as a matrix",
        ),
        "unknown-class": (
            {"class": "CustomMatrix"},
            UnsupportedMatrixOperation,
            "unknown or custom matrix class",
        ),
        "scalar-seed": (
            {"class": "DelayedMatrix", "slots": {"seed": 5}},
            TypeError,
            r"^matrix input at \$@seed must be a source or mapping$",
        ),
        "scalar-fragments": (
            {"class": "PeakMatrix", "slots": {"fragments": 5}},
            TypeError,
            r"^fragment input at \$@fragments must be a source or mapping$",
        ),
        "short-zero-dims": (
            {"class": "MatrixSubset", "slots": {"matrix": leaf, "zero_dims": [False]}},
            MatrixSourceError,
            r"^zero_dims at \$ must have two values$",
        ),
        "unknown-matrix-type": (
            {"class": "ConvertMatrixType", "slots": {"matrix": leaf, "type": "int64"}},
            UnsupportedMatrixOperation,
            r"\(unknown BPCells matrix type 'int64'\)$",
        ),
        "long-binary-parameters": (
            {
                "class": "TransformPow",
                "slots": {"matrix": leaf, "global_params": [1.0, 2.0, 3.0]},
            },
            MatrixSourceError,
            r"^global_params at \$ has more than 2 values$",
        ),
    }
    specification, error_type, message = cases[case]
    with pytest.raises(error_type, match=message):
        matrix_source_from_slots(specification)


@pytest.mark.parametrize(
    "case",
    [
        "invalid-version-utf8",
        "nontext-version",
        "bad-memory-shape",
        "negative-memory-shape",
        "memory-transpose-vector",
        "boolean-scalar-axis",
        "transpose-vector",
        "parameter-rank",
        "argument-vector",
        "boolean-argument",
        "round-parameter",
        "binary-parameter",
        "binarize-parameters",
        "active-shape",
        "incomplete-active-parameters",
        "missing-global-parameters",
        "empty-active-parameters",
        "pearson-parameter-shape",
        "memory-version-conflict",
        "byte-string-version",
        "two-type-names",
        "vector-axis",
    ],
)
def test_factory_validates_serialized_slot_forms(case: str) -> None:
    leaf = _dense_factory_spec()
    memory_class = ["UnpackedMatrixMem_uint32_t", "IterableMatrix"]
    cases: dict[str, tuple[dict[str, object], type[Exception], str]] = {
        "invalid-version-utf8": (
            {
                "class": memory_class,
                "slots": {"version": [b"\xff"], "dim": [1, 1]},
            },
            MatrixSourceError,
            "not valid UTF-8",
        ),
        "nontext-version": (
            {
                "class": memory_class,
                "slots": {"version": [1], "dim": [1, 1]},
            },
            MatrixSourceError,
            "must contain one string",
        ),
        "bad-memory-shape": (
            {
                "class": memory_class,
                "slots": {
                    "version": [b"unpacked-uint-matrix-v2"],
                    "dim": [1],
                },
            },
            MatrixSourceError,
            "must contain two integers",
        ),
        "negative-memory-shape": (
            {
                "class": memory_class,
                "slots": {
                    "version": ["unpacked-uint-matrix-v2"],
                    "dim": [-1, 1],
                },
            },
            MatrixSourceError,
            "cannot contain negative values",
        ),
        "memory-transpose-vector": (
            {
                "class": memory_class,
                "slots": {
                    "version": ["unpacked-uint-matrix-v2"],
                    "dim": [1, 1],
                    "transpose": [0, 1],
                },
            },
            MatrixSourceError,
            "transpose at .* must contain one logical value",
        ),
        "boolean-scalar-axis": (
            {
                "class": "DelayedAbind",
                "slots": {"seeds": [leaf], "along": [True]},
            },
            MatrixSourceError,
            "along.*must contain one integer",
        ),
        "transpose-vector": (
            {
                "class": "TransformMinByRow",
                "slots": {
                    "matrix": leaf,
                    "transpose": [False, True],
                    "row_params": [1.0, 2.0],
                },
            },
            MatrixSourceError,
            "must contain one logical value",
        ),
        "parameter-rank": (
            {
                "class": "TransformMinByRow",
                "slots": {
                    "matrix": leaf,
                    "row_params": np.zeros((1, 1, 1)),
                },
            },
            MatrixSourceError,
            "two-dimensional matrix",
        ),
        "argument-vector": (
            {
                "class": "DelayedUnaryIsoOpWithArgs",
                "slots": {
                    "seed": leaf,
                    "OP": "+",
                    "Rargs": [np.asarray([1.0, 2.0])],
                },
            },
            UnsupportedMatrixOperation,
            "scalar numeric arguments",
        ),
        "boolean-argument": (
            {
                "class": "DelayedUnaryIsoOpWithArgs",
                "slots": {"seed": leaf, "OP": "+", "Rargs": [True]},
            },
            UnsupportedMatrixOperation,
            "scalar numeric arguments",
        ),
        "round-parameter": (
            {
                "class": "TransformRound",
                "slots": {"matrix": leaf, "global_params": [1.5]},
            },
            MatrixSourceError,
            "requires one integer digit",
        ),
        "binary-parameter": (
            {
                "class": "TransformPow",
                "slots": {"matrix": leaf, "global_params": [1.0, 2.0]},
            },
            MatrixSourceError,
            "requires one parameter",
        ),
        "binarize-parameters": (
            {
                "class": "TransformBinarize",
                "slots": {"matrix": leaf, "global_params": [1.0]},
            },
            MatrixSourceError,
            "requires two parameters",
        ),
        "active-shape": (
            {
                "class": "TransformScaleShift",
                "slots": {"matrix": leaf, "active_transforms": [True]},
            },
            MatrixSourceError,
            "must have shape",
        ),
        "incomplete-active-parameters": (
            {
                "class": "TransformScaleShift",
                "slots": {
                    "matrix": leaf,
                    "active_transforms": [1, 0, 0, 0, 0, 0],
                },
            },
            MatrixSourceError,
            "parameters.*are incomplete",
        ),
        "missing-global-parameters": (
            {
                "class": "TransformScaleShift",
                "slots": {
                    "matrix": leaf,
                    "active_transforms": [0, 0, 1, 0, 0, 0],
                    "global_params": [1.0],
                },
            },
            MatrixSourceError,
            "must contain scale and shift",
        ),
        # An empty serialized vector holds no parameter row either.
        "empty-active-parameters": (
            {
                "class": "TransformScaleShift",
                "slots": {
                    "matrix": leaf,
                    "active_transforms": [1, 0, 0, 0, 0, 0],
                    "row_params": [],
                },
            },
            MatrixSourceError,
            r"^TransformScaleShift parameters at \$ are incomplete$",
        ),
        "pearson-parameter-shape": (
            {
                "class": "SCTransformPearson",
                "slots": {
                    "matrix": leaf,
                    "row_params": [1.0],
                    "col_params": [1.0],
                    "global_params": [1.0],
                },
            },
            MatrixSourceError,
            "parameters.*have invalid shapes",
        ),
        "memory-version-conflict": (
            {
                "class": memory_class,
                "slots": {
                    "version": ["packed-uint-matrix-v2"],
                    "dim": [1, 1],
                },
            },
            MatrixSourceError,
            "conflicts with format",
        ),
        "byte-string-version": (
            {
                "class": memory_class,
                "slots": {"version": b"unpacked-uint-matrix-v2", "dim": [1, 1]},
            },
            MatrixSourceError,
            r"^version at \$ must contain one string$",
        ),
        "two-type-names": (
            {
                "class": "ConvertMatrixType",
                "slots": {"matrix": leaf, "type": ["float", "double"]},
            },
            MatrixSourceError,
            r"^type at \$ must contain one string$",
        ),
        "vector-axis": (
            {"class": "DelayedAbind", "slots": {"seeds": [leaf], "along": [1, 2]}},
            MatrixSourceError,
            r"^along at \$ must contain one integer$",
        ),
    }
    specification, error_type, message = cases[case]
    with pytest.raises(error_type, match=message):
        matrix_source_from_slots(specification)


def test_fragment_factory_wrappers_expose_resources_and_records() -> None:
    base = fragment_source_from_slots(_memory_fragment_spec())
    assert base.recordCount == 7
    assert base.residentBytes >= base.metadataBytes
    with pytest.raises(IndexError, match="outside"):
        tuple(base.iter_chromosome(2))

    shifted = fragment_source_from_slots(
        {
            "class": ["ShiftFragments", "IterableFragments"],
            "slots": {
                "fragments": base,
                "shift_start": [1],
                "shift_end": [2],
            },
        }
    )
    assert shifted.chromosomeNames == base.chromosomeNames
    assert shifted.cellNames == base.cellNames
    assert shifted.recordCount == base.recordCount
    assert shifted.residentBytes == base.residentBytes
    assert shifted.metadataBytes == base.metadataBytes
    assert shifted.blockWorkingBytes == 2 * base.blockWorkingBytes
    original = next(base.iter_chromosome(0))
    moved = next(shifted.iter_chromosome(0))
    np.testing.assert_array_equal(moved.starts, original.starts + 1)
    np.testing.assert_array_equal(moved.ends, original.ends + 2)

    unbounded_length = fragment_source_from_slots(
        {
            "class": ["SelectLength", "IterableFragments"],
            "slots": {
                "fragments": base,
                "min_len": [R_INT_NA],
                "max_len": [R_INT_NA],
            },
        }
    )
    assert (
        sum(
            block.size
            for chromosome_id in range(len(unbounded_length.chromosomeNames))
            for block in unbounded_length.iter_chromosome(chromosome_id)
        )
        == base.recordCount
    )

    # Names cost their UTF-8 bytes plus an 8-byte reference: chr_ptr holds
    # 32 bytes, two chromosome names 24, and three cell names 30.
    assert base.metadataBytes == 86
    chromosome = fragment_source_from_slots(
        {
            "class": ["ChrSelectIndex", "IterableFragments"],
            "slots": {
                "fragments": base,
                "chr_index_selection": [2],
            },
        }
    )
    assert chromosome.chromosomeNames == ("chr2",)
    assert chromosome.metadataBytes == base.metadataBytes + 12
    assert [block.starts.tolist() for block in chromosome.iter_chromosome(0)] == [
        [1, 4]
    ]
    by_scalar_name = fragment_source_from_slots(
        {
            "class": ["ChrSelectName", "IterableFragments"],
            "slots": {"fragments": base, "chr_names": "chr2"},
        }
    )
    assert by_scalar_name.chromosomeNames == ("chr2",)
    with pytest.raises(IndexError, match="out of range"):
        tuple(chromosome.iter_chromosome(1))

    cells = fragment_source_from_slots(
        {
            "class": ["CellSelectName", "IterableFragments"],
            "slots": {
                "fragments": base,
                "cell_names": ["c3", "c1"],
            },
        }
    )
    assert cells.cellNames == ("c3", "c1")
    # The int64 mapping adds 24 bytes; the two new names add 20.
    assert cells.residentBytes == base.residentBytes + 24
    assert cells.metadataBytes == base.metadataBytes + 24 + 20
    # c3 becomes cell 0 and c1 cell 1; c2's fragments are dropped.
    assert [
        (block.cellIds.tolist(), block.starts.tolist())
        for block in cells.iter_chromosome(0)
    ] == [([1, 1, 0], [0, 10, 12])]

    renamed = fragment_source_from_slots(
        {
            "class": ["ChrRename", "IterableFragments"],
            "slots": {
                "fragments": base,
                "chr_names": ["one", "two"],
            },
        }
    )
    assert renamed.chromosomeNames == ("one", "two")
    assert renamed.metadataBytes == base.metadataBytes + 22 + 30
    assert next(renamed.iter_chromosome(0)).size == original.size

    region = fragment_source_from_slots(
        {
            "class": ["RegionSelect", "IterableFragments"],
            "slots": {
                "fragments": base,
                "chr_id": [0, 1],
                "start": [0, 100],
                "end": [10, 110],
                "chr_levels": ["chr1", "missing"],
                "invert_selection": False,
            },
        }
    )
    # Three int32 region vectors hold 24 bytes; the two levels need 27.
    assert region.residentBytes == base.residentBytes + 51
    assert region.metadataBytes == base.metadataBytes + 51
    region_blocks = tuple(region.iter_chromosome(0))
    np.testing.assert_array_equal(
        np.concatenate([block.starts for block in region_blocks]),
        [0, 5],
    )

    merged = fragment_source_from_slots(
        {
            "class": ["MergeFragments", "IterableFragments"],
            "slots": {"fragments_list": [base, base]},
        }
    )
    assert merged.recordCount == 2 * base.recordCount
    assert merged.residentBytes == 2 * base.residentBytes
    assert merged.metadataBytes == 2 * base.metadataBytes + 24 + 60
    # Buffered source blocks plus the merged output block.
    assert merged.blockWorkingBytes == 4 * base.blockWorkingBytes
    merged_blocks = tuple(merged.iter_chromosome(0))
    merged_starts = np.concatenate([block.starts for block in merged_blocks])
    merged_cells = np.concatenate([block.cellIds for block in merged_blocks])
    np.testing.assert_array_equal(merged_starts, [0, 0, 5, 5, 10, 10, 12, 12, 20, 20])
    assert set(merged_cells[merged_starts == 0].tolist()) == {0, len(base.cellNames)}
    with pytest.raises(IndexError, match="out of range"):
        tuple(merged.iter_chromosome(2))


@pytest.mark.parametrize(
    "case",
    [
        "not-mapping",
        "missing-class",
        "empty-class",
        "custom-base",
        "bad-slots",
        "bad-nested-source",
        "bad-merge-list",
        "empty-merge",
        "invalid-length-bounds",
        "duplicate-chromosome-names",
        "unknown-chromosome-name",
        "invalid-chromosome-index",
        "duplicate-chromosome-index",
        "duplicate-cell-names",
        "unknown-cell-name",
        "invalid-cell-index",
        "duplicate-cell-index",
        "invalid-cell-groups",
        "short-chromosome-renaming",
        "short-cell-renaming",
        "inconsistent-regions",
        "reversed-region",
        "invalid-region-logical",
        "missing-version",
        "invalid-version",
        "compression-conflict",
        "missing-directory",
        "missing-hdf5-group",
        "invalid-buffer-size",
        "invalid-prefix-utf8",
        "invalid-group-number",
        "nonnumeric-shift",
    ],
)
def test_fragment_factory_rejects_invalid_graphs(case: str) -> None:
    base = fragment_source_from_slots(_memory_fragment_spec())
    base_slots = dict(_memory_fragment_spec()["slots"])
    base_slots["buffer_size"] = [0]
    cases: dict[str, tuple[object, type[Exception], str]] = {
        "not-mapping": ([], TypeError, "must be a mapping"),
        "missing-class": (
            {},
            UnsupportedMatrixOperation,
            "fragment class is missing",
        ),
        "empty-class": (
            {"class": []},
            MatrixSourceError,
            "class vector cannot be empty",
        ),
        "custom-base": (
            {
                "class": ["UnpackedMemFragments", "CustomFragmentsBase"],
                "slots": {},
            },
            UnsupportedMatrixOperation,
            "unknown or custom fragment base class",
        ),
        "bad-slots": (
            {"class": "UnpackedMemFragments", "slots": []},
            TypeError,
            "slots.*must be a mapping",
        ),
        "bad-nested-source": (
            {
                "class": "ShiftFragments",
                "slots": {
                    "fragments": None,
                    "shift_start": [0],
                    "shift_end": [0],
                },
            },
            TypeError,
            "fragment input.*must be a source or mapping",
        ),
        "bad-merge-list": (
            {
                "class": "MergeFragments",
                "slots": {"fragments_list": base},
            },
            TypeError,
            "fragments_list.*must be a sequence",
        ),
        "empty-merge": (
            {
                "class": "MergeFragments",
                "slots": {"fragments_list": []},
            },
            MatrixSourceError,
            "requires at least one source",
        ),
        "invalid-length-bounds": (
            {
                "class": "SelectLength",
                "slots": {
                    "fragments": base,
                    "min_len": [5],
                    "max_len": [4],
                },
            },
            MatrixSourceError,
            "length bounds are invalid",
        ),
        "duplicate-chromosome-names": (
            {
                "class": "ChrSelectName",
                "slots": {
                    "fragments": base,
                    "chr_names": ["chr1", "chr1"],
                },
            },
            MatrixSourceError,
            "chromosome selection contains duplicates",
        ),
        "unknown-chromosome-name": (
            {
                "class": "ChrSelectName",
                "slots": {
                    "fragments": base,
                    "chr_names": ["missing"],
                },
            },
            MatrixSourceError,
            "unknown names",
        ),
        "invalid-chromosome-index": (
            {
                "class": "ChrSelectIndex",
                "slots": {
                    "fragments": base,
                    "chr_index_selection": [0],
                },
            },
            MatrixSourceError,
            "invalid R index",
        ),
        "duplicate-chromosome-index": (
            {
                "class": "ChrSelectIndex",
                "slots": {
                    "fragments": base,
                    "chr_index_selection": [1, 1],
                },
            },
            MatrixSourceError,
            "chromosome selection contains duplicates",
        ),
        "duplicate-cell-names": (
            {
                "class": "CellSelectName",
                "slots": {
                    "fragments": base,
                    "cell_names": ["c1", "c1"],
                },
            },
            MatrixSourceError,
            "cell selection contains duplicates",
        ),
        "unknown-cell-name": (
            {
                "class": "CellSelectName",
                "slots": {
                    "fragments": base,
                    "cell_names": ["missing"],
                },
            },
            MatrixSourceError,
            "unknown names",
        ),
        "invalid-cell-index": (
            {
                "class": "CellSelectIndex",
                "slots": {
                    "fragments": base,
                    "cell_index_selection": [0],
                },
            },
            MatrixSourceError,
            "invalid R index",
        ),
        "duplicate-cell-index": (
            {
                "class": "CellSelectIndex",
                "slots": {
                    "fragments": base,
                    "cell_index_selection": [1, 1],
                },
            },
            MatrixSourceError,
            "cell selection contains duplicates",
        ),
        "invalid-cell-groups": (
            {
                "class": "CellMerge",
                "slots": {
                    "fragments": base,
                    "group_names": ["one"],
                    "group_ids": [0, 0],
                },
            },
            MatrixSourceError,
            "groups do not match the source cells",
        ),
        "short-chromosome-renaming": (
            {
                "class": "ChrRename",
                "slots": {"fragments": base, "chr_names": ["one"]},
            },
            MatrixSourceError,
            "chromosome names have an invalid length",
        ),
        "short-cell-renaming": (
            {
                "class": "CellRename",
                "slots": {"fragments": base, "cell_names": ["one"]},
            },
            MatrixSourceError,
            "cell names have an invalid length",
        ),
        "inconsistent-regions": (
            {
                "class": "RegionSelect",
                "slots": {
                    "fragments": base,
                    "chr_id": [0, 1],
                    "start": [0],
                    "end": [10],
                    "chr_levels": ["chr1", "chr2"],
                },
            },
            MatrixSourceError,
            "region metadata is inconsistent",
        ),
        "reversed-region": (
            {
                "class": "RegionSelect",
                "slots": {
                    "fragments": base,
                    "chr_id": [0],
                    "start": [10],
                    "end": [5],
                    "chr_levels": ["chr1", "chr2"],
                },
            },
            MatrixSourceError,
            "region end precedes its start",
        ),
        "invalid-region-logical": (
            {
                "class": "RegionSelect",
                "slots": {
                    "fragments": base,
                    "chr_id": [0],
                    "start": [0],
                    "end": [5],
                    "chr_levels": ["chr1", "chr2"],
                    "invert_selection": [2],
                },
            },
            MatrixSourceError,
            "invert_selection must contain one logical value",
        ),
        "missing-version": (
            {
                "class": "UnpackedMemFragments",
                "slots": {},
            },
            MatrixSourceError,
            "has no version slot",
        ),
        "invalid-version": (
            {
                "class": "UnpackedMemFragments",
                "slots": {"version": ["custom-fragments-v1"]},
            },
            MatrixSourceError,
            "unsupported BPCells fragment format",
        ),
        "compression-conflict": (
            {
                "class": "PackedMemFragments",
                "slots": {"version": ["unpacked-fragments-v2"]},
            },
            MatrixSourceError,
            "cannot contain",
        ),
        "missing-directory": (
            {"class": "FragmentsDir", "slots": {}},
            MatrixSourceError,
            "has no dir slot",
        ),
        "missing-hdf5-group": (
            {
                "class": "FragmentsHDF5",
                "slots": {"path": ["fragments.h5"]},
            },
            MatrixSourceError,
            "requires path and group slots",
        ),
        "invalid-buffer-size": (
            {
                "class": "UnpackedMemFragments",
                "slots": base_slots,
            },
            MatrixSourceError,
            "must be a positive 32-bit integer",
        ),
        "invalid-prefix-utf8": (
            {
                "class": "CellPrefix",
                "slots": {"fragments": base, "prefix": [b"\xff"]},
            },
            MatrixSourceError,
            "not valid UTF-8",
        ),
        "invalid-group-number": (
            {
                "class": "CellMerge",
                "slots": {
                    "fragments": base,
                    "group_names": ["one", "two"],
                    "group_ids": [0.0, float("nan"), 1.0],
                },
            },
            MatrixSourceError,
            "invalid uint32 value",
        ),
        "nonnumeric-shift": (
            {
                "class": "ShiftFragments",
                "slots": {"fragments": base, "shift_start": ["a"], "shift_end": [0]},
            },
            TypeError,
            r"^integer value at \$@shift_start must be numeric$",
        ),
    }
    specification, error_type, message = cases[case]
    with pytest.raises(error_type, match=message):
        fragment_source_from_slots(specification)


def test_fragment_wrappers_enforce_shift_and_region_limits() -> None:
    base = fragment_source_from_slots(_memory_fragment_spec())
    shifted = fragment_source_from_slots(
        {
            "class": "ShiftFragments",
            "slots": {
                "fragments": base,
                "shift_start": [-1],
                "shift_end": [0],
            },
        }
    )
    with pytest.raises(MatrixSourceError, match="valid uint32 range"):
        tuple(shifted.iter_chromosome(0))

    with pytest.raises(ResourceLimitError, match="region metadata"):
        fragment_source_from_slots(
            {
                "class": "RegionSelect",
                "slots": {
                    "fragments": base,
                    "chr_id": [0, 0, 0, 0],
                    "start": [0, 2, 4, 6],
                    "end": [1, 3, 5, 7],
                    "chr_levels": ["chr1", "chr2"],
                },
            },
            limits=SourceLimits(maxMetadataBytes=64),
        )
    # Ten uint32 IDs need 40 bytes before any region is read.
    with pytest.raises(
        ResourceLimitError, match=r"^chr_id at \$ exceeds maxMetadataBytes=32$"
    ):
        fragment_source_from_slots(
            {
                "class": "RegionSelect",
                "slots": {
                    "fragments": base,
                    "chr_id": [0] * 10,
                    "start": [0] * 10,
                    "end": [1] * 10,
                    "chr_levels": ["chr1"],
                },
            },
            limits=SourceLimits(maxMetadataBytes=32),
        )
    # A 40-byte prefix adds 120 bytes to the source's 86 bytes of metadata.
    for maximum, accepted in ((206, True), (205, False)):
        prefixed_spec = {
            "class": "CellPrefix",
            "slots": {"fragments": base, "prefix": "x" * 40},
        }
        if accepted:
            prefixed = fragment_source_from_slots(
                prefixed_spec, limits=SourceLimits(maxMetadataBytes=maximum)
            )
            assert prefixed.cellNames[0] == "x" * 40 + "c1"
            continue
        with pytest.raises(
            ResourceLimitError,
            match=rf"^prefixed cell names at \$ exceed maxMetadataBytes={maximum}$",
        ):
            fragment_source_from_slots(
                prefixed_spec, limits=SourceLimits(maxMetadataBytes=maximum)
            )


def test_fragment_matrices_cover_tile_orientation_and_empty_reads() -> None:
    base = fragment_source_from_slots(_memory_fragment_spec())
    peak = matrix_source_from_slots(_fragment_matrix_spec(base))
    assert peak.shape == (1, 3)
    # An empty window still returns its one row pointer.
    assert peak.estimate_read_memory(0, 0).outputBytes == 8
    assert peak.read_cells(0, 0).shape == (0, 1)
    # Peak [0, 10) holds both ends of c1's [0, 10) and the start of c2's [5, 15).
    np.testing.assert_array_equal(peak.read_cells(0, 3).toarray(), [[2], [1], [0]])

    tile_spec = _fragment_matrix_spec(base, matrix_class="TileMatrix")
    tile_slots = tile_spec["slots"]
    assert isinstance(tile_slots, dict)
    tile_slots["start"] = np.asarray([10], dtype=np.int32)
    tile_slots["end"] = np.asarray([30], dtype=np.int32)
    tile_slots["dim"] = [4, 3]
    tile = matrix_source_from_slots(tile_spec)
    assert tile.shape == (4, 3)
    # Tiles [10, 15), [15, 20), [20, 25), [25, 30) count each fragment start and
    # each last base, end - 1.
    np.testing.assert_array_equal(
        tile.read_cells(0, 3).toarray(),
        [[1, 1, 0, 0], [1, 0, 1, 1], [1, 1, 0, 0]],
    )

    native_spec = _fragment_matrix_spec(base, matrix_class="TileMatrix")
    native_slots = native_spec["slots"]
    assert isinstance(native_slots, dict)
    native_slots["transpose"] = [0]
    native_slots["dim"] = [3, 2]
    native = matrix_source_from_slots(native_spec)
    assert native.shape == (3, 2)
    # Untransposed, each tile of [0, 5) and [5, 10) is a column over the cells.
    np.testing.assert_array_equal(
        native.read_cells(0, 2).toarray(), [[1, 0, 0], [1, 1, 0]]
    )


@pytest.mark.parametrize(
    "case",
    [
        "unequal-ranges",
        "reversed-range",
        "invalid-peak-mode",
        "invalid-tile-mode",
        "missing-tile-width",
        "tile-width-count",
        "zero-tile-width",
        "unsorted-peaks",
        "unsorted-tile-chromosomes",
        "overlapping-tile-ranges",
        "short-shape",
        "negative-shape",
        "nonnumeric-chromosome",
    ],
)
def test_fragment_matrix_factory_validates_materialized_ranges(case: str) -> None:
    base = fragment_source_from_slots(_memory_fragment_spec())

    def specification(
        matrix_class: str = "PeakMatrix",
        **overrides: object,
    ) -> dict[str, object]:
        result = _fragment_matrix_spec(base, matrix_class=matrix_class)
        slots = result["slots"]
        assert isinstance(slots, dict)
        slots.update(overrides)
        return result

    cases: dict[str, tuple[dict[str, object], type[Exception], str]] = {
        "unequal-ranges": (
            specification(end=np.asarray([10, 20], dtype=np.int32)),
            MatrixSourceError,
            "equal chr_id, start, and end lengths",
        ),
        "reversed-range": (
            specification(start=[10], end=[5]),
            MatrixSourceError,
            "end before its start",
        ),
        "invalid-peak-mode": (
            specification(mode=["custom"]),
            MatrixSourceError,
            "PeakMatrix mode.*is invalid",
        ),
        "invalid-tile-mode": (
            specification("TileMatrix", mode=["overlaps"]),
            MatrixSourceError,
            "TileMatrix mode.*is invalid",
        ),
        "missing-tile-width": (
            specification("TileMatrix", tile_width=None),
            MatrixSourceError,
            "has no tile_width slot",
        ),
        "tile-width-count": (
            specification("TileMatrix", tile_width=[5, 5]),
            MatrixSourceError,
            "requires one width per range",
        ),
        "zero-tile-width": (
            specification("TileMatrix", tile_width=[0]),
            MatrixSourceError,
            "contains a zero tile width",
        ),
        "unsorted-peaks": (
            specification(
                chr_id=[0, 0],
                start=[0, 0],
                end=[20, 10],
                dim=[2, 3],
            ),
            MatrixSourceError,
            "not sorted by",
        ),
        "unsorted-tile-chromosomes": (
            specification(
                "TileMatrix",
                chr_id=[1, 0],
                start=[0, 0],
                end=[10, 10],
                tile_width=[5, 5],
                dim=[4, 3],
            ),
            MatrixSourceError,
            "not sorted by chromosome",
        ),
        "overlapping-tile-ranges": (
            specification(
                "TileMatrix",
                chr_id=[0, 0],
                start=[0, 5],
                end=[10, 15],
                tile_width=[5, 5],
                dim=[4, 3],
            ),
            MatrixSourceError,
            "ranges.*overlap",
        ),
        "short-shape": (
            specification(dim=[1]),
            MatrixSourceError,
            "must contain two integers",
        ),
        "negative-shape": (
            specification(dim=[-1, 3]),
            MatrixSourceError,
            "cannot contain negative values",
        ),
        "nonnumeric-chromosome": (
            specification(chr_id=["chr1"]),
            TypeError,
            "must contain integers",
        ),
    }
    source_specification, error_type, message = cases[case]
    with pytest.raises(error_type, match=message):
        matrix_source_from_slots(source_specification)


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
        window = assay.counts.read_cells(7, 9)
        assert window.shape == (2, 17_195)
        np.testing.assert_array_equal(
            np.asarray(window.sum(axis=1)).ravel(), [5_008, 13_461]
        )
        _assert_counts_match_seurat_totals(reader)
        assert reader.cellMetadata.columnNames[:3] == (
            "orig.ident",
            "nCount_RNA",
            "nFeature_RNA",
        )
        assert reader.activeIdentity.levels == ("DC", "Mono CD14", "Mono FCGR3A")
        assert _decoded(reader.activeIdentity, 0, 3) == (
            "Mono CD14",
            "Mono CD14",
            "Mono CD14",
        )
        assert reader.get_reduction("pca").dimensions == (1_000, 50)
        loadings = reader.get_reduction("pca").featureLoadings
        assert loadings is not None
        assert loadings.shape == (2_000, 50)
        # The stored PCA is approximate (irlba), so its stdev matches to 1e-4.
        _assert_pca_matches_its_stdev_and_orthonormal_loadings(reader, rtol=1e-3)
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
        window = assay.counts.read_cells(11, 13)
        assert window.shape == (2, 500)
        np.testing.assert_array_equal(
            np.asarray(window.sum(axis=1)).ravel(), [1_448, 1_415]
        )
        _assert_counts_match_seurat_totals(reader)
        assert assay.cellMembership.allIncluded
        assert _decoded(reader.activeIdentity, 0, 3) == ("1", "0", "2")
        pca = reader.get_reduction("pca")
        assert pca.dimensions == (300, 20)
        assert pca.featureLoadings is not None
        assert pca.featureLoadings.shape == (200, 20)
        _assert_pca_matches_its_stdev_and_orthonormal_loadings(reader, rtol=1e-12)
        assert reader.get_reduction("umap").dimensions == (300, 2)
        assert reader.get_reduction("umap").role == "displayEmbedding"


def test_stream_reader_needs_a_sidecar_root_for_sidecar_layers(
    tmp_path: Path,
) -> None:
    path = _write_delayed_hdf5array_fixture(tmp_path / "sidecar.rds")
    payload = path.read_bytes()

    with SeuratReader(io.BytesIO(payload), reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "invalid_matrix"
        assert diagnostic.context == {"causeType": "UnsafeSidecarError"}
        assert "needs an anchor directory" in diagnostic.message

    with SeuratReader(
        io.BytesIO(payload),
        reductions=[],
        sidecar_root=tmp_path,
    ) as reader:
        np.testing.assert_array_equal(
            reader.get_assay("RNA").counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )


def _matrix_dir_node(wire: _Wire, directory: str, shape: tuple[int, int]) -> bytes:
    return wire.s4(
        [
            ("dir", wire.string_vector([directory])),
            ("compressed", wire.logical_vector([0])),
            ("buffer_size", wire.integer_vector([8192])),
            ("type", wire.string_vector(["uint32_t"])),
            ("dim", wire.real_vector([float(shape[0]), float(shape[1])])),
            ("transpose", wire.logical_vector([0])),
            ("dimnames", wire.vector([wire.nil(), wire.nil()])),
            ("class", wire.string_vector(["MatrixDir"])),
        ]
    )


def test_rename_dims_over_a_bpcells_directory_replaces_stale_names(
    tmp_path: Path,
) -> None:
    logical = np.asarray([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0]], dtype=np.uint32)
    # The directory keeps its original names, f0..f2 and c0..c3.
    _write_bpcells_directory(
        tmp_path / "counts",
        _bpcells_payload(logical, packed=False, version=2, storage_order="col"),
        version=2,
    )
    wire = _Wire()
    genes = ["g0", "g1", "g2"]
    cells = ["c1", "c2", "c3", "c4"]
    renamed = wire.s4(
        [
            ("matrix", _matrix_dir_node(wire, "counts", logical.shape)),
            ("dim", wire.real_vector([3.0, 4.0])),
            ("transpose", wire.logical_vector([0])),
            ("dimnames", wire.dimnames(genes, cells)),
            ("class", wire.string_vector(["RenameDims"])),
        ]
    )
    assay = wire.s4(
        [
            ("counts", renamed),
            ("meta.features", wire.data_frame([], 3)),
            ("class", wire.string_vector(["Assay"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame([("group", wire.string_vector(["a"] * 4))], cells),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1] * 4, ["cells"], names=cells)),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path = tmp_path / "renamed.rds"
    path.write_bytes(wire.document(root))

    with SeuratReader(path) as reader:
        assay_model = reader.get_assay("RNA")
        assert assay_model.featureIds == tuple(genes)
        assert assay_model.counts.row_names == tuple(genes)
        np.testing.assert_array_equal(
            assay_model.counts.read_cells(0, 4).toarray(), logical.T
        )


def test_assay5_logmap_names_place_layers_with_stale_sidecar_names(
    tmp_path: Path,
) -> None:
    logical = np.asarray([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0]], dtype=np.uint32)
    _write_bpcells_directory(
        tmp_path / "counts",
        _bpcells_payload(logical, packed=False, version=2, storage_order="col"),
        version=2,
    )
    wire = _Wire()
    genes = ["f0", "f1", "f2"]
    # merge(add.cell.ids = "A") renames only the LogMap and meta.data.
    cells = ["A_c0", "A_c1", "A_c2", "A_c3"]
    assay = wire.s4(
        [
            (
                "layers",
                wire.vector(
                    [_matrix_dir_node(wire, "counts", logical.shape)],
                    names=["counts"],
                ),
            ),
            ("cells", wire.logmap([1] * 4, cells, ["counts"])),
            ("features", wire.logmap([1] * 3, genes, ["counts"])),
            ("meta.data", wire.data_frame([], 3)),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([assay], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame([("group", wire.string_vector(["a"] * 4))], cells),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1] * 4, ["cells"], names=cells)),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path = tmp_path / "renamed-cells.rds"
    path.write_bytes(wire.document(root))

    with SeuratReader(path) as reader:
        assay_model = reader.get_assay("RNA")
        assert assay_model.cellIds == tuple(cells)
        assert assay_model.counts.column_names == tuple(cells)
        np.testing.assert_array_equal(
            assay_model.counts.read_cells(0, 4).toarray(), logical.T
        )


@pytest.mark.parametrize(
    ("levels", "expected_levels", "expected"),
    [
        (["", "A", "B"], ("", "A", "B"), ("A", "", "B")),
        ([None, "A", "B"], ("A", "B"), ("A", None, "B")),
    ],
)
def test_factor_levels_may_be_empty_or_missing(
    tmp_path: Path,
    levels: list[str | None],
    expected_levels: tuple[str, ...],
    expected: tuple[str | None, ...],
) -> None:
    wire = _Wire()
    path = _write_metadata_fixture(
        tmp_path / "levels.rds",
        wire=wire,
        columns=[("sample", wire.factor([2, 1, 3], levels))],  # type: ignore[arg-type]
    )

    with SeuratReader(path) as reader:
        column = reader.cellMetadata.column("sample")
        assert column.levels == expected_levels
        assert _decoded(column, 0, 3) == expected


def test_altrep_metadata_columns_are_expanded(tmp_path: Path) -> None:
    wire = _Wire()
    path = _write_metadata_fixture(
        tmp_path / "altrep.rds",
        wire=wire,
        columns=[
            # 1:3
            ("sequence", wire.altrep("compact_intseq", wire.real_vector([3, 1, 1]))),
            # seq(2.5, by = -1, length.out = 3)
            (
                "decreasing",
                wire.altrep("compact_realseq", wire.real_vector([3, 2.5, -1])),
            ),
            (
                "wrapped",
                wire.altrep(
                    "wrap_real",
                    wire.untagged_pair(
                        wire.real_vector([0.5, 1.5, 2.5]),
                        wire.integer_vector([0, 0]),
                    ),
                ),
            ),
            # as.character(c(7L, NA, 9L))
            (
                "deferred",
                wire.altrep(
                    "deferred_string",
                    wire.untagged_pair(
                        wire.integer_vector([7, R_INT_NA, 9]),
                        wire.integer_vector([0]),
                    ),
                ),
            ),
        ],
    )

    with SeuratReader(path) as reader:
        metadata = reader.cellMetadata
        assert [metadata.column(name).kind for name in metadata.columnNames] == [
            "integer",
            "real",
            "real",
            "character",
        ]
        np.testing.assert_array_equal(
            metadata.column("sequence").read_block(0, 3).values, [1, 2, 3]
        )
        np.testing.assert_array_equal(
            metadata.column("decreasing").read_block(1, 3).values, [1.5, 0.5]
        )
        np.testing.assert_array_equal(
            metadata.column("wrapped").read_block(0, 3).values, [0.5, 1.5, 2.5]
        )
        assert _decoded(metadata.column("deferred"), 0, 3) == ("7", None, "9")


@pytest.mark.parametrize(
    ("name", "state", "message"),
    [
        ("compact_intseq", [3, 1, 2], "state is invalid"),
        ("compact_intseq", [3, 1.5, 1], "state is invalid"),
        ("compact_realseq", [-1, 0, 1], "state is invalid"),
        ("compact_realseq", [2.5, 0, 1], "state is invalid"),
        ("compact_realseq", [3, float("nan"), 1], "state is invalid"),
        ("compact_bogus", [3, 1, 1], "is not supported"),
    ],
)
def test_invalid_altrep_metadata_columns_are_rejected(
    tmp_path: Path,
    name: str,
    state: list[float],
    message: str,
) -> None:
    wire = _Wire()
    path = _write_metadata_fixture(
        tmp_path / "altrep.rds",
        wire=wire,
        columns=[("bad", wire.altrep(name, wire.real_vector(state)))],
    )

    with pytest.raises(SeuratImportError, match=message) as error:
        with SeuratReader(path) as reader:
            reader.cellMetadata.column("bad").read_block(0, 1)
    assert error.value.code == "unsupported_altrep"


def test_altrep_sequence_views_read_bounded_windows() -> None:
    from types import SimpleNamespace

    from scarf.readers.seurat import _CompactSequence, _DeferredIntegerStrings

    sequence = _CompactSequence(5, 10.0, -1.0, np.dtype(np.int32))
    assert (len(sequence), sequence.nbytes) == (5, 20)
    np.testing.assert_array_equal(sequence.read_block(1, 4), [9, 8, 7])
    assert sequence[4] == 6
    for start, stop in ((-1, 2), (3, 2), (3, 6)):
        with pytest.raises(IndexError, match="outside"):
            sequence.read_block(start, stop)

    strings = _DeferredIntegerStrings(sequence)
    assert len(strings) == 5
    assert strings.read_block(0, 2) == ("10", "9")
    assert strings[2] == "8"
    missing = SimpleNamespace(
        read_block=lambda start, stop: np.array([1, R_INT_NA], dtype=np.int32)
    )
    assert _DeferredIntegerStrings(missing).read_block(0, 2) == ("1", None)  # type: ignore[arg-type]


@pytest.mark.parametrize("real", [False, True])
def test_missing_count_values_are_rejected(tmp_path: Path, real: bool) -> None:
    wire = _Wire()
    missing: int | float = float("nan") if real else R_INT_NA
    assay = wire.s4(
        [
            (
                "counts",
                wire.matrix(
                    [1, missing, 0, 2, 3, 0],  # type: ignore[list-item]
                    (2, 3),
                    rows=["g1", "g2"],
                    columns=["c1", "c2", "c3"],
                    real=real,
                ),
            ),
            ("meta.features", wire.data_frame([], ["g1", "g2"])),
            ("class", wire.string_vector(["Assay"])),
        ]
    )
    path = _write_single_assay_fixture(tmp_path / "missing.rds", wire=wire, assay=assay)

    with SeuratReader(path) as reader:
        counts = reader.get_assay("RNA").counts
        np.testing.assert_array_equal(counts.read_cells(1, 3), [[0, 2], [3, 0]])
        with pytest.raises(SeuratImportError) as error:
            counts.read_cells(0, 3)
    assert error.value.code == "missing_count_value"
    assert error.value.objectPath == "assays/RNA/counts"


def test_assay_counts_split_reads_that_exceed_the_block_limit(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "split.rds")
    with SeuratReader(
        path,
        assays=["RNA"],
        reductions=[],
        matrix_limits=SourceLimits(maxBlockBytes=20),
    ) as reader:
        counts = reader.get_assay("RNA").counts
        assert counts.estimate_read_memory(0, 3).blockBytes > 20
        np.testing.assert_array_equal(
            counts.read_cells(0, 3),
            [[1, 0], [0, 2], [3, 0]],
        )


def test_identifier_iteration_reads_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Values:
        def __init__(self) -> None:
            self.reads: list[tuple[int, int]] = []

        def __len__(self) -> int:
            return 5

        def read_block(self, start: int, stop: int) -> list[str]:
            self.reads.append((start, stop))
            return [f"c{index}" for index in range(start, stop)]

    class _Document:
        closed = False

    monkeypatch.setattr(seurat_module, "_VECTOR_BLOCK_SIZE", 2)
    values = _Values()
    vector = SeuratStringVector(
        values,  # type: ignore[arg-type]
        _Document(),  # type: ignore[arg-type]
        object_path="cells",
    )
    assert tuple(vector) == ("c0", "c1", "c2", "c3", "c4")
    assert values.reads == [(0, 2), (2, 4), (4, 5)]


def test_identifier_database_closes_its_connection_when_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class _Connection:
        def execute(self, statement: str) -> "_Connection":
            if statement.startswith("CREATE TABLE"):
                raise sqlite3.OperationalError("disk I/O error")
            return self

        def fetchone(self) -> tuple[int]:
            return (4096,)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(sqlite3, "connect", lambda path: _Connection())
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        seurat_module._identifier_database(
            scratch_dir=tmp_path,
            maximum_bytes=1024 * 1024,
            object_path="ids",
        )
    assert closed == [True]
    assert list(tmp_path.iterdir()) == []


def test_close_releases_the_document_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.readers._seurat.sources as sources_module

    def fail(source: object) -> None:
        raise OSError("cleanup failed")

    reader = SeuratReader(_write_fixture(tmp_path / "cleanup.rds"), reductions=[])
    document = reader.document
    monkeypatch.setattr(sources_module, "release_temporary_storage", fail)
    with pytest.raises(OSError, match="cleanup failed"):
        reader.close()
    assert document.closed


_CELLS = ["c1", "c2", "c3"]
_GENES = ["g1", "g2"]


def _seurat_document(
    wire: _Wire,
    *,
    assays: list[bytes] | None = None,
    assay_names: tuple[str, ...] = ("RNA",),
    metadata: bytes | None = None,
    active_assay: bytes | None = None,
    active_identity: bytes | None = None,
    reductions: bytes | None = None,
    extra_slots: tuple[tuple[str, bytes], ...] = (),
    classes: tuple[str, ...] = ("Seurat",),
) -> bytes:
    """Serialize a three-cell Seurat object; each argument replaces one slot."""
    slots = [
        (
            "assays",
            wire.vector(
                [_legacy_assay(wire)] if assays is None else assays,
                names=list(assay_names),
            ),
        ),
        (
            "meta.data",
            wire.data_frame([("group", wire.string_vector(["a", "b", "c"]))], _CELLS)
            if metadata is None
            else metadata,
        ),
        (
            "active.assay",
            wire.string_vector(["RNA"]) if active_assay is None else active_assay,
        ),
        (
            "active.ident",
            wire.factor([1, 1, 1], ["cells"], names=_CELLS)
            if active_identity is None
            else active_identity,
        ),
        (
            "reductions",
            wire.vector([], names=[]) if reductions is None else reductions,
        ),
        *extra_slots,
        ("class", wire.string_vector(list(classes))),
    ]
    return wire.document(wire.s4(slots))


def _write_document(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _assay_diagnostic(path: Path, name: str = "RNA", **options: object) -> object:
    with SeuratReader(path, reductions=[], **options) as reader:  # type: ignore[arg-type]
        inspection = reader.inspection.assay(name)
        assert not inspection.importable
        assert inspection.blockingDiagnostic is not None
        with pytest.raises(SeuratImportError) as error:
            reader.get_assay(name)
        assert error.value.code == inspection.blockingDiagnostic.code
        return inspection.blockingDiagnostic


def _custom_assay(wire: _Wire) -> bytes:
    return wire.s4([("class", wire.string_vector(["Custom"]))])


def _dgc(wire: _Wire, values: list[list[int]]) -> bytes:
    """Serialize a feature-by-cell dgCMatrix without dimnames."""
    matrix = csc_matrix(np.asarray(values, dtype=np.float64))
    return wire.s4(
        [
            ("i", wire.integer_vector(matrix.indices.tolist())),
            ("p", wire.integer_vector(matrix.indptr.tolist())),
            ("Dim", wire.integer_vector(list(matrix.shape))),
            ("x", wire.real_vector(matrix.data.tolist())),
            ("factors", wire.vector([])),
            ("class", wire.string_vector(["dgCMatrix"])),
        ]
    )


def _single_layer_assay(
    wire: _Wire,
    layer: bytes,
    *,
    cells: list[str] = _CELLS,
    genes: list[str] = _GENES,
) -> bytes:
    return wire.s4(
        [
            ("layers", wire.vector([layer], names=["counts"])),
            ("cells", wire.logmap([1] * len(cells), cells, ["counts"])),
            ("features", wire.logmap([1] * len(genes), genes, ["counts"])),
            ("meta.data", wire.data_frame([], len(genes))),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )


def _raw_char(wire: _Wire, raw: bytes | None, *, gp: int = 0) -> bytes:
    """A CHARSXP with explicit encoding flags; ``gp=2`` marks raw bytes."""
    if raw is None:
        return wire.integer(int(RType.CHAR)) + wire.integer(-1)
    return wire.integer(int(RType.CHAR) | (gp << 12)) + wire.integer(len(raw)) + raw


@pytest.mark.parametrize(
    ("case", "code", "object_path", "context"),
    [
        ("not-seurat", "not_seurat", "$", {"classNames": ("SingleCellExperiment",)}),
        ("empty-class", "empty_class", "$/class", {}),
        ("duplicate-assays", "duplicate_id", "assays/names/1", {"id": "RNA"}),
        (
            "missing-active-assay",
            "active_assay_missing",
            "active.assay",
            {"activeAssay": "ADT", "availableAssays": ("RNA",)},
        ),
        (
            "numeric-active-assay",
            "invalid_character_vector",
            "active.assay",
            {"rType": "INTEGER"},
        ),
        ("two-active-assays", "invalid_scalar", "active.assay", {"actualLength": 2}),
        ("vector-metadata", "invalid_data_frame", "meta.data", {}),
        ("unnamed-metadata", "missing_row_names", "meta.data/row.names", {}),
        ("compact-metadata", "invalid_row_names", "meta.data/row.names", {}),
        ("list-metadata", "invalid_data_frame", "meta.data", {"classNames": ("list",)}),
    ],
)
def test_root_structure_errors_stop_the_reader(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()
    group = [("group", wire.string_vector(["a", "b", "c"]))]
    unnamed_frame = wire.vector(
        [wire.string_vector(["a", "b", "c"])],
        attributes=[
            ("names", wire.string_vector(["group"])),
            ("class", wire.string_vector(["data.frame"])),
        ],
    )
    list_frame = wire.vector(
        [wire.string_vector(["a", "b", "c"])],
        attributes=[
            ("names", wire.string_vector(["group"])),
            ("row.names", wire.string_vector(_CELLS)),
            ("class", wire.string_vector(["list"])),
        ],
    )
    options: dict[str, object] = {
        "not-seurat": {"classes": ("SingleCellExperiment",)},
        "empty-class": {"classes": ()},
        "duplicate-assays": {
            "assays": [_legacy_assay(wire), _legacy_assay(wire)],
            "assay_names": ("RNA", "RNA"),
        },
        "missing-active-assay": {"active_assay": wire.string_vector(["ADT"])},
        "numeric-active-assay": {"active_assay": wire.integer_vector([1])},
        "two-active-assays": {"active_assay": wire.string_vector(["RNA", "RNA"])},
        "vector-metadata": {"metadata": wire.integer_vector([1, 2, 3])},
        "unnamed-metadata": {"metadata": unnamed_frame},
        # Cell metadata needs explicit identifiers, not R's compact c(NA, -3).
        "compact-metadata": {"metadata": wire.data_frame(group, 3)},
        "list-metadata": {"metadata": list_frame},
    }[case]
    path = _write_document(
        tmp_path / f"{case}.rds",
        _seurat_document(wire, **options),  # type: ignore[arg-type]
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)
    assert (error.value.code, error.value.objectPath, error.value.context) == (
        code,
        object_path,
        context,
    )


@pytest.mark.parametrize(
    ("case", "code", "object_path", "context"),
    [
        (
            "list-column",
            "unsupported_metadata_type",
            "meta.data/x",
            {"rType": "VECTOR", "classNames": ()},
        ),
        (
            "short-column",
            "length_mismatch",
            "meta.data/x",
            {"actual": 2, "expected": 3},
        ),
        ("real-factor", "invalid_factor", "meta.data/x", {"rType": "REAL"}),
        ("unleveled-factor", "missing_factor_levels", "meta.data/x/levels", {}),
        ("duplicate-levels", "duplicate_id", "meta.data/x/levels/1", {"id": "a"}),
        ("factor-code", "invalid_factor_value", "meta.data/x/1", {"value": 5}),
        ("logical-code", "invalid_logical_value", "meta.data/x/1", {"value": 2}),
    ],
)
def test_metadata_columns_are_validated_by_type(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()
    column = {
        "list-column": wire.vector([wire.nil(), wire.nil(), wire.nil()]),
        "short-column": wire.string_vector(["a", "b"]),
        "real-factor": wire.real_vector(
            [1.0, 1.0, 1.0],
            attributes=[
                ("levels", wire.string_vector(["a"])),
                ("class", wire.string_vector(["factor"])),
            ],
        ),
        "unleveled-factor": wire.integer_vector(
            [1, 1, 1], attributes=[("class", wire.string_vector(["factor"]))]
        ),
        "duplicate-levels": wire.factor([1, 1, 1], ["a", "a"]),
        "factor-code": wire.factor([1, 5, 1], ["a", "b"]),
        "logical-code": wire.logical_vector([1, 2, R_INT_NA]),
    }[case]
    path = _write_document(
        tmp_path / f"{case}.rds",
        _seurat_document(wire, metadata=wire.data_frame([("x", column)], _CELLS)),
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)
    assert (error.value.code, error.value.objectPath, error.value.context) == (
        code,
        object_path,
        context,
    )


@pytest.mark.parametrize(
    ("row", "code", "message"),
    [
        (None, "missing_id", "identifier is missing"),
        (b"", "missing_id", "identifier is empty"),
        (b"a\x00b", "invalid_id", "identifier contains a NUL character"),
        # Raw bytes that are not UTF-8.
        ((b"\xff", 2), "invalid_id_encoding", "identifier is not valid UTF-8"),
        # Unflagged bytes decode with escapes that are not valid Unicode.
        ((b"\xff", 0), "invalid_id_encoding", "identifier is not valid Unicode"),
    ],
)
def test_cell_identifiers_must_be_present_and_valid_text(
    tmp_path: Path,
    row: bytes | tuple[bytes, int] | None,
    code: str,
    message: str,
) -> None:
    wire = _Wire()
    raw, gp = row if isinstance(row, tuple) else (row, 0)
    row_names = (
        wire.integer(int(RType.STRING))
        + wire.integer(3)
        + _raw_char(wire, b"c1")
        + _raw_char(wire, raw, gp=gp)
        + _raw_char(wire, b"c3")
    )
    metadata = wire.vector(
        [wire.string_vector(["a", "b", "c"])],
        attributes=[
            ("names", wire.string_vector(["group"])),
            ("row.names", row_names),
            ("class", wire.string_vector(["data.frame"])),
        ],
    )
    path = _write_document(
        tmp_path / "ids.rds",
        _seurat_document(wire, assays=[_custom_assay(wire)], metadata=metadata),
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(path)
    assert error.value.code == code
    assert error.value.message == message
    assert error.value.objectPath == "meta.data/row.names/1"


def test_byte_encoded_cell_identifiers_are_decoded(tmp_path: Path) -> None:
    wire = _Wire()

    def identifiers() -> bytes:
        return (
            wire.integer(int(RType.STRING))
            + wire.integer(3)
            + _raw_char(wire, b"c1")
            + _raw_char(wire, "cé".encode(), gp=2)
            + _raw_char(wire, b"c3")
        )

    metadata = wire.vector(
        [wire.string_vector(["a", "b", "c"])],
        attributes=[
            ("names", wire.string_vector(["group"])),
            ("row.names", identifiers()),
            ("class", wire.string_vector(["data.frame"])),
        ],
    )
    identity = wire.integer_vector(
        [1, 1, 1],
        attributes=[
            ("levels", wire.string_vector(["cells"])),
            ("class", wire.string_vector(["factor"])),
            ("names", identifiers()),
        ],
    )
    path = _write_document(
        tmp_path / "bytes.rds",
        _seurat_document(
            wire,
            assays=[_custom_assay(wire)],
            metadata=metadata,
            active_identity=identity,
        ),
    )
    with SeuratReader(path) as reader:
        assert tuple(reader.cellIds) == ("c1", "cé", "c3")
        assert _decoded(reader.activeIdentity, 0, 3) == ("cells",) * 3


def test_identifier_indexes_respect_their_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    wire = _Wire()
    duplicated = _write_document(
        tmp_path / "duplicated.rds",
        _seurat_document(
            wire,
            assays=[_custom_assay(wire)],
            metadata=wire.data_frame(
                [("group", wire.string_vector(["a", "b", "c"]))], ["c1", "c1", "c3"]
            ),
        ),
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(duplicated)
    assert (error.value.code, error.value.objectPath, error.value.message) == (
        "duplicate_id",
        "meta.data/row.names",
        "identifiers are duplicated",
    )

    # 400 eleven-character identifiers outgrow two 4 KiB index pages.
    cells = [f"cell-{index:06d}" for index in range(400)]
    many = _write_document(
        tmp_path / "many.rds",
        wire.document(
            wire.s4(
                [
                    ("assays", wire.vector([_custom_assay(wire)], names=["RNA"])),
                    ("meta.data", wire.data_frame([], cells)),
                    ("active.assay", wire.string_vector(["RNA"])),
                    ("reductions", wire.vector([], names=[])),
                    ("class", wire.string_vector(["Seurat"])),
                ]
            )
        ),
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(many, matrix_limits=SourceLimits(maxMetadataBytes=8_192))
    assert (error.value.code, error.value.message) == (
        "metadata_index_limit",
        "identifier index exceeds its disk budget",
    )

    long_levels = _write_document(
        tmp_path / "levels.rds",
        _seurat_document(
            wire,
            metadata=wire.data_frame(
                [("x", wire.factor([1, 1, 1], ["x" * 9_000]))], _CELLS
            ),
        ),
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(long_levels, matrix_limits=SourceLimits(maxMetadataBytes=8_192))
    assert (error.value.code, error.value.objectPath, error.value.context) == (
        "metadata_index_limit",
        "meta.data/x/levels",
        {"requiredBytes": 9_008, "maximumBytes": 8_192},
    )

    monkeypatch.setattr(
        seurat_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=4_096)
    )
    with pytest.raises(SeuratImportError) as error:
        SeuratReader(_write_document(tmp_path / "small.rds", _seurat_document(wire)))
    assert error.value.code == "metadata_index_limit"
    assert error.value.message == "identifier index has insufficient scratch space"
    assert error.value.context == {"maximumBytes": 268_435_456, "freeBytes": 4_096}


@pytest.mark.parametrize(
    ("case", "code", "object_path"),
    [
        ("metadata-column", "active_identity_conflict", "active.ident"),
        ("character", "invalid_active_identity", "active.ident"),
        ("unnamed", "missing_active_identity_names", "active.ident/names"),
        ("unknown-cell", "metadata_id_conflict", "active.ident/names"),
    ],
)
def test_active_identity_problems_stay_local(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
) -> None:
    wire = _Wire()
    options = {
        "metadata-column": {
            "metadata": wire.data_frame(
                [("active.ident", wire.string_vector(["a", "b", "c"]))], _CELLS
            )
        },
        "character": {"active_identity": wire.string_vector(["a", "b", "c"])},
        "unnamed": {"active_identity": wire.factor([1, 1, 1], ["cells"])},
        "unknown-cell": {
            "active_identity": wire.factor(
                [1, 1, 1], ["cells"], names=["c1", "c2", "c9"]
            )
        },
    }[case]
    path = _write_document(tmp_path / f"{case}.rds", _seurat_document(wire, **options))
    with SeuratReader(path) as reader:
        assert reader.inspection.assay("RNA").importable
        diagnostic = reader.inspection.activeIdentity.blockingDiagnostic
        assert diagnostic is not None
        assert (diagnostic.code, diagnostic.objectPath) == (code, object_path)
        assert not reader.inspection.activeIdentity.importable
        with pytest.raises(SeuratImportError) as error:
            _ = reader.activeIdentity
        assert (error.value.code, error.value.objectPath) == (code, object_path)


def test_empty_root_slots_are_neither_imported_nor_reported(tmp_path: Path) -> None:
    wire = _Wire()
    path = _write_document(
        tmp_path / "empty-slots.rds",
        _seurat_document(
            wire,
            extra_slots=(
                ("misc", wire.nil()),
                ("tools", wire.nil()),
                ("commands", wire.vector([wire.integer_vector([1])], names=["x"])),
            ),
        ),
    )
    with SeuratReader(path) as reader:
        assert [
            (notice.code, notice.objectPath) for notice in reader.inspection.notices
        ] == [("ignored_seurat_slot", "commands")]


def _legacy_with(
    wire: _Wire,
    *,
    counts: bytes | None = None,
    features: bytes | None = None,
) -> bytes:
    return wire.s4(
        [
            (
                "counts",
                wire.matrix([1, 0, 0, 2, 3, 0], (2, 3), rows=_GENES, columns=_CELLS)
                if counts is None
                else counts,
            ),
            (
                "meta.features",
                wire.data_frame([("symbol", wire.string_vector(["G1", "G2"]))], _GENES)
                if features is None
                else features,
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )


@pytest.mark.parametrize(
    ("case", "code", "object_path", "context"),
    [
        (
            "custom-class",
            "unsupported_assay_class",
            "assays/RNA",
            {"classNames": ("Custom",)},
        ),
        ("no-dim", "missing_matrix_dimensions", "assays/RNA/counts/dim", {}),
        (
            "negative-dim",
            "invalid_matrix_dimensions",
            "assays/RNA/counts/dim",
            {"dimensions": (-2, 3)},
        ),
        (
            "real-dim",
            "invalid_integer_vector",
            "assays/RNA/counts/dim",
            {"rType": "REAL"},
        ),
        (
            "long-dim",
            "length_mismatch",
            "assays/RNA/counts/dim",
            {"actual": 3, "expected": 2},
        ),
        ("no-dimnames", "missing_dimnames", "assays/RNA/counts/Dimnames", {}),
        ("character-dimnames", "invalid_dimnames", "assays/RNA/counts/Dimnames", {}),
        (
            "three-dimnames",
            "invalid_dimnames",
            "assays/RNA/counts/Dimnames",
            {"actualLength": 3},
        ),
        ("null-axis", "missing_dimnames", "assays/RNA/counts/Dimnames/0", {}),
        (
            "long-axis",
            "dimnames_length_mismatch",
            "assays/RNA/counts/Dimnames",
            {"dimensions": (2, 3), "rowIds": 3, "columnIds": 3},
        ),
        ("cell-order", "assay_cell_id_conflict", "assays/RNA/counts/Dimnames/1", {}),
        (
            "character-counts",
            "unsupported_matrix_structure",
            "assays/RNA/counts",
            {"rType": "STRING", "classNames": ()},
        ),
        (
            "short-features",
            "length_mismatch",
            "assays/RNA/meta.features/row.names",
            {"actual": 1, "expected": 2},
        ),
        (
            "unknown-features",
            "metadata_id_conflict",
            "assays/RNA/meta.features/row.names",
            {"missing": ("g2",)},
        ),
        (
            "vector-features",
            "invalid_data_frame",
            "assays/RNA/meta.features",
            {"rType": "INTEGER"},
        ),
    ],
)
def test_legacy_assay_structure_is_validated(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()
    values = [1, 0, 0, 2, 3, 0]
    dimnames = ("dimnames", wire.dimnames(_GENES, _CELLS))

    def counts(*attributes: tuple[str, bytes]) -> bytes:
        return wire.integer_vector(values, attributes=list(attributes))

    shape = ("dim", wire.integer_vector([2, 3]))
    replaced = {
        "no-dim": {"counts": counts(dimnames)},
        "negative-dim": {
            "counts": counts(("dim", wire.integer_vector([-2, 3])), dimnames)
        },
        "real-dim": {"counts": counts(("dim", wire.real_vector([2.0, 3.0])), dimnames)},
        "long-dim": {
            "counts": counts(("dim", wire.integer_vector([2, 3, 1])), dimnames)
        },
        "no-dimnames": {"counts": counts(shape)},
        "character-dimnames": {
            "counts": counts(shape, ("dimnames", wire.string_vector(["g1", "g2"])))
        },
        "three-dimnames": {
            "counts": counts(
                shape,
                (
                    "dimnames",
                    wire.vector(
                        [
                            wire.string_vector(_GENES),
                            wire.string_vector(_CELLS),
                            wire.nil(),
                        ]
                    ),
                ),
            )
        },
        "null-axis": {
            "counts": counts(shape, ("dimnames", wire.dimnames(None, _CELLS)))
        },
        "long-axis": {
            "counts": counts(
                shape, ("dimnames", wire.dimnames(["g1", "g2", "g3"], _CELLS))
            )
        },
        "cell-order": {
            "counts": wire.matrix(
                values, (2, 3), rows=_GENES, columns=["c2", "c1", "c3"]
            )
        },
        "character-counts": {
            "counts": wire.string_vector(["a"] * 6, attributes=[shape, dimnames])
        },
        "short-features": {
            "features": wire.data_frame(
                [("symbol", wire.string_vector(["G1"]))], ["g1"]
            )
        },
        "unknown-features": {
            "features": wire.data_frame(
                [("symbol", wire.string_vector(["G1", "G3"]))], ["g1", "g3"]
            )
        },
        "vector-features": {
            "features": wire.integer_vector(
                [1, 2],
                attributes=[
                    ("row.names", wire.string_vector(_GENES)),
                    ("class", wire.string_vector(["data.frame"])),
                ],
            )
        },
    }
    assay = (
        _custom_assay(wire)
        if case == "custom-class"
        else _legacy_with(wire, **replaced[case])
    )
    path = _write_document(
        tmp_path / f"{case}.rds", _seurat_document(wire, assays=[assay])
    )
    diagnostic = _assay_diagnostic(path)
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (  # type: ignore[attr-defined]
        code,
        object_path,
        context,
    )


def test_feature_metadata_rows_follow_the_count_features(tmp_path: Path) -> None:
    wire = _Wire()
    # meta.features lists g2 before g1; columns are reordered to match counts.
    features = wire.data_frame(
        [
            ("symbol", wire.string_vector(["G2", "G1"])),
            ("rank", wire.integer_vector([2, 1])),
        ],
        ["g2", "g1"],
    )
    path = _write_document(
        tmp_path / "aligned.rds",
        _seurat_document(wire, assays=[_legacy_with(wire, features=features)]),
    )
    with SeuratReader(path) as reader:
        assay = reader.get_assay("RNA")
        metadata = assay.featureMetadata
        assert metadata.column("symbol").read_block(0, 2).values == ("G1", "G2")
        np.testing.assert_array_equal(
            metadata.column("rank").read_block(0, 2).values, [1, 2]
        )
        assert assay.cellMembership.allIncluded
        np.testing.assert_array_equal(
            assay.cellMembership.read_block(1, 3), [True, True]
        )
        with pytest.raises(
            IndexError, match=r"cell window \[2, 5\) is outside \[0, 3\)"
        ):
            assay.counts.read_cells(2, 5)


def _assay5_with(
    wire: _Wire,
    *,
    layers: list[bytes] | None = None,
    layer_names: tuple[str, ...] = ("counts",),
    cells: bytes | None = None,
    features: bytes | None = None,
) -> bytes:
    count = len(layer_names)
    return wire.s4(
        [
            (
                "layers",
                wire.vector(
                    [wire.matrix([1, 0, 0, 2, 3, 0], (2, 3))]
                    if layers is None
                    else layers,
                    names=list(layer_names),
                ),
            ),
            (
                "cells",
                wire.logmap([1] * 3 * count, _CELLS, list(layer_names))
                if cells is None
                else cells,
            ),
            (
                "features",
                wire.logmap([1] * 2 * count, _GENES, list(layer_names))
                if features is None
                else features,
            ),
            ("meta.data", wire.data_frame([], 2)),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )


@pytest.mark.parametrize(
    ("case", "code", "object_path", "context"),
    [
        ("plain-cells", "invalid_logmap", "assays/RNA/cells", {"classNames": ()}),
        ("integer-cells", "invalid_logmap", "assays/RNA/cells", {"rType": "INTEGER"}),
        (
            "short-names",
            "logmap_dimension_mismatch",
            "assays/RNA/cells/dim",
            {"dimensions": (3, 1), "rowIds": 2, "layerNames": 1},
        ),
        (
            "short-values",
            "logmap_dimension_mismatch",
            "assays/RNA/cells",
            {"values": 2, "dimensions": (3, 1)},
        ),
        ("cell-order", "assay_cell_order_conflict", "assays/RNA/cells/Dimnames/0", {}),
        ("no-counts", "counts_layer_missing", "assays/RNA/layers", {}),
        (
            "cells-lack-layer",
            "logmap_layer_missing",
            "assays/RNA/cells",
            {"layer": "counts"},
        ),
        (
            "features-lack-layer",
            "logmap_layer_missing",
            "assays/RNA/features",
            {"layer": "counts"},
        ),
        (
            "layer-shape",
            "layer_membership_dimension_mismatch",
            "assays/RNA/layers/counts",
            {"sourceDimensions": (2, 2), "featureMembership": 2, "cellMembership": 3},
        ),
    ],
)
def test_assay5_membership_is_validated(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()

    def cell_map(
        values: list[int], rows: list[str], r_type: RType = RType.LOGICAL
    ) -> bytes:
        return wire.atomic_vector(
            r_type,
            values,
            attributes=[
                ("dim", wire.integer_vector([3, 1])),
                ("class", wire.string_vector(["LogMap"])),
                ("dimnames", wire.dimnames(rows, ["counts"])),
            ],
        )

    options = {
        "plain-cells": {"cells": wire.logical_vector([1, 1, 1])},
        "integer-cells": {"cells": cell_map([1, 1, 1], _CELLS, RType.INTEGER)},
        "short-names": {"cells": cell_map([1, 1, 1], ["c1", "c2"])},
        "short-values": {"cells": cell_map([1, 1], _CELLS)},
        "cell-order": {"cells": wire.logmap([1, 1, 1], ["c3", "c1", "c2"], ["counts"])},
        "no-counts": {
            "layers": [wire.matrix([0.0] * 6, (2, 3), real=True)],
            "layer_names": ("data",),
        },
        "cells-lack-layer": {"cells": wire.logmap([1, 1, 1], _CELLS, ["other"])},
        "features-lack-layer": {"features": wire.logmap([1, 1], _GENES, ["other"])},
        "layer-shape": {"layers": [wire.matrix([1, 0, 0, 2], (2, 2))]},
    }[case]
    path = _write_document(
        tmp_path / f"{case}.rds",
        _seurat_document(wire, assays=[_assay5_with(wire, **options)]),  # type: ignore[arg-type]
    )
    diagnostic = _assay_diagnostic(path)
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (  # type: ignore[attr-defined]
        code,
        object_path,
        context,
    )


def test_assay5_sparse_reads_split_to_fit_the_block_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _Wire()
    path = _write_document(
        tmp_path / "split.rds", _seurat_document(wire, assays=[_assay5_with(wire)])
    )
    with SeuratReader(path, matrix_limits=SourceLimits(maxBlockBytes=300)) as reader:
        counts = reader.get_assay("RNA").counts
        # Three cells need 360 bytes; any two need 248.
        assert counts.estimate_read_memory(0, 3).blockBytes == 360
        assert counts.estimate_read_memory(1, 3).blockBytes == 248
        stitched = counts._source  # type: ignore[attr-defined]
        windows: list[tuple[int, int]] = []
        original = stitched.read_cells

        def tracked(start: int, stop: int) -> object:
            windows.append((start, stop))
            return original(start, stop)

        monkeypatch.setattr(stitched, "read_cells", tracked)
        block = counts.read_cells(0, 3)
    # Windows are halved only until each fits.
    assert windows == [(0, 1), (1, 3)]
    np.testing.assert_array_equal(block.toarray(), [[1, 0], [0, 2], [3, 0]])


@pytest.mark.parametrize(
    ("features", "required_bytes"),
    [
        # Positions of 100 cells (900 bytes) plus 150 feature and 100 cell
        # indexes exceed the budget before the layer is placed.
        (150, 4 * (900 + 150 * 8 + 100 * 8)),
        # One feature fits until the layer's global cell positions are added.
        (1, 4 * (900 + 8 + 100 * 8 + 100 * 8)),
    ],
)
def test_assay5_stitching_indexes_respect_the_metadata_budget(
    tmp_path: Path,
    features: int,
    required_bytes: int,
) -> None:
    wire = _Wire()
    cells = [f"c{index}" for index in range(100)]
    genes = [f"g{index}" for index in range(features)]
    assay = wire.s4(
        [
            (
                "layers",
                wire.vector(
                    [wire.matrix([1] * (features * 100), (features, 100))],
                    names=["counts"],
                ),
            ),
            ("cells", wire.logmap([1] * 100, cells, ["counts"])),
            ("features", wire.logmap([1] * features, genes, ["counts"])),
            ("meta.data", wire.data_frame([], features)),
            ("class", wire.string_vector(["Assay5"])),
        ]
    )
    payload = wire.document(
        wire.s4(
            [
                ("assays", wire.vector([assay], names=["RNA"])),
                ("meta.data", wire.data_frame([], cells)),
                ("active.assay", wire.string_vector(["RNA"])),
                ("reductions", wire.vector([], names=[])),
                ("class", wire.string_vector(["Seurat"])),
            ]
        )
    )
    path = _write_document(tmp_path / "budget.rds", payload)
    diagnostic = _assay_diagnostic(
        path, matrix_limits=SourceLimits(maxMetadataBytes=8_192)
    )
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (  # type: ignore[attr-defined]
        "metadata_index_limit",
        "assays/RNA/layers",
        {"requiredBytes": required_bytes, "maximumBytes": 8_192},
    )


def _reduction_with(
    wire: _Wire,
    *,
    embeddings: bytes | None = None,
    loadings: bytes | None = None,
    stdev: bytes | None = None,
    extra_slots: tuple[tuple[str, bytes], ...] = (),
    source_class: str = "DimReduc",
) -> bytes:
    slots = [
        (
            "cell.embeddings",
            wire.matrix(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                (3, 2),
                rows=_CELLS,
                columns=["PC_1", "PC_2"],
                real=True,
            )
            if embeddings is None
            else embeddings,
        ),
        ("assay.used", wire.string_vector(["RNA"])),
    ]
    if loadings is not None:
        slots.append(("feature.loadings", loadings))
    if stdev is not None:
        slots.append(("stdev", stdev))
    slots.extend(extra_slots)
    slots.append(("class", wire.string_vector([source_class])))
    return wire.s4(slots)


@pytest.mark.parametrize(
    ("case", "code", "object_path", "context"),
    [
        (
            "custom-class",
            "unsupported_reduction_class",
            "reductions/pca",
            {"classNames": ("Custom",)},
        ),
        (
            "short-names",
            "dimnames_length_mismatch",
            "reductions/pca/cell.embeddings/Dimnames",
            {"dimensions": (3, 2), "rowIds": 2, "columnIds": 2},
        ),
        (
            "cell-order",
            "reduction_cell_id_conflict",
            "reductions/pca/cell.embeddings/Dimnames/0",
            {},
        ),
        (
            "short-values",
            "matrix_length_mismatch",
            "reductions/pca/cell.embeddings",
            {},
        ),
        (
            "loading-components",
            "loading_component_id_conflict",
            "reductions/pca/feature.loadings/Dimnames/1",
            {},
        ),
        (
            "loading-features",
            "loading_feature_id_conflict",
            "reductions/pca/feature.loadings/Dimnames/0",
            {"missing": ("g9",)},
        ),
        (
            "long-stdev",
            "stdev_length_mismatch",
            "reductions/pca/stdev",
            {"actual": 3, "expected": 2},
        ),
        (
            "character-stdev",
            "invalid_atomic_vector",
            "reductions/pca/stdev",
            {"rType": "STRING"},
        ),
    ],
)
def test_reduction_structure_is_validated(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()
    components = ["PC_1", "PC_2"]
    options = {
        "custom-class": {"source_class": "Custom"},
        "short-names": {
            "embeddings": wire.matrix(
                [1.0] * 6, (3, 2), rows=["c1", "c2"], columns=components, real=True
            )
        },
        "cell-order": {
            "embeddings": wire.matrix(
                [1.0] * 6,
                (3, 2),
                rows=["c2", "c1", "c3"],
                columns=components,
                real=True,
            )
        },
        "short-values": {
            "embeddings": wire.real_vector(
                [1.0] * 5,
                attributes=[
                    ("dim", wire.integer_vector([3, 2])),
                    ("dimnames", wire.dimnames(_CELLS, components)),
                ],
            )
        },
        "loading-components": {
            "loadings": wire.matrix(
                [0.1, 0.2, 0.3, 0.4],
                (2, 2),
                rows=_GENES,
                columns=["PC_1", "PC_9"],
                real=True,
            )
        },
        "loading-features": {
            "loadings": wire.matrix(
                [0.1, 0.2, 0.3, 0.4],
                (2, 2),
                rows=["g1", "g9"],
                columns=components,
                real=True,
            )
        },
        "long-stdev": {"stdev": wire.real_vector([1.0, 2.0, 3.0])},
        "character-stdev": {"stdev": wire.string_vector(["a", "b"])},
    }[case]
    path = _write_document(
        tmp_path / f"{case}.rds",
        _seurat_document(
            wire,
            reductions=wire.vector([_reduction_with(wire, **options)], names=["pca"]),  # type: ignore[arg-type]
        ),
    )
    with SeuratReader(path) as reader:
        inspection = reader.inspection.reduction("pca")
        assert inspection.blockingDiagnostic is not None
        diagnostic = inspection.blockingDiagnostic
        assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (
            code,
            object_path,
            context,
        )
        with pytest.raises(SeuratImportError) as error:
            reader.get_reduction("pca")
        assert error.value.code == code


def test_reductions_resolve_their_assay_and_report_ignored_slots(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    reduction = _reduction_with(
        wire,
        extra_slots=(
            (
                "feature.loadings.projected",
                wire.matrix(
                    [0.1, 0.2], (1, 2), rows=["g1"], columns=["PC_1", "PC_2"], real=True
                ),
            ),
            ("misc", wire.vector([wire.integer_vector([1])], names=["x"])),
            ("jackstraw", wire.nil()),
        ),
    )
    reductions = wire.vector([reduction], names=["pca"])
    path = _write_document(
        tmp_path / "notices.rds",
        _seurat_document(
            wire,
            assays=[
                _legacy_assay(wire),
                _assay5(wire, invalid_membership=False, overlap=False),
            ],
            assay_names=("RNA", "ADT"),
            reductions=reductions,
        ),
    )
    # The reduction's legacy assay is built on demand although it was not selected.
    with SeuratReader(path, assays=["ADT"]) as reader:
        assert reader.assayNames == ("ADT",)
        pca = reader.get_reduction("pca")
        assert pca.assayUsed == "RNA"
        assert [(notice.code, notice.objectPath) for notice in pca.notices] == [
            ("ignored_projected_loadings", "reductions/pca/feature.loadings.projected"),
            ("ignored_reduction_slot", "reductions/pca/misc"),
        ]

    failed = _write_document(
        tmp_path / "failed-assay.rds",
        _seurat_document(wire, assays=[_custom_assay(wire)], reductions=reductions),
    )
    with SeuratReader(failed) as reader:
        diagnostic = reader.inspection.reduction("pca").blockingDiagnostic
        assert diagnostic is not None
        # The reduction reports the failure of the assay it depends on.
        assert (diagnostic.code, diagnostic.objectPath) == (
            "unsupported_assay_class",
            "assays/RNA",
        )


def test_unexpected_parse_errors_are_wrapped_as_item_diagnostics(
    tmp_path: Path,
) -> None:
    wire = _Wire()

    def with_integer_attributes(
        r_type: RType, values: list[int] | list[float]
    ) -> bytes:
        # An attribute list that is an integer vector, not a pairlist.
        encode = wire.real if r_type is RType.REAL else wire.integer
        return (
            wire.integer(wire.flags(r_type, attributes=True))
            + wire.integer(len(values))
            + b"".join(encode(value) for value in values)  # type: ignore[arg-type]
            + wire.integer_vector([1])
        )

    malformed_counts = _write_document(
        tmp_path / "counts.rds",
        _seurat_document(
            wire,
            assays=[
                _legacy_with(
                    wire,
                    counts=with_integer_attributes(RType.INTEGER, [1, 0, 0, 2, 3, 0]),
                )
            ],
        ),
    )
    diagnostic = _assay_diagnostic(malformed_counts)
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (  # type: ignore[attr-defined]
        "invalid_assay",
        "assays/RNA",
        {"causeType": "TypeError"},
    )

    malformed_reduction = _write_document(
        tmp_path / "reduction.rds",
        _seurat_document(
            wire,
            reductions=wire.vector(
                [
                    _reduction_with(
                        wire, embeddings=with_integer_attributes(RType.REAL, [1.0] * 6)
                    )
                ],
                names=["pca"],
            ),
        ),
    )
    with SeuratReader(malformed_reduction) as reader:
        diagnostic = reader.inspection.reduction("pca").blockingDiagnostic
        assert diagnostic is not None
        assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (
            "invalid_reduction",
            "reductions/pca",
            {"causeType": "TypeError"},
        )


_HDF5ARRAY_LOADER = (
    "function(x) HDF5Array::HDF5Array(filepath = x, name = 'counts', as.sparse = FALSE)"
)


def _cache_frame(
    wire: _Wire,
    rows: list[dict[str, str]],
    columns: tuple[str, ...] = ("layer", "path", "class", "pkg", "fxn", "assay"),
) -> bytes:
    return wire.data_frame(
        [(name, wire.string_vector([row[name] for row in rows])) for name in columns],
        len(rows),
    )


def _cache_row(**replaced: str) -> dict[str, str]:
    row = {
        "layer": "counts",
        "path": "counts.h5",
        "class": "DelayedMatrix",
        "pkg": "HDF5Array",
        "fxn": _HDF5ARRAY_LOADER,
        "assay": "RNA",
    }
    row.update(replaced)
    return row


def _cached_document(
    wire: _Wire,
    tools: bytes,
    *,
    in_object_counts: bool = False,
) -> bytes:
    if in_object_counts:
        assay = _assay5_with(wire)
    else:
        assay = wire.s4(
            [
                ("layers", wire.vector([], names=[])),
                ("cells", wire.logmap([], _CELLS, [])),
                ("features", wire.logmap([], _GENES, [])),
                ("meta.data", wire.data_frame([], 2)),
                ("class", wire.string_vector(["Assay5"])),
            ]
        )
    return _seurat_document(wire, assays=[assay], extra_slots=(("tools", tools),))


def _save_cache(wire: _Wire, frame: bytes) -> bytes:
    return wire.vector([frame], names=["SaveSeuratRds"])


def test_tools_without_a_save_cache_are_reported_as_ignored(tmp_path: Path) -> None:
    wire = _Wire()
    for name, tools in (
        ("unnamed", wire.vector([wire.integer_vector([1])])),
        ("other-tool", wire.vector([wire.integer_vector([1])], names=["Other"])),
    ):
        path = _write_document(
            tmp_path / f"{name}.rds",
            _cached_document(wire, tools, in_object_counts=True),
        )
        with SeuratReader(path) as reader:
            assert [
                (notice.code, notice.objectPath, notice.context)
                for notice in reader.inspection.notices
            ] == [("ignored_seurat_slot", "tools", {"rType": "VECTOR"})]
            np.testing.assert_array_equal(
                reader.get_assay("RNA").counts.read_cells(0, 3).toarray(),
                [[1, 0], [0, 2], [3, 0]],
            )


@pytest.mark.parametrize(
    ("case", "object_path", "context"),
    [
        ("list", "tools/SaveSeuratRds", {"classNames": ("list",)}),
        (
            "missing-columns",
            "tools/SaveSeuratRds",
            {"missingColumns": ("class", "pkg", "fxn", "assay")},
        ),
        (
            "uneven-columns",
            "tools/SaveSeuratRds",
            {
                "columnLengths": {
                    "layer": 1,
                    "path": 2,
                    "class": 1,
                    "pkg": 1,
                    "fxn": 1,
                    "assay": 1,
                }
            },
        ),
        (
            "duplicate-layer",
            "tools/SaveSeuratRds/1",
            {"assay": "RNA", "layer": "counts"},
        ),
    ],
)
def test_malformed_save_caches_block_only_assays_that_need_them(
    tmp_path: Path,
    case: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()
    row = _cache_row()
    frame = {
        "list": wire.vector(
            [wire.string_vector(["counts"])],
            attributes=[
                ("names", wire.string_vector(["layer"])),
                ("class", wire.string_vector(["list"])),
            ],
        ),
        "missing-columns": _cache_frame(wire, [row], columns=("layer", "path")),
        "uneven-columns": wire.data_frame(
            [
                ("layer", wire.string_vector(["counts"])),
                ("path", wire.string_vector(["a.h5", "b.h5"])),
                ("class", wire.string_vector(["DelayedMatrix"])),
                ("pkg", wire.string_vector(["HDF5Array"])),
                ("fxn", wire.string_vector([_HDF5ARRAY_LOADER])),
                ("assay", wire.string_vector(["RNA"])),
            ],
            1,
        ),
        "duplicate-layer": _cache_frame(wire, [row, row]),
    }[case]
    tools = _save_cache(wire, frame)
    # An assay with in-object counts does not depend on the broken cache.
    with SeuratReader(
        _write_document(
            tmp_path / "with-counts.rds",
            _cached_document(wire, tools, in_object_counts=True),
        )
    ) as reader:
        assert reader.inspection.assay("RNA").importable
        assert [
            (notice.code, notice.context) for notice in reader.inspection.notices
        ] == [("used_save_seurat_rds_cache", {"cachedAssays": (), "valid": False})]
    diagnostic = _assay_diagnostic(
        _write_document(tmp_path / "cache-only.rds", _cached_document(wire, tools))
    )
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (  # type: ignore[attr-defined]
        "invalid_sidecar_cache",
        object_path,
        context,
    )


def test_save_cache_rows_must_not_duplicate_layers_or_exceed_budgets(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    elsewhere = _write_document(
        tmp_path / "other-assay.rds",
        _cached_document(
            wire,
            _save_cache(wire, _cache_frame(wire, [_cache_row(assay="ATAC")])),
            in_object_counts=True,
        ),
    )
    with SeuratReader(elsewhere) as reader:
        # Rows for assays that the object lacks are ignored.
        assert [notice.context for notice in reader.inspection.notices] == [
            {"cachedAssays": (), "valid": True}
        ]
        assert reader.inspection.assay("RNA").importable

    duplicate = _assay_diagnostic(
        _write_document(
            tmp_path / "duplicate.rds",
            _cached_document(
                wire,
                _save_cache(wire, _cache_frame(wire, [_cache_row()])),
                in_object_counts=True,
            ),
        )
    )
    assert (duplicate.code, duplicate.context) == (  # type: ignore[attr-defined]
        "invalid_sidecar_cache",
        {"assay": "RNA", "duplicateLayers": ("counts",)},
    )

    oversized = _assay_diagnostic(
        _write_document(
            tmp_path / "oversized.rds",
            _cached_document(
                wire,
                _save_cache(wire, _cache_frame(wire, [_cache_row(path="x" * 9_000)])),
            ),
        ),
        matrix_limits=SourceLimits(maxMetadataBytes=8_192),
    )
    # The layer column already used 14 of the 8,192 bytes.
    assert (oversized.code, oversized.objectPath, oversized.context) == (  # type: ignore[attr-defined]
        "metadata_index_limit",
        "tools/SaveSeuratRds/path",
        {"requiredBytes": 9_008, "maximumBytes": 8_178},
    )


@pytest.mark.parametrize(
    ("path", "loader"),
    [
        ("counts", "function(x) BPCells::open_matrix_dir(dir = x)"),
        (
            "counts.h5",
            'function(x) BPCells::open_matrix_hdf5(path = x, group = "matrix")',
        ),
    ],
)
def test_save_cache_places_named_bpcells_layers_by_identifier(
    tmp_path: Path,
    path: str,
    loader: str,
) -> None:
    stored = np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.uint32)
    payload = _bpcells_payload(stored, packed=False, version=2, storage_order="col")
    # The sidecar stores its features and cells in another order.
    payload["row_names"] = ("g2", "g1")
    payload["col_names"] = ("c3", "c1", "c2")
    if path == "counts":
        _write_bpcells_directory(tmp_path / path, payload, version=2)
    else:
        _write_bpcells_hdf5(tmp_path / path, payload, version=2)
    wire = _Wire()
    row = _cache_row(
        path=path, pkg="BPCells", fxn=loader, **{"class": "IterableMatrix"}
    )
    source = _write_document(
        tmp_path / "cached.rds",
        _cached_document(wire, _save_cache(wire, _cache_frame(wire, [row]))),
    )
    with SeuratReader(source) as reader:
        assay = reader.get_assay("RNA")
        np.testing.assert_array_equal(
            assay.counts.read_cells(0, 3).toarray(),
            [[3, 0], [0, 2], [0, 1]],
        )
        assert [notice.code for notice in assay.notices] == [
            "restored_sidecar_cache_layer"
        ]


def _write_h5ad_sidecar(
    path: Path,
    values: list[list[int]] | np.ndarray,
    features: list[str],
    cells: list[str],
) -> None:
    with h5py.File(path, mode="w") as handle:
        handle.create_dataset("X", data=np.asarray(values, dtype=np.int32))
        for group_name, names in (("obs", cells), ("var", features)):
            group = handle.create_group(group_name)
            group.attrs["_index"] = "_index"
            group.create_dataset("_index", data=np.asarray(names, dtype="S"))


def test_save_cache_composites_unescape_recipes_and_validate_sidecar_ids(
    tmp_path: Path,
) -> None:
    _write_h5ad_sidecar(tmp_path / "one.h5ad", [[1, 0]], _GENES, ["c1"])
    _write_h5ad_sidecar(tmp_path / "two.h5ad", [[0, 2], [3, 0]], _GENES, ["c2", "c3"])
    _write_h5ad_sidecar(
        tmp_path / "three.h5ad", [[1, 0, 5]], ["g1", "g2", "g3"], ["c3"]
    )
    _write_h5ad_sidecar(
        tmp_path / "repeated.h5ad", [[1, 0], [0, 2], [3, 0]], ["g1", "g1"], _CELLS
    )
    many = [f"feature-{index:04d}" for index in range(400)]
    _write_h5ad_sidecar(tmp_path / "many.h5ad", np.zeros((3, 400)), many, _CELLS)
    # R deparses the inner double quotes of each recipe as \".
    escaped = 'function(x) BPCells::open_matrix_anndata_hdf5(path = x, group = \\"X\\")'
    plain = 'function(x) BPCells::open_matrix_anndata_hdf5(path = x, group = "X")'
    composite = _composite_cache_loader(f'"{escaped}", "{escaped}"')
    wire = _Wire()

    def document(name: str, path: str, loader: str) -> Path:
        row = _cache_row(
            path=path, pkg="BPCells", fxn=loader, **{"class": "IterableMatrix"}
        )
        return _write_document(
            tmp_path / f"{name}.rds",
            _cached_document(wire, _save_cache(wire, _cache_frame(wire, [row]))),
        )

    with SeuratReader(document("composite", "one.h5ad,two.h5ad", composite)) as reader:
        np.testing.assert_array_equal(
            reader.get_assay("RNA").counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )
    cases = (
        (
            document("unequal", "one.h5ad,three.h5ad", composite),
            {},
            (
                "invalid_matrix",
                "assays/RNA/layers/counts",
                {"causeType": "MatrixSourceError"},
            ),
            "cell bind sources must have equal feature counts",
        ),
        (
            document("repeated", "repeated.h5ad", plain),
            {},
            ("duplicate_id", "assays/RNA/layers/counts/Dimnames/0", {}),
            "identifiers are duplicated",
        ),
        (
            document("many", "many.h5ad", plain),
            {"matrix_limits": SourceLimits(maxMetadataBytes=8_192)},
            ("metadata_index_limit", "assays/RNA/layers/counts/Dimnames/0", {}),
            "identifier mapping exceeds its disk budget",
        ),
    )
    for path, options, expected, message in cases:
        diagnostic = _assay_diagnostic(path, **options)
        assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == expected  # type: ignore[attr-defined]
        assert diagnostic.message == message  # type: ignore[attr-defined]


def test_serialized_slots_convert_by_r_type(tmp_path: Path) -> None:
    wire = _Wire()
    # Features g1 = [1, 0, 3] and g2 = [0, 2, 0] over c1, c2, c3.
    stored = _dgc(wire, [[1, 0, 3], [0, 2, 0]])

    def counts(
        layer: bytes, *, cells: list[str] = _CELLS, **options: object
    ) -> np.ndarray:
        path = _write_document(
            tmp_path / "layer.rds",
            _seurat_document(
                wire, assays=[_single_layer_assay(wire, layer, cells=cells)]
            ),
        )
        with SeuratReader(path, reductions=[], **options) as reader:  # type: ignore[arg-type]
            block = reader.get_assay("RNA").counts.read_cells(0, 3)
        return block.toarray()

    # A NULL row selection keeps every feature.
    subset = wire.s4(
        [
            ("matrix", stored),
            ("row_selection", wire.nil()),
            ("col_selection", wire.integer_vector([2, 3])),
            ("zero_dims", wire.logical_vector([0, 0])),
            ("dim", wire.integer_vector([2, 2])),
            ("class", wire.string_vector(["MatrixSubset"])),
        ]
    )
    np.testing.assert_array_equal(
        counts(subset, cells=["c2", "c3"]), [[0, 0], [0, 2], [3, 0]]
    )
    # A matrix-shaped parameter is reshaped in R's column-major order.
    capped = wire.s4(
        [
            ("matrix", stored),
            (
                "row_params",
                wire.real_vector(
                    [2.0, 1.0], attributes=[("dim", wire.integer_vector([1, 2]))]
                ),
            ),
            ("transpose", wire.logical_vector([0])),
            ("class", wire.string_vector(["TransformMinByRow"])),
        ]
    )
    np.testing.assert_array_equal(counts(capped), [[1, 0], [0, 1], [2, 0]])
    # Named arguments arrive as a mapping: e1 = 10 makes 10 - x.
    subtracted = wire.s4(
        [
            ("seed", stored),
            ("OP", wire.builtin("-")),
            ("Largs", wire.vector([wire.real_vector([10.0])], names=["e1"])),
            ("Rargs", wire.vector([])),
            ("class", wire.string_vector(["DelayedUnaryIsoOpWithArgs"])),
        ]
    )
    np.testing.assert_array_equal(counts(subtracted), [[9, 10], [10, 8], [7, 10]])

    with h5py.File(tmp_path / "named.h5", mode="w") as handle:
        handle.create_dataset(
            "5", data=np.asarray([[1, 0], [0, 2], [3, 0]], dtype=np.int32)
        )
    # as.character(5L) serializes as a deferred string; a bare CHARSXP also works.
    for name in (
        wire.altrep(
            "deferred_string",
            wire.untagged_pair(wire.integer_vector([5]), wire.integer_vector([0])),
        ),
        wire.char("5"),
    ):
        seed = wire.s4(
            [
                ("filepath", wire.string_vector(["named.h5"])),
                ("name", name),
                ("class", wire.string_vector(["HDF5ArraySeed"])),
            ]
        )
        delayed = wire.s4(
            [("seed", seed), ("class", wire.string_vector(["DelayedMatrix"]))]
        )
        np.testing.assert_array_equal(counts(delayed), [[1, 0], [0, 2], [3, 0]])

    _write_h5ad_sidecar(
        tmp_path / "matrix.h5ad", [[1, 0], [0, 2], [3, 0]], _GENES, _CELLS
    )
    # An NA layer name selects the default matrix.
    h5ad_seed = wire.s4(
        [
            ("filepath", wire.string_vector(["matrix.h5ad"])),
            ("layer", wire.char(None)),
            ("class", wire.string_vector(["H5ADMatrixSeed"])),
        ]
    )
    np.testing.assert_array_equal(counts(h5ad_seed), [[1, 0], [0, 2], [3, 0]])

    def fragments(
        cell_ids: list[int], starts: list[int], ends: list[int], names: list[str]
    ) -> bytes:
        return wire.s4(
            [
                ("cell", wire.integer_vector(cell_ids)),
                ("start", wire.integer_vector(starts)),
                ("end", wire.integer_vector(ends)),
                ("end_max", wire.integer_vector([max(ends)])),
                ("chr_ptr", wire.real_vector([0.0, float(len(starts))])),
                ("chr_names", wire.string_vector(["chr1"])),
                ("cell_names", wire.string_vector(names)),
                ("version", wire.string_vector(["unpacked-fragments-v2"])),
                (
                    "class",
                    wire.string_vector(["UnpackedMemFragments", "IterableFragments"]),
                ),
            ]
        )

    merged = wire.s4(
        [
            (
                "fragments_list",
                wire.vector(
                    [
                        fragments([0, 0], [0, 4], [6, 9], ["c1"]),
                        fragments([0, 1], [2, 20], [8, 30], ["c2", "c3"]),
                    ]
                ),
            ),
            ("class", wire.string_vector(["MergeFragments", "IterableFragments"])),
        ]
    )
    peaks = wire.s4(
        [
            ("fragments", merged),
            ("chr_id", wire.integer_vector([0, 0])),
            ("start", wire.integer_vector([0, 10])),
            ("end", wire.integer_vector([10, 40])),
            ("chr_levels", wire.string_vector(["chr1"])),
            ("mode", wire.string_vector(["insertions"])),
            ("transpose", wire.logical_vector([1])),
            ("dim", wire.real_vector([2.0, 3.0])),
            ("class", wire.string_vector(["PeakMatrix", "IterableMatrix"])),
        ]
    )
    # Insertions: c1 has four ends in [0, 10), c2 two, and c3 two in [10, 40).
    np.testing.assert_array_equal(counts(peaks), [[4, 0], [2, 0], [0, 2]])


def test_renamed_layers_inherit_or_clear_sidecar_names(tmp_path: Path) -> None:
    stored = np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.uint32)
    payload = _bpcells_payload(stored, packed=False, version=2, storage_order="col")
    payload["row_names"] = ("f1", "f2")
    _write_bpcells_directory(tmp_path / "counts", payload, version=2)
    wire = _Wire()
    for rows, expected in (
        # DelayedArray stores -1L for an axis whose names the seed keeps.
        (wire.integer_vector([-1]), ("f1", "f2")),
        (wire.nil(), None),
    ):
        renamed = wire.s4(
            [
                ("matrix", _matrix_dir_node(wire, "counts", stored.shape)),
                ("dim", wire.real_vector([2.0, 3.0])),
                ("transpose", wire.logical_vector([0])),
                ("dimnames", wire.vector([rows, wire.string_vector(_CELLS)])),
                ("class", wire.string_vector(["RenameDims"])),
            ]
        )
        path = _write_document(
            tmp_path / "renamed.rds",
            _seurat_document(wire, assays=[_single_layer_assay(wire, renamed)]),
        )
        with SeuratReader(path, reductions=[]) as reader:
            counts = reader.get_assay("RNA").counts
            layer = counts._source.layers[0].source  # type: ignore[attr-defined]
            assert layer.row_names == expected
            assert layer.column_names == tuple(_CELLS)
            np.testing.assert_array_equal(counts.read_cells(0, 3).toarray(), stored.T)


@pytest.mark.parametrize(
    ("case", "code", "object_path", "context"),
    [
        (
            "three-dimensions",
            "invalid_matrix_parameter",
            "assays/RNA/layers/counts/row_params/dim",
            {},
        ),
        (
            "negative-dimension",
            "invalid_matrix_parameter",
            "assays/RNA/layers/counts/row_params/dim",
            {},
        ),
        (
            "value-count",
            "invalid_matrix_parameter",
            "assays/RNA/layers/counts/row_params",
            {},
        ),
        (
            "oversized",
            "metadata_index_limit",
            "assays/RNA/layers/counts/row_params",
            {"requiredBytes": 8_800, "maximumBytes": 8_192},
        ),
        (
            "system-call",
            "unsupported_matrix_function",
            "assays/RNA/layers/counts/OP",
            {"functionName": "system"},
        ),
        (
            "symbol",
            "unsupported_matrix_slot",
            "assays/RNA/layers/counts/OP",
            {"rType": "SYMBOL"},
        ),
        (
            "vector-fragments",
            "unsupported_fragment_structure",
            "assays/RNA/layers/counts/fragments/fragments_list/0",
            {"rType": "INTEGER", "classNames": ()},
        ),
        (
            "tsv-fragments",
            "unsupported_matrix",
            "assays/RNA/layers/counts/fragments",
            {"causeType": "UnsupportedMatrixOperation"},
        ),
    ],
)
def test_serialized_slots_with_unsupported_forms_are_rejected(
    tmp_path: Path,
    case: str,
    code: str,
    object_path: str,
    context: dict[str, object],
) -> None:
    wire = _Wire()
    stored = _dgc(wire, [[1, 0, 3], [0, 2, 0]])

    def capped(dimensions: list[int], values: list[float]) -> bytes:
        return wire.s4(
            [
                ("matrix", stored),
                (
                    "row_params",
                    wire.real_vector(
                        values, attributes=[("dim", wire.integer_vector(dimensions))]
                    ),
                ),
                ("class", wire.string_vector(["TransformMinByRow"])),
            ]
        )

    def peak(fragment_source: bytes) -> bytes:
        return wire.s4(
            [
                ("fragments", fragment_source),
                ("chr_id", wire.integer_vector([0])),
                ("start", wire.integer_vector([0])),
                ("end", wire.integer_vector([10])),
                ("chr_levels", wire.string_vector(["chr1"])),
                ("mode", wire.string_vector(["insertions"])),
                ("dim", wire.real_vector([1.0, 3.0])),
                ("class", wire.string_vector(["PeakMatrix"])),
            ]
        )

    def operation(op: bytes) -> bytes:
        return wire.s4(
            [
                ("seed", stored),
                ("OP", op),
                ("class", wire.string_vector(["DelayedUnaryIsoOpWithArgs"])),
            ]
        )

    layer = {
        "three-dimensions": lambda: capped([1, 2, 1], [2.0, 3.0]),
        "negative-dimension": lambda: capped([-1, 2], [2.0, 3.0]),
        "value-count": lambda: capped([2, 2], [2.0, 3.0]),
        "oversized": lambda: capped([1, 1_100], [1.0] * 1_100),
        "system-call": lambda: operation(wire.builtin("system")),
        "symbol": lambda: operation(wire.symbol("abs")),
        "vector-fragments": lambda: peak(
            wire.s4(
                [
                    ("fragments_list", wire.vector([wire.integer_vector([1])])),
                    (
                        "class",
                        wire.string_vector(["MergeFragments", "IterableFragments"]),
                    ),
                ]
            )
        ),
        "tsv-fragments": lambda: peak(
            wire.s4(
                [
                    ("path", wire.string_vector(["fragments.tsv.gz"])),
                    (
                        "class",
                        wire.string_vector(["FragmentsTsv", "IterableFragments"]),
                    ),
                ]
            )
        ),
    }[case]()
    path = _write_document(
        tmp_path / f"{case}.rds",
        _seurat_document(wire, assays=[_single_layer_assay(wire, layer)]),
    )
    diagnostic = _assay_diagnostic(
        path, matrix_limits=SourceLimits(maxMetadataBytes=8_192)
    )
    assert (diagnostic.code, diagnostic.objectPath, diagnostic.context) == (  # type: ignore[attr-defined]
        code,
        object_path,
        context,
    )


def test_delayed_array_over_an_ordinary_matrix_imports(tmp_path: Path) -> None:
    wire = _Wire()
    # DelayedArray(matrix(c(1L, 0L, 0L, 2L, 3L, 0L), 2)) keeps the matrix as its seed.
    layer = wire.s4(
        [
            ("seed", wire.matrix([1, 0, 0, 2, 3, 0], (2, 3))),
            ("class", wire.string_vector(["DelayedMatrix", "DelayedArray"])),
        ]
    )
    path = _write_document(
        tmp_path / "in-memory-seed.rds",
        _seurat_document(wire, assays=[_single_layer_assay(wire, layer)]),
    )
    with SeuratReader(path, reductions=[]) as reader:
        np.testing.assert_array_equal(
            reader.get_assay("RNA").counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )


def test_delayed_abind_of_ordinary_matrices_imports(tmp_path: Path) -> None:
    wire = _Wire()
    # cbind(DelayedArray(matrix(c(1L, 0L, 0L, 2L), 2)), matrix(c(3L, 0L), 2))
    # keeps both ordinary matrices in the seeds list of a DelayedAbind.
    bound = wire.s4(
        [
            (
                "seeds",
                wire.vector(
                    [wire.matrix([1, 0, 0, 2], (2, 2)), wire.matrix([3, 0], (2, 1))]
                ),
            ),
            ("along", wire.integer_vector([2])),
            ("class", wire.string_vector(["DelayedAbind"])),
        ]
    )
    layer = wire.s4(
        [
            ("seed", bound),
            ("class", wire.string_vector(["DelayedMatrix", "DelayedArray"])),
        ]
    )
    path = _write_document(
        tmp_path / "bound-in-memory-seeds.rds",
        _seurat_document(wire, assays=[_single_layer_assay(wire, layer)]),
    )
    with SeuratReader(path, reductions=[]) as reader:
        np.testing.assert_array_equal(
            reader.get_assay("RNA").counts.read_cells(0, 3).toarray(),
            [[1, 0], [0, 2], [3, 0]],
        )


def test_metadata_columns_reject_values_of_another_kind() -> None:
    from scarf.readers.seurat import _CompactSequence, _DeferredIntegerStrings

    class _Document:
        closed = False

    numbers = _CompactSequence(3, 1.0, 1.0, np.dtype(np.int32))
    strings = _DeferredIntegerStrings(numbers)

    def column(kind: str, values: object) -> SeuratMetadataColumn:
        return SeuratMetadataColumn(
            name="x",
            kind=kind,
            values=values,  # type: ignore[arg-type]
            length=3,
            document=_Document(),  # type: ignore[arg-type]
            object_path="meta.data/x",
        )

    assert column("character", strings).read_block(0, 3).values == ("1", "2", "3")
    np.testing.assert_array_equal(
        column("integer", numbers).read_block(0, 3).values, [1, 2, 3]
    )
    with pytest.raises(TypeError, match="^meta.data/x is not an atomic column$"):
        column("integer", strings).read_block(0, 1)
    with pytest.raises(TypeError, match="^meta.data/x is not a character column$"):
        column("character", numbers).read_block(0, 1)
    with pytest.raises(AssertionError, match="unknown metadata kind 'complex'"):
        column("complex", numbers).read_block(0, 1)


def test_logmap_membership_requires_its_layer_and_index_budget() -> None:
    from scarf.readers.seurat import _LogMap

    class _Values:
        # Two rows by two layers, column-major: a = [TRUE, FALSE], b = [TRUE, TRUE].
        def read_block(self, start: int, stop: int) -> np.ndarray:
            return np.asarray([1, 0, 1, 1], dtype=np.int32)[start:stop]

    values = _Values()
    logmap = _LogMap(
        rowIds=("c1", "c2"),
        layerNames=("a", "b"),
        values=values,  # type: ignore[arg-type]
        objectPath="assays/RNA/cells",
        maximumIndexBytes=24,
    )
    np.testing.assert_array_equal(logmap.membership("a"), [0])
    np.testing.assert_array_equal(logmap.membership("b"), [0, 1])
    with pytest.raises(SeuratImportError) as missing:
        logmap.membership("c")
    assert (missing.value.code, missing.value.context) == (
        "logmap_layer_missing",
        {"layer": "c"},
    )
    limited = _LogMap(
        rowIds=("c1", "c2"),
        layerNames=("a", "b"),
        values=values,  # type: ignore[arg-type]
        objectPath="assays/RNA/cells",
        maximumIndexBytes=23,
    )
    with pytest.raises(SeuratImportError) as budget:
        limited.membership("a")
    assert (budget.value.code, budget.value.objectPath, budget.value.context) == (
        "metadata_index_limit",
        "assays/RNA/cells/a",
        {"requiredBytes": 24, "maximumBytes": 23},
    )


def test_identifier_sources_must_return_the_requested_block() -> None:
    class _Truncating:
        def __len__(self) -> int:
            return 3

        def read_block(self, start: int, stop: int) -> tuple[str, ...]:
            return ("c1",)

    with pytest.raises(SeuratImportError) as error:
        seurat_module._identifier_block(_Truncating(), 0, 3, object_path="cells")  # type: ignore[arg-type]
    assert (error.value.code, error.value.context) == (
        "invalid_id_source",
        {"expected": 3, "actual": 1},
    )
