import gzip
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

import scarf.readers.seurat as seurat_module
from scarf.readers._rds import LazyAtomicVector, R_INT_NA, RdsClosedError, RType
from scarf.readers._seurat import (
    MatrixSourceError,
    ResourceLimitError,
    SourceLimits,
    UnsupportedMatrixOperation,
    fragment_source_from_slots,
    matrix_source_from_slots,
)
from scarf.readers.seurat import (
    SeuratImportError,
    SeuratMembership,
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
        assert reader.reductionNames == ()
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


def test_public_reader_containers_and_accessors_are_consistent(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path / "public-accessors.rds")
    with SeuratReader(
        path,
        assay_layers={"RNA": ["counts"]},
    ) as reader:
        assert not reader.closed
        assert reader.document.source.name == str(path)
        temp_paths = reader.tempPaths
        assert temp_paths == reader.document.temp_paths
        assert temp_paths
        assert all(Path(temp_path).exists() for temp_path in temp_paths)
        assert reader.inspect() is reader.inspection
        assert reader.assayNames == ("RNA", "ADT")
        assert reader.reductionNames == ("pca",)
        assert tuple(assay.name for assay in reader.assays) == ("RNA", "ADT")
        assert tuple(reduction.name for reduction in reader.reductions) == ("pca",)

        assert reader.cellIds.shape == (3,)
        assert reader.cellIds[-1] == "c3"
        assert reader.cellIds[::-1] == ("c3", "c2", "c1")
        assert tuple(reader.cellIds.iter_blocks(2)) == (("c1", "c2"), ("c3",))
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
        assert assay.matrix is assay.counts
        assert assay.dimensions == (2, 3)
        assert assay.counts.shape == assay.dimensions
        assert assay.counts.dtype == np.dtype(np.int32)
        assert not assay.counts.is_sparse
        assert assay.counts.zero_preserving
        assert assay.counts.resident_bytes >= 0
        assert assay.counts.row_names is None
        assert assay.counts.column_names is None
        assert assay.counts.estimate_read_memory(0, 1).peakBytes > 0

        reduction = reader.get_reduction("pca")
        assert reduction.stdev is not None
        assert len(reduction.stdev) == reduction.stdev.length == 2
        assert reduction.stdev.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(reduction.stdev.read_block(0, 2), [2.0, 1.0])
        np.testing.assert_array_equal(
            reduction.cellEmbeddings.read_cells(0, 1),
            [[1.0, 4.0]],
        )
        assert reader.activeIdentity.sourceIndices is not None
        np.testing.assert_array_equal(reader.activeIdentity.sourceIndices, [1, 2, 0])
        assert reader.cellMetadata.column("character").read_decoded_block(0, 3) == (
            "a",
            None,
            "c",
        )

    assert reader.closed
    assert all(not Path(temp_path).exists() for temp_path in temp_paths)
    with pytest.raises(RdsClosedError):
        _ = reader.cellIds[0]
    with pytest.raises(RdsClosedError):
        reader.cellMetadata.column("integer").read_block(0, 1)
    with pytest.raises(RdsClosedError):
        reduction.stdev.read_block(0, 1)


def test_public_sequence_bounds_and_membership_validation(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "public-bounds.rds")
    with SeuratReader(path) as reader:
        with pytest.raises(ValueError, match="block_size must be positive"):
            tuple(reader.cellIds.iter_blocks(0))
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
    np.testing.assert_array_equal(membership[::2], [False, False])
    assert membership[-1]
    np.testing.assert_array_equal(
        np.asarray(membership, dtype=np.uint8),
        [0, 1, 0, 1],
    )
    np.testing.assert_array_equal(
        np.array(membership, dtype=np.uint8, copy=True),
        [0, 1, 0, 1],
    )
    with pytest.raises(IndexError, match="membership window"):
        membership.read_block(0, 5)
    with pytest.raises(IndexError, match="membership index out of range"):
        _ = membership[4]
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
                "rowNames": ["f1", "f2"],
                "columnNames": ["c1", "c2"],
                "dim": [2, 2],
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
    assert row_bound.shape == (4, 2)
    assert column_bound.shape == (2, 4)
    assert row_bound.read_cells(0, 1).shape == (1, 4)
    assert column_bound.read_cells(0, 1).shape == (1, 2)

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
    converted = matrix_source_from_slots(
        {
            "class": ["ConvertMatrixType", "IterableMatrix"],
            "slots": {"matrix": leaf, "type": "float"},
        }
    )
    assert converted.dtype == np.dtype(np.float32)

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
    assert multiplied.read_cells(0, 2).shape == (2, 2)

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
    assert minimum.read_cells(0, 2).shape == (2, 2)
    np.testing.assert_array_equal(
        inactive_scale_shift.read_cells(0, 2),
        [[1, 3], [2, 4]],
    )


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
            "transpose slot must be scalar",
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
        "pearson-parameter-shape",
        "memory-version-conflict",
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
            "cannot be negative",
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
            "transpose slot must be scalar",
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
    assert chromosome.metadataBytes > base.metadataBytes
    assert sum(block.size for block in chromosome.iter_chromosome(0)) == 2
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
    assert cells.residentBytes > base.residentBytes
    assert cells.metadataBytes > base.metadataBytes
    selected_blocks = tuple(cells.iter_chromosome(0))
    assert selected_blocks
    assert all(np.all(block.cellIds < 2) for block in selected_blocks)

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
    assert renamed.metadataBytes > base.metadataBytes
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
    assert region.residentBytes > base.residentBytes
    assert region.metadataBytes > base.metadataBytes
    assert sum(block.size for block in region.iter_chromosome(0)) == 3

    merged = fragment_source_from_slots(
        {
            "class": ["MergeFragments", "IterableFragments"],
            "slots": {"fragments_list": [base, base]},
        }
    )
    assert merged.recordCount == 2 * base.recordCount
    assert merged.residentBytes == 2 * base.residentBytes
    assert merged.metadataBytes > 2 * base.metadataBytes
    assert merged.blockWorkingBytes == 2 * base.blockWorkingBytes
    merged_blocks = tuple(merged.iter_chromosome(0))
    assert len(merged_blocks) == 2
    assert int(merged_blocks[1].cellIds.min()) >= len(base.cellNames)
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
            TypeError,
            "must be TRUE or FALSE",
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
            "must be positive",
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


def test_fragment_matrices_cover_tile_orientation_and_empty_reads() -> None:
    base = fragment_source_from_slots(_memory_fragment_spec())
    peak = matrix_source_from_slots(_fragment_matrix_spec(base))
    assert peak.shape == (1, 3)
    assert peak.estimate_read_memory(0, 0).outputBytes > 0
    assert peak.read_cells(0, 0).shape == (0, 1)

    tile_spec = _fragment_matrix_spec(base, matrix_class="TileMatrix")
    tile_slots = tile_spec["slots"]
    assert isinstance(tile_slots, dict)
    tile_slots["start"] = np.asarray([10], dtype=np.int32)
    tile_slots["end"] = np.asarray([30], dtype=np.int32)
    tile_slots["dim"] = [4, 3]
    tile = matrix_source_from_slots(tile_spec)
    assert tile.shape == (4, 3)
    assert tile.read_cells(0, 3).shape == (3, 4)

    native_spec = _fragment_matrix_spec(base, matrix_class="TileMatrix")
    native_slots = native_spec["slots"]
    assert isinstance(native_slots, dict)
    native_slots["transpose"] = [0]
    native_slots["dim"] = [3, 2]
    native = matrix_source_from_slots(native_spec)
    assert native.shape == (3, 2)
    assert native.read_cells(0, 2).shape == (2, 3)


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
            "must contain two integers",
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
