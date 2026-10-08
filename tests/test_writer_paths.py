"""Edge paths of the H5AD, Seurat, sparse, Cell Ranger, subset, and CSV writers."""

import re
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.storage import LocalStore, MemoryStore

from scarf import DataStore
from scarf.readers import CSVReader, H5adReader
from scarf.readers._rds._types import RType
from scarf.readers.seurat import SeuratReader
from scarf.storage.artifacts import artifact_group
from scarf.storage.count_matrix import CountMatrixPolicy, plan_count_matrix_pair
from scarf.utils.count_values import CountValueRange
from scarf.writers import (
    CSVtoZarr,
    CrToZarr,
    H5adToZarr,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
)
from tests.test_seurat_reader import _Wire, _legacy_assay
from tests.test_writers import _write_h5ad

_CELLS = ["c1", "c2", "c3"]


def _sentinel_store() -> MemoryStore:
    store = MemoryStore()
    zarr.open_group(store=store, mode="w").create_group("sentinel")
    return store


def _untouched(store: MemoryStore) -> bool:
    return set(zarr.open_group(store=store, mode="r").group_keys()) == {"sentinel"}


def _store_bytes(store: MemoryStore) -> dict[str, bytes]:
    return {key: bytes(value.to_bytes()) for key, value in store._store_dict.items()}


def test_h5ad_analysis_assay_must_own_a_selected_output(tmp_path) -> None:
    source = _write_h5ad(tmp_path / "cells.h5ad", np.eye(3, dtype=np.uint8))
    with h5py.File(source, mode="r+") as h5:
        h5["obs"].create_dataset("clusters", data=np.array([0, 1, 0], dtype=np.int8))
    plain = H5adReader(str(source), feature_name_key="feature_name")
    clustered = H5adReader(
        str(source), feature_name_key="feature_name", cluster_keys=("clusters",)
    )
    destination = _sentinel_store()
    try:
        with pytest.raises(ValueError, match="requires at least one explicitly"):
            H5adToZarr(plain, zarr_loc=destination, analysis_assay="RNA")
        with pytest.raises(ValueError, match="must name an imported assay"):
            H5adToZarr(clustered, zarr_loc=destination, analysis_assay="ADT")
    finally:
        plain.close()
        clustered.close()
    assert _untouched(destination)


def test_h5ad_numeric_category_clusters_keep_missing_cells(tmp_path) -> None:
    counts = np.array([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.uint8)
    source = _write_h5ad(tmp_path / "scores.h5ad", counts)
    with h5py.File(source, mode="r+") as h5:
        score = h5["obs"].create_group("score")
        score.create_dataset("codes", data=np.array([0, -1, 1, 0], dtype=np.int8))
        score.create_dataset("categories", data=np.array([0.5, 1.5]))
    reader = H5adReader(
        str(source), feature_name_key="feature_name", cluster_keys=("score",)
    )
    destination = MemoryStore()
    try:
        # One-cell blocks leave the block of the unlabeled cell without values.
        result = H5adToZarr(reader, zarr_loc=destination, nthreads=1).dump(batch_size=1)
    finally:
        reader.close()

    labels = artifact_group(
        zarr.open_group(store=destination, mode="r"),
        result.clusterArtifacts["score"],
    )
    assert labels["values"].dtype == np.float64
    np.testing.assert_array_equal(labels["values"][:], [0.5, 0.0, 1.5, 0.5])
    np.testing.assert_array_equal(
        labels[labels["values"].attrs["missing_mask"]][:],
        [False, True, False, False],
    )


def test_h5ad_repeated_dump_reuses_its_artifacts(tmp_path) -> None:
    source = _write_h5ad(tmp_path / "twice.h5ad", np.eye(3, dtype=np.uint8))
    with h5py.File(source, mode="r+") as h5:
        leiden = h5["obs"].create_group("leiden")
        leiden.create_dataset("codes", data=np.array([0, 1, 1], dtype=np.int8))
        leiden.create_dataset("categories", data=np.array([b"a", b"b"]))
        h5.create_group("obsm").create_dataset(
            "X_umap", data=np.arange(6, dtype=np.float32).reshape(3, 2)
        )
    reader = H5adReader(
        str(source),
        feature_name_key="feature_name",
        embedding_roles={"X_umap": "umap"},
        cluster_keys=("leiden",),
    )
    try:
        writer = H5adToZarr(reader, zarr_loc=MemoryStore(), nthreads=1)
        first = writer.dump()
        second = writer.dump()
    finally:
        reader.close()

    assert second.cellSelection == first.cellSelection
    assert dict(second.clusterArtifacts) == dict(first.clusterArtifacts)
    assert dict(second.embeddingArtifacts) == dict(first.embeddingArtifacts)


def test_h5ad_dump_rejects_a_non_positive_batch_size(tmp_path) -> None:
    source = _write_h5ad(tmp_path / "cells.h5ad", np.eye(3, dtype=np.uint8))
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(reader, zarr_loc=MemoryStore(), nthreads=1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            writer.dump(batch_size=0)
    finally:
        reader.close()


def test_h5ad_cell_id_source_reads_ids_by_position(tmp_path) -> None:
    from scarf.writers.h5ad import _H5adCellIdSource

    def text(values) -> list[str]:
        return [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in values
        ]

    source = _write_h5ad(tmp_path / "ids.h5ad", np.eye(3, dtype=np.uint8))
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:
        ids = _H5adCellIdSource(reader)
        assert len(ids) == 3
        assert text([ids[0], ids[-1]]) == ["c0", "c2"]
        assert text(ids[1:3]) == ["c1", "c2"]
        assert text(ids) == ["c0", "c1", "c2"]
        with pytest.raises(IndexError):
            ids[3]
        with pytest.raises(ValueError, match="only contiguous reads"):
            ids[::2]
    finally:
        reader.close()


class _Connection:
    def __init__(self, on_send: Callable[[], object] | None = None) -> None:
        self.messages: list[tuple[str, object]] = []
        self.closed = False
        self._onSend = on_send

    def send(self, message: tuple[str, object]) -> None:
        self.messages.append(message)
        if self._onSend is not None:
            self._onSend()

    def close(self) -> None:
        self.closed = True


def test_h5ad_producer_sends_no_band_after_the_parent_stops(
    tmp_path, monkeypatch
) -> None:
    from scarf.writers import h5ad as h5ad_writer

    source = _write_h5ad(
        tmp_path / "bands.h5ad", np.arange(24, dtype=np.uint8).reshape(8, 3)
    )
    # 12 bytes of uint8 counts make bands of four cells.
    plan = plan_count_matrix_pair(
        8, 3, np.uint8, policy=CountMatrixPolicy(unitBytes=12, chunkBytes=12)
    )
    reader = H5adReader(str(source), feature_name_key="feature_name")
    try:

        def produce(connection: _Connection, stop: threading.Event) -> None:
            h5ad_writer._read_h5ad_process_window(
                reader._clone_kwargs(),
                4,
                0,
                8,
                {"RNA": plan.counts},
                ("RNA",),
                None,
                connection,
                stop,
            )

        # A stop that arrives while a band is sent ends the window after it.
        stop = threading.Event()
        connection = _Connection(on_send=stop.set)
        produce(connection, stop)
        assert [kind for kind, _payload in connection.messages] == ["band"]
        assert connection.closed

        # A stop that arrives while a band is cut keeps that band unsent.
        stop = threading.Event()
        cut_bands = h5ad_writer._count_matrix_bands

        def stop_while_cutting(*args):
            stop.set()
            yield from cut_bands(*args)

        monkeypatch.setattr(h5ad_writer, "_count_matrix_bands", stop_while_cutting)
        connection = _Connection()
        produce(connection, stop)
        assert connection.messages == []
        assert connection.closed
    finally:
        reader.close()


def _complex_matrix(wire: _Wire, values: list[float]) -> bytes:
    """Encode an R complex matrix of two genes and the three test cells."""
    return (
        wire.integer(wire.flags(RType.COMPLEX, attributes=True))
        + wire.integer(len(values))
        + b"".join(wire.real(value) + wire.real(0.0) for value in values)
        + wire.attributes(
            [
                ("dim", wire.integer_vector([2, len(_CELLS)])),
                ("dimnames", wire.dimnames(["g1", "g2"], _CELLS)),
                ("class", wire.string_vector(["matrix", "array"])),
            ]
        )
    )


def _write_seurat(
    path: Path,
    wire: _Wire,
    *,
    assay: bytes | None = None,
    cell_columns: list[tuple[str, bytes]] | None = None,
    reductions: list[tuple[str, bytes]] | None = None,
) -> Path:
    selected = reductions or []
    root = wire.s4(
        [
            ("assays", wire.vector([assay or _legacy_assay(wire)], names=["RNA"])),
            (
                "meta.data",
                wire.data_frame(
                    cell_columns or [("group", wire.string_vector(["a", "b", "c"]))],
                    _CELLS,
                ),
            ),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([1, 2, 1], ["x", "y"], names=_CELLS)),
            (
                "reductions",
                wire.vector(
                    [node for _name, node in selected],
                    names=[name for name, _node in selected],
                ),
            ),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _assay(wire: _Wire, counts: bytes, feature_column: str = "symbol") -> bytes:
    return wire.s4(
        [
            ("counts", counts),
            (
                "meta.features",
                wire.data_frame(
                    [(feature_column, wire.string_vector(["G1", "G2"]))],
                    ["g1", "g2"],
                ),
            ),
            ("class", wire.string_vector(["Assay"])),
        ]
    )


def test_seurat_complex_counts_fail_before_the_destination_exists(tmp_path) -> None:
    wire = _Wire()
    counts = _complex_matrix(wire, [1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
    path = _write_seurat(tmp_path / "complex.rds", wire, assay=_assay(wire, counts))
    destination = _sentinel_store()
    with SeuratReader(path, reductions=[]) as reader:
        with pytest.raises(TypeError, match="counts use unsupported dtype complex128"):
            SeuratToZarr(reader, destination, nthreads=1)
    assert _untouched(destination)


@pytest.mark.parametrize(
    ("cell_column", "feature_column", "message"),
    [
        (".", "symbol", "cell metadata column '.' cannot name a Zarr array"),
        ("group", "..", "RNA feature metadata column '..' cannot name a Zarr array"),
    ],
)
def test_seurat_metadata_names_must_name_an_array(
    tmp_path, cell_column, feature_column, message
) -> None:
    wire = _Wire()
    counts = wire.matrix([1, 0, 0, 2, 3, 0], (2, 3), rows=["g1", "g2"], columns=_CELLS)
    path = _write_seurat(
        tmp_path / "names.rds",
        wire,
        assay=_assay(wire, counts, feature_column),
        cell_columns=[(cell_column, wire.string_vector(["a", "b", "c"]))],
    )
    destination = _sentinel_store()
    with SeuratReader(path, reductions=[]) as reader:
        with pytest.raises(ValueError, match=re.escape(message)):
            SeuratToZarr(reader, destination, nthreads=1)
    assert _untouched(destination)


def test_seurat_repeated_dump_reuses_its_artifacts(tmp_path) -> None:
    wire = _Wire()
    pca = wire.s4(
        [
            (
                "cell.embeddings",
                wire.matrix(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    (3, 2),
                    rows=_CELLS,
                    columns=["PC_1", "PC_2"],
                    real=True,
                ),
            ),
            ("feature.loadings", wire.matrix([], (0, 0), real=True)),
            ("assay.used", wire.string_vector(["RNA"])),
            ("global", wire.logical_vector([0])),
            ("stdev", wire.real_vector([])),
            ("key", wire.string_vector(["PC_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
    )
    path = _write_seurat(tmp_path / "twice.rds", wire, reductions=[("pca", pca)])
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        writer = SeuratToZarr(reader, destination, nthreads=1)
        first = writer.dump()
        second = writer.dump()

    assert second.cellSelection == first.cellSelection
    assert second.activeIdentity == first.activeIdentity
    assert dict(second.reductionArtifacts) == dict(first.reductionArtifacts)
    # A reduction without stdev stores none.
    stored = artifact_group(
        zarr.open_group(store=destination, mode="r"),
        first.reductionArtifacts["pca"],
    )
    np.testing.assert_array_equal(stored["data"][:], [[1, 4], [2, 5], [3, 6]])
    assert "stdev" not in stored


def test_seurat_sparse_assay_without_cells_imports(tmp_path) -> None:
    wire = _Wire()
    counts = wire.s4(
        [
            ("i", wire.integer_vector([])),
            ("p", wire.integer_vector([0])),
            ("Dim", wire.integer_vector([2, 0])),
            ("Dimnames", wire.dimnames(["g1", "g2"], [])),
            ("x", wire.real_vector([])),
            ("factors", wire.vector([])),
            ("class", wire.string_vector(["dgCMatrix"])),
        ]
    )
    root = wire.s4(
        [
            ("assays", wire.vector([_assay(wire, counts)], names=["RNA"])),
            ("meta.data", wire.data_frame([("group", wire.string_vector([]))], [])),
            ("active.assay", wire.string_vector(["RNA"])),
            ("active.ident", wire.factor([], ["cells"], names=[])),
            ("reductions", wire.vector([], names=[])),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path = tmp_path / "empty.rds"
    path.write_bytes(wire.document(root))
    destination = MemoryStore()
    with SeuratReader(path) as reader:
        assert reader.get_assay("RNA").counts.is_sparse
        SeuratToZarr(reader, destination, nthreads=1).dump()

    counts_array = zarr.open_group(store=destination, mode="r")["RNA/counts"]
    assert (counts_array.shape, counts_array.dtype) == ((0, 2), np.uint8)


def test_sparse_writer_rejects_ids_that_do_not_match_the_matrix() -> None:
    matrix = csr_matrix(np.eye(3, dtype=np.uint8))
    destination = _sentinel_store()
    with pytest.raises(ValueError, match="Number of cell ids"):
        SparseToZarr(matrix, destination, ["c1", "c2"], ["f1", "f2", "f3"])
    with pytest.raises(ValueError, match="Number of feature ids"):
        SparseToZarr(matrix, destination, ["c1", "c2", "c3"], ["f1"])
    assert _untouched(destination)


def test_sparse_writer_batches_write_every_cell() -> None:
    values = np.arange(1, 16, dtype=np.uint8).reshape(5, 3)
    store = MemoryStore()
    writer = SparseToZarr(
        csr_matrix(values),
        store,
        [f"c{index}" for index in range(5)],
        ["f1", "f2", "f3"],
        nthreads=1,
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        writer.dump(batch_size=0)
    writer.dump(batch_size=2)

    # Batches of two cells leave one cell for the last batch.
    assert writer._lastImportPlan.batchRows == 2
    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    np.testing.assert_array_equal(counts[:], values)


class _CountsReader:
    """A Cell Ranger style reader of small in-memory counts."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        assay: str = "RNA",
        cell_columns: tuple[tuple[str, np.ndarray], ...] = (),
        feature_columns: tuple[tuple[str, np.ndarray], ...] = (),
    ) -> None:
        self._values = values
        self.nCells, self.nFeatures = values.shape
        self.matrix_dtype = values.dtype
        self.assayFeats = pd.DataFrame(
            {assay: ["Gene Expression", 0, self.nFeatures, self.nFeatures]},
            index=["type", "start", "end", "nFeatures"],
        )
        self._cellColumns = cell_columns
        self._featureColumns = feature_columns

    def cell_names(self) -> list[str]:
        return [f"c{index}" for index in range(self.nCells)]

    def feature_ids(self, assay_name: str) -> list[str]:
        return [f"f{index}" for index in range(self.nFeatures)]

    def feature_names(self, assay_name: str) -> list[str]:
        return [f"g{index}" for index in range(self.nFeatures)]

    def consume(self, batch_size: int, lines_in_mem: int) -> Iterator[coo_matrix]:
        for start in range(0, self.nCells, batch_size):
            yield coo_matrix(self._values[start : start + batch_size])

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        value_range = CountValueRange()
        value_range.update(self._values)
        return [value_range]

    def max_window_nnz(self, window_rows: int) -> int:
        return int(np.count_nonzero(self._values))

    def producer_staging_bytes(self, batch_size: int, lines_in_mem: int) -> int:
        return 0

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._cellColumns

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._featureColumns


def test_cellranger_assay_types_must_name_imported_assays_and_presets() -> None:
    reader = _CountsReader(np.eye(3, dtype=np.uint8))
    destination = _sentinel_store()
    with pytest.raises(ValueError, match="names assays that are not imported: GEX"):
        CrToZarr(reader, zarr_loc=destination, assay_types={"GEX": "RNA"})
    # The error names the assay, as every writer's does.
    with pytest.raises(ValueError, match="assay_type 'rna' of assay 'RNA' is not a"):
        CrToZarr(reader, zarr_loc=destination, assay_types={"RNA": "rna"})
    assert _untouched(destination)


def test_cellranger_assay_type_makes_a_custom_assay_rna() -> None:
    values = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint8)
    store = MemoryStore()
    writer = CrToZarr(
        _CountsReader(values, assay="GEX"),
        zarr_loc=store,
        nthreads=1,
        assay_types={"GEX": "RNA"},
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        writer.dump(batch_size=0)
    writer.dump()

    root = zarr.open_group(store=store, mode="r")
    assert root.attrs["assayTypes"] == {"GEX": "RNA"}
    np.testing.assert_array_equal(root["GEX/countsT"][:], values.T)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (
            {"cell_columns": (("batch", np.array(["a", "b"])),)},
            r"Cell metadata column 'batch' has shape \(2,\); expected \(3,\)",
        ),
        (
            {"feature_columns": (("tag", np.zeros((3, 2))),)},
            r"Feature metadata column 'tag' has shape \(3, 2\); expected \(3,\)",
        ),
    ],
)
def test_cellranger_reader_metadata_must_align_with_its_axis(columns, message) -> None:
    reader = _CountsReader(np.eye(3, dtype=np.uint8), **columns)
    with pytest.raises(ValueError, match=message):
        CrToZarr(reader, zarr_loc=MemoryStore(), nthreads=1)


_IMPORTED = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint8)

type _Importer = Callable[[Any, ExitStack, Path, bool], Any]


def _sparse(destination: Any, _stack: ExitStack, _tmp: Path, overwrite: bool) -> Any:
    return SparseToZarr(
        csr_matrix(_IMPORTED),
        destination,
        _CELLS,
        ["f1", "f2", "f3"],
        nthreads=1,
        overwrite=overwrite,
    )


def _csv(destination: Any, _stack: ExitStack, tmp: Path, overwrite: bool) -> Any:
    path = tmp / "counts.csv"
    path.write_text("g1,g2,g3\n1,0,2\n0,3,0\n4,0,5\n")
    return CSVtoZarr(
        CSVReader(str(path)),
        destination,
        assay_name="RNA",
        nthreads=1,
        overwrite=overwrite,
    )


def _cellranger(
    destination: Any, _stack: ExitStack, _tmp: Path, overwrite: bool
) -> Any:
    return CrToZarr(
        _CountsReader(_IMPORTED), zarr_loc=destination, nthreads=1, overwrite=overwrite
    )


def _h5ad(destination: Any, stack: ExitStack, tmp: Path, overwrite: bool) -> Any:
    reader = H5adReader(
        str(_write_h5ad(tmp / "counts.h5ad", _IMPORTED)),
        feature_name_key="feature_name",
    )
    stack.callback(reader.close)
    return H5adToZarr(reader, zarr_loc=destination, nthreads=1, overwrite=overwrite)


def _seurat(destination: Any, stack: ExitStack, tmp: Path, overwrite: bool) -> Any:
    path = _write_seurat(tmp / "counts.rds", _Wire())
    reader = stack.enter_context(SeuratReader(path, reductions=[]))
    return SeuratToZarr(reader, destination, nthreads=1, overwrite=overwrite)


_IMPORTERS: dict[str, _Importer] = {
    "sparse": _sparse,
    "csv": _csv,
    "cellranger": _cellranger,
    "h5ad": _h5ad,
    "seurat": _seurat,
}


def _unread(*_args: Any) -> Any:
    raise AssertionError("the source counts were read before the destination check")


@pytest.mark.parametrize("importer", sorted(_IMPORTERS))
def test_importers_replace_a_store_only_with_overwrite(
    tmp_path, monkeypatch, importer
) -> None:
    # Before, an import opened its destination with mode "w", which deleted
    # whatever it held.
    destination = MemoryStore()
    with ExitStack() as stack:
        _sparse(destination, stack, tmp_path, False).dump()
    before = _store_bytes(destination)
    with (
        ExitStack() as stack,
        monkeypatch.context() as patch,
        pytest.raises(FileExistsError, match="is not empty"),
    ):
        # Importers that pass over their source refuse the destination first.
        for writer in ("cellranger", "h5ad", "seurat"):
            patch.setattr(f"scarf.writers.{writer}.count_storage_dtype", _unread)
        _IMPORTERS[importer](destination, stack, tmp_path, False)
    assert _store_bytes(destination) == before

    with ExitStack() as stack:
        _IMPORTERS[importer](destination, stack, tmp_path, True).dump()
    root = zarr.open_group(store=destination, mode="r")
    assert root["RNA/counts"].attrs["complete"] is True


_SUBSET_COUNTS = np.array(
    [[1, 4, 9], [2, 20, 3], [12, 2, 2], [5, 0, 1]], dtype=np.uint16
)


def _subset_source(location: str | MemoryStore) -> DataStore:
    SparseToZarr(
        csr_matrix(_SUBSET_COUNTS),
        location,
        ["c1", "c2", "c3", "c4"],
        ["f1", "f2", "f3"],
        nthreads=1,
    ).dump()
    return DataStore(location, default_assay="RNA", min_features_per_cell=0, nthreads=1)


def test_subset_refuses_its_source_even_when_it_is_not_prepared(tmp_path) -> None:
    """The overlap check protects a source that no DataStore prepared.

    An assay built directly over imported counts is not prepared, so the
    refusal to replace a prepared store does not protect it.
    """
    on_disk = str(tmp_path / "source.zarr")
    dataset = _subset_source(on_disk)
    dataset.z["RNA"].attrs["prepared"] = False
    for destination in (on_disk, LocalStore(on_disk)):
        with pytest.raises(ValueError, match="overlaps a source store"):
            SubsetZarr(
                destination,
                assays=[dataset.RNA],
                cell_idx=np.array([0]),
                overwrite_existing_file=True,
                nthreads=1,
            )
    np.testing.assert_array_equal(dataset.RNA.rawData.compute(), _SUBSET_COUNTS)


def test_subset_refuses_a_destination_that_overlaps_its_source(tmp_path) -> None:
    on_disk = str(tmp_path / "source.zarr")
    dataset = _subset_source(on_disk)
    with pytest.raises(ValueError, match="overlaps a source store"):
        SubsetZarr(
            f"{on_disk}/nested.zarr",
            assays=[dataset.RNA],
            cell_idx=np.array([0]),
            nthreads=1,
        )

    in_memory = MemoryStore()
    dataset = _subset_source(in_memory)
    with pytest.raises(ValueError, match="overlaps a source store"):
        SubsetZarr(
            in_memory,
            assays=[dataset.RNA],
            cell_idx=np.array([0]),
            overwrite_existing_file=True,
            nthreads=1,
        )
    # Before, a second store object over the source's keys was replaced, which
    # emptied the source. The source is prepared, so overwrite refuses it.
    with pytest.raises(FileExistsError, match=r"holds the prepared assays \['RNA'\]"):
        SubsetZarr(
            MemoryStore(in_memory._store_dict),
            assays=[dataset.RNA],
            cell_idx=np.array([0]),
            overwrite_existing_file=True,
            nthreads=1,
        )
    np.testing.assert_array_equal(dataset.RNA.rawData.compute(), _SUBSET_COUNTS)


@pytest.mark.parametrize(
    ("cell_idx", "message"),
    [
        # A repeated index would write a store whose cell IDs repeat.
        (np.array([1, 1]), "cannot contain duplicate indices"),
        # An empty index used to fail inside Python's max(), and an empty
        # list was reported as an array of the wrong dtype.
        (np.array([], dtype=np.int64), "cannot be empty"),
        ([], "cannot be empty"),
        # A negative index wrapped around, so -4 repeated the first cell.
        (np.array([0, -4]), "cannot contain negative indices"),
    ],
    ids=["repeated", "empty", "empty-list", "negative"],
)
def test_subset_rejects_invalid_cell_indices(cell_idx, message) -> None:
    dataset = _subset_source(MemoryStore())
    destination = _sentinel_store()
    with pytest.raises(ValueError, match=message):
        SubsetZarr(
            destination,
            assays=[dataset.RNA],
            cell_idx=cell_idx,
            overwrite_existing_file=True,
            nthreads=1,
        )
    assert _untouched(destination)


def test_subset_by_a_cell_key_without_selected_cells_writes_no_cells() -> None:
    # A key is data, not an index argument: one that selects no cells writes
    # a store without cells, as an import of a source without cells does.
    dataset = _subset_source(MemoryStore())
    dataset.cells.insert("none", np.zeros(4, dtype=bool))
    destination = MemoryStore()
    SubsetZarr(destination, assays=[dataset.RNA], cell_key="none", nthreads=1).dump()

    root = zarr.open_group(store=destination, mode="r")
    assert root["cellData/ids"].shape == (0,)
    assert root["RNA/counts"].shape == (0, 3)
    assert root["RNA/countsT"].shape == (3, 0)


@pytest.mark.parametrize(
    ("reset", "expected"),
    [(True, [True, True, True]), (False, [True, False, True])],
)
def test_subset_keeps_the_source_cell_filter_unless_reset(reset, expected) -> None:
    dataset = _subset_source(MemoryStore())
    dataset.cells.update_key(np.array([True, False, True, False]), key="I")
    destination = MemoryStore()
    SubsetZarr(
        destination,
        assays=[dataset.RNA],
        cell_idx=np.array([0, 1, 2]),
        reset_cell_filter=reset,
        nthreads=1,
    ).dump()

    cells = zarr.open_group(store=destination, mode="r")["cellData"]
    np.testing.assert_array_equal(cells["I"][:], expected)
    np.testing.assert_array_equal(cells["ids"][:], ["c1", "c2", "c3"])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (2, "Dense stream contains 2 rows, expected 4"),
        (5, "Dense stream contains more rows than its destination"),
    ],
)
def test_csv_dump_refuses_a_file_whose_rows_changed(tmp_path, rows, message) -> None:
    path = tmp_path / "counts.csv"

    def write(n_rows: int) -> None:
        path.write_text(
            "g1,g2\n" + "".join(f"{row},{row + 1}\n" for row in range(n_rows))
        )

    write(4)
    destination = str(tmp_path / "counts.zarr")
    writer = CSVtoZarr(
        CSVReader(str(path), batch_size=2),
        destination,
        assay_name="RNA",
        nthreads=1,
    )
    write(rows)
    with pytest.raises(ValueError, match=message):
        writer.dump()
    counts = zarr.open_group(destination, mode="r")["RNA/counts"]
    assert counts.attrs["complete"] is False
