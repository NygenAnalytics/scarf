import dataclasses
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.graph.state import read_assay_state
from scarf.readers._rds import R_INT_NA, RdsClosedError
from scarf.readers.seurat import SeuratReader
from scarf.storage.artifacts import artifact_group
from scarf.writers.seurat import SeuratImportResult, SeuratToZarr
from tests.test_seurat_reader import (
    _Wire,
    _legacy_assay,
    _reduction,
    _write_chromatin_fixture,
    _write_fixture,
)


def _ordered_factor(
    wire: _Wire,
    values: list[int],
    levels: list[str],
) -> bytes:
    return wire.integer_vector(
        values,
        attributes=[
            ("levels", wire.string_vector(levels)),
            ("class", wire.string_vector(["ordered", "factor"])),
        ],
    )


def _partial_assay5(
    wire: _Wire,
    *,
    source_class: str = "Assay5",
) -> bytes:
    layer_names = ["counts", "data"]
    layers = wire.vector(
        [
            wire.matrix([1, 0, 2, 3], (2, 2)),
            wire.matrix([0.0] * 4, (2, 2), real=True),
        ],
        names=layer_names,
    )
    metadata = wire.data_frame(
        [("kind", wire.string_vector(["a", "b"]))],
        2,
    )
    return wire.s4(
        [
            ("layers", layers),
            (
                "cells",
                wire.logmap([1, 1, 1, 1], ["c1", "c3"], layer_names),
            ),
            (
                "features",
                wire.logmap([1, 1, 1, 1], ["p1", "p2"], layer_names),
            ),
            ("meta.data", metadata),
            ("class", wire.string_vector([source_class])),
        ]
    )


def _write_partial_fixture(
    path: Path,
    *,
    reduction_name: str = "pca",
    assay5_source_class: str = "Assay5",
) -> Path:
    wire = _Wire()
    metadata = wire.data_frame(
        [
            ("logical", wire.logical_vector([1, 0, R_INT_NA])),
            ("integer", wire.integer_vector([1, R_INT_NA, 3])),
            ("real", wire.real_vector([1.5, float("nan"), 3.5])),
            ("character", wire.string_vector(["a", None, "c"])),
            (
                "group",
                _ordered_factor(
                    wire,
                    [1, 2, R_INT_NA],
                    ["first", "second"],
                ),
            ),
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
                        _partial_assay5(
                            wire,
                            source_class=assay5_source_class,
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
                wire.vector([_reduction(wire)], names=[reduction_name]),
            ),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _missing(group: zarr.Group, column_name: str) -> np.ndarray:
    column = group[column_name]
    return np.asarray(group[column.attrs["missing_mask"]][:], dtype=bool)


def test_import_materializes_metadata_counts_membership_and_pca(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(tmp_path / "partial.rds")
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        result = SeuratToZarr(
            reader,
            destination,
            mem_budget="64M",
            nthreads=1,
            targetChunkBytes=1024,
            targetShardBytes=4096,
        ).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    assert isinstance(result, SeuratImportResult)
    assert result.assayNames == ("RNA", "ADT")
    assert result.defaultAssay == "RNA"
    assert root.attrs["complete"] is True
    assert root.attrs["scarf:import_complete"] is True
    assert root.attrs["scarf:import_source"] == "seurat"
    assert len(root.attrs["scarf:import_source_sha256"]) == 64
    assert len(root.attrs["scarf:import_payload_sha256"]) == 64
    assert root.attrs["defaultAssay"] == "RNA"
    assert root["cellData/ids"][:].tolist() == ["c1", "c2", "c3"]
    assert root["cellData/names"][:].tolist() == ["c1", "c2", "c3"]

    cells = root["cellData"]
    assert cells["group"][:].tolist() == ["first", "second", ""]
    assert cells["group"].attrs["levels"] == ["first", "second"]
    assert cells["group"].attrs["ordered"] is True
    np.testing.assert_array_equal(_missing(cells, "group"), [False, False, True])
    assert cells["active.ident"][:].tolist() == ["zero", "", "one"]
    assert cells["active.ident"].attrs["levels"] == ["zero", "one"]
    np.testing.assert_array_equal(
        _missing(cells, "active.ident"),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        _missing(cells, "character"),
        [False, True, False],
    )
    np.testing.assert_array_equal(cells["ADT_I"][:], [True, False, True])

    np.testing.assert_array_equal(
        root["RNA/counts"][:],
        [[1, 0], [0, 2], [3, 0]],
    )
    np.testing.assert_array_equal(
        root["ADT/counts"][:],
        [[1, 0], [0, 0], [2, 3]],
    )
    assert root["RNA/featureData/ids"][:].tolist() == ["g1", "g2"]
    assert root["RNA/featureData/names"][:].tolist() == ["g1", "g2"]
    assert root["RNA/featureData/symbol"][:].tolist() == ["G1", "G2"]
    assert root["ADT/featureData/kind"][:].tolist() == ["a", "b"]

    selection = artifact_group(root, result.cellSelection)
    np.testing.assert_array_equal(selection["values"][:], [True, True, True])
    pca_ref = result.reductionArtifacts["pca"]
    pca = artifact_group(root, pca_ref)
    np.testing.assert_array_equal(
        pca["data"][:],
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    )
    np.testing.assert_array_equal(pca["stdev"][:], [2.0, 1.0])
    state = read_assay_state(root, "RNA")
    assert state is not None
    assert state.reduction is None
    assert state.named_results["pca"] == pca_ref
    assert any(
        notice.code == "ignored_normalized_layer"
        and notice.objectPath == "assays/ADT/layers/data"
        for notice in result.notices
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.defaultAssay = "ADT"  # type: ignore[misc]


def test_assay5_subclass_preserves_partial_cell_membership(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "partial-subclass.rds",
        assay5_source_class="ChromatinAssay5",
    )
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        assert reader.get_assay("ADT").sourceClass == "ChromatinAssay5"
        SeuratToZarr(
            reader,
            destination,
            mem_budget="64M",
            nthreads=1,
            targetChunkBytes=1024,
            targetShardBytes=4096,
        ).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    np.testing.assert_array_equal(
        root["cellData/ADT_I"][:],
        [True, False, True],
    )


def test_imported_store_is_readable_by_datastore(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "datastore.rds")
    destination = MemoryStore()
    with SeuratReader(source) as reader:
        SeuratToZarr(
            reader,
            destination,
            mem_budget="64M",
            nthreads=1,
            targetChunkBytes=1024,
            targetShardBytes=4096,
        ).dump(batch_size=2)

    store = DataStore(
        destination,
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
        mem_budget="64M",
    )

    assert store._defaultAssay == "RNA"
    assert store.assay_names == ["ADT", "RNA"]
    assert store.cells.fetch_all("ids").tolist() == ["c1", "c2", "c3"]
    assert store.get_assay("RNA").rawData.shape == (3, 2)
    assert store.get_assay_state("RNA") is not None
    assert store.get_assay_state("RNA").reduction is None  # type: ignore[union-attr]


def test_chromatin_assay_streams_counts_and_round_trips_to_zarr(
    tmp_path: Path,
) -> None:
    source = _write_chromatin_fixture(tmp_path / "chromatin.rds")
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        result = SeuratToZarr(
            reader,
            destination,
            mem_budget="64M",
            nthreads=1,
            targetChunkBytes=1024,
            targetShardBytes=4096,
        ).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    assert result.assayNames == ("ATAC",)
    assert result.defaultAssay == "ATAC"
    assert "lsi" in result.reductionArtifacts
    np.testing.assert_array_equal(
        root["ATAC/counts"][:],
        [[1, 0], [0, 2], [3, 0]],
    )
    assert any(
        notice.code == "ignored_assay_slot"
        and notice.objectPath == "assays/ATAC/ranges"
        for notice in result.notices
    )
    store = DataStore(
        destination,
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
        mem_budget="64M",
    )
    assert store.assay_names == ["ATAC"]
    assert store.ATAC.rawData.shape == (3, 2)


def test_import_normalizes_seurat_reduction_name_for_assay_state(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "named-reduction.rds",
        reduction_name="UMAP.RNA",
    )
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        result = SeuratToZarr(
            reader,
            destination,
            mem_budget="64M",
            nthreads=1,
            targetChunkBytes=1024,
            targetShardBytes=4096,
        ).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    state = read_assay_state(root, "RNA")
    assert state is not None
    assert result.reductionArtifacts["UMAP.RNA"] == state.named_results["umap_rna"]


def test_reader_must_remain_open_until_dump_finishes(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "closed.rds")
    destination = MemoryStore()
    reader = SeuratReader(source)
    writer = SeuratToZarr(
        reader,
        destination,
        mem_budget="64M",
        nthreads=1,
        targetChunkBytes=1024,
        targetShardBytes=4096,
    )
    reader.close()

    with pytest.raises(RdsClosedError, match="RDS document is closed"):
        writer.dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    assert root.attrs["complete"] is False
    assert root.attrs["scarf:import_complete"] is False
    with pytest.raises(RuntimeError, match="seurat import is incomplete"):
        DataStore(
            destination,
            min_features_per_cell=0,
            min_cells_per_feature=0,
            nthreads=1,
            mem_budget="64M",
        )


def test_matrix_metadata_and_reduction_reads_stay_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_partial_fixture(tmp_path / "bounded.rds")
    destination = MemoryStore()
    with SeuratReader(source) as reader:
        writer = SeuratToZarr(
            reader,
            destination,
            mem_budget="64M",
            nthreads=1,
            targetChunkBytes=1024,
            targetShardBytes=4096,
        )
        matrix_reads: dict[str, list[tuple[int, int]]] = {}
        for assay_name in reader.assayNames:
            matrix = reader.get_assay(assay_name).counts
            original = matrix.read_cells
            reads: list[tuple[int, int]] = []
            matrix_reads[assay_name] = reads

            def tracked_matrix_read(
                start: int,
                stop: int,
                *,
                read=original,
                target=reads,
            ):
                target.append((start, stop))
                return read(start, stop)

            monkeypatch.setattr(matrix, "read_cells", tracked_matrix_read)

        character = reader.cellMetadata.column("character")
        original_metadata_read = character.read_block
        metadata_reads: list[tuple[int, int]] = []

        def tracked_metadata_read(start: int, stop: int):
            metadata_reads.append((start, stop))
            return original_metadata_read(start, stop)

        monkeypatch.setattr(character, "read_block", tracked_metadata_read)
        membership = reader.get_assay("ADT").cellMembership
        original_membership_read = membership.read_block
        membership_reads: list[tuple[int, int]] = []

        def tracked_membership_read(start: int, stop: int):
            membership_reads.append((start, stop))
            return original_membership_read(start, stop)

        monkeypatch.setattr(membership, "read_block", tracked_membership_read)
        embeddings = reader.get_reduction("pca").cellEmbeddings
        original_reduction_read = embeddings.read_rows
        reduction_reads: list[tuple[int, int]] = []

        def tracked_reduction_read(start: int, stop: int):
            reduction_reads.append((start, stop))
            return original_reduction_read(start, stop)

        monkeypatch.setattr(embeddings, "read_rows", tracked_reduction_read)
        writer.dump(batch_size=1)

    assert all(
        stop - start <= 1 for reads in matrix_reads.values() for start, stop in reads
    )
    assert all(stop - start <= 1 for start, stop in metadata_reads)
    assert all(stop - start <= 1 for start, stop in membership_reads)
    assert all(stop - start <= 1 for start, stop in reduction_reads)
    assert {start for start, _ in matrix_reads["RNA"]} == {0, 1, 2}
    assert {start for start, _ in matrix_reads["ADT"]} == {0, 1, 2}
    assert {start for start, _ in membership_reads} == {0, 1, 2}
    assert {start for start, _ in reduction_reads} == {0, 1, 2}
