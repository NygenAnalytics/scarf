import dataclasses
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.readers._rds import R_INT_NA, RdsClosedError
from scarf.readers.seurat import SeuratImportError, SeuratReader
from scarf.storage.artifacts import artifact_group
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.writers.seurat import SeuratImportResult, SeuratToZarr
from tests.test_seurat_reader import (
    _Wire,
    _legacy_assay,
    _reduction,
    _write_delayed_hdf5array_fixture,
    _write_chromatin_fixture,
    _write_fixture,
    _write_single_assay_fixture,
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


def _reduction_with_loadings(
    wire: _Wire,
    *,
    assay_used: str = "RNA",
) -> bytes:
    embeddings = wire.matrix(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (3, 2),
        rows=["c1", "c2", "c3"],
        columns=["PC_1", "PC_2"],
        real=True,
    )
    loadings = wire.matrix(
        [0.1, 0.2, 0.3, 0.4],
        (2, 2),
        rows=["g1", "g2"],
        columns=["PC_1", "PC_2"],
        real=True,
    )
    return wire.s4(
        [
            ("cell.embeddings", embeddings),
            ("feature.loadings", loadings),
            ("assay.used", wire.string_vector([assay_used])),
            ("global", wire.logical_vector([0])),
            ("stdev", wire.real_vector([2.0, 1.0])),
            ("key", wire.string_vector(["PC_"])),
            ("class", wire.string_vector(["DimReduc"])),
        ]
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
    reduction_names: list[str] | None = None,
    reduction_assay: str = "RNA",
    reduction_loadings: bool = False,
    assay5_source_class: str = "Assay5",
    extra_cell_metadata_name: str | None = None,
) -> Path:
    wire = _Wire()
    metadata_columns = [
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
    ]
    if extra_cell_metadata_name is not None:
        metadata_columns.append(
            (
                extra_cell_metadata_name,
                wire.string_vector(["x", "y", "z"]),
            )
        )
    metadata = wire.data_frame(
        metadata_columns,
        ["c1", "c2", "c3"],
    )
    selected_reduction_names = (
        [reduction_name] if reduction_names is None else reduction_names
    )
    reduction_writer = _reduction_with_loadings if reduction_loadings else _reduction
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
                wire.vector(
                    [
                        reduction_writer(wire, assay_used=reduction_assay)
                        for _ in selected_reduction_names
                    ],
                    names=selected_reduction_names,
                ),
            ),
            ("class", wire.string_vector(["Seurat"])),
        ]
    )
    path.write_bytes(wire.document(root))
    return path


def _missing(group: zarr.Group, column_name: str) -> np.ndarray:
    column = group[column_name]
    return np.asarray(group[column.attrs["missing_mask"]][:], dtype=bool)


def _new_writer(reader: SeuratReader, destination: MemoryStore) -> SeuratToZarr:
    return SeuratToZarr(
        reader,
        destination,
        mem_budget="64M",
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
    )


def _destination_with_sentinel() -> MemoryStore:
    destination = MemoryStore()
    zarr.open_group(store=destination, mode="w").create_group("sentinel")
    return destination


def _assert_destination_untouched(destination: MemoryStore) -> None:
    root = zarr.open_group(store=destination, mode="r")
    assert set(root.group_keys()) == {"sentinel"}


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
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
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
    np.testing.assert_array_equal(cells["logical"][:], [True, False, False])
    np.testing.assert_array_equal(
        _missing(cells, "logical"),
        [False, False, True],
    )
    np.testing.assert_array_equal(cells["integer"][:], [1, 0, 3])
    np.testing.assert_array_equal(
        _missing(cells, "integer"),
        [False, True, False],
    )
    np.testing.assert_allclose(
        cells["real"][:],
        [1.5, np.nan, 3.5],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        _missing(cells, "real"),
        [False, True, False],
    )
    assert cells["group"][:].tolist() == ["first", "second", ""]
    assert cells["group"].attrs["levels"] == ["first", "second"]
    assert cells["group"].attrs["ordered"] is True
    np.testing.assert_array_equal(_missing(cells, "group"), [False, False, True])
    assert "active.ident" not in cells
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
    active_identity = artifact_group(root, result.activeIdentity)
    assert result.activeIdentity.kind == "cluster_labels"
    assert result.activeIdentity.assay == "RNA"
    assert active_identity["values"][:].tolist() == ["zero", "", "one"]
    assert active_identity["values"].attrs["levels"] == ["zero", "one"]
    np.testing.assert_array_equal(
        active_identity["__scarf_missing__values"][:],
        [False, True, False],
    )
    pca_ref = result.reductionArtifacts["pca"]
    pca = artifact_group(root, pca_ref)
    np.testing.assert_array_equal(
        pca["data"][:],
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    )
    np.testing.assert_array_equal(pca["stdev"][:], [2.0, 1.0])
    assert result.artifactRefs == (
        result.cellSelection,
        result.activeIdentity,
        pca_ref,
    )
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
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
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
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
        ).dump(batch_size=2)

    store = DataStore(
        destination,
        min_features_per_cell=0,
        nthreads=1,
        mem_budget="64M",
    )

    assert store._defaultAssay == "RNA"
    assert store.assay_names == ["ADT", "RNA"]
    assert store.cells.fetch_all("ids").tolist() == ["c1", "c2", "c3"]
    assert store.get_assay("RNA").rawData.shape == (3, 2)
    assert store.list_artifacts(
        kind="imported_coordinates",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )


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
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
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
        nthreads=1,
        mem_budget="64M",
    )
    assert store.assay_names == ["ATAC"]
    assert store.ATAC.rawData.shape == (3, 2)


def test_import_preserves_source_reduction_name_in_explicit_artifact(
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
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
        ).dump(batch_size=1)

    ref = result.reductionArtifacts["UMAP.RNA"]
    assert ref.kind == "imported_coordinates"
    root = zarr.open_group(store=destination, mode="r")
    provenance = artifact_group(root, ref).attrs["provenance"]
    assert provenance["parameters"]["dimreduc_key"] == "UMAP.RNA"
    assert provenance["parameters"]["role"] == "graphcoordinates"


def test_import_persists_reduction_loadings_and_feature_ids(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "reduction-loadings.rds",
        reduction_loadings=True,
    )
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        result = _new_writer(reader, destination).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    reduction = artifact_group(root, result.reductionArtifacts["pca"])
    np.testing.assert_allclose(
        reduction["loadings"][:],
        [[0.1, 0.3], [0.2, 0.4]],
    )
    assert reduction["feature_ids"][:].tolist() == ["g1", "g2"]


def test_imported_umap_writes_only_embedding_artifact(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "umap.rds",
        reduction_name="umap",
    )
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        result = _new_writer(reader, destination).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    embedding = artifact_group(root, result.reductionArtifacts["umap"])
    np.testing.assert_array_equal(
        embedding["values"][:],
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    )
    assert "RNA_UMAP1" not in root["cellData"]
    assert "RNA_UMAP2" not in root["cellData"]


def test_reader_must_remain_open_until_dump_finishes(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "closed.rds")
    destination = MemoryStore()
    reader = SeuratReader(source)
    writer = SeuratToZarr(
        reader,
        destination,
        mem_budget="64M",
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
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
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
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


def test_writer_rejects_unselected_active_assay_before_mutating_destination(
    tmp_path: Path,
) -> None:
    source = _write_fixture(tmp_path / "unselected-active.rds")
    destination = _destination_with_sentinel()

    with SeuratReader(source, assays=["ADT"], reductions=[]) as reader:
        with pytest.raises(
            ValueError,
            match="Active assay 'RNA' is not selected for import",
        ):
            _new_writer(reader, destination)

    _assert_destination_untouched(destination)


def test_writer_rejects_blocked_reduction_before_mutating_destination(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "blocked-reduction.rds",
        reduction_assay="missing",
    )
    destination = _destination_with_sentinel()

    with SeuratReader(source) as reader:
        diagnostic = reader.inspection.reduction("pca").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "reduction_assay_missing"
        with pytest.raises(
            ValueError,
            match=r"pca \(reduction_assay_missing\)",
        ):
            _new_writer(reader, destination)

    _assert_destination_untouched(destination)


def test_writer_rejects_reduction_for_unselected_assay_before_mutation(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "unselected-reduction-assay.rds",
        reduction_assay="ADT",
    )
    destination = _destination_with_sentinel()

    with SeuratReader(source, assays=["RNA"]) as reader:
        assert reader.inspection.reduction("pca").importable
        with pytest.raises(
            ValueError,
            match="Selected reductions reference assays.*pca",
        ):
            _new_writer(reader, destination)

    _assert_destination_untouched(destination)


@pytest.mark.parametrize(
    "reduction_names",
    [
        pytest.param(["123", "123?"], id="numeric-prefix"),
        pytest.param(["---", "..."], id="empty-normalization"),
    ],
)
def test_writer_preserves_reduction_names_without_normalized_name_constraints(
    tmp_path: Path,
    reduction_names: list[str],
) -> None:
    source = _write_partial_fixture(
        tmp_path / "colliding-reductions.rds",
        reduction_names=reduction_names,
    )
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        result = _new_writer(reader, destination).dump(batch_size=1)

    assert list(result.reductionArtifacts) == reduction_names
    refs = tuple(result.reductionArtifacts.values())
    assert len(set(refs)) == len(reduction_names)
    root = zarr.open_group(store=destination, mode="r")
    assert [
        artifact_group(root, ref).attrs["provenance"]["parameters"]["dimreduc_key"]
        for ref in refs
    ] == reduction_names


@pytest.mark.parametrize(
    ("column_name", "message"),
    [
        pytest.param("ids", "metadata column 'ids' is reserved", id="reserved"),
        pytest.param(
            "__scarf_missing__quality",
            "uses Scarf's internal prefix",
            id="internal-prefix",
        ),
        pytest.param(
            "ADT_I",
            "membership columns conflict with cell metadata",
            id="assay-membership",
        ),
    ],
)
def test_writer_rejects_conflicting_metadata_before_mutating_destination(
    tmp_path: Path,
    column_name: str,
    message: str,
) -> None:
    source = _write_partial_fixture(
        tmp_path / "conflicting-metadata.rds",
        extra_cell_metadata_name=column_name,
    )
    destination = _destination_with_sentinel()

    with SeuratReader(source) as reader:
        with pytest.raises(ValueError, match=message):
            _new_writer(reader, destination)

    _assert_destination_untouched(destination)


def test_dump_rejects_boolean_batch_size_and_stays_incomplete(
    tmp_path: Path,
) -> None:
    source = _write_partial_fixture(tmp_path / "invalid-batch.rds")
    destination = MemoryStore()

    with SeuratReader(source) as reader:
        writer = _new_writer(reader, destination)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            writer.dump(batch_size=False)

    root = zarr.open_group(store=destination, mode="r")
    assert root.attrs["complete"] is False
    assert root.attrs["scarf:import_complete"] is False


def test_layer_override_controls_converted_assay_counts(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "layer-override.rds")
    destination = MemoryStore()

    with SeuratReader(
        source,
        assay_layers={"ADT": ["counts.1"]},
        reductions=[],
    ) as reader:
        result = _new_writer(reader, destination).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    np.testing.assert_array_equal(
        root["ADT/counts"][:],
        [[1, 3, 0], [2, 4, 0], [0, 0, 0]],
    )
    assert any(
        notice.code == "ignored_unselected_count_layer"
        and notice.objectPath == "assays/ADT/layers/counts.2"
        for notice in result.notices
    )


def test_dense_assay_import_round_trips_in_requested_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_fixture(tmp_path / "dense.rds")
    destination = MemoryStore()

    with SeuratReader(source, assays=["RNA"], reductions=[]) as reader:
        counts = reader.get_assay("RNA").counts
        assert counts.is_sparse is False
        original_read = counts.read_cells
        reads: list[tuple[int, int]] = []

        def tracked_read(start: int, stop: int):
            reads.append((start, stop))
            return original_read(start, stop)

        monkeypatch.setattr(counts, "read_cells", tracked_read)
        result = _new_writer(reader, destination).dump(batch_size=2)

    root = zarr.open_group(store=destination, mode="r")
    assert result.assayNames == ("RNA",)
    assert reads == [(0, 2), (2, 3)]
    np.testing.assert_array_equal(
        root["RNA/counts"][:],
        [[1, 0], [0, 2], [3, 0]],
    )


def test_dense_assay_import_rejects_a_budget_below_one_output_band(
    tmp_path: Path,
) -> None:
    source = _write_fixture(tmp_path / "dense-budget.rds")
    destination = MemoryStore()

    with SeuratReader(source, assays=["RNA"], reductions=[]) as reader:
        writer = SeuratToZarr(
            reader,
            destination,
            mem_budget="1K",
            nthreads=1,
            policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=1024),
        )
        with pytest.raises(
            MemoryError,
            match="cannot fit one source row and one destination row band",
        ):
            writer.dump()

    root = zarr.open_group(store=destination, mode="r")
    assert root.attrs["complete"] is False
    assert root.attrs["scarf:import_complete"] is False


@pytest.mark.parametrize(
    ("fixture_options", "expected"),
    [
        pytest.param(
            {"transformed": True},
            [[1, 0], [0, 1], [2, 0]],
            id="parameter-transform",
        ),
        pytest.param(
            {"delayed_primitive": True},
            [[3, 2], [2, 4], [5, 2]],
            id="delayed-primitive",
        ),
    ],
)
def test_structural_matrix_factory_nodes_convert_to_zarr(
    tmp_path: Path,
    fixture_options: dict[str, bool],
    expected: list[list[int]],
) -> None:
    source = _write_delayed_hdf5array_fixture(
        tmp_path / "structural.rds",
        **fixture_options,
    )
    destination = MemoryStore()

    with SeuratReader(source, reductions=[]) as reader:
        result = _new_writer(reader, destination).dump(batch_size=1)

    root = zarr.open_group(store=destination, mode="r")
    assert result.assayNames == ("RNA",)
    np.testing.assert_array_equal(root["RNA/counts"][:], expected)


def test_failed_conversion_stays_incomplete_and_retry_replaces_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_fixture(tmp_path / "retry.rds")
    destination = MemoryStore()

    with SeuratReader(source, assays=["RNA"], reductions=[]) as reader:
        counts = reader.get_assay("RNA").counts
        original_read = counts.read_cells

        def failing_read(start: int, stop: int):
            if start >= 1:
                raise OSError("synthetic count read failure")
            return original_read(start, stop)

        monkeypatch.setattr(counts, "read_cells", failing_read)
        writer = _new_writer(reader, destination)
        with pytest.raises(OSError, match="synthetic count read failure"):
            writer.dump(batch_size=1)

        failed = zarr.open_group(store=destination, mode="r")
        assert failed.attrs["complete"] is False
        assert failed.attrs["scarf:import_complete"] is False
        with pytest.raises(RuntimeError, match="seurat import is incomplete"):
            DataStore(
                destination,
                min_features_per_cell=0,
                nthreads=1,
                mem_budget="64M",
            )

        monkeypatch.setattr(counts, "read_cells", original_read)
        _new_writer(reader, destination).dump(batch_size=1)

    recovered = zarr.open_group(store=destination, mode="r")
    assert recovered.attrs["complete"] is True
    assert recovered.attrs["scarf:import_complete"] is True
    np.testing.assert_array_equal(
        recovered["RNA/counts"][:],
        [[1, 0], [0, 2], [3, 0]],
    )


def test_malformed_assay_error_is_preserved_without_destination_mutation(
    tmp_path: Path,
) -> None:
    wire = _Wire()
    source = _write_single_assay_fixture(
        tmp_path / "malformed-assay.rds",
        wire=wire,
        assay=wire.s4(
            [
                (
                    "layers",
                    wire.vector(
                        [wire.matrix([1, 0, 0, 2, 3, 0], (2, 3))],
                        names=["counts"],
                    ),
                ),
                ("class", wire.string_vector(["Assay5"])),
            ]
        ),
    )
    destination = _destination_with_sentinel()

    with SeuratReader(source, reductions=[]) as reader:
        diagnostic = reader.inspection.assay("RNA").blockingDiagnostic
        assert diagnostic is not None
        assert diagnostic.code == "missing_slot"
        assert diagnostic.objectPath == "assays/RNA/cells"
        with pytest.raises(SeuratImportError) as error:
            _new_writer(reader, destination)

    assert error.value.code == "missing_slot"
    assert error.value.objectPath == "assays/RNA/cells"
    _assert_destination_untouched(destination)
