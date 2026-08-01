import gzip
import stat
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf import DataStore
from scarf.readers import CrH5Reader, MtxCandidate, MtxReader, inspect_mtx
from scarf.writers import CrToZarr, MtxToZarr


def _write_text(path: Path, text: str) -> None:
    if path.name.endswith(".gz"):
        with gzip.open(path, mode="wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)


def _write_mex(
    path: Path,
    coordinates: list[tuple[int, int, int]],
    *,
    n_features: int = 4,
    n_cells: int = 4,
    matrix_name: str = "matrix.mtx",
    prefix: str = "",
    compressed_sidecars: bool = False,
    feature_types: list[str] | None = None,
) -> None:
    matrix_path = path / matrix_name
    rows = "\n".join(f"{row} {column} {value}" for row, column, value in coordinates)
    _write_text(
        matrix_path,
        "%%MatrixMarket matrix coordinate integer general\n"
        "% fixture\n"
        f"{n_features} {n_cells} {len(coordinates)}\n"
        f"{rows}\n",
    )
    suffix = ".gz" if compressed_sidecars else ""
    types = feature_types or ["Gene Expression"] * n_features
    _write_text(
        path / f"{prefix}features.tsv{suffix}",
        "".join(
            f"feature-{index}\tgene-{index}\t{types[index]}\n"
            for index in range(n_features)
        ),
    )
    _write_text(
        path / f"{prefix}barcodes.tsv{suffix}",
        "".join(f"cell-{index}\n" for index in range(n_cells)),
    )


@pytest.mark.parametrize(
    ("matrix_name", "compressed_sidecars"),
    [("matrix.mtx", True), ("matrix.mtx.gz", False)],
)
def test_inspect_and_stream_canonical_mixed_compression(
    tmp_path: Path,
    matrix_name: str,
    compressed_sidecars: bool,
) -> None:
    coordinates = [(1, 1, 2), (3, 1, 4), (2, 2, 3), (4, 4, 5)]
    _write_mex(
        tmp_path,
        coordinates,
        matrix_name=matrix_name,
        compressed_sidecars=compressed_sidecars,
    )

    candidate = inspect_mtx(tmp_path)[0]
    assert isinstance(candidate, MtxCandidate)
    assert candidate.matrixOrientation == "featuresByCells"
    reader = MtxReader(candidate)
    explicit = MtxReader(
        candidate.matrixPath,
        candidate.featurePath,
        candidate.cellPath,
    )
    try:
        assert reader.coordinateOrder == "cellMajor"
        observed = np.vstack(
            [batch.toarray() for batch in reader.consume(2, lines_in_mem=2)]
        )
        expected = np.vstack(
            [batch.toarray() for batch in explicit.consume(3, lines_in_mem=3)]
        )
    finally:
        reader.close()
        explicit.close()
    np.testing.assert_array_equal(observed, expected)


def test_filtered_cell_major_stream_does_not_prescan_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 1), (3, 1, 2), (2, 2, 3), (4, 3, 4)],
        n_cells=3,
    )

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("filtered cell-major input must not be prescanned")

    monkeypatch.setattr(
        "scarf.readers.mtx._MtxEngine._scan_matrix",
        unexpected_scan,
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        observed = np.vstack(
            [batch.toarray() for batch in reader.consume(2, lines_in_mem=2)]
        )
    finally:
        reader.close()
    np.testing.assert_array_equal(
        observed,
        [[1, 0, 2, 0], [0, 3, 0, 0], [0, 0, 0, 4]],
    )


def test_matrix_market_count_dtype_overflow_requires_explicit_widening(
    tmp_path: Path,
) -> None:
    value = int(np.iinfo(np.uint32).max) + 1
    _write_mex(
        tmp_path,
        [(1, 1, value)],
        n_features=1,
        n_cells=1,
    )
    candidate = inspect_mtx(tmp_path)[0]

    default_reader = MtxReader(candidate)
    try:
        with pytest.raises(
            OverflowError,
            match="Matrix Market count exceeds dtype uint32",
        ):
            list(default_reader.consume(1))
    finally:
        default_reader.close()

    wide_reader = MtxReader(candidate, dtype=np.uint64)
    try:
        observed = list(wide_reader.consume(1))[0].toarray()
    finally:
        wide_reader.close()
    np.testing.assert_array_equal(observed, np.array([[value]], dtype=np.uint64))


def test_feature_major_disk_csr_parity_filtering_and_cleanup(tmp_path: Path) -> None:
    feature_major = tmp_path / "feature-major"
    cell_major = tmp_path / "cell-major"
    feature_major.mkdir()
    cell_major.mkdir()
    feature_coordinates = [
        (1, 1, 1),
        (1, 3, 2),
        (2, 2, 1),
        (2, 2, 2),
        (3, 1, 4),
        (4, 4, 1),
    ]
    cell_coordinates = [
        (1, 1, 1),
        (3, 1, 4),
        (2, 2, 3),
        (1, 3, 2),
        (4, 4, 1),
    ]
    _write_mex(feature_major, feature_coordinates)
    _write_mex(cell_major, cell_coordinates)

    feature_reader = MtxReader(
        inspect_mtx(feature_major)[0],
        is_filtered=False,
        filtering_cutoff=1,
        temp_dir=str(tmp_path),
    )
    cell_reader = MtxReader(
        inspect_mtx(cell_major)[0],
        is_filtered=False,
        filtering_cutoff=1,
    )
    assert feature_reader.coordinateOrder == "featureMajor"
    assert feature_reader.cell_names() == ["cell-0", "cell-1", "cell-2"]
    feature_reader._prepare_sparse_import()
    assert feature_reader.temporaryDiskBytes == 64
    prepared = list(feature_reader.consume(2, lines_in_mem=2))
    assert all(batch.tocsr().has_sorted_indices for batch in prepared)
    widened = list(feature_reader.consume(2, lines_in_mem=2, dtype=np.uint64))
    assert all(batch.dtype == np.dtype(np.uint64) for batch in widened)
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in widened]),
        np.vstack([batch.toarray() for batch in prepared]),
    )

    stores = [MemoryStore(), MemoryStore()]
    MtxToZarr(
        feature_reader,
        stores[0],
        mem_budget="64M",
        targetChunkBytes=16,
        targetShardBytes=32,
    ).dump(lines_in_mem=2)
    MtxToZarr(
        cell_reader,
        stores[1],
        mem_budget="64M",
        targetChunkBytes=16,
        targetShardBytes=32,
    ).dump(lines_in_mem=2)

    first = zarr.open_group(store=stores[0], mode="r")["RNA/counts"][:]
    second = zarr.open_group(store=stores[1], mode="r")["RNA/counts"][:]
    np.testing.assert_array_equal(first, second)
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))


def _write_parse(path: Path, *, both_aliases: bool = False) -> None:
    _write_text(
        path / "DGE.mtx.gz",
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 2 5\n"
        "1 1 1\n"
        "1 2 2\n"
        "2 1 3\n"
        "3 2 1\n"
        "3 2 4\n",
    )
    (path / "genes.csv").write_text(
        "gene_id,gene_name\nfeature-0,Gene 0\nfeature-1,Gene 1\n"
    )
    header = (
        "bc_wells,bc_index,sample,quality"
        if both_aliases
        else ("bc_wells,sample,quality")
    )
    rows = (
        "cell-0,index-0,A,1\ncell-1,index-1,A,2\ncell-2,index-2,B,3\n"
        if both_aliases
        else "cell-0,A,1\ncell-1,A,2\ncell-2,B,3\n"
    )
    (path / "cell_metadata.csv").write_text(f"{header}\n{rows}")


def test_parse_orientation_duplicate_sum_and_metadata(tmp_path: Path) -> None:
    _write_parse(tmp_path)
    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.matrixOrientation == "cellsByFeatures"
    assert candidate.cellIdKeys == ("bc_wells",)
    reader = MtxReader(candidate)
    store = MemoryStore()
    MtxToZarr(
        reader,
        store,
        mem_budget="64M",
        targetChunkBytes=8,
        targetShardBytes=16,
    ).dump(lines_in_mem=2)

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(
        root["RNA/counts"][:],
        [[1, 2], [3, 0], [0, 5]],
    )
    np.testing.assert_array_equal(root["cellData/sample"][:], ["A", "A", "B"])
    np.testing.assert_array_equal(root["cellData/quality"][:], [1, 2, 3])


def test_parse_requires_explicit_ambiguous_cell_id_key(tmp_path: Path) -> None:
    _write_parse(tmp_path, both_aliases=True)
    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.cellIdKeys == ("bc_wells", "bc_index")
    with pytest.raises(ValueError, match="ambiguous"):
        MtxReader(candidate)
    reader = MtxReader(candidate, cell_id_key="bc_index")
    try:
        assert reader.cell_names() == ["index-0", "index-1", "index-2"]
        assert "bc_wells" in dict(reader.get_cell_columns())
    finally:
        reader.close()


def test_prefixed_dragen_triplet_and_ambiguous_candidates(tmp_path: Path) -> None:
    coordinates = [(1, 1, 1)]
    _write_mex(
        tmp_path,
        coordinates,
        n_features=1,
        n_cells=1,
        matrix_name="sample_matrix.mtx.gz",
        prefix="sample_",
    )
    other = tmp_path / "other"
    other.mkdir()
    _write_mex(other, coordinates, n_features=1, n_cells=1)

    candidate = inspect_mtx(tmp_path)[0]
    assert Path(candidate.featurePath).name == "sample_features.tsv"
    assert Path(candidate.cellPath).name == "sample_barcodes.tsv"

    _write_mex(
        tmp_path,
        coordinates,
        n_features=1,
        n_cells=1,
        matrix_name="second_matrix.mtx",
        prefix="second_",
    )
    assert len(inspect_mtx(tmp_path)) == 2
    with pytest.raises(ValueError, match="requires one"):
        from scarf.readers import CrDirReader

        CrDirReader(str(tmp_path))


def _write_zip_mex(path: Path) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            "bundle/sample_matrix.mtx",
            "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n2 2 2\n",
        )
        archive.writestr(
            "bundle/sample_features.tsv",
            "feature-0\tGene 0\nfeature-1\tGene 1\n",
        )
        archive.writestr(
            "bundle/sample_barcodes.tsv",
            "cell-0\ncell-1\n",
        )
        archive.writestr(
            "bundle/protospacer_calls_per_cell.csv",
            "cell_id,feature_call\ncell-0,feature-0\n",
        )


def test_direct_mex_zip_and_archive_safety(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample_MEX.zip"
    _write_zip_mex(archive_path)
    candidate = inspect_mtx(archive_path)[0]
    assert candidate.relatedFiles == ("bundle/protospacer_calls_per_cell.csv",)
    reader = MtxReader(candidate, temp_dir=str(tmp_path))
    try:
        assert candidate.relatedFiles[0] not in reader._engine._archivePaths
        observed = np.vstack([batch.toarray() for batch in reader.consume(1)])
    finally:
        reader.close()
    np.testing.assert_array_equal(observed, np.eye(2, dtype=np.uint32) * [1, 2])
    assert not list(tmp_path.glob("scarf-mtx-archive-*"))

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, mode="w") as archive:
        archive.writestr("../matrix.mtx", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        inspect_mtx(traversal)

    linked = tmp_path / "linked.zip"
    info = zipfile.ZipInfo("matrix.mtx")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(linked, mode="w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="links"):
        inspect_mtx(linked)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(duplicate, mode="w") as archive:
            archive.writestr("matrix.mtx", "one")
            archive.writestr("matrix.mtx", "two")
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_mtx(duplicate)


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        ([(1, 1, 1), (2, 2, 1), (1, 1, 1)], "neither cell-major"),
        ([(0, 1, 1)], "outside the declared"),
        ([(1, 1, -1)], "non-negative"),
    ],
)
def test_invalid_matrix_market_coordinates(
    tmp_path: Path,
    coordinates: list[tuple[int, int, int]],
    message: str,
) -> None:
    _write_mex(
        tmp_path,
        coordinates,
        n_features=2,
        n_cells=2,
    )
    with pytest.raises(ValueError, match=message):
        MtxReader(inspect_mtx(tmp_path)[0])


def test_duplicate_ids_and_dimension_mismatch_are_rejected(tmp_path: Path) -> None:
    _write_mex(tmp_path, [(1, 1, 1)], n_features=1, n_cells=1)
    (tmp_path / "barcodes.tsv").write_text("cell-0\ncell-0\n")
    with pytest.raises(ValueError, match="do not match"):
        inspect_mtx(tmp_path)

    (tmp_path / "barcodes.tsv").write_text("cell-0\n")
    (tmp_path / "features.tsv").write_text("feature-0\tGene 0\nfeature-0\tGene 1\n")
    _write_text(
        tmp_path / "matrix.mtx",
        "%%MatrixMarket matrix coordinate integer general\n2 1 1\n1 1 1\n",
    )
    with pytest.raises(ValueError, match="unique"):
        MtxReader(inspect_mtx(tmp_path)[0])


def test_feature_major_cleanup_after_planning_and_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 1), (2, 2, 2), (3, 1, 3)],
        n_features=3,
        n_cells=2,
    )
    candidate = inspect_mtx(tmp_path)[0]

    planning_reader = MtxReader(candidate, temp_dir=str(tmp_path))
    writer = MtxToZarr(
        planning_reader,
        MemoryStore(),
        mem_budget=1,
        targetChunkBytes=8,
        targetShardBytes=8,
    )
    with pytest.raises(MemoryError, match="one source row"):
        writer.dump(lines_in_mem=1)
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))

    write_reader = MtxReader(candidate, temp_dir=str(tmp_path))

    def fail_after_first(writes, **kwargs):
        source = iter(writes)
        next(source)
        close = getattr(source, "close", None)
        if callable(close):
            close()
        raise RuntimeError("injected cancellation")

    monkeypatch.setattr(
        "scarf.storage.sharding.write_sparse_bands",
        fail_after_first,
    )
    writer = MtxToZarr(
        write_reader,
        MemoryStore(),
        mem_budget="64M",
        targetChunkBytes=8,
        targetShardBytes=8,
    )
    with pytest.raises(RuntimeError, match="injected cancellation"):
        writer.dump(lines_in_mem=1)
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))


def test_feature_major_consume_uses_parse_budget_and_cleans_up_on_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 1), (1, 3, 2), (2, 2, 3), (3, 1, 4)],
        n_features=3,
        n_cells=3,
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0], temp_dir=str(tmp_path))
    observed_lines: list[int] = []
    original = reader._engine._coalesced_chunks

    def recording_chunks(lines_in_mem: int):
        observed_lines.append(lines_in_mem)
        yield from original(lines_in_mem)

    monkeypatch.setattr(reader._engine, "_coalesced_chunks", recording_chunks)
    source = reader.consume(1, lines_in_mem=2)
    next(source)
    assert observed_lines == [2]
    assert list(tmp_path.glob("scarf-mtx-csr-*"))
    source.close()
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))
    completed = list(reader.consume(2, lines_in_mem=2))
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))
    assert np.vstack([batch.toarray() for batch in completed]).sum() == 10


def test_multimodal_mex_names_feature_reference_and_related_files(
    tmp_path: Path,
) -> None:
    types = [
        "mRNA",
        "CRISPR Guide Capture",
        "Multiplexing Capture",
        "Antigen Capture",
        "Custom",
        "AbSeq",
    ]
    _write_mex(
        tmp_path,
        [(index, 1, index) for index in range(1, 7)],
        n_features=6,
        n_cells=1,
        feature_types=types,
    )
    (tmp_path / "feature_reference.csv").write_text(
        "id,sequence,pattern,read,target_gene_id\n"
        + "".join(
            f"feature-{index},SEQ{index},PAT{index},R2,target-{index}\n"
            for index in range(6)
        )
    )
    (tmp_path / "protospacer_calls_per_cell.csv").write_text(
        "cell_id,feature_call\ncell-0,feature-1\n"
    )
    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.relatedFiles == (str(tmp_path / "protospacer_calls_per_cell.csv"),)
    reader = MtxReader(candidate)
    store = MemoryStore()
    MtxToZarr(
        reader,
        store,
        mem_budget="64M",
        targetChunkBytes=8,
        targetShardBytes=8,
    ).dump(lines_in_mem=2)

    root = zarr.open_group(store=store, mode="r")
    assert set(root.group_keys()) == {
        "RNA",
        "CRISPR",
        "HTO",
        "ANTIGEN",
        "CUSTOM",
        "ADT",
        "cellData",
    }
    np.testing.assert_array_equal(
        root["CRISPR/featureData/sequence"][:],
        ["SEQ1"],
    )
    assert "feature_call" not in root["cellData"]
    datastore = DataStore(
        store,
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
        mem_budget="64M",
        nthreads=1,
    )
    summary = datastore.summary()
    assert {assay.name: assay.assay_type for assay in summary.assays} == {
        "ADT": "ADT",
        "ANTIGEN": "ANTIGEN",
        "CRISPR": "CRISPR",
        "CUSTOM": "CUSTOM",
        "HTO": "HTO",
        "RNA": "RNA",
    }


def test_bd_guide_reclassification_is_explicit(tmp_path: Path) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 1), (2, 1, 2)],
        n_features=2,
        n_cells=1,
        feature_types=["mRNA", "mRNA"],
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    reader.reclassify_features(
        [1],
        "CRISPR Guide Capture",
        require_previous="mRNA",
    )
    assert tuple(reader.assayFeats.columns) == ("RNA", "CRISPR")


def _write_tagged_h5(path: Path) -> None:
    values = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint16))
    with h5py.File(path, mode="w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=values.data)
        group.create_dataset("indices", data=values.indices)
        group.create_dataset("indptr", data=values.indptr)
        group.create_dataset("barcodes", data=np.array([b"cell-0", b"cell-1"]))
        features = group.create_group("features")
        features.create_dataset("id", data=np.array([b"gene-0", b"guide-0"]))
        features.create_dataset("name", data=np.array([b"Gene 0", b"Guide 0"]))
        features.create_dataset(
            "feature_type",
            data=np.array([b"Gene Expression", b"CRISPR Guide Capture"]),
        )
        features.create_dataset(
            "_all_tag_keys",
            data=np.array([b"sequence", b"pattern", b"read", b"target_gene_id"]),
        )
        features.create_dataset("sequence", data=np.array([b"", b"ACGT"]))
        features.create_dataset("pattern", data=np.array([b"", b"5P(BC)"]))
        features.create_dataset("read", data=np.array([b"", b"R2"]))
        features.create_dataset("target_gene_id", data=np.array([b"", b"gene-0"]))


def test_cellranger_h5_preserves_guide_feature_tags(tmp_path: Path) -> None:
    path = tmp_path / "tagged.h5"
    _write_tagged_h5(path)
    reader = CrH5Reader(str(path))
    store = MemoryStore()
    try:
        CrToZarr(
            reader,
            store,
            mem_budget="64M",
            targetChunkBytes=8,
            targetShardBytes=8,
        ).dump()
    finally:
        reader.close()
    feature_data = zarr.open_group(store=store, mode="r")["CRISPR/featureData"]
    np.testing.assert_array_equal(feature_data["sequence"][:], ["ACGT"])
    np.testing.assert_array_equal(feature_data["pattern"][:], ["5P(BC)"])
    np.testing.assert_array_equal(feature_data["read"][:], ["R2"])
    np.testing.assert_array_equal(feature_data["target_gene_id"][:], ["gene-0"])


def test_archive_parse_failure_removes_temporary_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid_MEX.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "matrix.mtx",
            "%%MatrixMarket matrix coordinate integer general\n"
            "2 2 3\n1 1 1\n2 2 1\n1 1 1\n",
        )
        archive.writestr("features.tsv", "f1\tg1\nf2\tg2\n")
        archive.writestr("barcodes.tsv", "c1\nc2\n")
    candidate = inspect_mtx(archive_path)[0]
    with pytest.raises(ValueError, match="neither cell-major"):
        MtxReader(candidate, temp_dir=str(tmp_path))
    assert not list(tmp_path.glob("scarf-mtx-archive-*"))
