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
from scarf.storage.count_matrix import CountMatrixPolicy
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
    # Rows are cells and columns features; cell 3 holds no counts.
    cells_by_features = [[2, 0, 4, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5]]
    np.testing.assert_array_equal(observed, cells_by_features)
    np.testing.assert_array_equal(expected, cells_by_features)


@pytest.mark.parametrize(
    ("matrix_compressed", "features_compressed", "barcodes_compressed"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_directory_triplet_supports_independent_compression(
    tmp_path: Path,
    matrix_compressed: bool,
    features_compressed: bool,
    barcodes_compressed: bool,
) -> None:
    matrix_name = "matrix.mtx.gz" if matrix_compressed else "matrix.mtx"
    _write_mex(
        tmp_path,
        [(1, 1, 7), (2, 2, 9)],
        n_features=2,
        n_cells=2,
        matrix_name=matrix_name,
    )
    for name, compressed in (
        ("features.tsv", features_compressed),
        ("barcodes.tsv", barcodes_compressed),
    ):
        if not compressed:
            continue
        plain_path = tmp_path / name
        contents = plain_path.read_text()
        plain_path.unlink()
        _write_text(tmp_path / f"{name}.gz", contents)

    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.matrixPath.endswith(".gz") is matrix_compressed
    assert candidate.featurePath.endswith(".gz") is features_compressed
    assert candidate.cellPath.endswith(".gz") is barcodes_compressed
    reader = MtxReader(candidate)
    try:
        observed = np.vstack([batch.toarray() for batch in reader.consume(1)])
    finally:
        reader.close()
    np.testing.assert_array_equal(observed, [[7, 0], [0, 9]])


def test_explicit_custom_triplet_paths_and_constructor_validation(
    tmp_path: Path,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 4), (2, 1, 3), (2, 2, 5)],
        n_features=2,
        n_cells=2,
        matrix_name="quantification.mtx.gz",
    )
    feature_path = tmp_path / "measurements.txt"
    (tmp_path / "features.tsv").replace(feature_path)
    cell_path = tmp_path / "observations.txt.gz"
    cell_contents = (tmp_path / "barcodes.tsv").read_text()
    (tmp_path / "barcodes.tsv").unlink()
    _write_text(cell_path, cell_contents)
    matrix_path = tmp_path / "quantification.mtx.gz"

    with pytest.raises(ValueError, match="requires feature_path and cell_path"):
        MtxReader(str(matrix_path))

    with pytest.raises(TypeError, match="dtype"):
        MtxReader(  # type: ignore[call-arg]
            str(matrix_path), str(feature_path), str(cell_path), dtype=np.uint16
        )

    reader = MtxReader(
        str(matrix_path),
        str(feature_path),
        str(cell_path),
    )
    try:
        observed = np.vstack([batch.toarray() for batch in reader.consume(1)])
        assert reader.matrix_dtype == np.dtype(np.uint8)
        assert reader.feature_names() == ["gene-0", "gene-1"]
        assert reader.cell_names() == ["cell-0", "cell-1"]
    finally:
        reader.close()
    np.testing.assert_array_equal(
        observed,
        np.array([[4, 3], [0, 5]], dtype=np.uint8),
    )


def test_directory_discovery_is_sorted_and_direct_file_is_selected(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, [(1, 1, 1)], n_features=1, n_cells=1)
    _write_mex(
        tmp_path,
        [(1, 1, 2)],
        n_features=1,
        n_cells=1,
        matrix_name="sample_matrix.mtx",
        prefix="sample_",
    )
    _write_text(
        tmp_path / "orphan_matrix.mtx",
        "%%MatrixMarket matrix coordinate integer general\n1 1 0\n",
    )

    candidates = inspect_mtx(tmp_path)
    assert [Path(candidate.matrixPath).name for candidate in candidates] == [
        "matrix.mtx",
        "sample_matrix.mtx",
    ]

    selected = inspect_mtx(tmp_path / "sample_matrix.mtx")
    assert len(selected) == 1
    assert Path(selected[0].featurePath).name == "sample_features.tsv"
    assert selected[0].source == str(tmp_path / "sample_matrix.mtx")


def test_archive_discovers_nested_compressed_prefixed_members_and_cleans_up(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "provider.zip"
    matrix = "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 3\n2 2 4\n"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "bundle/sample_matrix.mtx.gz",
            gzip.compress(matrix.encode()),
        )
        archive.writestr(
            "bundle/sample_genes.tsv.gz",
            gzip.compress(b"feature-0\tGene 0\nfeature-1\tGene 1\n"),
        )
        archive.writestr(
            "bundle/sample_barcodes.tsv",
            "cell-0\ncell-1\n",
        )
        archive.writestr(
            "orphan_matrix.mtx",
            "%%MatrixMarket matrix coordinate integer general\n1 1 0\n",
        )
        archive.writestr("features.tsv", "orphan\n")

    candidates = inspect_mtx(archive_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.matrixPath == "bundle/sample_matrix.mtx.gz"
    assert candidate.featurePath == "bundle/sample_genes.tsv.gz"
    assert candidate.cellPath == "bundle/sample_barcodes.tsv"
    assert candidate.archivePath == str(archive_path)

    reader = MtxReader(candidate, temp_dir=str(tmp_path))
    assert list(tmp_path.glob("scarf-mtx-archive-*"))
    try:
        observed = np.vstack([batch.toarray() for batch in reader.consume(1)])
    finally:
        reader.close()
    np.testing.assert_array_equal(observed, [[3, 0], [0, 4]])
    assert not list(tmp_path.glob("scarf-mtx-archive-*"))


@pytest.mark.parametrize(
    ("sidecar_name", "contents", "expected_names", "expected_type"),
    [
        (
            "features.tsv",
            "feature-0\nfeature-1\n",
            ["feature-0", "feature-1"],
            "Gene Expression",
        ),
        (
            "genes.tsv.gz",
            "feature-0\tGene zero\nfeature-1\tGene one\n",
            ["Gene zero", "Gene one"],
            "Gene Expression",
        ),
        (
            "peaks.bed.gz",
            "chr1\t10109\t10357\nchr1\t180730\t181630\n",
            ["chr1:10109-10357", "chr1:180730-181630"],
            "Peaks",
        ),
    ],
)
def test_feature_sidecar_suffix_and_column_fallbacks(
    tmp_path: Path,
    sidecar_name: str,
    contents: str,
    expected_names: list[str],
    expected_type: str,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 1), (2, 1, 2)],
        n_features=2,
        n_cells=1,
    )
    original = tmp_path / "features.tsv"
    if sidecar_name != original.name:
        original.unlink()
    _write_text(tmp_path / sidecar_name, contents)

    candidate = inspect_mtx(tmp_path)[0]
    assert Path(candidate.featurePath).name == sidecar_name
    reader = MtxReader(candidate)
    try:
        assert reader.feature_names() == expected_names
        assert reader.feature_types() == [expected_type, expected_type]
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("chr1\t10109\nchr1\t180730\n", "chrom, start, and end columns"),
        ("chr1\tstart\t10357\nchr1\t180730\t181630\n", "must hold integers"),
    ],
)
def test_peak_bed_sidecars_are_validated(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    _write_mex(tmp_path, [(1, 1, 1), (2, 1, 2)], n_features=2, n_cells=1)
    (tmp_path / "features.tsv").unlink()
    _write_text(tmp_path / "peaks.bed", contents)

    candidate = inspect_mtx(tmp_path)[0]
    with pytest.raises(ValueError, match=message):
        reader = MtxReader(candidate)
        try:
            reader.feature_names()
        finally:
            reader.close()


def test_parse_modern_compressed_names_and_feature_name_fallback(
    tmp_path: Path,
) -> None:
    _write_text(
        tmp_path / "count_matrix.mtx.gz",
        "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 2\n2 2 3\n",
    )
    _write_text(
        tmp_path / "all_genes.csv.gz",
        "feature_id\nfeature-0\nfeature-1\n",
    )
    _write_text(
        tmp_path / "cell_metadata.csv.gz",
        "bc_index,donor\ncell-0,A\ncell-1,B\n",
    )

    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.matrixOrientation == "cellsByFeatures"
    assert Path(candidate.featurePath).name == "all_genes.csv.gz"
    assert Path(candidate.cellPath).name == "cell_metadata.csv.gz"
    assert candidate.cellIdKeys == ("bc_index",)

    reader = MtxReader(candidate)
    try:
        assert reader.feature_ids() == ["feature-0", "feature-1"]
        assert reader.feature_names() == ["feature-0", "feature-1"]
        assert reader.feature_types() == ["Gene Expression", "Gene Expression"]
        np.testing.assert_array_equal(
            dict(reader.get_cell_columns())["donor"],
            ["A", "B"],
        )
    finally:
        reader.close()


def test_cells_by_features_real_counts_zero_based_indices_and_dtype(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, [], n_features=2, n_cells=3)
    _write_text(
        tmp_path / "matrix.mtx",
        "%%MatrixMarket matrix coordinate real general\n"
        "% zero-based fixture\n"
        "3 2 3\n"
        "0 0 1.0\n"
        "1 1 2.0\n"
        "2 0 3.0\n",
    )
    candidate = inspect_mtx(tmp_path)[0]
    assert candidate.matrixOrientation == "cellsByFeatures"

    with pytest.raises(ValueError, match="outside the declared dimensions"):
        MtxReader(candidate)

    reader = MtxReader(candidate, index_offset=0)
    try:
        batches = list(reader.consume(2, lines_in_mem=1))
    finally:
        reader.close()
    # Real-valued integral counts are held like integer ones.
    assert all(batch.dtype == np.dtype(np.uint8) for batch in batches)
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in batches]),
        [[1, 0], [0, 2], [3, 0]],
    )


@pytest.mark.parametrize(
    ("coordinates", "is_filtered", "order"),
    [
        ([(1, 1, 1), (3, 1, 2), (2, 2, 3), (4, 3, 4)], True, "cellMajor"),
        ([(1, 1, 1), (2, 2, 3), (3, 1, 2), (4, 3, 4)], False, "featureMajor"),
    ],
)
def test_construction_scans_the_matrix_once_for_every_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coordinates: list[tuple[int, int, int]],
    is_filtered: bool,
    order: str,
) -> None:
    from scarf.readers.mtx import _MtxEngine

    _write_mex(tmp_path, coordinates, n_cells=3)
    scans: list[int] = []
    original = _MtxEngine._scan_matrix

    def counted_scan(self, lines_in_mem):
        scans.append(lines_in_mem)
        return original(self, lines_in_mem)

    monkeypatch.setattr(_MtxEngine, "_scan_matrix", counted_scan)
    reader = MtxReader(
        inspect_mtx(tmp_path)[0], is_filtered=is_filtered, filtering_cutoff=0
    )
    try:
        # One scan fixes the order, the cells, and the count range.
        assert len(scans) == 1
        assert reader.coordinateOrder == order
        assert reader.count_value_ranges(0)[0].maximum == 4
        reader._prepare_sparse_import()
        observed = np.vstack(
            [batch.toarray() for batch in reader.consume(2, lines_in_mem=2)]
        )
    finally:
        reader.close()
    assert len(scans) == 1
    np.testing.assert_array_equal(
        observed,
        [[1, 0, 2, 0], [0, 3, 0, 0], [0, 0, 0, 4]],
    )


def test_matrix_market_counts_past_uint32_import_as_uint64(tmp_path: Path) -> None:
    value = int(np.iinfo(np.uint32).max) + 1
    _write_mex(
        tmp_path,
        [(1, 1, value)],
        n_features=1,
        n_cells=1,
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        assert reader.matrix_dtype == np.uint64
        observed = list(reader.consume(1))[0].toarray()
        store = MemoryStore()
        MtxToZarr(reader, store, nthreads=1).dump()
    finally:
        reader.close()
    np.testing.assert_array_equal(observed, np.array([[value]], dtype=np.uint64))
    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == np.uint64
    np.testing.assert_array_equal(counts[:], [[value]])


def test_matrix_market_counts_that_change_after_the_scan_raise(
    tmp_path: Path,
) -> None:
    _write_mex(tmp_path, [(1, 1, 200), (2, 2, 5)], n_features=2, n_cells=2)
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    try:
        assert reader.matrix_dtype == np.uint8
        # The file is rewritten with a count that the scanned dtype cannot hold.
        _write_mex(tmp_path, [(1, 1, 300), (2, 2, 5)], n_features=2, n_cells=2)
        with pytest.raises(OverflowError, match="the file changed"):
            list(reader.consume(1))
    finally:
        reader.close()


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
    # Four kept uint8 counts with int32 indices, and four int64 row pointers.
    assert feature_reader.temporaryDiskBytes == 4 * (1 + 4) + 4 * 8
    prepared = list(feature_reader.consume(2, lines_in_mem=2))
    assert all(batch.tocsr().has_sorted_indices for batch in prepared)
    assert all(batch.dtype == np.dtype(np.uint8) for batch in prepared)

    stores = [MemoryStore(), MemoryStore()]
    MtxToZarr(
        feature_reader,
        stores[0],
        mem_budget="64M",
        policy=CountMatrixPolicy(unitBytes=32, chunkBytes=16),
        lines_in_mem=2,
    ).dump()
    MtxToZarr(
        cell_reader,
        stores[1],
        mem_budget="64M",
        policy=CountMatrixPolicy(unitBytes=32, chunkBytes=16),
        lines_in_mem=2,
    ).dump()

    first = zarr.open_group(store=stores[0], mode="r")["RNA/counts"][:]
    second = zarr.open_group(store=stores[1], mode="r")["RNA/counts"][:]
    # Feature 2 of cell 2 sums its two entries, and cell 4, whose total does
    # not exceed the cutoff, is dropped.
    expected = [[1, 0, 4, 0], [0, 3, 0, 0], [2, 0, 0, 0]]
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
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
        policy=CountMatrixPolicy(unitBytes=16, chunkBytes=8),
        lines_in_mem=2,
    ).dump()

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


@pytest.mark.parametrize("archive_source", [False, True], ids=["directory", "archive"])
@pytest.mark.parametrize("missing_name", ["matrix.mtx", "features.tsv", "barcodes.tsv"])
def test_inspection_rejects_each_missing_triplet_member(
    tmp_path: Path,
    archive_source: bool,
    missing_name: str,
) -> None:
    members = {
        "matrix.mtx": ("%%MatrixMarket matrix coordinate integer general\n1 1 0\n"),
        "features.tsv": "feature-0\n",
        "barcodes.tsv": "cell-0\n",
    }
    members.pop(missing_name)
    if archive_source:
        source = tmp_path / "incomplete.zip"
        with zipfile.ZipFile(source, mode="w") as archive:
            for name, contents in members.items():
                archive.writestr(name, contents)
    else:
        source = tmp_path
        for name, contents in members.items():
            _write_text(tmp_path / name, contents)

    with pytest.raises(ValueError, match="No complete Matrix Market"):
        inspect_mtx(source)


def test_archive_missing_selected_member_cleans_partial_extraction(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "changed.zip"
    _write_zip_mex(archive_path)
    candidate = inspect_mtx(archive_path)[0]

    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            candidate.matrixPath,
            "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n2 2 2\n",
        )
        archive.writestr(candidate.cellPath, "cell-0\ncell-1\n")

    with pytest.raises(ValueError, match="missing selected members"):
        MtxReader(candidate, temp_dir=str(tmp_path))
    assert not list(tmp_path.glob("scarf-mtx-archive-*"))


@pytest.mark.parametrize(
    ("matrix_text", "message"),
    [
        (
            "not a Matrix Market banner\n1 1 0\n",
            "coordinate matrix format",
        ),
        (
            "%%MatrixMarket matrix array integer general\n1 1 0\n",
            "coordinate matrix format",
        ),
        (
            "%%MatrixMarket matrix coordinate complex general\n1 1 0\n",
            "integer or real coordinate values",
        ),
        (
            "%%MatrixMarket matrix coordinate integer symmetric\n1 1 0\n",
            "general symmetry",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n",
            "dimensions line must contain three integers",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\none 1 0\n",
            "dimensions line must contain three integers",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n-1 1 0\n",
            "dimensions cannot be negative",
        ),
    ],
)
def test_malformed_matrix_market_headers_are_rejected(
    tmp_path: Path,
    matrix_text: str,
    message: str,
) -> None:
    _write_mex(tmp_path, [], n_features=1, n_cells=1)
    _write_text(tmp_path / "matrix.mtx", matrix_text)

    with pytest.raises(ValueError, match=message):
        inspect_mtx(tmp_path)


@pytest.mark.parametrize(
    ("field", "coordinate", "message"),
    [
        ("integer", "1 1", "Could not parse Matrix Market coordinates"),
        ("integer", "axis 1 1", "Could not parse Matrix Market coordinates"),
        ("integer", "1 1 1.5", "Could not parse Matrix Market coordinates"),
        ("integer", "2 1 1", "outside the declared dimensions"),
        ("integer", "1 2 1", "outside the declared dimensions"),
        ("real", "1 1 1.5", "finite non-negative integers"),
        ("real", "1 1 nan", "finite non-negative integers"),
        ("real", "1 1 inf", "finite non-negative integers"),
    ],
)
def test_malformed_matrix_market_counts_and_indices_are_rejected(
    tmp_path: Path,
    field: str,
    coordinate: str,
    message: str,
) -> None:
    _write_mex(tmp_path, [], n_features=1, n_cells=1)
    _write_text(
        tmp_path / "matrix.mtx",
        f"%%MatrixMarket matrix coordinate {field} general\n1 1 1\n{coordinate}\n",
    )
    candidate = inspect_mtx(tmp_path)[0]

    with pytest.raises(ValueError, match=message):
        MtxReader(candidate)


@pytest.mark.parametrize(
    ("declared_entries", "coordinates", "observed_entries"),
    [
        (2, "1 1 1\n", 1),
        (1, "1 1 1\n1 1 2\n", 2),
    ],
)
def test_matrix_market_coordinate_count_must_match_header(
    tmp_path: Path,
    declared_entries: int,
    coordinates: str,
    observed_entries: int,
) -> None:
    _write_mex(tmp_path, [], n_features=1, n_cells=1)
    _write_text(
        tmp_path / "matrix.mtx",
        "%%MatrixMarket matrix coordinate integer general\n"
        f"1 1 {declared_entries}\n"
        f"{coordinates}",
    )
    candidate = inspect_mtx(tmp_path)[0]

    with pytest.raises(
        ValueError,
        match=(
            f"header declares {declared_entries} entries, "
            f"but {observed_entries} were read"
        ),
    ):
        MtxReader(candidate)


def test_duplicate_cell_ids_are_rejected_after_dimension_validation(
    tmp_path: Path,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 1), (1, 2, 2)],
        n_features=1,
        n_cells=2,
    )
    (tmp_path / "barcodes.tsv").write_text("cell-0\ncell-0\n")
    candidate = inspect_mtx(tmp_path)[0]

    with pytest.raises(ValueError, match="Cell IDs must contain unique values"):
        MtxReader(candidate)


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
    destination = tmp_path / "planning.zarr"
    # The layout is admitted before the destination or temporary rows exist.
    with pytest.raises(MemoryError, match="requested count-matrix policy"):
        MtxToZarr(
            planning_reader,
            str(destination),
            mem_budget=1,
            policy=CountMatrixPolicy(unitBytes=8, chunkBytes=8),
        )
    assert not list(tmp_path.glob("scarf-mtx-csr-*"))
    assert not destination.exists()
    planning_reader.close()

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
        policy=CountMatrixPolicy(unitBytes=8, chunkBytes=8),
        lines_in_mem=1,
    )
    with pytest.raises(RuntimeError, match="injected cancellation"):
        writer.dump()
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
    np.testing.assert_array_equal(
        np.vstack([batch.toarray() for batch in completed]),
        [[1, 0, 4], [0, 3, 0], [2, 0, 0]],
    )


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
        policy=CountMatrixPolicy(unitBytes=8, chunkBytes=8),
        lines_in_mem=2,
    ).dump()

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
    # Feature i of the single cell holds the count i, one feature per assay.
    for count, assay in enumerate(
        ("RNA", "CRISPR", "HTO", "ANTIGEN", "CUSTOM", "ADT"), start=1
    ):
        np.testing.assert_array_equal(root[f"{assay}/counts"][:], [[count]])
        np.testing.assert_array_equal(
            root[f"{assay}/featureData/ids"][:], [f"feature-{count - 1}"]
        )
    assert "feature_call" not in root["cellData"]
    datastore = DataStore(
        store,
        default_assay="RNA",
        min_features_per_cell=0,
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

    store = MemoryStore()
    MtxToZarr(reader, store, mem_budget="64M", lines_in_mem=2).dump()
    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(
        root["CRISPR/featureData/feature_type"][:],
        ["CRISPR Guide Capture"],
    )
    np.testing.assert_array_equal(root["RNA/featureData/feature_type"][:], ["mRNA"])


def test_feature_reference_covers_only_feature_barcode_features(
    tmp_path: Path,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 3), (2, 1, 1), (3, 1, 7)],
        n_features=3,
        n_cells=1,
        feature_types=["Gene Expression", "Gene Expression", "Antibody Capture"],
    )
    # A 10x Feature Reference lists Feature Barcode features only.
    (tmp_path / "feature_reference.csv").write_text(
        "id,name,read,pattern,sequence,feature_type\n"
        "feature-2,CD3,R2,5PNNNNNNNNNN(BC),AACAAGACCCTTGAG,Antibody Capture\n"
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    store = MemoryStore()
    MtxToZarr(reader, store, mem_budget="64M", lines_in_mem=2).dump()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(
        root["ADT/featureData/sequence"][:], ["AACAAGACCCTTGAG"]
    )
    np.testing.assert_array_equal(root["ADT/featureData/names"][:], ["gene-2"])
    np.testing.assert_array_equal(
        root["ADT/featureData/feature_type"][:],
        ["Antibody Capture"],
    )
    assert "sequence" not in root["RNA/featureData"]

    (tmp_path / "feature_reference.csv").write_text("id,sequence\nunknown,ACGT\n")
    with pytest.raises(ValueError, match="not matrix features: unknown"):
        MtxReader(inspect_mtx(tmp_path)[0])


def test_feature_reference_column_with_a_separator_is_renamed(
    tmp_path: Path,
) -> None:
    _write_mex(
        tmp_path,
        [(1, 1, 3), (2, 1, 7)],
        n_features=2,
        n_cells=1,
        feature_types=["Gene Expression", "Antibody Capture"],
    )
    (tmp_path / "feature_reference.csv").write_text(
        "id,name,read,pattern,sequence,feature_type,tag/x\n"
        "feature-1,CD3,R2,5PNNNNNNNNNN(BC),AACAAGACCCTTGAG,Antibody Capture,T1\n"
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    store = MemoryStore()
    MtxToZarr(reader, store, mem_budget="64M", lines_in_mem=2).dump()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["ADT/featureData/tag_x"][:], ["T1"])
    assert list(root["ADT/featureData"].group_keys()) == []
    assert "tag_x" not in root["RNA/featureData"]


def test_feature_reference_column_empty_for_every_feature_takes_no_key(
    tmp_path: Path,
) -> None:
    from scarf.utils.logging import logger

    _write_mex(
        tmp_path,
        [(1, 1, 3), (2, 1, 7)],
        n_features=2,
        n_cells=1,
        feature_types=["Gene Expression", "Antibody Capture"],
    )
    (tmp_path / "feature_reference.csv").write_text(
        "id,name,read,pattern,sequence,feature_type,tag/x,tag\\x\n"
        "feature-1,CD3,R2,5PNNNNNNNNNN(BC),AACAAGACCCTTGAG,Antibody Capture,,T1\n"
    )
    reader = MtxReader(inspect_mtx(tmp_path)[0])
    store = MemoryStore()
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        MtxToZarr(reader, store, mem_budget="64M", lines_in_mem=2).dump()
    finally:
        logger.remove(handler)

    root = zarr.open_group(store=store, mode="r")
    # The empty 'tag/x' column is never written, so 'tag\\x' keeps the base key.
    np.testing.assert_array_equal(root["ADT/featureData/tag_x"][:], ["T1"])
    assert "tag_x_2" not in root["ADT/featureData"]
    assert not any("'tag/x'" in message for message in messages)


def test_symbolic_link_sidecars_keep_their_names(tmp_path: Path) -> None:
    content = tmp_path / "objects"
    content.mkdir()
    _write_mex(content, [(1, 1, 2)], n_features=1, n_cells=1)
    work = tmp_path / "work"
    work.mkdir()
    for name in ("matrix.mtx", "features.tsv", "barcodes.tsv"):
        target = content / f"sha256-{name.replace('.', '-')}"
        (content / name).rename(target)
        (work / name).symlink_to(target)

    candidates = inspect_mtx(work)

    assert [Path(candidate.matrixPath).name for candidate in candidates] == [
        "matrix.mtx"
    ]
    reader = MtxReader(candidates[0])
    try:
        assert np.vstack([batch.toarray() for batch in reader.consume(1)]).tolist() == [
            [2]
        ]
    finally:
        reader.close()


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
            policy=CountMatrixPolicy(unitBytes=8, chunkBytes=8),
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


def test_mtx_import_fits_the_parse_buffer_that_it_reads(tmp_path):
    values = np.arange(200 * 50).reshape(200, 50) % 7 + 1
    coordinates = [
        (feature + 1, cell + 1, int(values[cell, feature]))
        for cell in range(200)
        for feature in range(50)
    ]
    directory = tmp_path / "mex"
    directory.mkdir()
    _write_mex(directory, coordinates, n_features=50, n_cells=200)
    reader = MtxReader(inspect_mtx(directory)[0])
    store = MemoryStore()
    try:
        # The default buffer of 100,000 parsed lines needs about 9.7 MB, so the
        # import is refused before the destination exists.
        destination = tmp_path / "default.zarr"
        with pytest.raises(MemoryError, match="count shards of one row"):
            MtxToZarr(reader, str(destination), mem_budget="8M", nthreads=1)
        assert not destination.exists()
        with pytest.raises(ValueError, match="lines_in_mem must be positive"):
            MtxToZarr(reader, str(tmp_path / "invalid.zarr"), lines_in_mem=0)
        # A smaller buffer fits the same budget.
        MtxToZarr(
            reader, store, mem_budget="8M", nthreads=1, lines_in_mem=10_000
        ).dump()
    finally:
        reader.close()
    np.testing.assert_array_equal(
        zarr.open_group(store=store, mode="r")["RNA/counts"][:], values
    )
