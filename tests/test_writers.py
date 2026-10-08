import os

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

from scarf.readers import CSVReader
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.utils.count_values import CountValueRange
from scarf.writers import (
    CSVtoZarr,
    CrToZarr,
    SubsetZarr,
    subset_assay_zarr,
)


def _value_range(values: np.ndarray) -> CountValueRange:
    """Return the count range that a test reader reports for its values."""
    value_range = CountValueRange()
    value_range.update(values)
    return value_range


class _FakeCells:
    def __init__(
        self, n_cells: int, columns: dict[str, np.ndarray] | None = None
    ) -> None:
        self.N = n_cells
        self._columns = columns or {}

    def fetch_all(self, key: str) -> np.ndarray:
        return self._columns[key]


class _FakeAssay:
    def __init__(
        self,
        name: str,
        n_cells: int,
        columns: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self.cells = _FakeCells(n_cells, columns)
        self.z = zarr.group()
        self.matrixGroup = self.z


@pytest.fixture(scope="module")
def memory_source():
    """An in-memory DataStore with one RNA assay of two cells, which tests only read."""
    from scipy.sparse import csr_matrix

    from scarf import DataStore
    from scarf.writers import SparseToZarr

    store = MemoryStore()
    SparseToZarr(
        csr_matrix(np.arange(1, 7, dtype=np.uint8).reshape(2, 3)),
        store,
        ["c0", "c1"],
        ["f1", "f2", "f3"],
        nthreads=1,
    ).dump()
    return DataStore(store, min_features_per_cell=0, nthreads=1)


def _assert_counts_equal(array, expected) -> None:
    """Compare stored counts with a SciPy matrix, one stored chunk at a time."""
    assert tuple(array.shape) == expected.shape
    expected = expected.tocsr()
    rows, columns = (int(value) for value in array.chunks)
    for start in range(0, array.shape[0], rows):
        band = expected[start : start + rows]
        for column in range(0, array.shape[1], columns):
            stored = array[start : start + rows, column : column + columns]
            wanted = band[:, column : column + columns].toarray()
            if not np.array_equal(stored, wanted):
                np.testing.assert_array_equal(stored, wanted)


def _smallest_unsigned_dtype(counts) -> np.dtype:
    """Return the narrowest unsigned dtype that holds non-negative integer counts."""
    return np.dtype(np.min_scalar_type(int(counts.max())))


def _read_cellranger_h5(path):
    """Read a Cell Ranger 3 HDF5 matrix with h5py and SciPy alone."""
    import h5py
    from scipy.sparse import csc_matrix

    with h5py.File(path, mode="r") as h5:
        group = h5["matrix"]
        features_by_cells = csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(int(value) for value in group["shape"][:]),
        )
        barcodes = group["barcodes"][:].astype(str)
        features = {
            key: group[f"features/{key}"][:].astype(str)
            for key in ("id", "name", "feature_type")
        }
    return features_by_cells.T.tocsr(), barcodes, features


def test_crtozarr(crh5_reader, tmp_path):
    fn = str(tmp_path / "dummy_1K_pbmc_citeseq.zarr")
    CrToZarr(crh5_reader, zarr_loc=fn).dump()

    counts, barcodes, features = _read_cellranger_h5(crh5_reader.h5obj.filename)
    root = zarr.open_group(fn, mode="r")
    assert set(root.group_keys()) == {"cellData", "RNA", "ADT"}
    assert root.attrs["assayTypes"] == {"RNA": "RNA", "ADT": "ADT"}
    np.testing.assert_array_equal(root["cellData/ids"][:], barcodes)
    for assay, feature_type in (
        ("RNA", "Gene Expression"),
        ("ADT", "Antibody Capture"),
    ):
        columns = np.flatnonzero(features["feature_type"] == feature_type)
        expected = counts[:, columns]
        stored = root[f"{assay}/counts"]
        assert stored.dtype == _smallest_unsigned_dtype(expected)
        _assert_counts_equal(stored, expected)
        feature_data = root[f"{assay}/featureData"]
        np.testing.assert_array_equal(feature_data["ids"][:], features["id"][columns])
        np.testing.assert_array_equal(
            feature_data["names"][:], features["name"][columns]
        )
        np.testing.assert_array_equal(
            feature_data["feature_type"][:], features["feature_type"][columns]
        )
    # Only the RNA assay stores the feature-major transpose; the import
    # contract tests compare its values with every writer's input.
    assert root["RNA/countsT"].attrs["complete"] is True
    assert root["RNA/countsT"].shape == root["RNA/counts"].shape[::-1]
    assert "countsT" not in root["ADT"]


def test_crtozarr_fromdir(crdir_reader, mtx_dir, tmp_path):
    import gzip
    from pathlib import Path

    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    fn = str(tmp_path / "1K_pbmc_citeseq_dir.zarr")
    CrToZarr(crdir_reader, zarr_loc=fn).dump()

    source = Path(mtx_dir)
    expected = csr_matrix(mmread(source / "matrix.mtx.gz", spmatrix=False).T)
    with gzip.open(source / "barcodes.tsv.gz", "rt") as handle:
        barcodes = handle.read().split()
    with gzip.open(source / "features.tsv.gz", "rt") as handle:
        features = [line.rstrip("\n").split("\t") for line in handle]
    root = zarr.open_group(fn, mode="r")
    assert set(root.group_keys()) == {"cellData", "RNA"}
    np.testing.assert_array_equal(root["cellData/ids"][:], barcodes)
    assert root["RNA/counts"].dtype == _smallest_unsigned_dtype(expected)
    _assert_counts_equal(root["RNA/counts"], expected)
    np.testing.assert_array_equal(
        root["RNA/featureData/ids"][:], [row[0] for row in features]
    )
    np.testing.assert_array_equal(
        root["RNA/featureData/names"][:], [row[1] for row in features]
    )


def test_crtozarr_writes_a_directory_without_cells(toy_crdir_empty, tmp_path):
    from scarf import DataStore
    from scarf.writers import CrToZarr

    fn = str(tmp_path / "empty.zarr")
    CrToZarr(toy_crdir_empty, zarr_loc=fn, nthreads=1).dump()

    store = DataStore(fn, default_assay="RNA")
    assert store.cells.N == 0
    assert store.assay_names == ["ADT", "RNA"]


def test_crtozarr_preserves_exact_counts_metadata_and_transpose():
    import pandas as pd
    from scipy.sparse import coo_matrix

    values = np.array(
        [[1, 0, 2], [0, 3, 0], [4, 5, 6]],
        dtype=np.uint16,
    )

    class ExactReader:
        nCells, nFeatures = values.shape
        matrix_dtype = values.dtype
        assayFeats = pd.DataFrame(
            {"RNA": ["Gene Expression", 0, values.shape[1], values.shape[1]]},
            index=["type", "start", "end", "nFeatures"],
        )

        def cell_names(self):
            return ["c1", "c2", "c3"]

        def feature_ids(self, assay_name):
            return ["f1", "f2", "f3"]

        def feature_names(self, assay_name):
            return ["g1", "g2", "g3"]

        def consume(self, batch_size, lines_in_mem):
            for start in range(0, self.nCells, batch_size):
                yield coo_matrix(values[start : start + batch_size])

        def count_value_ranges(self, maxBytes, featureGroups=None):
            return [_value_range(values)]

        def max_window_nnz(self, window_rows):
            width = min(window_rows, self.nCells)
            return max(
                np.count_nonzero(values[start : start + width])
                for start in range(self.nCells - width + 1)
            )

        def producer_staging_bytes(self, batch_size, lines_in_mem):
            return 0

    store = MemoryStore()
    # The counts store as uint8, so 6 bytes give two-row shards.
    writer = CrToZarr(
        ExactReader(),
        zarr_loc=store,
        policy=CountMatrixPolicy(unitBytes=6, chunkBytes=6),
    )
    writer.dump(batch_size=2)

    root = zarr.open_group(store=store, mode="r")
    assert root["RNA/counts"].dtype == np.uint8
    assert root["RNA/counts"].metadata.shards[0] == 2
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(root["cellData/ids"][:], ["c1", "c2", "c3"])
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], ["f1", "f2", "f3"])


def test_h5adtozarr(h5ad_reader, bastidas_ponce_data, tmp_path):
    import h5py
    from scipy.sparse import csr_matrix

    from scarf.writers import H5adImportResult, H5adToZarr

    fn = str(tmp_path / "bastidas.zarr")
    writer = H5adToZarr(h5ad_reader, zarr_loc=fn)
    result = writer.dump()

    assert isinstance(result, H5adImportResult)
    assert result.assayNames == ("RNA",)
    assert result.analysisAssay is None
    assert result.embeddingArtifacts == {}
    assert result.clusterArtifacts == {}

    # The AnnData 0.6 file holds its tables as compound datasets.
    with h5py.File(bastidas_ponce_data, mode="r") as h5:
        obs = h5["obs"][:]
        var = h5["var"][:]
        expected = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=(obs.shape[0], var.shape[0]),
        )
    root = zarr.open_group(fn, mode="r")
    counts = root["RNA/counts"]
    # The float32 source holds integral counts up to 2286.
    assert counts.dtype == np.uint16
    _assert_counts_equal(counts, expected)
    np.testing.assert_array_equal(
        root["cellData/ids"][:], [value.decode() for value in obs["index"]]
    )
    np.testing.assert_array_equal(
        root["RNA/featureData/ids"][:], [value.decode() for value in var["index"]]
    )
    for column in ("S_score", "G2M_score"):
        np.testing.assert_array_equal(root[f"cellData/{column}"][:], obs[column])


def test_h5adtozarr_splits_noncontiguous_feature_types():
    import tempfile
    from pathlib import Path

    import h5py
    from scipy.sparse import csr_matrix

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = np.array(
        [
            [1, 2, 0, 3],
            [4, 0, 5, 0],
            [0, 6, 7, 8],
        ],
        dtype=np.uint16,
    )
    matrix = csr_matrix(values)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "multi.h5ad"
        with h5py.File(path, mode="w") as h5:
            sparse = h5.create_group("X")
            sparse.attrs["encoding-type"] = "csr_matrix"
            sparse.attrs["shape"] = matrix.shape
            sparse.create_dataset("data", data=matrix.data)
            sparse.create_dataset("indices", data=matrix.indices)
            sparse.create_dataset("indptr", data=matrix.indptr)

            obs = h5.create_group("obs")
            obs.create_dataset("_index", data=np.array([b"c1", b"c2", b"c3"]))
            obs.create_dataset("batch", data=np.array([b"A", b"A", b"B"]))

            var = h5.create_group("var")
            var.create_dataset(
                "_index",
                data=np.array([b"f1", b"a1", b"f2", b"a2"]),
            )
            var.create_dataset(
                "feature_name",
                data=np.array([b"g1", b"p1", b"g2", b"p2"]),
            )
            var.create_dataset(
                "feature_types",
                data=np.array(
                    [
                        b"Gene Expression",
                        b"Antibody Capture",
                        b"Gene Expression",
                        b"Antibody Capture",
                    ]
                ),
            )
            var.create_dataset(
                "chromosome",
                data=np.array([b"1", b"na", b"2", b"na"]),
            )

        reader = H5adReader(str(path), feature_name_key="feature_name")
        store = MemoryStore()
        try:
            assert tuple(
                reader.assay_feature_slices(
                    "feature_types",
                    {"Antibody Capture": "HTO"},
                )
            ) == ("RNA", "HTO")
            writer = H5adToZarr(
                reader,
                zarr_loc=store,
                assay_name="ignored",
                assay_split_key="feature_types",
                policy=CountMatrixPolicy(unitBytes=8, chunkBytes=8),
            )
            writer.dump(batch_size=2)
        finally:
            reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    assert set(root.group_keys()) == {"artifacts", "cellData", "RNA", "ADT"}
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, [0, 2]])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, [1, 3]])
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    assert "countsT" not in root["ADT"]
    np.testing.assert_array_equal(root["cellData/batch"][:], ["A", "A", "B"])
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], ["f1", "f2"])
    np.testing.assert_array_equal(root["ADT/featureData/ids"][:], ["a1", "a2"])
    np.testing.assert_array_equal(
        root["RNA/featureData/chromosome"][:],
        ["1", "2"],
    )
    np.testing.assert_array_equal(
        root["ADT/featureData/feature_types"][:],
        ["Antibody Capture", "Antibody Capture"],
    )


def _write_h5ad(
    path,
    values: np.ndarray,
    *,
    encoding: str = "csr",
    feature_types: list[bytes] | None = None,
):
    """Write a minimal AnnData file with the requested matrix encoding."""
    import h5py
    from scipy.sparse import csc_matrix, csr_matrix

    n_cells, n_feats = values.shape
    with h5py.File(path, mode="w") as h5:
        if encoding == "dense":
            h5.create_dataset("X", data=values)
        else:
            matrix = csr_matrix(values) if encoding == "csr" else csc_matrix(values)
            group = h5.create_group("X")
            group.attrs["encoding-type"] = f"{encoding}_matrix"
            group.attrs["shape"] = values.shape
            group.create_dataset("data", data=matrix.data)
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)

        obs = h5.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.array([f"c{i}".encode() for i in range(n_cells)]),
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index",
            data=np.array([f"f{i}".encode() for i in range(n_feats)]),
        )
        var.create_dataset(
            "feature_name",
            data=np.array([f"g{i}".encode() for i in range(n_feats)]),
        )
        if feature_types is not None:
            var.create_dataset("feature_types", data=np.array(feature_types))
    return path


def test_h5ad_import_returns_cold_loadable_analytical_artifacts(
    tmp_path,
    monkeypatch,
):
    import h5py

    from scarf import DataStore
    from scarf.readers import H5adReader
    from scarf.storage import ArtifactRef
    from scarf.writers import H5adImportResult, H5adToZarr

    counts = np.array(
        [[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]],
        dtype=np.uint16,
    )
    umap = np.array(
        [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4]],
        dtype=np.float32,
    )
    tsne = np.array(
        [[1, 5], [2, 6], [3, 7], [4, 8]],
        dtype=np.int16,
    )
    source = _write_h5ad(tmp_path / "analytical.h5ad", counts)
    with h5py.File(source, mode="r+") as h5:
        obs = h5["obs"]
        obs.create_dataset("batch", data=np.array([b"A", b"A", b"B", b"B"]))
        clusters = obs.create_group("leiden")
        clusters.create_dataset("codes", data=np.array([0, 1, -1, 1], dtype=np.int8))
        clusters.create_dataset("categories", data=np.array([b"alpha", b"beta"]))
        obsm = h5.create_group("obsm")
        obsm.create_dataset("X_umap", data=umap)
        obsm.create_dataset("X_tsne", data=tsne)
        obsm.create_dataset("X_pca", data=np.arange(12).reshape(4, 3))

    destination = tmp_path / "analytical.zarr"
    reader = H5adReader(
        str(source),
        feature_name_key="feature_name",
        embedding_roles={"X_umap": "umap", "X_tsne": "tsne"},
        cluster_keys=("leiden",),
    )
    embedding_block_rows: list[int] = []
    cluster_block_rows: list[int] = []
    original_obsm_blocks = H5adReader._iter_obsm_blocks
    original_cell_block = H5adReader._cell_column_block

    def tracked_obsm_blocks(self, key, block_rows, dtype):
        for block in original_obsm_blocks(self, key, block_rows, dtype):
            embedding_block_rows.append(int(block.shape[0]))
            yield block

    def tracked_cell_block(self, key, start, stop):
        if key == "leiden":
            cluster_block_rows.append(stop - start)
        return original_cell_block(self, key, start, stop)

    monkeypatch.setattr(H5adReader, "_iter_obsm_blocks", tracked_obsm_blocks)
    monkeypatch.setattr(H5adReader, "_cell_column_block", tracked_cell_block)
    try:
        result = H5adToZarr(
            reader,
            zarr_loc=str(destination),
            mem_budget="16M",
            nthreads=1,
        ).dump(batch_size=2)
    finally:
        reader.h5.close()

    assert isinstance(result, H5adImportResult)
    assert embedding_block_rows and max(embedding_block_rows) <= 2
    assert cluster_block_rows and max(cluster_block_rows) <= 2
    assert result.assayNames == ("RNA",)
    assert result.analysisAssay == "RNA"
    assert isinstance(result.cellSelection, ArtifactRef)
    assert set(result.embeddingArtifacts) == {"X_umap", "X_tsne"}
    assert set(result.clusterArtifacts) == {"leiden"}
    assert {ref.kind for ref in result.embeddingArtifacts.values()} == {"embedding"}
    assert result.clusterArtifacts["leiden"].kind == "cluster_labels"
    with pytest.raises(TypeError):
        result.embeddingArtifacts["alias"] = result.embeddingArtifacts["X_umap"]  # type: ignore[index]

    root = zarr.open_group(str(destination), mode="r")
    cells = root["cellData"]
    assert set(cells.array_keys()) == {"I", "ids", "names", "batch"}
    np.testing.assert_array_equal(cells["batch"][:], ["A", "A", "B", "B"])
    assert "leiden" not in cells
    assert "X_umap1" not in cells
    assert "X_tsne1" not in cells
    assert "X_pca1" not in cells

    reopened = DataStore(str(destination), default_assay="RNA")

    def metadata_snapshot() -> dict[str, tuple[list[object], dict[str, object]]]:
        group = reopened.zw["cellData"]
        return {
            name: (np.asarray(group[name][:]).tolist(), dict(group[name].attrs))
            for name in group.array_keys()
        }

    metadata_before = metadata_snapshot()
    loaded_umap = reopened.load_artifact(result.embeddingArtifacts["X_umap"])
    loaded_tsne = reopened.load_artifact(result.embeddingArtifacts["X_tsne"])
    loaded_clusters = reopened.load_artifact(result.clusterArtifacts["leiden"])
    cluster_status = reopened.inspect_artifact(result.clusterArtifacts["leiden"])
    assert cluster_status.parameters == {"source": "h5ad", "source_key": "leiden"}
    assert cluster_status.inputs["cell_selection"] == result.cellSelection.to_dict()
    for group in (loaded_umap, loaded_tsne, loaded_clusters):
        assert "source_artifact" not in group.attrs
        assert "schema_version" not in group.attrs
    np.testing.assert_allclose(loaded_umap["values"][:], umap)
    np.testing.assert_allclose(loaded_tsne["values"][:], tsne.astype(np.float64))
    np.testing.assert_array_equal(
        loaded_clusters["values"][:],
        ["alpha", "beta", "", "beta"],
    )
    assert loaded_clusters["values"].attrs["missing_mask"] == (
        "__scarf_missing__values"
    )
    np.testing.assert_array_equal(
        loaded_clusters["__scarf_missing__values"][:],
        [False, False, True, False],
    )
    np.testing.assert_array_equal(
        reopened.load_artifact(result.cellSelection)["values"][:],
        np.ones(4, dtype=bool),
    )
    assert metadata_snapshot() == metadata_before


def test_h5ad_multi_assay_analytical_import_requires_explicit_assay(tmp_path):
    import h5py

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    source = _write_h5ad(
        tmp_path / "multi_analysis.h5ad",
        np.array([[1, 2], [3, 4]], dtype=np.uint16),
        feature_types=[b"Gene Expression", b"Antibody Capture"],
    )
    with h5py.File(source, mode="r+") as h5:
        h5["obs"].create_dataset("clusters", data=np.array([0, 1], dtype=np.int8))

    reader = H5adReader(
        str(source),
        feature_name_key="feature_name",
        cluster_keys=("clusters",),
    )
    try:
        with pytest.raises(ValueError, match="analysis_assay is required"):
            H5adToZarr(
                reader,
                zarr_loc=MemoryStore(),
                assay_split_key="feature_types",
            )
        result = H5adToZarr(
            reader,
            zarr_loc=MemoryStore(),
            assay_split_key="feature_types",
            analysis_assay="RNA",
        ).dump(batch_size=1)
    finally:
        reader.h5.close()

    assert result.analysisAssay == "RNA"
    assert result.clusterArtifacts["clusters"].assay == "RNA"


def test_aligned_row_windows_are_shard_aligned() -> None:
    from scarf.storage.sharding import aligned_row_windows

    windows = aligned_row_windows(12, 4, 3)
    assert windows == [(0, 4), (4, 8), (8, 12)]
    assert aligned_row_windows(0, 4, 2) == []
    assert aligned_row_windows(5, 4, 8) == [(0, 4), (4, 5)]


# Sizes a 12-row uint8 assay of three features into (4, 3) row shards.
_SHARD_BAND_BUDGET = {
    "mem_budget": 1024**2,
    "nthreads": 4,
    "policy": CountMatrixPolicy(unitBytes=12, chunkBytes=12),
}


class _Pipe:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.messages: list[object] = []
        self.closed = False
        self.fail_send = fail_send

    def send(self, item: object) -> None:
        if self.fail_send:
            raise OSError("closed")
        self.messages.append(item)

    def close(self) -> None:
        self.closed = True


def _band_counts(n_cells: int, n_feats: int) -> np.ndarray:
    """Return small counts, which every encoding imports as uint8."""
    rng = np.random.default_rng(7)
    values = rng.integers(0, 5, size=(n_cells, n_feats), dtype=np.uint32)
    values[4:8] = 0
    return values


def test_h5ad_parallel_producers_stop_when_consumer_closes(tmp_path):
    import time
    from multiprocessing import active_children
    from multiprocessing.connection import wait
    from threading import Thread

    from scarf.readers import H5adReader
    from scarf.storage.schema import load_count_array
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "early_close.h5ad", values, encoding="csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        writer = H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET)
        destination = load_count_array(writer.z, "RNA", None)
        iterator = writer._parallel_count_bands(
            2,
            {"RNA": destination},
            None,
            [(0, 4), (4, 8), (8, 12)],
        )
        others = set(active_children())
        next(iterator)
        # Every producer has started; one may already have finished its rows.
        producers = [child for child in active_children() if child not in others]
        closer = Thread(target=iterator.close, daemon=True)
        closer.start()
        # A producer still starting under load sees the stop only once it
        # runs, so closing can take seconds. The deadline bounds only a hang:
        # the wait ends as soon as every producer has exited.
        deadline = time.monotonic() + 60.0
        running = {producer.sentinel for producer in producers}
        while running and time.monotonic() < deadline:
            running.difference_update(wait(running, deadline - time.monotonic()))
        assert not running, "H5AD producers outlived the closed consumer"
        closer.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not closer.is_alive()
    finally:
        reader.h5.close()


def test_h5ad_parallel_producer_count_is_memory_admitted(tmp_path):
    from scarf.readers import H5adReader
    from scarf.storage.budget import ResourceBudget
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "admitted_producers.h5ad", values, encoding="csr")

    def write(budget: int) -> tuple[int, MemoryStore]:
        reader = H5adReader(str(path), feature_name_key="feature_name")
        store = MemoryStore()
        try:
            writer = H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET)
            writer.resources = ResourceBudget(budget, 4)
            writer._write_counts(batch_size=2)
            return writer._lastImportProducerCount, store
        finally:
            reader.h5.close()

    # Producer processes keep buffering while the parent writes, so two of
    # them need both of their reserves beside the parent's write band.
    producers, store = write(7_736)
    assert producers == 2
    assert write(7_735)[0] == 1

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)


def test_h5ad_direct_writers_fit_their_window_summaries(tmp_path, monkeypatch):
    import scarf.storage.sharding as sharding
    from scarf.readers import H5adReader
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.identity import CountSummary
    from scarf.storage.io_policy import StorageIoPolicy
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "window_summaries.h5ad", values, encoding="csr")

    class Planned(Exception):
        pass

    def smallest_process_budget() -> int:
        """Return the smallest budget at which the parent starts writer processes."""
        chosen: list[bool] = []

        def stop(processes: bool):
            def record(*args, **kwargs):
                chosen.append(processes)
                raise Planned

            return record

        def uses_processes(budget: int) -> bool:
            chosen.clear()
            writer.resources = ResourceBudget(budget, 4)
            try:
                writer._write_counts(batch_size=2)
            except (Planned, MemoryError):
                pass
            return chosen == [True]

        reader = H5adReader(str(path), feature_name_key="feature_name")
        try:
            # Each probe only plans, so one writer serves every budget. A
            # second search replaces the unprepared planning store of the first.
            writer = H5adToZarr(
                reader,
                zarr_loc=str(tmp_path / "planning.zarr"),
                io=StorageIoPolicy(readWorkers=2),
                overwrite=True,
                **_SHARD_BAND_BUDGET,
            )
            with monkeypatch.context() as patch:
                patch.setattr(H5adToZarr, "_write_parallel_count_windows", stop(True))
                patch.setattr(sharding, "write_sparse_bands", stop(False))
                low, high = 1_000, 1_000_000
                assert not uses_processes(low)
                assert uses_processes(high)
                while high - low > 1:
                    middle = (low + high) // 2
                    if uses_processes(middle):
                        high = middle
                    else:
                        low = middle
        finally:
            reader.close()
        return high

    budget = smallest_process_budget()
    # Two processes write the row windows (0, 8) and (8, 12), and each holds
    # the count summaries of its window. Growing the summaries of a window by
    # 1,000 bytes therefore raises the budget the parent needs by 2,000 bytes.
    original = CountSummary.nbytes_for
    with monkeypatch.context() as patch:
        patch.setattr(
            CountSummary,
            "nbytes_for",
            staticmethod(
                lambda rows, columns: (
                    original(rows, columns) + (1_000 if rows < values.shape[0] else 0)
                )
            ),
        )
        assert smallest_process_budget() == budget + 2_000

    # At that budget two processes, of two workers each, write every count
    # into disjoint row windows of the destination.
    location = str(tmp_path / "written.zarr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=location,
            io=StorageIoPolicy(readWorkers=2),
            **_SHARD_BAND_BUDGET,
        )
        writer.resources = ResourceBudget(budget, 4)
        writer._write_counts(batch_size=2)
    finally:
        reader.close()
    assert writer._lastImportProducerCount == 2
    assert writer._lastImportWorkersPerProcess == 2
    root = zarr.open_group(store=location, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)


def test_h5ad_read_width_caps_parallel_producers(tmp_path):
    from scarf.readers import H5adReader
    from scarf.storage.io_policy import StorageIoPolicy
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "read_width.h5ad", values, encoding="csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=MemoryStore(),
            io=StorageIoPolicy(readWorkers=1),
            **_SHARD_BAND_BUDGET,
        )
        writer._write_counts(batch_size=2)
        assert writer._lastImportProducerCount == 1
        assert writer._lastImportWriteWorkers == 4
    finally:
        reader.h5.close()


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_h5adtozarr_writes_shard_bands_for_every_encoding(tmp_path, encoding):
    from scarf.readers import H5adReader
    from scarf.storage.types import array_metadata_shards
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / f"{encoding}.h5ad", values, encoding=encoding)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET).dump(batch_size=5)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    counts = root["RNA/counts"]
    assert array_metadata_shards(counts) == (4, 3)
    np.testing.assert_array_equal(counts[:], values)
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True


def test_h5adtozarr_write_counts_defers_counts_t(tmp_path):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "defer_counts_t.h5ad", values, encoding="csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET)._write_counts(
            batch_size=5,
        )
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" not in root["RNA"]


@pytest.mark.parametrize(
    ("fractional", "expected_dtype"),
    [(False, np.dtype("uint16")), (True, np.dtype("float32"))],
)
def test_h5adtozarr_uses_smallest_lossless_dtype_for_float_counts(
    tmp_path,
    fractional,
    expected_dtype,
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3).astype(np.float32)
    values[0, 0] = 1.5 if fractional else 3308
    path = _write_h5ad(tmp_path / "float_counts.h5ad", values, encoding="csr")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            mem_budget=1024**2,
            nthreads=2,
            policy=CountMatrixPolicy(unitBytes=48, chunkBytes=48),
        ).dump(batch_size=4)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    assert np.dtype(root["RNA/counts"].dtype) == expected_dtype
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True


def _half_precision_h5ad(path, values: np.ndarray, encoding: str):
    """Write ``values`` as float16, which SciPy sparse matrices cannot hold."""
    import h5py

    _write_h5ad(path, values.astype(np.float32), encoding=encoding)
    with h5py.File(path, mode="r+") as h5:
        parent, name = (h5, "X") if encoding == "dense" else (h5["X"], "data")
        half = np.asarray(parent[name][...], dtype=np.float16)
        del parent[name]
        parent.create_dataset(name, data=half)
    return path


@pytest.mark.parametrize(
    ("encoding", "fractional", "expected_dtype"),
    [
        ("csr", False, np.dtype("uint16")),
        ("csr", True, np.dtype("float32")),
        ("csc", True, np.dtype("float32")),
        ("dense", False, np.dtype("uint16")),
    ],
)
def test_h5adtozarr_imports_float16_counts_as_float32(
    tmp_path, encoding, fractional, expected_dtype
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 3).astype(np.float16)
    values[0, 0] = 1.5 if fractional else 2048
    path = _half_precision_h5ad(tmp_path / "half.h5ad", values, encoding)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        assert reader.sourceMatrixDtype == np.float32
        H5adToZarr(
            reader,
            zarr_loc=store,
            mem_budget=1024**2,
            nthreads=2,
            policy=CountMatrixPolicy(unitBytes=48, chunkBytes=48),
        ).dump(batch_size=4)
    finally:
        reader.h5.close()

    root = zarr.open_group(store=store, mode="r")
    assert np.dtype(root["RNA/counts"].dtype) == expected_dtype
    np.testing.assert_array_equal(root["RNA/counts"][:], values)
    np.testing.assert_array_equal(root["RNA/countsT"][:], values.T)


def test_float16_is_rejected_as_a_count_storage_dtype():
    from scarf.storage.identity import CountSummary
    from scarf.writers import create_zarr_count_assay

    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts = create_zarr_count_assay(
        root, "RNA", None, 6, ["g0", "g1"], ["g0", "g1"], np.float16
    )
    # Every count writer summarizes its bands, so none stores float16.
    with pytest.raises(
        ValueError, match="float16 is not a supported count storage dtype"
    ):
        CountSummary(counts)


def _duplicate_entry_h5ad(path, data: np.ndarray, encoding: str):
    """Write one cell and one feature whose single coordinate repeats."""
    import h5py

    _write_h5ad(path, np.zeros((1, 1), dtype=np.float32), encoding=encoding)
    with h5py.File(path, mode="r+") as h5:
        group = h5["X"]
        for name in ("data", "indices", "indptr"):
            del group[name]
        group.create_dataset("data", data=data)
        group.create_dataset("indices", data=np.zeros(data.size, dtype=np.int32))
        group.create_dataset("indptr", data=np.array([0, data.size], dtype=np.int32))
    return path


_PAST_ENTRY_MAXIMUM = np.array([200, 100], dtype=np.float32)
_FRACTIONAL_ENTRIES = np.array([100.5, 99.5], dtype=np.float32)


@pytest.mark.parametrize(
    ("encoding", "data", "expected_dtype", "expected"),
    [
        # The sum passes the per-entry maximum, so uint8 would not hold it.
        ("csr", _PAST_ENTRY_MAXIMUM, np.dtype("uint16"), 300),
        ("csc", _PAST_ENTRY_MAXIMUM, np.dtype("uint16"), 300),
        # Fractional entries with an integral sum store the canonical sum.
        ("csr", _FRACTIONAL_ENTRIES, np.dtype("uint8"), 200),
        ("csc", _FRACTIONAL_ENTRIES, np.dtype("uint8"), 200),
        # The sum passes int32, which an unsigned dtype holds.
        (
            "csr",
            np.array([2**31 - 1, 2**31 - 1], dtype=np.int32),
            np.dtype("uint32"),
            2**32 - 2,
        ),
        # The CSC conversion sums integers in 64 bits, so a sum past a narrow
        # source dtype is kept, as a CSR import keeps it.
        ("csr", np.array([200, 100], dtype=np.uint8), np.dtype("uint16"), 300),
        ("csc", np.array([200, 100], dtype=np.uint8), np.dtype("uint16"), 300),
        ("csc", np.array([100, 100], dtype=np.int8), np.dtype("uint8"), 200),
        (
            "csc",
            np.array([2**31 - 1, 2**31 - 1], dtype=np.int32),
            np.dtype("uint32"),
            2**32 - 2,
        ),
    ],
)
def test_h5adtozarr_resolves_the_dtype_from_duplicate_sums(
    tmp_path, encoding, data, expected_dtype, expected
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    path = _duplicate_entry_h5ad(
        tmp_path / f"duplicate_{encoding}.h5ad", data, encoding
    )
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            mem_budget=1024**2,
            nthreads=2,
            policy=CountMatrixPolicy(unitBytes=16, chunkBytes=16),
        ).dump(batch_size=1)
    finally:
        reader.close()

    counts = zarr.open_group(store=store, mode="r")["RNA/counts"]
    assert counts.dtype == expected_dtype
    assert int(counts[0, 0]) == expected


def test_h5adtozarr_reads_the_source_once_for_all_assays(tmp_path, monkeypatch):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = _band_counts(12, 4)
    # An empty leading band for the second assay must not shift its row offsets.
    values[:5, [1, 3]] = 0
    path = _write_h5ad(
        tmp_path / "multi.h5ad",
        values,
        feature_types=[
            b"Gene Expression",
            b"Antibody Capture",
            b"Gene Expression",
            b"Antibody Capture",
        ],
    )

    consumed: list[tuple[int, int]] = []
    original = H5adReader.consume_row_range

    def spy(self, batch_size, row_start=0, row_end=None):
        stop = self.nCells if row_end is None else int(row_end)
        consumed.append((int(row_start), stop))
        return original(self, batch_size, row_start, stop)

    monkeypatch.setattr(H5adReader, "consume_row_range", spy)

    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = MemoryStore()
    try:
        H5adToZarr(
            reader,
            zarr_loc=store,
            assay_split_key="feature_types",
            mem_budget=_SHARD_BAND_BUDGET["mem_budget"],
            nthreads=1,
            policy=_SHARD_BAND_BUDGET["policy"],
        ).dump(batch_size=5)
    finally:
        reader.h5.close()

    assert consumed
    covered = 0
    for start, stop in sorted(consumed):
        assert start == covered
        covered = stop
    assert covered == values.shape[0]
    root = zarr.open_group(store=store, mode="r")
    assert set(root.group_keys()) == {"artifacts", "cellData", "RNA", "ADT"}
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, [0, 2]])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, [1, 3]])
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    assert "countsT" not in root["ADT"]
    # One dtype, resolved over the whole matrix, serves every split assay.
    assert root["RNA/counts"].dtype == root["ADT/counts"].dtype == np.uint8


def test_h5adtozarr_small_assay_does_not_serialize_row_band_writes(
    tmp_path,
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr
    from tests.store_probes import RecordingStore

    values = _band_counts(12, 4)
    path = _write_h5ad(
        tmp_path / "uneven_multi.h5ad",
        values,
        feature_types=[
            b"Gene Expression",
            b"Gene Expression",
            b"Gene Expression",
            b"Antibody Capture",
        ],
    )
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = RecordingStore(delay=0.01)
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=store,
            assay_split_key="feature_types",
            **_SHARD_BAND_BUDGET,
        )
        store.reset()
        writer.dump(batch_size=5)
    finally:
        reader.h5.close()

    assert store.max_in_flight > 1
    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], values[:, :3])
    np.testing.assert_array_equal(root["ADT/counts"][:], values[:, 3:])


def test_h5adtozarr_propagates_band_write_failure(
    tmp_path,
):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr
    from tests.store_probes import RecordingStore

    values = _band_counts(12, 3)
    path = _write_h5ad(tmp_path / "failing.h5ad", values)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    store = RecordingStore(fail_on="RNA/counts/c/2/0")
    try:
        writer = H5adToZarr(reader, zarr_loc=store, **_SHARD_BAND_BUDGET)
        with pytest.raises(RuntimeError, match="injected write failure"):
            writer.dump(batch_size=5)
    finally:
        reader.h5.close()

    assert ("set", "RNA/counts/c/2/0") in store.ops


def test_h5adtozarr_spills_csc_with_bounded_resident_memory(tmp_path):
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    values = (
        np.arange(400 * 400, dtype=np.uint32).reshape(400, 400) % 65_534 + 1
    ).astype(np.uint16)
    path = _write_h5ad(tmp_path / "resident_csc.h5ad", values, encoding="csc")
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=MemoryStore(),
            mem_budget=4 * 1024 * 1024,
            nthreads=2,
            # Twenty-row shards, whose bands the budget writes one at a time.
            policy=CountMatrixPolicy(unitBytes=16 * 1024, chunkBytes=4 * 1024),
        )
        assert reader.materialized_csr_bytes() == (values.shape[0] + 1) * 8
        writer.dump(batch_size=8)
        np.testing.assert_array_equal(writer.z["RNA/counts"][:], values)
        np.testing.assert_array_equal(writer.z["RNA/countsT"][:], values.T)
    finally:
        reader.close()


@pytest.mark.parametrize("from_inspect", [False, True])
def test_h5ad_csc_spill_is_removed_when_the_reader_closes(tmp_path, from_inspect):
    from pathlib import Path
    from scarf.readers import H5adReader, inspect_h5ad

    values = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.uint16)
    path = _write_h5ad(tmp_path / "spilled_csc.h5ad", values, encoding="csc")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    if from_inspect:
        reader = H5adReader.from_inspect(inspect_h5ad(path), temp_dir=scratch)
    else:
        reader = H5adReader(
            str(path), feature_name_key="feature_name", temp_dir=str(scratch)
        )
    try:
        reader.materialize_csc()
        directory = Path(reader._convertedCsr._directory.name)
        assert directory.parent == scratch
        np.testing.assert_array_equal(next(reader.consume(3)).toarray(), values)
    finally:
        reader.close()
    assert not directory.exists()
    assert list(scratch.iterdir()) == []


def test_sparsetozarr(tmp_path):
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    cols = [1, 3, 8, 2, 3, 1, 2, 8, 9]
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    data = [1, 10, 15, 10, 20, 2, 3, 1, 5]
    mat = (data, (rows, cols))
    mat = csr_matrix(mat, shape=(3, 10))

    fn = str(tmp_path / "dummy_sparse.zarr")

    writer = SparseToZarr(
        mat,
        zarr_loc=fn,
        cell_ids=[f"cell_{x}" for x in range(3)],
        feature_ids=[f"feat_{x}" for x in range(10)],
    )
    writer.dump()
    root = zarr.open_group(fn, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], mat.toarray())
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True


def test_sparsetozarr_sharded_layout(tmp_path):
    from scipy.sparse import csr_matrix

    from scarf.storage.layout import count_array_spec
    from scarf.writers import SparseToZarr

    n_cells, n_feats = 5000, 200
    rng = np.random.default_rng(0)
    rows = rng.integers(0, n_cells, size=50_000)
    cols = rng.integers(0, n_feats, size=50_000)
    data = np.ones(50_000, dtype=np.uint32)
    mat = csr_matrix((data, (rows, cols)), shape=(n_cells, n_feats))
    fn = str(tmp_path / "dummy_sparse_sharded.zarr")
    writer = SparseToZarr(
        mat,
        zarr_loc=fn,
        cell_ids=[f"cell_{x}" for x in range(n_cells)],
        feature_ids=[f"feat_{x}" for x in range(n_feats)],
    )
    writer.dump()
    store = zarr.open_group(fn, mode="r")
    counts = store["RNA/counts"]
    # csr_matrix summed the repeated coordinates, which stay below 256.
    expected = count_array_spec(n_cells, n_feats, dtype=np.uint8, profile="fast_local")
    assert counts.dtype == np.uint8
    assert counts.chunks == expected.chunks
    assert counts.metadata.shards == expected.shards
    np.testing.assert_array_equal(counts[:], mat.toarray())
    np.testing.assert_array_equal(store["RNA/countsT"][:], mat.toarray().T)
    np.testing.assert_array_equal(
        store["cellData/ids"][:], [f"cell_{x}" for x in range(n_cells)]
    )


def test_csv_to_zarr_round_trip(tmp_path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text(
        "quality,geneA,geneB,geneC\n"
        "10,1,0,2\n"
        "20,0,3,0\n"
        "30,4,5,6\n"
        "40,7,0,8\n"
        "50,9,10,0\n",
        encoding="utf-8",
    )
    reader = CSVReader(
        str(csv_path),
        cell_data_cols=["quality"],
        batch_size=2,
    )
    store = MemoryStore()
    writer = CSVtoZarr(
        reader,
        zarr_loc=store,
        assay_name="RNA",
    )

    writer.dump()

    root = zarr.open_group(store=store, mode="r")
    expected = np.array(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 5, 6],
            [7, 0, 8],
            [9, 10, 0],
        ],
        dtype=np.uint8,
    )
    assert root["RNA/counts"].dtype == np.uint8
    np.testing.assert_array_equal(root["RNA/counts"][:], expected)
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        root["RNA/featureData/ids"][:],
        np.array(["geneA", "geneB", "geneC"]),
    )
    np.testing.assert_array_equal(
        root["cellData/ids"][:],
        np.array(["cell_0", "cell_1", "cell_2", "cell_3", "cell_4"]),
    )
    np.testing.assert_array_equal(
        root["cellData/quality"][:],
        np.array([10, 20, 30, 40, 50]),
    )


def test_csv_to_zarr_writes_extra_cell_columns_into_workspace(tmp_path):
    csv_path = tmp_path / "counts.csv"
    csv_path.write_text(
        "quality,geneA,geneB\n1,1,2\n2,3,4\n3,5,6\n",
        encoding="utf-8",
    )
    reader = CSVReader(
        str(csv_path),
        cell_data_cols=["quality"],
        batch_size=2,
    )
    store = MemoryStore()
    writer = CSVtoZarr(
        reader,
        zarr_loc=store,
        assay_name="RNA",
        workspace="run1",
    )

    writer.dump()

    root = zarr.open_group(store=store, mode="r")
    assert "cellData" not in root
    np.testing.assert_array_equal(
        root["run1/cellData/quality"][:],
        np.array([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        root["matrices/RNA/counts"][:],
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint16),
    )


@pytest.mark.parametrize(
    "columns",
    [
        ["quality", "batch", "score"],
        ["score", "batch", "quality"],
        ["score", "quality"],
    ],
)
@pytest.mark.parametrize("text_dtype", [None, object])
def test_csv_to_zarr_preserves_metadata_column_order_and_counts(
    tmp_path, columns, text_dtype
):
    path = tmp_path / "counts.csv"
    path.write_text(
        "cell,quality,g1,batch,score,g2,drop\n"
        "c1,9007199254740993,1,batch_A,0.5,2,unused_A\n"
        "c2,8,3,batch_B,1.5,4,unused_B\n"
        "c3,9,5,batch_γ_longer,2.5,6,unused_C\n"
    )
    reader = CSVReader(
        str(path),
        id_column=0,
        cell_data_cols=columns,
        skip_cols=["drop"] + ([] if "batch" in columns else ["batch"]),
        batch_size=2,
        pandas_kwargs={"dtype": {"batch": text_dtype}} if text_dtype else None,
    )
    store = MemoryStore()
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1).dump()

    root = zarr.open_group(store=store, mode="r")
    counts = np.array([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(root["RNA/counts"][:], counts)
    np.testing.assert_array_equal(root["RNA/countsT"][:], counts.T)
    np.testing.assert_array_equal(root["cellData/ids"][:], ["c1", "c2", "c3"])
    np.testing.assert_array_equal(root["cellData/quality"][:], [2**53 + 1, 8, 9])
    np.testing.assert_array_equal(root["cellData/score"][:], [0.5, 1.5, 2.5])
    if "batch" in columns:
        np.testing.assert_array_equal(
            root["cellData/batch"][:], ["batch_A", "batch_B", "batch_γ_longer"]
        )


def test_csv_to_zarr_preserves_supplied_cell_ids(tmp_path):
    from scarf import DataStore

    csv_path = tmp_path / "counts.csv"
    csv_path.write_text("cell,g1,g2\ncell_A,1,2\ncell_B,3,4\n")
    reader = CSVReader(str(csv_path), id_column=0, batch_size=1)
    store = MemoryStore()

    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1).dump()

    result = DataStore(store, min_features_per_cell=0, nthreads=1)
    np.testing.assert_array_equal(result.cells.fetch_all("ids"), ["cell_A", "cell_B"])
    np.testing.assert_array_equal(result.RNA.rawData.compute(), [[1, 2], [3, 4]])


@pytest.mark.parametrize(
    ("dtypes", "message"),
    [
        ([], "holds 0 dtypes for 1 cell_data_cols columns"),
        (
            [np.dtype(np.int64), np.dtype(object)],
            "holds 2 dtypes for 1 cell_data_cols columns",
        ),
    ],
)
def test_csv_cell_data_dtypes_must_match_the_columns_before_writing(
    tmp_path, dtypes, message
) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("g1,g2,quality\n1,0,7\n0,3,8\n")
    reader = CSVReader(str(path), cell_data_cols=["quality"])
    reader.cellDataDtypes = dtypes
    destination = MemoryStore()
    zarr.open_group(store=destination, mode="w").create_group("sentinel")
    with pytest.raises(ValueError, match=message):
        CSVtoZarr(reader, destination, assay_name="RNA", nthreads=1)
    root = zarr.open_group(store=destination, mode="r")
    assert set(root.group_keys()) == {"sentinel"}


def _write_reserved_h5ad(tmp_path):
    import h5py

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    path = _write_h5ad(tmp_path / "reserved.h5ad", np.eye(3, dtype=np.float32))
    with h5py.File(path, mode="a") as h5:
        h5["obs"].create_dataset("ids", data=np.array([b"dup", b"dup", b"dup"]))
        h5["obs"].create_dataset("names", data=np.array([b"n0", b"n1", b"n2"]))
        h5["obs"].create_dataset("I", data=np.array([False, True, False]))
        h5["obs"].create_dataset("quality", data=np.array([1, 2, 3]))
        h5["var"].create_dataset("ids", data=np.array([b"x", b"x", b"x"]))
    store = MemoryStore()
    H5adToZarr(H5adReader(str(path)), zarr_loc=store).dump()
    return store, ["c0", "c1", "c2"], ["f0", "f1", "f2"]


def _write_reserved_csv(tmp_path):
    path = tmp_path / "reserved.csv"
    path.write_text(
        "cell,ids,names,I,quality,g0,g1,g2\n"
        "c0,dup,n0,False,1,1,0,0\n"
        "c1,dup,n1,True,2,0,1,0\n"
        "c2,dup,n2,False,3,0,0,1\n"
    )
    reader = CSVReader(
        str(path),
        id_column=0,
        cell_data_cols=["ids", "names", "I", "quality"],
        batch_size=2,
    )
    store = MemoryStore()
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1).dump()
    return store, ["c0", "c1", "c2"], ["g0", "g1", "g2"]


def _write_reserved_cellranger(tmp_path):
    import pandas as pd
    from scipy.sparse import coo_matrix

    values = np.eye(3, dtype=np.uint16)

    class ReservedReader:
        nCells = 3
        nFeatures = 3
        matrix_dtype = values.dtype
        assayFeats = pd.DataFrame(
            {"RNA": ["Gene Expression", 0, 3, 3]},
            index=["type", "start", "end", "nFeatures"],
        )

        def cell_names(self):
            return ["c0", "c1", "c2"]

        def feature_ids(self, assay_name):
            return ["f0", "f1", "f2"]

        def feature_names(self, assay_name):
            return ["g0", "g1", "g2"]

        def get_cell_columns(self):
            yield "ids", np.array(["dup", "dup", "dup"])
            yield "names", np.array(["n0", "n1", "n2"])
            yield "I", np.array([False, True, False])
            yield "quality", np.array([1, 2, 3])

        def get_feature_columns(self):
            yield "ids", np.array(["x", "x", "x"])

        def consume(self, batch_size, lines_in_mem):
            for start in range(0, self.nCells, batch_size):
                yield coo_matrix(values[start : start + batch_size])

        def count_value_ranges(self, maxBytes, featureGroups=None):
            return [_value_range(values)]

        def max_window_nnz(self, window_rows):
            return min(window_rows, self.nCells)

        def producer_staging_bytes(self, batch_size, lines_in_mem):
            return 0

    store = MemoryStore()
    CrToZarr(ReservedReader(), zarr_loc=store).dump(batch_size=2)
    return store, ["c0", "c1", "c2"], ["f0", "f1", "f2"]


@pytest.mark.parametrize(
    "write",
    [
        _write_reserved_h5ad,
        _write_reserved_csv,
        _write_reserved_cellranger,
    ],
    ids=["h5ad", "csv", "cellranger"],
)
def test_writers_skip_reserved_source_metadata_columns(tmp_path, write):
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        store, cell_ids, feature_ids = write(tmp_path)
    finally:
        logger.remove(sink)

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["cellData/ids"][:], cell_ids)
    np.testing.assert_array_equal(root["cellData/names"][:], cell_ids)
    np.testing.assert_array_equal(root["cellData/I"][:], [True, True, True])
    np.testing.assert_array_equal(root["cellData/quality"][:], [1, 2, 3])
    np.testing.assert_array_equal(root["RNA/featureData/ids"][:], feature_ids)
    skipped = [message for message in messages if "reserves the column" in message]
    assert any("cell metadata column 'ids'" in message for message in skipped)
    assert any("cell metadata column 'I'" in message for message in skipped)


def test_subset_assay_zarr_selects_ordered_rows_and_columns():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    source = root.create_array(
        "source",
        shape=(5, 4),
        chunks=(2, 2),
        dtype=np.uint16,
        fill_value=0,
    )
    values = np.arange(20, dtype=np.uint16).reshape(5, 4)
    source[:] = values
    cells = np.array([4, 1, 3])
    features = np.array([3, 0])

    result = subset_assay_zarr(
        store,
        in_grp="source",
        out_grp="selected",
        cells_idx=cells,
        feat_idx=features,
        policy=CountMatrixPolicy(unitBytes=8, chunkBytes=8),
    )

    selected = root["selected"]
    assert result is None
    assert selected.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(
        selected[:],
        values[np.ix_(cells, features)],
    )


def test_subset_assay_zarr_counts_carry_the_count_matrix_layout():
    from scarf.storage.counts_t_contract import validate_count_matrix

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("source", data=np.arange(20, dtype=np.uint16).reshape(5, 4))
    root.create_group("subset")

    subset_assay_zarr(
        store,
        in_grp="source",
        out_grp="subset/counts",
        cells_idx=np.array([4, 1, 3]),
        feat_idx=np.array([3, 0]),
    )

    counts, _ = validate_count_matrix(
        zarr.open_group(store=store, path="subset", mode="r"),
        require_transpose=False,
    )
    np.testing.assert_array_equal(
        counts[:], np.arange(20, dtype=np.uint16).reshape(5, 4)[[4, 1, 3]][:, [3, 0]]
    )
    # Zarr ignores extra separators, so the layout goes on the group "other".
    root.create_group("other")
    subset_assay_zarr(
        store,
        in_grp="source",
        out_grp="/other//counts/",
        cells_idx=np.array([0]),
        feat_idx=np.array([1]),
    )
    counts, _ = validate_count_matrix(
        zarr.open_group(store=store, path="other", mode="r"),
        require_transpose=False,
    )
    np.testing.assert_array_equal(counts[:], [[1]])


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1.25, 4.5], [2.75, 3.25]], dtype=np.float32),
        np.array([[1, 2**32 + 1], [2**40, 3]], dtype=np.uint64),
    ],
)
def test_subset_assay_zarr_preserves_numeric_dtype_and_values(values):
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("source", data=values)

    subset_assay_zarr(
        store,
        "source",
        "selected",
        cells_idx=np.array([1, 0]),
        feat_idx=np.array([1, 0]),
        nthreads=1,
    )

    assert root["selected"].dtype == values.dtype
    np.testing.assert_array_equal(root["selected"][:], values[::-1, ::-1])


@pytest.mark.parametrize(
    ("cells", "features", "error", "match"),
    [
        (
            np.array([-1, 0]),
            np.array([0]),
            IndexError,
            "cells_idx contains an out-of-range",
        ),
        (
            np.array([5]),
            np.array([0]),
            IndexError,
            "cells_idx contains an out-of-range",
        ),
        (
            np.array([1, 1]),
            np.array([0]),
            ValueError,
            "cells_idx cannot contain duplicate",
        ),
        (np.array([0.5]), np.array([0]), TypeError, "cells_idx must contain integers"),
        (
            np.array([[0]]),
            np.array([0]),
            ValueError,
            "cells_idx must be one-dimensional",
        ),
        (np.array([0]), np.array([4]), IndexError, "feat_idx contains an out-of-range"),
        (
            np.array([0]),
            np.array([], dtype=np.int64),
            ValueError,
            "at least one feature",
        ),
    ],
)
def test_subset_assay_zarr_rejects_unusable_indices_before_writing(
    cells, features, error, match
):
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("source", data=np.arange(20, dtype=np.uint16).reshape(5, 4))

    with pytest.raises(error, match=match):
        subset_assay_zarr(
            store, "source", "selected", cells_idx=cells, feat_idx=features
        )
    assert "selected" not in root


def test_subset_assay_zarr_writes_only_new_nodes_of_its_store(monkeypatch):
    from zarr.errors import ContainsArrayError

    import scarf.writers.subset as subset_module

    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_group("RNA").create_array(
        "counts", data=np.arange(12, dtype=np.uint16).reshape(4, 3)
    )
    root.create_group("selected")

    def subset(in_grp: str, out_grp: str) -> None:
        subset_assay_zarr(
            store,
            in_grp,
            out_grp,
            cells_idx=np.array([0, 1]),
            feat_idx=np.array([0, 1]),
            nthreads=1,
        )

    def stored() -> dict[str, bytes]:
        return {
            key: bytes(value.to_bytes()) for key, value in store._store_dict.items()
        }

    subset("RNA/counts", "selected/counts")
    before = stored()
    overlap = "must not be, contain, or lie inside"
    # Before, an out_grp equal to in_grp zeroed its counts, and out_grp="RNA"
    # deleted the assay.
    for in_grp, out_grp, error, message in (
        ("RNA/counts", "RNA/counts", ValueError, overlap),
        ("RNA/counts", "/RNA//counts/", ValueError, overlap),
        ("RNA/counts", "RNA", ValueError, overlap),
        ("RNA/counts", "", ValueError, overlap),
        ("RNA/counts", "RNA/counts/subset", ValueError, overlap),
        ("/RNA/counts", "RNA\\counts", ValueError, overlap),
        ("RNA/counts", "selected/../RNA", ValueError, "must not contain '.' or '..'"),
        ("RNA/counts", "selected/counts", FileExistsError, "already exists"),
        # The layout record of a matrix group describes the counts it holds.
        ("RNA/counts", "selected/again", FileExistsError, "records the layout"),
    ):
        with pytest.raises(error, match=message):
            subset(in_grp, out_grp)
    assert stored() == before

    # A node that another writer creates after the check is never replaced.
    def racing_check(z, out_grp: str, _parts) -> None:
        z.create_array(out_grp, data=np.array([7]))

    monkeypatch.setattr(subset_module, "_check_new_output", racing_check)
    with pytest.raises(ContainsArrayError):
        subset("RNA/counts", "raced")
    np.testing.assert_array_equal(root["raced"][:], [7])


def test_v2_fixture_read_only(datastore):
    from tests import full_path

    # The bundled store holds the counts of the Cell Ranger file it was built
    # from, which h5py and SciPy read independently of Scarf.
    counts, barcodes, features = _read_cellranger_h5(full_path("1K_pbmc_citeseq.h5"))
    # Session fixtures may add derived assays to this shared store.
    assert {"RNA", "assay2"} <= set(datastore.assay_names)
    np.testing.assert_array_equal(datastore.cells.fetch_all("ids"), barcodes)
    rna_columns = np.flatnonzero(features["feature_type"] == "Gene Expression")
    adt_columns = np.flatnonzero(features["feature_type"] == "Antibody Capture")
    rna = datastore.get_assay("RNA")
    adt = datastore.get_assay("assay2")
    for assay, columns in ((rna, rna_columns), (adt, adt_columns)):
        assert assay.rawData.shape == (barcodes.size, columns.size)
        np.testing.assert_array_equal(
            assay.feats.fetch_all("ids"), features["id"][columns]
        )
    np.testing.assert_array_equal(
        adt.rawData.compute(), counts[:, adt_columns].toarray()
    )
    # Reading the last stored chunk of RNA genes decodes only that chunk; the
    # stored cell totals cover the other genes.
    start = int(rna.matrixGroup["counts"].chunks[1])
    tail = np.arange(start, rna_columns.size)
    np.testing.assert_array_equal(
        rna.rawData[:, tail].compute(), counts[:, rna_columns[tail]].toarray()
    )
    expected_rna = counts[:, rna_columns]
    np.testing.assert_array_equal(
        datastore.cells.fetch_all("RNA_nCounts"),
        np.asarray(expected_rna.sum(axis=1)).ravel(),
    )
    np.testing.assert_array_equal(
        datastore.cells.fetch_all("RNA_nFeatures"), np.diff(expected_rna.indptr)
    )


# RNA counts of cells b1, b2, and b3 over genes g1 to g4 in toy_cr_dir.tar.gz.
_TOY_RNA_COUNTS = np.array([[5, 0, 0, 2], [3, 3, 0, 7], [3, 3, 0, 7]])


@pytest.fixture
def export_assay_store(toy_crdir_writer, tmp_path):
    import shutil

    from scarf.datastore.datastore import DataStore

    destination = tmp_path / "export_toy.zarr"
    shutil.copytree(toy_crdir_writer, destination)
    return DataStore(
        str(destination),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


def test_to_h5ad_preserves_counts_metadata_and_embeddings(
    export_assay_store, tmp_path, monkeypatch
):
    import h5py
    from scipy.sparse import csr_matrix

    from scarf.matrix import ChunkedArray
    from scarf.writers import to_h5ad
    from scarf.writers.export import h5ad_conversion_bytes, largest_block_rows

    charged = []
    stream = ChunkedArray._stream_blocks

    def charging(self, **options):
        charged.append(options["resident_bytes"])
        return stream(self, **options)

    monkeypatch.setattr(ChunkedArray, "_stream_blocks", charging)
    assay = export_assay_store.RNA
    n_cells = assay.cells.N
    umap = np.column_stack(
        [
            np.linspace(0.0, 1.0, n_cells),
            np.linspace(2.0, 3.0, n_cells),
        ]
    )
    assay.cells.insert("RNA_UMAP1", umap[:, 0], overwrite=True)
    assay.cells.insert("RNA_UMAP2", umap[:, 1], overwrite=True)
    assay.cells.insert(
        "export_batch", np.array(["a", "b", "a"][:n_cells]), overwrite=True
    )

    path = tmp_path / "toy_export.h5ad"
    to_h5ad(assay, str(path), embeddings_cols=["UMAP"])
    # The stream charges the conversion of its largest block to CSR.
    counts = assay.rawData
    rows = largest_block_rows(counts)
    assert charged == [h5ad_conversion_bytes(counts.dtype, counts.shape[1], rows)]

    with h5py.File(path, "r") as h5:
        shape = tuple(int(x) for x in h5["X"].attrs["shape"])
        assert shape == _TOY_RNA_COUNTS.shape
        exported = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=shape,
        )
        np.testing.assert_array_equal(exported.toarray(), _TOY_RNA_COUNTS)
        np.testing.assert_array_equal(h5["obs/_index"].asstr()[:], ["b1", "b2", "b3"])
        np.testing.assert_array_equal(
            h5["obs/export_batch"].asstr()[:], ["a", "b", "a"]
        )
        np.testing.assert_array_equal(
            h5["var/_index"].asstr()[:], ["g1", "g2", "g3", "g4"]
        )
        np.testing.assert_array_equal(
            h5["var/gene_short_name"].asstr()[:], ["g1", "g2", "g3", "g4"]
        )
        emb_cols = sorted(
            column for column in assay.cells.columns if column.startswith("RNA_UMAP")
        )
        assert emb_cols == ["RNA_UMAP1", "RNA_UMAP2"]
        np.testing.assert_allclose(h5["obsm/X_umap"][:], umap)
        assert "RNA_UMAP1" not in h5["obs"]
        assert "RNA_UMAP2" not in h5["obs"]

    # Two prefixes that would both export as obsm["X_umap"] are refused.
    assay.cells.insert("RNA_umap1", umap[:, 0], overwrite=True)
    clash = tmp_path / "clash.h5ad"
    with pytest.raises(ValueError, match=r"both export as obsm\['X_umap'\]"):
        to_h5ad(assay, str(clash), embeddings_cols=["UMAP", "umap"])
    assert not clash.exists()


@pytest.mark.parametrize("skip_recalc", [True, False])
@pytest.mark.parametrize("preserve_total", [True, False])
def test_to_h5ad_recalculates_counts_without_mutating_qc_metadata(
    export_assay_store,
    tmp_path,
    skip_recalc,
    preserve_total,
):
    import h5py
    from scipy.sparse import csr_matrix

    from scarf.writers import to_h5ad

    assay = export_assay_store.RNA
    source_qc = np.full(assay.cells.N, 99, dtype=np.int64)
    if preserve_total:
        source_qc = np.count_nonzero(_TOY_RNA_COUNTS, axis=1)
        donor = int(np.flatnonzero(source_qc)[0])
        source_qc[donor] -= 1
        source_qc[(donor + 1) % assay.cells.N] += 1
    assay.cells._get_array("RNA_nFeatures")[:] = source_qc
    columns_before = set(assay.cells.columns)

    path = tmp_path / "recalculated_export.h5ad"
    to_h5ad(assay, str(path), skip_recalc_nfeats=skip_recalc)

    assert set(assay.cells.columns) == columns_before
    np.testing.assert_array_equal(
        assay.cells.fetch_all("RNA_nFeatures"),
        source_qc,
    )
    with h5py.File(path, "r") as h5:
        shape = tuple(int(value) for value in h5["X"].attrs["shape"])
        assert h5["X/data"].chunks == (65_536,)
        assert h5["X/indices"].chunks == (65_536,)
        exported = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=shape,
        )
        np.testing.assert_array_equal(exported.toarray(), _TOY_RNA_COUNTS)


@pytest.mark.parametrize(("row_delta", "column_delta"), [(1, 0), (-1, 0), (0, 1)])
def test_to_h5ad_rejects_count_metadata_shape_mismatches_and_closes_output(
    export_assay_store, tmp_path, row_delta, column_delta
):
    import h5py

    from scarf.matrix import ChunkedArray
    from scarf.writers import to_h5ad

    assay = export_assay_store.RNA
    values = np.ones(
        (assay.cells.N + row_delta, assay.feats.N + column_delta), dtype=np.uint16
    )
    group = zarr.group(store=MemoryStore())
    assay.rawData = ChunkedArray(group.create_array("counts", data=values), nthreads=1)
    path = tmp_path / "invalid.h5ad"
    handles = h5py.h5f.get_obj_count()
    with pytest.raises(ValueError, match="The X matrix has .* the export declares"):
        to_h5ad(assay, str(path), nthreads=1)
    assert h5py.h5f.get_obj_count() == handles
    # The file is written under another name and moved into place only once
    # it is complete, so a failed export leaves nothing behind.
    assert list(tmp_path.glob("invalid.h5ad*")) == []


def _plan(blocks, *, shape=(3, 2), dtype=np.int32, obs=(), obsm=None):
    """A one-matrix H5AD plan whose X yields ``blocks``."""
    from scarf.writers.export import H5adColumn, H5adExportPlan, H5adMatrix

    def ids(prefix: str, count: int) -> H5adColumn:
        names = np.asarray([f"{prefix}{index}" for index in range(count)])
        return H5adColumn("_index", lambda: (names, None))

    return H5adExportPlan(
        x=H5adMatrix(
            shape=shape,
            dtype=np.dtype(dtype),
            blocks=lambda: iter(blocks),
        ),
        obs_index=ids("c", shape[0]),
        obs=tuple(obs),
        var_index=ids("g", shape[1]),
        var=(),
        obsm={} if obsm is None else obsm,
    )


# Blocks of other shapes are refused through to_h5ad in
# test_to_h5ad_rejects_count_metadata_shape_mismatches_and_closes_output.
@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        (
            [np.ones((3, 2), dtype=np.float64)],
            "The X matrix has a row block of dtype float64; the export declares int32",
        ),
        (
            [np.ones(2, dtype=np.int32)],
            "The X matrix has a row block with 1 dimension; blocks are two-dimensional",
        ),
    ],
)
def test_h5ad_plan_writer_rejects_blocks_that_do_not_fit_the_plan(
    tmp_path, blocks, message
):
    from scarf.writers.export import materialize_h5ad_matrix, write_h5ad_plan

    path = tmp_path / "export.h5ad"
    path.write_bytes(b"an earlier export")

    with pytest.raises(ValueError, match=message):
        write_h5ad_plan(_plan(blocks), path)
    with pytest.raises(ValueError, match=message):
        materialize_h5ad_matrix(_plan(blocks).x)

    # The earlier file is kept, and no partial or temporary file is left.
    assert path.read_bytes() == b"an earlier export"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_h5ad_plan_writer_keeps_the_permissions_of_a_replaced_file(
    tmp_path, monkeypatch
):
    import stat

    import scarf.writers.export as export
    from scarf.writers.export import write_h5ad_plan

    path = tmp_path / "export.h5ad"
    path.write_bytes(b"an earlier export")
    path.chmod(0o640)
    new = tmp_path / "new.h5ad"
    write_plan = export._write_plan
    modes = []

    def recorded(h5, plan):
        modes.append(stat.S_IMODE(os.stat(h5.filename).st_mode))
        write_plan(h5, plan)

    monkeypatch.setattr(export, "_write_plan", recorded)
    previous = os.umask(0o022)
    try:
        write_h5ad_plan(_plan([np.ones((3, 2), dtype=np.int32)]), path)
        write_h5ad_plan(_plan([np.ones((3, 2), dtype=np.int32)]), new)
    finally:
        os.umask(previous)

    # The data of a replaced file is private until the file takes its mode.
    assert modes[0] == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    # A new file takes the mode that the umask gives any new file.
    assert stat.S_IMODE(new.stat().st_mode) == 0o644
    assert sorted(tmp_path.iterdir()) == [path, new]


def test_h5ad_plan_writer_leaves_nothing_behind_when_interrupted(tmp_path):
    import dataclasses

    from scipy.sparse import csr_matrix

    from scarf.writers.export import H5adColumn, write_h5ad_plan

    def interrupted_blocks():
        yield csr_matrix(np.ones((1, 2), dtype=np.int32))
        raise KeyboardInterrupt

    def unreadable():
        raise OSError("metadata is unreadable")

    path = tmp_path / "export.h5ad"
    plan = _plan([])
    interrupted = dataclasses.replace(
        plan, x=dataclasses.replace(plan.x, blocks=interrupted_blocks)
    )
    with pytest.raises(KeyboardInterrupt):
        write_h5ad_plan(interrupted, path)
    assert list(tmp_path.iterdir()) == []

    rows = [np.ones((3, 2), dtype=np.int32)]
    failing = _plan(rows, obs=[H5adColumn("batch", lambda: unreadable())])
    with pytest.raises(OSError, match="metadata is unreadable"):
        write_h5ad_plan(failing, path)
    assert list(tmp_path.iterdir()) == []


def test_h5ad_plan_writer_checks_columns_and_replaces_a_file_once_written(tmp_path):
    import dataclasses

    from scipy.sparse import csr_matrix

    from scarf.writers.export import (
        H5adColumn,
        h5ad_frame,
        materialize_h5ad_matrix,
        write_h5ad_plan,
    )

    rows = [csr_matrix(np.asarray([[1, 0], [0, 2]], dtype=np.int32))]
    rows.append(np.asarray([[0, 3]], dtype=np.int32))
    path = tmp_path / "export.h5ad"
    path.write_bytes(b"an earlier export")
    short = H5adColumn("batch", lambda: (np.asarray(["a", "b"]), None))
    misaligned = H5adColumn(
        "batch", lambda: (np.asarray(["a", "b", "c"]), np.zeros(2, dtype=bool))
    )
    table = H5adColumn("batch", lambda: (np.zeros((3, 1)), None))
    duplicated = H5adColumn("_index", lambda: (np.asarray(["a", "b", "c"]), None))
    incomplete = H5adColumn(
        "_index", lambda: (np.asarray(["a", "", "c"]), np.asarray([False, True, False]))
    )
    wide = {"X_umap": lambda: np.zeros((2, 2))}
    flat = {"X_umap": lambda: np.zeros(3)}
    small = _plan([], shape=(2, 2)).x
    for plan, message in (
        (_plan(rows, obs=[short]), "Column 'batch' has 2 rows; the export declares 3"),
        (
            _plan(rows, obs=[misaligned]),
            "Column 'batch' has a missing mask of shape \\(2,\\); its values have 3 rows",
        ),
        (_plan(rows, obs=[table]), "Column 'batch' has 2 dimensions"),
        (_plan(rows, obs=[duplicated]), "Column '_index' is written twice"),
        (
            dataclasses.replace(_plan(rows), obs_index=incomplete),
            "Index '_index' holds missing values",
        ),
        (_plan(rows, obsm=wide), "obsm 'X_umap' has 2 rows; the export declares 3"),
        (_plan(rows, obsm=flat), "obsm 'X_umap' has 1 dimensions"),
        (
            dataclasses.replace(_plan(rows), layers={"raw": small}),
            "Layer 'raw' has shape \\(2, 2\\); X has shape \\(3, 2\\)",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            write_h5ad_plan(plan, path)
        assert path.read_bytes() == b"an earlier export"
        assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match="Index '_index' holds missing values"):
        h5ad_frame(incomplete, (), 3)
    empty = materialize_h5ad_matrix(_plan([], shape=(0, 2)).x)
    assert empty.shape == (0, 2) and empty.dtype == np.int32

    plan = _plan(rows)
    write_h5ad_plan(dataclasses.replace(plan, layers={"raw": plan.x}), path)

    anndata = pytest.importorskip("anndata")
    exported = anndata.read_h5ad(path)
    assert list(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(
        exported.X.toarray(), np.asarray([[1, 0], [0, 2], [0, 3]])
    )
    np.testing.assert_array_equal(
        exported.layers["raw"].toarray(), exported.X.toarray()
    )
    assert exported.obs_names.tolist() == ["c0", "c1", "c2"]
    assert exported.var_names.tolist() == ["g0", "g1"]


def test_h5ad_export_of_a_tiny_matrix_fits_a_small_budget(monkeypatch) -> None:
    import zarr

    from scarf.datastore._operations.presentation import _stored_row_blocks
    from scarf.matrix import ChunkedArray
    from scarf.storage.budget import ResourceBudget
    from scarf.writers.export import h5ad_conversion_bytes

    charged: list[int] = []
    stream = ChunkedArray._stream_blocks

    def record(self, *args, **kwargs):
        charged.append(kwargs["resident_bytes"])
        return stream(self, *args, **kwargs)

    monkeypatch.setattr(ChunkedArray, "_stream_blocks", record)

    # The reservation follows the rows of the largest block that the stream
    # yields; before, it held a full step of 2**20 values, about 20 MiB, for
    # any matrix, and then for every row of the matrix.
    assert h5ad_conversion_bytes(np.float32, 2, 2) < 1024
    assert h5ad_conversion_bytes(np.float32, 2, 0) == 0
    for shape, chunks, budget in (
        ((2, 2), None, 8 * 1024**2),
        ((2_000, 2), (10, 2), 1024**2),
        # Converting every row of this matrix at once would hold about 12 MB.
        ((200_000, 2), (1_000, 2), 1024**2),
    ):
        data = zarr.create_array(
            store=zarr.storage.MemoryStore(),
            shape=shape,
            chunks=chunks or shape,
            dtype="float32",
        )
        data[:] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 3
        blocks = _stored_row_blocks(data, 1, ResourceBudget(budget, 1), "export")
        np.testing.assert_array_equal(np.vstack(list(blocks)), data[:])
        # The stream charges the conversion of its blocks to CSR as resident.
        assert len(charged) == 1 and charged.pop() > 0


def test_h5ad_writer_converts_dense_blocks_within_the_charged_bytes(
    tmp_path, monkeypatch
):
    import tracemalloc

    import h5py
    from scipy.sparse import csr_matrix

    import scarf.writers.export as export
    from scarf.writers.export import (
        h5ad_conversion_bytes,
        iter_h5ad_blocks,
        write_h5ad_plan,
    )

    monkeypatch.setattr(export, "_CSR_STEP_VALUES", 1 << 15)
    # Every value is nonzero, the worst case for a conversion to CSR.
    rng = np.random.default_rng(5)
    dense = rng.integers(1, 100, size=(1024, 512)).astype(np.float32)
    plan = _plan([dense], shape=dense.shape, dtype=np.float32)
    charged = h5ad_conversion_bytes(np.float32, 512, 1024)

    tracemalloc.start()
    try:
        for piece in iter_h5ad_blocks(plan.x):
            del piece
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Converting the 2 MiB block through COO held about 14 MiB.
    assert peak <= charged < dense.nbytes
    path = tmp_path / "dense.h5ad"
    write_h5ad_plan(plan, path)
    with h5py.File(path, "r") as h5:
        stored = csr_matrix(
            (h5["X/data"][:], h5["X/indices"][:], h5["X/indptr"][:]),
            shape=dense.shape,
        )
    np.testing.assert_array_equal(stored.toarray(), dense)


@pytest.mark.parametrize(
    ("shape", "dtype", "order"),
    [
        ((37, 23), np.float32, "C"),
        ((37, 23), np.float32, "F"),
        ((5, 3000), np.int64, "C"),
        ((0, 4), np.float64, "C"),
        ((6, 0), np.float32, "C"),
    ],
)
def test_dense_blocks_become_the_csr_pieces_that_scipy_builds(
    monkeypatch, shape, dtype, order
):
    from scipy.sparse import csr_matrix, vstack

    import scarf.writers.export as export

    monkeypatch.setattr(export, "_CSR_STEP_VALUES", 64)
    rng = np.random.default_rng(9)
    values = rng.integers(0, 3, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.floating) and values.size:
        values.flat[::7] = np.nan
        values.flat[1::11] = -0.0
    block = np.asarray(values, order=order)

    pieces = list(export._dense_csr_pieces(block))
    expected = csr_matrix(values)

    assert sum(piece.shape[0] for piece in pieces) == shape[0]
    if pieces:
        assert len(pieces) == -(-shape[0] // export._csr_step_rows(shape[1]))
        joined = vstack(pieces, format="csr")
        assert joined.dtype == expected.dtype
        np.testing.assert_array_equal(joined.indptr, expected.indptr)
        np.testing.assert_array_equal(joined.indices, expected.indices)
        np.testing.assert_array_equal(joined.data, expected.data)
    else:
        assert shape[0] == 0


def _completed_export_run(umap=True):
    """A completed run whose datastore owner hands the writer a fixed plan."""
    from types import SimpleNamespace

    from scipy.sparse import csr_matrix

    from scarf.datastore.pipeline_run import PipelineRun
    from scarf.storage.pipeline_runs import PipelineRunRecord
    from scarf.writers.export import H5adColumn, H5adExportPlan, H5adMatrix

    assay = SimpleNamespace(name="RNA")

    def frozen(name, values, missing=None):
        array = np.asarray(values)
        return H5adColumn(name, lambda: (array, missing), "categorical")

    class Owner:
        def __init__(self):
            self.zw = zarr.open_group(store=MemoryStore(), mode="w")
            self.cells = SimpleNamespace()
            self.requests = []

        def _get_assay(self, name):
            assert name == "RNA"
            return assay

        def _h5ad_run_plan(self, run, *, matrix):
            self.requests.append((run, matrix))
            counts = csr_matrix(np.asarray([[2, 0], [5, 7]], dtype=np.int32))
            coordinates = np.asarray([[1.5, 10.5], [3.5, 30.5]], dtype=np.float32)
            return H5adExportPlan(
                x=H5adMatrix(
                    shape=(2, 2),
                    dtype=np.dtype(np.int32),
                    blocks=lambda: iter([counts]),
                ),
                obs_index=frozen("ids", ["c1", "c3"]),
                obs=(
                    frozen("I", [True, True]),
                    frozen("names", ["frozen-a", "frozen-c"]),
                    frozen("batch", ["x", "x"]),
                    frozen("clusters", [0, 2]),
                    frozen("site", ["s1", "s1"], np.asarray([False, True])),
                ),
                var_index=frozen("gene_ids", ["g1", "g3"]),
                var=(frozen("names", ["frozen-g1", "frozen-g3"]),),
                obsm={"X_umap": lambda: coordinates} if umap else {},
            )

    owner = Owner()
    record = PipelineRunRecord(
        run_id="a" * 64,
        recipe="basic_rna_analysis",
        requested_label="export",
        label="export",
        assay="RNA",
        started_at_ns=1,
        finished_at_ns=2,
        status="completed",
        complete=True,
        scarf_version="1.0.0",
        config={},
        stage_order=("input_snapshot",),
        outputs=(),
        fields=(),
        error=None,
        interruption=None,
    )
    return assay, owner, PipelineRun(owner, record)


def test_to_h5ad_exports_completed_run_frozen_fields_and_artifact_layout(tmp_path):
    import h5py
    from anndata import read_h5ad

    from scarf.writers import to_h5ad

    assay, owner, run = _completed_export_run()
    path = tmp_path / "run_export.h5ad"

    to_h5ad(assay, str(path), run=run)

    assert owner.requests == [(run, "raw")]
    exported = read_h5ad(path)
    assert list(exported.obs_names) == ["c1", "c3"]
    assert list(exported.var_names) == ["g1", "g3"]
    assert exported.obs.index.name == "ids"
    assert exported.var.index.name == "gene_ids"
    assert exported.obs["names"].tolist() == ["frozen-a", "frozen-c"]
    assert exported.obs["batch"].tolist() == ["x", "x"]
    assert exported.obs["clusters"].tolist() == [0, 2]
    assert exported.obs["site"].isna().tolist() == [False, True]
    assert "umap_1" not in exported.obs
    assert "umap_2" not in exported.obs
    np.testing.assert_allclose(
        exported.obsm["X_umap"],
        np.asarray([[1.5, 10.5], [3.5, 30.5]]),
    )
    np.testing.assert_array_equal(
        exported.X.toarray(),
        np.asarray([[2, 0], [5, 7]]),
    )
    # A run export keeps AnnData's encoding: text that repeats or is missing
    # is categorical, distinct text a string array, and the indexes keep
    # their names.
    with h5py.File(path, "r") as h5:
        assert h5["obs"].attrs["_index"] == "ids"
        assert h5["var"].attrs["_index"] == "gene_ids"
        assert h5["obs/batch"].attrs["encoding-type"] == "categorical"
        assert h5["obs/site"].attrs["encoding-type"] == "categorical"
        assert h5["obs/site/codes"].dtype == np.int8
        assert h5["obs/names"].attrs["encoding-type"] == "string-array"


def test_to_h5ad_run_export_does_not_invent_umap_without_frozen_umap(tmp_path):
    from anndata import read_h5ad

    from scarf.writers import to_h5ad

    assay, _owner, run = _completed_export_run(umap=False)
    path = tmp_path / "run_export_no_umap.h5ad"
    to_h5ad(assay, str(path), run=run)
    exported = read_h5ad(path)
    assert "X_umap" not in exported.obsm
    assert exported.obs["clusters"].tolist() == [0, 2]


def test_to_h5ad_run_export_rejects_foreign_assay_and_live_options(tmp_path):
    from types import SimpleNamespace

    from scarf.writers import to_h5ad

    assay, owner, run = _completed_export_run()
    path = tmp_path / "rejected_run_export.h5ad"

    with pytest.raises(ValueError, match="exact run assay"):
        to_h5ad(SimpleNamespace(name="RNA"), str(path), run=run)
    with pytest.raises(ValueError, match="embeddings_cols"):
        to_h5ad(assay, str(path), embeddings_cols=[], run=run)
    with pytest.raises(ValueError, match="skip_recalc_nfeats"):
        to_h5ad(assay, str(path), skip_recalc_nfeats=False, run=run)
    with pytest.raises(ValueError, match="nthreads"):
        to_h5ad(assay, str(path), nthreads=2, run=run)
    with pytest.raises(TypeError, match="PipelineRun"):
        to_h5ad(assay, str(path), run=object())
    with pytest.raises(ValueError, match="matrix must be either 'raw' or 'normed'"):
        to_h5ad(assay, str(path), run=run, matrix="scaled")
    with pytest.raises(ValueError, match="matrix='normed' requires run"):
        to_h5ad(assay, str(path), matrix="normed")
    assert owner.requests == []
    assert not path.exists()


def test_to_h5ad_run_export_writes_without_anndata_from_a_datastore_owner(
    tmp_path, monkeypatch
):
    import sys

    import h5py

    from scarf.writers import to_h5ad

    assay, owner, run = _completed_export_run()
    path = tmp_path / "run_export.h5ad"
    with monkeypatch.context() as patched:
        patched.setitem(sys.modules, "anndata", None)
        to_h5ad(assay, str(path), run=run, matrix="normed")
    assert owner.requests == [(run, "normed")]
    with h5py.File(path, "r") as h5:
        assert h5.attrs["encoding-type"] == "anndata"
        assert h5["obs/ids"].asstr()[:].tolist() == ["c1", "c3"]
        np.testing.assert_array_equal(h5["X/indptr"][:], [0, 1, 3])

    owner._h5ad_run_plan = None
    other = tmp_path / "unowned.h5ad"
    with pytest.raises(TypeError, match="run must be opened from a DataStore"):
        to_h5ad(assay, str(other), run=run)
    assert not other.exists()


def test_to_h5ad_skips_a_metadata_column_of_unsupported_dtype(
    export_assay_store, tmp_path
):
    import h5py
    from loguru import logger

    from scarf.writers import to_h5ad

    assay = export_assay_store.RNA
    pairs = np.zeros(3, dtype=[("a", "<i4"), ("b", "<f4")])
    assay.cells.insert("pairs", pairs, overwrite=True)
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        to_h5ad(assay, str(tmp_path / "pairs.h5ad"))
    finally:
        logger.remove(sink)

    assert any(
        f"Skipping metadata column 'pairs' with unsupported dtype {pairs.dtype}"
        in message
        for message in messages
    )
    with h5py.File(tmp_path / "pairs.h5ad", "r") as h5:
        assert "pairs" not in h5["obs"]
        np.testing.assert_array_equal(h5["obs/_index"].asstr()[:], ["b1", "b2", "b3"])
        np.testing.assert_array_equal(
            h5["obs/RNA_nCounts"][:], _TOY_RNA_COUNTS.sum(axis=1)
        )


def test_to_h5ad_with_a_skipped_column_stays_readable_by_anndata(
    export_assay_store, tmp_path
):
    anndata = pytest.importorskip("anndata")

    from scarf.writers import to_h5ad

    import h5py

    pairs = [("a", "<i4"), ("b", "<f4")]
    assay = export_assay_store.RNA
    assay.cells.insert("pairs", np.zeros(3, dtype=pairs), overwrite=True)
    assay.feats.insert("pairs", np.zeros(assay.feats.N, dtype=pairs), overwrite=True)
    path = tmp_path / "pairs.h5ad"
    to_h5ad(assay, str(path))

    # column-order names exactly the columns written, which AnnData reads.
    with h5py.File(path, mode="r") as h5:
        for table in ("obs", "var"):
            listed = list(h5[table].attrs["column-order"])
            assert "pairs" not in listed and "pairs" not in h5[table]
            assert set(listed) == set(h5[table]) - {"_index"}
    adata = anndata.read_h5ad(path)
    assert "pairs" not in adata.obs.columns and "pairs" not in adata.var.columns
    assert list(adata.obs_names) == ["b1", "b2", "b3"]
    np.testing.assert_array_equal(
        adata.obs["RNA_nCounts"], assay.cells.fetch_all("RNA_nCounts")
    )


def test_to_mtx_preserves_counts_barcodes_and_features(export_assay_store, tmp_path):
    from scipy.io import mmread

    from scarf.writers import to_mtx

    assay = export_assay_store.RNA
    assay.cells._get_array("RNA_nFeatures")[:] = 99
    out_dir = tmp_path / "toy_mtx"
    to_mtx(assay, str(out_dir), compress=False)

    exported = mmread(out_dir / "matrix.mtx", spmatrix=False).tocsr()
    np.testing.assert_array_equal(exported.toarray(), _TOY_RNA_COUNTS.T)

    barcodes = (out_dir / "barcodes.tsv").read_text().splitlines()
    assert barcodes == ["b1", "b2", "b3"]

    features = [
        line.split("\t")
        for line in (out_dir / "genes.tsv").read_text().splitlines()
        if line
    ]
    assert features == [[name, name] for name in ("g1", "g2", "g3", "g4")]


def test_to_mtx_compress_writes_gzipped_matrix_market(export_assay_store, tmp_path):
    import gzip

    from scipy.io import mmread

    from scarf.writers import to_mtx

    assay = export_assay_store.RNA
    assay.cells._get_array("RNA_nFeatures")[:] = 99
    out_dir = tmp_path / "toy_mtx_gz"
    to_mtx(assay, str(out_dir), compress=True)

    assert (out_dir / "matrix.mtx.gz").is_file()
    assert (out_dir / "barcodes.tsv.gz").is_file()
    assert (out_dir / "features.tsv.gz").is_file()

    exported = mmread(
        gzip.open(out_dir / "matrix.mtx.gz", "rt"),
        spmatrix=False,
    ).tocsr()
    np.testing.assert_array_equal(exported.toarray(), _TOY_RNA_COUNTS.T)

    with gzip.open(out_dir / "barcodes.tsv.gz", "rt") as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    assert barcodes == ["b1", "b2", "b3"]

    # Cell Ranger 3 readers expect id, name, and feature type columns.
    with gzip.open(out_dir / "features.tsv.gz", "rt") as handle:
        features = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    assert features == [
        [name, name, "Gene Expression"] for name in ("g1", "g2", "g3", "g4")
    ]


@pytest.mark.parametrize("compress", [False, True])
def test_to_mtx_preserves_fractional_counts(tmp_path, compress):
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    from scarf import DataStore
    from scarf.writers import SparseToZarr, to_mtx

    values = np.array([[1.5, 0, 2.75], [0, 3.25, 1]], dtype=np.float32)
    store = MemoryStore()
    SparseToZarr(
        csr_matrix(values),
        store,
        cell_ids=["c1", "c2"],
        feature_ids=["g1", "g2", "g3"],
        nthreads=1,
    ).dump()
    dataset = DataStore(store, min_features_per_cell=0, nthreads=1)
    out_dir = tmp_path / "fractional_counts"

    to_mtx(dataset.RNA, str(out_dir), compress=compress)

    filename = "matrix.mtx.gz" if compress else "matrix.mtx"
    np.testing.assert_array_equal(
        mmread(out_dir / filename, spmatrix=False).toarray(),
        values.T,
    )


def test_zarr_subset(datastore, tmp_path):
    from tests import full_path

    cell_idx = np.array([1, 10, 100, 500])
    zarr_path = str(tmp_path / "subset.zarr")
    writer = SubsetZarr(zarr_loc=zarr_path, assays=[datastore.RNA], cell_idx=cell_idx)
    writer.dump()
    counts, barcodes, features = _read_cellranger_h5(full_path("1K_pbmc_citeseq.h5"))
    expected = counts[cell_idx][:, features["feature_type"] == "Gene Expression"]
    root = zarr.open_group(zarr_path, mode="r")
    np.testing.assert_array_equal(root["cellData/ids"][:], barcodes[cell_idx])
    np.testing.assert_array_equal(root["RNA/counts"][:], expected.toarray())
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(root["RNA/countsT"][:], expected.T.toarray())
    # Subset store must open as RNAassay under the strip contract.
    from scarf import DataStore

    subset_ds = DataStore(zarr_path, default_assay="RNA", assay_types={"RNA": "RNA"})
    assert subset_ds.RNA.rawDataT is not None
    assert subset_ds.RNA.rawDataT.shape == (root["RNA/counts"].shape[1], 4)


@pytest.mark.parametrize("workspace", [None, "source"])
@pytest.mark.parametrize(
    ("assay_name", "assay_type"), [("expression", "RNA"), ("protein", "ADT")]
)
def test_zarr_subset_preserves_custom_assay_types(workspace, assay_name, assay_type):
    from scipy.sparse import csr_matrix

    from scarf import DataStore
    from scarf.assay import ADTassay, RNAassay
    from scarf.writers import SparseToZarr
    from scarf.writers.counts_t import finalize_writer_counts_t

    values = np.array([[1, 4, 9], [2, 20, 3], [12, 2, 2]], dtype=np.uint16)
    source = MemoryStore()
    writer = SparseToZarr(
        csr_matrix(values),
        source,
        cell_ids=["c1", "c2", "c3"],
        feature_ids=["f1", "f2", "f3"],
        assay_name=assay_name,
        workspace=workspace,
        nthreads=1,
    )
    writer.dump()
    finalize_writer_counts_t(
        writer.z, assay_name, workspace, assay_type=assay_type, nthreads=1
    )
    dataset = DataStore(
        source,
        default_assay=assay_name,
        workspace=workspace,
        min_features_per_cell=0,
        nthreads=1,
    )
    destination = MemoryStore()
    out_workspace = "selected" if workspace is not None else None

    SubsetZarr(
        destination,
        assays=[getattr(dataset, assay_name)],
        in_workspace=workspace,
        out_workspace=out_workspace,
        cell_idx=np.array([0, 2]),
        nthreads=1,
    ).dump()

    result = DataStore(
        destination,
        default_assay=assay_name,
        workspace=out_workspace,
        min_features_per_cell=0,
        nthreads=1,
    )
    assert result.zw.attrs["assayTypes"][assay_name] == assay_type
    assay = getattr(result, assay_name)
    np.testing.assert_array_equal(assay.rawData.compute(), values[[0, 2]])
    if assay_type == "RNA":
        assert isinstance(assay, RNAassay)
        np.testing.assert_array_equal(assay.rawDataT[:], values[[0, 2]].T)
    else:
        assert isinstance(assay, ADTassay)
        selected = values[[0, 2]].astype(float)
        expected = np.log1p(selected / np.exp(np.log1p(selected).mean(axis=0))[None, :])
        np.testing.assert_allclose(assay.normed().compute(), expected, rtol=1e-6)
        assert assay.rawDataT is None


def test_zarr_subset_does_not_copy_source_pipeline_runs(
    datastore_ephemeral,
    tmp_path,
):
    datastore_ephemeral.zw.create_group(f"pipeline/runs/{'e' * 64}/stages")
    zarr_path = str(tmp_path / "subset_without_runs.zarr")

    SubsetZarr(
        zarr_loc=zarr_path,
        assays=[datastore_ephemeral.RNA],
        cell_idx=np.array([1, 10, 100, 500]),
    ).dump()

    assert "pipeline" in datastore_ephemeral.zw
    assert "pipeline" not in zarr.open_group(zarr_path, mode="r")


def test_subset_zarr_rejects_invalid_assay_inputs():
    with pytest.raises(TypeError, match="should be a list"):
        SubsetZarr._check_assays("RNA")
    with pytest.raises(ValueError, match="at least one assay"):
        SubsetZarr._check_assays([])
    with pytest.raises(ValueError, match="actual assay objects"):
        SubsetZarr._check_assays([object()])
    with pytest.raises(ValueError, match="same numer of cells"):
        SubsetZarr._check_assays(
            [
                _FakeAssay("RNA", 3),
                _FakeAssay("ATAC", 4),
            ]
        )
    # Assays of two datastores with as many cells hold two cell tables.
    with pytest.raises(ValueError, match="not from the same DataStore"):
        SubsetZarr._check_assays([_FakeAssay("RNA", 3), _FakeAssay("ATAC", 3)])
    rna = _FakeAssay("RNA", 3)
    with pytest.raises(ValueError, match="must not repeat"):
        SubsetZarr._check_assays([rna, rna])
    adt = _FakeAssay("ADT", 3)
    adt.cells = rna.cells
    assert SubsetZarr._check_assays([rna, adt]) == [rna, adt]


def test_subset_zarr_requires_cell_key_or_indices():
    subset = object.__new__(SubsetZarr)
    subset.assays = []
    subset.assays = [_FakeAssay("RNA", 3)]

    with pytest.raises(ValueError, match="cannot be None"):
        subset._check_idx(None, None)


@pytest.mark.parametrize(
    ("cell_idx", "message"),
    [
        (np.array([0.5]), "integer type"),
        (np.array([3]), "max value"),
    ],
)
def test_subset_zarr_rejects_invalid_explicit_indices(cell_idx, message):
    subset = object.__new__(SubsetZarr)
    subset.assays = []
    subset.assays = [_FakeAssay("RNA", 3)]

    with pytest.raises(ValueError, match=message):
        subset._check_idx(None, cell_idx)


def test_subset_zarr_validates_cell_key():
    subset = object.__new__(SubsetZarr)
    subset.assays = []
    subset.assays = [_FakeAssay("RNA", 3)]
    with pytest.raises(ValueError, match="was not found"):
        subset._check_idx("selected", None)

    subset.assays = [
        _FakeAssay("RNA", 3, {"selected": np.array([1, 0, 1])}),
    ]
    with pytest.raises(ValueError, match="not of boolean type"):
        subset._check_idx("selected", None)


def test_subset_zarr_resolves_consistent_cell_key():
    selected = np.array([True, False, True, False])
    subset = object.__new__(SubsetZarr)
    subset.assays = []
    subset.assays = [
        _FakeAssay("RNA", 4, {"selected": selected}),
        _FakeAssay("ATAC", 4, {"selected": selected.copy()}),
    ]

    np.testing.assert_array_equal(
        subset._check_idx("selected", None),
        np.array([0, 2]),
    )


def test_subset_zarr_rejects_different_cell_masks():
    subset = object.__new__(SubsetZarr)
    subset.assays = []
    subset.assays = [
        _FakeAssay("RNA", 2, {"selected": np.array([True, False])}),
        _FakeAssay("ATAC", 2, {"selected": np.array([False, True])}),
    ]
    with pytest.raises(ValueError, match="cell_key selected is not consistent"):
        subset._check_idx("selected", None)


@pytest.mark.parametrize("on_disk", [False, True])
def test_subset_zarr_never_replaces_foreign_content(tmp_path, memory_source, on_disk):
    store = LocalStore(tmp_path / "existing.zarr") if on_disk else MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("existing", data=np.array([123]))

    for overwrite, reason in (
        (False, "is not empty"),
        (True, "holds 'existing', which is not part of a Scarf store"),
    ):
        with pytest.raises(FileExistsError, match=reason):
            SubsetZarr(
                store,
                [memory_source.RNA],
                cell_idx=np.array([0]),
                overwrite_existing_file=overwrite,
            )

    np.testing.assert_array_equal(root["existing"][:], [123])


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("assays", "actual assay objects"),
        ("cell_idx", "max value"),
        ("out_workspace", "must not contain path separators"),
    ],
)
def test_subset_zarr_validates_inputs_before_overwriting(invalid, message):
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("existing", data=np.array([123]))

    with pytest.raises(ValueError, match=message):
        SubsetZarr(
            store,
            assays=[object()] if invalid == "assays" else [_FakeAssay("RNA", 2)],
            cell_idx=np.array([2 if invalid == "cell_idx" else 0]),
            out_workspace="a/b" if invalid == "out_workspace" else None,
            overwrite_existing_file=True,
        )

    np.testing.assert_array_equal(root["existing"][:], [123])


def test_subset_zarr_remote_uri_checks_contents(memory_source, monkeypatch):
    calls = []
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("existing", data=np.array([123]))

    def make_store(location, storage_options=None, read_only=False):
        calls.append((location, storage_options, read_only))
        return store

    monkeypatch.setattr("scarf.storage.stores.make_store", make_store)
    options = {"access_key_id": "key"}
    with pytest.raises(FileExistsError, match="is not empty"):
        SubsetZarr(
            "s3://bucket/out.zarr",
            [memory_source.RNA],
            cell_idx=np.array([0]),
            storage_options=options,
        )
    assert calls == [("s3://bucket/out.zarr", options, True)]
    np.testing.assert_array_equal(root["existing"][:], [123])


def test_subset_zarr_refuses_to_overwrite_after_probe_failure(
    memory_source, monkeypatch
):
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_array("existing", data=np.array([123]))

    async def fail_probe(prefix):
        raise OSError("Cannot inspect destination")

    monkeypatch.setattr(store, "is_empty", fail_probe)
    with pytest.raises(OSError, match="Cannot inspect destination"):
        SubsetZarr(
            store,
            [memory_source.RNA],
            cell_idx=np.array([0]),
            overwrite_existing_file=True,
        )

    np.testing.assert_array_equal(root["existing"][:], [123])


def test_crtozarr_forwards_storage_options(monkeypatch):
    captured = {}
    probed = []

    def fake_load_zarr(zarr_loc, mode, storage_options=None):
        captured["zarr_loc"] = zarr_loc
        captured["mode"] = mode
        captured["storage_options"] = storage_options
        return zarr.open_group(store=MemoryStore(), mode="w")

    def fake_make_store(location, storage_options=None, read_only=False):
        # The destination check reads the store before it is created.
        probed.append(storage_options)
        return MemoryStore()

    monkeypatch.setattr("scarf.storage.stores.load_zarr", fake_load_zarr)
    monkeypatch.setattr("scarf.storage.stores.make_store", fake_make_store)
    monkeypatch.setattr("scarf.storage.schema.create_cell_data", lambda **kwargs: None)
    monkeypatch.setattr(
        "scarf.storage.schema.create_zarr_count_assay",
        lambda **kwargs: None,
    )

    import pandas as pd

    class FakeCr:
        nCells = nFeatures = 1
        matrix_dtype = np.dtype(np.uint8)
        assayFeats = pd.DataFrame(
            {"RNA": ["Gene Expression", 0, 1, 1]},
            index=["type", "start", "end", "nFeatures"],
        )

        def cell_names(self):
            return ["c1"]

        def feature_ids(self, assay_name):
            return ["f1"]

        def feature_names(self, assay_name):
            return ["f1"]

        def count_value_ranges(self, maxBytes, featureGroups=None):
            return [CountValueRange()]

        def max_window_nnz(self, window_rows):
            return 0

        def producer_staging_bytes(self, batch_size, lines_in_mem):
            return 0

    CrToZarr(
        FakeCr(),
        zarr_loc="s3://bucket/out.zarr",
        storage_options={"access_key_id": "id"},
    )
    assert captured["storage_options"] == {"access_key_id": "id"}
    assert probed and all(options == {"access_key_id": "id"} for options in probed)


def test_h5adtozarr_applies_storage_resources_and_chunk_controls(tmp_path):
    from scarf.readers import H5adReader
    from scarf.storage.layout import count_array_spec
    from scarf.writers import H5adToZarr

    values = (np.arange(100 * 50, dtype=np.uint32).reshape(100, 50) % 1_000).astype(
        np.uint16
    )
    path = _write_h5ad(tmp_path / "layout_controls.h5ad", values)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(
            reader,
            zarr_loc=MemoryStore(),
            mem_budget="2G",
            nthreads=3,
            policy=CountMatrixPolicy(unitBytes=20_480, chunkBytes=4_096),
        )
    finally:
        reader.h5.close()

    expected = count_array_spec(
        100,
        50,
        dtype=np.uint16,
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=20_480, chunkBytes=4_096),
    )
    counts = writer.z["RNA/counts"]
    assert writer.resources.memoryBytes == 2 * 1024**3
    assert writer.resources.workers == 3
    assert counts.chunks == expected.chunks
    assert counts.metadata.shards == expected.shards


def test_count_matrix_bands_project_and_reject_uninitialized_split() -> None:
    from scipy.sparse import coo_matrix

    from scarf.storage.sharding import SparseShardBuffer
    from scarf.writers.h5ad import _count_matrix_bands, _finished_count_bands

    root = zarr.open_group(store=MemoryStore(), mode="w")
    rna = root.create_array(
        "RNA",
        shape=(4, 2),
        chunks=(2, 2),
        shards=(2, 2),
        dtype=np.uint32,
    )
    adt = root.create_array(
        "ADT",
        shape=(4, 2),
        chunks=(2, 2),
        shards=(2, 2),
        dtype=np.uint32,
    )
    buffers = {
        "RNA": SparseShardBuffer(rna, startRow=0, endRow=4),
        "ADT": SparseShardBuffer(adt, startRow=0, endRow=4),
    }
    chunk = coo_matrix(np.arange(16, dtype=np.uint32).reshape(4, 4))
    with pytest.raises(RuntimeError, match="Multi-assay projection"):
        list(
            _count_matrix_bands(
                chunk,
                buffers,
                ("RNA", "ADT"),
                None,
            )
        )
    # Source features 0 and 2 belong to RNA and 1 and 3 to ADT.
    codes = np.array([0, 1, 0, 1], dtype=np.int64)
    columns = np.array([0, 0, 1, 1], dtype=np.int64)
    projected = list(
        _count_matrix_bands(
            chunk,
            buffers,
            ("RNA", "ADT"),
            (codes, columns),
        )
    )
    # The four rows fill both two-row bands of each assay, so none is left.
    assert [(name, band.start, band.end) for name, band, _bytes in projected] == [
        ("RNA", 0, 2),
        ("RNA", 2, 4),
        ("ADT", 0, 2),
        ("ADT", 2, 4),
    ]
    assert list(_finished_count_bands(buffers)) == []
    values = chunk.toarray()
    for name, source_columns in (("RNA", [0, 2]), ("ADT", [1, 3])):
        np.testing.assert_array_equal(
            np.vstack(
                [
                    band.dense()
                    for band_name, band, _bytes in projected
                    if band_name == name
                ]
            ),
            values[:, source_columns],
        )
    # A producer holds at least the bytes of the bands it projected.
    assert all(
        producer_bytes >= band.sparseBytes for _name, band, producer_bytes in projected
    )


def test_h5ad_process_windows_run_in_the_parent_process(tmp_path) -> None:
    import threading

    from scarf.readers import H5adReader
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.count_matrix import plan_count_matrix_pair
    from scarf.writers import H5adToZarr
    from scarf.writers.h5ad import (
        _read_h5ad_process_window,
        _write_h5ad_process_window,
    )

    values = np.arange(8 * 4, dtype=np.uint32).reshape(8, 4)
    h5ad_path = _write_h5ad(tmp_path / "cells.h5ad", values)
    zarr_loc = str(tmp_path / "cells.zarr")
    missing = {"h5ad_fn": str(tmp_path / "missing.h5ad")}

    class _StopOnCall:
        """A stop that is set from its ``trigger``-th check on."""

        def __init__(self, trigger: int) -> None:
            self.calls = 0
            self.trigger = trigger

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls >= self.trigger

    def stopped() -> threading.Event:
        stop = threading.Event()
        stop.set()
        return stop

    reader = H5adReader(str(h5ad_path))
    try:
        # Constructing the writer creates the counts that write windows fill.
        writer = H5adToZarr(reader, zarr_loc=zarr_loc, **_SHARD_BAND_BUDGET)
        plan = plan_count_matrix_pair(
            values.shape[0],
            values.shape[1],
            writer.storageDtypes["RNA"],
            policy=_SHARD_BAND_BUDGET["policy"],
        )
        # Twelve bytes of uint8 counts make bands of three cells, so the last
        # band of the eight cells holds two and is cut only when they end.
        assert writer.storageDtypes["RNA"] == np.uint8
        assert int(plan.counts.shards[0]) == 3
        counts = writer.z["RNA/counts"]

        def read(stop, kwargs=None, connection=None) -> list:
            connection = _Pipe() if connection is None else connection
            _read_h5ad_process_window(
                reader._clone_kwargs() if kwargs is None else kwargs,
                2,
                0,
                values.shape[0],
                {"RNA": plan.counts},
                ("RNA",),
                None,
                connection,
                stop,
            )
            assert connection.closed
            return connection.messages

        def write(stop, kwargs=None, connection=None) -> list:
            counts[:] = 0
            connection = _Pipe() if connection is None else connection
            _write_h5ad_process_window(
                reader._clone_kwargs() if kwargs is None else kwargs,
                2,
                0,
                values.shape[0],
                zarr_loc,
                None,
                None,
                ("RNA",),
                None,
                ResourceBudget(1024 * 1024, 2),
                0,
                1,
                None,
                connection,
                stop,
            )
            assert connection.closed
            return connection.messages

        def band_rows(messages) -> list[tuple[str, int, int]]:
            return [
                (payload[0], payload[1].start, payload[1].end)
                for kind, payload in messages
                if kind == "band"
            ]

        # A complete read window sends every band of its rows, then done.
        complete = read(threading.Event())
        assert [kind for kind, _payload in complete] == ["band"] * 3 + ["done"]
        assert band_rows(complete[:3]) == [("RNA", 0, 3), ("RNA", 3, 6), ("RNA", 6, 8)]
        np.testing.assert_array_equal(
            np.vstack([band.dense() for _kind, (_name, band, _bytes) in complete[:3]]),
            values,
        )
        # A window stopped before its first batch, or after its first batch
        # but before any band is cut, sends nothing.
        assert read(stopped()) == []
        assert read(_StopOnCall(2)) == []
        # A stop between batches ends the window after the bands it sent.
        assert band_rows(read(_StopOnCall(5))) == [("RNA", 0, 3)]
        # A stop that follows the last band suppresses done.
        final = _Pipe()

        class _StopAfterBands:
            def is_set(self) -> bool:
                return sum(kind == "band" for kind, _payload in final.messages) >= 3

        assert band_rows(read(_StopAfterBands(), connection=final)) == band_rows(
            complete
        )
        assert [kind for kind, _payload in final.messages] == ["band"] * 3
        # A reader that cannot open its file reports the error, and a pipe
        # that cannot carry the report is still closed.
        ((kind, message),) = read(threading.Event(), kwargs=missing)
        assert kind == "error"
        assert message.startswith("FileNotFoundError: ")
        assert (
            read(threading.Event(), kwargs=missing, connection=_Pipe(fail_send=True))
            == []
        )

        # A complete write window fills its rows and reports their summaries.
        ((kind, (_reports, windows)),) = write(threading.Event())
        assert kind == "done"
        np.testing.assert_array_equal(counts[:], values)
        start, _digests, row_sums, row_positive, column_positive = windows["RNA"]
        assert start == 0
        np.testing.assert_array_equal(row_sums, values.sum(axis=1))
        np.testing.assert_array_equal(row_positive, np.count_nonzero(values, axis=1))
        np.testing.assert_array_equal(column_positive, np.count_nonzero(values, axis=0))
        # A stopped write window writes nothing.
        assert write(stopped()) == []
        np.testing.assert_array_equal(counts[:], 0)
        # A stop after the last source batch leaves the final band unwritten
        # and reports nothing.
        source_batches = (values.shape[0] + 2 - 1) // 2
        assert write(_StopOnCall(source_batches + 1)) == []
        np.testing.assert_array_equal(counts[:6], values[:6])
        np.testing.assert_array_equal(counts[6:], 0)
        assert (
            write(threading.Event(), kwargs=missing, connection=_Pipe(fail_send=True))
            == []
        )
        np.testing.assert_array_equal(counts[:], 0)
    finally:
        reader.close()


def test_h5ad_import_links_missing_masks_and_keeps_nullable_booleans(tmp_path):
    import h5py

    from scarf.readers import H5adReader
    from scarf.storage.arrays import linked_missing_mask
    from scarf.writers import H5adToZarr

    path = _write_h5ad(tmp_path / "nullable.h5ad", np.eye(3, dtype=np.uint16))
    text = h5py.string_dtype()
    with h5py.File(path, "r+") as h5:
        obs = h5["obs"]
        batch = obs.create_group("batch")
        batch.attrs["encoding-type"] = "categorical"
        batch.create_dataset("codes", data=np.array([0, -1, 1], dtype=np.int8))
        batch.create_dataset(
            "categories", data=np.array(["a", "b"], dtype=object), dtype=text
        )
        flag = obs.create_group("flag")
        flag.attrs["encoding-type"] = "nullable-boolean"
        flag.create_dataset("values", data=np.array([True, False, False]))
        flag.create_dataset("mask", data=np.array([False, True, False]))
        note = obs.create_group("note")
        note.attrs["encoding-type"] = "nullable-string-array"
        note.create_dataset(
            "values", data=np.array(["x", "", ""], dtype=object), dtype=text
        )
        note.create_dataset("mask", data=np.array([False, True, False]))
        obs.create_dataset("score", data=np.array([0.5, 1.5, 2.5]))

    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        writer = H5adToZarr(reader, zarr_loc=MemoryStore(), nthreads=1)
        writer.dump()
    finally:
        reader.close()

    cells = writer.z["cellData"]
    expected = {
        "batch": (["a", "", "b"], [False, True, False]),
        "flag": ([True, False, False], [False, True, False]),
        # A genuine empty string stays distinct from a missing value.
        "note": (["x", "", ""], [False, True, False]),
    }
    for name, (values, missing) in expected.items():
        mask = linked_missing_mask(cells, name)
        assert mask is not None, name
        np.testing.assert_array_equal(cells[name][:], values)
        np.testing.assert_array_equal(mask[:], missing)
    assert cells["flag"].dtype == np.dtype(bool)
    assert linked_missing_mask(cells, "score") is None


def test_to_h5ad_writes_anndata_encodings_that_round_trip(export_assay_store, tmp_path):
    anndata = pytest.importorskip("anndata")
    import warnings

    from anndata._warnings import OldFormatWarning

    from scarf.writers import to_h5ad

    assay = export_assay_store.RNA
    path = tmp_path / "encoded.h5ad"
    to_h5ad(assay, str(path))

    with warnings.catch_warnings():
        warnings.simplefilter("error", OldFormatWarning)
        adata = anndata.read_h5ad(path)
    assert "_index" not in adata.obs.columns
    assert "_index" not in adata.var.columns
    # The toy Cell Ranger directory holds these RNA counts and identifiers.
    assert list(adata.obs_names) == ["b1", "b2", "b3"]
    assert list(adata.var_names) == ["g1", "g2", "g3", "g4"]
    np.testing.assert_array_equal(adata.X.toarray(), _TOY_RNA_COUNTS)
    rewritten = tmp_path / "rewritten.h5ad"
    adata.write_h5ad(rewritten)
    reread = anndata.read_h5ad(rewritten)
    assert list(reread.obs_names) == list(adata.obs_names)
    assert list(reread.var_names) == list(adata.var_names)
    np.testing.assert_array_equal(reread.X.toarray(), _TOY_RNA_COUNTS)


def test_to_h5ad_orders_embedding_components_numerically(export_assay_store, tmp_path):
    import h5py

    from scarf.writers import to_h5ad

    assay = export_assay_store.RNA
    n_cells = assay.cells.N
    for component in range(1, 13):
        assay.cells.insert(
            f"RNA_PCA{component}",
            np.full(n_cells, float(component)),
            overwrite=True,
        )
    path = tmp_path / "components.h5ad"
    to_h5ad(assay, str(path), embeddings_cols=["PCA"])

    with h5py.File(path, "r") as h5:
        assert set(h5["obsm"]) == {"X_pca"}
        np.testing.assert_array_equal(h5["obsm/X_pca"][0], np.arange(1, 13))


def test_csv_import_keeps_later_fractional_counts_and_missing_text(tmp_path):
    from scarf.readers import CSVReader
    from scarf.storage.arrays import linked_missing_mask

    path = tmp_path / "counts.csv"
    path.write_text("cell,g1,g2,label\nc1,1,0,a\nc2,0,2,b\nc3,2.5,7,\n")
    reader = CSVReader(str(path), id_column=0, batch_size=2, cell_data_cols=["label"])
    store = MemoryStore()
    CSVtoZarr(reader, store, assay_name="RNA", nthreads=1).dump()

    root = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root["RNA/counts"][:], [[1, 0], [0, 2], [2.5, 7]])
    np.testing.assert_array_equal(root["cellData/label"][:], ["a", "b", ""])
    mask = linked_missing_mask(root["cellData"], "label")
    assert mask is not None
    np.testing.assert_array_equal(mask[:], [False, False, True])


def test_import_writers_accept_an_explicit_assay_type(tmp_path):
    from scipy.sparse import csr_matrix

    from scarf.writers import SparseToZarr

    matrix = csr_matrix(np.arange(1, 13, dtype=np.uint32).reshape(4, 3))
    store = MemoryStore()
    SparseToZarr(
        matrix,
        store,
        cell_ids=[f"c{i}" for i in range(4)],
        feature_ids=["g0", "g1", "g2"],
        assay_name="GEX",
        assay_type="RNA",
        nthreads=1,
    ).dump()

    root = zarr.open_group(store=store, mode="r")
    assert root.attrs["assayTypes"] == {"GEX": "RNA"}
    np.testing.assert_array_equal(root["GEX/countsT"][:], matrix.toarray().T)

    untouched = MemoryStore()
    zarr.open_group(store=untouched, mode="w").create_group("sentinel")
    # The error names the assay, which defaults to RNA.
    with pytest.raises(ValueError, match="assay_type 'rna' of assay 'RNA' is not a"):
        SparseToZarr(
            matrix,
            untouched,
            cell_ids=[f"c{i}" for i in range(4)],
            feature_ids=["g0", "g1", "g2"],
            assay_type="rna",
        )
    assert set(zarr.open_group(store=untouched, mode="r").group_keys()) == {"sentinel"}


def test_h5ad_worker_messages_reject_closed_pipes_and_worker_errors():
    from multiprocessing import Pipe
    from types import SimpleNamespace

    from scarf.writers.h5ad import _worker_messages

    worker = SimpleNamespace(exitcode=None, name="h5ad-writer-0")
    receiver, sender = Pipe(duplex=False)
    sender.close()
    with pytest.raises(RuntimeError, match="writer 0 closed without a result"):
        list(_worker_messages({receiver: 0}, [worker], "writer"))

    receiver, sender = Pipe(duplex=False)
    sender.send(("error", "ValueError: boom"))
    with pytest.raises(RuntimeError, match="producer 0 failed: ValueError: boom"):
        list(_worker_messages({receiver: 0}, [worker], "producer"))
    sender.close()


def test_h5ad_stop_workers_escalates_and_drains_pipes(monkeypatch):
    import threading
    from multiprocessing import Pipe
    from types import SimpleNamespace

    import scarf.writers.h5ad as h5ad_writer

    class Worker:
        def __init__(self, survives_terminate: bool) -> None:
            self.alive = True
            self.survives_terminate = survives_terminate
            self.calls: list[str] = []

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.calls.append("join")

        def terminate(self) -> None:
            self.calls.append("terminate")
            self.alive = self.survives_terminate

        def kill(self) -> None:
            self.calls.append("kill")
            self.alive = False

    clock = iter(range(0, 100, 3))
    monkeypatch.setattr(
        h5ad_writer, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    receiver, sender = Pipe(duplex=False)
    sender.send(("rows", "blocked"))
    closed, closed_sender = Pipe(duplex=False)
    closed.close()
    stop = threading.Event()
    stubborn, stopping = Worker(True), Worker(False)

    h5ad_writer._stop_workers(stop, [stubborn, stopping], [receiver, closed])

    assert stop.is_set()
    # One drain pass joins without waiting, then the deadline forces termination.
    assert stubborn.calls == ["join", "terminate", "join", "kill", "join"]
    assert stopping.calls == ["join", "terminate", "join"]
    assert receiver.closed and closed.closed
    sender.close()
    closed_sender.close()


def test_h5ad_writer_rejects_assay_type_with_assay_split_key(tmp_path):
    from scarf.writers import H5adToZarr

    with pytest.raises(ValueError, match="cannot be combined"):
        H5adToZarr(
            None,  # type: ignore[arg-type]
            str(tmp_path / "out.zarr"),
            assay_type="RNA",
            assay_split_key="feature_types",
        )
    assert not (tmp_path / "out.zarr").exists()


def test_h5ad_worker_messages_wait_for_a_clean_exit_result(monkeypatch):
    import multiprocessing.connection as connection_module
    from multiprocessing import Pipe
    from types import SimpleNamespace

    from scarf.writers.h5ad import _worker_messages

    real_wait = connection_module.wait
    waits: list[float | None] = []

    def first_wait_times_out(objects, timeout=None):
        waits.append(timeout)
        return [] if len(waits) == 1 else real_wait(objects, timeout)

    monkeypatch.setattr(connection_module, "wait", first_wait_times_out)
    receiver, sender = Pipe(duplex=False)
    sender.send(("done", "result"))
    sender.close()
    # The worker already exited cleanly, so a wait that times out before its
    # result is read is not a failure; the next wait reads the result.
    worker = SimpleNamespace(exitcode=0, name="h5ad-writer-0")
    messages = list(_worker_messages({receiver: 0}, [worker], "writer"))
    assert messages == [(0, "done", "result")]
    assert len(waits) == 2

    waits.clear()
    receiver, sender = Pipe(duplex=False)
    failed = SimpleNamespace(exitcode=3, name="h5ad-writer-1")
    with pytest.raises(RuntimeError, match="h5ad-writer-1 exitcode=3"):
        list(_worker_messages({receiver: 0}, [failed], "writer"))
    assert len(waits) == 1
    sender.close()
