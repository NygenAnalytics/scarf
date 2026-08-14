import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay import Assay, RNAassay
from scarf.features.genomic.melding import coordinate_melding
from scarf.features.markers import find_markers_by_rank, find_markers_by_regression
from scarf.metadata import MetaData
from scarf.quality_control.doublets import write_doublet_target_zarr
from scarf.storage.budget import ResourceBudget
from scarf.storage.layout import ZarrArraySpec
from scarf.storage.profiles import resolve_storage_profile
from scarf.storage.sharding import counts_t_spec, write_counts_t
from scarf.writers import (
    create_cell_data,
    create_zarr_count_assay,
)
from tests.store_probes import RecordingStore


def _memory_root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _write_small_assay(
    root: zarr.Group,
    *,
    workspace: str | None,
    values: np.ndarray,
) -> zarr.Array:
    n_cells, n_feats = values.shape
    create_cell_data(
        root,
        workspace,
        ids=np.array([f"c{i}" for i in range(n_cells)]),
        names=np.array([f"c{i}" for i in range(n_cells)]),
    )
    create_zarr_count_assay(
        z=root,
        assay_name="RNA",
        workspace=workspace,
        n_cells=n_cells,
        feat_ids=np.array([f"f{i}" for i in range(n_feats)]),
        feat_names=np.array([f"g{i}" for i in range(n_feats)]),
        dtype="uint32",
    )
    if workspace is None:
        counts = root["RNA/counts"]
    else:
        counts = root["matrices/RNA/counts"]
    counts[:] = values
    group = root["RNA"] if workspace is None else root["matrices/RNA"]
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(1024**3, 2),
    )
    return counts


@pytest.mark.parametrize("workspace", [None, "ws"])
def test_explicit_write_counts_t_builds_complete_counts_t(workspace):
    root = _memory_root()
    values = np.arange(20, dtype=np.uint32).reshape(5, 4)
    counts = _write_small_assay(root, workspace=workspace, values=values)

    if workspace is None:
        group = root["RNA"]
    else:
        group = root["matrices/RNA"]
    assert "countsT" in group
    counts_t = group["countsT"]
    assert counts_t.attrs["complete"] is True
    assert counts_t.shape == (4, 5)
    assert counts_t.dtype == counts.dtype
    np.testing.assert_array_equal(counts_t[:], values.T)


def test_counts_t_spec_matches_write_layout_and_data():
    root = _memory_root()
    group = root.create_group("RNA")
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    counts = group.create_array(
        "counts",
        data=values,
        chunks=(3, 2),
    )
    profile = resolve_storage_profile(root.store)
    preview = counts_t_spec(
        ZarrArraySpec(
            shape=tuple(int(value) for value in counts.shape),
            chunks=tuple(int(value) for value in counts.chunks),
            dtype=counts.dtype,
            compressors=None,
            shards=None,
            fillValue=0,
            overwrite=True,
        ),
        profile=profile,
    )
    written = write_counts_t(counts, group, profile=profile)
    assert written is not None
    assert tuple(int(value) for value in written.shape) == preview.shape
    assert np.dtype(written.dtype) == np.dtype(preview.dtype)
    assert tuple(int(value) for value in written.chunks) == preview.chunks
    assert written.attrs.get("complete") is True
    np.testing.assert_array_equal(written[:], values.T)


def test_write_counts_t_marks_incomplete_until_finished():
    root = _memory_root()
    group = root.create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=(4, 3),
        chunks=(2, 2),
        dtype=np.uint32,
        fill_value=0,
    )
    counts[:] = np.arange(12, dtype=np.uint32).reshape(4, 3)
    counts_t = write_counts_t(counts, group)
    assert counts_t is not None
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(counts_t[:], counts[:].T)


def test_write_counts_t_uses_strip_layout_independent_of_source_chunks():
    from scarf.storage.sharding import choose_strip_layout
    from scarf.storage.types import array_metadata_shards

    root = _memory_root()
    group = root.create_group("RNA")
    values = np.arange(22 * 7, dtype=np.uint32).reshape(22, 7)
    counts = group.create_array(
        "counts",
        data=values,
        chunks=(6, 4),
    )

    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(1024**2, 2),
    )

    assert counts_t is not None
    expected = choose_strip_layout(7, 22, values.dtype.itemsize)
    assert counts_t.chunks == (expected["geneStrip"], expected["cellChunk"])
    assert array_metadata_shards(counts_t) == (
        expected["geneStrip"],
        expected["shardCells"],
    )
    np.testing.assert_array_equal(counts_t[:], values.T)


def test_assay_loads_counts_t_and_falls_back_when_incomplete():
    root = _memory_root()
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)

    cells = MetaData(root["cellData"])
    assay = Assay(root, None, "RNA", cells, nthreads=1)
    assert assay.rawDataT is not None
    np.testing.assert_array_equal(assay.rawDataT[:], values.T)

    root["RNA/countsT"].attrs["complete"] = False
    broken = Assay(root, None, "RNA", cells, nthreads=1)
    assert broken.rawDataT is None

    root["RNA/countsT"].attrs["complete"] = "false"
    string_flag = Assay(root, None, "RNA", cells, nthreads=1)
    assert string_flag.rawDataT is None


def test_assay_falls_back_on_wrong_shape_dtype_or_group_node():
    root = _memory_root()
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])

    del root["RNA/countsT"]
    root["RNA"].create_array(
        "countsT",
        shape=(2, 3),
        chunks=(2, 2),
        dtype=np.uint32,
        fill_value=0,
    )
    root["RNA/countsT"].attrs["complete"] = True
    assert Assay(root, None, "RNA", cells, nthreads=1).rawDataT is None

    del root["RNA/countsT"]
    root["RNA"].create_array(
        "countsT",
        shape=(4, 3),
        chunks=(2, 2),
        dtype=np.float32,
        fill_value=0,
    )
    root["RNA/countsT"].attrs["complete"] = True
    assert Assay(root, None, "RNA", cells, nthreads=1).rawDataT is None

    del root["RNA/countsT"]
    root["RNA"].create_group("countsT")
    assert Assay(root, None, "RNA", cells, nthreads=1).rawDataT is None


def test_rna_requires_strip_counts_t():
    root = _memory_root()
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(root, "RNA", cells, workspace=None, nthreads=1)
    assay.sf = 1000
    assert assay.rawDataT is not None
    cell_idx = np.array([0, 2, 3])
    feat_idx = np.array([1, 3, 0])
    stats = assay._streaming_feature_stats(cell_idx, feat_idx)
    assert set(stats) == {"normed_tot", "normed_n", "sigmas"}
    assert stats["normed_tot"].shape == (3,)

    del root["RNA/countsT"]
    with pytest.raises(ValueError, match="requires a complete strip-sharded"):
        RNAassay(root, "RNA", cells, workspace=None, nthreads=1)


def test_rna_rejects_unsharded_counts_t():
    root = _memory_root()
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    del root["RNA/countsT"]
    root["RNA"].create_array("countsT", data=values.T, chunks=(1, 2))
    root["RNA/countsT"].attrs["complete"] = True
    with pytest.raises(ValueError, match="unsupported countsT layout"):
        RNAassay(root, "RNA", cells, workspace=None, nthreads=1)


def test_marker_results_on_strip_counts_t():
    root = _memory_root()
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    cells.insert("cluster", np.array(["a", "a", "b", "b"]), overwrite=True)
    assay = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    assay.sf = 1000.0
    results = find_markers_by_rank(
        assay,
        group_key="cluster",
        cell_key="I",
        feat_key="I",
        batch_size=2,
        n_threads=1,
    )
    assert set(results) == {"a", "b"}
    assert len(results["a"]) == 4


def test_iter_normed_feature_wise_on_strip_counts_t():
    root = _memory_root()
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    assay.sf = 1000.0
    batches = list(
        assay.iter_normed_feature_wise(
            cell_key="I",
            feat_key="I",
            batch_size=2,
            msg=None,
            as_dataframe=True,
        )
    )
    assert batches
    joined = np.concatenate([batch.to_numpy() for batch in batches], axis=1)
    assert joined.shape[0] == 4


def test_regression_on_strip_counts_t():
    root = _memory_root()
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    assay.sf = 1000.0
    regressor = np.linspace(0.0, 1.0, values.shape[0])
    table = find_markers_by_regression(
        assay,
        cell_key="I",
        feat_key="I",
        regressor=regressor,
        min_cells=1,
        batch_size=2,
    )
    assert len(table) == 4


def test_iter_normed_feature_wise_uses_base_path_when_ineligible(monkeypatch):
    root = _memory_root()
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    assay.sf = 1000.0
    called = {"base": False}

    def fake_base(self, *args, **kwargs):
        called["base"] = True
        if False:
            yield None

    monkeypatch.setattr(Assay, "iter_normed_feature_wise", fake_base)
    list(
        assay.iter_normed_feature_wise(
            cell_key="I",
            feat_key="I",
            batch_size=2,
            msg=None,
            renormalize_subset=True,
        )
    )
    assert called["base"] is True


def test_renormalize_subset_path_batches_features():
    root = _memory_root()
    values = np.arange(1, 29, dtype=np.uint32).reshape(4, 7)
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    assay = RNAassay(
        root,
        "RNA",
        cells,
        workspace=None,
        nthreads=1,
        min_cells_per_feature=1,
    )
    assay.sf = 1000.0
    frames = list(
        assay.iter_normed_feature_wise(
            "I",
            "I",
            4,
            None,
            renormalize_subset=True,
        )
    )
    assert sum(frame.shape[1] for frame in frames) == 7


def test_coordinate_melding_leaves_counts_t_on_demand():
    root = _memory_root()
    values = np.array(
        [
            [1, 0, 2, 3],
            [0, 4, 0, 0],
            [0, 0, 5, 0],
        ],
        dtype=np.uint32,
    )
    n_cells, n_feats = values.shape
    create_cell_data(
        root,
        None,
        ids=np.array([f"c{i}" for i in range(n_cells)]),
        names=np.array([f"c{i}" for i in range(n_cells)]),
    )
    peak_ids = np.array(
        ["chr1:100-200", "chr1:150-250", "chr1:400-500", "chr2:100-200"]
    )
    create_zarr_count_assay(
        z=root,
        assay_name="ATAC",
        workspace=None,
        n_cells=n_cells,
        feat_ids=peak_ids,
        feat_names=peak_ids,
        dtype="uint32",
    )
    root["ATAC/counts"][:] = values
    cells = MetaData(root["cellData"])
    n_features = (values > 0).sum(axis=1).astype(np.float64)
    n_counts = values.sum(axis=1, dtype=np.float64)
    n_cells_per_peak = (values > 0).sum(axis=0).astype(np.float64)
    cells.insert("ATAC_nFeatures", n_features, overwrite=True)
    cells.insert("ATAC_nCounts", n_counts, overwrite=True)
    assay = Assay(root, None, "ATAC", cells, nthreads=1, min_cells_per_feature=1)
    assay.feats.insert("nCells", n_cells_per_peak, overwrite=True)

    feature_bed = pd.DataFrame(
        {
            0: ["chr1", "chr1", "chr2"],
            1: [120, 300, 150],
            2: [160, 350, 180],
            3: ["a", "b", "c"],
            4: ["A", "B", "C"],
        }
    )
    coordinate_melding(
        assay,
        workspace=None,
        feature_bed=feature_bed,
        new_assay_name="GENE",
        peaks_col="ids",
        renormalization=False,
    )
    assert "counts" in root["GENE"]
    assert "countsT" not in root["GENE"]


def test_count_assay_leaves_counts_t_on_demand():
    root = _memory_root()
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    n_cells, n_feats = values.shape
    create_cell_data(
        root,
        None,
        ids=np.array([f"c{i}" for i in range(n_cells)]),
        names=np.array([f"c{i}" for i in range(n_cells)]),
    )
    create_zarr_count_assay(
        z=root,
        assay_name="PTIME_MODULES",
        workspace=None,
        n_cells=n_cells,
        feat_ids=np.array([f"group_{i}" for i in range(n_feats)]),
        feat_names=np.array([f"group_{i}" for i in range(n_feats)]),
        dtype="float",
    )
    root["PTIME_MODULES/counts"][:] = values
    assert "countsT" not in root["PTIME_MODULES"]


def test_write_doublet_target_zarr_writes_strip_counts_t(tmp_path):
    sim = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.uint32))
    zarr_loc = str(tmp_path / "doublets.zarr")
    root = write_doublet_target_zarr(
        zarr_loc=zarr_loc,
        assay_name="RNA",
        sim_counts=sim,
        feat_ids=np.array(["f0", "f1", "f2"]),
        feat_names=np.array(["g0", "g1", "g2"]),
        dtype="uint32",
    )
    assert "counts" in root["RNA"]
    assert "countsT" in root["RNA"]
    assert root["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(
        root["RNA/countsT"][:],
        np.asarray(root["RNA/counts"][:]).T,
    )


def test_custom_assay_name_seeds_generic_type_and_loads(tmp_path):
    from scarf import DataStore
    from scarf.writers import SparseToZarr

    path = str(tmp_path / "custom.zarr")
    SparseToZarr(
        csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint32)),
        zarr_loc=path,
        cell_ids=["c0", "c1"],
        feature_ids=["f0", "f1"],
        assay_name="CUSTOM_NAME",
    ).dump()
    root = zarr.open_group(path, mode="r")
    assert root.attrs["assayTypes"]["CUSTOM_NAME"] == "Assay"
    assert "countsT" not in root["CUSTOM_NAME"]
    store = DataStore(
        path,
        default_assay="CUSTOM_NAME",
        min_features_per_cell=0,
        min_cells_per_feature=0,
    )
    assert type(store.CUSTOM_NAME).__name__ == "Assay"


def test_explicit_assay_type_can_declare_custom_group_as_rna(tmp_path):
    from scarf.writers import create_cell_data, create_zarr_count_assay
    from scarf.writers.counts_t import finalize_writer_counts_t

    path = str(tmp_path / "declared_rna.zarr")
    root = zarr.open_group(path, mode="w")
    values = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    create_cell_data(
        root,
        None,
        ids=np.array(["c0", "c1"]),
        names=np.array(["c0", "c1"]),
    )
    create_zarr_count_assay(
        root,
        "CUSTOM_NAME",
        None,
        2,
        feat_ids=np.array(["f0", "f1"]),
        feat_names=np.array(["g0", "g1"]),
        dtype="uint32",
    )
    root["CUSTOM_NAME/counts"][:] = values
    finalize_writer_counts_t(root, "CUSTOM_NAME", None, assay_type="RNA")
    assert root.attrs["assayTypes"]["CUSTOM_NAME"] == "RNA"
    assert root["CUSTOM_NAME/countsT"].attrs["complete"] is True


def test_write_doublet_target_rejects_summary_before_truncating_destination():
    store = MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    root.create_group("sentinel")

    with pytest.raises(ValueError, match=r"reserved for DataStore\.summary"):
        write_doublet_target_zarr(
            zarr_loc=store,
            assay_name="summary",
            sim_counts=csr_matrix(np.ones((1, 1), dtype=np.uint32)),
            feat_ids=np.array(["f0"]),
            feat_names=np.array(["g0"]),
        )

    preserved = zarr.open_group(store=store, mode="r")
    assert set(preserved.group_keys()) == {"sentinel"}


def _counts_array(
    values: np.ndarray,
    *,
    chunks: tuple[int, int] = (3, 2),
    shards: tuple[int, int] | None = None,
    store: MemoryStore | None = None,
) -> tuple[zarr.Group, zarr.Array]:
    root = zarr.open_group(
        store=store if store is not None else MemoryStore(), mode="w"
    )
    group = root.create_group("RNA")
    counts = group.create_array(
        "counts",
        shape=values.shape,
        chunks=chunks,
        shards=shards,
        dtype=values.dtype,
        fill_value=0,
    )
    counts[:] = values
    return group, counts


def _dense_values(n_cells: int, n_feats: int) -> np.ndarray:
    return (np.arange(n_cells * n_feats, dtype=np.uint32) + 1).reshape(n_cells, n_feats)


@pytest.mark.parametrize("workers", [1, 4])
def test_write_counts_t_transposes_exactly(workers):
    values = _dense_values(24, 8)
    group, counts = _counts_array(values)
    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, workers),
    )
    assert counts_t is not None
    assert counts_t.attrs["complete"] is True
    np.testing.assert_array_equal(counts_t[:], values.T)


def test_write_counts_t_geometry_matches_serial_baseline():
    from scarf.storage.sharding import choose_strip_layout
    from scarf.storage.types import array_metadata_shards

    values = _dense_values(22, 7)
    metadata = []
    for workers in (1, 4):
        group, counts = _counts_array(values)
        counts_t = write_counts_t(
            counts,
            group,
            resources=ResourceBudget(8 * 1024**3, workers),
        )
        np.testing.assert_array_equal(counts_t[:], values.T)
        metadata.append(
            (
                tuple(counts_t.shape),
                tuple(counts_t.chunks),
                array_metadata_shards(counts_t),
            )
        )

    assert metadata[0] == metadata[1]
    expected = choose_strip_layout(7, 22, values.dtype.itemsize)
    assert metadata[0][0] == (7, 22)
    assert metadata[0][1] == (expected["geneStrip"], expected["cellChunk"])
    assert metadata[0][2] == (expected["geneStrip"], expected["shardCells"])


def test_write_counts_t_covers_edge_strips():
    values = _dense_values(22, 7)
    group, counts = _counts_array(values)
    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    np.testing.assert_array_equal(counts_t[:], values.T)
    np.testing.assert_array_equal(counts_t[0:1, :], values[:, 0:1].T)
    np.testing.assert_array_equal(counts_t[6:7, :], values[:, 6:7].T)


def test_write_counts_t_writes_each_strip_once():
    store = RecordingStore()
    values = _dense_values(22, 7)
    group, counts = _counts_array(values, store=store)
    store.reset()
    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    chunk_ops = store.chunk_ops("RNA/countsT/c/")
    assert {kind for kind, _ in chunk_ops} == {"set"}
    keys = [key for _, key in chunk_ops]
    assert len(keys) == len(set(keys))
    gene_strip = int(counts_t.chunks[0])
    n_strips = (7 + gene_strip - 1) // gene_strip
    assert len(keys) == n_strips
    np.testing.assert_array_equal(counts_t[:], values.T)


def test_counts_t_write_plan_grows_with_memory():
    from scarf.storage.sharding import COUNTS_T_MAX_UPLOAD_WORKERS, plan_counts_t_write

    # Force many small strips so concurrency is the admission variable.
    tight = plan_counts_t_write(
        nFeats=2000,
        nCells=100_000,
        dtype=np.uint16,
        sourceCellRows=1000,
        resources=ResourceBudget(80 * 1024**2, 2),
        residentBytes=0,
        profile="fast_local",
        maxShardBytes=100 * 100_000 * 2,
    )
    roomy = plan_counts_t_write(
        nFeats=2000,
        nCells=100_000,
        dtype=np.uint16,
        sourceCellRows=1000,
        resources=ResourceBudget(320 * 1024**2, 4),
        residentBytes=0,
        profile="fast_local",
        maxShardBytes=100 * 100_000 * 2,
    )
    assert tight.nStrips == 20
    assert tight.stripsPerBatch >= 1
    assert tight.stripsPerBatch <= COUNTS_T_MAX_UPLOAD_WORKERS
    assert roomy.stripsPerBatch <= COUNTS_T_MAX_UPLOAD_WORKERS
    assert roomy.stripsPerBatch <= roomy.nStrips
    assert roomy.uploadWorkers >= tight.uploadWorkers
    assert roomy.uploadWorkers >= roomy.stripsPerBatch


def test_counts_t_write_plan_caps_in_memory_strips_and_prefers_uploads():
    from scarf.storage.sharding import COUNTS_T_MAX_UPLOAD_WORKERS, plan_counts_t_write

    plan = plan_counts_t_write(
        nFeats=45_000,
        nCells=5_000_000,
        dtype=np.uint16,
        sourceCellRows=7370,
        resources=ResourceBudget(48 * 1024**3, 16),
        residentBytes=0,
        profile="fast_local",
        maxShardBytes=1024**3,
    )
    assert plan.nStrips > COUNTS_T_MAX_UPLOAD_WORKERS
    assert plan.stripsPerBatch <= COUNTS_T_MAX_UPLOAD_WORKERS
    assert plan.uploadWorkers == COUNTS_T_MAX_UPLOAD_WORKERS
    assert plan.uploadWorkers == plan.stripsPerBatch
    assert plan.readWorkers >= plan.uploadWorkers
    # Four ~1 GiB strips plus encode scratch; previously this shape admitted ~45 strips.
    assert plan.peakBytes < 16 * 1024**3
    assert plan.peakBytes < 48 * 1024**3 // 2


def test_counts_t_write_plan_rejects_when_one_strip_cannot_fit():
    from scarf.storage.sharding import plan_counts_t_write

    with pytest.raises(MemoryError, match="countsT strip write needs"):
        plan_counts_t_write(
            nFeats=8,
            nCells=1_000_000,
            dtype=np.uint16,
            sourceCellRows=1000,
            resources=ResourceBudget(8 * 1024**2, 2),
            residentBytes=0,
            profile="fast_local",
            maxShardBytes=2 * 1_000_000 * 2,
        )


def test_choose_strip_layout_pads_awkward_cell_counts():
    from scarf.storage.sharding import choose_strip_layout, is_strip_counts_t_layout

    layout = choose_strip_layout(30_000, 1_000_003, 4)
    assert layout["cellChunk"] > 1
    assert layout["shardCells"] >= 1_000_003
    assert layout["shardCells"] % layout["cellChunk"] == 0
    assert is_strip_counts_t_layout(
        shape=(30_000, 1_000_003),
        chunks=(layout["geneStrip"], layout["cellChunk"]),
        shards=(layout["geneStrip"], layout["shardCells"]),
        dtype=np.uint32,
    )


def test_is_strip_counts_t_layout_accepts_structural_variants():
    from scarf.storage.sharding import is_strip_counts_t_layout

    assert is_strip_counts_t_layout(
        shape=(100, 1000),
        chunks=(10, 128),
        shards=(10, 1024),
        dtype=np.uint16,
    )
    assert not is_strip_counts_t_layout(
        shape=(100, 1000),
        chunks=(10, 128),
        shards=(20, 1024),
        dtype=np.uint16,
    )
    assert not is_strip_counts_t_layout(
        shape=(100, 1000),
        chunks=(10, 128),
        shards=(10, 512),
        dtype=np.uint16,
    )


def test_write_counts_t_uploads_disjoint_strips_concurrently(monkeypatch):
    import threading

    import zarr

    import scarf.storage.sharding as sharding

    original = sharding.choose_strip_layout

    def tiny_strips(n_feats, n_cells, itemsize, **kwargs):
        return original(
            n_feats,
            n_cells,
            itemsize,
            maxShardBytes=64,
            targetChunkBytes=32,
        )

    monkeypatch.setattr(sharding, "choose_strip_layout", tiny_strips)
    store = RecordingStore()
    values = _dense_values(32, 16)
    group, counts = _counts_array(values, store=store)

    active = 0
    max_active = 0
    lock = threading.Lock()
    release = threading.Event()
    real_setitem = zarr.Array.__setitem__

    def counted_setitem(self, selection, value):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active >= 2:
                release.set()
        if not release.wait(timeout=5):
            raise TimeoutError("concurrent strip uploads did not overlap")
        try:
            return real_setitem(self, selection, value)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(zarr.Array, "__setitem__", counted_setitem)
    store.reset()
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    assert max_active >= 2
    np.testing.assert_array_equal(group["countsT"][:], values.T)


def test_write_counts_t_overwrite_leaves_no_stale_chunks():
    store = RecordingStore()
    values = _dense_values(22, 7)
    group, counts = _counts_array(values, store=store)
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    stale = sorted(k for k in store._store_dict if k.startswith("RNA/countsT/c/"))

    smaller = values[:6]
    counts.resize(smaller.shape)
    counts[:] = smaller
    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )

    live = sorted(k for k in store._store_dict if k.startswith("RNA/countsT/c/"))
    assert live == sorted(set(live))
    assert len(live) <= len(stale)
    np.testing.assert_array_equal(counts_t[:], smaller.T)


def test_write_counts_t_restores_without_async_concurrency_mutation():
    before = zarr.config.get("async.concurrency")
    values = _dense_values(24, 8)
    group, counts = _counts_array(values)
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    assert zarr.config.get("async.concurrency") == before


def test_write_counts_t_stays_incomplete_when_a_strip_fails(monkeypatch):
    values = _dense_values(24, 8)
    group, counts = _counts_array(values)

    def boom(self, selection):
        raise RuntimeError("boom")

    monkeypatch.setattr(type(counts), "__getitem__", boom)
    with pytest.raises(RuntimeError, match="boom"):
        write_counts_t(
            counts,
            group,
            resources=ResourceBudget(8 * 1024**3, 2),
        )
    assert group["countsT"].attrs.get("complete") is False


def test_repack_rebuilds_complete_counts_t(tmp_path):
    from scarf.tools.repack_zarr import repack_store

    src_path = tmp_path / "src.zarr"
    dst_path = tmp_path / "dst.zarr"
    root = zarr.open_group(str(src_path), mode="w")
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)
    assert root["RNA/countsT"].attrs["complete"] is True

    repack_store(str(src_path), str(dst_path), profile="fast_local")
    dst = zarr.open_group(str(dst_path), mode="r")
    assert "countsT" in dst["RNA"]
    assert dst["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(dst["RNA/countsT"][:], values.T)


def test_inspect_counts_t_reports_ready_and_missing(tmp_path):
    from scarf.storage.counts_t_contract import inspect_counts_t

    path = tmp_path / "inspect.zarr"
    root = zarr.open_group(str(path), mode="w")
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)
    ready = inspect_counts_t(root, "RNA")
    assert ready.status == "ready"
    assert ready.assayType == "RNA"

    del root["RNA/countsT"]
    missing = inspect_counts_t(root, "RNA")
    assert missing.status == "missing"


def test_inspect_counts_t_reports_shape_dtype_mismatch(tmp_path):
    from scarf.storage.counts_t_contract import inspect_counts_t

    path = tmp_path / "mismatch.zarr"
    root = zarr.open_group(str(path), mode="w")
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)
    del root["RNA/counts"]
    root["RNA"].create_array(
        "counts",
        shape=(5, 4),
        chunks=(5, 4),
        dtype="uint32",
        fill_value=0,
    )
    mismatch = inspect_counts_t(root, "RNA")
    assert mismatch.status == "shape-dtype-mismatch"


def test_inspect_counts_t_reports_dtype_mismatch(tmp_path):
    from scarf.storage.counts_t_contract import inspect_counts_t

    path = tmp_path / "dtype-mismatch.zarr"
    root = zarr.open_group(str(path), mode="w")
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)
    del root["RNA/counts"]
    root["RNA"].create_array(
        "counts",
        shape=(3, 4),
        chunks=(3, 4),
        dtype="uint16",
        fill_value=0,
    )
    mismatch = inspect_counts_t(root, "RNA")
    assert mismatch.status == "shape-dtype-mismatch"


def test_assess_counts_t_reuse_outcomes(tmp_path):
    from scarf.merge.writer import assess_counts_t_reuse

    path = tmp_path / "reuse.zarr"
    root = zarr.open_group(str(path), mode="w")
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    _write_small_assay(root, workspace=None, values=values)

    ok = assess_counts_t_reuse(
        root, "RNA", None, n_cells=3, n_features=4, dtype="uint32"
    )
    assert ok.outcome == "reusable"

    root["RNA/countsT"].attrs["complete"] = False
    incomplete = assess_counts_t_reuse(
        root, "RNA", None, n_cells=3, n_features=4, dtype="uint32"
    )
    assert incomplete.outcome == "incomplete"

    root["RNA/countsT"].attrs["complete"] = True
    blocked = assess_counts_t_reuse(
        root, "RNA", None, n_cells=3, n_features=4, dtype="float32"
    )
    assert blocked.outcome == "block-shape/dtype"

    # Non-strip: rewrite by replacing with an unsharded destination.
    del root["RNA/countsT"]
    counts = root["RNA/counts"]
    unsharded = root["RNA"].create_array(
        "countsT",
        shape=(4, 3),
        chunks=(4, 3),
        dtype=counts.dtype,
        fill_value=0,
    )
    unsharded[:] = np.asarray(counts[:]).T
    unsharded.attrs["complete"] = True
    layout = assess_counts_t_reuse(
        root, "RNA", None, n_cells=3, n_features=4, dtype="uint32"
    )
    assert layout.outcome == "rewrite-layout"


def test_subset_preserves_gene_activity_alias(tmp_path):
    from scarf import DataStore
    from scarf.writers import SparseToZarr
    from scarf.writers.subset import SubsetZarr

    src = str(tmp_path / "src.zarr")
    SparseToZarr(
        csr_matrix(np.array([[1, 0], [0, 2], [3, 4]], dtype=np.uint32)),
        zarr_loc=src,
        cell_ids=["c0", "c1", "c2"],
        feature_ids=["f0", "f1"],
        assay_name="GeneActivity",
    ).dump()
    root = zarr.open_group(src, mode="r+")
    root.attrs["assayTypes"] = {"GeneActivity": "GeneActivity"}
    store = DataStore(
        src,
        default_assay="GeneActivity",
        min_features_per_cell=0,
        min_cells_per_feature=0,
    )
    out = str(tmp_path / "subset.zarr")
    SubsetZarr(
        out,
        [store.GeneActivity],
        cell_idx=np.array([0, 2]),
        overwrite_existing_file=True,
    ).dump()
    subset_root = zarr.open_group(out, mode="r")
    assert subset_root.attrs["assayTypes"]["GeneActivity"] == "GeneActivity"
    assert subset_root["GeneActivity/countsT"].attrs["complete"] is True
