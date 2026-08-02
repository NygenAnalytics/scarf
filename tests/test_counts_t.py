import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay import Assay, RNAassay, lib_size_feature_stream_eligible, norm_dummy
from scarf.features.markers.batching import feature_column_chunk
from scarf.features.genomic.melding import coordinate_melding
from scarf.features.markers import find_markers_by_rank, find_markers_by_regression
from scarf.metadata import MetaData
from scarf.quality_control.doublets import write_doublet_target_zarr
from scarf.storage.budget import ResourceBudget
from scarf.storage.sharding import write_counts_t
from tests.store_probes import RecordingStore
from scarf.writers import (
    create_cell_data,
    create_zarr_count_assay,
)


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


def test_write_counts_t_feature_major_candidate_inherits_chunk_area():
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
        feature_major_layout=True,
    )

    assert counts_t is not None
    assert counts_t.chunks == (1, 22)
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


def test_rna_feature_reads_match_with_and_without_counts_t():
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
    del root["RNA/countsT"]
    counts_t = root["RNA"].create_array(
        "countsT",
        data=values.T,
        chunks=(1, 2),
    )
    counts_t.attrs["complete"] = True
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)

    with_t = RNAassay(root, "RNA", cells, workspace=None, nthreads=1)
    with_t.sf = 1000
    assert with_t.rawDataT is not None

    cell_idx = np.array([0, 2, 3])
    feat_idx = np.array([1, 3, 0])
    stats_with = with_t._streaming_feature_stats(cell_idx, feat_idx)
    blocks_with = [
        raw.copy()
        for _, raw, _, _, _ in with_t.iter_raw_column_blocks(
            cell_idx, feat_idx, batch_size=2
        )
    ]

    root["RNA/countsT"].attrs["complete"] = False
    without_t = RNAassay(root, "RNA", cells, workspace=None, nthreads=1)
    without_t.sf = 1000
    assert without_t.rawDataT is None
    stats_without = without_t._streaming_feature_stats(cell_idx, feat_idx)
    blocks_without = [
        raw.copy()
        for _, raw, _, _, _ in without_t.iter_raw_column_blocks(
            cell_idx, feat_idx, batch_size=2
        )
    ]

    for key in stats_with:
        np.testing.assert_allclose(stats_with[key], stats_without[key], rtol=1e-10)
    assert len(blocks_with) == len(blocks_without)
    for left, right in zip(blocks_with, blocks_without, strict=True):
        np.testing.assert_array_equal(left, right)

    assert feature_column_chunk(with_t, n_features=4) == int(with_t.rawDataT.chunks[0])
    assert feature_column_chunk(without_t, n_features=4) == int(
        without_t.rawData._backing.chunks[1]
    )


def test_marker_results_match_with_and_without_counts_t():
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
    del root["RNA/countsT"]
    counts_t = root["RNA"].create_array(
        "countsT",
        data=values.T,
        chunks=(1, 2),
    )
    counts_t.attrs["complete"] = True
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    cells.insert("cluster", np.array(["a", "a", "b", "b"]), overwrite=True)

    with_t = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    with_t.sf = 1000.0
    assert with_t.rawDataT is not None
    results_with = find_markers_by_rank(
        with_t,
        group_key="cluster",
        cell_key="I",
        feat_key="I",
        batch_size=2,
        n_threads=1,
    )

    root["RNA/countsT"].attrs["complete"] = False
    without_t = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    without_t.sf = 1000.0
    assert without_t.rawDataT is None
    results_without = find_markers_by_rank(
        without_t,
        group_key="cluster",
        cell_key="I",
        feat_key="I",
        batch_size=2,
        n_threads=1,
    )

    assert set(results_with) == set(results_without)
    for group in results_with:
        left = results_with[group].sort_index()
        right = results_without[group].sort_index()
        np.testing.assert_allclose(
            left.to_numpy(dtype=np.float64),
            right.to_numpy(dtype=np.float64),
            rtol=1e-10,
            equal_nan=True,
        )


def test_marker_stream_releases_consumed_block_before_reading_next(
    monkeypatch: pytest.MonkeyPatch,
):
    import gc
    import weakref

    import scarf.assay.rna as rna_module

    root = _memory_root()
    values = np.arange(1, 37, dtype=np.uint32).reshape(6, 6)
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    cells.insert("cluster", np.array(["a", "a", "a", "b", "b", "b"]), overwrite=True)
    assay = RNAassay(
        root,
        "RNA",
        cells,
        workspace=None,
        nthreads=1,
        min_cells_per_feature=1,
    )
    assay.sf = 1_000.0

    original_read = rna_module._read_facade_block
    previous: weakref.ReferenceType[np.ndarray] | None = None
    read_count = 0

    def tracked_read(
        array: zarr.Array,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> np.ndarray:
        nonlocal previous, read_count
        gc.collect()
        if previous is not None:
            assert previous() is None
        raw = np.ascontiguousarray(original_read(array, rows, columns))
        previous = weakref.ref(raw)
        read_count += 1
        return raw

    monkeypatch.setattr(rna_module, "_read_facade_block", tracked_read)
    find_markers_by_rank(
        assay,
        group_key="cluster",
        cell_key="I",
        feat_key="I",
        batch_size=2,
        n_threads=1,
    )

    gc.collect()
    assert read_count == 3
    assert previous is not None
    assert previous() is None


def _small_rna_with_ptime(values: np.ndarray) -> tuple[zarr.Group, MetaData]:
    root = _memory_root()
    _write_small_assay(root, workspace=None, values=values)
    cells = MetaData(root["cellData"])
    cells.insert("RNA_nCounts", values.sum(axis=1).astype(np.float64), overwrite=True)
    cells.insert("ptime", np.linspace(0.0, 1.0, values.shape[0]), overwrite=True)
    return root, cells


def test_iter_normed_feature_wise_matches_with_and_without_counts_t():
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    root, cells = _small_rna_with_ptime(values)

    with_t = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    with_t.sf = 1000.0
    assert with_t.rawDataT is not None
    frames_with = list(
        with_t.iter_normed_feature_wise("I", "I", 2, "normed feats", log_transform=True)
    )

    root["RNA/countsT"].attrs["complete"] = False
    without_t = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    without_t.sf = 1000.0
    assert without_t.rawDataT is None
    frames_without = list(
        without_t.iter_normed_feature_wise(
            "I", "I", 2, "normed feats", log_transform=True
        )
    )

    mat_with = pd.concat(frames_with, axis=1)
    mat_without = pd.concat(frames_without, axis=1)
    assert list(mat_with.columns) == list(mat_without.columns)
    np.testing.assert_allclose(
        mat_with.to_numpy(dtype=np.float64),
        mat_without.to_numpy(dtype=np.float64),
        rtol=1e-10,
    )


def test_slow_feature_iterator_never_exceeds_batch_size():
    values = np.arange(1, 29, dtype=np.uint32).reshape(4, 7)
    root, cells = _small_rna_with_ptime(values)
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

    assert [frame.shape[1] for frame in frames] == [4, 3]


def test_regression_and_aggregation_match_with_and_without_counts_t():
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    root, cells = _small_rna_with_ptime(values)
    ptime = cells.fetch("ptime", key="I")

    with_t = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    with_t.sf = 1000.0
    assert with_t.rawDataT is not None
    regression_with = find_markers_by_regression(
        with_t,
        cell_key="I",
        feat_key="I",
        regressor=ptime,
        min_cells=1,
        batch_size=2,
    )
    agg_with, feats_with = with_t.save_aggregated_ordering(
        cell_key="I",
        feat_key="I",
        ordering_key="ptime",
        min_exp=0.0,
        smoothen=False,
        z_scale=False,
        window_size=2,
        chunk_size=2,
        batch_size=2,
    )
    agg_with_vals = np.asarray(agg_with.compute())
    location = "aggregated_I_I_ptime"
    del root["RNA"][location]

    root["RNA/countsT"].attrs["complete"] = False
    without_t = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    without_t.sf = 1000.0
    assert without_t.rawDataT is None
    regression_without = find_markers_by_regression(
        without_t,
        cell_key="I",
        feat_key="I",
        regressor=ptime,
        min_cells=1,
        batch_size=2,
    )
    agg_without, feats_without = without_t.save_aggregated_ordering(
        cell_key="I",
        feat_key="I",
        ordering_key="ptime",
        min_exp=0.0,
        smoothen=False,
        z_scale=False,
        window_size=2,
        chunk_size=2,
        batch_size=2,
    )

    np.testing.assert_allclose(
        regression_with.sort_index().to_numpy(dtype=np.float64),
        regression_without.sort_index().to_numpy(dtype=np.float64),
        rtol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(feats_with, feats_without)
    np.testing.assert_allclose(
        agg_with_vals,
        np.asarray(agg_without.compute()),
        rtol=1e-10,
    )


def test_iter_normed_feature_wise_uses_slow_path_when_ineligible(monkeypatch):
    values = np.array(
        [
            [4, 0, 1, 2],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    root, cells = _small_rna_with_ptime(values)
    assay = RNAassay(
        root, "RNA", cells, workspace=None, nthreads=1, min_cells_per_feature=1
    )
    assay.sf = 1000.0
    assert lib_size_feature_stream_eligible(assay) is True
    assert lib_size_feature_stream_eligible(assay, renormalize_subset=True) is False

    calls = {"raw": 0, "base": 0}
    orig_raw = RNAassay._iter_raw_feature_columns
    orig_base = Assay.iter_normed_feature_wise

    def spy_raw(self, *args, **kwargs):
        calls["raw"] += 1
        yield from orig_raw(self, *args, **kwargs)

    def spy_base(self, *args, **kwargs):
        calls["base"] += 1
        yield from orig_base(self, *args, **kwargs)

    monkeypatch.setattr(RNAassay, "_iter_raw_feature_columns", spy_raw)
    monkeypatch.setattr(Assay, "iter_normed_feature_wise", spy_base)

    list(assay.iter_normed_feature_wise("I", "I", 2, None))
    assert calls["raw"] >= 1
    assert calls["base"] == 0

    calls["raw"] = 0
    calls["base"] = 0
    list(assay.iter_normed_feature_wise("I", "I", 2, None, renormalize_subset=True))
    assert calls["raw"] == 0
    assert calls["base"] >= 1

    calls["raw"] = 0
    calls["base"] = 0
    assay.normMethod = norm_dummy
    assert lib_size_feature_stream_eligible(assay) is False
    list(assay.iter_normed_feature_wise("I", "I", 2, None))
    assert calls["raw"] == 0
    assert calls["base"] >= 1


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

    import pandas as pd

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


def test_write_doublet_target_zarr_skips_counts_t(tmp_path):
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
    assert "countsT" not in root["RNA"]


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
        metadata.append(counts_t.metadata.to_dict())

    assert metadata[0] == metadata[1]
    assert metadata[0]["shape"] == (7, 22)
    assert metadata[0]["chunk_grid"]["configuration"]["chunk_shape"] == (2, 3)
    assert metadata[0]["fill_value"] == 0


def test_write_counts_t_covers_edge_tiles():
    # 7 features over a 2-wide grid and 22 cells over a 6-wide grid leave a
    # partial tile on both axes, including the shared corner tile.
    values = _dense_values(22, 7)
    group, counts = _counts_array(values)
    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    np.testing.assert_array_equal(counts_t[0:2, 0:3], values[0:3, 0:2].T)
    np.testing.assert_array_equal(counts_t[6:7, 0:3], values[0:3, 6:7].T)
    np.testing.assert_array_equal(counts_t[0:2, 21:22], values[21:22, 0:2].T)
    np.testing.assert_array_equal(counts_t[6:7, 21:22], values[21:22, 6:7].T)


def test_write_counts_t_writes_each_chunk_once_without_destination_read():
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
    n_tiles = len(range(0, 7, 2)) * len(range(0, 22, 3))
    assert len(keys) == n_tiles
    np.testing.assert_array_equal(counts_t[:], values.T)


def test_write_counts_t_reads_each_source_inner_chunk_once():
    store = RecordingStore()
    values = _dense_values(24, 8)
    group, counts = _counts_array(
        values,
        chunks=(3, 2),
        shards=(6, 8),
        store=store,
    )
    store.reset()

    counts_t = write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )

    source_gets = [
        (key, byte_range)
        for kind, key, byte_range in store.byte_ranges
        if kind == "get" and key.startswith("RNA/counts/c/")
    ]
    shard_rows, shard_columns = counts.metadata.shards
    expected_shard_keys = {
        f"RNA/counts/c/{row}/{column}"
        for row in range((counts.shape[0] + shard_rows - 1) // shard_rows)
        for column in range((counts.shape[1] + shard_columns - 1) // shard_columns)
    }
    assert {key for key, _ in source_gets} == expected_shard_keys
    assert all(byte_range is not None for _, byte_range in source_gets)
    index_reads = [
        (key, byte_range.suffix)
        for key, byte_range in source_gets
        if hasattr(byte_range, "suffix")
    ]
    chunk_reads = [
        (key, byte_range.start, byte_range.end)
        for key, byte_range in source_gets
        if hasattr(byte_range, "start")
    ]
    n_tasks = ((counts.shape[0] + shard_rows - 1) // shard_rows) * (
        (counts.shape[1] + counts.chunks[1] - 1) // counts.chunks[1]
    )
    chunks_per_task = (shard_rows + counts.chunks[0] - 1) // counts.chunks[0]
    assert len(index_reads) == n_tasks
    assert len(chunk_reads) == len(set(chunk_reads))
    assert len(chunk_reads) == n_tasks * chunks_per_task
    np.testing.assert_array_equal(counts_t[:], values.T)


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
    assert len(live) < len(stale)
    np.testing.assert_array_equal(counts_t[:], smaller.T)


def test_write_counts_t_overlaps_tiles_within_worker_budget():
    store = RecordingStore(delay=0.01)
    values = _dense_values(24, 8)
    group, counts = _counts_array(values, store=store)
    store.reset()
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    assert store.max_in_flight_for("set") > 1
    assert store.max_in_flight_for("set") <= 4


def test_write_counts_t_keeps_concurrency_within_worker_budget():
    store = RecordingStore(delay=0.005)
    values = _dense_values(24, 8)
    group, counts = _counts_array(values, store=store)
    store.reset()
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 2),
    )
    assert store.max_in_flight_for("set") <= 2


def test_write_counts_t_restores_zarr_concurrency():
    before = zarr.config.get("async.concurrency")
    values = _dense_values(24, 8)

    group, counts = _counts_array(values)
    write_counts_t(
        counts,
        group,
        resources=ResourceBudget(8 * 1024**3, 4),
    )
    assert zarr.config.get("async.concurrency") == before

    store = RecordingStore(fail_on="RNA/countsT/c/1/1")
    group, counts = _counts_array(values, store=store)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_counts_t(
            counts,
            group,
            resources=ResourceBudget(8 * 1024**3, 4),
        )
    assert zarr.config.get("async.concurrency") == before


def test_write_counts_t_stays_incomplete_when_a_tile_fails():
    store = RecordingStore(fail_on="RNA/countsT/c/1/1")
    values = _dense_values(24, 8)
    group, counts = _counts_array(values, store=store)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_counts_t(
            counts,
            group,
            resources=ResourceBudget(8 * 1024**3, 4),
        )
    assert group["countsT"].attrs["complete"] is False


def test_write_counts_t_rejects_oversized_tile():
    values = _dense_values(24, 8)
    group, counts = _counts_array(values)
    with pytest.raises(MemoryError):
        write_counts_t(
            counts,
            group,
            resources=ResourceBudget(100, 8),
        )


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
