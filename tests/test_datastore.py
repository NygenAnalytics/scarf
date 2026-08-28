from importlib import import_module, util

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score
from zarr.storage import MemoryStore

import scarf
import scarf.plotting as splt
from scarf.assay import Assay
from scarf.datastore.datastore import DataStore
from scarf.datastore.mapping_datastore import MappingDatastore
from scarf.metadata import MetaData
from scarf.metadata.artifacts import artifact_values
from scarf.storage.artifacts import ArtifactRef, artifact_group
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.trajectory.results import (
    PseudotimeAggregationResult,
    PseudotimeMarkerResult,
    PseudotimeScoreResult,
)
from scarf.writers import create_cell_data, create_zarr_count_assay
from tests.fixtures_datastore import build_neighbourhood_graph
from tests.store_probes import RecordingStore

from . import full_path


_QC_VALUES = np.array(
    [
        [5, 0, 1, 0, 0, 2],
        [0, 3, 0, 4, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [2, 1, 3, 1, 0, 0],
    ],
    dtype=np.uint32,
)
_QC_FEATURE_NAMES = np.array(["MT-CO1", "RPS3", "GENE_A", "RPL5", "ZERO", "GENE_B"])


def _qc_store() -> tuple[RecordingStore, int]:
    store = RecordingStore()
    root = zarr.open_group(store=store, mode="w")
    n_cells, n_features = _QC_VALUES.shape
    create_cell_data(
        root,
        None,
        ids=np.array([f"c{i}" for i in range(n_cells)]),
        names=np.array([f"c{i}" for i in range(n_cells)]),
        profile="fast_local",
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        n_cells,
        feat_ids=np.array([f"f{i}" for i in range(n_features)]),
        feat_names=_QC_FEATURE_NAMES,
        dtype="uint32",
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=48, chunkBytes=16),
    )
    counts[:] = _QC_VALUES
    assert counts.shards is not None
    from scarf.writers.counts_t import finalize_writer_counts_t

    finalize_writer_counts_t(root, "RNA", None, profile="fast_local")
    expected_reads = int(np.ceil(n_cells / counts.shards[0]))
    store.reset()
    return store, expected_reads


def _open_qc_store(store: RecordingStore, **overrides) -> DataStore:
    options = {
        "default_assay": "RNA",
        "min_features_per_cell": 0,
        "mito_pattern": "^MT-",
        "ribo_pattern": "^(RPS|RPL)",
        "nthreads": 1,
        "zarrProfile": "fast_local",
    }
    options.update(overrides)
    return DataStore(store, **options)


def _count_chunk_gets(store: RecordingStore) -> list[str]:
    return [
        key for operation, key in store.chunk_ops("RNA/counts/c/") if operation == "get"
    ]


def _assert_one_counts_stream(store: RecordingStore, expected_reads: int) -> None:
    gets = _count_chunk_gets(store)
    assert len(gets) == expected_reads
    assert len(set(gets)) == expected_reads


def test_initialization_fuses_qc_stats_in_one_counts_stream():
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(store)
    report = datastore.last_execution_report

    assert report is not None
    assert report.unitKind == "initializationRowBand"
    assert report.actualReadWorkers == min(expected_reads, report.plan.readWorkers)
    assert report.actualComputeWorkers == 1
    assert report.extra["effectiveChunkReadsInFlight"] == (
        report.actualReadWorkers * report.plan.ioConcurrency
    )
    assert report.plan.reservedBytes <= datastore.memoryBytes
    assert report.unitsCompleted == expected_reads
    assert report.fetchSeconds >= 0
    assert report.computeSeconds >= 0
    assert report.extra["fusedReadCompute"] is True

    expected_n_counts = _QC_VALUES.sum(axis=1).astype(np.float64)
    expected_n_features = (_QC_VALUES > 0).sum(axis=1).astype(np.float64)
    expected_n_cells = (_QC_VALUES > 0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_mito = 100 * _QC_VALUES[:, 0] / expected_n_counts
        expected_ribo = 100 * _QC_VALUES[:, [1, 3]].sum(axis=1) / expected_n_counts

    np.testing.assert_array_equal(
        datastore.cells.fetch_all("RNA_nCounts"),
        expected_n_counts,
    )
    np.testing.assert_array_equal(
        datastore.cells.fetch_all("RNA_nFeatures"),
        expected_n_features,
    )
    np.testing.assert_allclose(
        datastore.cells.fetch_all("RNA_percentMito"),
        expected_mito,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        datastore.cells.fetch_all("RNA_percentRibo"),
        expected_ribo,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        datastore.RNA.feats.fetch_all("nCells"),
        expected_n_cells,
    )
    np.testing.assert_array_equal(
        datastore.RNA.feats.fetch_all("dropOuts"),
        _QC_VALUES.shape[0] - expected_n_cells,
    )
    np.testing.assert_array_equal(
        datastore.RNA.feats.fetch_all("I"),
        np.ones(_QC_VALUES.shape[1], dtype=bool),
    )
    _assert_one_counts_stream(store, expected_reads)

    for column in (
        "RNA_nCounts",
        "RNA_nFeatures",
        "RNA_percentMito",
        "RNA_percentRibo",
    ):
        assert "source_artifact" not in datastore.zw["cellData"][column].attrs
    for column in ("nCells", "dropOuts"):
        assert "source_artifact" not in datastore.RNA.z["featureData"][column].attrs


def test_initialization_concurrency_respects_a_tight_memory_budget():
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(
        store,
        mem_budget=300,
        nthreads=4,
    )
    report = datastore.last_execution_report

    assert report is not None
    assert report.unitKind == "initializationRowBand"
    assert report.plan.reservedBytes <= 300
    assert report.actualReadWorkers == 1
    assert report.actualComputeWorkers == 1
    assert report.unitsCompleted == expected_reads
    _assert_one_counts_stream(store, expected_reads)


def test_cached_initialization_is_read_and_write_free():
    store, _ = _qc_store()
    _open_qc_store(store)
    store.reset()

    _open_qc_store(store, zarr_mode="r")

    assert _count_chunk_gets(store) == []
    assert [operation for operation, _ in store.ops if operation == "set"] == []


def test_partial_initialization_preserves_feature_summary_cache(monkeypatch):
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(store)
    selection = datastore.select_detected_features(
        datastore.snapshot_cell_selection(),
        min_cells=1,
    )
    summary = ArtifactRef.from_dict(
        datastore.inspect_artifact(selection).inputs["feature_summary"]
    )
    feature_index = datastore.RNA.feats.fetch_all("I")
    datastore.cells.drop("RNA_nFeatures")
    store.reset()

    reopened = _open_qc_store(store)

    _assert_one_counts_stream(store, expected_reads)
    assert reopened.inspect_artifact(summary).complete is True
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch_all("I"),
        feature_index,
    )
    monkeypatch.setattr(
        reopened.RNA,
        "_compute_feature_summary",
        lambda *_: pytest.fail("valid feature summary should be reused"),
    )
    assert (
        reopened.select_detected_features(
            reopened.snapshot_cell_selection(),
            min_cells=1,
        )
        == selection
    )


def test_percent_cache_remains_attribute_only():
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(store)
    datastore.cells.drop("RNA_percentMito")
    store.reset()

    cached = _open_qc_store(store)

    assert _count_chunk_gets(store) == []
    assert "RNA_percentMito" not in cached.cells.columns
    store.reset()

    refreshed = _open_qc_store(store, mito_pattern="^MT-|^GENE_A$")

    _assert_one_counts_stream(store, expected_reads)
    assert "RNA_percentMito" in refreshed.cells.columns


def test_partial_feature_props_are_recomputed_together():
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(store)
    datastore.RNA.feats.insert(
        "nCells",
        np.zeros(_QC_VALUES.shape[1], dtype=np.int64),
        overwrite=True,
    )
    datastore.RNA.feats.drop("dropOuts")
    feature_index = datastore.RNA.feats.fetch_all("I")
    store.reset()

    reopened = _open_qc_store(store)

    _assert_one_counts_stream(store, expected_reads)
    expected_n_cells = (_QC_VALUES > 0).sum(axis=0)
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch_all("nCells"),
        expected_n_cells,
    )
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch_all("I"),
        feature_index,
    )
    expected_dropouts = _QC_VALUES.shape[0] - expected_n_cells[feature_index]
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch("dropOuts"),
        expected_dropouts,
    )


def test_missing_feature_props_do_not_read_or_modify_legacy_feature_i():
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(store)
    legacy_mask = np.array([True, False, True, False, False, True])
    feature_i = datastore.RNA.z["featureData/I"]
    feature_i[:] = legacy_mask
    feature_i.attrs["legacy_marker"] = {"preserve": True}
    original_attrs = dict(feature_i.attrs)
    datastore.RNA.feats.drop("nCells")
    datastore.RNA.feats.drop("dropOuts")
    store.reset()

    reopened = _open_qc_store(store)

    _assert_one_counts_stream(store, expected_reads)
    feature_i_chunk_ops = store.chunk_ops("RNA/featureData/I/c/")
    assert feature_i_chunk_ops == []
    expected_n_cells = (_QC_VALUES > 0).sum(axis=0)
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch_all("nCells"),
        expected_n_cells,
    )
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch_all("dropOuts"),
        _QC_VALUES.shape[0] - expected_n_cells,
    )
    np.testing.assert_array_equal(reopened.RNA.feats.fetch_all("I"), legacy_mask)
    assert dict(reopened.RNA.z["featureData/I"].attrs) == original_attrs


def test_standalone_assay_keeps_eager_feature_initialization():
    store, _ = _qc_store()
    root = zarr.open_group(store=store, mode="r+")
    assay = Assay(
        root,
        None,
        "RNA",
        MetaData(root["cellData"]),
        nthreads=1,
    )

    assert {"nCells", "dropOuts"}.issubset(assay.feats.columns)
    assert assay._deferred_feature_props is False


class TestToyDataStore:
    def test_toy_crdir_metadata(self, toy_crdir_ds):
        assert np.all(
            toy_crdir_ds.RNA.feats.fetch_all("ids") == ["g1", "g2", "g3", "g4"]
        )
        assert np.all(toy_crdir_ds.ADT.feats.fetch_all("ids") == ["a1", "a2", "a3"])
        assert np.all(toy_crdir_ds.HTO.feats.fetch_all("ids") == ["h1"])
        assert np.all(toy_crdir_ds.cells.fetch_all("ids") == ["b1", "b2", "b3"])

    def test_toy_crdir_rawdata(self, toy_crdir_ds):
        assert np.all(
            toy_crdir_ds.RNA.rawData.compute()
            == [[5, 0, 0, 2], [3, 3, 0, 7], [3, 3, 0, 7]]
        )
        assert np.all(
            toy_crdir_ds.ADT.rawData.compute()
            == [[30, 40, 30], [30, 50, 20], [0, 50, 20]]
        )
        assert np.all(toy_crdir_ds.HTO.rawData.compute() == [[200], [100], [100]])


class TestDataStore:
    def test_init_wrong_zarr_mode(self, tmp_path):
        import pytest
        import tarfile

        from scarf.datastore.datastore import DataStore

        fn = full_path("1K_pbmc_citeseq.zarr.tar.gz")
        out_fn = tmp_path / "1K_pbmc_citeseq.zarr"
        with tarfile.open(fn, "r:gz") as tar:
            tar.extractall(out_fn, filter="data")
        with pytest.raises(ValueError):
            DataStore(str(out_fn), zarr_mode="wrong", default_assay="RNA")

    @pytest.mark.parametrize("zarr_mode", ["r", "r+"])
    def test_init_rejects_legacy_assay_state_without_mutation(
        self,
        zarr_mode,
    ):
        store = MemoryStore()
        root = zarr.open_group(store=store, mode="w")
        assay = root.create_group("RNA")
        assay.attrs["is_assay"] = True
        assay.create_group("featureData")
        assay.create_array("counts", shape=(1, 1), dtype=np.uint32)
        state = assay.create_group("state")
        state.attrs["state"] = {"assay": "RNA", "legacy": True}
        root_attrs = dict(root.attrs)
        state_attrs = dict(state.attrs)

        with pytest.raises(
            ValueError,
            match=r"RNA/state.*never reads or migrates.*rebuild",
        ):
            DataStore(
                store,
                default_assay="RNA",
                min_features_per_cell=0,
                zarr_mode=zarr_mode,
            )

        reopened = zarr.open_group(store=store, mode="r")
        assert dict(reopened.attrs) == root_attrs
        assert dict(reopened["RNA/state"].attrs) == state_attrs

    def test_nthreads_env_and_explicit_precedence(
        self, toy_crdir_writer, tmp_path, monkeypatch
    ):
        import shutil

        from scarf.datastore.datastore import DataStore

        store_path = tmp_path / "nthreads_budget.zarr"
        shutil.copytree(toy_crdir_writer, store_path)
        monkeypatch.setenv("SCARF_WORKERS", "3")
        auto = DataStore(
            str(store_path),
            default_assay="RNA",
            min_features_per_cell=0,
        )
        assert auto.nthreads == 3
        assert auto.resources.workers == 3
        explicit = DataStore(
            str(store_path),
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=2,
        )
        assert explicit.nthreads == 2
        assert explicit.resources.workers == 2

    def test_auto_filter_cells(self, datastore_ephemeral):
        before = np.asarray(datastore_ephemeral.cells.fetch_all("I")).copy()
        ref = datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts", "RNA_nFeatures"],
        )
        assert ref.kind == "cell_selection"
        assert datastore_ephemeral.inspect_artifact(ref).complete
        np.testing.assert_array_equal(
            datastore_ephemeral.cells.fetch_all("I"),
            before,
        )

    def test_auto_filter_cells_does_not_mix_computation_with_plotting(
        self, datastore_ephemeral, monkeypatch
    ):
        before = np.asarray(datastore_ephemeral.cells.fetch_all("I")).copy()
        monkeypatch.setattr(
            splt,
            "distribution",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("filtering must not invoke plotting")
            ),
        )
        ref = datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts", "RNA_nFeatures"],
        )
        assert ref.kind == "cell_selection"
        np.testing.assert_array_equal(datastore_ephemeral.cells.fetch_all("I"), before)

    def test_auto_filter_cells_skips_qc_when_no_attrs(
        self, datastore_ephemeral, monkeypatch
    ):
        monkeypatch.setattr(
            splt,
            "distribution",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("distribution should not be called")
            ),
        )
        ref = datastore_ephemeral.auto_filter_cells(
            attrs=[],
        )
        assert ref.kind == "cell_selection"

    def test_assay_names_tolerates_repeated_group_listings(
        self, datastore_ephemeral, monkeypatch
    ):
        # Object-store listings can yield the same group more than once.
        root = datastore_ephemeral.z
        keys = list(root.group_keys())
        expected = sorted(
            name
            for name in dict.fromkeys(keys)
            if "is_assay" in root[name].attrs.keys()
        )
        assert expected

        class RepeatingGroup:
            def group_keys(self):
                return iter(keys + keys + keys)

            def __getitem__(self, key):
                return root[key]

        monkeypatch.setattr(
            type(datastore_ephemeral),
            "zw",
            property(lambda self: RepeatingGroup()),
        )
        assert datastore_ephemeral.assay_names == expected

    def test_filter_cells(self, datastore_ephemeral):
        before = np.asarray(datastore_ephemeral.cells.fetch_all("I")).copy()
        ref = datastore_ephemeral.filter_cells(
            attrs=["RNA_nCounts", "RNA_nFeatures"],
            lows=[None, None],
            highs=[None, None],
        )
        assert ref.kind == "cell_selection"
        np.testing.assert_array_equal(datastore_ephemeral.cells.fetch_all("I"), before)

    def test_filtering_rejects_explicit_missing_metadata_columns(
        self,
        datastore_ephemeral,
    ):
        before = np.asarray(datastore_ephemeral.cells.fetch_all("I")).copy()
        calls = (
            (
                datastore_ephemeral.filter_cells,
                {
                    "attrs": ["missing_a", "missing_b"],
                    "lows": [None, None],
                    "highs": [None, None],
                },
            ),
            (
                datastore_ephemeral.auto_filter_cells,
                {"attrs": ["missing_a", "missing_b"]},
            ),
        )
        for operation, kwargs in calls:
            with pytest.raises(
                KeyError,
                match="Cell metadata columns not found: 'missing_a', 'missing_b'",
            ):
                operation(**kwargs)

        np.testing.assert_array_equal(datastore_ephemeral.cells.fetch_all("I"), before)

    def test_graph_indices(self, graph_artifacts, datastore):
        expected = np.load(full_path("knn_indices.npy"))
        observed = np.asarray(datastore.z[graph_artifacts]["indices"][:])
        row_ids = np.arange(observed.shape[0])[:, None]
        overlap = np.mean(
            [
                len(set(expected_row) & set(observed_row)) / len(expected_row)
                for expected_row, observed_row in zip(
                    expected,
                    observed,
                    strict=True,
                )
            ]
        )

        assert expected.shape == observed.shape
        assert observed.min() >= 0
        assert observed.max() < observed.shape[0]
        assert not np.any(observed == row_ids)
        assert all(np.unique(row).size == row.size for row in observed)
        assert overlap >= 0.35

    def test_graph_distances(self, graph_artifacts, datastore):
        expected = np.sqrt(np.load(full_path("knn_distances.npy")))
        observed = np.asarray(datastore.z[graph_artifacts]["distances"][:])

        assert expected.shape == observed.shape
        assert np.isfinite(observed).all()
        assert np.all(observed > 0)
        assert np.all(np.diff(observed, axis=1) >= 0)
        relative_error = np.abs(observed - expected) / np.maximum(expected, 1e-12)
        assert np.median(relative_error) < 0.25

    def test_graph_weights(self, graph_artifacts, connectivity_graph, datastore):
        from umap.umap_ import compute_membership_strengths, smooth_knn_dist

        indices = np.asarray(datastore.z[graph_artifacts]["indices"][:])
        distances = np.asarray(
            datastore.z[graph_artifacts]["distances"][:],
            dtype=np.float32,
        )
        sigmas, rhos = smooth_knn_dist(
            distances,
            k=indices.shape[1],
            local_connectivity=1.0,
            bandwidth=1.5,
        )
        _, _, expected, _ = compute_membership_strengths(
            indices,
            distances,
            sigmas,
            rhos,
        )
        a = np.asarray(expected, dtype=np.float32)
        a = a[a > 0]
        graph_path = datastore.inspect_artifact(connectivity_graph).path
        b = datastore.z[graph_path]["weights"][:]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)

    def test_atac_graph_indices(self, make_atac_graph, atac_datastore):
        expected = np.load(full_path("atac_knn_indices.npy"))
        observed = np.asarray(atac_datastore.z[make_atac_graph]["indices"][:])
        row_ids = np.arange(observed.shape[0])[:, None]
        overlap = np.mean(
            [
                len(set(expected_row) & set(observed_row)) / len(expected_row)
                for expected_row, observed_row in zip(
                    expected,
                    observed,
                    strict=True,
                )
            ]
        )

        assert expected.shape == observed.shape
        assert observed.min() >= 0
        assert observed.max() < observed.shape[0]
        assert not np.any(observed == row_ids)
        assert all(np.unique(row).size == row.size for row in observed)
        assert overlap >= 0.15

    def test_atac_graph_distances(self, make_atac_graph, atac_datastore):
        from scipy.stats import spearmanr

        expected = np.sqrt(np.load(full_path("atac_knn_distances.npy")))
        observed = np.asarray(atac_datastore.z[make_atac_graph]["distances"][:])

        assert expected.shape == observed.shape
        assert np.isfinite(observed).all()
        assert np.all(observed > 0)
        assert np.all(np.diff(observed, axis=1) >= 0)
        # Total-count TF changes cell-specific scale, so compare each local
        # distance profile after removing that scale.
        expected_profile = expected / expected.mean(axis=1, keepdims=True)
        observed_profile = observed / observed.mean(axis=1, keepdims=True)
        assert (
            spearmanr(expected_profile.ravel(), observed_profile.ravel()).statistic
            >= 0.8
        )
        relative_mae = np.mean(np.abs(expected_profile - observed_profile)) / np.mean(
            expected_profile
        )
        assert relative_mae < 0.1

    def test_leiden_values(self, leiden_clustering, cell_attrs, datastore):
        labels = artifact_values(
            artifact_group(datastore.zw, leiden_clustering),
            "values",
        )
        unique = np.unique(labels)
        assert np.array_equal(unique, np.arange(1, unique.size + 1))
        assert unique.size >= 2
        expected = cell_attrs["RNA_leiden_cluster"].values
        agreement = adjusted_rand_score(expected, labels)
        assert agreement >= 0.6

    def test_paris_values(self, paris_clustering, datastore):
        labels = artifact_values(
            artifact_group(datastore.zw, paris_clustering),
            "labels",
        ).astype(np.int32)
        assert labels.ndim == 1
        assert np.array_equal(np.unique(labels), np.arange(1, 11))
        assert np.bincount(labels)[1:].min() > 0

    def test_paris_adaptive_values(self, paris_clustering_auto, datastore):
        labels = artifact_values(
            artifact_group(datastore.zw, paris_clustering_auto),
            "labels",
        ).astype(np.int32)
        unique = np.unique(labels)
        assert labels.ndim == 1
        assert np.array_equal(unique, np.arange(1, unique.size + 1))
        assert np.bincount(labels)[1:].min() >= 10

    def test_run_cell_cycle_scoring(self, cell_cycle_scoring, datastore):
        group = artifact_group(datastore.zw, cell_cycle_scoring)
        phase = artifact_values(group, "phase")
        selection = ArtifactRef.from_dict(
            datastore.inspect_artifact(cell_cycle_scoring).inputs["cell_selection"]
        )
        n_selected = int(
            artifact_values(artifact_group(datastore.zw, selection), "values").sum()
        )
        assert phase.shape == (n_selected,)
        assert set(np.unique(phase)) <= {"G1", "S", "G2M"}
        assert {"G1", "S"} <= set(phase)
        status = datastore.inspect_artifact(cell_cycle_scoring)
        assert status.operation == "run_cell_cycle_scoring"
        assert set(status.inputs or {}) == {"feature_summary", "cell_selection"}
        assert np.isfinite(artifact_values(group, "s_score")).all()
        assert np.isfinite(artifact_values(group, "g2m_score")).all()
        assert "RNA_cell_cycle_phase" not in datastore.cells.columns

    def test_umap_values(self, umap, cell_attrs, datastore):
        values = artifact_values(artifact_group(datastore.zw, umap), "values")
        precalc_umap = cell_attrs[["RNA_UMAP1", "RNA_UMAP2"]].values
        assert values.shape == precalc_umap.shape
        _, _, disparity = procrustes(precalc_umap, values)
        assert disparity < 0.25
        assert "RNA_UMAP1" not in datastore.cells.columns

    def test_get_markers(self, marker_search, paris_clustering, datastore):
        markers = datastore.get_markers(
            marker=marker_search,
            group_id=1,
        )

        assert not markers.empty
        assert set(markers.group_id) == {1}
        assert markers.feature_name.is_unique
        assert {"score", "fold_change", "p_value"}.issubset(markers.columns)
        assert np.isfinite(markers.score).all()
        assert np.isfinite(markers.fold_change).all()
        assert markers.p_value.between(0, 1).all()

    def test_get_markers_all_groups(self, marker_search, paris_clustering, datastore):
        all_markers = datastore.get_markers(
            marker=marker_search,
            group_id=None,
        )
        assert "group_id" in all_markers.columns
        groups = {
            str(value)
            for value in artifact_values(
                artifact_group(datastore.zw, paris_clustering),
                "labels",
            )
        }
        assert set(all_markers["group_id"]).issubset(groups)
        one = datastore.get_markers(
            marker=marker_search,
            group_id=1,
        )
        from_all = all_markers[all_markers["group_id"] == "1"].reset_index(drop=True)
        assert len(from_all) == len(one)
        assert from_all["feature_name"].equals(one["feature_name"])

    def test_export_markers_to_csv(
        self, marker_search, paris_clustering, datastore, tmp_path
    ):
        out_file = str(tmp_path / "test_values_markers.csv")
        datastore.export_markers_to_csv(
            marker=marker_search,
            csv_filename=out_file,
        )
        markers = pd.read_csv(out_file)
        groups = sorted(
            str(value)
            for value in np.unique(
                artifact_values(
                    artifact_group(datastore.zw, paris_clustering),
                    "labels",
                )
            )
        )
        assert list(markers.columns) == groups
        for group in groups:
            expected = datastore.get_markers(
                marker=marker_search,
                group_id=group,
            ).feature_name.reset_index(drop=True)
            actual = markers[group].dropna().reset_index(drop=True)
            assert actual.equals(expected)

    def test_repr(self, datastore):
        text = repr(datastore)
        active = datastore.cells.active_index("I").shape[0]
        assert f"DataStore has {active} ({datastore.cells.N}) cells" in text
        for assay_name in datastore.assay_names:
            assert assay_name in text
            assay = datastore._get_assay(assay_name)
            assert f"{assay_name} assay has {assay.feats.N} features" in text
        assert "Cell metadata:" in text
        assert "ids" in text

    def test_get_imputed(self, connectivity_graph, datastore):
        feature_name = "CD4"
        diffusion = datastore.run_diffusion_operator(connectivity_graph, t=2)
        values = datastore.get_imputed(
            feature_name=feature_name,
            diffusion=diffusion,
            from_assay="RNA",
        )
        graph = datastore.load_graph(
            graph=connectivity_graph,
            symmetric=True,
            upper_only=False,
        )
        assert values.shape == (graph.shape[0],)
        assert np.all(np.isfinite(values))

        smoother_diffusion = datastore.run_diffusion_operator(
            connectivity_graph,
            t=4,
        )
        smoother = datastore.get_imputed(
            feature_name=feature_name,
            diffusion=smoother_diffusion,
            from_assay="RNA",
        )
        assert smoother.std() < values.std()

    def test_mean_features(self, datastore):
        import pytest

        names = list(datastore.RNA.feats.fetch("names", key="I")[:3])
        values = datastore.RNA.mean_features(names)
        active = datastore.cells.active_index("I")
        feat_idx = datastore.RNA.feats.get_index_by(names, "names", None)
        expected = (
            datastore.RNA.normed(cell_idx=active, feat_idx=np.sort(feat_idx))
            .mean(axis=1)
            .compute()
        )
        assert values.shape == (len(active),)
        np.testing.assert_allclose(values, expected)
        with pytest.raises(ValueError, match="not found"):
            datastore.RNA.mean_features(["__missing_feature__"])
        skipped = datastore.RNA.mean_features(
            [names[0], "__missing_feature__"],
            missing="skip",
        )
        assert skipped.shape == (len(active),)

    def test_run_doublet_detection(
        self,
        connectivity_graph,
        paris_clustering,
        datastore,
    ):
        columns_before = set(datastore.cells.columns)
        score_ref = datastore.run_doublet_detection(
            paris_clustering,
            connectivity_graph,
            simulation_ratio=0.5,
            random_seed=1,
        )
        assert score_ref.kind == "doublet_score"
        scores = artifact_values(artifact_group(datastore.zw, score_ref), "values")
        n_graph = datastore.load_graph(connectivity_graph).shape[0]
        assert scores.shape == (n_graph,)
        assert not np.isnan(scores).any()
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        assert set(datastore.cells.columns) == columns_before
        assert not datastore.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
        score_status = datastore.inspect_artifact(score_ref)
        neighbors = ArtifactRef.from_dict(score_status.inputs["neighbors"])
        reference_refs = datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
            scope="assay",
            complete_only=True,
        )
        matching = [
            ref
            for ref in reference_refs
            if datastore.get_mapping_reference(ref).neighbors == neighbors
        ]
        assert matching
        reference = datastore.get_mapping_reference(matching[-1])
        assert reference.neighbors == neighbors
        assert reference.symphony_state is None

    def test_run_doublet_detection_bad_cluster_key(
        self,
        connectivity_graph,
        datastore,
    ):
        import pytest

        with pytest.raises(ValueError):
            datastore.run_doublet_detection(
                connectivity_graph,
                connectivity_graph,
            )

    def test_run_pseudotime_scoring(self, pseudotime_scoring, datastore):
        result = datastore.load_pseudotime_scoring(pseudotime_scoring)
        values = result.values[result.valid]
        assert values.ndim == 1
        assert np.isfinite(values).all()
        assert values.min() >= 0
        assert values.max() <= 1
        assert np.ptp(values) > 0.5

    def test_run_pseudotime_scoring_current_contract(
        self,
        datastore_ephemeral,
        monkeypatch,
    ):
        selection = datastore_ephemeral.auto_filter_cells()
        features = datastore_ephemeral.select_hvgs(
            selection,
            from_assay="RNA",
            top_n=100,
            show_plot=False,
            bin_strategy="fixed",
        )
        graph = build_neighbourhood_graph(
            datastore_ephemeral,
            features=features,
            local_cache=False,
        )
        leiden_clustering = datastore_ephemeral.run_leiden_clustering(graph)
        labels = artifact_values(
            artifact_group(datastore_ephemeral.zw, leiden_clustering),
            "values",
        )
        unique = np.unique(labels)
        arguments = {
            "source_sink": leiden_clustering,
            "sources": [int(unique[0])],
            "sinks": [int(unique[-1])],
            "n_singular_vals": 10,
        }
        ref = datastore_ephemeral.run_pseudotime_scoring(
            graph,
            **arguments,
        )

        result = datastore_ephemeral.load_pseudotime_scoring(ref)
        assert isinstance(result, PseudotimeScoreResult)
        assert result.values.shape == result.valid.shape
        np.testing.assert_array_equal(result.valid, np.ones_like(result.valid))
        values = result.values[result.valid]
        assert np.isfinite(values).all()
        assert values.min() >= 0.0
        assert values.max() <= 1.0

        from scarf.datastore._operations import trajectory as trajectory_operations

        def fail_if_recomputed(*_args, **_kwargs):
            raise AssertionError("pseudotime should have been reused")

        monkeypatch.setattr(
            trajectory_operations,
            "_truncated_pba_potential_impl",
            fail_if_recomputed,
        )
        cached = datastore_ephemeral.run_pseudotime_scoring(
            graph,
            **arguments,
        )
        assert cached == ref

    def test_run_pseudotime_marker_search(
        self,
        pseudotime_markers,
        datastore,
    ):
        result = datastore.load_pseudotime_markers(pseudotime_markers)
        assert isinstance(result, PseudotimeMarkerResult)
        feature_mask = artifact_values(
            artifact_group(datastore.zw, result.feature_selection),
            "values",
        ).astype(bool)
        expected_indices = np.flatnonzero(feature_mask)
        np.testing.assert_array_equal(
            result.table["feature_index"].to_numpy(),
            expected_indices,
        )
        assert np.isfinite(result.table["r_value"]).all()
        assert result.table["p_value"].between(0, 1).all()
        assert pseudotime_markers.kind == "pseudotime_markers"

    def test_run_pseudotime_aggregation(
        self,
        pseudotime_aggregation,
        datastore,
        monkeypatch,
    ):
        result = datastore.load_pseudotime_aggregation(pseudotime_aggregation)
        agg_group = artifact_group(datastore.zw, pseudotime_aggregation)
        test_values = agg_group["feature_indices"][:]
        assert isinstance(result, PseudotimeAggregationResult)
        feature_mask = artifact_values(
            artifact_group(datastore.zw, result.feature_selection),
            "values",
        ).astype(bool)
        np.testing.assert_array_equal(
            test_values.astype(np.int64),
            np.flatnonzero(feature_mask),
        )

        assert agg_group.attrs["complete"] is True
        assert "input_fingerprints" in agg_group.attrs
        assert "valid_features" in agg_group
        assert np.isfinite(agg_group["data"][:]).all()
        valid_features = np.asarray(agg_group["valid_features"][:], dtype=bool)
        np.testing.assert_array_equal(
            result.feature_indices,
            test_values[valid_features],
        )
        assert result.data.shape[0] == valid_features.sum()
        clusters = np.asarray(agg_group["cluster_values"][:])
        assigned = clusters[test_values[valid_features].astype(int)]
        assert assigned.min() >= 1
        assert assigned.max() <= 15
        assert len(np.unique(assigned)) == 15
        np.testing.assert_array_equal(
            result.feature_clusters,
            assigned,
        )
        assert np.all(clusters[test_values[~valid_features].astype(int)] == -1)

        def fail_if_recomputed(*_args, **_kwargs):
            raise AssertionError("pseudotime aggregation should have been reused")

        monkeypatch.setattr(
            "scarf.trajectory.feature_dynamics.knn_clustering",
            fail_if_recomputed,
        )
        cached = datastore.run_pseudotime_aggregation(
            result.pseudotime,
            features=result.feature_selection,
            n_clusters=15,
            window_size=50,
            chunk_size=10,
        )
        assert cached == pseudotime_aggregation

    def test_add_grouped_assay(self, grouped_assay, datastore):
        test_values = datastore.get_cell_vals(
            from_assay="PTIME_MODULES", cell_key="I", k="group_1"
        )
        groups = np.asarray(
            artifact_group(datastore.zw, grouped_assay)["cluster_values"][:]
        )
        feature_index = np.where(groups == 1)[0]
        expected = (
            datastore.RNA.normed(
                cell_idx=datastore.cells.active_index("I"),
                feat_idx=feature_index,
            )
            .mean(axis=1)
            .compute()
        )
        assert np.allclose(expected, test_values)

    def test_make_bulk(self, leiden_clustering, datastore):
        df = datastore.make_bulk(leiden_clustering)
        groups = np.unique(
            artifact_values(
                artifact_group(datastore.zw, leiden_clustering),
                "values",
            )
        )
        assert df.shape == (18850, len(groups))
        np.testing.assert_array_equal(df.columns.to_numpy(), groups.astype(str))
        assert np.isfinite(df.values).all()
        assert np.all(df.values >= 0)

    def test_to_anndata(self, datastore):
        from scipy import sparse

        adata = datastore.to_anndata()
        assert sparse.isspmatrix_csr(adata.X)
        assert adata.n_obs == len(datastore.cells.active_index("I"))
        assert adata.n_vars == datastore.RNA.feats.N
        assert list(adata.obs_names) == list(datastore.cells.fetch("ids", key="I"))
        assert list(adata.var_names) == list(datastore.RNA.feats.fetch_all("ids"))
        np.testing.assert_array_equal(
            adata.X.toarray(),
            datastore.RNA.to_raw_sparse("I").toarray(),
        )

    def test_run_topacedo_sampler(
        self,
        paris_clustering,
        topacedo_sampler,
        datastore,
    ):
        assert topacedo_sampler.kind == "sampling"
        group = artifact_group(datastore.zw, topacedo_sampler)
        sampled = artifact_values(group, "sampled").astype(bool)
        edges = artifact_values(
            group,
            "edges",
        )
        labels = artifact_values(
            artifact_group(datastore.zw, paris_clustering),
            "labels",
        )
        assert sampled.shape == labels.shape
        assert sampled.any()
        assert edges.ndim == 2 and edges.shape[1] == 2
        assert edges.min() >= 0
        assert edges.max() < len(labels)

    def test_plot_distributions(self, datastore):
        result = splt.distribution(
            datastore,
            keys=["RNA_nCounts", "RNA_nFeatures"],
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert set(result.tables) == {"RNA_nCounts", "RNA_nFeatures"}
        result.close()

    def test_plot_embedding(self, umap, paris_clustering, datastore):
        result = datastore.plots.embedding(
            layout=umap,
            color_by=paris_clustering,
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert result.provenance.notes[:2] == ("embedding", "artifact")
        result.close()

    def test_plot_embedding_raster(self, umap, datastore):
        result = splt.embedding_raster(
            datastore,
            layout=umap,
            color_by="RNA_nCounts",
            pixels=64,
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert result.provenance.renderer == "matplotlib-raster"
        result.close()

    def test_plot_cluster_tree(
        self,
        paris_clustering,
        connectivity_graph,
        datastore,
    ):
        result = splt.cluster_tree(
            datastore,
            graph=connectivity_graph,
            clusters=paris_clustering,
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert "cluster_summary" in result.tables
        result.close()

    def test_plot_marker_heatmap(self, marker_search, datastore):
        result = splt.marker_heatmap(
            datastore,
            marker=marker_search,
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert "matrix" in result.tables
        result.close()

    def test_plot_pseudotime_heatmap(self, pseudotime_aggregation, datastore):
        result = splt.pseudotime_heatmap(
            datastore,
            aggregation=pseudotime_aggregation,
            show_features=["Wsb1", "Rest"],
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert "pseudotime_bins" in result.tables
        result.close()

    def test_legacy_plotting_apis_are_absent(self):
        removed_datastore_methods = (
            "plot_cells_dists",
            "plot_layout",
            "plot_marker_heatmap",
            "plot_cluster_tree",
            "plot_pseudotime_heatmap",
            "plot_unified_layout",
        )
        for method_name in removed_datastore_methods:
            assert not hasattr(DataStore, method_name)
        assert not hasattr(MappingDatastore, "plot_unified_layout")

        assert not hasattr(scarf, "plots")
        assert not hasattr(splt, "_legacy")
        for module_name in ("scarf.plots", "scarf.plotting._legacy"):
            assert util.find_spec(module_name) is None
            with pytest.raises(
                ModuleNotFoundError, match=module_name.replace(".", r"\.")
            ):
                import_module(module_name)

    def test_mark_hvgs_with_atac_assay(self, atac_datastore):
        with pytest.raises(TypeError):
            atac_datastore.select_hvgs(
                atac_datastore.snapshot_cell_selection(),
                show_plot=False,
            )

    def test_mark_hvgs_default_max_cells_excludes_ubiquitous(self, datastore_ephemeral):
        datastore = datastore_ephemeral
        n_selected = int(np.asarray(datastore.cells.fetch_all("I"), dtype=bool).sum())
        expected_max = n_selected - 20
        selection = datastore.snapshot_cell_selection()
        default_ref = datastore.select_hvgs(
            selection,
            show_plot=False,
            top_n=10,
        )
        default_parameters = datastore.inspect_artifact(default_ref).parameters
        assert default_parameters["max_cells"] == expected_max
        assert default_parameters["min_cells"] == 20
        assert default_parameters["top_n"] == 10
        assert default_parameters["bin_strategy"] == "adaptive"

        infinite_ref = datastore.select_hvgs(
            selection,
            show_plot=False,
            top_n=10,
            max_cells=np.inf,
        )
        infinite_parameters = datastore.inspect_artifact(infinite_ref).parameters
        assert infinite_parameters["max_cells"] == {"special_float": "inf"}
        assert infinite_parameters["bin_strategy"] == "adaptive"

    def test_adaptive_hvg_stats_reuse_single_matrix_pass(
        self,
        auto_filter_cells,
        datastore,
        monkeypatch,
    ):
        assay = datastore.RNA
        fixed = datastore.select_hvgs(
            auto_filter_cells,
            show_plot=False,
            bin_strategy="fixed",
        )
        fixed_summary = ArtifactRef.from_dict(
            datastore.inspect_artifact(fixed).inputs["feature_summary"]
        )

        def fail_matrix_pass(*args, **kwargs):
            pytest.fail("HVG calculation must reuse the feature-summary artifact")

        monkeypatch.setattr(assay, "_compute_feature_summary", fail_matrix_pass)
        adaptive = datastore.select_hvgs(
            auto_filter_cells,
            show_plot=False,
            bin_strategy="adaptive",
        )
        adaptive_summary = ArtifactRef.from_dict(
            datastore.inspect_artifact(adaptive).inputs["feature_summary"]
        )
        assert adaptive_summary == fixed_summary
        assert not any(
            name.startswith("summary_stats_") for name in assay.z.group_keys()
        )

    def test_mark_prevalent_peaks_with_rna_assay(self, datastore):
        with pytest.raises(TypeError):
            datastore.select_prevalent_peaks(datastore.snapshot_cell_selection())

    def test_mark_prevalent_peaks_links_selection_artifact(
        self,
        mark_prevalent_peaks,
        atac_datastore,
        monkeypatch,
    ):
        ref = mark_prevalent_peaks
        assert ref.kind == "feature_selection"
        status = atac_datastore.inspect_artifact(ref)
        assert status.operation == "select_prevalent_peaks"
        assert status.inputs["feature_summary"]["kind"] == "feature_summary"
        assert status.parameters == {"top_n": 5000}
        refreshed = atac_datastore.select_prevalent_peaks(
            atac_datastore.snapshot_cell_selection(),
            top_n=5000,
        )
        assert refreshed == ref
        columns_before = set(atac_datastore.ATAC.feats.columns)
        monkeypatch.setattr(
            atac_datastore.ATAC,
            "_prevalent_peak_mask",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("interrupted selection computation")
            ),
        )
        with pytest.raises(RuntimeError, match="interrupted selection computation"):
            atac_datastore.select_prevalent_peaks(
                atac_datastore.snapshot_cell_selection(),
                top_n=4000,
            )
        assert set(atac_datastore.ATAC.feats.columns) == columns_before
        assert atac_datastore.inspect_artifact(ref).complete

    def test_run_marker_search_requires_explicit_clusters(self, datastore, mark_hvgs):
        with pytest.raises(TypeError, match="ArtifactRef"):
            datastore.run_marker_search(
                None,  # type: ignore[arg-type]
                features=mark_hvgs,
            )

    def test_run_marker_search_with_explicit_refs(
        self,
        datastore,
        paris_clustering,
        mark_hvgs,
    ):
        ref = datastore.run_marker_search(
            paris_clustering,
            features=mark_hvgs,
        )
        assert ref.kind == "marker_table"

    def test_streaming_feature_stats_matches_three_pass(self, datastore):
        from scarf.utils import controlled_compute

        assay = datastore.RNA
        cell_idx = assay.cells.active_index("I")
        feat_idx = np.arange(assay.feats.N, dtype=np.int64)

        tiled = assay._streaming_feature_stats(cell_idx, feat_idx)

        normed = assay.normed(cell_idx, feat_idx)
        legacy_n = controlled_compute((normed > 0).sum(axis=0), assay.nthreads)
        legacy_tot = controlled_compute(normed.sum(axis=0), assay.nthreads)
        legacy_sigmas = controlled_compute(normed.var(axis=0), assay.nthreads)

        np.testing.assert_allclose(tiled["normed_n"], legacy_n, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            tiled["normed_tot"], legacy_tot, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(tiled["sigmas"], legacy_sigmas, rtol=1e-5, atol=1e-6)

    def test_streaming_feature_stats_match_across_read_widths(self, datastore):
        from scarf.storage.io_policy import StorageIoPolicy

        assay = datastore.RNA
        cell_idx = assay.cells.active_index("I")
        feat_idx = np.arange(assay.feats.N, dtype=np.int64)
        results = []
        original = getattr(assay, "storageIo", None)
        try:
            for width in (2, 8, 32):
                assay.storageIo = StorageIoPolicy(readWorkers=width)
                results.append(assay._streaming_feature_stats(cell_idx, feat_idx))
        finally:
            assay.storageIo = original
        first = results[0]
        for other in results[1:]:
            np.testing.assert_array_equal(first["normed_n"], other["normed_n"])
            np.testing.assert_array_equal(first["normed_tot"], other["normed_tot"])
            np.testing.assert_array_equal(first["sigmas"], other["sigmas"])

    def test_streaming_feature_stats_uses_cell_band_counts_t(
        self, datastore, monkeypatch
    ):
        import scarf.storage.feature_stream as feature_stream

        assay = datastore.RNA
        cell_idx = assay.cells.active_index("I")
        feat_idx = np.arange(assay.feats.N, dtype=np.int64)
        counts_t = assay.rawDataT
        assert counts_t is not None
        from scarf.storage.types import array_metadata_shards

        shards = array_metadata_shards(counts_t)
        assert shards is not None
        expected = int(np.ceil(int(counts_t.shape[0]) / int(counts_t.chunks[0])))
        calls = {"n": 0}
        original = feature_stream.map_feature_cell_bands

        def counted(*args, **kwargs):
            calls["n"] += 1
            yield from original(*args, **kwargs)

        monkeypatch.setattr(feature_stream, "map_feature_cell_bands", counted)
        assay._streaming_feature_stats(cell_idx, feat_idx)
        assert calls["n"] == 1
        assert expected >= 1

    def test_streaming_feature_stats_requires_sf(self, datastore):
        import pytest

        assay = datastore.RNA
        cell_idx = assay.cells.active_index("I")
        feat_idx = np.arange(assay.feats.N, dtype=np.int64)
        original = assay.sf
        try:
            assay.sf = None
            with pytest.raises(ValueError):
                assay._streaming_feature_stats(cell_idx, feat_idx)
        finally:
            assay.sf = original

    def test_read_block_preserves_order_and_selection(self, datastore):
        from scarf.assay import _read_block

        backing = datastore.RNA.rawData._backing

        rows = np.array([3, 4, 5, 6])
        cols = np.array([0, 1, 2])
        contiguous = _read_block(backing, rows, cols)
        reference = np.asarray(backing.get_orthogonal_selection((rows, cols)))
        np.testing.assert_array_equal(contiguous, reference)

        scattered_rows = np.array([7, 2, 9])
        scattered_cols = np.array([4, 0, 2])
        scattered = _read_block(backing, scattered_rows, scattered_cols)
        ref2 = np.asarray(
            backing.get_orthogonal_selection((scattered_rows, scattered_cols))
        )
        np.testing.assert_array_equal(scattered, ref2)
