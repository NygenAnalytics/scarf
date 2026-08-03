from importlib import import_module, util

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score

import scarf
import scarf.plotting as splt
from scarf.assay import Assay
from scarf.datastore.datastore import DataStore
from scarf.datastore.mapping_datastore import MappingDatastore
from scarf.metadata import MetaData
from scarf.storage.artifacts import ArtifactRef
from scarf.trajectory.results import (
    PseudotimeAggregationResult,
    PseudotimeScoreResult,
)
from scarf.utils.arrays import array_digest
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
        targetChunkBytes=16,
        targetShardBytes=48,
    )
    counts[:] = _QC_VALUES
    assert counts.shards is not None
    expected_reads = int(np.ceil(n_cells / counts.shards[0]))
    store.reset()
    return store, expected_reads


def _open_qc_store(store: RecordingStore, **overrides) -> DataStore:
    options = {
        "default_assay": "RNA",
        "min_features_per_cell": 0,
        "min_cells_per_feature": 0,
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
        expected_n_cells > 0,
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


def test_cached_initialization_is_read_and_write_free():
    store, _ = _qc_store()
    _open_qc_store(store)
    store.reset()

    _open_qc_store(store, zarr_mode="r")

    assert _count_chunk_gets(store) == []
    assert [operation for operation, _ in store.ops if operation == "set"] == []


def test_partial_initialization_preserves_hvg_cache():
    store, expected_reads = _qc_store()
    datastore = _open_qc_store(store)
    datastore.RNA.set_feature_stats("I")
    subset_hash = datastore.RNA.z["summary_stats_I"].attrs["subset_hash"]
    feature_index = datastore.RNA.feats.fetch_all("I")
    datastore.cells.drop("RNA_nFeatures")
    store.reset()

    reopened = _open_qc_store(store)

    _assert_one_counts_stream(store, expected_reads)
    assert reopened.RNA.z["summary_stats_I"].attrs["subset_hash"] == subset_hash
    np.testing.assert_array_equal(
        reopened.RNA.feats.fetch_all("I"),
        feature_index,
    )
    reopened.RNA._streaming_feature_stats = lambda *_: pytest.fail(
        "valid feature statistics should be reused"
    )
    reopened.RNA.set_feature_stats("I")


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


def test_standalone_assay_keeps_eager_feature_initialization():
    store, _ = _qc_store()
    root = zarr.open_group(store=store, mode="r+")
    assay = Assay(
        root,
        None,
        "RNA",
        MetaData(root["cellData"]),
        nthreads=1,
        min_cells_per_feature=0,
    )

    assert {"nCells", "dropOuts"}.issubset(assay.feats.columns)
    assert assay._deferred_min_cells_per_feature is None


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

    def test_auto_filter_cells(self, datastore_ephemeral):
        assert (
            datastore_ephemeral.auto_filter_cells(
                attrs=["nCounts", "nFeatures", "non_existing_column"],
                show_qc_plots=False,
            )
            is None
        )

    def test_auto_filter_cells_uses_modern_distribution(
        self, datastore_ephemeral, monkeypatch
    ):
        calls = []
        active_before = int(datastore_ephemeral.cells.fetch_all("I").sum())

        def record_distribution(store, *, keys, cell_key=None, **kwargs):
            if cell_key is None:
                n_cells = store.cells.N
            else:
                n_cells = len(store.cells.active_index(cell_key))
            calls.append(
                {
                    "keys": list(keys),
                    "cell_key": cell_key,
                    "n_cells": n_cells,
                    "active_i": int(store.cells.fetch_all("I").sum()),
                    **kwargs,
                }
            )

        monkeypatch.setattr(splt, "distribution", record_distribution)
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts", "RNA_nFeatures"],
            show_qc_plots=True,
        )

        assert [call["title"] for call in calls] == [
            "Pre-filtering distribution",
            "Post-filtering distribution",
        ]
        assert [call["color"] for call in calls] == ["steelblue", "coral"]
        assert [call["cell_key"] for call in calls] == [None, "I"]
        assert all(call["keys"] == ["RNA_nCounts", "RNA_nFeatures"] for call in calls)
        assert all(call["show"] is True for call in calls)
        # Filtering happens before both plots; pre includes filtered-out cells.
        assert calls[0]["active_i"] < active_before
        assert calls[0]["n_cells"] == datastore_ephemeral.cells.N
        assert calls[1]["n_cells"] == calls[1]["active_i"]
        assert calls[0]["n_cells"] >= calls[1]["n_cells"]

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
        assert (
            datastore_ephemeral.auto_filter_cells(
                attrs=["missing_a", "missing_b"],
                show_qc_plots=True,
            )
            is None
        )

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
        assert (
            datastore_ephemeral.filter_cells(
                attrs=["nCounts", "nFeatures", "non_existing_column"],
                lows=[None, None, None],
                highs=[None, None, None],
                reset_previous=True,
            )
            is None
        )
        # still doesn't access `if j is None:` cases for j and k

    def test_graph_indices(self, graph_artifacts, datastore):
        a = np.load(full_path("knn_indices.npy"))
        b = datastore.z[graph_artifacts]["indices"][:]
        assert np.array_equal(a, b)

    def test_graph_distances(self, graph_artifacts, datastore):
        a = np.sqrt(np.load(full_path("knn_distances.npy")))
        b = datastore.z[graph_artifacts]["distances"][:]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-3)

    def test_graph_weights(self, graph_artifacts, datastore):
        from umap.umap_ import compute_membership_strengths, smooth_knn_dist

        indices = np.load(full_path("knn_indices.npy"))
        distances = np.sqrt(np.load(full_path("knn_distances.npy"))).astype(np.float32)
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
        state = datastore.get_assay_state("RNA")
        assert state is not None and state.connectivity_map is not None
        graph_path = datastore.inspect_artifact(state.connectivity_map).path
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

    def test_leiden_values(self, leiden_clustering, cell_attrs):
        assert len(set(leiden_clustering)) == 10
        expected = cell_attrs["RNA_leiden_cluster"].values
        agreement = adjusted_rand_score(expected, leiden_clustering)
        assert agreement == pytest.approx(0.8850162225, abs=1e-6)

    def test_paris_values(self, paris_clustering):
        labels = np.asarray(paris_clustering, dtype=np.int32)
        assert labels.ndim == 1
        assert np.array_equal(np.unique(labels), np.arange(1, 11))
        assert array_digest(labels) == ("306518457a86d4a4fe0d49a7cfeaeb39")

    def test_paris_adaptive_values(self, paris_clustering_auto):
        labels = np.asarray(paris_clustering_auto, dtype=np.int32)
        unique = np.unique(labels)
        assert labels.ndim == 1
        assert np.array_equal(unique, np.arange(1, unique.size + 1))
        assert np.bincount(labels)[1:].min() >= 10
        assert array_digest(labels) == ("aeb885f85abcb41ec2b970f7fe1aaca2")

    def test_run_cell_cycle_scoring(self, cell_cycle_scoring, cell_attrs):
        assert np.array_equal(
            cell_cycle_scoring, cell_attrs["RNA_cell_cycle_phase"].values
        )

    def test_umap_values(self, umap, cell_attrs):
        precalc_umap = cell_attrs[["RNA_UMAP1", "RNA_UMAP2"]].values
        assert umap.shape == precalc_umap.shape
        _, _, disparity = procrustes(precalc_umap, umap)
        assert disparity < 0.25

    def test_get_markers(self, marker_search, paris_clustering, datastore):
        markers = datastore.get_markers(group_key="RNA_cluster", group_id=1)

        assert not markers.empty
        assert set(markers.group_id) == {1}
        assert markers.feature_name.is_unique
        assert {"score", "fold_change", "p_value"}.issubset(markers.columns)
        assert np.isfinite(markers.score).all()
        assert np.isfinite(markers.fold_change).all()
        assert markers.p_value.between(0, 1).all()

    def test_get_markers_all_groups(self, marker_search, paris_clustering, datastore):
        all_markers = datastore.get_markers(group_key="RNA_cluster", group_id=None)
        assert "group_id" in all_markers.columns
        groups = set(datastore.cells.fetch("RNA_cluster", key="I"))
        assert set(all_markers["group_id"]).issubset(groups)
        one = datastore.get_markers(group_key="RNA_cluster", group_id=1)
        from_all = all_markers[all_markers["group_id"] == 1].reset_index(drop=True)
        assert len(from_all) == len(one)
        assert from_all["feature_name"].equals(one["feature_name"])

    def test_export_markers_to_csv(
        self, marker_search, paris_clustering, datastore, tmp_path
    ):
        out_file = str(tmp_path / "test_values_markers.csv")
        datastore.export_markers_to_csv(group_key="RNA_cluster", csv_filename=out_file)
        markers = pd.read_csv(out_file)
        groups = sorted(np.unique(paris_clustering))
        assert list(markers.columns) == [str(group) for group in groups]
        for group in groups:
            expected = datastore.get_markers(
                group_key="RNA_cluster",
                group_id=int(group),
            ).feature_name.reset_index(drop=True)
            actual = markers[str(group)].dropna().reset_index(drop=True)
            assert actual.equals(expected)

    def test_repr(self, datastore):
        text = repr(datastore)
        active = datastore.cells.active_index("I").shape[0]
        assert f"DataStore has {active} ({datastore.cells.N}) cells" in text
        for assay_name in datastore.assay_names:
            assert assay_name in text
            assay = datastore._get_assay(assay_name)
            assert (
                f"{assay_name} assay has {assay.feats.fetch_all('I').sum()} "
                f"({assay.feats.N}) features"
            ) in text
        assert "Cell metadata:" in text
        assert "ids" in text

    def test_get_imputed(self, graph_artifacts, datastore):
        feature_name = "CD4"
        raw = datastore.get_cell_vals(from_assay="RNA", cell_key="I", k=feature_name)
        values = datastore.get_imputed(
            feature_name=feature_name,
            from_assay="RNA",
            cell_key="I",
            feat_key="hvgs",
            t=2,
            cache_operator=False,
        )
        assert values.shape == raw.shape
        assert np.all(np.isfinite(values))
        assert not np.allclose(values, raw)

        from scarf.neighbors.diffusion import diffusion_operator

        graph = datastore.load_graph(
            from_assay="RNA",
            cell_key="I",
            feat_key="hvgs",
            symmetric=True,
            upper_only=False,
        )
        expected = diffusion_operator(graph, power=2).dot(raw)
        np.testing.assert_allclose(values, expected)

        smoother = datastore.get_imputed(
            feature_name=feature_name,
            from_assay="RNA",
            cell_key="I",
            feat_key="hvgs",
            t=4,
            cache_operator=False,
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
        graph_artifacts,
        paris_clustering,
        datastore,
    ):
        score_col = datastore.run_doublet_detection(
            cluster_key="RNA_cluster", simulation_ratio=0.5, random_seed=1
        )
        col = "RNA_doublet_score"
        assert score_col == col
        assert col in datastore.cells.columns
        scores = datastore.cells.fetch(col)
        n_active = datastore.cells.active_index("I").shape[0]
        assert scores.shape[0] == n_active
        assert not np.isnan(scores).any()
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        raw_ref = datastore.zw["cellData"][col].attrs["source_artifact"]
        assert ArtifactRef.from_dict(raw_ref).kind == "doublet_score"
        # scratch column used for smoothing should be removed
        assert "RNA_doublet_score__raw" not in datastore.cells.columns
        # the temporary query owns and removes its projection
        assert not datastore.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
        state = datastore.get_assay_state("RNA")
        assert state is not None
        assert state.connectivity_map is not None
        reference = datastore.get_mapping_reference()
        assert reference.neighbors == state.neighbors
        assert reference.symphony_state is None

    def test_run_doublet_detection_bad_cluster_key(
        self,
        graph_artifacts,
        datastore,
    ):
        import pytest

        with pytest.raises(ValueError):
            datastore.run_doublet_detection(cluster_key="not_a_column")

    def test_run_pseudotime_scoring(self, pseudotime_scoring, cell_attrs):
        diff = pseudotime_scoring - cell_attrs["RNA_pseudotime"].values
        assert np.all(np.abs(diff) < 0.08)

    def test_run_pseudotime_scoring_current_contract(
        self,
        leiden_clustering,
        datastore_ephemeral,
        monkeypatch,
    ):
        datastore_ephemeral.auto_filter_cells(show_qc_plots=False)
        datastore_ephemeral.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            top_n=100,
            hvg_key_name="hvgs",
            show_plot=False,
            bin_strategy="fixed",
        )
        build_neighbourhood_graph(
            datastore_ephemeral,
            feat_key="hvgs",
            local_cache=False,
        )
        datastore_ephemeral.cells.insert(
            "RNA_leiden_cluster",
            leiden_clustering,
            key="I",
            overwrite=True,
        )
        arguments = {
            "source_sink_key": "RNA_leiden_cluster",
            "sources": [6],
            "sinks": [3],
            "n_singular_vals": 10,
        }
        result = datastore_ephemeral.run_pseudotime_scoring(
            **arguments,
            label="reliability_test",
        )

        output_key = "RNA_reliability_test"
        validity_key = f"{output_key}__valid"
        assert isinstance(result, PseudotimeScoreResult)
        assert result.pseudotime_key == output_key
        assert result.validity_key == validity_key
        assert result.values.shape == result.valid.shape
        np.testing.assert_array_equal(result.valid, np.ones_like(result.valid))
        assert validity_key in datastore_ephemeral.cells.columns
        pseudotime_ref = ArtifactRef.from_dict(
            datastore_ephemeral.zw["cellData"][output_key].attrs["source_artifact"]
        )
        assert pseudotime_ref.kind == "pseudotime"
        assert (
            ArtifactRef.from_dict(
                datastore_ephemeral.zw["cellData"][validity_key].attrs[
                    "source_artifact"
                ]
            )
            == pseudotime_ref
        )
        values = datastore_ephemeral.cells.fetch(output_key, key=validity_key)
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
            **arguments,
            label="reliability_cached",
        )
        cached_ref = ArtifactRef.from_dict(
            datastore_ephemeral.zw["cellData"][cached.pseudotime_key].attrs[
                "source_artifact"
            ]
        )
        assert cached_ref == pseudotime_ref
        np.testing.assert_allclose(cached.values, result.values)

    def test_run_pseudotime_marker_search(
        self,
        pseudotime_markers,
        datastore,
    ):
        precalc_markers = pd.read_csv(
            full_path("pseudotime_markers_r_values.csv"), index_col=0
        )
        assert np.all(precalc_markers.index == pseudotime_markers.index)
        assert np.all(precalc_markers.names.values == pseudotime_markers.names.values)
        assert np.allclose(
            precalc_markers.I__RNA_pseudotime__r.values,
            pseudotime_markers.I__RNA_pseudotime__r.values,
            rtol=0.15,
            atol=0.15,
        )
        marker_ref = ArtifactRef.from_dict(
            datastore.RNA.z["featureData/I__RNA_pseudotime__r"].attrs["source_artifact"]
        )
        assert marker_ref.kind == "pseudotime_markers"
        assert (
            ArtifactRef.from_dict(
                datastore.RNA.z["featureData/I__RNA_pseudotime__p"].attrs[
                    "source_artifact"
                ]
            )
            == marker_ref
        )

    def test_run_pseudotime_aggregation(
        self,
        pseudotime_aggregation,
        datastore,
        monkeypatch,
    ):
        precalc_values = np.load(full_path("aggregated_feat_idx.npy"))
        agg_group = datastore.zw[pseudotime_aggregation.storage_path]
        test_values = agg_group["feature_indices"][:]
        assert isinstance(pseudotime_aggregation, PseudotimeAggregationResult)
        assert (
            ArtifactRef.from_dict(
                datastore.RNA.z["featureData/pseudotime_clusters"].attrs[
                    "source_artifact"
                ]
            ).kind
            == "pseudotime_aggregation"
        )
        assert np.array_equal(
            precalc_values.astype(np.int64), test_values.astype(np.int64)
        )

        assert agg_group.attrs["complete"] is True
        assert "input_fingerprints" in agg_group.attrs
        assert "valid_features" in agg_group
        assert np.isfinite(agg_group["data"][:]).all()
        valid_features = np.asarray(agg_group["valid_features"][:], dtype=bool)
        np.testing.assert_array_equal(
            pseudotime_aggregation.feature_indices,
            test_values[valid_features],
        )
        assert pseudotime_aggregation.data.shape[0] == valid_features.sum()
        clusters = datastore.RNA.feats.fetch_all("pseudotime_clusters")
        assigned = clusters[test_values[valid_features].astype(int)]
        assert assigned.min() >= 1
        assert assigned.max() <= 15
        assert len(np.unique(assigned)) == 15
        np.testing.assert_array_equal(
            pseudotime_aggregation.feature_clusters,
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
            pseudotime_key="RNA_pseudotime",
            cluster_label="pseudotime_clusters_cached",
            n_clusters=15,
            window_size=50,
            chunk_size=10,
        )
        assert cached.storage_path == pseudotime_aggregation.storage_path
        cached_ref = ArtifactRef.from_dict(
            datastore.RNA.z["featureData/pseudotime_clusters_cached"].attrs[
                "source_artifact"
            ]
        )
        assert cached_ref == ArtifactRef.from_dict(
            datastore.RNA.z["featureData/pseudotime_clusters"].attrs["source_artifact"]
        )

    def test_add_grouped_assay(self, grouped_assay, datastore):
        test_values = datastore.get_cell_vals(
            from_assay="PTIME_MODULES", cell_key="I", k="group_1"
        )
        groups = datastore.RNA.feats.fetch_all("pseudotime_clusters")
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
        df = datastore.make_bulk(group_key="RNA_leiden_cluster")
        assert df.shape == (18850, 10)
        assert hash(tuple((df.values.flatten()))) == -3925915741848261436

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

    def test_run_topacedo_sampler(self, paris_clustering, topacedo_sampler):
        assert topacedo_sampler.dtype == bool
        assert topacedo_sampler.shape == paris_clustering.shape
        cluster_sizes = np.bincount(paris_clustering)[1:]
        sampled_sizes = np.bincount(
            paris_clustering[topacedo_sampler],
            minlength=int(paris_clustering.max()) + 1,
        )[1:]
        assert np.all(sampled_sizes >= 3)
        assert np.all(sampled_sizes < cluster_sizes)

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
            layout_key="RNA_UMAP",
            color_by="RNA_cluster",
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert result.provenance.notes[:2] == ("embedding", "materialized")
        result.close()

    def test_plot_embedding_raster(self, umap, datastore):
        result = splt.embedding_raster(
            datastore,
            layout_key="RNA_UMAP",
            color_by="RNA_nCounts",
            pixels=64,
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert result.provenance.renderer == "matplotlib-raster"
        result.close()

    def test_plot_cluster_tree(self, paris_clustering, datastore):
        result = splt.cluster_tree(
            datastore,
            cluster_key="RNA_cluster",
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert "cluster_summary" in result.tables
        result.close()

    def test_plot_marker_heatmap(self, marker_search, datastore):
        result = splt.marker_heatmap(
            datastore,
            group_key="RNA_cluster",
            show=False,
        )
        assert isinstance(result, splt.PlotResult)
        assert "matrix" in result.tables
        result.close()

    def test_plot_pseudotime_heatmap(self, pseudotime_aggregation, datastore):
        result = splt.pseudotime_heatmap(
            datastore,
            cell_key="I",
            feat_key="I",
            feature_cluster_key="pseudotime_clusters",
            pseudotime_key="RNA_pseudotime",
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
        import pytest

        with pytest.raises(TypeError):
            atac_datastore.mark_hvgs()

    def test_mark_hvgs_default_max_cells_excludes_ubiquitous(self, datastore_ephemeral):
        from unittest.mock import patch

        datastore = datastore_ephemeral
        n_selected = int(np.asarray(datastore.cells.fetch_all("I"), dtype=bool).sum())
        expected_max = n_selected - 20
        assay = datastore.RNA

        def _fake_mark_hvgs(*args, **kwargs):
            assay.feats.insert(
                "I__hvgs",
                np.zeros(assay.feats.N, dtype=bool),
                fill_value=False,
                overwrite=True,
            )

        with patch.object(assay, "mark_hvgs", side_effect=_fake_mark_hvgs) as mock:
            datastore.mark_hvgs(show_plot=False, top_n=10)
        assert mock.call_args.kwargs["max_cells"] == expected_max
        assert mock.call_args.kwargs["min_cells"] == 20
        assert mock.call_args.kwargs["top_n"] == 10
        assert mock.call_args.kwargs["bin_strategy"] == "adaptive"

        with patch.object(assay, "mark_hvgs", side_effect=_fake_mark_hvgs) as mock:
            datastore.mark_hvgs(
                show_plot=False,
                top_n=10,
                max_cells=np.inf,
                bin_strategy="adaptive",
            )
        assert mock.call_args.kwargs["max_cells"] == np.inf
        assert mock.call_args.kwargs["bin_strategy"] == "adaptive"

    def test_adaptive_hvg_stats_reuse_single_matrix_pass(
        self,
        auto_filter_cells,
        datastore,
        monkeypatch,
    ):
        import scarf.features.variability as variability

        assay = datastore.RNA
        identifier, fixed_column = assay.set_summary_stats(
            "I",
            bin_strategy="fixed",
        )

        def fail_matrix_pass(*args, **kwargs):
            pytest.fail("adaptive correction must reuse cached feature statistics")

        def fail_metadata_trend(*args, **kwargs):
            pytest.fail("HVG correction must not route through metadata")

        monkeypatch.setattr(assay, "_streaming_feature_stats", fail_matrix_pass)
        monkeypatch.setattr(assay.feats, "remove_trend", fail_metadata_trend)
        adaptive_identifier, adaptive_column = assay.set_summary_stats("I")

        assert adaptive_identifier == identifier
        assert fixed_column == "c_var__200__0.1"
        assert adaptive_column == "c_var__adaptive__200__0.1"
        assert f"{identifier}_{fixed_column}" in assay.feats.columns
        assert f"{identifier}_{adaptive_column}" in assay.feats.columns

        def fail_correction(*args, **kwargs):
            pytest.fail("cached adaptive correction must be reused")

        monkeypatch.setattr(variability, "fit_lowess", fail_correction)
        assert assay.set_summary_stats("I") == (
            identifier,
            adaptive_column,
        )

    def test_mark_prevalent_peaks_with_rna_assay(self, datastore):
        import pytest

        with pytest.raises(TypeError):
            datastore.mark_prevalent_peaks()

    def test_mark_prevalent_peaks_links_selection_artifact(
        self,
        mark_prevalent_peaks,
        atac_datastore,
        monkeypatch,
    ):
        column = atac_datastore.ATAC.z["featureData/I__prevalent_peaks"]
        ref = ArtifactRef.from_dict(column.attrs["source_artifact"])

        assert ref.kind == "feature_selection"
        status = atac_datastore.inspect_artifact(ref)
        assert status.operation == "mark_prevalent_peaks"
        assert status.inputs["feature_selection"]["kind"] == "feature_selection"
        assert "values_fingerprint" not in status.inputs
        atac_datastore.mark_prevalent_peaks(top_n=5000)
        refreshed = ArtifactRef.from_dict(
            atac_datastore.ATAC.z["featureData/I__prevalent_peaks"].attrs[
                "source_artifact"
            ]
        )
        assert refreshed == ref
        before = np.asarray(atac_datastore.ATAC.feats.fetch_all("I__prevalent_peaks"))
        monkeypatch.setattr(
            "scarf.datastore._operations.quality_control."
            "resolve_generated_selection_artifact",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("interrupted artifact write")
            ),
        )
        with pytest.raises(RuntimeError, match="interrupted artifact write"):
            atac_datastore.mark_prevalent_peaks(top_n=4000)
        np.testing.assert_array_equal(
            atac_datastore.ATAC.feats.fetch_all("I__prevalent_peaks"),
            before,
        )
        assert (
            ArtifactRef.from_dict(
                atac_datastore.ATAC.z["featureData/I__prevalent_peaks"].attrs[
                    "source_artifact"
                ]
            )
            == ref
        )

    def test_run_marker_search_with_no_groupkey(self, datastore):
        import pytest

        with pytest.raises(ValueError):
            datastore.run_marker_search(group_key=None)

    def test_run_marker_search_with_cellkey(self, datastore, paris_clustering):
        datastore.run_marker_search(group_key="RNA_cluster", cell_key="I")

    def test_streaming_feature_stats_matches_three_pass(self, datastore):
        from scarf.utils import controlled_compute

        assay = datastore.RNA
        cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")

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

    def test_streaming_feature_stats_one_decode_per_physical_chunk(
        self, datastore, monkeypatch
    ):
        import scarf.assay as assay_mod

        assay = datastore.RNA
        cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")
        zarr_arr = assay.rawData._backing
        row_chunk, col_chunk = zarr_arr.chunks[:2]
        expected = len({int(i) // row_chunk for i in cell_idx}) * len(
            {int(i) // col_chunk for i in feat_idx}
        )
        calls = {"n": 0}
        original = assay_mod._read_block

        def counted(zarr_arr_arg, rows, cols):
            calls["n"] += 1
            return original(zarr_arr_arg, rows, cols)

        monkeypatch.setattr(assay_mod, "_read_block", counted)
        assay._streaming_feature_stats(cell_idx, feat_idx)
        assert calls["n"] == expected

    def test_streaming_feature_stats_requires_sf(self, datastore):
        import pytest

        assay = datastore.RNA
        cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")
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
