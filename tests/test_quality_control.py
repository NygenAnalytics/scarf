from pathlib import Path

import numpy as np
import pytest

from scarf.quality_control.cell_cycle import assign_cell_cycle_phase
from scarf.quality_control.doublets import simulate_doublet_pairs
from scarf.quality_control.filtering import gaussian_quantile_bounds


def test_simulate_doublet_pairs_is_seeded_and_heterotypic():
    clusters = np.array([0, 0, 1, 1])
    left, right = simulate_doublet_pairs(
        clusters,
        n_sim=12,
        heterotypic_fraction=1.0,
        rng=np.random.default_rng(11),
        max_tries=100,
    )

    np.testing.assert_array_equal(left, [0, 0, 3, 1, 2, 2, 2, 0, 1, 0, 1, 3])
    np.testing.assert_array_equal(right, [2, 3, 1, 3, 1, 1, 0, 2, 3, 2, 3, 0])
    assert np.all(clusters[left] != clusters[right])


def test_assign_cell_cycle_phase_preserves_rule_precedence():
    phases = assign_cell_cycle_phase(
        s_score=np.array([1.0, 0.1, -2.0, 0.0, -1.0]),
        g2m_score=np.array([0.5, 0.2, -1.0, 0.0, 0.5]),
    )

    np.testing.assert_array_equal(phases, ["S", "G2M", "G1", "S", "G2M"])


def test_gaussian_quantile_bounds_uses_median_and_population_deviation():
    bounds = gaussian_quantile_bounds(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        min_p=0.1,
        max_p=0.9,
    )

    np.testing.assert_allclose(
        bounds,
        (1.1876123951263535, 4.8123876048736465),
        rtol=0,
        atol=1e-12,
    )


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _insert_doublet_clusters(datastore) -> str:
    column = "doublet_test_clusters"
    datastore.cells.insert(
        column,
        np.arange(datastore.cells.N) % 4,
        overwrite=True,
    )
    return column


def test_doublet_mapping_is_query_owned_and_leaves_reference_unprojected(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    cluster_key = _insert_doublet_clusters(datastore)
    selected_state = datastore.get_assay_state("RNA")
    assert selected_state is not None
    assert selected_state.connectivity_map is not None
    selected_connectivity = selected_state.connectivity_map
    reference_projections = set(
        datastore.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
    )
    temporary_paths: list[Path] = []
    mapping_calls = 0
    original_run_mapping = type(datastore).run_mapping

    def observe_mapping(query, reference, mapping_name, **kwargs):
        nonlocal mapping_calls
        mapping_calls += 1
        assert query is not datastore
        assert mapping_name == "_doublet_sim_RNA"
        assert kwargs == {"query_assay": "RNA", "save_k": 3}
        temporary_path = Path(query.zarr_loc)
        temporary_paths.append(temporary_path)
        assert temporary_path.exists()
        before = _snapshot_store(datastore.zarr_loc)
        result = original_run_mapping(
            query,
            reference,
            mapping_name,
            **kwargs,
        )
        assert _snapshot_store(datastore.zarr_loc) == before
        return result

    monkeypatch.setattr(type(datastore), "run_mapping", observe_mapping)

    score_column = datastore.run_doublet_detection(
        cluster_key=cluster_key,
        cluster_sample_fraction=0.01,
        max_cells_per_cluster=2,
        simulation_ratio=0.01,
        save_k=3,
        smoothing_t=1,
        random_seed=19,
    )

    assert mapping_calls == 1
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert (
        set(
            datastore.list_artifacts(
                kind="projection",
                from_assay="RNA",
            )
        )
        == reference_projections
    )
    scores = datastore.cells.fetch(score_column)
    assert scores.shape == (len(datastore.cells.active_index("I")),)
    assert np.all(np.isfinite(scores))
    assert "RNA_doublet_score__raw" not in datastore.cells.columns

    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map == selected_connectivity
    reference = datastore.get_mapping_reference()
    assert reference.neighbors == state.neighbors
    assert reference.method == "pca"
    assert reference.symphony_state is None

    first_ref = datastore.zw["cellData"][score_column].attrs["source_artifact"]
    assert (
        datastore.run_doublet_detection(
            cluster_key=cluster_key,
            cluster_sample_fraction=0.01,
            max_cells_per_cluster=2,
            simulation_ratio=0.01,
            save_k=3,
            smoothing_t=1,
            random_seed=19,
        )
        == score_column
    )
    assert mapping_calls == 1
    assert datastore.zw["cellData"][score_column].attrs["source_artifact"] == first_ref


def test_doublet_mapping_failure_removes_temporary_query_store(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    cluster_key = _insert_doublet_clusters(datastore)
    temporary_paths: list[Path] = []
    original_run_mapping = type(datastore).run_mapping

    def capture_mapping(query, reference, mapping_name, **kwargs):
        temporary_paths.append(Path(query.zarr_loc))
        return original_run_mapping(
            query,
            reference,
            mapping_name,
            **kwargs,
        )

    def wrong_length_score(_query, _result, *_args, **_kwargs):
        yield 0, np.zeros(len(datastore.cells.active_index("I")) - 1)

    monkeypatch.setattr(type(datastore), "run_mapping", capture_mapping)
    monkeypatch.setattr(type(datastore), "get_mapping_score", wrong_length_score)

    with pytest.raises(RuntimeError, match="selected reference cells"):
        datastore.run_doublet_detection(
            cluster_key=cluster_key,
            cluster_sample_fraction=0.01,
            max_cells_per_cluster=2,
            simulation_ratio=0.01,
            save_k=3,
            smoothing_t=1,
            random_seed=23,
        )

    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert not datastore.list_artifacts(
        kind="projection",
        from_assay="RNA",
    )
    assert "RNA_doublet_score__raw" not in datastore.cells.columns


def test_doublet_detection_rejects_legacy_graph_without_following_it(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    cluster_key = _insert_doublet_clusters(datastore)
    del datastore.zw["RNA/state"].attrs["state"]

    def fail_legacy_lookup(*_args, **_kwargs):
        raise AssertionError("Legacy graph lookup must not run")

    monkeypatch.setattr(datastore, "get_latest_graph_loc", fail_legacy_lookup)

    with pytest.raises(
        ValueError,
        match="artifact-backed connectivity chain.*Rebuild",
    ):
        datastore.run_doublet_detection(
            cluster_key=cluster_key,
            from_assay="RNA",
            cell_key="I",
            feat_key="hvgs",
            simulation_ratio=0.01,
        )


def test_doublet_detection_rejects_symphony_connectivity_chain(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    cluster_key = _insert_doublet_clusters(datastore)
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    datastore.cells.insert(
        "doublet_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        ["doublet_batch"],
        state.reduction,
        harmony_params={"nclust": 5},
        update_state=False,
    )
    ann_index = datastore.build_ann_index(
        correction,
        update_state=False,
    )
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
        update_state=False,
    )
    connectivity = datastore.build_connectivity_map(neighbors)
    references_before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )

    def fail_temporary_store(*_args, **_kwargs):
        raise AssertionError("Temporary query store must not be created")

    monkeypatch.setattr(
        datastore,
        "_create_temporary_datastore",
        fail_temporary_store,
    )

    with pytest.raises(
        ValueError,
        match="plain scaled-PCA mapping reference.*Symphony",
    ):
        datastore.run_doublet_detection(
            cluster_key=cluster_key,
            simulation_ratio=0.01,
        )

    failed_state = datastore.get_assay_state("RNA")
    assert failed_state is not None
    assert failed_state.connectivity_map == connectivity
    assert (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        == references_before
    )
