import inspect
from pathlib import Path
from threading import Event, get_ident

import pytest
from scipy.sparse import tril

import scarf.datastore.pipeline_accessor as pipeline_accessor_module
from scarf.datastore.pipeline_accessor import PipelineEvent
from scarf.storage.artifacts import ArtifactRef
from scarf.utils import logger


def test_basic_rna_pipeline_returns_only_named_artifact_refs(
    datastore_ephemeral,
) -> None:
    artifacts = datastore_ephemeral.pipeline.run(
        pipeline_id="basic_rna_analysis",
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "label": "pipeline_hvgs",
        },
        pca={"dims": 5, "n_centroids": 10},
        ann_index={"ann_m": 16},
        neighbors={"k": 3},
        connectivity={},
        umap={"n_epochs": 10, "label": "pipeline_umap"},
        leiden={
            0.5: {},
            1.0: {},
        },
        paris={"n_clusters": 3, "label": "pipeline_paris"},
        doublet_scoring=False,
        markers={
            "clusters": {"leiden": 1.0},
        },
    )

    assert list(artifacts) == [
        "cell_selection",
        "highly_variable_features",
        "normalized",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "embedding_initialization",
        "umap",
        "leiden_0.5",
        "leiden_1.0",
        "paris",
        "selected_clusters",
        "markers",
    ]
    assert all(isinstance(ref, ArtifactRef) for ref in artifacts.values())
    assert artifacts["leiden_0.5"] != artifacts["leiden_1.0"]
    assert all(
        datastore_ephemeral.inspect_artifact(ref).complete for ref in artifacts.values()
    )
    assert (
        datastore_ephemeral.resolve_features("RNA", "pipeline_hvgs")
        == artifacts["highly_variable_features"]
    )
    assay = datastore_ephemeral.get_assay("RNA")
    assert "pipeline_hvgs" in assay.feats.columns
    assert "I__pipeline_hvgs" not in assay.feats.columns
    marker_status = datastore_ephemeral.inspect_artifact(artifacts["markers"])
    assert marker_status.inputs is not None
    marker_features = ArtifactRef.from_dict(marker_status.inputs["feature_selection"])
    assert marker_features == datastore_ephemeral.resolve_features(
        "RNA", "all_features"
    )
    cluster_column = "RNA_leiden_1.0"
    edited = datastore_ephemeral.cells.fetch(
        cluster_column,
        key="I",
    ).copy()
    edited[0] = int(edited.max()) + 1
    datastore_ephemeral.cells.insert(
        cluster_column,
        edited,
        key="I",
        overwrite=True,
    )
    with pytest.raises(ValueError, match="cluster labels are stale"):
        datastore_ephemeral.get_markers(
            from_assay="RNA",
            cell_key="I",
            group_key=cluster_column,
        )


def test_pipeline_emits_ordered_events_and_omits_skipped_stages(
    datastore_ephemeral,
) -> None:
    events: list[PipelineEvent] = []
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
    )

    try:
        artifacts = datastore_ephemeral.pipeline.run(
            filtering=False,
            cell_cycle_scoring=False,
            highly_variable_features={
                "top_n": 50,
                "label": "pipeline_callback_hvgs",
            },
            pca={"dims": 3, "n_centroids": 5},
            neighbors={"k": 3},
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers=False,
            callback=events.append,
        )
    finally:
        logger.remove(sink)

    expected_stages = [
        "highly_variable_features",
        "normalization",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity",
        "embedding_initialization",
    ]
    assert [(event.kind, event.stage) for event in events] == [
        (kind, stage)
        for stage in expected_stages
        for kind in ("stage_started", "stage_completed")
    ]
    assert all(event.error is None for event in events)
    lifecycle_messages = [
        message
        for message in messages
        if message.startswith("Running pipeline stage:")
        or message.startswith("Completed pipeline stage:")
        or message.startswith("Pipeline completed")
    ]
    assert lifecycle_messages == [
        *[
            message
            for stage in expected_stages
            for message in (
                f"Running pipeline stage: {stage.replace('_', ' ')}",
                f"Completed pipeline stage: {stage.replace('_', ' ')}",
            )
        ],
        f"Pipeline completed with {len(artifacts)} artifacts",
    ]


def test_pipeline_emits_failed_stage_and_reraises_original_error(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    events: list[PipelineEvent] = []
    expected_error = RuntimeError("PCA callback test failure")

    def fail_pca(self, *_args, **_kwargs):
        raise expected_error

    monkeypatch.setattr(type(datastore), "run_pca", fail_pca)

    with pytest.raises(RuntimeError, match="PCA callback test failure") as raised:
        datastore.pipeline.run(
            filtering=False,
            cell_cycle_scoring=False,
            highly_variable_features={
                "top_n": 50,
                "label": "pipeline_callback_failure_hvgs",
            },
            pca={"dims": 3, "n_centroids": 5},
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers=False,
            callback=events.append,
        )

    assert raised.value is expected_error
    assert [(event.kind, event.stage) for event in events[-2:]] == [
        ("stage_started", "pca"),
        ("stage_failed", "pca"),
    ]
    assert events[-1].error is expected_error


def test_pipeline_retry_reuses_completed_artifacts_after_stage_failure(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    expected_error = RuntimeError("transient neighbor query failure")
    observed: dict[str, list[ArtifactRef]] = {
        "normalized": [],
        "pca": [],
        "ann_index": [],
    }
    original_methods = {
        "normalized": type(datastore).run_normalization,
        "pca": type(datastore).run_pca,
        "ann_index": type(datastore).build_ann_index,
    }

    def recording_method(stage: str):
        original = original_methods[stage]

        def wrapped(self, *args, **kwargs):
            ref = original(self, *args, **kwargs)
            observed[stage].append(ref)
            return ref

        return wrapped

    for stage, method_name in (
        ("normalized", "run_normalization"),
        ("pca", "run_pca"),
        ("ann_index", "build_ann_index"),
    ):
        monkeypatch.setattr(
            type(datastore),
            method_name,
            recording_method(stage),
        )

    original_query_neighbors = type(datastore).query_neighbors
    query_attempts = 0

    def fail_first_neighbor_query(self, *args, **kwargs):
        nonlocal query_attempts
        query_attempts += 1
        if query_attempts == 1:
            raise expected_error
        return original_query_neighbors(self, *args, **kwargs)

    monkeypatch.setattr(
        type(datastore),
        "query_neighbors",
        fail_first_neighbor_query,
    )
    options = {
        "filtering": False,
        "cell_cycle_scoring": False,
        "highly_variable_features": {
            "top_n": 50,
            "label": "pipeline_resume_hvgs",
        },
        "pca": {"dims": 3, "n_centroids": 5},
        "neighbors": {"k": 3},
        "umap": False,
        "leiden": {},
        "paris": False,
        "doublet_scoring": False,
        "markers": False,
    }
    failed_events: list[PipelineEvent] = []

    with pytest.raises(
        RuntimeError, match="transient neighbor query failure"
    ) as raised:
        datastore.pipeline.run(**options, callback=failed_events.append)

    assert raised.value is expected_error
    assert [(event.kind, event.stage) for event in failed_events[-2:]] == [
        ("stage_started", "neighbors"),
        ("stage_failed", "neighbors"),
    ]
    assert failed_events[-1].error is expected_error
    assert not {
        "connectivity",
        "embedding_initialization",
    }.intersection(event.stage for event in failed_events)
    first_refs = {stage: refs[0] for stage, refs in observed.items()}
    assert all(datastore.inspect_artifact(ref).complete for ref in first_refs.values())

    retry_events: list[PipelineEvent] = []
    artifacts = datastore.pipeline.run(**options, callback=retry_events.append)

    assert query_attempts == 2
    assert all(event.kind != "stage_failed" for event in retry_events)
    assert artifacts["normalized"] == first_refs["normalized"]
    assert artifacts["pca"] == first_refs["pca"]
    assert artifacts["ann_index"] == first_refs["ann_index"]
    assert {stage: refs[1] for stage, refs in observed.items()} == first_refs


def test_pipeline_logs_callback_errors_and_continues(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    callback_logs: list[str] = []

    class RecordingLogger:
        def info(self, _message: str) -> None:
            pass

        def exception(self, message: str) -> None:
            callback_logs.append(message)

    def fail_callback(_event: PipelineEvent) -> None:
        raise RuntimeError("Callback failed")

    monkeypatch.setattr(
        pipeline_accessor_module,
        "logger",
        RecordingLogger(),
    )

    artifacts = datastore_ephemeral.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 50,
            "label": "pipeline_broken_callback_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        neighbors={"k": 3},
        umap=False,
        leiden={},
        paris=False,
        doublet_scoring=False,
        markers=False,
        callback=fail_callback,
    )

    assert artifacts["pca"].kind == "reduction"
    assert callback_logs
    assert all("Pipeline callback failed" in message for message in callback_logs)


def test_pipeline_rejects_unknown_id(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="basic_rna_analysis"):
        datastore_ephemeral.pipeline.run(pipeline_id="unknown")


def test_pipeline_requires_highly_variable_features(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="highly_variable_features"):
        datastore_ephemeral.pipeline.run(
            highly_variable_features=False,
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers=False,
        )


def test_pipeline_selects_clusters_by_pca_silhouette(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    events: list[PipelineEvent] = []
    store = datastore_ephemeral
    original = store._store_to_sparse
    graph_reads = 0

    def counted_store_to_sparse(*args, **kwargs):
        nonlocal graph_reads
        graph_reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_store_to_sparse", counted_store_to_sparse)
    artifacts = store.pipeline.run(
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "label": "pipeline_silhouette_hvgs",
        },
        pca={"dims": 5, "n_centroids": 10},
        neighbors={"k": 3},
        umap=False,
        leiden={
            0.5: {},
            1.0: {},
        },
        paris={"n_clusters": 3, "label": "pipeline_silhouette_paris"},
        clustering_concurrency=2,
        doublet_scoring=False,
        markers={},
        callback=events.append,
    )

    assert "selected_clusters" in artifacts
    selected = artifacts["selected_clusters"]
    candidates = {
        artifacts["leiden_0.5"]: "RNA_leiden_0.5",
        artifacts["leiden_1.0"]: "RNA_leiden_1.0",
        artifacts["paris"]: "RNA_pipeline_silhouette_paris",
    }
    assert selected in candidates
    assert artifacts["markers"].kind == "marker_table"
    assert graph_reads == 1
    assert store._graphMemoryCache is None
    for stage in ("cluster_selection", "markers"):
        assert [event.kind for event in events if event.stage == stage] == [
            "stage_started",
            "stage_completed",
        ]

    labels = store.cells.fetch("RNA_clusters")
    assert (labels == store.cells.fetch(candidates[selected])).all()
    column = store.zw["cellData"]["RNA_clusters"]
    assert ArtifactRef.from_dict(column.attrs["source_artifact"]) == selected
    assert store.get_markers(group_key="RNA_clusters", group_id=labels[0]) is not None


def test_pipeline_names_a_single_partition_without_silhouette(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("silhouette selection is unnecessary for one partition")

    monkeypatch.setattr(
        pipeline_accessor_module.PipelineAccessor,
        "_select_clusters_by_pca_silhouette",
        fail,
    )
    artifacts = datastore_ephemeral.pipeline.run(
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={"top_n": 100, "label": "pipeline_single_hvgs"},
        pca={"dims": 5, "n_centroids": 10},
        neighbors={"k": 3},
        umap=False,
        leiden={1.0: {}},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )

    assert artifacts["selected_clusters"] == artifacts["leiden_1.0"]
    store = datastore_ephemeral
    assert (
        store.cells.fetch("RNA_clusters") == store.cells.fetch("RNA_leiden_1.0")
    ).all()


def test_pipeline_rejects_downstream_steps_without_clustering(
    datastore_ephemeral,
) -> None:
    with pytest.raises(ValueError, match="Marker needs a clustering result"):
        datastore_ephemeral.pipeline.run(
            filtering={},
            cell_cycle_scoring=False,
            highly_variable_features={"top_n": 100, "label": "pipeline_no_hvgs"},
            pca={"dims": 5, "n_centroids": 10},
            neighbors={"k": 3},
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers={},
        )


def test_pipeline_rejects_invalid_clustering_concurrency(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="clustering_concurrency"):
        datastore_ephemeral.pipeline.run(
            clustering_concurrency=0,
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers=False,
        )


def test_pipeline_reuses_leiden_without_recomputing(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    options = {
        "filtering": False,
        "cell_cycle_scoring": False,
        "highly_variable_features": {
            "top_n": 100,
            "label": "pipeline_cached_leiden_hvgs",
        },
        "pca": {"dims": 3, "n_centroids": 5},
        "neighbors": {"k": 3},
        "umap": False,
        "leiden": {0.5: {}},
        "paris": False,
        "doublet_scoring": False,
        "markers": False,
    }
    first = datastore.pipeline.run(**options)

    def fail_compute(*_args, **_kwargs):
        pytest.fail("A reusable Leiden artifact must not be recomputed")

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(fail_compute),
    )
    events: list[PipelineEvent] = []
    second = datastore.pipeline.run(**options, callback=events.append)

    assert second["leiden_0.5"] == first["leiden_0.5"]
    assert [
        (event.kind, event.stage) for event in events if event.stage == "leiden_0.5"
    ] == [
        ("stage_started", "leiden_0.5"),
        ("stage_completed", "leiden_0.5"),
    ]


def test_pipeline_computes_leiden_with_requested_graph_options(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    original_compute = type(datastore)._compute_prepared_leiden
    observed = False

    def capture_graph(prepared, graph):
        nonlocal observed
        observed = True
        assert prepared.symmetric_graph is True
        assert prepared.graph_upper_only is True
        assert tril(graph, k=-1).nnz == 0
        return original_compute(prepared, graph)

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(capture_graph),
    )
    datastore.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "label": "pipeline_graph_options_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        neighbors={"k": 3},
        umap=False,
        leiden={
            0.5: {
                "symmetric_graph": True,
                "graph_upper_only": True,
            }
        },
        paris=False,
        doublet_scoring=False,
        markers=False,
    )

    assert observed is True


def test_pipeline_reports_failed_and_aborted_clustering_jobs(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    original_compute = type(datastore)._compute_prepared_leiden
    expected_error = RuntimeError("Leiden callback test failure")
    events: list[PipelineEvent] = []

    def fail_one_resolution(prepared, graph):
        if prepared.resolution == 0.5:
            raise expected_error
        return original_compute(prepared, graph)

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(fail_one_resolution),
    )

    with pytest.raises(RuntimeError, match="Leiden callback test failure") as raised:
        datastore.pipeline.run(
            filtering=False,
            cell_cycle_scoring=False,
            highly_variable_features={
                "top_n": 100,
                "label": "pipeline_failed_clustering_hvgs",
            },
            pca={"dims": 3, "n_centroids": 5},
            neighbors={"k": 3},
            umap=False,
            leiden={0.5: {}, 1.0: {}},
            paris=False,
            clustering_concurrency=2,
            doublet_scoring=False,
            markers=False,
            callback=events.append,
        )

    assert raised.value is expected_error
    failed_events = {
        event.stage: event
        for event in events
        if event.kind == "stage_failed" and event.stage in {"leiden_0.5", "leiden_1.0"}
    }
    assert failed_events["leiden_0.5"].error is expected_error
    assert isinstance(failed_events["leiden_1.0"].error, RuntimeError)
    assert "was not written" in str(failed_events["leiden_1.0"].error)
    for stage in ("leiden_0.5", "leiden_1.0"):
        assert [event.kind for event in events if event.stage == stage] == [
            "stage_started",
            "stage_failed",
        ]


def test_leiden_public_api_does_not_accept_precomputed_membership(
    datastore_ephemeral,
) -> None:
    parameters = inspect.signature(datastore_ephemeral.run_leiden_clustering).parameters
    assert "precomputed_membership" not in parameters


def test_pipeline_overlaps_paris_compute_but_serializes_leiden_publish(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    compute_started = Event()
    paris_started = Event()
    paris_finished = Event()
    original_compute = type(datastore)._compute_prepared_leiden
    original_paris = type(datastore).run_paris_clustering
    original_publish = type(datastore)._publish_prepared_leiden
    callback_events: list[PipelineEvent] = []
    callback_threads: list[int] = []
    messages: list[str] = []
    caller_thread = get_ident()

    def record_event(event: PipelineEvent) -> None:
        callback_events.append(event)
        callback_threads.append(get_ident())

    def observed_compute(prepared, graph):
        compute_started.set()
        assert paris_started.wait(timeout=5)
        return original_compute(prepared, graph)

    def observed_paris(self, *args, **kwargs):
        paris_started.set()
        assert compute_started.wait(timeout=5)
        result = original_paris(self, *args, **kwargs)
        paris_finished.set()
        return result

    def observed_publish(self, prepared, membership):
        assert paris_finished.is_set()
        return original_publish(self, prepared, membership)

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(observed_compute),
    )
    monkeypatch.setattr(
        type(datastore),
        "run_paris_clustering",
        observed_paris,
    )
    monkeypatch.setattr(
        type(datastore),
        "_publish_prepared_leiden",
        observed_publish,
    )

    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
    )
    try:
        datastore.pipeline.run(
            filtering=False,
            cell_cycle_scoring=False,
            highly_variable_features={
                "top_n": 100,
                "label": "pipeline_concurrency_hvgs",
            },
            pca={"dims": 3, "n_centroids": 5},
            neighbors={"k": 3},
            umap=False,
            leiden={0.5: {}},
            paris={"n_clusters": 3, "label": "pipeline_concurrency_paris"},
            clustering_concurrency=2,
            doublet_scoring=False,
            markers=False,
            callback=record_event,
        )
    finally:
        logger.remove(sink)

    assert paris_finished.is_set()
    assert set(callback_threads) == {caller_thread}
    for stage in ("leiden_0.5", "paris"):
        assert [event.kind for event in callback_events if event.stage == stage] == [
            "stage_started",
            "stage_completed",
        ]
        label = stage.replace("_", " ")
        expected_messages = [
            f"Running pipeline stage: {label}",
            f"Completed pipeline stage: {label}",
        ]
        assert [message for message in messages if message in expected_messages] == (
            expected_messages
        )


def test_pipeline_rejects_unsaved_markers_before_writes(
    datastore_ephemeral,
) -> None:
    before = set(datastore_ephemeral.cells.columns)

    with pytest.raises(ValueError, match="skip_save"):
        datastore_ephemeral.pipeline.run(
            markers={"skip_save": True},
        )

    assert set(datastore_ephemeral.cells.columns) == before


def test_basic_rna_pipeline_supports_optional_harmony(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    datastore.cells.insert(
        "pipeline_batch",
        ["a" if i % 2 else "b" for i in range(datastore.cells.N)],
        overwrite=True,
    )

    artifacts = datastore.pipeline.run(
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 50,
            "label": "pipeline_harmony_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        harmony={
            "batch_columns": ["pipeline_batch"],
            "harmony_params": {"nclust": 3},
        },
        neighbors={"k": 3},
        umap=False,
        leiden={},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )

    state = datastore.get_assay_state("RNA")
    assert "harmony" in artifacts
    assert state is not None
    assert state.batch_correction == artifacts["harmony"]


def test_basic_rna_pipeline_runs_score_steps_after_clustering(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    events: list[PipelineEvent] = []
    temporary_paths: list[Path] = []
    original_run_mapping = type(datastore).run_mapping

    def observe_mapping(query, reference, mapping_name, **kwargs):
        assert query is not datastore
        temporary_paths.append(Path(query.zarr_loc))
        return original_run_mapping(
            query,
            reference,
            mapping_name,
            **kwargs,
        )

    monkeypatch.setattr(type(datastore), "run_mapping", observe_mapping)
    artifacts = datastore.pipeline.run(
        filtering={},
        cell_cycle_scoring={},
        highly_variable_features={
            "top_n": 50,
            "label": "pipeline_full_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        neighbors={"k": 3},
        umap={"n_epochs": 5, "label": "pipeline_full_umap"},
        leiden={0.5: {}},
        paris={"n_clusters": 3, "label": "pipeline_full_paris"},
        doublet_scoring={
            "clusters": "paris",
            "cluster_sample_fraction": 0.01,
            "max_cells_per_cluster": 2,
            "simulation_ratio": 0.01,
            "save_k": 3,
            "smoothing_t": 1,
            "random_seed": 9,
        },
        markers={
            "clusters": {"leiden": 0.5},
            "features": "pipeline_full_hvgs",
        },
        callback=events.append,
    )

    assert artifacts["cell_cycle"].kind == "cell_cycle"
    assert artifacts["doublets"].kind == "doublet_score"
    assert artifacts["markers"].kind == "marker_table"
    assert list(artifacts).index("doublets") > list(artifacts).index("paris")
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert not datastore.list_artifacts(
        kind="projection",
        from_assay="RNA",
    )
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map == artifacts["connectivity_map"]
    reference = datastore.get_mapping_reference()
    assert reference.neighbors == artifacts["neighbors"]
    assert reference.symphony_state is None
    for stage in (
        "cell_cycle_scoring",
        "umap",
        "doublet_scoring",
        "markers",
    ):
        assert [event.kind for event in events if event.stage == stage] == [
            "stage_started",
            "stage_completed",
        ]
