"""Diagnostic accounting records calls and verified reuse without inventing work."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.parameter_tuning import diagnostics as d
from scarf.agent.parameter_tuning import execution as e
from scarf.agent.parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidateEvaluation,
)
from tests.test_agent_diagnostic_boundaries import _capture_scoring_store, _ref
from tests.test_agent_parameter_tuning import _FakeStore, _dependencies


def test_operation_counts_preserve_arguments_results_failures_and_scope() -> None:
    value = object()
    error = RuntimeError("diagnostic failed")

    def success(argument: Any, *, flag: bool) -> Any:
        assert argument is value and flag is False
        return value

    def failure() -> Any:
        raise error

    assert e.diagnostic_call("core.operation", success, value, flag=False) is value
    with e.diagnostic_work() as outer:
        assert e.diagnostic_call("core.operation", success, value, flag=False) is value
        with e.diagnostic_work() as inner:
            with pytest.raises(RuntimeError) as caught:
                e.diagnostic_call("core.operation", failure)
            assert caught.value is error
        e.diagnostic_call("core.operation", success, value, flag=False)
        assert inner["core.operation"] == {
            "attempted": 1,
            "completed": 0,
            "failed": 1,
            "cacheHits": 0,
            "restored": 0,
            "artifactReuses": 0,
        }
    assert outer["core.operation"]["attempted"] == 2
    assert outer["core.operation"]["completed"] == 2
    assert outer["core.operation"]["failed"] == 0
    assert outer["core.operation"]["artifactReuses"] == 0
    e.diagnostic_reuse("outside", "restored")
    assert "outside" not in outer


def test_failed_metric_is_retried_and_exact_cache_hits_are_counted() -> None:
    attempts = []
    value = object()

    def calculate() -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("metric unavailable")
        return value

    with e.diagnostic_work() as counts, e.candidate_metric_cache():
        key = (123, "graph_connectivity", "exact graph", "exact labels")
        with pytest.raises(ValueError, match="unavailable"):
            e._cached_candidate_metric(key, calculate)
        assert e._cached_candidate_metric(key, calculate) is value
        assert e._cached_candidate_metric(key, calculate) is value
        assert (
            e._cached_candidate_metric((*key, "changed selection"), calculate) is value
        )
    assert attempts == [1, 1, 1]
    assert counts["metric.graph_connectivity"] == {
        "attempted": 3,
        "completed": 2,
        "failed": 1,
        "cacheHits": 1,
        "restored": 0,
        "artifactReuses": 0,
    }
    with e.diagnostic_work() as uncached:
        e._cached_candidate_metric(key, calculate)
        e._cached_candidate_metric(key, calculate)
    assert uncached["metric.graph_connectivity"]["completed"] == 2
    assert uncached["metric.graph_connectivity"]["cacheHits"] == 0


@pytest.mark.parametrize("fails", [False, True])
def test_capture_diagnostic_operations_are_counted_including_partial_failure(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    labels = np.asarray(["tiny"] * 2 + ["small"] * 3 + ["large"] * 5)
    store, selected, _, calls = _capture_scoring_store(monkeypatch, labels)
    original = store.run_doublet_detection
    attempted = []

    def score(*args: Any, **kwargs: Any) -> Any:
        attempted.append(1)
        if fails and len(attempted) == 2:
            raise RuntimeError("second capture interrupted")
        return original(*args, **kwargs)

    store.run_doublet_detection = score
    with e.diagnostic_work() as counts:
        if fails:
            with pytest.raises(RuntimeError, match="interrupted"):
                d.score_advisory_doublets(
                    store,
                    selected,
                    [selected],
                    assay="RNA",
                    feature_selection=_ref("feature_selection"),
                    capture_column="capture",
                )
        else:
            evidence = d.score_advisory_doublets(
                store,
                selected,
                [selected],
                assay="RNA",
                feature_selection=_ref("feature_selection"),
                capture_column="capture",
            )
            assert evidence.capture_coverage == 0.8
    assert counts["core.doubletDetection"] == {
        "attempted": 2,
        "completed": 1 if fails else 2,
        "failed": int(fails),
        "cacheHits": 0,
        "restored": 0,
        "artifactReuses": 0,
    }
    for name in ("Normalization", "Pca", "Ann", "Neighbors", "Graph", "Partition"):
        assert counts[f"core.capture{name}"]["completed"] == 2
    assert counts["core.captureSelection"]["completed"] == (2 if fails else 3)
    assert [row[2]["dims"] for row in calls if row[0] == "pca"] == [4, 2]
    assert [row[2]["k"] for row in calls if row[0] == "neighbors"] == [4, 2]


def test_candidate_reuse_does_not_add_core_calls_or_claim_numerical_rebuilds() -> None:
    store = _FakeStore()
    deps = _dependencies(store)
    identifier = next(iter(deps.candidates))
    with e.diagnostic_work() as counts:
        first = e.execute_parameter_candidate(deps, identifier)
        assert first.status == "done"
        before = {name: dict(row) for name, row in counts.items()}
        assert e.execute_parameter_candidate(deps, identifier) is first
    assert counts["candidate.evaluation"]["cacheHits"] == 1
    for name, row in before.items():
        assert counts[name] == row
    for name in ("pca", "ann", "neighbors", "graph", "partition"):
        assert counts[f"core.{name}"]["completed"] == 1
        assert counts[f"core.{name}"]["artifactReuses"] == 0


def test_doublet_restoration_is_recorded_only_after_complete_inventory_validation() -> (
    None
):
    evaluation = ParameterCandidateEvaluation(
        artifacts={
            "doubletScore:0": ArtifactRecord.from_ref(_ref("quality_metric")),
            "doubletCellSelection:0": ArtifactRecord.from_ref(_ref("cell_selection")),
            "doubletNativeGraph": ArtifactRecord.from_ref(_ref("connectivity_map")),
            "doubletNativeClusters": ArtifactRecord.from_ref(_ref("cluster_labels")),
        }
    )
    evaluation.metrics.doubletScoreByCapture = {"capture": {"p90": 0.8}}
    with e.diagnostic_work() as counts:
        evidence = d.restore_advisory_doublets(evaluation, capture_column="capture")
        assert len(evidence.scores) == 1
        del evaluation.artifacts["doubletCellSelection:0"]
        with pytest.raises(ValueError, match="lacks"):
            d.restore_advisory_doublets(evaluation, capture_column="capture")
    assert counts == {
        "diagnostic.advisoryDoublets": {
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "cacheHits": 0,
            "restored": 1,
            "artifactReuses": 0,
        }
    }


def test_validated_pca_artifact_reuse_precedes_numerical_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "component_variance": np.ones(2),
        "explained_variance_ratio": np.ones(2),
        "top_loading_feature_indices": np.tile(np.arange(3), (2, 1)),
        "top_loading_values": np.ones((2, 3)),
        "family_enrichment": np.ones((0, 2)),
        "covariate_association": np.ones((0, 2)),
    }

    class Saved(dict):
        attrs = {"payload_fingerprint": "validated"}

    saved = Saved(payload)
    store = SimpleNamespace(
        zw=None,
        cells=None,
        inspect_artifact=lambda _: SimpleNamespace(parameters={"feat_scaling": True}),
        load_artifact=lambda ref: (
            {"data": np.ones((4, 2)), "loadings": np.ones((3, 2))}
            if ref.kind == "reduction"
            else saved
        ),
    )
    evaluation = ParameterCandidateEvaluation(
        artifacts={
            "pca": ArtifactRecord.from_ref(_ref("reduction")),
            "neighbors": ArtifactRecord.from_ref(_ref("neighbors")),
        }
    )
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    monkeypatch.setattr(d, "fingerprint_stored_arrays", lambda *_: "validated")
    monkeypatch.setattr(
        d,
        "plan_artifact",
        lambda *_, **__: SimpleNamespace(reused=True, ref=_ref("feature_summary")),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Validated PCA diagnostics must bypass numerical arrays")

    for name in ("_component_variance", "_scaled_total_variance", "_top_loadings"):
        monkeypatch.setattr(d, name, forbidden)
    with e.diagnostic_work() as counts:
        result = e.diagnostic_call(
            "diagnostic.pca",
            d._write_pca_diagnostic,
            store,
            evaluation,
            feature_selection=_ref("feature_selection"),
            selected_indices=np.arange(3),
            family_masks={},
            covariate_columns=[],
            covariate_roles=[],
            adjacent_overlap=None,
        )
    assert result[0] == _ref("feature_summary")
    assert counts == {
        "diagnostic.pca": {
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "cacheHits": 0,
            "restored": 0,
            "artifactReuses": 1,
        }
    }


def test_stability_markers_and_local_metric_reuse_are_separate_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd
    from scipy.sparse import block_diag, csr_matrix
    from tests.agent_examples import example

    labels = np.asarray([1] * 8 + [2] * 8)
    community = csr_matrix(np.ones((8, 8)) - np.eye(8))
    graph = block_diag((community, community), format="csr")
    calls = []

    def partition(*args: Any, **kwargs: Any) -> Any:
        calls.append(("partition", kwargs))
        return _ref("cluster_labels", 2)

    def marker(*args: Any, **kwargs: Any) -> Any:
        calls.append(("markers", kwargs))
        return _ref("marker_table")

    store = SimpleNamespace(
        cells=SimpleNamespace(columns=["donor", "batch"]),
        load_graph=lambda _: graph,
        run_leiden_clustering=partition,
        run_marker_search=marker,
        get_markers=lambda *_, **__: pd.DataFrame(
            {
                "group_id": ["1", "2"],
                "feature_name": ["A", "B"],
                "score": [1.0, 1.0],
                "auc": [0.9, 0.9],
            }
        ),
    )
    monkeypatch.setattr(d, "_cluster_labels", lambda *_: labels)
    monkeypatch.setattr(
        d, "_selected_feature_names", lambda *_: (np.arange(2), np.asarray(["A", "B"]))
    )
    monkeypatch.setattr(d, "_aligned_metadata", lambda *_: np.tile(["d1", "d2"], 8))
    evaluation = example(ParameterCandidateEvaluation)
    evaluation.artifacts.update(
        {
            "clusters": ArtifactRecord.from_ref(_ref("cluster_labels")),
            "connectivityMap": ArtifactRecord.from_ref(_ref("connectivity_map")),
        }
    )
    with e.diagnostic_work() as counts, e.candidate_metric_cache():
        results = [
            d.augment_cluster_evaluations(
                store,
                [evaluation],
                marker_assay="RNA",
                marker_features=_ref("feature_selection"),
                independent_unit_columns=["donor"],
                technical_columns=["batch"],
            )[0]
            for _ in range(2)
        ]
    assert results[0].metrics == results[1].metrics
    assert results[0].metrics.subsampleStability == pytest.approx(1.0)
    assert counts["metric.subsample_stability"]["completed"] == 1
    assert counts["metric.subsample_stability"]["cacheHits"] == 1
    assert counts["core.subsamplePartition"]["completed"] == 1
    assert counts["core.alternateSeedPartition"]["completed"] == 2
    assert counts["core.markers"]["completed"] == 2
    assert counts["core.markers"]["artifactReuses"] == 0
    assert counts["metric.crossUnitSupport"]["completed"] == 2
    assert counts["metric.technicalAssociation"]["completed"] == 2
    assert all(row[1]["random_seed"] == 9173 for row in calls if row[0] == "partition")
    assert all(
        row[1]["features"] == _ref("feature_selection")
        for row in calls
        if row[0] == "markers"
    )
