"""Exact-input reuse and scientific gates for the automated RNA path."""

from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from scarf.agent.parameter_tuning import diagnostics, execution
from scarf.agent.parameter_tuning.selection import harmony_acceptance_gate
from scarf.agent.parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterTuningDependencies,
)
from scarf.agent.types import ArtifactReferenceModel
from scarf.storage.artifacts import fingerprint_stored_arrays
from tests.test_agent_parameter_tuning import _FakeStore, _artifact, _cell_selection


def test_stage_metric_reuse_tracks_artifacts_and_live_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    metadata = {"batch": np.asarray(["a", "b"], dtype=object)}
    store.cells = metadata
    monkeypatch.setattr(
        execution,
        "iter_metadata_column_blocks",
        lambda cells, column: iter([cells[column]]),
    )

    def run(candidate_id: str) -> ParameterCandidateEvaluation:
        candidate = ParameterCandidate(candidateId=candidate_id)
        return execution.execute_parameter_candidate(
            ParameterTuningDependencies(
                store=store,
                normalized=store.normalized,
                normalizedShape=store.normalized_shape,
                cellSelection=store.cell_selection,
                fromAssay="RNA",
                candidates={candidate_id: candidate},
                batchColumns=("batch",),
                preservationColumns=("batch",),
            ),
            candidate_id,
        )

    def counts() -> Counter[str]:
        return Counter(name for name, _args, _kwargs in store.calls)

    with execution.candidate_metric_cache():
        first = run("first")
        same = run("another_phase")
        assert first.status == same.status == "done"
        assert first.metrics == same.metrics
        assert all("another_phase" in value for value in same.evidenceIds)
        assert counts()["metric_cluster_separability"] == 1
        assert counts()["metric_proportional_batch_mixing"] == 1

        metadata["batch"][0] = "changed"
        run("changed_metadata")
        assert counts()["metric_proportional_batch_mixing"] == 2
        assert counts()["metric_clisi"] == 2
        assert counts()["metric_cluster_separability"] == 1

        store._artifacts["neighbors"] = _artifact("neighbors", 70)
        run("changed_neighbors")
        assert counts()["metric_proportional_batch_mixing"] == 3
        assert counts()["metric_graph_silhouette"] == 2

        store._artifacts["clusters"] = _artifact("cluster_labels", 71)
        run("changed_clusters")
        assert counts()["metric_cluster_separability"] == 2

    with execution.candidate_metric_cache():
        run("new_stage")
    assert counts()["metric_cluster_separability"] == 3
    assert counts()["metric_proportional_batch_mixing"] == 4


@pytest.mark.parametrize(
    ("kind", "artifact_covariate"),
    [("categorical", False), ("continuous", False), ("continuous", True)],
)
def test_pca_diagnostic_reuse_precedes_numerical_work(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    artifact_covariate: bool,
) -> None:
    root = zarr.open_group(str(tmp_path / "diagnostic.zarr"), mode="w")
    reduction = root.create_group("reduction")
    reduction.create_array("data", data=np.ones((4, 2)))
    reduction.create_array("loadings", data=np.ones((3, 2)))
    stored = root.create_group("diagnostic")
    payload = {
        "component_variance": np.asarray([2.0, 1.0]),
        "explained_variance_ratio": np.asarray([0.6, 0.3]),
        "top_loading_feature_indices": np.asarray([[0, 1, 2], [2, 1, 0]]),
        "top_loading_values": np.ones((2, 3)),
        "family_enrichment": np.ones((1, 2)),
        "covariate_association": np.ones((1, 2)),
        "adjacent_neighbor_overlap": np.asarray([0.8]),
    }
    for name, values in payload.items():
        stored.create_array(name, data=values)
    stored.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        stored, diagnostics._PCA_DIAGNOSTIC_ARRAYS
    )
    pca_ref = _artifact("reduction", 2)
    diagnostic_ref = _artifact("feature_summary", 3)
    store = SimpleNamespace(
        zw=root,
        cells={},
        inspect_artifact=lambda ref: SimpleNamespace(parameters={"feat_scaling": True}),
        load_artifact=lambda ref: reduction if ref == pca_ref else stored,
    )
    planned: list[dict[str, Any]] = []

    def reuse(*_args: Any, **kwargs: Any) -> Any:
        planned.append(kwargs)
        return SimpleNamespace(ref=diagnostic_ref, reused=True)

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Persisted diagnostic must not recompute evidence")

    monkeypatch.setattr(diagnostics, "plan_artifact", reuse)
    monkeypatch.setattr(
        diagnostics, "_metadata_column_fingerprint", lambda *_: "current"
    )
    for name in (
        "_component_variance",
        "_scaled_total_variance",
        "_top_loadings",
        "_covariate_associations",
    ):
        monkeypatch.setattr(diagnostics, name, unexpected)
    covariate_artifacts = (
        {"batch": _artifact("quality_metric", 6)} if artifact_covariate else {}
    )
    result = diagnostics._write_pca_diagnostic(
        store,
        ParameterCandidateEvaluation(
            artifacts={
                "pca": ArtifactRecord.from_ref(pca_ref),
                "neighbors": ArtifactRecord.from_ref(_artifact("neighbors", 4)),
            },
        ),
        feature_selection=_artifact("feature_selection", 5),
        selected_indices=np.arange(3),
        family_masks={"protected": np.asarray([True, False, False])},
        covariate_columns=("batch",),
        covariate_roles=("technical",),
        adjacent_overlap=0.8,
        column_kinds={"batch": kind},
        column_artifacts=covariate_artifacts,
    )
    assert result[0] == diagnostic_ref
    np.testing.assert_array_equal(result[1], payload["component_variance"])
    assert planned[0]["parameters"]["covariate_fingerprints"] == {
        "batch": covariate_artifacts["batch"].to_dict()
        if artifact_covariate
        else "current"
    }
    assert planned[0]["inputs"]["covariate_artifacts"] == covariate_artifacts
    assert planned[0]["parameters"]["covariate_kinds"] == {"batch": kind}
    assert (
        planned[0]["parameters"]["covariate_method"] == "typedCompleteCaseAssociation"
    )


@pytest.mark.parametrize(
    "damage",
    ["none", "doublets", "protected", "selection", "clisi", "graphConnectivity"],
)
def test_workflow_harmony_gate_keeps_matched_evidence_requirements(damage: str) -> None:
    native = ParameterCandidateEvaluation(
        candidateId="native",
        status="done",
        eligible=True,
        cellSelection=ArtifactReferenceModel.from_artifact_ref(_cell_selection()),
        parameters=ParameterCandidate(candidateId="native"),
        metrics=ParameterMetrics(
            batchMixing={"batch": 0.4},
            biologicalPreservation={
                "condition": {"clisi": 0.8, "graphConnectivity": 0.8}
            },
            markerCoherence=0.8,
            doubletHighScoreConcentration=0.1,
        ),
    )
    corrected = native.model_copy(deep=True)
    corrected.candidateId = "corrected"
    corrected.parameters.candidateId = "corrected"
    corrected.parameters.useHarmony = True
    corrected.metrics.batchMixing = {"batch": 0.6}
    if damage == "doublets":
        corrected.metrics.doubletHighScoreConcentration = None
    elif damage == "protected":
        corrected.metrics.biologicalPreservation["condition"]["extra"] = 0.9
    elif damage == "selection":
        corrected.cellSelection = ArtifactReferenceModel.from_artifact_ref(
            _cell_selection(99)
        )
    elif damage in {"clisi", "graphConnectivity"}:
        del native.metrics.biologicalPreservation["condition"][damage]
        del corrected.metrics.biologicalPreservation["condition"][damage]
    accepted, reasons = harmony_acceptance_gate(
        native,
        corrected,
        batch_columns=["batch", "batch"],
        protected_columns=["condition"],
        independent_unit_columns=[],
        require_doublet_evidence=True,
    )
    assert accepted is (damage == "none")
    assert bool(reasons) is (damage != "none")


def test_restore_doublets_keeps_frozen_artifacts_and_summaries() -> None:
    evaluation = ParameterCandidateEvaluation(
        artifacts={
            "doubletScore:0": ArtifactRecord.from_ref(_artifact("doublet_score", 20)),
            "doubletCellSelection:0": ArtifactRecord.from_ref(_cell_selection()),
            "doubletNativeGraph": ArtifactRecord.from_ref(
                _artifact("connectivity_map", 21)
            ),
            "doubletNativeClusters": ArtifactRecord.from_ref(
                _artifact("cluster_labels", 22)
            ),
        },
        metrics=ParameterMetrics(
            doubletScoreByCapture={"captureA": {"p50": 0.1}},
            doubletScoreQuantiles={"p95": 0.4},
            doubletCaptureCoverage=1.0,
        ),
    )
    restored = diagnostics.restore_advisory_doublets(
        evaluation, capture_column="capture"
    )
    assert restored.scores == (_artifact("doublet_score", 20),)
    assert restored.cell_selections == (_cell_selection(),)
    assert restored.capture_values == ("captureA",)
    assert restored.score_quantiles == {"p95": 0.4}


def test_revised_design_metrics_reuse_primary_analysis_and_drop_stale_findings() -> (
    None
):
    store = _FakeStore()
    candidate = ParameterCandidate(candidateId="baseline")
    deps = ParameterTuningDependencies(
        store=store,
        normalized=store.normalized,
        normalizedShape=store.normalized_shape,
        cellSelection=store.cell_selection,
        fromAssay="RNA",
        candidates={"baseline": candidate},
        batchColumns=("batch",),
        preservationColumns=("condition",),
    )
    observed = execution.execute_parameter_candidate(deps, "baseline")
    assert observed.status == "done"
    observed.metrics.biologicalPreservation["old"] = {"clisi": 1.0}
    observed.evidenceIds.append("candidate:baseline:clisi:old")
    observed.warnings.append("cLISI for 'old' unavailable: missing column")
    before = Counter(name for name, _args, _kwargs in store.calls)
    deps.preservationColumns = ("age",)
    deps.columnKinds = {"age": "continuous"}
    revised = execution.refresh_candidate_design_evidence(deps, observed)
    after = Counter(name for name, _args, _kwargs in store.calls)
    assert revised.artifacts == observed.artifacts
    assert revised.metrics.biologicalPreservation == {}
    assert not any(":clisi:old" in item for item in revised.evidenceIds)
    assert not any("'old'" in item for item in revised.warnings)
    assert any(
        "continuous column 'age' is unsupported" in item for item in revised.warnings
    )
    assert "old" in observed.metrics.biologicalPreservation
    assert after - before == Counter({"metric_proportional_batch_mixing": 1})
