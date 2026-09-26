"""Selection cannot erase failed preservation, alternative evidence, or lineage."""

from types import SimpleNamespace
from typing import Any

import pytest

from scarf.agent.parameter_tuning import selection
from scarf.agent.parameter_tuning.contracts import (
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterTuningReport,
)
from scarf.agent.tools import artifact_reference
from tests.test_agent_parameter_tuning import (
    _FakeStore,
    _artifact,
    _cell_selection,
)


def _evaluation(
    name: str = "baseline", **parameters: Any
) -> ParameterCandidateEvaluation:
    return ParameterCandidateEvaluation(
        candidateId=name,
        parameters=ParameterCandidate(candidateId=name, **parameters),
        status="done",
        eligible=True,
        cellSelection=artifact_reference(_cell_selection()),
        clusterColumn=f"clusters_{name}",
        artifacts={
            "clusters": artifact_reference(_artifact("cluster_labels", 7)).model_dump()
        },
        metrics=ParameterMetrics(
            seedStability=0.8,
            markerCoherence=0.8,
            markerSpecificityMedian=0.8,
            clusterConnectivity=0.8,
            membershipStrengthMean=0.8,
            crossUnitSupport=0.8,
            batchMixing={"library": 0.2},
            biologicalPreservation={
                "condition": {"clisi": 0.8, "graphConnectivity": 0.8}
            },
            doubletHighScoreConcentration=0.1,
        ),
        evidenceIds=[
            f"candidate:{name}:{metric}"
            for metric in (
                "seedStability",
                "markers",
                "crossUnitSupport",
                "protected:condition",
                "doublet",
                "batchMixing",
                "geometry",
            )
        ],
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"status": "failed"}, "not an eligible execution"),
        (
            {"parameters": ParameterCandidate(candidateId="corrected")},
            "correction modes",
        ),
        (
            {
                "parameters": ParameterCandidate(
                    candidateId="corrected", useHarmony=True, dimensions=10
                )
            },
            "parameters are not matched",
        ),
        (
            {"cellSelection": artifact_reference(_cell_selection(99))},
            "different cell selections",
        ),
        ({"metrics": {"batchMixing": {}}}, "Batch comparison is missing"),
        ({"metrics": {"batchMixing": {"library": 0.1}}}, "materially worsened"),
        (
            {"metrics": {"biologicalPreservation": {}}},
            "Protected comparison is missing",
        ),
        (
            {"metrics": {"biologicalPreservation": {"condition": {"clisi": 0.8}}}},
            "Required protected metrics",
        ),
        (
            {
                "metrics": {
                    "biologicalPreservation": {
                        "condition": {
                            "clisi": 0.8,
                            "graphConnectivity": 0.8,
                            "extra": 0.9,
                        }
                    }
                }
            },
            "Protected metrics do not align",
        ),
        (
            {
                "metrics": {
                    "biologicalPreservation": {
                        "condition": {"clisi": 0.6, "graphConnectivity": 0.8}
                    }
                }
            },
            "degraded protected evidence",
        ),
        (
            {"metrics": {"crossUnitSupport": None}},
            "Cross-unit support comparison is missing",
        ),
        ({"metrics": {"crossUnitSupport": 0.6}}, "degraded cross-unit support"),
        (
            {"metrics": {"markerCoherence": None}},
            "Marker-coherence comparison is missing",
        ),
        ({"metrics": {"markerCoherence": 0.6}}, "degraded marker coherence"),
        (
            {"metrics": {"markerSpecificityMedian": None}},
            "Matched marker specificity comparison is missing",
        ),
        ({"metrics": {"clusterConnectivity": 0.6}}, "degraded cluster connectivity"),
        (
            {"metrics": {"doubletHighScoreConcentration": 0.3}},
            "increased doublet concentration",
        ),
        (
            {"metrics": {"doubletHighScoreConcentration": None}},
            "Matched doublet-concentration comparison is missing",
        ),
    ],
)
def test_harmony_batch_gain_cannot_hide_missing_or_contradictory_evidence(
    change: dict[str, Any], reason: str
) -> None:
    native = _evaluation()
    corrected = _evaluation("corrected", useHarmony=True)
    corrected.metrics.batchMixing = {"library": 0.7}
    assert selection.harmony_acceptance_gate(
        native,
        corrected,
        batch_columns=["library", "library"],
        protected_columns=["condition", "condition"],
        independent_unit_columns=["donor"],
        require_doublet_evidence=True,
    ) == (True, [])
    updates = dict(change)
    if "metrics" in updates:
        updates["metrics"] = corrected.metrics.model_copy(update=updates["metrics"])
    accepted, reasons = selection.harmony_acceptance_gate(
        native,
        corrected.model_copy(update=updates),
        batch_columns=["library", "library"],
        protected_columns=["condition", "condition"],
        independent_unit_columns=["donor"],
        require_doublet_evidence=True,
    )
    assert not accepted
    assert any(reason in item for item in reasons)
    assert len(reasons) == len(set(reasons))


def test_harmony_requires_approved_batch_evidence_and_valid_tolerance() -> None:
    native, corrected = _evaluation(), _evaluation("corrected", useHarmony=True)
    for tolerance in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tolerance"):
            selection.harmony_acceptance_gate(
                native,
                corrected,
                batch_columns=["library"],
                protected_columns=[],
                tolerance=tolerance,
            )
    accepted, reasons = selection.harmony_acceptance_gate(
        native.model_copy(update={"eligible": False}),
        corrected,
        batch_columns=[],
        protected_columns=[],
    )
    assert not accepted
    assert any("matched native candidate" in reason for reason in reasons)
    assert "No approved batch metric was supplied." in reasons


def test_pareto_dominance_marks_one_factor_superiority_idempotently() -> None:
    native = _evaluation()
    alternative = _evaluation("neighbors_21", neighborsK=21)
    alternative.metrics.seedStability = 0.95
    alternative.metrics.markerCoherence = 0.95
    native.metrics.technicalAssociation = {"depth": 0.2}
    alternative.metrics.technicalAssociation = {"depth": 0.2}
    annotated = selection.annotate_candidate_dominance([native, alternative])
    assert annotated[0].metrics.dominatedByCandidateIds == ["neighbors_21"]
    assert annotated[1].metrics.dominatesCandidateIds == ["baseline"]
    assert selection.annotate_candidate_dominance(annotated) == annotated


def test_pareto_does_not_treat_incomparable_or_missing_metrics_as_superiority() -> None:
    baseline, alternative = _evaluation(), _evaluation("alternative", dimensions=10)
    alternative.metrics.seedStability = 0.95
    alternative.metrics.markerCoherence = 0.6
    assert not selection.annotate_candidate_dominance([baseline, alternative])[
        0
    ].metrics.dominatedByCandidateIds
    alternative.metrics = ParameterMetrics(seedStability=0.95)
    baseline.metrics = ParameterMetrics(seedStability=0.8)
    assert not selection.annotate_candidate_dominance([baseline, alternative])[
        0
    ].metrics.dominatedByCandidateIds
    alternative.parameters.useHarmony = True
    result = selection.annotate_candidate_dominance([baseline, alternative])
    assert all(item.metrics.paretoOptimal is None for item in result)
    with pytest.raises(ValueError, match="tolerance"):
        selection.annotate_candidate_dominance(result, tolerance=float("nan"))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"exists": False}, "unavailable or incomplete"),
        ({"complete": False}, "unavailable or incomplete"),
        ({"inputs": {}}, "no cell-selection input"),
        (
            {"inputs": {"cell_selection": _cell_selection(9).to_dict()}},
            "candidate does not match normalized artifact lineage",
        ),
    ],
)
def test_promoting_selected_artifacts_checks_exact_normalization_lineage(
    change: dict[str, Any], reason: str
) -> None:
    candidate = _evaluation()
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=candidate.cellSelection,
        recommendedCandidateId=candidate.candidateId,
        evaluations=[candidate],
    )
    inspections: list[Any] = []

    def inspect(ref: Any) -> Any:
        inspections.append(ref)
        return SimpleNamespace(
            **(
                {
                    "exists": True,
                    "complete": True,
                    "inputs": {"cell_selection": _cell_selection().to_dict()},
                }
                | change
            )
        )

    store = SimpleNamespace(inspect_artifact=inspect)
    with pytest.raises(ValueError, match=reason):
        selection.promote_parameter_candidate(
            store, report=report, normalized=_artifact("normalized", 1)
        )
    assert inspections == [_artifact("normalized", 1)]


@pytest.mark.parametrize(
    ("report_change", "candidate_change", "normalized", "limit", "reason"),
    [
        (
            {"status": "needsInput"},
            {},
            _artifact("normalized", 1),
            64,
            "completed native tuning recommendation",
        ),
        (
            {"recommendedCandidateId": "absent"},
            {},
            _artifact("normalized", 1),
            64,
            "not an eligible execution",
        ),
        (
            {},
            {"artifacts": {}},
            _artifact("normalized", 1),
            64,
            "exact cluster artifact",
        ),
        (
            {},
            {},
            _artifact("normalized", 1, "ADT"),
            64,
            "exact normalized assay artifact",
        ),
        (
            {"cellSelection": artifact_reference(_cell_selection(99))},
            {},
            _artifact("normalized", 1),
            64,
            "report does not match normalized artifact lineage",
        ),
        ({}, {}, _artifact("normalized", 1), 1, "at least two"),
    ],
)
def test_native_promotion_rejects_an_incomplete_or_mismatched_recommendation(
    report_change: dict[str, Any],
    candidate_change: dict[str, Any],
    normalized: Any,
    limit: int,
    reason: str,
) -> None:
    candidate = _evaluation().model_copy(update=candidate_change)
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=artifact_reference(_cell_selection()),
        recommendedCandidateId="baseline",
        evaluations=[candidate],
    ).model_copy(update=report_change)
    store = _FakeStore()
    with pytest.raises(ValueError, match=reason):
        selection.promote_parameter_candidate(
            store, report=report, normalized=normalized, identity_feature_limit=limit
        )
    assert not any(
        name in {"run_pca", "run_leiden_clustering"} for name, _, _ in store.calls
    )


@pytest.mark.parametrize(
    ("report_change", "candidate_change", "kwargs", "reason"),
    [
        ({"status": "needsInput"}, {}, {}, "must be done"),
        ({}, {}, {"marker_assay": ""}, "must be non-empty"),
        ({"cellSelection": None}, {}, {}, "exact cell selection"),
        ({}, {}, {"marker_assay": "ADT"}, "Unknown marker assay"),
        ({}, {}, {"native_assay": "ADT"}, "lacks a native tuning recommendation"),
        ({}, {"artifacts": {}}, {}, "lacks exact clusters"),
        (
            {},
            {"cellSelection": artifact_reference(_cell_selection(99))},
            {},
            "different cell selection",
        ),
        (
            {},
            {},
            {"native_assay": "RNA", "recommended_integration_id": "unexecuted"},
            "either an integrated graph",
        ),
        ({}, {}, {"recommended_integration_id": "unexecuted"}, "was not evaluated"),
    ],
)
def test_finalization_cannot_publish_an_unexecuted_or_unmatched_native_branch(
    report_change: dict[str, Any],
    candidate_change: dict[str, Any],
    kwargs: dict[str, Any],
    reason: str,
) -> None:
    candidate = _evaluation().model_copy(update=candidate_change)
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=artifact_reference(_cell_selection()),
        recommendedCandidateId="baseline",
        evaluations=[candidate],
    ).model_copy(update=report_change)
    with pytest.raises(ValueError, match=reason):
        selection.finalize_parameter_tuning_selection(
            report, **({"marker_assay": "RNA"} | kwargs)
        )
