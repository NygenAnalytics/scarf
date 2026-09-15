"""Safe correction designs require an executed matched Harmony comparison."""

from typing import Any

import pytest

from scarf.agent.orchestrator import rna_tuning
from scarf.agent.orchestrator.budget import CandidateBudgetExceeded
from scarf.agent.parameter_tuning.comparisons import (
    setting_changes,
    validate_comparison_review,
)
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from tests.agent_comparison_examples import observed_action
from tests.test_agent_required_comparisons import panel_run  # noqa: F401


def _set_design(run: rna_tuning.RnaTuningRun, *, safe: bool = True) -> None:
    run.study = run.study.model_copy(
        update={
            "correctionLicense": "safe" if safe else "unsafeConfounded",
            "technicalBatchColumns": ["capture"],
        }
    )


def _action(
    selected: ParameterCandidateEvaluation,
    *,
    combining: bool = False,
    choices: dict[str, str] | None = None,
    safe: bool = True,
) -> rna_tuning.TuningAction:
    return rna_tuning.TuningAction(
        action="combine" if combining else "accept",
        selectedCandidateId=selected.candidateId,
        correctionNeed="notNeeded" if safe else "notApplicable",
        combinedSettings=choices,
        evidenceIds=[f"candidate:{selected.candidateId}"],
        quantitativeFindings=["Compare the observed matched representations."],
        qualitativeFindings=["Retain the reference population marker program."],
        comparisonConclusions=[],
        plainLanguageSummary="Prefer the native representation after comparison.",
        objectivePreservation="Retain the reference population.",
        rationale="The execution test nominates native; correction must still run.",
    )


def _assert_matched_pairs(
    run: rna_tuning.RnaTuningRun, scope: str, *, partitions: int
) -> None:
    corrected = [row for row in run.evaluations[scope] if row.parameters.useHarmony]
    assert len(corrected) == partitions
    for harmony in corrected:
        setting = run.settings[harmony.candidateId]
        native = [
            row
            for row in run.evaluations[scope]
            if not row.parameters.useHarmony
            and row.parameters.model_dump(exclude={"candidateId", "useHarmony"})
            == harmony.parameters.model_dump(exclude={"candidateId", "useHarmony"})
            and run.settings[row.candidateId].features == setting.features
        ]
        assert len(native) == 1
        assert native[0].cellSelection == harmony.cellSelection
        assert native[0].status == harmony.status == "done"
        assert run.settings[native[0].candidateId].model_dump(
            exclude={"parameters"}
        ) == (setting.model_dump(exclude={"parameters"}))


@pytest.mark.parametrize("safe", [False, True])
def test_combined_native_preference_still_executes_harmony_for_safe_design(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    safe: bool,
) -> None:
    run = request.getfixturevalue("panel_run")
    _set_design(run, safe=safe)
    reviews = []

    def review(scope: str, index: int, selected: Any, coverage: Any) -> Any:
        reviews.append(index)
        if index == 0:
            assert len(run.evaluations[scope]) == 12
            assert not any(row.parameters.useHarmony for row in run.evaluations[scope])
            return _action(
                selected,
                combining=True,
                safe=safe,
                choices={
                    name: selected.candidateId
                    for name in (
                        "hvgCountCandidateId",
                        "hvgRankingCandidateId",
                        "featurePolicyCandidateId",
                        "pcaCandidateId",
                        "neighborsCandidateId",
                    )
                },
            )
        _assert_matched_pairs(run, scope, partitions=4 if safe else 0)
        native = next(
            row
            for row in run.evaluations[scope]
            if not row.parameters.useHarmony
            and row.parameters.dimensions == 21
            and row.parameters.neighborsK == 11
            and row.parameters.leidenResolution == 1.0
        )
        return _action(native, safe=safe)

    monkeypatch.setattr(run, "review", review)
    status, selected = run.assess_scope("sample0", run.cells, None)
    assert status == "accept" and selected is not None
    assert not selected.parameters.useHarmony
    assert reviews == [0, 1]
    assert run.budget.summary()["scopes"]["sample0"]["completed"] == {
        "graphs": 10 if safe else 9,
        "partitions": 16 if safe else 12,
    }


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
def test_native_full_validation_and_recovery_admit_required_counterpart(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    safe: bool,
    recovery: bool,
) -> None:
    run = request.getfixturevalue("panel_run")
    _set_design(run, safe=safe)
    if recovery:
        run.recovery_scope = "sample1"
    else:
        run.discovery_scope = "sample0"
    initial = run.baseline()
    monkeypatch.setattr(
        run,
        "review",
        lambda scope, index, selected, coverage: _action(selected, safe=safe),
    )
    status, selected = run.assess_scope("full", run.cells, initial)
    assert status == "accept" and selected is not None
    assert not selected.parameters.useHarmony
    assert selected.parameters.model_dump(exclude={"candidateId"}) == (
        initial.parameters.model_dump(exclude={"candidateId"})
    )
    _assert_matched_pairs(run, "full", partitions=(4 if recovery else 1) if safe else 0)
    counts = {
        "graphs": 2 if safe else 1,
        "partitions": (4 if recovery else 1) * (2 if safe else 1),
    }
    assert run.budget.summary()["scopes"]["full"] == {
        "reserved": counts,
        "completed": counts,
    }
    if recovery:
        coverage = run.comparison_coverage("full", run.cells)
        for row in coverage["comparisons"]:
            assert set(
                setting_changes(
                    coverage["candidateSettings"][row["baselineCandidateId"]],
                    coverage["candidateSettings"][row["alternativeCandidateId"]],
                )
            ) == {"partition"}


@pytest.mark.parametrize(
    ("recovery", "limit", "value"),
    [
        (False, "maxFullGraphs", 1),
        (False, "maxFullPartitions", 1),
        (True, "maxFullGraphs", 1),
        (True, "maxFullPartitions", 7),
    ],
)
def test_full_matched_plan_rejects_insufficient_capacity_before_execution(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    recovery: bool,
    limit: str,
    value: int,
) -> None:
    run = request.getfixturevalue("panel_run")
    _set_design(run)
    setattr(run.request.config, limit, value)
    if recovery:
        run.recovery_scope = "sample1"
    monkeypatch.setattr(
        run, "execute", lambda *args: pytest.fail("The whole pair must fit first")
    )
    monkeypatch.setattr(
        run, "review", lambda *args: pytest.fail("No incomplete pair may be reviewed")
    )
    with pytest.raises(CandidateBudgetExceeded):
        run.assess_scope("full", run.cells, run.baseline())
    assert run.evaluations["full"] == []
    assert run.budget.summary()["scopes"]["full"] == {
        "reserved": {"graphs": 0, "partitions": 0},
        "completed": {"graphs": 0, "partitions": 0},
    }


def test_safe_combined_plan_admits_both_modes_before_new_native_partitions(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = request.getfixturevalue("panel_run")
    _set_design(run)
    run.request.config.maxScreeningEvaluations = 18

    def review(scope: str, index: int, selected: Any, coverage: Any) -> Any:
        assert index == 0
        choices = {
            name: selected.candidateId
            for name in (
                "hvgCountCandidateId",
                "hvgRankingCandidateId",
                "featurePolicyCandidateId",
                "pcaCandidateId",
                "neighborsCandidateId",
            )
        }
        choices["pcaCandidateId"] = next(
            row.candidateId
            for row in run.evaluations[scope]
            if row.parameters.dimensions == 10
        )
        return _action(selected, combining=True, choices=choices)

    monkeypatch.setattr(run, "review", review)
    with pytest.raises(CandidateBudgetExceeded):
        run.assess_scope("sample0", run.cells, None)
    assert len(run.evaluations["sample0"]) == 12
    assert not any(row.parameters.useHarmony for row in run.evaluations["sample0"])
    assert run.budget.summary()["scopes"]["sample0"] == {
        "reserved": {"graphs": 9, "partitions": 12},
        "completed": {"graphs": 9, "partitions": 12},
    }


def test_native_recovery_choice_validates_against_matched_resolution_panel(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = request.getfixturevalue("panel_run")
    _set_design(run)
    sample = rna_tuning.artifact_model_to_ref(
        run.handoff.cellSelection.model_copy(update={"artifactId": "e" * 64})
    )
    run.scope_sizes["sample1"] = 100
    baseline = run._resolution_panel("sample1", sample, run.baseline())
    run._sensitivity_panel("sample1", sample, baseline)
    run.combined_candidates["sample1"] = baseline.candidateId
    run.recovery_scope = "sample1"

    def review(scope: str, index: int, selected: Any, coverage: Any) -> Any:
        assert scope == "full" and index == 0
        assert not selected.parameters.useHarmony
        selected.metrics.topMarkerGenes = {"0": ["MS4A1"], "1": ["CD3D"]}
        selected.metrics.nClusters = 2
        comparison = run.comparison_coverage(scope, run.cells)
        action = observed_action(
            {
                "comparisonCoverage": comparison,
                "currentCandidateId": selected.candidateId,
            }
        )
        action["correctionNeed"] = "notNeeded"
        validate_comparison_review(comparison, action)
        return rna_tuning.TuningAction.model_validate(action)

    monkeypatch.setattr(run, "review", review)
    status, selected = run.assess_scope("full", run.cells, run.baseline())
    assert status == "accept" and selected is not None
    assert not selected.parameters.useHarmony
    _assert_matched_pairs(run, "full", partitions=4)
