"""Executed sensitivity coverage and exact scientific comparison conclusions."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.orchestrator import journal, rna_tuning
from scarf.agent.orchestrator.budget import candidate_identity, CandidateBudgetExceeded
from scarf.agent.orchestrator.models import (
    AutomatedWorkflowConfig,
    PreprocessedAssayHandoff,
    AutomatedPreprocessingPlan,
)
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.parameter_tuning.contracts import (
    ParameterCandidateEvaluation,
    ArtifactRecord,
)
from scarf.agent.parameter_tuning.comparisons import (
    validate_comparison_review,
    setting_changes,
)
from scarf.agent.types import ArtifactReferenceModel
from tests.agent_comparison_examples import comparison_review
from tests.agent_examples import example
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401


@pytest.mark.parametrize(
    "damage",
    [
        "missing_row",
        "different_cells",
        "two_axes",
        "failed_alternative",
        "missing_conclusion",
        "unexecuted_combination",
        "wrong_final_graph",
        "wrong_preference",
    ],
)
def test_acceptance_requires_actual_matched_coverage(damage: str) -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    if damage == "missing_row":
        coverage["comparisons"].pop(0)
    elif damage == "different_cells":
        coverage["candidateSettings"]["genes-two"]["cellSelection"] = {"changed": True}
    elif damage == "two_axes":
        coverage["candidateSettings"]["genes-two"]["parameters"]["dimensions"] += 1
    elif damage == "failed_alternative":
        coverage["candidateSettings"]["genes-two"]["status"] = "failed"
    elif damage == "missing_conclusion":
        review["comparisonConclusions"].pop()
    elif damage == "unexecuted_combination":
        coverage["combinedCandidateId"] = "unexecuted"
    elif damage == "wrong_final_graph":
        coverage["candidateSettings"]["resolution-half"]["parameters"][
            "neighborsK"
        ] += 1
    else:
        review["selectedCandidateId"] = "genes-two"
    with pytest.raises(ValueError):
        validate_comparison_review(coverage, review)


def test_changed_gene_sets_are_matched_only_on_the_declared_axis() -> None:
    coverage = comparison_review()["comparisonCoverage"]
    left = coverage["candidateSettings"]["baseline"]
    right = deepcopy(left)
    right["hvgCount"] = 2000
    right["features"] = {"distinct": "genes"}
    assert set(setting_changes(left, right)) == {"hvgCount"}
    right["ranking"] = "batchAware"
    right["rankingColumn"] = "capture"
    assert set(setting_changes(left, right)) == {"hvgCount", "hvgRanking"}


def test_old_checkpoints_fail_with_actionable_contract_error() -> None:
    review = comparison_review()
    review.pop("comparisonConclusions")
    with pytest.raises(ValueError, match="start a new workflow"):
        rna_tuning.validate_completed_comparison_evidence(review)


def test_better_alternative_requires_exact_observed_tradeoff() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    coverage["candidateSettings"]["genes-two"]["metrics"]["seedStability"] = 0.99
    with pytest.raises(ValueError, match="tradeoffs.*seedStability"):
        validate_comparison_review(coverage, review)
    conclusion = next(
        row for row in review["comparisonConclusions"] if row["axis"] == "hvgCount"
    )
    conclusion["tradeoffs"] = [
        {
            "alternativeCandidateId": "genes-two",
            "metric": "seedStability",
            "preferredValue": 0.92,
            "alternativeValue": 0.99,
            "interpretation": "The measured gain requires weighing preserved marker programs against extra selected genes.",
        }
    ]
    validate_comparison_review(coverage, review)
    conclusion["tradeoffs"][0]["alternativeValue"] = 1.0
    with pytest.raises(ValueError, match="exact preferred"):
        validate_comparison_review(coverage, review)


def test_marker_poor_selected_population_needs_explicit_objective_resolution() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    coverage["candidateSettings"]["candidate-two"]["metrics"]["topMarkerGenes"][
        "1"
    ] = []
    with pytest.raises(ValueError, match="without qualifying markers"):
        validate_comparison_review(coverage, review)
    review["populationConcerns"] = [
        {
            "candidateId": "candidate-two",
            "clusterId": "1",
            "status": "unresolvedEssential",
            "evidenceIds": review["evidenceIds"],
            "explanation": "This is the target population and its identity is unsupported.",
        }
    ]
    with pytest.raises(ValueError, match="essential population"):
        validate_comparison_review(coverage, review)
    review["populationConcerns"][0]["status"] = "nonEssentialLimitation"
    review["populationConcerns"][0]["explanation"] = (
        "Retain this unlabelled background group without assigning a cell identity."
    )
    validate_comparison_review(coverage, review)


def test_generic_not_applicable_and_off_axis_preferences_cannot_pass() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    row = next(row for row in coverage["comparisons"] if row["axis"] == "pca")
    row.update(
        status="notApplicable",
        reason="Defaults look sufficient",
        alternativeCandidateId=None,
    )
    with pytest.raises(ValueError, match="observed proof"):
        validate_comparison_review(coverage, review)
    review = comparison_review()
    conclusion = next(
        row for row in review["comparisonConclusions"] if row["axis"] == "pca"
    )
    conclusion["candidateIds"].append("genes-two")
    conclusion["preferredCandidateId"] = "genes-two"
    with pytest.raises(ValueError, match="observed candidate"):
        validate_comparison_review(review["comparisonCoverage"], review)


def test_equivalent_core_feature_masks_can_have_distinct_artifact_operations() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    baseline = coverage["candidateSettings"]["baseline"]
    alternative = deepcopy(baseline)
    alternative["features"] = {**baseline["features"], "artifactId": "7" * 64}
    row = next(
        row for row in coverage["comparisons"] if row["comparisonId"] == "hvgCount:2000"
    )
    row.update(
        status="notApplicable",
        alternativeCandidateId=None,
        reason="The exact observed selected-gene masks are equal.",
        observedProof={
            "kind": "identicalSelectedGenes",
            "baselineFeatures": baseline["features"],
            "verifiedEqualMasks": True,
            "alternativeSetting": alternative,
        },
    )
    validate_comparison_review(coverage, review)


@pytest.fixture
def panel_run(monkeypatch: pytest.MonkeyPatch) -> rna_tuning.RnaTuningRun:
    saved: dict[str, dict] = {}
    monkeypatch.setattr(journal, "_ensure_orchestration_store", lambda store: "test")

    def load(
        store: Any, prefix: str, workflow: str, key: str, inputs: Any = None
    ) -> Any:
        row = saved.get(key)
        if row is None:
            return None
        assert inputs is None or inputs == row["inputs"]
        return deepcopy(row["outputs"])

    def save(
        store: Any, prefix: str, workflow: str, key: str, inputs: Any, outputs: Any
    ) -> Any:
        row = {"inputs": deepcopy(inputs), "outputs": deepcopy(outputs)}
        assert key not in saved or saved[key] == row
        saved[key] = row
        return deepcopy(outputs)

    monkeypatch.setattr(journal, "load_checkpoint", load)
    monkeypatch.setattr(
        journal,
        "read_checkpoint",
        lambda store, prefix, workflow, key: deepcopy(saved.get(key)),
    )
    monkeypatch.setattr(journal, "save_checkpoint", save)
    handoff = example(PreprocessedAssayHandoff)
    handoff.nCells = 100
    handoff.nFeatures = 1000
    handoff.graphFeatureCandidates = {
        "eligibleDefault": handoff.graphFeatures,
        "eligibleAll": handoff.graphFeatures,
    }
    masks = {handoff.graphFeatures.artifactId: np.arange(5000) < 1000}
    store = SimpleNamespace(
        zw=None,
        load_artifact=lambda ref: {"values": masks[ref.artifact_id]},
        inspect_artifact=lambda ref: SimpleNamespace(exists=True, complete=True),
    )
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=object()),
        store,
        SimpleNamespace(workflowRunId="panel"),
        SimpleNamespace(config=AutomatedWorkflowConfig()),
        example(AutomatedPreprocessingPlan),
        handoff,
        StudyContract.get_blank(),
        {},
        {},
    )
    run.batch_columns = ["capture"]
    run.scope_sizes["sample0"] = 100
    monkeypatch.setattr(
        rna_tuning, "read_stored_selection_indices", lambda *a, **kw: np.arange(100)
    )
    monkeypatch.setattr(
        rna_tuning,
        "read_metadata_rows_chunkwise",
        lambda *a, **kw: np.repeat(["a", "b"], 50),
    )
    store.cells = SimpleNamespace()
    monkeypatch.setattr(
        run,
        "_feature_nomination",
        lambda setting: {"parameter": "excludeFeature", "value": "nominated"},
    )

    def prepared(
        scope: str, key: str, baseline: Any, experiment: dict, cells: Any
    ) -> Any:
        setting = run.settings[baseline.candidateId]
        field = experiment["parameter"]
        number = {
            "hvgCount": int(experiment["value"]) if field == "hvgCount" else 1000,
            "hvgRanking": 1000,
            "excludeFeature": 1000,
        }[field]
        identity = f"{len(masks) + 1:064x}"
        mask = np.arange(5000) < number
        updates: dict[str, Any] = {"hvgCount": number}
        if field != "hvgCount":
            mask[[0, number]] = [False, True]
        if field == "hvgRanking":
            updates.update(ranking="batchAware", rankingColumn="capture")
        if field == "excludeFeature":
            updates["eligibleFeatures"] = setting.eligibleFeatures.model_copy(
                update={"artifactId": "f" * 64}
            )
        masks[identity] = mask
        updates["features"] = setting.features.model_copy(
            update={"artifactId": identity}
        )
        return setting.model_copy(update=updates)

    monkeypatch.setattr(run, "_prepared_setting", prepared)

    def execute(scope: str, cells: Any, setting: Any) -> Any:
        inputs = run.execution_inputs(cells, setting)
        admission = run.budget.admit(scope, inputs)
        candidate_id = "rna_" + candidate_identity(inputs)[:24]
        old = next(
            (
                item
                for item in run.evaluations[scope]
                if item.candidateId == candidate_id
            ),
            None,
        )
        if old is not None:
            return old
        parameters = setting.parameters.model_copy(update={"candidateId": candidate_id})
        run.settings[candidate_id] = setting.model_copy(
            update={"parameters": parameters}
        )
        evaluation = example(ParameterCandidateEvaluation)
        evaluation.candidateId = candidate_id
        evaluation.parameters = parameters
        evaluation.status = "done"
        evaluation.cellSelection = ArtifactReferenceModel.from_artifact_ref(cells)
        evaluation.artifacts["graphFeatures"] = ArtifactRecord.model_validate(
            setting.features.model_dump()
        )
        run.evaluations[scope].append(evaluation)
        run.budget.complete(
            admission, {"evaluation": evaluation.model_dump(mode="json")}
        )
        return evaluation

    monkeypatch.setattr(run, "execute", execute)
    return run


def test_required_panel_executes_twelve_rows_and_holds_other_settings_fixed(
    panel_run: rna_tuning.RnaTuningRun,
) -> None:
    run = panel_run
    baseline = run._resolution_panel("sample0", run.cells, run.baseline())
    run.resolution_candidates.clear()
    run._sensitivity_panel("sample0", run.cells, baseline)
    assert len(run.evaluations["sample0"]) == 12
    assert run.budget.summary()["scopes"]["sample0"]["completed"] == {
        "graphs": 9,
        "partitions": 12,
    }
    coverage = run.comparison_coverage("sample0", run.cells)
    assert len(coverage["comparisons"]) == 11
    for row in coverage["comparisons"]:
        assert row["status"] == "completed"
        assert set(
            setting_changes(
                coverage["candidateSettings"][row["baselineCandidateId"]],
                coverage["candidateSettings"][row["alternativeCandidateId"]],
            )
        ) == {row["axis"]}
    assert {
        (item.parameters.dimensions, item.parameters.neighborsK)
        for item in run.evaluations["sample0"]
    } == {(21, 11), (10, 11), (30, 11), (21, 21), (21, 41)}


def test_full_validation_reuses_exact_all_cell_discovery_without_charge(
    panel_run: rna_tuning.RnaTuningRun,
) -> None:
    run = panel_run
    baseline = run.execute("sample0", run.cells, run.baseline())
    observed = rna_tuning.RnaTuningRun.execute(
        run, "full", run.cells, run.settings[baseline.candidateId]
    )
    assert observed == baseline
    assert run.budget.summary()["scopes"]["full"]["reserved"] == {
        "graphs": 0,
        "partitions": 0,
    }
    assert run.validation_sources[baseline.candidateId]["scope"] == "sample0"


def test_full_fallback_declines_unfinishable_panel_before_additional_work(
    panel_run: rna_tuning.RnaTuningRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = panel_run
    run.scope_sizes["full"] = 100
    baseline = run._resolution_panel("full", run.cells, run.baseline())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Feature preparation must follow admission of the required panel")

    monkeypatch.setattr(run, "_prepared_setting", forbidden)
    with pytest.raises(CandidateBudgetExceeded):
        run._sensitivity_panel("full", run.cells, baseline)
    assert run.budget.summary()["scopes"]["full"]["completed"] == {
        "graphs": 1,
        "partitions": 4,
    }


def test_feature_policy_waits_for_evidence_nomination_before_combining(
    panel_run: rna_tuning.RnaTuningRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = panel_run
    monkeypatch.setattr(run, "_feature_nomination", lambda setting: None)
    monkeypatch.setattr(
        run,
        "_feature_experiments",
        lambda setting: {
            "excludeFeature:nominated": {
                "parameter": "excludeFeature",
                "value": "nominated",
                "affectedEligibleGenes": 1,
            }
        },
    )
    baseline = run._resolution_panel("sample0", run.cells, run.baseline())
    run.resolution_candidates.clear()
    run._sensitivity_panel("sample0", run.cells, baseline)
    coverage = run.comparison_coverage("sample0", run.cells)
    row = next(row for row in coverage["comparisons"] if row["axis"] == "featurePolicy")
    assert row["status"] == "pending"
    with pytest.raises(ValueError, match="nomination and execution"):
        validate_comparison_review(coverage, {"action": "combine"})


def test_completed_feature_comparisons_replay_at_exact_candidate_limit(
    panel_run: rna_tuning.RnaTuningRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = panel_run
    run.request.config.maxScreeningEvaluations = 12
    prepare = run._prepared_setting

    def persisted(
        scope: str, key: str, baseline: Any, experiment: dict, cells: Any
    ) -> Any:
        checkpoint = f"parameter_tuning/{scope}/{key}/setting"
        inputs = {
            "baseline": run.settings[baseline.candidateId].model_dump(mode="json"),
            "experiment": experiment,
            "cells": run.cells.to_dict(),
        }
        saved = journal.load_checkpoint(
            run.store, run.prefix, run.workflow.workflowRunId, checkpoint, inputs
        )
        if saved is not None:
            return rna_tuning.RnaSetting.model_validate(saved["setting"])
        setting = prepare(scope, key, baseline, experiment, cells)
        journal.save_checkpoint(
            run.store,
            run.prefix,
            run.workflow.workflowRunId,
            checkpoint,
            inputs,
            {"setting": setting.model_dump(mode="json")},
        )
        return setting

    monkeypatch.setattr(run, "_prepared_setting", persisted)
    baseline = run._resolution_panel("sample0", run.cells, run.baseline())
    run.resolution_candidates.clear()
    run._sensitivity_panel("sample0", run.cells, baseline)
    before = run.budget.summary()
    assert before["scopes"]["sample0"]["completed"]["partitions"] == 12
    observed = run.comparison_coverage("sample0", run.cells)
    run._sensitivity_panel("sample0", run.cells, baseline)
    assert run.budget.summary() == before
    assert run.comparison_coverage("sample0", run.cells) == observed


def test_evaluated_ranking_matches_both_mode_and_column(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    import json

    from tests.test_agent_rna_evidence_mode import assess, make_run, comparison_coverage

    request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))
    run.batch_columns = ["capture", "other_capture"]
    other = selected.model_copy(deep=True)
    other.candidateId = "batch-aware-observed"
    other.parameters.candidateId = other.candidateId
    other.artifacts["graphFeatures"] = ArtifactRecord.model_validate(
        {
            **run.settings[selected.candidateId].features.model_dump(),
            "artifactId": "d" * 64,
        }
    )
    run.evaluations["full"].append(other)
    run.settings[other.candidateId] = run.settings[selected.candidateId].model_copy(
        update={
            "parameters": other.parameters,
            "features": other.artifacts["graphFeatures"],
            "ranking": "batchAware",
            "rankingColumn": "capture",
        }
    )
    coverage = comparison_coverage(run, "full")
    row = next(row for row in coverage["comparisons"] if row["axis"] == "hvgRanking")
    row.update(status="completed", alternativeCandidateId=other.candidateId)
    monkeypatch.setattr(run, "comparison_coverage", lambda *a: coverage)

    def inspect(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert "hvgRanking:batchAware:capture" not in evidence["experiments"]
        assert "hvgRanking:batchAware:other_capture" in evidence["experiments"]
        assert (
            evidence["assessmentContext"]["alreadyEvaluatedExperiments"][
                "hvgRanking:batchAware:capture"
            ]
            == other.candidateId
        )
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", inspect)
    assert run.review("full", 0, selected, {}).action == "accept"
