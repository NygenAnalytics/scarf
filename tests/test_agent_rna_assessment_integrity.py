"""Scientific choice guards preserve exact observed comparisons and prior reviews."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent.orchestrator import rna_tuning
from scarf.agent.experimental_context.contracts import (
    CovariateComparison,
    CovariateProposal,
)
from scarf.agent.experimental_context.study import unsupported_comparison_limitations
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from tests.agent_comparison_examples import observed_action
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401
from tests.test_agent_rna_evidence_mode import assess, make_run


pytestmark = pytest.mark.usefixtures("memory_checkpoints")


def _run(
    monkeypatch: pytest.MonkeyPatch, *, confounded: bool = False
) -> tuple[rna_tuning.RnaTuningRun, ParameterCandidateEvaluation]:
    run, selected = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))
    run.study = run.study.model_copy(
        update={
            "independentUnitColumns": ["donor"],
            "correctionLicense": "unsafeConfounded" if confounded else "notApplicable",
            "limitations": ["Library and protected tissue are confounded."]
            if confounded
            else [],
        }
    )
    selected.metrics.crossUnitSupport = 0.9
    return run, selected


@pytest.mark.parametrize(
    ("retries", "request_limit", "repair", "expected_calls"),
    [
        (5, 10, False, 3),
        (1, 10, False, 2),
        (0, 10, False, 1),
        (5, 2, False, 2),
        (5, 10, True, 2),
    ],
)
def test_review_repairs_are_bounded_without_changing_saved_config_or_evidence(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    retries: int,
    request_limit: int,
    repair: bool,
    expected_calls: int,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    calls = []
    key = "parameter_tuning/full/review0"

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Output repair must not repeat candidate or diagnostic work")

    def provider(messages: Any, info: Any) -> ModelResponse:
        evidence = json.loads(
            next(
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart)
            )
        )
        prepared = saved[f"{key}/evidence/structured"]
        assert prepared["outputs"]["evidence"] == evidence
        calls.append(json.dumps(prepared, sort_keys=True))
        # Prepared evidence must also be reused across the SDK's repair attempts.
        monkeypatch.setattr(run, "execute", forbidden)
        monkeypatch.setattr(run, "_augment_evaluation", forbidden)
        monkeypatch.setattr(run, "feature_evidence", forbidden)
        monkeypatch.setattr(run, "comparison_coverage", forbidden)
        monkeypatch.setattr(rna_tuning, "population_support_evidence", forbidden)
        monkeypatch.setattr(rna_tuning, "partition_comparison_evidence", forbidden)
        monkeypatch.setattr(rna_tuning, "_neighbor_overlap", forbidden)
        action = observed_action(evidence)
        if not repair or len(calls) == 1:
            conclusion = next(
                c for c in action["comparisonConclusions"] if c["axis"] == "partition"
            )
            conclusion["tradeoffs"] = [
                {
                    "alternativeCandidateId": "resolution-half",
                    "metric": "seedStability",
                    "interpretation": "The stability measurements tie, so neither has an advantage.",
                }
            ]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, action)])

    model = FunctionModel(provider, profile={"supports_image_input": False})
    run, selected = make_run(monkeypatch, model)
    run.request.config.agentRunConfig = run.request.config.agentRunConfig.model_copy(
        update={"retries": retries, "requestLimit": request_limit}
    )
    original_config = run.request.config.model_dump_json()
    if repair:
        assert run.review("full", 0, selected, {}).action == "accept"
        assert key in saved
    else:
        with pytest.raises((UnexpectedModelBehavior, UsageLimitExceeded)):
            run.review("full", 0, selected, {})
        assert key not in saved
    assert len(calls) == expected_calls
    assert len(set(calls)) == 1
    assert run.request.config.model_dump_json() == original_config
    attempts = [
        record["outputs"]["runInfo"]
        for name, record in saved.items()
        if name.startswith(f"{key}/model_attempts/")
    ]
    assert len(attempts) == 1
    assert attempts[0]["usage"]["requests"] == expected_calls
    assert attempts[0]["status"] == ("done" if repair else "failed")
    assert len(attempts[0]["validationRetries"]) == (1 if repair else expected_calls)


def _experiment_response(kwargs: dict[str, Any], correction_need: str) -> Any:
    action = assess(**{**kwargs, "output_validator": lambda value: value}).output
    evidence = json.loads(kwargs["user_prompt"])
    action = action.model_copy(
        update={
            "action": "experiment",
            "experimentId": next(iter(evidence["experiments"])),
            "correctionNeed": correction_need,
            "concern": "Assess sensitivity of population support to one setting.",
            "expectedImprovement": "The comparison may improve population support.",
            "rationale": "Request one observed-evidence-driven parameter comparison.",
        }
    )
    return SimpleNamespace(output=kwargs["output_validator"](action))


@pytest.mark.parametrize("correction_need", ["needed", "notNeeded"])
def test_confounded_design_rejects_identifiable_correction_claims_before_commit(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    correction_need: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = _run(monkeypatch, confounded=True)
    monkeypatch.setattr(
        rna_tuning,
        "run_agent_sync",
        lambda **kwargs: _experiment_response(kwargs, correction_need),
    )
    with pytest.raises(ValueError, match="design confounds"):
        run.review("full", 0, selected, {})
    assert "parameter_tuning/full/review0" not in saved


def test_confounded_design_can_accept_supported_native_descriptive_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, selected = _run(monkeypatch, confounded=True)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    action = run.review("full", 0, selected, {})
    assert action.action == "accept"
    assert action.correctionNeed == "notApplicable"
    assert not selected.parameters.useHarmony


@pytest.mark.parametrize("correction_need", ["needed", "notNeeded"])
def test_historical_confounded_experiment_replays_with_a_scientific_warning(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    correction_need: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = _run(monkeypatch, confounded=True)
    monkeypatch.setattr(
        rna_tuning,
        "run_agent_sync",
        lambda **kwargs: _experiment_response(kwargs, "notApplicable"),
    )
    run.review("full", 0, selected, {})
    key = "parameter_tuning/full/review0"
    # Represent a committed action produced before this scientific guard existed.
    saved[key]["outputs"]["action"]["correctionNeed"] = correction_need
    expected = rna_tuning.TuningAction.model_validate(saved[key]["outputs"]["action"])
    before = json.dumps(saved[key], sort_keys=True)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "A committed experiment must replay without new model or support work"
        )

    monkeypatch.setattr(rna_tuning, "run_agent_sync", unexpected)
    monkeypatch.setattr(rna_tuning, "population_support_evidence", unexpected)
    assert run.review("full", 0, selected, {}) == expected
    assert json.dumps(saved[key], sort_keys=True) == before
    assert any(
        "claimed identifiable correction necessity" in row.get("reason", "")
        for row in run.history
    )


def _alternative(
    run: rna_tuning.RnaTuningRun,
    selected: ParameterCandidateEvaluation,
    *,
    difference: str = "oneParameter",
) -> tuple[ParameterCandidateEvaluation, str]:
    offered = run.experiments(selected)
    experiment_id, experiment = next(
        (name, value)
        for name, value in offered.items()
        if value["parameter"] == "dimensions"
    )
    alternative = selected.model_copy(deep=True)
    alternative.candidateId = "alternative"
    alternative.parameters.candidateId = alternative.candidateId
    alternative.parameters.dimensions = experiment["value"]
    alternative.evidenceIds = ["candidate:alternative:seedStability"]
    if difference == "twoParameters":
        alternative.parameters.leidenResolution += 0.25
    elif difference == "selection":
        assert alternative.cellSelection is not None
        alternative.cellSelection = alternative.cellSelection.model_copy(
            update={"artifactId": "f" * 64}
        )
    elif difference == "reductionMethod":
        alternative.parameters.reductionMethod = "lsi"
    setting = run.settings[selected.candidateId].model_copy(
        update={"parameters": alternative.parameters}
    )
    if difference == "features":
        setting.features = setting.features.model_copy(update={"artifactId": "e" * 64})
    run.settings[alternative.candidateId] = setting
    run.evaluations["full"].append(alternative)
    return alternative, experiment_id


@pytest.mark.parametrize(
    "difference",
    ["oneParameter", "twoParameters", "selection", "features", "reductionMethod"],
)
def test_matched_comparisons_require_one_change_on_the_exact_representation(
    monkeypatch: pytest.MonkeyPatch, difference: str
) -> None:
    run, selected = _run(monkeypatch)
    alternative, experiment_id = _alternative(run, selected, difference=difference)
    observed: dict[str, Any] = {}

    def record(**kwargs: Any) -> Any:
        observed.update(json.loads(kwargs["user_prompt"]))
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", record)
    run.review("full", 0, selected, {})
    context = observed["assessmentContext"]
    if difference == "oneParameter":
        assert (
            context["alreadyEvaluatedExperiments"][experiment_id]
            == alternative.candidateId
        )
        assert context["matchedComparisons"] == [
            {
                "currentCandidateId": selected.candidateId,
                "alternativeCandidateId": alternative.candidateId,
                "changedParameter": {
                    "pca": {
                        "current": selected.parameters.dimensions,
                        "alternative": alternative.parameters.dimensions,
                    }
                },
                "partitionEvidence": context["matchedComparisons"][0][
                    "partitionEvidence"
                ],
                "basis": context["matchedComparisons"][0]["basis"],
            }
        ]
        partition = context["matchedComparisons"][0]["partitionEvidence"]
        assert partition["matchedCells"] == 100
        assert all(
            row["fractionOutsideLargestMatch"] == 0.0 for row in partition["splits"]
        )
        assert experiment_id not in observed["experiments"]
    else:
        assert not context["matchedComparisons"]
        assert not context["alreadyEvaluatedExperiments"]
        assert experiment_id in observed["experiments"]


def test_completed_numeric_comparison_remains_selectable_without_reexecution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, selected = _run(monkeypatch)
    alternative, experiment_id = _alternative(run, selected)

    def choose_alternative(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert experiment_id not in evidence["experiments"]
        assert alternative.candidateId in {
            row["candidateId"] for row in evidence["candidates"]
        }
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action = action.model_copy(
            update={
                "selectedCandidateId": alternative.candidateId,
                "evidenceIds": [f"candidate:{alternative.candidateId}"],
            }
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", choose_alternative)
    assert (
        run.review("full", 0, selected, {}).selectedCandidateId
        == alternative.candidateId
    )


def test_committed_catalogue_replays_without_new_filtering_or_population_work(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = _run(monkeypatch)
    _alternative_candidate, experiment_id = _alternative(run, selected)
    old_experiment = run.experiments(selected)[experiment_id]
    support_calls = []

    def support(_store: Any, candidate: Any, columns: Any) -> dict[str, Any]:
        support_calls.append(candidate.candidateId)
        return {"candidateId": candidate.candidateId, "columns": list(columns)}

    monkeypatch.setattr(rna_tuning, "population_support_evidence", support)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    expected = run.review("full", 0, selected, {})
    key = "parameter_tuning/full/review0"
    assert len(run.evaluations["full"]) == 2
    assert support_calls == [item.candidateId for item in run.evaluations["full"]]
    assert set(saved[key]["inputs"]["assessmentContext"]["populationSupport"]) == {
        item.candidateId for item in run.evaluations["full"]
    }
    # Prior reviews could offer an already-completed comparison and lacked this context.
    saved[key]["inputs"]["experiments"][experiment_id] = old_experiment
    del saved[key]["inputs"]["assessmentContext"]
    saved[key]["inputs"]["availableEvidenceIds"].remove("assessmentContext")
    before = json.dumps(saved[key], sort_keys=True)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Committed catalogues and support evidence must not be rebuilt")

    monkeypatch.setattr(run, "experiments", unexpected)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", unexpected)
    monkeypatch.setattr(rna_tuning, "population_support_evidence", unexpected)
    assert run.review("full", 0, selected, {}) == expected
    assert json.dumps(saved[key], sort_keys=True) == before


@pytest.mark.parametrize("already_in_study", [False, True])
def test_saved_unsupported_design_evidence_reaches_assessment_and_report_once(
    monkeypatch: pytest.MonkeyPatch, already_in_study: bool
) -> None:
    run, selected = _run(monkeypatch)
    comparison = CovariateComparison(
        proposal=CovariateProposal(
            response="mitochondrialFraction",
            explanatoryColumns=["tissue"],
            observationUnit="sample",
            independentUnit="donor",
            rationale="Assess tissue-associated quality differences across donors.",
        ),
        status="unsupported",
        evidence={"independentUnits": 3, "missingCells": 17},
        reasons=["withinIndependentUnitComparisonsAreUnsupported"],
        evidenceId="designComparison:observedUnsupported",
    )
    run.design_comparisons = (comparison,)
    limitation = unsupported_comparison_limitations([comparison])[0]
    run.study = run.study.model_copy(
        update={
            "evidenceIds": [],
            "limitations": [limitation] if already_in_study else [],
        }
    )
    original_study = run.study.model_dump_json()

    def defer_with_observed_design_evidence(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert evidence["assessmentContext"]["designComparisons"] == [
            comparison.model_dump(mode="json")
        ]
        assert comparison.evidenceId in evidence["availableEvidenceIds"]
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action = action.model_copy(
            update={
                "action": "defer",
                "evidenceIds": [
                    f"candidate:{selected.candidateId}",
                    comparison.evidenceId,
                ],
                "rationale": "Essential design evidence remains unsupported.",
            }
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(
        rna_tuning, "run_agent_sync", defer_with_observed_design_evidence
    )
    assert run.review("full", 0, selected, {}).action == "defer"
    report = run.report(None, "Essential design evidence remains unsupported.")
    assert report.limitations.count(limitation) == 1
    assert comparison.evidenceId in limitation
    assert comparison.reasons[0] in limitation
    assert "no supported association or absence finding" in limitation
    assert run.study.model_dump_json() == original_study


@pytest.mark.parametrize("license", ["safe", "unsafeConfounded"])
def test_uncertain_correction_can_request_harmony_only_with_safe_design(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    license: str,
) -> None:
    run, selected = _run(monkeypatch, confounded=license == "unsafeConfounded")
    run.study = run.study.model_copy(
        update={"correctionLicense": license, "technicalBatchColumns": ["batch"]}
    )
    run.batch_columns = ["batch"]

    def request_harmony(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert ("useHarmony:true" in evidence["experiments"]) is (license == "safe")
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action = action.model_copy(
            update={
                "action": "experiment",
                "experimentId": "useHarmony:true",
                "correctionNeed": "uncertain",
                "concern": "Determine whether observed batch structure is removable.",
                "expectedImprovement": "Compare mixing and protected biology against matched native settings.",
                "rationale": "Request a matched native and Harmony comparison.",
            }
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", request_harmony)
    if license == "safe":
        action = run.review("full", 0, selected, {})
        assert action.action == "experiment"
        assert action.experimentId == "useHarmony:true"
        assert action.correctionNeed == "uncertain"
    else:
        with pytest.raises(ValueError, match="Unknown experiment ID"):
            run.review("full", 0, selected, {})
        saved = request.getfixturevalue("memory_checkpoints")
        assert "parameter_tuning/full/review0" not in saved


@pytest.mark.parametrize(
    "comparison",
    ["missing", "failed", "cells", "features", "parameters", "batch", "matched"],
)
def test_safe_not_needed_requires_an_exact_completed_harmony_comparison(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    comparison: str,
) -> None:
    run, selected = _run(monkeypatch)
    run.study.correctionLicense = "safe"
    run.study.technicalBatchColumns = ["batch"]
    run.batch_columns = ["batch"]
    selected.metrics.batchMixing = {"batch": 0.8}
    if comparison != "missing":
        corrected = selected.model_copy(deep=True)
        corrected.parameters.candidateId = "matched_harmony"
        corrected.candidateId = "matched_harmony"
        corrected.parameters.useHarmony = True
        corrected.harmonyBatchColumns = ["batch"]
        corrected.metrics.batchMixing = {"batch": 0.7}
        setting = run.settings[selected.candidateId].model_copy(deep=True)
        setting.parameters = corrected.parameters
        if comparison == "failed":
            corrected.status = "failed"
        elif comparison == "cells":
            corrected.cellSelection.artifactId = "a" * 64
        elif comparison == "features":
            setting.features.artifactId = "b" * 64
        elif comparison == "parameters":
            corrected.parameters.neighborsK += 1
        elif comparison == "batch":
            corrected.harmonyBatchColumns = ["another_batch"]
        run.evaluations["full"].append(corrected)
        run.settings[corrected.candidateId] = setting

    def prefer_native(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action.correctionNeed = "notNeeded"
        action.quantitativeFindings = [
            "Native batch mixing is 0.8; corrected mixing is 0.7."
        ]
        action.rationale = (
            "The observed Harmony comparison worsens batch mixing; retain native."
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", prefer_native)
    if comparison == "matched":
        passed, _ = run.harmony_gate("full", corrected)
        assert not passed
        action = run.review("full", 0, selected, {})
        assert action.action == "accept"
        assert action.selectedCandidateId == selected.candidateId
        assert action.correctionNeed == "notNeeded"
    else:
        with pytest.raises(
            ValueError, match="requires a completed, matched Harmony experiment"
        ):
            run.review("full", 0, selected, {})
        saved = request.getfixturevalue("memory_checkpoints")
        assert "parameter_tuning/full/review0" not in saved
