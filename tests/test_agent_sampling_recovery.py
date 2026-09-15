"""Bounded sampling and recovery preserve exact scientific work."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from scarf.agent.config import AgentRunConfig
from scarf.agent.config.agent_exec import run_agent_sync
from scarf.agent.orchestrator import rna_tuning
from scarf.agent.orchestrator.main import AgentOrchestrator
from scarf.agent.orchestrator.models import AutomatedWorkflowConfig
from scarf.agent.parameter_tuning.comparisons import (
    bind_comparison_measurements,
    comparison_advantages,
    validate_comparison_review,
)
from tests.agent_comparison_examples import comparison_review
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401
from tests.test_agent_rna_evidence_mode import assess, make_run


@pytest.mark.parametrize(
    ("population", "expected"),
    [
        (1, (1,)),
        (8000, (8000,)),
        (10_000, (10_000,)),
        (62_721, (10_000, 62_721)),
        (100_001, (10_001, 100_000)),
        (621_200, (62_120, 100_000)),
        (1_000_000, (100_000,)),
        (10_000_000, (100_000,)),
    ],
)
def test_automatic_sampling_bounds(population: int, expected: tuple[int, ...]) -> None:
    assert rna_tuning.screening_sizes(population, AutomatedWorkflowConfig()) == expected


def test_fixed_saved_policy_and_explicit_resume_overrides() -> None:
    original = AutomatedWorkflowConfig(screeningCells=50_000)
    wire = original.model_dump_json()
    restored = AutomatedWorkflowConfig.model_validate_json(wire)
    assert restored.model_dump_json() == wire
    assert rna_tuning.screening_sizes(62_721, restored) == (50_000, 62_721)
    assert rna_tuning.screening_sizes(621_200, restored) == (50_000, 100_000)
    AgentOrchestrator(object())._validate_resume_config(restored)
    with pytest.raises(ValueError, match="execution settings differ"):
        AgentOrchestrator(
            object(), config=AutomatedWorkflowConfig(screeningCells=10_000)
        )._validate_resume_config(restored)
    with pytest.raises(ValueError, match="non-empty"):
        rna_tuning.screening_sizes(0, restored)


def test_inventory_exposes_small_advantages_and_exact_values() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    baseline = coverage["candidateSettings"]["baseline"]["metrics"]
    baseline["macroF1"] = 0.8
    alternative = coverage["candidateSettings"]["genes-two"]["metrics"]
    alternative["seedStability"] = baseline["seedStability"] + 1e-10
    alternative["macroF1"] = 0.999
    rows = [
        row
        for row in comparison_advantages(coverage)
        if row["preferredCandidateId"] == "baseline"
        and row["alternativeCandidateId"] == "genes-two"
    ]
    assert {row["metric"] for row in rows} == {"seedStability", "macroF1"}
    assert (
        next(row for row in rows if row["metric"] == "seedStability")["difference"] > 0
    )
    with pytest.raises(ValueError) as failure:
        validate_comparison_review(coverage, review)
    for required in ("hvgCount", "baseline", "genes-two", "seedStability", "macroF1"):
        assert required in str(failure.value)


def test_model_repairs_multiple_missing_tradeoffs_without_transcribing_values() -> None:
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    review = comparison_review()
    coverage = review["comparisonCoverage"]
    coverage["candidateSettings"]["baseline"]["metrics"]["macroF1"] = 0.8
    coverage["candidateSettings"]["genes-two"]["metrics"].update(
        seedStability=0.99, macroF1=0.999
    )
    action = {
        key: value
        for key, value in review.items()
        if key in rna_tuning.TuningAction.model_fields
    }
    for conclusion in action["comparisonConclusions"]:
        conclusion.setdefault("tradeoffs", [])
    output_type = rna_tuning._assessment_output_type(
        list(coverage["candidateSettings"]), [], scope="full"
    )
    schema = output_type.model_json_schema()
    tradeoff_schema = schema["$defs"]["ObservedTradeoffInterpretation"]["properties"]
    assert "preferredValue" not in tradeoff_schema
    assert "alternativeValue" not in tradeoff_schema
    calls = []

    def provider(messages: Any, info: Any) -> Any:
        proposed = deepcopy(action)
        calls.append(messages)
        if len(calls) == 2:
            feedback = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            ]
            assert "genes-two" in str(feedback) and "macroF1" in str(feedback)
            conclusion = next(
                row
                for row in proposed["comparisonConclusions"]
                if row["axis"] == "hvgCount"
            )
            conclusion["tradeoffs"] = [
                {
                    "alternativeCandidateId": "genes-two",
                    "metric": metric,
                    "interpretation": "This observed advantage must be weighed against the retained population marker programs.",
                }
                for metric in ("seedStability", "macroF1")
            ]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, proposed)])

    def validate(proposed: Any) -> Any:
        result = rna_tuning.TuningAction.model_validate(
            bind_comparison_measurements(coverage, proposed.model_dump(mode="json"))
        )
        validate_comparison_review(coverage, result.model_dump(mode="json"))
        return result

    result = run_agent_sync(
        model=FunctionModel(provider),
        output_type=output_type,
        system_prompt="Explain the supplied comparisons.",
        user_prompt="Saved evidence",
        config=AgentRunConfig(retries=2),
        output_validator=validate,
    )
    assert len(calls) == 2
    rows = next(
        row.tradeoffs
        for row in result.output.comparisonConclusions
        if row.axis == "hvgCount"
    )
    assert {row.metric: row.alternativeValue for row in rows} == {
        "seedStability": 0.99,
        "macroF1": 0.999,
    }


@pytest.mark.usefixtures("memory_checkpoints")
@pytest.mark.parametrize("visual", [False, True])
def test_uncommitted_review_reuses_prepared_evidence(
    monkeypatch: pytest.MonkeyPatch, visual: bool
) -> None:
    from scarf.agent.orchestrator import tuning

    model = object() if visual else SimpleNamespace(supports_image_input=False)
    run, selected = make_run(monkeypatch, model)
    calls = []

    def fail(**kwargs: Any) -> Any:
        calls.append(kwargs["user_prompt"])
        raise RuntimeError("Interrupted during provider call")

    monkeypatch.setattr(rna_tuning, "run_agent_sync", fail)
    with pytest.raises(RuntimeError, match="Interrupted"):
        run.review("full", 0, selected, {})
    resumed, selected = make_run(monkeypatch, model)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "Committed evidence must precede diagnostic, feature and image work"
        )

    monkeypatch.setattr(resumed, "feature_evidence", forbidden)
    monkeypatch.setattr(tuning, "_analysis_visual_content", forbidden)
    monkeypatch.setattr(rna_tuning, "population_support_evidence", forbidden)
    monkeypatch.setattr(rna_tuning, "partition_comparison_evidence", forbidden)
    monkeypatch.setattr(rna_tuning, "_neighbor_overlap", forbidden)

    def recovered(**kwargs: Any) -> Any:
        if visual:
            assert kwargs["user_prompt"][0] == calls[0][0]
        else:
            assert kwargs["user_prompt"] == calls[0]
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", recovered)
    assert resumed.review("full", 0, selected, {}).action == "accept"


def test_binding_rejects_invented_values_and_unknown_advantages() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    coverage["candidateSettings"]["genes-two"]["metrics"]["seedStability"] = 0.99
    conclusion = next(
        row for row in review["comparisonConclusions"] if row["axis"] == "hvgCount"
    )
    conclusion["tradeoffs"] = [
        {
            "alternativeCandidateId": "genes-two",
            "metric": "seedStability",
            "interpretation": "Observed tradeoff",
            "alternativeValue": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="exact preferred"):
        bind_comparison_measurements(coverage, review)
    conclusion["tradeoffs"][0]["alternativeCandidateId"] = "invented"
    with pytest.raises(ValueError, match="observed advantage"):
        bind_comparison_measurements(coverage, review)


@pytest.mark.usefixtures("memory_checkpoints")
@pytest.mark.parametrize("joint_revision", [False, True])
def test_context_revision_preserves_primary_artifacts_and_admissions(
    monkeypatch: pytest.MonkeyPatch, joint_revision: bool
) -> None:
    from scarf.agent.orchestrator.budget import CandidateBudget, candidate_identity

    run, old_evaluation = make_run(monkeypatch, object())
    original = {"studyContract": run.study.model_dump(mode="json")}
    run.provenance = original
    run.budget = CandidateBudget(
        run.store, run.prefix, run.workflow.workflowRunId, run.request.config, original
    )
    setting = run.baseline()
    admission = run.budget.admit("sample0", run.execution_inputs(run.cells, setting))
    identifier = (
        "rna_" + candidate_identity(run.execution_inputs(run.cells, setting))[:24]
    )
    old_evaluation.candidateId = identifier
    old_evaluation.parameters = setting.parameters.model_copy(
        update={"candidateId": identifier}
    )
    old_evaluation.artifacts["normalized"] = run.handoff.normalized
    run.budget.complete(
        admission, {"evaluation": old_evaluation.model_dump(mode="json")}
    )
    revised = run.study.model_copy(
        update={"protectedCombinations": [["sex", "condition"]]}
        if joint_revision
        else {
            "studyContext": run.study.studyContext
            + " Additional design evidence is now measured."
        }
    )
    provenance = {"studyContract": revised.model_dump(mode="json")}
    resumed = rna_tuning.RnaTuningRun(
        run.owner,
        run.store,
        run.workflow,
        run.request,
        run.plan,
        run.handoff,
        revised,
        {},
        provenance,
        previous_provenances=[original],
    )
    counter = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "A context revision must not recompute the primary graph, stability or doublets"
        )

    def refresh(deps: Any, evaluation: Any) -> Any:
        counter.append(deps.protectedCombinations)
        updated = evaluation.model_copy(deep=True)
        updated.metrics.biologicalPreservation['joint:["sex","condition"]'] = {
            "clisi": 0.8
        }
        return updated

    monkeypatch.setattr(
        rna_tuning,
        "prepare_parameter_tuning_dependencies",
        lambda *a, **kw: (SimpleNamespace(), []),
    )
    monkeypatch.setattr(rna_tuning, "refresh_candidate_design_evidence", refresh)
    monkeypatch.setattr(rna_tuning, "execute_parameter_candidate", forbidden)
    monkeypatch.setattr(rna_tuning, "score_advisory_doublets", forbidden)
    monkeypatch.setattr(rna_tuning, "augment_pca_evaluations", forbidden)
    monkeypatch.setattr(rna_tuning, "augment_cluster_evaluations", forbidden)
    result = resumed.execute("sample0", resumed.cells, setting)
    assert (
        result.model_dump(mode="json")["artifacts"]
        == old_evaluation.model_dump(mode="json")["artifacts"]
    )
    assert len(counter) == int(joint_revision)
    assert resumed.budget.summary()["scopes"]["sample0"]["reserved"]["partitions"] == 1
    assert resumed.budget.completed(admission)[
        "evaluation"
    ] == old_evaluation.model_dump(mode="json")
    assert resumed.execute("sample0", resumed.cells, setting) == result
    assert len(counter) == int(joint_revision)


from tests.test_agent_required_comparisons import panel_run  # noqa: E402, F401


@pytest.mark.parametrize("harmony", [False, True])
def test_targeted_recovery_admits_complete_resolution_panel_and_controls(
    request: pytest.FixtureRequest, harmony: bool
) -> None:
    from scarf.agent.orchestrator.budget import CandidateBudgetExceeded

    run = request.getfixturevalue("panel_run")
    run.recovery_scope = "sample1"
    run.study = run.study.model_copy(update={"correctionLicense": "safe"})
    setting = run.baseline()
    setting.parameters.useHarmony = harmony
    run._resolution_panel("full", run.cells, setting)
    counts = run.budget.summary()["scopes"]["full"]["completed"]
    assert counts == {"graphs": 2 if harmony else 1, "partitions": 8 if harmony else 4}
    repair = setting.model_copy(
        update={"parameters": setting.parameters.model_copy(update={"dimensions": 30})}
    )
    if harmony:
        with pytest.raises(CandidateBudgetExceeded):
            run._resolution_panel("full", run.cells, repair)
        assert run.budget.summary()["scopes"]["full"]["completed"] == counts
    else:
        run._resolution_panel("full", run.cells, repair)
        assert run.budget.summary()["scopes"]["full"]["completed"] == {
            "graphs": 2,
            "partitions": 8,
        }
        with pytest.raises(CandidateBudgetExceeded):
            run._resolution_panel(
                "full",
                run.cells,
                setting.model_copy(
                    update={
                        "parameters": setting.parameters.model_copy(
                            update={"dimensions": 10}
                        )
                    }
                ),
            )


@pytest.mark.usefixtures("memory_checkpoints")
def test_revised_feature_preparation_reuses_exact_inputs_but_not_other_ranking_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.orchestrator import journal

    run, _ = make_run(monkeypatch, object())
    run.study.technicalBatchColumns = ["old_batch"]
    run.batch_columns = ["old_batch"]
    original = {"studyContract": run.study.model_dump(mode="json")}
    baseline = run.baseline()
    experiment = {"parameter": "hvgRanking", "value": "batchAware"}
    key, inputs, saved = run._setting_checkpoint(
        "sample0", "sensitivity/hvgRanking", baseline, experiment
    )
    assert saved is None
    journal.save_checkpoint(
        run.store,
        run.prefix,
        run.workflow.workflowRunId,
        key,
        inputs,
        {"setting": baseline.model_dump(mode="json")},
    )
    run.previous_provenances = (original,)
    run.evidence_revision = "a" * 64
    assert (
        run._setting_checkpoint(
            "sample0", "sensitivity/hvgRanking", baseline, experiment
        )[2]
        is not None
    )
    run.batch_columns = ["new_batch"]
    revised_key, revised_inputs, saved = run._setting_checkpoint(
        "sample0", "sensitivity/hvgRanking", baseline, experiment
    )
    assert saved is None and revised_key != key
    assert revised_inputs["rankingColumns"] == ["new_batch"]
    assert journal.load_checkpoint(
        run.store, run.prefix, run.workflow.workflowRunId, key, inputs
    )["setting"] == baseline.model_dump(mode="json")
    different = {"parameter": "excludeFeature", "value": "MT-CO1"}
    assert (
        run._setting_checkpoint(
            "sample0", "sensitivity/hvgRanking", baseline, different
        )[2]
        is None
    )


@pytest.mark.usefixtures("memory_checkpoints")
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("unknownCandidate", "observed candidate"),
        ("missingCandidateCitation", "selected numerical evidence"),
        ("unknownExperiment", "Unknown experiment ID"),
        ("safeNotApplicable", "observed necessity assessment"),
        ("confoundedNotNeeded", "confounds the batch columns"),
        ("uncertain", "further evidence or deferral"),
        ("ineligible", "failed required full-cell checks"),
        ("missingUnits", "independent-unit support"),
        ("unsupportedProtection", "protection is unsupported"),
        ("neededNative", "required correction unresolved"),
        ("indeterminate", "authorization remains indeterminate"),
        ("harmonyRejected", "Harmony acceptance failed"),
    ],
)
def test_assessment_cannot_hide_unresolved_design_or_scientific_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str, reason: str
) -> None:
    run, selected = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))
    if failure == "safeNotApplicable":
        run.study.correctionLicense = "safe"
    elif failure == "confoundedNotNeeded":
        run.study.correctionLicense = "unsafeConfounded"
    elif failure == "ineligible":
        selected.eligible = False
    elif failure == "missingUnits":
        run.study.independentUnitColumns = ["donor"]
        selected.metrics.crossUnitSupport = None
    elif failure == "unsupportedProtection":
        run.study.unsupportedProtection = ["condition/sex"]
    elif failure == "indeterminate":
        run.study.correctionLicense = "indeterminate"
    elif failure == "harmonyRejected":
        monkeypatch.setattr(
            run,
            "harmony_gate",
            lambda *args: (False, ["Matched doublet agreement is missing"]),
        )

    def invalid(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        if failure == "unknownCandidate":
            action.selectedCandidateId = "unobserved"
        elif failure == "missingCandidateCitation":
            action.evidenceIds = ["featureEvidence"]
        elif failure == "unknownExperiment":
            action.action = "experiment"
            action.experimentId = "nearly-the-offered-id"
            action.concern = "A supported population may merge"
            action.expectedImprovement = "Test membership at another neighborhood size"
        elif failure == "confoundedNotNeeded":
            action.correctionNeed = "notNeeded"
        elif failure == "uncertain":
            action.correctionNeed = "uncertain"
        elif failure in {"unsupportedProtection", "neededNative"}:
            action.correctionNeed = "needed"
        return kwargs["output_validator"](action)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", invalid)
    with pytest.raises(ValueError, match=reason):
        run.review("full", 0, selected, {})
    assert not any("review" in row for row in run.history)
