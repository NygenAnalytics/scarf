"""Context retries retain measured evidence and cannot erase study questions."""

from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from scarf.agent.experimental_context import tools, validation
from scarf.agent.experimental_context.contracts import (
    CaptureFailureEvidence,
    CellQcPlan,
    CellQcProfileEvidence,
    CovariateCharacterization,
    CovariateEvidence,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
)
from scarf.agent.experimental_context.requirements import (
    active_batch_safety,
    objective_evidence,
)
from tests.test_agent_objective_requirements import _repeated_design


def test_provider_schema_omits_derived_fields_without_changing_serialization():
    decision = ExperimentalContextDecision()
    before = decision.model_dump(mode="json")
    schema = ExperimentalContextDecision.model_json_schema()["properties"]
    derived = {
        "cellQc",
        "protectedCombinations",
        "physicalCaptureColumn",
        "pooledReferenceCaptures",
        "unsupportedProtection",
    }
    assert not derived.intersection(schema)
    assert derived <= before.keys()
    assert (
        ExperimentalContextDecision.model_validate(before).model_dump(mode="json")
        == before
    )


def test_illegal_qc_is_rejected_before_characterization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An illegal field must be rejected before metadata work")

    monkeypatch.setattr(validation, "characterize_context", forbidden)
    with pytest.raises(ModelRetry, match="cellQc blank"):
        validation.validate_experimental_context(
            ExperimentalContextDecision(cellQc=CellQcPlan(rationale="skip")),
            ExperimentalContextDependencies(),
        )


def test_model_evidence_deduplicates_design_without_dropping_contradictions():
    safety = [
        {
            "coefficient": "condition",
            "preservesConditionCoverage": False,
            "reason": "A protected group would disappear",
        }
    ]
    failure = CaptureFailureEvidence(capture="library", conditionAndUnitSafety=safety)
    profiles = [
        CellQcProfileEvidence(profileId=name, captureFailureEvidence=[failure])
        for name in ("global", "capture")
    ]
    full = CovariateEvidence(qcProfiles=profiles)
    original = full.model_dump(mode="json")
    compact = tools.compact_context_evidence(full)
    assert len(compact["captureDesignSafety"]) == 1
    for profile in compact["qcProfiles"]:
        ref = profile["captureFailureEvidence"][0]["designSafetyRef"]
        assert compact["captureDesignSafety"][ref] == safety
    assert full.model_dump(mode="json") == original


def test_context_details_restore_exact_capture_bounds_and_missingness(monkeypatch):
    import asyncio

    failure = CaptureFailureEvidence(
        capture="capture-a",
        metricMissingFractions={"mitochondrial": 0.2, "counts": 0.0},
        conditionAndUnitSafety=[
            {
                "coefficient": "condition",
                "requiredGroups": ["rare", "common"],
                "remainingGroups": ["common"],
                "preservesConditionCoverage": False,
                "preservesIndependentUnitCoverage": False,
            }
        ],
    )
    profile = CellQcProfileEvidence(
        profileId="capture-policy",
        action="sampleMad",
        attributes=["counts"],
        sampleColumn="capture",
        resolvedBounds=[
            {"group": "capture-a", "upperRemoval": 12.0},
            {"group": "capture-b", "upperRemoval": 15.0},
        ],
        captureFailureEvidence=[failure],
    )
    full = CovariateEvidence(qcProfiles=[profile])
    view = tools.compact_context_evidence(full)
    capture = view["qcProfiles"][0]["captureFailureEvidence"][0]
    assert capture["metricMissingness"] == {
        "measuredSources": 2,
        "nonzeroOrUnavailable": {"mitochondrial": 0.2},
    }
    assert view["captureDesignSafety"][capture["designSafetyRef"]][0]["lostGroups"] == [
        "rare"
    ]
    deps = ExperimentalContextDependencies(
        characterization=CovariateCharacterization(status="done"),
        qcProfiles={profile.profileId: profile},
    )
    monkeypatch.setattr(
        tools,
        "characterize_covariates",
        lambda *a, **k: pytest.fail("Details must not recompute"),
    )
    detail = asyncio.run(
        tools.inspect_context_evidence(
            SimpleNamespace(deps=deps), "qcProfile", profile.profileId, "capture-a"
        )
    )
    assert detail["capture"] == failure.model_dump(mode="json")
    assert detail["resolvedBounds"] == [profile.resolvedBounds[0]]
    with pytest.raises(ModelRetry, match="Choose one capture"):
        asyncio.run(
            tools.inspect_context_evidence(
                SimpleNamespace(deps=deps), "qcProfile", profile.profileId, "unknown"
            )
        )


def test_compact_context_keeps_repeated_donor_failures_and_full_details():
    import asyncio

    record = {
        "name": "condition",
        "kind": "categorical",
        "missingness": [{"missing": 3}],
        "pairedCoverage": {
            "complete": False,
            "incompletePairs": 2,
            "incompleteExamples": [{"pair": "donor-a"}, {"pair": "donor-b"}],
        },
    }
    characterization = CovariateCharacterization(status="done", coefficients=[record])
    view = tools.compact_context_evidence(
        CovariateEvidence(characterization=characterization)
    )
    coefficient = view["characterization"]["coefficients"][0]
    assert coefficient["pairedCoverage"]["complete"] is False
    assert coefficient["pairedCoverage"]["incompletePairs"] == 2
    assert coefficient["pairedCoverage"]["incompleteExamplesInSavedDetails"] == 2
    assert coefficient["missingness"] == [{"missing": 3}]
    detail = asyncio.run(
        tools.inspect_context_evidence(
            SimpleNamespace(
                deps=ExperimentalContextDependencies(characterization=characterization)
            ),
            "coefficient",
            "condition",
        )
    )
    assert detail == record


def test_completed_context_round_resumes_without_resetting_allowance():
    saved = {}
    deps = ExperimentalContextDependencies(
        characterization=CovariateCharacterization(status="done"),
        characterizationInputs={"exact": "inputs"},
        designRounds=1,
        toolCalls=["inspect_cell_covariates", "analyze_experimental_design"],
        checkpointWrite=lambda key, value: saved.update({key: value}),
    )
    tools.persist_context_evidence(deps, "design1")
    resumed = ExperimentalContextDependencies(checkpointRead=saved.get)
    assert tools.restore_context_evidence(resumed)
    assert resumed.designRounds == 1
    assert resumed.characterizationInputs == deps.characterizationInputs
    assert resumed.toolCalls == deps.toolCalls
    assert resumed.characterization == deps.characterization


def test_explicit_joint_request_cannot_be_satisfied_by_marginal_proposal():
    result = _repeated_design()
    objective = "Describe supported populations. Assess tissue and condition jointly"
    requirements, coverage = objective_evidence(
        study_context="The observations contain repeated donors and incomplete pairing.",
        study_objective=objective,
        experimental_result=result,
    )
    missing = [
        item
        for item in requirements
        if item.requirementId.startswith("requestedDesign:")
    ]
    assert len(missing) == 1
    assert missing[0].columns == ["condition", "tissue"]
    assert (
        next(
            item for item in coverage if item.requirementId == missing[0].requirementId
        ).status
        == "unsupported"
    )


def test_batch_alternatives_do_not_create_untested_union_or_stale_license():
    first = SimpleNamespace(batchColumns=["library"], status="unsafe")
    second = SimpleNamespace(batchColumns=["chemistry"], status="safe")
    result = SimpleNamespace(
        batchSafety=[first, second], decision=ExperimentalContextDecision()
    )
    with pytest.raises(ValueError, match="exact assessed batch set"):
        active_batch_safety(result)
    result.decision.batchCorrection.batchColumns = ["chemistry"]
    assert active_batch_safety(result) == [second]
    result.decision.batchCorrection.batchColumns = ["library"]
    assert active_batch_safety(result) == [first]


def test_exact_context_characterization_reuses_work_and_invalidates_changes(
    monkeypatch,
):
    from tests.test_agent_experimental_context import _Store, _context

    store = _Store()
    deps = _context(store).deps
    calls = []

    def measured(*args, **kwargs):
        calls.append(kwargs["directions"])
        return CovariateCharacterization(status="done")

    monkeypatch.setattr(tools, "characterize_covariates", measured)
    first = tools.characterize_context(deps, {"columnDomains": {"batch": "technical"}})
    assert (
        tools.characterize_context(deps, {"columnDomains": {"batch": "technical"}})
        is first
    )
    assert len(calls) == 1
    store.cells._values["batch"][0] = "b2"
    tools.characterize_context(deps, {"columnDomains": {"batch": "technical"}})
    assert len(calls) == 2
    tools.characterize_context(deps, {"columnDomains": {"batch": "design"}})
    assert len(calls) == 3


def test_context_identity_binds_added_columns_without_repeating_resume_scan(
    monkeypatch,
):
    from scarf.agent.orchestrator.context import _context_metadata_identity
    from scarf.agent.parameter_tuning import execution

    values = {"capture": "first"}
    measured = []

    def fingerprint(metadata, column):
        measured.append(column)
        return values[column]

    monkeypatch.setattr(execution, "_metadata_column_fingerprint", fingerprint)
    store = SimpleNamespace(cells=SimpleNamespace(columns=["original", "capture"]))
    request = SimpleNamespace(
        inputIdentity={"data": {"metadata": {"original": "validated"}}}
    )
    first = _context_metadata_identity(store, request)
    assert first == {"original": "validated", "capture": "first"}
    assert measured == ["capture"]
    values["capture"] = "changed"
    assert _context_metadata_identity(store, request) != first


def test_generic_joint_request_also_requires_a_joint_proposal():
    result = _repeated_design()
    requirements, coverage = objective_evidence(
        study_context="The observations contain repeated donors and incomplete pairing.",
        study_objective="Describe supported populations. Assess individual and combined covariates",
        experimental_result=result,
    )
    assert any(
        item.requirementId.startswith("requestedDesign:") for item in requirements
    )
    assert any(
        item.requirementId.startswith("requestedDesign:")
        and item.status == "unsupported"
        for item in coverage
    )


@pytest.mark.parametrize(
    "change", [None, "metadata", "features", "cohort", "noRevision"]
)
def test_tuning_context_revision_preserves_only_exact_numerical_inputs(
    monkeypatch, change
):
    from copy import deepcopy
    from scarf.agent.orchestrator import journal
    from scarf.agent.orchestrator.models import StageEvidenceReference
    from scarf.agent.orchestrator.tuning import _tuning_revision_provenances

    request = SimpleNamespace(requestSha256="request", configSha256="config")

    def reference(name):
        return StageEvidenceReference(
            workflowRunId="workflow",
            stage="experimental_context",
            key=name,
            contentSha256=name,
        )

    old_ref, new_ref = reference("old"), reference("new")
    previous = {
        "preprocessedAssays": [{"cells": "frozen", "features": "genes"}],
        "featureMetadataFingerprints": {"names": "same"},
        "metadataFingerprints": {"condition": "same"},
        "studyContract": {"protectedCombinations": []},
    }
    current = deepcopy(previous)
    current["studyContract"] = {"protectedCombinations": [["condition", "tissue"]]}
    current["metadataFingerprints"]["tissue"] = "newly measured"
    old = SimpleNamespace(
        attemptId="old",
        status="done",
        reportReferences=[old_ref],
        inputs={},
        outputs={"studyContract": previous["studyContract"]},
        requestSha256="request",
        configSha256="config",
    )
    new = SimpleNamespace(
        attemptId="new",
        status="done",
        reportReferences=[new_ref],
        inputs={"reassessContextReport": old_ref.model_dump(mode="json")},
        outputs={"studyContract": current["studyContract"]},
        requestSha256="request",
        configSha256="config",
    )
    if change == "metadata":
        current["metadataFingerprints"]["condition"] = "changed"
    elif change == "features":
        current["featureMetadataFingerprints"]["names"] = "changed"
    elif change == "cohort":
        current["preprocessedAssays"][0]["cells"] = "different"
    elif change == "noRevision":
        new.inputs = {}
    monkeypatch.setattr(journal, "_stage_outcomes", lambda *args: [old, new])
    before = deepcopy(previous)

    def invoke():
        return _tuning_revision_provenances(
            SimpleNamespace(zw=None),
            "prefix",
            "workflow",
            request,
            new_ref,
            current,
            [SimpleNamespace(inputs=previous)],
        )

    if change is not None:
        with pytest.raises(ValueError, match="changed beyond"):
            invoke()
    else:
        assert invoke() == [
            {**previous, "requestSha256": "request", "configSha256": "config"}
        ]
    assert previous == before
