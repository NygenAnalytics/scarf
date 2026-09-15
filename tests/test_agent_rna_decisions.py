"""Exact evidence and payload validation for the live RNA QC decisions."""

import pytest
from pydantic import ValidationError

from scarf.agent.decisions.kernel import (
    DecisionEvidence,
    DecisionRecord,
    EvidenceBundle,
)
from scarf.agent.decisions.rna import (
    CellQualityExecutorPayload,
    RnaDecisionCompilationError,
    RnaDecisionGateError,
    build_cell_quality_decision,
    build_qc_grouping_decision,
    compile_rna_decision,
    require_option_evidence,
)


def _bundle(
    decision_id: str,
    bundle_id: str,
    classes: list[str],
) -> EvidenceBundle:
    return EvidenceBundle(
        bundleId=bundle_id,
        decisionId=decision_id,
        evidence=[
            DecisionEvidence(
                evidenceId=f"evidence:{evidence_class}:{index}",
                evidenceClass=evidence_class,
                summary=f"Observed {evidence_class} evidence.",
            )
            for index, evidence_class in enumerate(classes)
        ],
    )


def _record(
    definition: object,
    bundle: EvidenceBundle,
    option_id: str,
    *,
    source: str = "agent",
    evidence_ids: list[str] | None = None,
    override_of: str | None = None,
    override_evidence_ids: list[str] | None = None,
) -> DecisionRecord:
    bundle = bundle.with_content_sha256()
    assert bundle.contentSha256 is not None
    spec = definition.spec
    option = spec.option_by_id()[option_id]
    return DecisionRecord(
        recordId=f"record:{spec.decisionId}:1",
        decisionId=spec.decisionId,
        definitionVersion=spec.definitionVersion,
        evidenceBundleId=bundle.bundleId,
        evidenceBundleSha256=bundle.contentSha256,
        offeredOptionIds=[item.optionId for item in spec.options],
        availableEvidenceIds=[item.evidenceId for item in bundle.evidence],
        selectedOptionId=option_id,
        status=option.status,
        source=source,
        evidenceIds=evidence_ids
        if evidence_ids is not None
        else [item.evidenceId for item in bundle.evidence],
        rationale="The exact cited evidence supports this registered option.",
        confidence="medium",
        overrideOfOptionId=override_of,
        overrideEvidenceIds=override_evidence_ids or [],
    )


def test_cell_quality_registry_gates_capture_profiles() -> None:
    global_grouping = build_qc_grouping_decision(
        evidence_bundle_id="bundle:cellQuality",
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )

    assert [option.optionId for option in global_grouping.spec.options] == [
        "qcGrouping:global",
        "qcGrouping:defer",
    ]
    global_only = build_cell_quality_decision(
        evidence_bundle_id="bundle:cellQuality",
        available_profiles=["retainWithFlags", "globalMad5"],
    )
    assert [option.optionId for option in global_only.spec.options] == [
        "cellQuality:retainWithFlags",
        "cellQuality:globalMad5",
        "cellQuality:defer",
    ]
    global_payload = global_only.executor_option("cellQuality:globalMad5").payload
    assert global_payload.operation == "cellQualityProfile"
    assert global_payload.lowerCountMad == 5.0
    assert global_payload.flagHighCounts is True

    with pytest.raises(RnaDecisionGateError, match="physical capture"):
        build_qc_grouping_decision(
            evidence_bundle_id="bundle:cellQuality",
            physical_capture_eligible=False,
            pooled_reference_eligible=True,
        )


def test_cell_quality_registry_offers_only_eligible_pooled_reference() -> None:
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:cellQuality",
        available_profiles=[
            "retainWithFlags",
            "captureMad5",
            "captureMad3Sensitivity",
            "pooledReferenceMad5",
        ],
    )

    assert "cellQuality:captureMad5" in definition.spec.option_by_id()
    assert "cellQuality:captureMad3Sensitivity" in definition.spec.option_by_id()
    assert "cellQuality:pooledReferenceMad5" in definition.spec.option_by_id()


def test_qc_grouping_offers_licensed_capture_and_pooled_modes() -> None:
    definition = build_qc_grouping_decision(
        evidence_bundle_id="bundle:cellQuality",
        physical_capture_eligible=True,
        pooled_reference_eligible=True,
    )

    assert [option.optionId for option in definition.spec.options] == [
        "qcGrouping:global",
        "qcGrouping:physicalCapture",
        "qcGrouping:pooledReference",
        "qcGrouping:defer",
    ]
    assert [
        definition.executor_option(option_id).payload.groupingMode
        for option_id in (
            "qcGrouping:global",
            "qcGrouping:physicalCapture",
            "qcGrouping:pooledReference",
        )
    ] == ["global", "physicalCapture", "pooledReference"]


@pytest.mark.parametrize(
    "profile,changes,message",
    [
        ("retainWithFlags", {"lowerCountMad": 3.0}, "cannot define removal"),
        ("retainWithFlags", {"groupByCapture": True}, "cannot enable filtering"),
        ("coreGlobalGaussian", {"lowerCountMad": 3.0}, "exact core bounds"),
        ("coreSampleMad3", {"groupByCapture": False}, "requires capture grouping"),
        ("globalMad5", {"lowerCountMad": None}, "require all three"),
        ("captureMad5", {"groupByCapture": False}, "groupByCapture"),
        ("globalMad5", {"pooledReference": True}, "pooledReference"),
        ("globalMad5", {"sensitivityOnly": True}, "sensitivityOnly"),
    ],
)
def test_qc_payload_rejects_inconsistent_execution_modes(profile, changes, message):
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:qc", available_profiles=[profile]
    )
    original = definition.executor_option(f"cellQuality:{profile}").payload
    with pytest.raises(ValidationError, match=message):
        CellQualityExecutorPayload.model_validate({**original.model_dump(), **changes})


@pytest.mark.parametrize("profiles", [[], ["globalMad5", "globalMad5"]])
def test_qc_choices_reject_empty_or_duplicate_inventory(profiles):
    with pytest.raises(RnaDecisionGateError):
        build_cell_quality_decision(
            evidence_bundle_id="bundle:qc", available_profiles=profiles
        )


@pytest.mark.parametrize(
    "change",
    ["specCheckpoint", "duplicateOption", "payloadCheckpoint", "missingPayload"],
)
def test_qc_definition_rejects_visible_and_executable_drift(change):
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:qc",
        available_profiles=["coreGlobalGaussian", "globalMad5"],
    )
    values = definition.model_dump()
    if change == "specCheckpoint":
        values["spec"]["checkpoint"] = "qcGrouping"
    elif change == "duplicateOption":
        values["executorOptions"][1]["optionId"] = values["executorOptions"][0][
            "optionId"
        ]
    elif change == "payloadCheckpoint":
        values["executorOptions"][0]["checkpoint"] = "qcGrouping"
    else:
        values["executorOptions"].pop()
    with pytest.raises(ValidationError):
        type(definition).model_validate(values)


def test_qc_compilation_requires_exact_policy_evidence_and_hides_thresholds():
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:qc", available_profiles=["globalMad5"]
    )
    bundle = _bundle("cellQuality", "bundle:qc", ["qualityControl", "design"])
    quality, design = [item.evidenceId for item in bundle.evidence]
    definition = require_option_evidence(
        definition, {"cellQuality:globalMad5": [quality]}
    )
    missing = _record(
        definition, bundle, "cellQuality:globalMad5", evidence_ids=[design]
    )
    with pytest.raises(RnaDecisionCompilationError, match="deterministic verification"):
        compile_rna_decision(definition, bundle, missing)
    record = _record(
        definition, bundle, "cellQuality:globalMad5", evidence_ids=[quality]
    )
    compiled = compile_rna_decision(definition, bundle, record)
    assert compiled.executorPayload.lowerCountMad == 5.0
    assert "lowerCountMad" not in record.model_dump_json()
    with pytest.raises(RnaDecisionCompilationError):
        compile_rna_decision(
            definition,
            bundle,
            record.model_copy(update={"evidenceBundleSha256": "0" * 64}),
        )


def test_qc_defer_has_no_executable_filter_and_evidence_binding_rejects_unknown_options():
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:qc", available_profiles=["coreGlobalGaussian"]
    )
    bundle = _bundle("cellQuality", "bundle:qc", ["qualityControl"])
    compiled = compile_rna_decision(
        definition, bundle, _record(definition, bundle, "cellQuality:defer")
    )
    assert compiled.status == "defer"
    assert compiled.executorPayload.operation == "noExecution"
    with pytest.raises(ValueError, match="unknown options"):
        require_option_evidence(
            definition, {"invented": [bundle.evidence[0].evidenceId]}
        )
    with pytest.raises(TypeError, match="sequences"):
        require_option_evidence(
            definition, {"cellQuality:coreGlobalGaussian": "invented"}
        )
