"""Contract tests for the decision kernel and deterministic auditor."""

import pytest
from pydantic import ValidationError

from scarf.agent.decisions.kernel import (
    DecisionEvidence,
    DecisionOption,
    DecisionRecord,
    DecisionSpec,
    DeterministicDecisionAuditor,
    EvidenceBundle,
    ProtectedVariableEffect,
)
from scarf.agent.types import ArtifactReferenceModel


def _evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundleId="bundle:clusterPartition",
        decisionId="clusterPartition",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:silhouette",
                evidenceClass="geometric",
                summary="The coarse partition has the largest silhouette.",
            ),
            DecisionEvidence(
                evidenceId="evidence:markers",
                evidenceClass="markerCoherence",
                summary="The finer partition has distinct marker programs.",
            ),
            DecisionEvidence(
                evidenceId="evidence:stability",
                evidenceClass="resamplingStability",
                summary="The finer partition is stable under subsampling.",
            ),
            DecisionEvidence(
                evidenceId="evidence:replicates",
                evidenceClass="crossUnitSupport",
                summary="The finer populations occur in independent units.",
            ),
        ],
    )


def _decision_spec() -> DecisionSpec:
    return DecisionSpec(
        decisionId="clusterPartition",
        definitionVersion=1,
        checkpoint="clustering",
        question="Which registered partition is defensible?",
        evidenceBundleId="bundle:clusterPartition",
        options=[
            DecisionOption(
                optionId="partition:coarse",
                status="apply",
                label="Coarse partition",
                description="Use the metric-preferred coarse partition.",
            ),
            DecisionOption(
                optionId="partition:fine",
                status="apply",
                label="Fine partition",
                description="Use the finer marker-supported partition.",
                requiredEvidenceClasses=["markerCoherence"],
            ),
            DecisionOption(
                optionId="partition:abstain",
                status="abstain",
                label="No discrete partition",
                description="Do not claim that a discrete partition is supported.",
            ),
        ],
        baselineOptionId="partition:coarse",
        metricPreferredOptionId="partition:coarse",
        requireIndependentOverrideEvidence=True,
    )


def _decision_record(
    *,
    record_id: str = "decision:cluster:1",
    selected_option_id: str = "partition:coarse",
    status: str = "apply",
    source: str = "agent",
    evidence_ids: list[str] | None = None,
    override_of: str | None = None,
    override_evidence_ids: list[str] | None = None,
    available_evidence_ids: list[str] | None = None,
    supersedes: str | None = None,
) -> DecisionRecord:
    bundle = _evidence_bundle().with_content_sha256()
    assert bundle.contentSha256 is not None
    return DecisionRecord(
        recordId=record_id,
        decisionId="clusterPartition",
        definitionVersion=1,
        evidenceBundleId=bundle.bundleId,
        evidenceBundleSha256=bundle.contentSha256,
        offeredOptionIds=[
            "partition:coarse",
            "partition:fine",
            "partition:abstain",
        ],
        availableEvidenceIds=available_evidence_ids
        if available_evidence_ids is not None
        else [item.evidenceId for item in bundle.evidence],
        selectedOptionId=selected_option_id,
        status=status,
        source=source,
        evidenceIds=evidence_ids or ["evidence:silhouette"],
        rationale="The cited evidence supports this registered option.",
        confidence="medium",
        overrideOfOptionId=override_of,
        overrideEvidenceIds=override_evidence_ids or [],
    )


def test_option_contract_rejects_freeform_execution_parameters() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DecisionOption.model_validate(
            {
                "optionId": "pca:30",
                "status": "apply",
                "label": "Thirty PCs",
                "description": "Use the registered thirty-PC prefix.",
                "dimensions": 30,
            }
        )


def test_evidence_bundle_rejects_duplicate_evidence_ids() -> None:
    item = DecisionEvidence(
        evidenceId="evidence:qc",
        evidenceClass="qualityControl",
        summary="Observed cell quality evidence.",
    )
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        EvidenceBundle(
            bundleId="bundle:qc",
            decisionId="cellQuality",
            evidence=[item, item],
        )


def test_evidence_contract_rejects_duplicate_artifacts_and_bad_checksums() -> None:
    reference = ArtifactReferenceModel(
        scope="assay",
        assay="RNA",
        kind="pca",
        artifactId="a" * 64,
    )
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        DecisionEvidence(
            evidenceId="evidence:pca",
            evidenceClass="geometric",
            summary="Observed PCA evidence.",
            artifactReferences=[reference, reference],
        )

    values = _evidence_bundle().model_dump()
    values["contentSha256"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match"):
        EvidenceBundle.model_validate(values)
    values["contentSha256"] = "INVALID"
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        EvidenceBundle.model_validate(values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"selectedOptionId": "partition:invented"},
            "selectedOptionId must reference an offered option",
        ),
        (
            {"evidenceIds": ["evidence:invented"]},
            "evidenceIds must reference only available evidence",
        ),
        (
            {
                "overrideOfOptionId": None,
                "evidenceIds": ["evidence:markers"],
                "overrideEvidenceIds": ["evidence:markers"],
            },
            "overrideEvidenceIds require overrideOfOptionId",
        ),
    ],
)
def test_decision_record_rejects_non_exact_references(
    changes: dict[str, object], message: str
) -> None:
    values = _decision_record().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        DecisionRecord.model_validate(values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "overrideOfOptionId": "partition:fine",
                "overrideEvidenceIds": ["evidence:stability"],
            },
            "overrideEvidenceIds must be included",
        ),
        (
            {"overrideOfOptionId": "partition:invented"},
            "overrideOfOptionId must reference",
        ),
        (
            {"overrideOfOptionId": "partition:coarse"},
            "overrideOfOptionId must differ",
        ),
        (
            {"supersedes": "decision:cluster:1"},
            "Extra inputs are not permitted",
        ),
        (
            {
                "protectedVariableEffects": [
                    ProtectedVariableEffect(
                        variable="disease",
                        status="preserved",
                        evidenceIds=["evidence:invented"],
                        summary="Disease structure is preserved.",
                    ).model_dump()
                ]
            },
            "protectedVariableEffects must reference",
        ),
    ],
)
def test_decision_record_rejects_invalid_override_and_lineage_references(
    changes: dict[str, object],
    message: str,
) -> None:
    values = _decision_record().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        DecisionRecord.model_validate(values)


def test_auditor_accepts_an_exact_metric_preferred_decision() -> None:
    record = _decision_record()

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record, created_at_ns=10
    )

    assert all(check.status == "passed" for check in verification)
    assert {check.status for check in verification} == {"passed"}


def test_auditor_rejects_a_tampered_evidence_bundle_checksum() -> None:
    values = _decision_record().model_dump()
    values["evidenceBundleSha256"] = "f" * 64
    record = DecisionRecord.model_validate(values)

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(),
        _evidence_bundle(),
        record,
    )

    assert any(check.status == "failed" for check in verification)
    assert [check.checkId for check in verification if check.status == "failed"] == [
        "decisionIdentity"
    ]


def test_auditor_requires_evidence_bound_to_the_selected_option() -> None:
    spec_values = _decision_spec().model_dump()
    spec_values["options"][0]["requiredEvidenceIds"] = ["evidence:markers"]
    spec = DecisionSpec.model_validate(spec_values)

    verification = DeterministicDecisionAuditor.audit(
        spec,
        _evidence_bundle(),
        _decision_record(),
    )

    assert any(check.status == "failed" for check in verification)
    assert [check.checkId for check in verification if check.status == "failed"] == [
        "requiredEvidence"
    ]


def test_auditor_rejects_a_tampered_option_or_evidence_inventory() -> None:
    record = _decision_record(
        available_evidence_ids=["evidence:silhouette"],
    )
    values = record.model_dump()
    values["offeredOptionIds"] = ["partition:coarse", "partition:abstain"]
    record = DecisionRecord.model_validate(values)

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    assert any(check.status == "failed" for check in verification)
    failed = {check.checkId for check in verification if check.status == "failed"}
    assert failed == {"exactOptionInventory", "exactEvidenceInventory"}


def test_auditor_requires_two_independent_non_geometric_override_classes() -> None:
    record = _decision_record(
        selected_option_id="partition:fine",
        evidence_ids=["evidence:silhouette", "evidence:markers"],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:silhouette", "evidence:markers"],
    )

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    assert any(check.status == "failed" for check in verification)
    failed = [check for check in verification if check.status == "failed"]
    assert [check.checkId for check in failed] == ["independentOverrideEvidence"]


def test_auditor_accepts_two_independent_non_geometric_override_classes() -> None:
    record = _decision_record(
        selected_option_id="partition:fine",
        evidence_ids=["evidence:markers", "evidence:stability"],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:markers", "evidence:stability"],
    )

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    assert all(check.status == "passed" for check in verification)


def test_auditor_rejects_override_evidence_from_another_option() -> None:
    spec_values = _decision_spec().model_dump()
    spec_values["options"][1]["requiredEvidenceIds"] = [
        "evidence:markers",
        "evidence:stability",
    ]
    spec = DecisionSpec.model_validate(spec_values)
    record = _decision_record(
        selected_option_id="partition:fine",
        evidence_ids=[
            "evidence:markers",
            "evidence:stability",
            "evidence:replicates",
        ],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:markers", "evidence:replicates"],
    )

    verification = DeterministicDecisionAuditor.audit(
        spec,
        _evidence_bundle(),
        record,
    )

    assert any(check.status == "failed" for check in verification)
    assert [check.checkId for check in verification if check.status == "failed"] == [
        "independentOverrideEvidence"
    ]


def test_human_choices_obey_the_same_source_and_status_contracts() -> None:
    spec_values = _decision_spec().model_dump()
    spec_values["allowedSources"] = ["rule", "agent"]
    spec = DecisionSpec.model_validate(spec_values)
    record = _decision_record(source="human", status="skip")

    verification = DeterministicDecisionAuditor.audit(spec, _evidence_bundle(), record)

    assert any(check.status == "failed" for check in verification)
    failed = {check.checkId for check in verification if check.status == "failed"}
    assert failed == {"selectedOption", "decisionSource"}
