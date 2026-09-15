"""Constrained RNA choices persisted with their owning stage evidence."""

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...utils.logging import logger
from .. import record_io
from ..config.agent_exec import run_agent_sync
from ..decisions.kernel import (
    DecisionRecord,
    DecisionSelection,
    DecisionSource,
    EvidenceBundle,
)
from ..decisions.rna import (
    CompiledRnaDecision,
    RnaDecisionDefinition,
    compile_rna_decision,
)
from . import journal
from .models import OrchestrationRequestRecord, WorkflowQuestion


@dataclass(frozen=True, slots=True)
class DecisionResolution:
    """One validated choice or an explicit unresolved scientific question."""

    record: DecisionRecord | None
    compiled: CompiledRnaDecision | None
    checkpointSha256: str
    pending: WorkflowQuestion | None = None


def _sha256(value: object) -> str:
    return hashlib.sha256(record_io.canonical_json_bytes(value)).hexdigest()


def _record_from_selection(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    selection: DecisionSelection,
    source: DecisionSource,
    model_name: str | None,
    prompt_sha256: str | None,
    created_at_ns: int,
) -> DecisionRecord:
    if evidence.contentSha256 is None:
        raise ValueError("Decision evidence must have a content digest")
    option = definition.spec.option_by_id().get(selection.selectedOptionId)
    if option is None:
        raise ValueError("Selected option is not offered")
    return DecisionRecord(
        recordId=f"decision:{definition.spec.decisionId}:{_sha256(selection.model_dump(mode='json'))[:24]}",
        decisionId=definition.spec.decisionId,
        definitionVersion=definition.spec.definitionVersion,
        evidenceBundleId=evidence.bundleId,
        evidenceBundleSha256=evidence.contentSha256,
        offeredOptionIds=[v.optionId for v in definition.spec.options],
        availableEvidenceIds=[v.evidenceId for v in evidence.evidence],
        selectedOptionId=selection.selectedOptionId,
        status=option.status,
        source=source,
        evidenceIds=list(selection.evidenceIds),
        rationale=selection.rationale,
        confidence=selection.confidence,
        protectedVariableEffects=list(selection.protectedVariableEffects),
        overrideOfOptionId=selection.overrideOfOptionId,
        overrideEvidenceIds=list(selection.overrideEvidenceIds),
        modelName=model_name,
        promptSha256=prompt_sha256,
        createdAtNs=created_at_ns,
        softwareSha256=_sha256(definition.model_dump(mode="json")),
    )


def _validate_selection(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    selection: DecisionSelection,
) -> DecisionSelection:
    record = _record_from_selection(
        definition, evidence, selection, "agent", None, None, 0
    )
    compile_rna_decision(definition, evidence, record)
    return selection


def _human_selection(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    answer: Mapping[str, Any],
) -> DecisionSelection:
    if (
        set(answer) != {"decisionId", "optionId", "rationale"}
        or answer["decisionId"] != definition.spec.decisionId
    ):
        raise ValueError(
            "A decision answer must name this decisionId, optionId, and rationale"
        )
    option = definition.spec.option_by_id().get(answer["optionId"])
    if option is None:
        raise ValueError("Human answer does not select an offered option")
    ids = [
        v.evidenceId
        for v in evidence.evidence
        if not option.requiredEvidenceClasses
        or v.evidenceClass in option.requiredEvidenceClasses
    ]
    ids = list(dict.fromkeys([*option.requiredEvidenceIds, *ids]))
    override = (
        definition.spec.requireIndependentOverrideEvidence
        and definition.spec.metricPreferredOptionId is not None
        and option.optionId != definition.spec.metricPreferredOptionId
        and option.status in {"apply", "skip"}
    )
    override_ids = (
        [
            v.evidenceId
            for v in evidence.evidence
            if v.evidenceClass
            in {
                "markerCoherence",
                "resamplingStability",
                "crossUnitSupport",
                "protectedVariablePreservation",
            }
            and (
                not option.requiredEvidenceIds
                or v.evidenceId in option.requiredEvidenceIds
            )
        ]
        if override
        else []
    )
    return DecisionSelection(
        selectedOptionId=option.optionId,
        evidenceIds=list(dict.fromkeys([*ids, *override_ids])),
        rationale=answer["rationale"],
        confidence="notApplicable",
        overrideOfOptionId=definition.spec.metricPreferredOptionId
        if override
        else None,
        overrideEvidenceIds=override_ids,
    )


class DecisionStagesMixin:
    """Resolve choices through scientific validation and one stage-owned checkpoint."""

    model: Any

    def _resolve_rna_decision(
        self,
        store: Any,
        request_record: OrchestrationRequestRecord,
        definition: RnaDecisionDefinition,
        evidence: EvidenceBundle,
        answers: Mapping[str, Any],
        *,
        rule_selection: DecisionSelection | None = None,
        agent_selection: DecisionSelection | None = None,
        agent_model_name: str | None = None,
        qc_evidence: Mapping[str, Any] | None = None,
    ) -> DecisionResolution:
        evidence = (
            evidence if evidence.contentSha256 else evidence.with_content_sha256()
        )
        if (
            evidence.decisionId != definition.spec.decisionId
            or evidence.bundleId != definition.spec.evidenceBundleId
        ):
            raise ValueError("Decision and exact evidence identities differ")
        if rule_selection is not None and agent_selection is not None:
            raise ValueError("A decision cannot have two supplied owners")
        decision_id = definition.spec.decisionId
        stage = (
            "preprocessing_plan"
            if decision_id in {"qcGrouping", "cellQuality"}
            else "preprocessing"
        )
        question_id = f"decision:{decision_id}"
        answer = answers.get(question_id)
        identity = {
            "requestSha256": request_record.requestSha256,
            "configSha256": request_record.configSha256,
            "definition": definition.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
            "answer": answer,
            "ruleSelection": rule_selection.model_dump(mode="json")
            if rule_selection
            else None,
            "agentSelection": agent_selection.model_dump(mode="json")
            if agent_selection
            else None,
        }
        if qc_evidence is not None:
            identity["qcPolicyEvidence"] = dict(qc_evidence)
        digest = _sha256(identity)
        key = f"{stage}/decisions/{decision_id}/{digest}"
        prefix = journal._ensure_orchestration_store(store)
        saved = journal.load_checkpoint(
            store, prefix, request_record.workflowRunId, key, identity
        )
        if saved is not None:
            record = DecisionRecord.model_validate(saved["record"])
            compiled = compile_rna_decision(definition, evidence, record)
            if [v.model_dump(mode="json") for v in compiled.checks] != saved["checks"]:
                raise ValueError("Saved decision checks differ from exact replay")
            pending = (
                WorkflowQuestion.model_validate(saved["pending"])
                if saved.get("pending")
                else None
            )
            return DecisionResolution(
                record, None if pending else compiled, digest, pending
            )
        source: DecisionSource = "agent"
        model_name = agent_model_name
        prompt_hash = None
        if rule_selection is not None:
            if answer is not None:
                raise ValueError("A rule-owned decision cannot accept a human answer")
            selection = rule_selection
            source = "rule"
        elif agent_selection is not None:
            if answer is not None:
                raise ValueError(
                    "A supplied agent decision cannot accept a human answer"
                )
            selection = agent_selection
        elif answer is not None:
            if not isinstance(answer, Mapping):
                raise ValueError("Human decision answer must be a mapping")
            selection = _human_selection(definition, evidence, answer)
            source = "human"
        else:
            payload = {
                "studyContext": request_record.request.studyContext,
                "studyObjective": request_record.request.studyObjective,
                "question": definition.spec.question,
                "spec": definition.spec.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
            }
            if qc_evidence is not None:
                payload["qcPolicyEvidence"] = dict(qc_evidence)
            prompt = json.dumps(payload, indent=2, sort_keys=True)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            execution = run_agent_sync(
                model=self.model,
                output_type=DecisionSelection,
                system_prompt=(
                    "Assess the offered settings against the study objective using the observed quantitative and qualitative evidence. "
                    "The objective identifies questions and biology to protect; it does not predetermine the answer. "
                    "Select only an offered option and cite its required evidence. "
                    "Write the rationale as two or three plain-language sentences for the analysis report: "
                    "state the chosen outcome, the relevant measured comparison, and its scientific tradeoff or limitation. "
                    "Use readable study and measurement names. Keep option identifiers, artifact identifiers and "
                    "internal field names out of the rationale; cite identifiers in evidenceIds instead. "
                    "Do not infer nuisance from gene-family names alone. Retain defaults only when evidence supports them. "
                    "Defer essential unresolved questions. Model failure or a work limit never justifies an unsupported choice. "
                    "For QC, distinguish retained group coverage from preserved biological structure: "
                    "group counts and absence of unsafe flags do not establish balanced retention, "
                    "cell validity, or preservation of marker programs. Compare retention fractions "
                    "and metric-specific flags where supplied; unmeasured effects remain unknown. "
                    "Use qcPolicyEvidence for exact thresholds, distributions, retention "
                    "and limitations. Its Ref fields point to sharedMeasurements in the "
                    "same payload. Do not substitute the short evidence summaries for "
                    "these measurements or dismiss an unsafe rejected policy as untested. "
                    "Within-capture QC does not require an independent biological unit or a healthy reference. "
                    "An override needs the independent evidence required by the supplied specification."
                ),
                user_prompt=prompt,
                config=request_record.config.agentRunConfig,
                name=f"rna_{decision_id}_decision",
                output_validator=lambda value: _validate_selection(
                    definition, evidence, value
                ),
                on_attempt=journal.model_attempt_callback(
                    store, prefix, request_record.workflowRunId, key, payload
                ),
            )
            selection = execution.output
            if not isinstance(selection, DecisionSelection):
                raise TypeError("Decision model returned an unexpected result")
            model_name = execution.runInfo.modelName
        record = _record_from_selection(
            definition,
            evidence,
            selection,
            source,
            model_name,
            prompt_hash,
            time.time_ns(),
        )
        compiled = compile_rna_decision(definition, evidence, record)
        pending = (
            WorkflowQuestion(
                questionId=question_id,
                decisionId=decision_id,
                question=f"{definition.spec.question} {record.rationale}",
                options=[v.optionId for v in definition.spec.options],
                evidenceIds=[v.evidenceId for v in evidence.evidence],
                planChecksum=digest,
            )
            if record.status in {"defer", "abstain"}
            else None
        )
        value = {
            "stage": stage,
            "checkpointSha256": digest,
            "spec": definition.spec.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
            "checks": [v.model_dump(mode="json") for v in compiled.checks],
            "pending": pending.model_dump(mode="json") if pending else None,
        }
        journal.save_checkpoint(
            store, prefix, request_record.workflowRunId, key, identity, value
        )
        option = definition.spec.option_by_id()[record.selectedOptionId]
        logger.info(
            f"{definition.spec.question} Selected {option.label}: {record.rationale}"
        )
        return DecisionResolution(
            record, None if pending else compiled, digest, pending
        )

    @staticmethod
    def _pending_decision_question(
        resolution: DecisionResolution, definition: RnaDecisionDefinition
    ) -> WorkflowQuestion:
        if resolution.pending is None:
            raise ValueError("Decision has no pending question")
        return resolution.pending
