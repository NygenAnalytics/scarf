"""Immutable orchestration journal and stage lifecycle operations."""

import hashlib
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from .. import record_io
from ..persistence import (
    AgentInvocation,
    AgentName,
    AgentReport,
    AgentReportLink,
    AgentReportReference,
    AgentWorkflowRun,
    finalize_agent_workflow,
    list_agent_reports,
    load_agent_record,
    load_agent_report,
    load_agent_workflow,
    save_agent_report,
)
from ..types import AgentDataModel, ArtifactReferenceModel
from .models import (
    _ORCHESTRATION_FORMAT,
    _ORCHESTRATION_VERSION,
    _STAGE_ORDER,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowResult,
    AutomatedWorkflowStatus,
    FinalAnalysisHandoff,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    WorkflowNeedsInput,
    WorkflowStageAttempt,
    WorkflowStageLink,
    WorkflowStageName,
    artifact_model_to_ref,
)


def _sha256_model(value: AgentDataModel) -> str:
    return hashlib.sha256(
        record_io.canonical_json_bytes(value.model_dump(mode="json"))
    ).hexdigest()


def _record_checksum(value: AgentDataModel) -> str:
    return hashlib.sha256(
        record_io.canonical_json_bytes(
            value.model_dump(mode="json", exclude={"contentSha256"})
        )
    ).hexdigest()


def _write_key_once(group: zarr.Group, key: str, payload: bytes) -> None:
    if record_io.read_key(group, key) is not None:
        raise FileExistsError(f"Immutable orchestration record {key!r} exists")
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(group.store.set_if_not_exists(key, buffer))
    if record_io.read_key(group, key) != payload:
        raise FileExistsError(f"Immutable orchestration record {key!r} raced")


def _list_keys(group: zarr.Group, prefix: str) -> list[str]:
    if not group.store.supports_listing:
        raise NotImplementedError("Orchestration persistence requires listing")
    return record_io.list_keys(group, prefix)


def _orchestration_prefix(store: DataStore) -> str:
    root_path = str(getattr(store.zw, "path", "")).strip("/")
    return record_io.join_key(root_path, "agents", "orchestrations")


def _ensure_orchestration_store(store: DataStore) -> str:
    if "agents" not in store.zw:
        raise RuntimeError("Create the agent workflow before orchestration records")
    agents = store.zw["agents"]
    if not isinstance(agents, zarr.Group):
        raise ValueError("The agents namespace must be a Zarr group")
    if "orchestrations" not in agents:
        candidate_prefix = _orchestration_prefix(store)
        if _list_keys(store.zw, candidate_prefix):
            raise ValueError(
                "A non-Zarr object already occupies the orchestrations namespace"
            )
        agents.create_group(
            "orchestrations",
            attributes={
                "format": _ORCHESTRATION_FORMAT,
                "format_version": _ORCHESTRATION_VERSION,
            },
        )
        logger.info("Initialized the automated workflow orchestration journal")
    node = agents["orchestrations"]
    if not isinstance(node, zarr.Group):
        raise ValueError("The orchestrations namespace must be a Zarr group")
    if (
        node.attrs.get("format") != _ORCHESTRATION_FORMAT
        or node.attrs.get("format_version") != _ORCHESTRATION_VERSION
    ):
        raise ValueError("Unrecognized orchestration persistence format")
    return _orchestration_prefix(store)


def _request_key(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(prefix, workflow_run_id, "request.json")


def _resume_key(prefix: str, workflow_run_id: str, resume_id: str) -> str:
    return record_io.join_key(
        prefix,
        workflow_run_id,
        "resumes",
        f"{resume_id}.json",
    )


def _stage_prefix(
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
) -> str:
    return record_io.join_key(prefix, workflow_run_id, "stages", stage)


def _stage_key(
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
    attempt_id: str,
    filename: Literal["started.json", "outcome.json"],
) -> str:
    return record_io.join_key(
        _stage_prefix(prefix, workflow_run_id, stage),
        attempt_id,
        filename,
    )


def _result_key(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(prefix, workflow_run_id, "result.json")


def _read_model(
    group: zarr.Group,
    key: str,
    model_type: type[AgentDataModel],
) -> AgentDataModel:
    raw = record_io.read_key(group, key)
    if raw is None:
        raise FileNotFoundError(key)
    try:
        return model_type.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError(f"Malformed orchestration record {key!r}") from exc


def _write_model_once(group: zarr.Group, key: str, value: AgentDataModel) -> None:
    _write_key_once(
        group,
        key,
        record_io.display_json_bytes(value.model_dump(mode="json")),
    )


def _stage_checksum(attempt: WorkflowStageAttempt) -> str:
    return _record_checksum(attempt)


def _complete_attempt(
    started: WorkflowStageAttempt,
    *,
    status: Literal["done", "needsInput", "failed"],
    report_references: Sequence[AgentReportReference] = (),
    artifacts: Mapping[str, ArtifactReferenceModel] | None = None,
    outputs: Mapping[str, Any] | None = None,
    actions: Sequence[str] = (),
    notes: Sequence[str] = (),
    needs_input: WorkflowNeedsInput | None = None,
    error: str | None = None,
) -> WorkflowStageAttempt:
    outcome = started.model_copy(
        update={
            "status": status,
            "completedAtNs": time.time_ns(),
            "reportReferences": list(report_references),
            "artifacts": dict(artifacts or {}),
            "outputs": dict(outputs or {}),
            "actions": list(actions),
            "notes": list(notes),
            "needsInput": needs_input,
            "error": error,
        }
    )
    return outcome.model_copy(update={"contentSha256": _stage_checksum(outcome)})


def _start_attempt(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
    request_record: OrchestrationRequestRecord,
    parent_attempts: Sequence[WorkflowStageLink],
    *,
    inputs: Mapping[str, Any] | None = None,
    resume_record: OrchestrationResumeRecord | None = None,
) -> WorkflowStageAttempt:
    attempt_inputs = dict(inputs or {})
    if resume_record is not None:
        attempt_inputs["resumeLineage"] = {
            "resumeId": resume_record.resumeId,
            "answeredAttempt": (
                resume_record.answeredAttempt.model_dump(mode="json")
                if resume_record.answeredAttempt is not None
                else None
            ),
            "questionIds": list(resume_record.questionIds),
        }
    attempt = WorkflowStageAttempt(
        workflowRunId=workflow_run_id,
        stage=stage,
        attemptId=uuid.uuid4().hex,
        status="started",
        startedAtNs=time.time_ns(),
        requestSha256=request_record.requestSha256,
        configSha256=request_record.configSha256,
        parentAttempts=list(parent_attempts),
        inputs=attempt_inputs,
    )
    attempt = attempt.model_copy(update={"contentSha256": _stage_checksum(attempt)})
    _write_model_once(
        group,
        _stage_key(
            prefix,
            workflow_run_id,
            stage,
            attempt.attemptId,
            "started.json",
        ),
        attempt,
    )
    logger.info(
        f"Workflow {workflow_run_id}: started stage={stage!r} "
        f"attempt={attempt.attemptId}"
    )
    return attempt


def _save_outcome(
    group: zarr.Group,
    prefix: str,
    outcome: WorkflowStageAttempt,
) -> None:
    _write_model_once(
        group,
        _stage_key(
            prefix,
            outcome.workflowRunId,
            outcome.stage,
            outcome.attemptId,
            "outcome.json",
        ),
        outcome,
    )
    elapsed_seconds = (
        (outcome.completedAtNs - outcome.startedAtNs) / 1_000_000_000
        if outcome.completedAtNs is not None
        else 0.0
    )
    details = (
        f"reports={len(outcome.reportReferences)}, "
        f"artifacts={len(outcome.artifacts)}, actions={len(outcome.actions)}"
    )
    if outcome.status == "failed":
        error_kind = (outcome.error or "unknown error").partition(":")[0]
        logger.error(
            f"Workflow {outcome.workflowRunId}: stage={outcome.stage!r} "
            f"failed ({error_kind}; {details}; {elapsed_seconds:.1f}s)"
        )
    elif outcome.status == "needsInput":
        question_count = (
            len(outcome.needsInput.questions) if outcome.needsInput is not None else 0
        )
        logger.info(
            f"Workflow {outcome.workflowRunId}: stage={outcome.stage!r} paused "
            f"for {question_count} input question(s) ({details}; "
            f"{elapsed_seconds:.1f}s)"
        )
    else:
        logger.info(
            f"Workflow {outcome.workflowRunId}: completed stage={outcome.stage!r} "
            f"({details}; {elapsed_seconds:.1f}s)"
        )


def _stage_outcomes(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
) -> list[WorkflowStageAttempt]:
    stage_prefix = _stage_prefix(prefix, workflow_run_id, stage)
    outcomes: list[WorkflowStageAttempt] = []
    for key in _list_keys(group, stage_prefix):
        if not key.endswith("/outcome.json"):
            continue
        path_attempt_id = key.rsplit("/", 2)[-2]
        outcome = cast(
            WorkflowStageAttempt,
            _read_model(group, key, WorkflowStageAttempt),
        )
        if (
            outcome.workflowRunId != workflow_run_id
            or outcome.stage != stage
            or outcome.attemptId != path_attempt_id
        ):
            raise ValueError("Stage outcome identity does not match its path")
        if outcome.contentSha256 != _stage_checksum(outcome):
            raise ValueError("Stage outcome checksum does not match its content")
        outcomes.append(outcome)
    return sorted(outcomes, key=lambda value: (value.startedAtNs, value.attemptId))


def _stage_starts(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
) -> list[WorkflowStageAttempt]:
    stage_prefix = _stage_prefix(prefix, workflow_run_id, stage)
    starts: list[WorkflowStageAttempt] = []
    for key in _list_keys(group, stage_prefix):
        if not key.endswith("/started.json"):
            continue
        path_attempt_id = key.rsplit("/", 2)[-2]
        started = cast(
            WorkflowStageAttempt,
            _read_model(group, key, WorkflowStageAttempt),
        )
        if (
            started.workflowRunId != workflow_run_id
            or started.stage != stage
            or started.attemptId != path_attempt_id
            or started.status != "started"
        ):
            raise ValueError("Stage start identity does not match its path")
        if started.contentSha256 != _stage_checksum(started):
            raise ValueError("Stage start checksum does not match its content")
        starts.append(started)
    return sorted(starts, key=lambda value: (value.startedAtNs, value.attemptId))


def _has_resume_answer(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping | Sequence):
        return bool(value)
    return True


def _unsafe_context_resolution(value: Any) -> Literal["skip", "clarify"] | None:
    if value == "skipHarmony":
        return "skip"
    if not isinstance(value, Mapping):
        return None
    selection = value.get("selection")
    batch_correction = value.get("batchCorrection")
    if (
        selection == "skipHarmony"
        or value.get("batchCorrectionAction") == "skip"
        or batch_correction == "skip"
        or (
            isinstance(batch_correction, Mapping)
            and batch_correction.get("action") == "skip"
        )
    ):
        return "skip"
    clarification = value.get("clarification")
    if (
        selection in {None, "provideClarification"}
        and isinstance(clarification, str)
        and clarification.strip()
    ):
        return "clarify"
    return None


def _resume_answer_errors(
    paused: WorkflowStageAttempt,
    answers: Mapping[str, Any],
) -> list[str]:
    needs_input = paused.needsInput
    if needs_input is None:
        return ["The latest paused stage does not contain persisted questions"]
    questions = {question.questionId: question for question in needs_input.questions}
    supplied_ids = set(answers)
    expected_ids = set(questions)
    errors = [
        f"Unknown resume answer key {question_id!r}"
        for question_id in sorted(supplied_ids - expected_ids)
    ]
    errors.extend(
        f"Missing resume answer for {question_id!r}"
        for question_id in sorted(expected_ids - supplied_ids)
    )
    unsafe_context = (
        paused.stage == "experimental_context"
        and paused.outputs.get("unsafeBatchCorrection") is True
    )
    for question_id in sorted(expected_ids & supplied_ids):
        question = questions[question_id]
        answer = answers[question_id]
        if unsafe_context and question_id == "experimentalDirections":
            if _unsafe_context_resolution(answer) is None:
                errors.append(
                    "Experimental Context requires an explicit skipHarmony choice "
                    "or a non-empty provideClarification response"
                )
            continue
        if question.planChecksum is not None:
            if answer != question.planChecksum:
                errors.append(
                    f"Resume answer for {question_id!r} does not match the "
                    "persisted plan checksum"
                )
            continue
        if question.options and question_id in {
            "finalGraphOptionId",
            "primaryCoefficient",
        }:
            if not isinstance(answer, str) or answer not in question.options:
                errors.append(
                    f"Resume answer for {question_id!r} must be one of the "
                    f"persisted options {question.options!r}"
                )
            continue
        if not _has_resume_answer(answer):
            errors.append(f"Resume answer for {question_id!r} must be non-empty")
    return errors


def _validated_resume_record(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    resume_id: str,
) -> OrchestrationResumeRecord:
    record = cast(
        OrchestrationResumeRecord,
        _read_model(
            store.zw,
            _resume_key(prefix, workflow_run_id, resume_id),
            OrchestrationResumeRecord,
        ),
    )
    if record.workflowRunId != workflow_run_id or record.resumeId != resume_id:
        raise ValueError("Persisted resume record identity does not match its path")
    if record.contentSha256 != _record_checksum(record):
        raise ValueError("Persisted resume record checksum does not match its content")
    if record.answeredAttempt is None:
        if record.questionIds or record.answers:
            raise ValueError(
                "A resume without an answered attempt cannot contain question answers"
            )
        return record
    matches = [
        outcome
        for outcome in _stage_outcomes(
            store.zw,
            prefix,
            workflow_run_id,
            record.answeredAttempt.stage,
        )
        if outcome.attemptId == record.answeredAttempt.attemptId
        and outcome.contentSha256 == record.answeredAttempt.contentSha256
    ]
    if len(matches) != 1 or matches[0].status != "needsInput":
        raise ValueError(
            "Persisted resume record does not cite one paused stage attempt"
        )
    answered_outcome = matches[0]
    assert answered_outcome.needsInput is not None
    expected_question_ids = [
        question.questionId for question in answered_outcome.needsInput.questions
    ]
    if record.questionIds != expected_question_ids:
        raise ValueError("Persisted resume record question IDs are stale")
    answer_errors = _resume_answer_errors(answered_outcome, record.answers)
    if answer_errors:
        raise ValueError(
            "Persisted resume record contains invalid answers: "
            + "; ".join(answer_errors)
        )
    return record


def _validated_done_outcome(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
    request_record: OrchestrationRequestRecord,
    parent_attempts: Sequence[WorkflowStageLink],
    *,
    required_status: Literal["done", "needsInput"] = "done",
) -> WorkflowStageAttempt | None:
    """Return the newest lineage-matching stage whose persisted outputs resolve."""

    candidates = [
        value
        for value in _stage_outcomes(store.zw, prefix, workflow_run_id, stage)
        if value.status == required_status
    ]
    for outcome in reversed(candidates):
        if (
            outcome.requestSha256 != request_record.requestSha256
            or outcome.configSha256 != request_record.configSha256
        ):
            raise ValueError("Stage request or configuration checksum is stale")
        if outcome.parentAttempts != list(parent_attempts):
            continue
        if _stage_outcome_resolves(
            store,
            prefix,
            workflow_run_id,
            request_record,
            outcome,
        ):
            logger.debug(
                f"Workflow {workflow_run_id}: reusing stage={stage!r} "
                f"attempt={outcome.attemptId} status={required_status!r}"
            )
            return outcome
    return None


def _stage_outcome_resolves(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    request_record: OrchestrationRequestRecord,
    outcome: WorkflowStageAttempt,
) -> bool:
    """Validate one exact stage record and every reference in its journal."""
    if (
        outcome.requestSha256 != request_record.requestSha256
        or outcome.configSha256 != request_record.configSha256
    ):
        raise ValueError("Stage request or configuration checksum is stale")
    if outcome.status == "started":
        raise ValueError("A stage outcome cannot retain started status")
    started_key = _stage_key(
        prefix,
        workflow_run_id,
        outcome.stage,
        outcome.attemptId,
        "started.json",
    )
    started = cast(
        WorkflowStageAttempt,
        _read_model(store.zw, started_key, WorkflowStageAttempt),
    )
    if (
        started.status != "started"
        or started.contentSha256 != _stage_checksum(started)
        or started.workflowRunId != workflow_run_id
        or started.stage != outcome.stage
        or started.attemptId != outcome.attemptId
        or started.startedAtNs != outcome.startedAtNs
        or started.requestSha256 != outcome.requestSha256
        or started.configSha256 != outcome.configSha256
        or started.parentAttempts != outcome.parentAttempts
        or started.inputs != outcome.inputs
    ):
        raise ValueError("Stage started and outcome records do not match")
    for parent in outcome.parentAttempts:
        matches = [
            value
            for value in _stage_outcomes(
                store.zw,
                prefix,
                workflow_run_id,
                parent.stage,
            )
            if value.attemptId == parent.attemptId
        ]
        if (
            len(matches) != 1
            or matches[0].status != "done"
            or matches[0].contentSha256 != parent.contentSha256
        ):
            return False
    try:
        for reference in outcome.reportReferences:
            if reference.workflowRunId != workflow_run_id:
                raise ValueError("Stage report belongs to a different workflow")
            load_agent_report(store, reference)
        for artifact_reference in outcome.artifacts.values():
            store.load_artifact(artifact_model_to_ref(artifact_reference))
        metadata_columns = outcome.outputs.get("metadataColumns", [])
        if not isinstance(metadata_columns, list) or any(
            not isinstance(column, str) or column not in store.cells.columns
            for column in metadata_columns
        ):
            raise ValueError("Stage metadata columns no longer resolve")
    except (KeyError, FileNotFoundError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _parent_link(outcome: WorkflowStageAttempt) -> WorkflowStageLink:
    return WorkflowStageLink(
        stage=outcome.stage,
        attemptId=outcome.attemptId,
        contentSha256=outcome.contentSha256,
    )


def _safe_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return clean or "assay"


def _report_link(reference: AgentReportReference) -> AgentReportLink:
    return AgentReportLink.from_reference(reference)


def _stage_execution_id(started: WorkflowStageAttempt) -> str:
    """Return the stable identity of one logical agent-stage invocation."""
    inputs = dict(started.inputs)
    inputs.pop("resumeLineage", None)
    payload = {
        "workflowRunId": started.workflowRunId,
        "stage": started.stage,
        "requestSha256": started.requestSha256,
        "configSha256": started.configSha256,
        "parentAttempts": [
            value.model_dump(mode="json") for value in started.parentAttempts
        ],
        "inputs": inputs,
    }
    digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
    return f"orchestrator_{started.stage}_{digest[:40]}"


def _stage_invocation(
    started: WorkflowStageAttempt,
    invocation: AgentInvocation,
) -> AgentInvocation:
    execution_id = _stage_execution_id(started)
    inputs = dict(invocation.inputs)
    observed = inputs.get("orchestrationExecutionId")
    if observed is not None and observed != execution_id:
        raise ValueError("Agent invocation has a conflicting orchestration identity")
    inputs["orchestrationExecutionId"] = execution_id
    return invocation.model_copy(update={"inputs": inputs})


def _recover_persisted_stage_report(
    store: DataStore,
    started: WorkflowStageAttempt,
    *,
    agent_name: AgentName,
    expected_type: type[AgentDataModel],
) -> tuple[AgentDataModel, AgentReportReference] | None:
    """Recover a report committed before its stage outcome was persisted."""
    execution_id = _stage_execution_id(started)
    matches = [
        reference
        for reference in list_agent_reports(
            store,
            started.workflowRunId,
            agent_name=agent_name,
        )
        if reference.agentRunId == execution_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("A logical stage execution has multiple persisted reports")
    reference = matches[0]
    record = load_agent_record(store, reference)
    if record.invocation.inputs.get("orchestrationExecutionId") != execution_id:
        raise ValueError("Persisted stage report has a stale execution identity")
    report = load_agent_report(store, reference)
    if not isinstance(report, expected_type):
        raise TypeError(
            f"Persisted {agent_name!r} report is not {expected_type.__name__}"
        )
    for artifact in record.invocation.artifacts.values():
        store.load_artifact(artifact_model_to_ref(artifact))
    logger.info(
        f"Workflow {started.workflowRunId}: recovered {agent_name!r} report for "
        f"stage={started.stage!r}"
    )
    return report, reference


def _save_stage_report(
    store: DataStore,
    started: WorkflowStageAttempt,
    report: AgentDataModel,
    *,
    invocation: AgentInvocation,
    expected_type: type[AgentDataModel],
) -> tuple[AgentDataModel, AgentReportReference]:
    """Persist a stage report under its stable logical execution identity."""
    tagged_invocation = _stage_invocation(started, invocation)
    try:
        reference = save_agent_report(
            store,
            started.workflowRunId,
            cast(AgentReport, report),
            invocation=tagged_invocation,
            agent_run_id=_stage_execution_id(started),
        )
        logger.info(
            f"Workflow {started.workflowRunId}: persisted "
            f"{tagged_invocation.agentName!r} report for stage={started.stage!r}"
        )
        return report, reference
    except FileExistsError:
        recovered = _recover_persisted_stage_report(
            store,
            started,
            agent_name=tagged_invocation.agentName,
            expected_type=expected_type,
        )
        if recovered is None:
            raise
        logger.debug(
            f"Workflow {started.workflowRunId}: reused concurrently persisted "
            f"{tagged_invocation.agentName!r} report"
        )
        return recovered


def _load_terminal_result(
    store: DataStore,
    prefix: str,
    workflow: AgentWorkflowRun,
) -> AutomatedWorkflowResult | None:
    raw = record_io.read_key(
        store.zw,
        _result_key(prefix, workflow.workflowRunId),
    )
    if raw is None:
        return None
    try:
        result = AutomatedWorkflowResult.model_validate_json(raw)
    except ValueError as exc:
        raise ValueError("Malformed automated workflow result") from exc
    if result.contentSha256 != _record_checksum(result):
        raise ValueError("Automated workflow result checksum is invalid")
    if result.workflowRun is None:
        raise ValueError("Terminal workflow result is missing its workflow identity")
    stored_workflow = result.workflowRun
    identity_fields = (
        "workflowRunId",
        "workspace",
        "status",
        "finalizedAtNs",
        "finalizationMessage",
        "analysisStore",
        "datasetFingerprints",
    )
    if any(
        getattr(stored_workflow, field) != getattr(workflow, field)
        for field in identity_fields
    ):
        raise ValueError("Automated workflow result has stale workflow metadata")
    if result.status != workflow.status:
        raise ValueError("Automated workflow result has a stale terminal status")
    if result.reportReferences != workflow.reports:
        raise ValueError("Automated workflow result has stale report references")
    return result


def _persist_terminal_result(
    store: DataStore,
    prefix: str,
    workflow: AgentWorkflowRun,
    result: AutomatedWorkflowResult,
) -> AutomatedWorkflowResult:
    existing = _load_terminal_result(store, prefix, workflow)
    if existing is not None:
        logger.debug(
            f"Workflow {workflow.workflowRunId}: reused terminal result "
            f"status={workflow.status!r}"
        )
        return existing
    if result.contentSha256 != _record_checksum(result):
        raise ValueError("Terminal result must carry its exact content checksum")
    try:
        _write_model_once(
            store.zw,
            _result_key(prefix, workflow.workflowRunId),
            result,
        )
    except FileExistsError:
        pass
    persisted = _load_terminal_result(store, prefix, workflow)
    if persisted is None:
        raise RuntimeError("Terminal workflow result was not persisted")
    logger.info(
        f"Workflow {workflow.workflowRunId}: persisted terminal result "
        f"status={workflow.status!r}"
    )
    return persisted


def load_stage_report(
    store: DataStore,
    outcome: WorkflowStageAttempt,
    expected_type: type[AgentDataModel],
) -> AgentDataModel:
    references = list(outcome.reportReferences)
    if outcome.stage == "parameter_tuning" and len(references) > 1:
        execution_id = _stage_execution_id(outcome)
        references = [
            reference
            for reference in references
            if reference.agentRunId == execution_id
        ]
    if len(references) != 1:
        raise ValueError(
            f"Stage {outcome.stage!r} must have exactly one report reference"
        )
    report = load_agent_report(store, references[0])
    if not isinstance(report, expected_type):
        raise TypeError(
            f"Stage {outcome.stage!r} report is not {expected_type.__name__}"
        )
    return report


def failed_stage(
    store: DataStore,
    workflow: AgentWorkflowRun,
    request_record: OrchestrationRequestRecord,
    stage: WorkflowStageName,
    parents: Sequence[WorkflowStageLink],
    error: str,
    *,
    resume_record: OrchestrationResumeRecord | None = None,
) -> WorkflowStageAttempt:
    logger.error(
        f"Workflow {workflow.workflowRunId}: stage={stage!r} failed validation"
    )
    prefix = _ensure_orchestration_store(store)
    started = _start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        stage,
        request_record,
        parents,
        resume_record=resume_record,
    )
    outcome = _complete_attempt(started, status="failed", error=error)
    _save_outcome(store.zw, prefix, outcome)
    finalize_failed(store, workflow, error)
    return outcome


def finish_exception(
    store: DataStore,
    prefix: str,
    workflow: AgentWorkflowRun,
    started: WorkflowStageAttempt,
    exc: BaseException,
    *,
    artifacts: Mapping[str, ArtifactReferenceModel] | None = None,
    actions: Sequence[str] = (),
    outputs: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> WorkflowStageAttempt:
    error = f"{type(exc).__name__}: {exc}"
    logger.error(
        f"Workflow {workflow.workflowRunId}: stage={started.stage!r} raised "
        f"{type(exc).__name__}; details were persisted in the stage outcome"
    )
    execution_id = _stage_execution_id(started)
    report_references = [
        reference
        for reference in list_agent_reports(store, workflow.workflowRunId)
        if reference.agentRunId == execution_id
        or reference.agentRunId.startswith(f"{execution_id}_integration_")
    ]
    outcome = _complete_attempt(
        started,
        status="failed",
        report_references=report_references,
        artifacts=artifacts,
        outputs=outputs,
        actions=actions,
        notes=notes,
        error=error,
    )
    _save_outcome(store.zw, prefix, outcome)
    finalize_failed(store, workflow, error)
    return outcome


def finalize_failed(
    store: DataStore,
    workflow: AgentWorkflowRun,
    message: str,
) -> None:
    current = load_agent_workflow(store, workflow.workflowRunId)
    if current.status == "running":
        logger.warning(f"Finalizing workflow {workflow.workflowRunId} as failed")
        finalize_agent_workflow(
            store,
            workflow.workflowRunId,
            status="failed",
            message=message,
        )


def all_report_references(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
) -> list[AgentReportReference]:
    references: dict[tuple[str, str], AgentReportReference] = {}
    for stage in _STAGE_ORDER:
        for outcome in _stage_outcomes(store.zw, prefix, workflow_run_id, stage):
            for reference in outcome.reportReferences:
                if reference.workflowRunId != workflow_run_id:
                    raise ValueError("Stage report belongs to a different workflow")
                load_agent_report(store, reference)
                references[(reference.agentName, reference.agentRunId)] = reference
    return sorted(
        references.values(),
        key=lambda value: (value.createdAtNs, value.agentName, value.agentRunId),
    )


def paused_or_failed_result(
    store: DataStore,
    workflow: AgentWorkflowRun,
    request_record: OrchestrationRequestRecord,
    outcome: WorkflowStageAttempt,
    *,
    preprocessing_plan: AutomatedPreprocessingPlan | None = None,
    final_analysis: FinalAnalysisHandoff | None = None,
) -> AutomatedWorkflowResult:
    prefix = _ensure_orchestration_store(store)
    current = load_agent_workflow(store, workflow.workflowRunId)
    status: AutomatedWorkflowStatus = (
        "needsInput" if outcome.status == "needsInput" else "failed"
    )
    result = AutomatedWorkflowResult(
        status=status,
        currentStage=outcome.stage,
        zarrPath=str(store.zarr_loc),
        workflowRun=current,
        reportReferences=list(current.reports),
        preprocessingPlan=preprocessing_plan,
        finalAnalysis=final_analysis,
        needsInput=outcome.needsInput,
        notes=[*outcome.notes, *([outcome.error] if outcome.error else [])],
    )
    result = result.model_copy(update={"contentSha256": _record_checksum(result)})
    if status == "failed":
        return _persist_terminal_result(store, prefix, current, result)
    logger.info(
        f"Workflow {workflow.workflowRunId}: returning needsInput at "
        f"stage={outcome.stage!r}"
    )
    return result
