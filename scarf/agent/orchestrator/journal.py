"""One immutable RNA workflow history containing stage evidence and decisions."""

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from .. import record_io
from ..experimental_context.study import StudyContract
from ..types import AgentDataModel, AgentRunInfo, ArtifactReferenceModel
from .models import (
    _STAGE_ORDER,
    AutomatedWorkflowConfig,
    AutomatedWorkflowResult,
    FinalAnalysisHandoff,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    StageEvidenceReference,
    WorkflowIdentity,
    WorkflowNeedsInput,
    WorkflowStageAttempt,
    WorkflowStageLink,
    WorkflowStageName,
    artifact_model_to_ref,
)

_INCOMPATIBLE = (
    "Unsupported saved agent workflow. Start a new RNA workflow with this release; "
    "older requests cannot be resumed or regenerated. Existing analysis artifacts "
    "remain accessible through Scarf's artifact APIs."
)


def _sha256_model(value: AgentDataModel) -> str:
    return hashlib.sha256(
        record_io.canonical_json_bytes(value.model_dump(mode="json"))
    ).hexdigest()


def model_attempt_callback(
    store: Any,
    prefix: str,
    workflow_run_id: str,
    key: str,
    inputs: Mapping[str, Any],
) -> Callable[[AgentRunInfo], None]:
    """Save provider attempts beside their exact owning scientific evidence."""
    identity = hashlib.sha256(record_io.canonical_json_bytes(dict(inputs))).hexdigest()

    def save(run_info: AgentRunInfo) -> None:
        save_checkpoint(
            store,
            prefix,
            workflow_run_id,
            f"{key}/model_attempts/{uuid.uuid4().hex}",
            inputs={"evidenceSha256": identity},
            outputs={
                "runInfo": run_info.model_dump(mode="json"),
                "recordedAtNs": time.time_ns(),
            },
        )

    return save


@contextmanager
def diagnostic_attempt(
    store: Any,
    prefix: str,
    workflow_run_id: str,
    inputs: Mapping[str, Any],
) -> Iterator[dict[str, dict[str, int]]]:
    """Record observed diagnostic calls, including interrupted and repeated work."""
    from ..parameter_tuning.execution import diagnostic_work

    key = f"parameter_tuning/diagnostic_attempts/{uuid.uuid4().hex}"
    identity = dict(inputs)
    started = time.time_ns()
    save_checkpoint(
        store,
        prefix,
        workflow_run_id,
        key + "/started",
        identity,
        {"recordedAtNs": started},
    )
    with diagnostic_work() as counts:
        status = "completed"
        error = None
        failure: BaseException | None = None
        try:
            yield counts
        except BaseException as exc:
            failure = exc
            status = "failed" if isinstance(exc, Exception) else "interrupted"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                logger.info(
                    f"Diagnostic work: {sum(row['completed'] for row in counts.values())}/"
                    f"{sum(row['attempted'] for row in counts.values())} "
                    "observed calls completed/attempted, including "
                    f"{counts.get('core.doubletDetection', {}).get('attempted', 0)} "
                    "doublet-scoring calls; "
                    f"{sum(row['cacheHits'] for row in counts.values())} metric cache hits, "
                    f"{sum(row['restored'] for row in counts.values())} saved-evidence restores. "
                    "Composite diagnostics include their underlying calls; these counts "
                    "do not measure internal rebuilds or simulated-doublet writes."
                )
                save_checkpoint(
                    store,
                    prefix,
                    workflow_run_id,
                    key + "/finished",
                    identity,
                    {
                        "recordedAtNs": time.time_ns(),
                        "elapsedSeconds": (time.time_ns() - started) / 1e9,
                        "status": status,
                        "error": error,
                        "operations": counts,
                        "interpretation": (
                            "Counts are observed agent calls and explicit reuse events. "
                            "Composite diagnostics include underlying calls; elapsed time "
                            "covers the tuning invocation, including model waiting. "
                            "A core call may reuse artifacts or still repeat internal work; "
                            "internal numerical rebuilds and simulated-doublet writes are "
                            "not observable from returned references. Historical attempts "
                            "without these records have unknown operation counts."
                        ),
                    },
                )
            except BaseException as persistence_error:
                if failure is None:
                    raise
                failure.add_note(
                    "Saving diagnostic work also failed: "
                    f"{type(persistence_error).__name__}: {persistence_error}"
                )


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


def _checkpoint_key(prefix: str, workflow_run_id: str, key: str) -> str:
    parts = key.split("/")
    if not parts or any(
        re.fullmatch(r"[A-Za-z0-9_.:-]+", part) is None or part in {".", ".."}
        for part in parts
    ):
        raise ValueError("Checkpoint keys must contain safe, non-empty path components")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", workflow_run_id) is None:
        raise ValueError("Invalid workflow identifier")
    return record_io.join_key(prefix, workflow_run_id, "checkpoints", key + ".json")


def read_checkpoint(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    key: str,
) -> dict[str, Any] | None:
    """Read an exact journal checkpoint with its validated inputs and outputs."""
    raw = record_io.read_key(store.zw, _checkpoint_key(prefix, workflow_run_id, key))
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "inputs",
        "outputs",
        "contentSha256",
    }:
        raise ValueError(
            "Unsupported RNA checkpoint contract; start a new workflow. Existing analysis artifacts remain accessible."
        )
    payload = {"inputs": value["inputs"], "outputs": value["outputs"]}
    digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
    if value["contentSha256"] != digest:
        raise ValueError("RNA checkpoint checksum does not match its contents")
    if not isinstance(value["inputs"], dict) or not isinstance(value["outputs"], dict):
        raise ValueError("RNA checkpoint inputs and outputs must be mappings")
    return value


def load_checkpoint(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    key: str,
    inputs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Read committed outputs only when the exact scientific inputs agree."""
    value = read_checkpoint(store, prefix, workflow_run_id, key)
    if value is None:
        return None
    if inputs is not None and record_io.canonical_json_bytes(
        inputs
    ) != record_io.canonical_json_bytes(value["inputs"]):
        raise ValueError(f"Checkpoint {key!r} has different scientific inputs")
    return cast(dict[str, Any], value["outputs"])


def save_checkpoint(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    key: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit evidence or admission before its dependent work; exact replay is idempotent."""
    payload = {"inputs": dict(inputs), "outputs": dict(outputs)}
    value = {
        **payload,
        "contentSha256": hashlib.sha256(
            record_io.canonical_json_bytes(payload)
        ).hexdigest(),
    }
    path = _checkpoint_key(prefix, workflow_run_id, key)
    try:
        _write_key_once(store.zw, path, record_io.display_json_bytes(value))
    except FileExistsError:
        existing = load_checkpoint(store, prefix, workflow_run_id, key, inputs)
        if existing != outputs:
            raise ValueError(
                f"Checkpoint {key!r} already contains a different outcome"
            ) from None
        return existing
    return dict(outputs)


def _orchestration_prefix(store: DataStore) -> str:
    root_path = str(getattr(store.zw, "path", "")).strip("/")
    return record_io.join_key(root_path, "agents", "orchestrations")


def _request_key(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(prefix, workflow_run_id, "request.json")


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
        hint = (
            "; recreate the request with the current AutomatedWorkflowConfig "
            "and start a new workflow. Older saved request/config shapes are "
            "unsupported and are not migrated"
            if model_type is OrchestrationRequestRecord
            else ""
        )
        raise ValueError(f"Malformed orchestration record {key!r}{hint}") from exc


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
    status: Literal["done", "needsInput", "abstained", "failed"],
    report_references: Sequence[StageEvidenceReference] = (),
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
        attempt_inputs["resumeAnswers"] = dict(resume_record.answers)
        attempt_inputs["answeredAttempt"] = (
            resume_record.answeredAttempt.model_dump(mode="json")
            if resume_record.answeredAttempt is not None
            else None
        )
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
    logger.info(f"{stage.replace('_', ' ').capitalize()}: started")
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
    label = outcome.stage.replace("_", " ").capitalize()
    if outcome.status == "failed":
        logger.error(f"{label}: {outcome.error} ({elapsed_seconds:.1f}s)")
    elif outcome.status in {"needsInput", "abstained"}:
        reasons = "; ".join(outcome.notes)
        if outcome.needsInput is not None:
            reasons = "; ".join(q.question for q in outcome.needsInput.questions)
        logger.warning(f"{label}: {outcome.status}: {reasons}")
    else:
        logger.info(f"{label}: completed ({elapsed_seconds:.1f}s)")
    logger.debug(f"Workflow {outcome.workflowRunId}, attempt {outcome.attemptId}")


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
        if question.decisionId is not None:
            if not isinstance(answer, Mapping):
                errors.append(
                    f"Resume answer for {question_id!r} must contain decisionId, "
                    "optionId, and rationale"
                )
                continue
            if set(answer) != {"decisionId", "optionId", "rationale"}:
                errors.append(
                    f"Resume answer for {question_id!r} must contain exactly "
                    "decisionId, optionId, and rationale"
                )
                continue
            if answer.get("decisionId") != question.decisionId:
                errors.append(
                    f"Resume answer for {question_id!r} does not match decision "
                    f"{question.decisionId!r}"
                )
            option_id = answer.get("optionId")
            if not isinstance(option_id, str) or option_id not in question.options:
                errors.append(
                    f"Resume answer for {question_id!r} must select one persisted "
                    f"option {question.options!r}"
                )
            rationale = answer.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append(
                    f"Resume answer for {question_id!r} requires a non-empty rationale"
                )
            continue
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


def _validated_done_outcome(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    stage: WorkflowStageName,
    request_record: OrchestrationRequestRecord,
    parent_attempts: Sequence[WorkflowStageLink],
    *,
    required_status: Literal["done", "needsInput", "failed"] = "done",
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
            read_stage_evidence(store, reference)
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


def _stage_execution_id(started: WorkflowStageAttempt) -> str:
    """Return the stable identity of one logical agent-stage invocation."""
    inputs = dict(started.inputs)
    inputs.pop("answeredAttempt", None)
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


def _ensure_orchestration_store(store: DataStore) -> str:
    """Initialize only the journal namespace, never a second workflow ledger."""
    agents = store.zw.require_group("agents")
    if "orchestrations" not in agents:
        agents.create_group(
            "orchestrations",
            attributes={"format": "scarf_agent_orchestrations", "format_version": 2},
        )
    node = agents["orchestrations"]
    if not isinstance(node, zarr.Group) or dict(node.attrs) != {
        "format": "scarf_agent_orchestrations",
        "format_version": 2,
    }:
        raise ValueError(_INCOMPATIBLE)
    return _orchestration_prefix(store)


def read_request(
    group: zarr.Group, prefix: str, workflow_run_id: str
) -> OrchestrationRequestRecord:
    try:
        value = cast(
            OrchestrationRequestRecord,
            _read_model(
                group, _request_key(prefix, workflow_run_id), OrchestrationRequestRecord
            ),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(_INCOMPATIBLE) from exc
    if (
        value.workflowRunId != workflow_run_id
        or value.requestSha256 != _sha256_model(value.request)
        or value.configSha256 != _sha256_model(value.config)
        or value.contentSha256 != _record_checksum(value)
    ):
        raise ValueError("RNA workflow request identity or checksum is invalid")
    return value


def open_analysis_store(
    target: str | Path, workflow_run_id: str, *, workspace: str | None = None
) -> DataStore:
    """Open the journal's selected RNA assay read-only, before resolving any result."""
    from ...storage.schema import validate_workspace_name

    validate_workspace_name(workspace)
    root = zarr.open_group(str(target), mode="r")
    active = root if workspace is None else root[workspace]
    if not isinstance(active, zarr.Group):
        raise ValueError("Analysis workspace is not a group")
    prefix = record_io.join_key(active.path, "agents", "orchestrations")
    record = read_request(active, prefix, workflow_run_id)
    if record.request.zarrPath is None:
        raise ValueError("Saved workflow has no store path")
    if (
        record.request.workspace != workspace
        or Path(record.request.zarrPath).resolve() != Path(target).resolve()
    ):
        raise ValueError("Analysis address does not match the saved request")
    if not record.request.primaryAssay:
        raise ValueError("Saved analysis has no selected RNA assay")
    return DataStore(
        str(target),
        workspace=workspace,
        default_assay=record.request.primaryAssay,
        zarr_mode="r",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
    )


def _report_checkpoint(started: WorkflowStageAttempt) -> str:
    return f"{started.stage}/report/{_stage_execution_id(started)}"


def _recover_persisted_stage_report(
    store: DataStore,
    started: WorkflowStageAttempt,
    *,
    expected_type: type[AgentDataModel],
) -> tuple[AgentDataModel, StageEvidenceReference] | None:
    key = _report_checkpoint(started)
    inputs = {"execution": _stage_execution_id(started)}
    data = load_checkpoint(
        store, _orchestration_prefix(store), started.workflowRunId, key, inputs
    )
    if data is None:
        return None
    if (
        set(data) != {"reportType", "report"}
        or data["reportType"] != expected_type.__name__
    ):
        raise ValueError("Stage evidence has a different scientific result type")
    report = expected_type.model_validate(data["report"])
    reference = StageEvidenceReference(
        workflowRunId=started.workflowRunId,
        stage=started.stage,
        key=key,
        contentSha256=_sha256_model(report),
    )
    return report, reference


def _save_stage_report(
    store: DataStore,
    started: WorkflowStageAttempt,
    report: AgentDataModel,
    *,
    expected_type: type[AgentDataModel],
    attempt_owned: bool = False,
) -> tuple[AgentDataModel, StageEvidenceReference]:
    report = expected_type.model_validate(report.model_dump(mode="json"))
    # Revisable stages recover individual evidence checkpoints across attempts;
    # their aggregate reports can change as additional evidence is completed.
    key = (
        f"{started.stage}/report/attempt_{started.attemptId}"
        if attempt_owned
        else _report_checkpoint(started)
    )
    data = {
        "reportType": expected_type.__name__,
        "report": report.model_dump(mode="json"),
    }
    save_checkpoint(
        store,
        _orchestration_prefix(store),
        started.workflowRunId,
        key,
        {"execution": _stage_execution_id(started)},
        data,
    )
    return report, StageEvidenceReference(
        workflowRunId=started.workflowRunId,
        stage=started.stage,
        key=key,
        contentSha256=_sha256_model(report),
    )


def read_stage_evidence(
    store: DataStore, reference: StageEvidenceReference
) -> dict[str, Any]:
    if not reference.key.startswith(reference.stage + "/report/"):
        raise ValueError("Evidence checkpoint is not owned by its declared stage")
    value = load_checkpoint(
        store,
        _orchestration_prefix(store),
        reference.workflowRunId,
        reference.key,
        None,
    )
    if value is None or set(value) != {"reportType", "report"}:
        raise ValueError("Stage evidence checkpoint is missing or malformed")
    report = value["report"]
    if (
        not isinstance(report, dict)
        or hashlib.sha256(record_io.canonical_json_bytes(report)).hexdigest()
        != reference.contentSha256
    ):
        raise ValueError("Stage evidence does not match its exact reference")
    return report


def load_stage_report(
    store: DataStore, outcome: WorkflowStageAttempt, expected_type: type[AgentDataModel]
) -> AgentDataModel:
    if len(outcome.reportReferences) != 1:
        raise ValueError("A scientific stage must own exactly one evidence report")
    return expected_type.model_validate(
        read_stage_evidence(store, outcome.reportReferences[0])
    )


def failed_stage(
    store: DataStore,
    workflow: WorkflowIdentity,
    request_record: OrchestrationRequestRecord,
    stage: WorkflowStageName,
    parents: Sequence[WorkflowStageLink],
    error: str,
    *,
    artifacts: Mapping[str, ArtifactReferenceModel] | None = None,
    resume_record: OrchestrationResumeRecord | None = None,
) -> WorkflowStageAttempt:
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
    outcome = _complete_attempt(
        started, status="failed", artifacts=artifacts, error=error
    )
    _save_outcome(store.zw, prefix, outcome)
    return outcome


def finish_exception(
    store: DataStore,
    prefix: str,
    workflow: WorkflowIdentity,
    started: WorkflowStageAttempt,
    exc: BaseException,
    *,
    artifacts: Mapping[str, ArtifactReferenceModel] | None = None,
    actions: Sequence[str] = (),
    outputs: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> WorkflowStageAttempt:
    from ..config.agent_exec import describe_agent_error

    error = describe_agent_error(exc)
    saved_outputs = dict(outputs or {})
    run_info = getattr(exc, "agent_run_info", None)
    if isinstance(run_info, AgentRunInfo):
        saved_outputs["runInfo"] = run_info.model_dump(mode="json")
    outcome = _complete_attempt(
        started,
        status="failed",
        artifacts=artifacts,
        outputs=saved_outputs,
        actions=actions,
        notes=notes,
        error=error,
    )
    _save_outcome(store.zw, prefix, outcome)
    return outcome


def paused_or_failed_result(
    store: DataStore,
    workflow: WorkflowIdentity,
    request_record: OrchestrationRequestRecord,
    outcome: WorkflowStageAttempt,
    *,
    study_contract: StudyContract | None = None,
) -> AutomatedWorkflowResult:
    status: Literal["needsInput", "abstained", "failed"] = (
        "needsInput"
        if outcome.status == "needsInput"
        else "abstained"
        if outcome.status == "abstained"
        else "failed"
    )
    questions = (
        [q.question for q in outcome.needsInput.questions] if outcome.needsInput else []
    )
    return AutomatedWorkflowResult(
        status=status,
        currentStage=outcome.stage,
        zarrPath=str(store.zarr_loc),
        workspace=workflow.workspace,
        workflowRunId=workflow.workflowRunId,
        needsInput=outcome.needsInput,
        notes=[*outcome.notes, *([outcome.error] if outcome.error else []), *questions],
        limitations=list(study_contract.limitations) if study_contract else [],
        unresolvedClaims=questions,
    )


def _analysis_review_views(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    stages: list[dict[str, Any]],
    config: AutomatedWorkflowConfig,
) -> list[dict[str, Any]]:
    """Derive bounded scientific views from the active exact review checkpoints."""
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stage in stages:
        if stage["stage"] != "parameter_tuning":
            continue
        history = stage["outputs"].get("tuningEvidence", {}).get("history", [])
        for entry in history:
            if "review" not in entry:
                continue
            key = entry.get("checkpointKey", "")
            match = re.fullmatch(
                r"parameter_tuning/(?:evidence_revisions/[a-f0-9]{64}/)?"
                r"(sample0|sample1|full)/review([0-9]+)(?:/answer)?",
                key,
            )
            if match is None or match[1] != entry.get("scope"):
                raise ValueError("Analysis review has an invalid checkpoint address")
            if key in seen:
                raise ValueError("Analysis review repeats a checkpoint")
            seen.add(key)
            limit = (
                config.maxFullPartitions
                if match[1] == "full"
                else config.maxScreeningEvaluations
            )
            if int(match[2]) > limit:
                raise ValueError("Analysis review exceeds its declared work allowance")
            value = read_checkpoint(store, prefix, workflow_run_id, key)
            if value is None:
                raise ValueError("Analysis review checkpoint is missing")
            if value["contentSha256"] != entry.get("checkpointSha256"):
                raise ValueError("Analysis review does not match its exact checkpoint")
            inputs, outputs = value["inputs"], value["outputs"]
            if (
                inputs.get("scope") != entry["scope"]
                or outputs.get("action") != entry["review"]
                or inputs.get("imageHashes") != entry.get("imageHashes")
                or inputs.get("evidenceMode") != entry.get("evidenceMode")
                or inputs.get("visualInspection") != entry.get("visualInspection")
            ):
                raise ValueError("Analysis review evidence bindings differ")
            mode, inspection = (
                inputs.get("evidenceMode"),
                inputs.get("visualInspection"),
            )
            if (
                mode not in {"visual", "structured"}
                or inspection != ("available" if mode == "visual" else "unavailable")
                or bool(inputs.get("imageHashes")) != (mode == "visual")
            ):
                raise ValueError(
                    "Analysis review has inconsistent evidence availability"
                )
            candidates = inputs.get("candidates", [])
            settings = inputs.get("settings", {})
            features = inputs.get("featureEvidence", {})
            candidate_ids = [item["candidateId"] for item in candidates]
            if (
                not 0 < len(candidates) <= limit
                or len(set(candidate_ids)) != len(candidate_ids)
                or set(settings) != set(candidate_ids)
                or set(features) != set(candidate_ids)
                or entry["review"].get("selectedCandidateId") not in candidate_ids
            ):
                raise ValueError("Analysis review candidate evidence does not align")
            for candidate in candidates:
                setting = settings[candidate["candidateId"]]
                if setting.get("parameters") != candidate.get(
                    "parameters"
                ) or setting.get("features") != candidate.get("artifacts", {}).get(
                    "graphFeatures"
                ):
                    raise ValueError(
                        "Analysis review settings do not match its artifacts"
                    )
            views.append(
                {
                    "scope": entry["scope"],
                    "evidenceMode": mode,
                    "visualInspection": inspection,
                    **entry["review"],
                    "coverage": inputs.get("coverage", {}),
                    "comparisonCoverage": inputs.get("comparisonCoverage"),
                    "populationSupport": inputs.get("assessmentContext", {}).get(
                        "populationSupport", {}
                    ),
                    "harmonyGates": inputs.get("harmonyGates", {}),
                    "candidates": [
                        {
                            name: item[name]
                            for name in (
                                "candidateId",
                                "parameters",
                                "metrics",
                                "cellSelection",
                                "artifacts",
                            )
                        }
                        for item in candidates
                    ],
                    "settings": {
                        identity: {
                            name: setting.get(name)
                            for name in (
                                "hvgCount",
                                "ranking",
                                "rankingColumn",
                                "features",
                                "eligibleFeatures",
                            )
                        }
                        for identity, setting in settings.items()
                    },
                    "featureEvidence": {
                        identity: {
                            name: evidence.get(name)
                            for name in (
                                "selectedGenes",
                                "eligibleGenes",
                                "families",
                                "topSelectedGenes",
                            )
                        }
                        for identity, evidence in features.items()
                    },
                }
            )
    return views


def _model_usage(
    model_attempts: Sequence[Mapping[str, Any]],
    saved_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count measured invocations once across callbacks and copied report views."""
    invocations: dict[str, AgentRunInfo] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            raw = value.get("runInfo")
            if isinstance(raw, Mapping):
                info = AgentRunInfo.model_validate(raw)
                # Empty, code-created outcomes are not model invocations.
                if (
                    info.runId
                    or info.modelName
                    or info.status
                    or any(
                        (
                            info.usage.requests,
                            info.usage.inputTokens,
                            info.usage.outputTokens,
                            info.usage.totalTokens,
                            info.usage.toolCalls,
                        )
                    )
                ):
                    identity = info.runId or _sha256_model(info)
                    invocations.setdefault(identity, info)
            for key, child in value.items():
                if key != "runInfo":
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    # Invocation callbacks own measured usage; reports can repeat these records.
    collect(list(model_attempts))
    collect(list(saved_views))
    infos = list(invocations.values())
    partial = sum(info.usage.availability == "partial" for info in infos)
    unavailable = sum(info.usage.availability == "unavailable" for info in infos)
    unspecified = sum(info.usage.availability is None for info in infos)
    known = any(
        info.usage.inputTokens or info.usage.outputTokens or info.usage.totalTokens
        for info in infos
    )
    return {
        "invocations": len(infos),
        "failedInvocations": sum(info.status == "failed" for info in infos),
        "validationRetries": sum(len(info.validationRetries) for info in infos),
        "durationSeconds": sum(info.durationSeconds for info in infos),
        **{
            field: sum(getattr(info.usage, field) for info in infos)
            for field in (
                "requests",
                "toolCalls",
                "inputTokens",
                "outputTokens",
                "totalTokens",
            )
        },
        "availability": (
            "reported"
            if infos and not (partial or unavailable or unspecified)
            else "partial"
            if known
            else "unavailable"
        ),
        "partialUsageInvocations": partial,
        "unavailableUsageInvocations": unavailable,
        "unspecifiedUsageInvocations": unspecified,
    }


def analysis_snapshot(store: DataStore, workflow_run_id: str) -> dict[str, Any]:
    """Validate one journal and derive its status, final artifacts, and report view."""
    prefix = _orchestration_prefix(store)
    request = read_request(store.zw, prefix, workflow_run_id)
    outcomes_by_stage = {
        stage: _stage_outcomes(store.zw, prefix, workflow_run_id, stage)
        for stage in _STAGE_ORDER
    }
    usage_views: list[dict[str, Any]] = [
        outcome.outputs
        for outcomes in outcomes_by_stage.values()
        for outcome in outcomes
    ]
    stages: list[dict[str, Any]] = []
    parents: list[WorkflowStageLink] = []
    final: dict[str, Any] | None = None
    status = "running"
    for stage in _STAGE_ORDER:
        outcomes = outcomes_by_stage[stage]
        matching = [v for v in outcomes if v.parentAttempts == parents]
        if not matching:
            break
        outcome = matching[-1]
        if not _stage_outcome_resolves(
            store, prefix, workflow_run_id, request, outcome
        ):
            raise ValueError(
                f"Stage {stage!r} contains unresolved artifact or evidence references"
            )
        stage_view = outcome.model_dump(mode="json")
        stage_view["report"] = (
            read_stage_evidence(store, outcome.reportReferences[0])
            if outcome.reportReferences
            else None
        )
        stage_view["decisions"] = []
        stages.append(stage_view)
        if outcome.status != "done":
            status = outcome.status
            break
        parents = [_parent_link(outcome)]
        if stage == "analysis_finalization":
            resolved = FinalAnalysisHandoff.model_validate(
                outcome.outputs["finalAnalysis"]
            )
            if (
                resolved.workflowRunId != workflow_run_id
                or resolved.primaryAssay != request.request.primaryAssay
                or resolved.markerAssay != resolved.primaryAssay
            ):
                raise ValueError("Final analysis belongs to a different RNA workflow")
            for name, kind in {
                "cellSelection": "cell_selection",
                "graph": "connectivity_map",
                "clusters": "cluster_labels",
                "umap": "embedding",
                "embeddingInitialization": "embedding_initialization",
                "markerFeatures": "feature_selection",
                "markers": "marker_table",
            }.items():
                ref = getattr(resolved, name)
                if (
                    ref is None
                    or ref.kind != kind
                    or ref != outcome.artifacts.get(name)
                ):
                    raise ValueError(
                        f"Final analysis lacks its validated {name} artifact"
                    )
                if name != "cellSelection" and ref.assay != resolved.primaryAssay:
                    raise ValueError(f"Final {name} belongs to another assay")
            inputs = outcome.inputs.get("preprocessedAssays", [])
            assert resolved.cellSelection is not None
            if len(inputs) != 1 or inputs[0].get(
                "cellSelection"
            ) != resolved.cellSelection.model_dump(mode="json"):
                raise ValueError(
                    "Final analysis does not use the full preprocessing cohort"
                )
            final = resolved.model_dump(mode="json")
            status = "completed"
    # Decisions belong to this same history and retain their offered evidence.
    checkpoint_prefix = record_io.join_key(prefix, workflow_run_id, "checkpoints")
    decisions: list[dict[str, Any]] = []
    model_attempts: list[dict[str, Any]] = []
    diagnostic_attempts: dict[str, dict[str, Any]] = {}
    for path in _list_keys(store.zw, checkpoint_prefix):
        if not path.endswith(".json") or not any(
            part in path
            for part in (
                "/decisions/",
                "/model_attempts/",
                "/report/",
                "/diagnostic_attempts/",
            )
        ):
            continue
        key = path[len(checkpoint_prefix) + 1 : -5]
        checkpoint = read_checkpoint(store, prefix, workflow_run_id, key)
        value = checkpoint["outputs"] if checkpoint is not None else None
        if value is not None:
            usage_views.append(value)
        if value is not None and "record" in value:
            decisions.append(value)
        if value is not None and "/model_attempts/" in path:
            run_info = AgentRunInfo.model_validate(value["runInfo"])
            model_attempts.append(
                {
                    "checkpoint": key,
                    "recordedAtNs": value.get("recordedAtNs"),
                    "runInfo": run_info.model_dump(mode="json"),
                }
            )
        if value is not None and "/diagnostic_attempts/" in path:
            assert checkpoint is not None
            attempt, _, phase = key.rpartition("/")
            if phase not in {"started", "finished"}:
                raise ValueError("Unknown diagnostic attempt checkpoint")
            entry = diagnostic_attempts.setdefault(
                attempt, {"checkpoint": attempt, "inputs": checkpoint["inputs"]}
            )
            if entry["inputs"] != checkpoint["inputs"]:
                raise ValueError(
                    "Diagnostic attempt inputs changed between start and finish"
                )
            entry[phase] = value

    if any(
        "finished" in row and "started" not in row
        for row in diagnostic_attempts.values()
    ):
        raise ValueError("Completed diagnostic attempt has no matching start record")

    def contains(value: Any, digest: str) -> bool:
        if isinstance(value, Mapping):
            return any(contains(v, digest) for v in value.values())
        if isinstance(value, list):
            return any(contains(v, digest) for v in value)
        return bool(value == digest)

    for value in decisions:
        digest = value.get("checkpointSha256")
        if not isinstance(digest, str):
            raise ValueError("Decision checkpoint has no input identity")
        owner = next(
            (
                stage
                for stage in stages
                if stage["stage"] == value.get("stage")
                and (
                    contains(stage["inputs"], digest)
                    or contains(stage["outputs"], digest)
                )
            ),
            None,
        )
        if owner is not None:
            owner["decisions"].append(value)
    reviews = _analysis_review_views(
        store, prefix, workflow_run_id, stages, request.config
    )
    if status == "completed":
        assert final is not None
        tuning = next(stage for stage in stages if stage["stage"] == "parameter_tuning")
        report = tuning["report"]
        full_reviews = [value for value in reviews if value["scope"] == "full"]
        if (
            not full_reviews
            or full_reviews[-1]["action"] != "accept"
            or full_reviews[-1]["selectedCandidateId"]
            != report.get("recommendedCandidateId")
            or report.get("finalClusterArtifact") != final["clusters"]
        ):
            raise ValueError(
                "Final analysis lacks its exact full-cohort acceptance evidence"
            )
    return {
        "runId": workflow_run_id,
        "status": status,
        "request": request.request.model_dump(mode="json"),
        "stages": stages,
        "modelAttempts": sorted(
            model_attempts, key=lambda row: row["recordedAtNs"] or 0
        ),
        "modelUsage": _model_usage(model_attempts, usage_views),
        "diagnosticAttempts": sorted(
            diagnostic_attempts.values(),
            key=lambda row: row.get("started", {}).get("recordedAtNs", 0),
        ),
        "finalAnalysis": final,
        "modelIdentity": request.modelIdentity,
        "analysisReviews": reviews,
        "config": request.config.model_dump(mode="json"),
    }
