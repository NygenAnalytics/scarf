"""Focused edge coverage for orchestration journal contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
import zarr
from pydantic import ValidationError
from zarr.storage import MemoryStore

import scarf.agent.orchestrator.journal as journal_module
from scarf.agent.orchestrator import (
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    AutomatedWorkflowResumeRequest,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
)
from scarf.agent.orchestrator.models import (
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
)
from scarf.agent.persistence import (
    AgentInvocation,
    AgentReportReference,
    AgentWorkflowRun,
)


def _started(
    *,
    stage: str = "ingest",
    parents: list[WorkflowStageLink] | None = None,
    inputs: dict[str, Any] | None = None,
) -> WorkflowStageAttempt:
    attempt = WorkflowStageAttempt(
        workflowRunId="workflow-1",
        stage=stage,
        attemptId="attempt-1",
        status="started",
        startedAtNs=1,
        requestSha256="a" * 64,
        configSha256="b" * 64,
        parentAttempts=parents or [],
        inputs=inputs or {},
    )
    return attempt.model_copy(
        update={"contentSha256": journal_module._stage_checksum(attempt)}
    )


def _request_record() -> OrchestrationRequestRecord:
    return OrchestrationRequestRecord(
        workflowRunId="workflow-1",
        requestSha256="a" * 64,
        configSha256="b" * 64,
    )


def _with_checksum(record: Any) -> Any:
    return record.model_copy(
        update={"contentSha256": journal_module._record_checksum(record)}
    )


def _terminal_workflow() -> AgentWorkflowRun:
    return AgentWorkflowRun(
        workflowRunId="workflow-1",
        createdAtNs=1,
        finalizedAtNs=2,
        status="completed",
        finalizationMessage="completed",
        analysisStore="analysis.zarr",
        datasetFingerprints={"RNA": "fingerprint"},
    )


def _terminal_result(workflow: AgentWorkflowRun) -> AutomatedWorkflowResult:
    return _with_checksum(
        AutomatedWorkflowResult(
            status="completed",
            currentStage="biological_interpretation",
            zarrPath="analysis.zarr",
            workflowRun=workflow,
            reportReferences=list(workflow.reports),
        )
    )


def test_orchestration_model_validation_edges() -> None:
    for field, value, message in (
        ("attemptId", "UPPER CASE", "lowercase run identifier"),
        ("contentSha256", "not-a-checksum", "lowercase SHA-256"),
    ):
        with pytest.raises(ValidationError, match=message):
            WorkflowStageLink(**{field: value})

    invalid_attempts = (
        ({"status": "started", "completedAtNs": 1}, "cannot have completedAtNs"),
        (
            {"status": "done", "startedAtNs": 2, "completedAtNs": 1},
            "must not precede",
        ),
        ({"status": "needsInput"}, "require questions"),
        ({"status": "failed"}, "require an error"),
    )
    for updates, message in invalid_attempts:
        with pytest.raises(ValidationError, match=message):
            WorkflowStageAttempt(**updates)

    invalid_requests = (
        ({"sourcePath": " ", "studyContext": "study"}, "sourcePath"),
        ({"sourcePath": "data", "studyContext": " "}, "studyContext"),
        (
            {
                "sourcePath": "data",
                "studyContext": "study",
                "analysisAssays": ["RNA", "RNA"],
            },
            "analysisAssays",
        ),
        (
            {
                "sourcePath": "data",
                "studyContext": "study",
                "pairedAssays": ["RNA", "RNA"],
            },
            "pairedAssays must be unique",
        ),
        (
            {
                "sourcePath": "data",
                "studyContext": "study",
                "pairedAssays": ["RNA"],
            },
            "at least two",
        ),
    )
    for values, message in invalid_requests:
        with pytest.raises(ValidationError, match=message):
            AutomatedWorkflowRequest(**values)

    with pytest.raises(ValidationError, match="zarrPath"):
        AutomatedWorkflowResumeRequest(zarrPath=" ", workflowRunId="workflow-1")
    with pytest.raises(ValidationError, match="workflowRunId"):
        AutomatedWorkflowResumeRequest(zarrPath="data.zarr", workflowRunId="BAD")

    assert OrchestrationRequestRecord.get_blank().workflowRunId == ""
    assert OrchestrationRequestRecord.get_example().workflowRunId == "workflow-1"
    assert OrchestrationResumeRecord.get_blank().resumeId == ""
    assert OrchestrationResumeRecord.get_example().resumeId == "resume-1"


def test_journal_storage_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    journal_module._write_key_once(root, "record.json", b"first")
    with pytest.raises(FileExistsError, match="exists"):
        journal_module._write_key_once(root, "record.json", b"second")

    with monkeypatch.context() as patch:
        observed = iter((None, b"different"))
        patch.setattr(
            journal_module.record_io,
            "read_key",
            lambda *_args: next(observed),
        )
        with pytest.raises(FileExistsError, match="raced"):
            journal_module._write_key_once(root, "raced.json", b"payload")

    unsupported = SimpleNamespace(store=SimpleNamespace(supports_listing=False))
    with pytest.raises(NotImplementedError, match="requires listing"):
        journal_module._list_keys(unsupported, "prefix")

    with monkeypatch.context() as patch:
        patch.setattr(journal_module.record_io, "read_key", lambda *_args: b"{")
        with pytest.raises(ValueError, match="Malformed orchestration record"):
            journal_module._read_model(root, "bad.json", WorkflowQuestion)


def test_orchestration_namespace_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(RuntimeError, match="Create the agent workflow"):
        journal_module._ensure_orchestration_store(SimpleNamespace(zw=root))

    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_array("agents", shape=(1,), dtype="i1")
    with pytest.raises(ValueError, match="agents namespace"):
        journal_module._ensure_orchestration_store(SimpleNamespace(zw=root))

    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("agents")
    with monkeypatch.context() as patch:
        patch.setattr(journal_module, "_list_keys", lambda *_args: ["occupied"])
        with pytest.raises(ValueError, match="non-Zarr object"):
            journal_module._ensure_orchestration_store(SimpleNamespace(zw=root))

    root = zarr.open_group(store=MemoryStore(), mode="w")
    agents = root.create_group("agents")
    agents.create_array("orchestrations", shape=(1,), dtype="i1")
    with pytest.raises(ValueError, match="orchestrations namespace"):
        journal_module._ensure_orchestration_store(SimpleNamespace(zw=root))

    root = zarr.open_group(store=MemoryStore(), mode="w")
    agents = root.create_group("agents")
    agents.create_group(
        "orchestrations",
        attributes={"format": "foreign", "format_version": 99},
    )
    with pytest.raises(ValueError, match="Unrecognized orchestration"):
        journal_module._ensure_orchestration_store(SimpleNamespace(zw=root))


def test_stage_record_identity_and_checksum_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome_key = "root/workflow-1/stages/ingest/attempt-1/outcome.json"
    start_key = "root/workflow-1/stages/ingest/attempt-1/started.json"
    done = journal_module._complete_attempt(_started(), status="done")

    with monkeypatch.context() as patch:
        patch.setattr(journal_module, "_list_keys", lambda *_args: [outcome_key])
        patch.setattr(
            journal_module,
            "_read_model",
            lambda *_args: done.model_copy(update={"attemptId": "other"}),
        )
        with pytest.raises(ValueError, match="outcome identity"):
            journal_module._stage_outcomes(object(), "root", "workflow-1", "ingest")
        patch.setattr(
            journal_module,
            "_read_model",
            lambda *_args: done.model_copy(update={"contentSha256": "0" * 64}),
        )
        with pytest.raises(ValueError, match="outcome checksum"):
            journal_module._stage_outcomes(object(), "root", "workflow-1", "ingest")

    started = _started()
    with monkeypatch.context() as patch:
        patch.setattr(journal_module, "_list_keys", lambda *_args: [start_key])
        patch.setattr(
            journal_module,
            "_read_model",
            lambda *_args: started.model_copy(update={"status": "done"}),
        )
        with pytest.raises(ValueError, match="start identity"):
            journal_module._stage_starts(object(), "root", "workflow-1", "ingest")
        patch.setattr(
            journal_module,
            "_read_model",
            lambda *_args: started.model_copy(update={"contentSha256": "0" * 64}),
        )
        with pytest.raises(ValueError, match="start checksum"):
            journal_module._stage_starts(object(), "root", "workflow-1", "ingest")


def test_resume_answer_helper_edges() -> None:
    assert not journal_module._has_resume_answer(None)
    assert not journal_module._has_resume_answer("  ")
    assert journal_module._has_resume_answer("answer")
    assert not journal_module._has_resume_answer({})
    assert journal_module._has_resume_answer({"answer": 1})
    assert not journal_module._has_resume_answer([])
    assert journal_module._has_resume_answer([1])
    assert journal_module._has_resume_answer(0)

    assert journal_module._unsafe_context_resolution("skipHarmony") == "skip"
    assert (
        journal_module._unsafe_context_resolution(
            {"batchCorrection": {"action": "skip"}}
        )
        == "skip"
    )
    assert journal_module._unsafe_context_resolution({"selection": "other"}) is None

    assert journal_module._resume_answer_errors(_started(), {}) == [
        "The latest paused stage does not contain persisted questions"
    ]
    choice = journal_module._complete_attempt(
        _started(),
        status="needsInput",
        needs_input=WorkflowNeedsInput(
            questions=[
                WorkflowQuestion(
                    questionId="finalGraphOptionId",
                    options=["graph-a", "graph-b"],
                )
            ]
        ),
    )
    assert (
        "must be one of"
        in journal_module._resume_answer_errors(
            choice, {"finalGraphOptionId": "graph-c"}
        )[0]
    )
    generic = journal_module._complete_attempt(
        _started(),
        status="needsInput",
        needs_input=WorkflowNeedsInput(
            questions=[WorkflowQuestion(questionId="freeText")]
        ),
    )
    assert (
        "must be non-empty"
        in journal_module._resume_answer_errors(generic, {"freeText": " "})[0]
    )


def test_persisted_resume_record_validation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SimpleNamespace(zw=object())

    def validate(record: OrchestrationResumeRecord) -> OrchestrationResumeRecord:
        monkeypatch.setattr(journal_module, "_read_model", lambda *_args: record)
        return journal_module._validated_resume_record(
            store, "root", "workflow-1", "resume-1"
        )

    mismatched = _with_checksum(
        OrchestrationResumeRecord(workflowRunId="other", resumeId="resume-1")
    )
    with pytest.raises(ValueError, match="identity"):
        validate(mismatched)

    with pytest.raises(ValueError, match="checksum"):
        validate(
            OrchestrationResumeRecord(
                workflowRunId="workflow-1",
                resumeId="resume-1",
                contentSha256="0" * 64,
            )
        )

    unanswered = _with_checksum(
        OrchestrationResumeRecord(
            workflowRunId="workflow-1",
            resumeId="resume-1",
            answers={"unexpected": "answer"},
        )
    )
    with pytest.raises(ValueError, match="without an answered attempt"):
        validate(unanswered)

    empty = _with_checksum(
        OrchestrationResumeRecord(workflowRunId="workflow-1", resumeId="resume-1")
    )
    assert validate(empty) == empty

    paused = journal_module._complete_attempt(
        _started(),
        status="needsInput",
        needs_input=WorkflowNeedsInput(
            questions=[WorkflowQuestion(questionId="freeText")]
        ),
    )
    link = journal_module._parent_link(paused)
    answered = _with_checksum(
        OrchestrationResumeRecord(
            workflowRunId="workflow-1",
            resumeId="resume-1",
            answeredAttempt=link,
            questionIds=["freeText"],
            answers={"freeText": "answer"},
        )
    )

    monkeypatch.setattr(journal_module, "_stage_outcomes", lambda *_args: [])
    with pytest.raises(ValueError, match="does not cite one paused"):
        validate(answered)

    monkeypatch.setattr(journal_module, "_stage_outcomes", lambda *_args: [paused])
    stale = _with_checksum(answered.model_copy(update={"questionIds": ["stale"]}))
    with pytest.raises(ValueError, match="question IDs are stale"):
        validate(stale)

    invalid = _with_checksum(answered.model_copy(update={"answers": {"freeText": ""}}))
    with pytest.raises(ValueError, match="invalid answers"):
        validate(invalid)


def test_stage_outcome_resolution_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request_record()
    store = SimpleNamespace(
        zw=object(),
        cells=SimpleNamespace(columns=[]),
        load_artifact=lambda *_args: object(),
    )
    done = journal_module._complete_attempt(_started(), status="done")

    with pytest.raises(ValueError, match="checksum is stale"):
        journal_module._stage_outcome_resolves(
            store,
            "root",
            "workflow-1",
            request,
            done.model_copy(update={"requestSha256": "c" * 64}),
        )
    with pytest.raises(ValueError, match="cannot retain started"):
        journal_module._stage_outcome_resolves(
            store, "root", "workflow-1", request, _started()
        )

    monkeypatch.setattr(
        journal_module,
        "_read_model",
        lambda *_args: _started().model_copy(update={"inputs": {"changed": True}}),
    )
    with pytest.raises(ValueError, match="do not match"):
        journal_module._stage_outcome_resolves(
            store, "root", "workflow-1", request, done
        )

    parent_outcome = journal_module._complete_attempt(_started(), status="done")
    parent = journal_module._parent_link(parent_outcome)
    child_started = _started(stage="data_enrichment", parents=[parent])
    child = journal_module._complete_attempt(child_started, status="done")
    monkeypatch.setattr(journal_module, "_read_model", lambda *_args: child_started)
    monkeypatch.setattr(journal_module, "_stage_outcomes", lambda *_args: [])
    assert not journal_module._stage_outcome_resolves(
        store, "root", "workflow-1", request, child
    )

    reference = AgentReportReference.get_example().model_copy(
        update={"workflowRunId": "other-workflow"}
    )
    report_outcome = journal_module._complete_attempt(
        _started(), status="done", report_references=[reference]
    )
    monkeypatch.setattr(journal_module, "_read_model", lambda *_args: _started())
    assert not journal_module._stage_outcome_resolves(
        store, "root", "workflow-1", request, report_outcome
    )

    metadata_outcome = journal_module._complete_attempt(
        _started(), status="done", outputs={"metadataColumns": ["missing"]}
    )
    assert not journal_module._stage_outcome_resolves(
        store, "root", "workflow-1", request, metadata_outcome
    )

    monkeypatch.setattr(
        journal_module,
        "_stage_outcomes",
        lambda *_args: [done.model_copy(update={"configSha256": "c" * 64})],
    )
    with pytest.raises(ValueError, match="checksum is stale"):
        journal_module._validated_done_outcome(
            store, "root", "workflow-1", "ingest", request, []
        )


def test_stage_invocation_and_report_loading_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _started(stage="parameter_tuning")
    with pytest.raises(ValueError, match="conflicting orchestration identity"):
        journal_module._stage_invocation(
            started,
            AgentInvocation(inputs={"orchestrationExecutionId": "other"}),
        )

    with pytest.raises(ValueError, match="exactly one report reference"):
        journal_module.load_stage_report(object(), started, WorkflowQuestion)

    execution_id = journal_module._stage_execution_id(started)
    matching = AgentReportReference.get_example().model_copy(
        update={"agentRunId": execution_id}
    )
    other = AgentReportReference.get_example().model_copy(
        update={"agentRunId": "other-report"}
    )
    tuning = started.model_copy(update={"reportReferences": [matching, other]})
    monkeypatch.setattr(
        journal_module,
        "load_agent_report",
        lambda *_args: WorkflowQuestion(questionId="answer"),
    )
    assert isinstance(
        journal_module.load_stage_report(object(), tuning, WorkflowQuestion),
        WorkflowQuestion,
    )

    wrong = started.model_copy(update={"reportReferences": [matching]})
    monkeypatch.setattr(
        journal_module,
        "load_agent_report",
        lambda *_args: WorkflowNeedsInput(),
    )
    with pytest.raises(TypeError, match="report is not WorkflowQuestion"):
        journal_module.load_stage_report(object(), wrong, WorkflowQuestion)


def test_persisted_stage_report_recovery_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _started(stage="data_enrichment")
    execution_id = journal_module._stage_execution_id(started)
    reference = AgentReportReference.get_example().model_copy(
        update={"agentRunId": execution_id}
    )
    store = SimpleNamespace(load_artifact=lambda *_args: object())

    monkeypatch.setattr(
        journal_module,
        "list_agent_reports",
        lambda *_args, **_kwargs: [reference, reference],
    )
    with pytest.raises(ValueError, match="multiple persisted reports"):
        journal_module._recover_persisted_stage_report(
            store,
            started,
            agent_name="data_enrichment",
            expected_type=WorkflowQuestion,
        )

    monkeypatch.setattr(
        journal_module,
        "list_agent_reports",
        lambda *_args, **_kwargs: [reference],
    )
    monkeypatch.setattr(
        journal_module,
        "load_agent_record",
        lambda *_args: SimpleNamespace(
            invocation=AgentInvocation(
                agentName="data_enrichment",
                inputs={"orchestrationExecutionId": "stale"},
            )
        ),
    )
    with pytest.raises(ValueError, match="stale execution identity"):
        journal_module._recover_persisted_stage_report(
            store,
            started,
            agent_name="data_enrichment",
            expected_type=WorkflowQuestion,
        )

    monkeypatch.setattr(
        journal_module,
        "load_agent_record",
        lambda *_args: SimpleNamespace(
            invocation=AgentInvocation(
                agentName="data_enrichment",
                inputs={"orchestrationExecutionId": execution_id},
            )
        ),
    )
    monkeypatch.setattr(
        journal_module,
        "load_agent_report",
        lambda *_args: WorkflowNeedsInput(),
    )
    with pytest.raises(TypeError, match="is not WorkflowQuestion"):
        journal_module._recover_persisted_stage_report(
            store,
            started,
            agent_name="data_enrichment",
            expected_type=WorkflowQuestion,
        )

    report = WorkflowQuestion(questionId="question")
    invocation = AgentInvocation(agentName="data_enrichment")
    monkeypatch.setattr(
        journal_module,
        "save_agent_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError),
    )
    monkeypatch.setattr(
        journal_module,
        "_recover_persisted_stage_report",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(FileExistsError):
        journal_module._save_stage_report(
            store,
            started,
            report,
            invocation=invocation,
            expected_type=WorkflowQuestion,
        )

    monkeypatch.setattr(
        journal_module,
        "_recover_persisted_stage_report",
        lambda *_args, **_kwargs: (report, reference),
    )
    assert journal_module._save_stage_report(
        store,
        started,
        report,
        invocation=invocation,
        expected_type=WorkflowQuestion,
    ) == (report, reference)


def test_terminal_result_validation_and_persistence_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _terminal_workflow()
    result = _terminal_result(workflow)
    store = SimpleNamespace(zw=object())

    def load(payload: bytes) -> AutomatedWorkflowResult | None:
        monkeypatch.setattr(
            journal_module.record_io, "read_key", lambda *_args: payload
        )
        return journal_module._load_terminal_result(store, "root", workflow)

    with pytest.raises(ValueError, match="Malformed automated workflow result"):
        load(b"{")
    with pytest.raises(ValueError, match="checksum is invalid"):
        load(
            journal_module.record_io.display_json_bytes(
                result.model_dump(mode="json") | {"contentSha256": "0" * 64}
            )
        )

    missing_workflow = _with_checksum(
        AutomatedWorkflowResult(
            status="completed", currentStage="biological_interpretation"
        )
    )
    with pytest.raises(ValueError, match="missing its workflow identity"):
        load(
            journal_module.record_io.display_json_bytes(
                missing_workflow.model_dump(mode="json")
            )
        )

    stale = _with_checksum(
        result.model_copy(
            update={
                "workflowRun": workflow.model_copy(
                    update={"analysisStore": "other.zarr"}
                )
            }
        )
    )
    with pytest.raises(ValueError, match="stale workflow metadata"):
        load(journal_module.record_io.display_json_bytes(stale.model_dump(mode="json")))

    wrong_status = _with_checksum(result.model_copy(update={"status": "failed"}))
    with pytest.raises(ValueError, match="stale terminal status"):
        load(
            journal_module.record_io.display_json_bytes(
                wrong_status.model_dump(mode="json")
            )
        )

    wrong_reports = _with_checksum(
        result.model_copy(
            update={"reportReferences": [AgentReportReference.get_example()]}
        )
    )
    with pytest.raises(ValueError, match="stale report references"):
        load(
            journal_module.record_io.display_json_bytes(
                wrong_reports.model_dump(mode="json")
            )
        )

    with monkeypatch.context() as patch:
        patch.setattr(journal_module, "_load_terminal_result", lambda *_args: result)
        assert (
            journal_module._persist_terminal_result(store, "root", workflow, result)
            == result
        )

    with monkeypatch.context() as patch:
        patch.setattr(journal_module, "_load_terminal_result", lambda *_args: None)
        with pytest.raises(ValueError, match="exact content checksum"):
            journal_module._persist_terminal_result(
                store,
                "root",
                workflow,
                result.model_copy(update={"contentSha256": ""}),
            )

    with monkeypatch.context() as patch:
        persisted = iter((None, result))
        patch.setattr(
            journal_module,
            "_load_terminal_result",
            lambda *_args: next(persisted),
        )
        patch.setattr(
            journal_module,
            "_write_model_once",
            lambda *_args: (_ for _ in ()).throw(FileExistsError),
        )
        assert (
            journal_module._persist_terminal_result(store, "root", workflow, result)
            == result
        )

    with monkeypatch.context() as patch:
        patch.setattr(journal_module, "_load_terminal_result", lambda *_args: None)
        patch.setattr(journal_module, "_write_model_once", lambda *_args: None)
        with pytest.raises(RuntimeError, match="was not persisted"):
            journal_module._persist_terminal_result(store, "root", workflow, result)
