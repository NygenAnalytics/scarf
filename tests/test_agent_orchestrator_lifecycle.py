"""Lifecycle, persistence, resume, and recovery orchestrator contracts."""

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import zarr
from pydantic_ai import ModelHTTPError

import scarf.agent.orchestrator.context as context_module
import scarf.agent.orchestrator.journal as journal_module
import scarf.agent.orchestrator.main as main_module
from scarf.agent.config import AgentRunConfig
from scarf.agent.data_enrichment import DataEnrichmentReport
from scarf.agent.ingest import IngestResult
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
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
    create_agent_workflow,
    finalize_agent_workflow,
    list_agent_reports,
    load_agent_report,
    load_agent_workflow,
    save_agent_report,
)
from scarf.agent.types import AgentRunInfo, ArtifactReferenceModel
from scarf.datastore.datastore import DataStore
from tests.agent_orchestrator_store import create_store


_PLAN_CHECKSUM = "a" * 64


def _record_checksum(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("contentSha256", None)
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_record_checksum(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert payload["contentSha256"] == _record_checksum(payload)
    return payload


def _mock_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "detect_format", lambda _path: "zarr")

    def fake_ingest(
        path: str,
        *,
        zarrPath: str | None,
        model: Any,
        directions: Mapping[str, Any],
    ) -> IngestResult:
        del model, directions
        resolved = str(Path(zarrPath or path).resolve())
        return IngestResult(
            status="done",
            format="zarr",
            zarrPath=resolved,
            assayNames=["RNA"],
            summary={"assays": ["RNA"]},
            actions=["summarize_zarr"],
        )

    monkeypatch.setattr(main_module, "ingest", fake_ingest)


class _CheckpointOrchestrator(AgentOrchestrator):
    """Minimal deterministic stage machine exercising persistence and resume."""

    def __init__(self) -> None:
        super().__init__(object())
        self.enrichmentExecutions = 0

    def _continue(
        self,
        store: Any,
        workflow: Any,
        request_record: Any,
        *,
        answers: Mapping[str, Any],
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> AutomatedWorkflowResult:
        prefix = journal_module._ensure_orchestration_store(store)
        ingest = journal_module._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
        )
        assert ingest is not None
        parents = [journal_module._parent_link(ingest)]

        enrichment = journal_module._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "data_enrichment",
            request_record,
            parents,
        )
        if enrichment is None:
            self.enrichmentExecutions += 1
            report = DataEnrichmentReport.get_example()
            report = report.model_copy(
                update={
                    "runInfo": report.runInfo.model_copy(
                        update={"runId": uuid.uuid4().hex}
                    )
                }
            )
            reference = save_agent_report(
                store,
                workflow.workflowRunId,
                report,
                invocation=AgentInvocation(
                    agentName="data_enrichment",
                    inputs={"studyContext": request_record.request.studyContext},
                ),
            )
            started = journal_module._start_attempt(
                store.zw,
                prefix,
                workflow.workflowRunId,
                "data_enrichment",
                request_record,
                parents,
                inputs={"studyContext": request_record.request.studyContext},
                resume_record=resume_record,
            )
            enrichment = journal_module._complete_attempt(
                started,
                status="done",
                report_references=[reference],
            )
            journal_module._save_outcome(store.zw, prefix, enrichment)
        parents = [journal_module._parent_link(enrichment)]

        approved = answers.get("approvePlanChecksum") == _PLAN_CHECKSUM
        completed_plan = journal_module._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
        )
        if completed_plan is None:
            started = journal_module._start_attempt(
                store.zw,
                prefix,
                workflow.workflowRunId,
                "preprocessing_plan",
                request_record,
                parents,
                inputs={"planChecksum": _PLAN_CHECKSUM},
                resume_record=resume_record,
            )
            if not approved:
                paused = journal_module._complete_attempt(
                    started,
                    status="needsInput",
                    needs_input=WorkflowNeedsInput(
                        questions=[
                            WorkflowQuestion(
                                questionId="approvePlanChecksum",
                                question="Approve the preprocessing plan?",
                                planChecksum=_PLAN_CHECKSUM,
                            )
                        ]
                    ),
                )
                journal_module._save_outcome(store.zw, prefix, paused)
                return journal_module.paused_or_failed_result(
                    store,
                    workflow,
                    request_record,
                    paused,
                )
            completed_plan = journal_module._complete_attempt(
                started,
                status="done",
                outputs={"planChecksum": _PLAN_CHECKSUM},
                actions=["approve_preprocessing_plan"],
            )
            journal_module._save_outcome(store.zw, prefix, completed_plan)

        terminal = finalize_agent_workflow(
            store,
            workflow.workflowRunId,
            status="completed",
            message="Checkpoint workflow completed",
        )
        result = AutomatedWorkflowResult(
            status="completed",
            currentStage="preprocessing_plan",
            zarrPath=str(Path(store.zarr_loc).resolve()),
            workflowRun=terminal,
            reportReferences=journal_module.all_report_references(
                store,
                prefix,
                workflow.workflowRunId,
            ),
        )
        result = result.model_copy(
            update={"contentSha256": journal_module._record_checksum(result)}
        )
        journal_module._write_model_once(
            store.zw,
            journal_module._result_key(prefix, workflow.workflowRunId),
            result,
        )
        return result


def _start_paused_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workspace: str | None = None,
) -> tuple[_CheckpointOrchestrator, AutomatedWorkflowResult, Path]:
    path = create_store(tmp_path / "data.zarr", workspace=workspace)
    _mock_ingest(monkeypatch)
    orchestrator = _CheckpointOrchestrator()
    result = orchestrator.run(
        AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic test study.",
            workspace=workspace,
        )
    )
    assert result.status == "needsInput"
    assert result.workflowRun is not None
    return orchestrator, result, path


def test_orchestration_records_are_plain_json_with_valid_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _orchestrator, result, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = result.workflowRun.workflowRunId
    agents = path / "agents"
    orchestration = agents / "orchestrations"
    workflow_path = orchestration / workflow_id

    assert (agents / "zarr.json").is_file()
    assert (agents / "store.json").is_file()
    assert (orchestration / "zarr.json").is_file()
    assert not (workflow_path / "zarr.json").exists()
    assert not list(orchestration.rglob("c"))

    request = _assert_record_checksum(workflow_path / "request.json")
    assert request["workflowRunId"] == workflow_id
    records = sorted((workflow_path / "stages").rglob("*.json"))
    assert records
    for record_path in records:
        _assert_record_checksum(record_path)
        assert record_path.name in {"started.json", "outcome.json"}


def test_approval_resume_reuses_completed_stages_and_persists_answer_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    workflow_path = path / "agents" / "orchestrations" / workflow_id
    paused_outcome_path = next(
        (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
    )
    paused_outcome = json.loads(paused_outcome_path.read_text())

    completed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
            answers={"approvePlanChecksum": _PLAN_CHECKSUM},
        )
    )

    assert completed.status == "completed"
    assert completed.workflowRun is not None
    assert completed.workflowRun.status == "completed"
    assert orchestrator.enrichmentExecutions == 1
    assert len(list((workflow_path / "stages" / "ingest").glob("*/outcome.json"))) == 1
    assert (
        len(list((workflow_path / "stages" / "data_enrichment").glob("*/outcome.json")))
        == 1
    )
    assert (
        len(
            list(
                (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
            )
        )
        == 2
    )

    resume_path = next((workflow_path / "resumes").glob("*.json"))
    resume = _assert_record_checksum(resume_path)
    assert resume["answeredAttempt"]["attemptId"] == paused_outcome["attemptId"]
    assert resume["questionIds"] == ["approvePlanChecksum"]
    assert resume["answers"] == {"approvePlanChecksum": _PLAN_CHECKSUM}
    resumed_start = next(
        payload
        for payload in (
            json.loads(path.read_text())
            for path in (workflow_path / "stages" / "preprocessing_plan").glob(
                "*/started.json"
            )
        )
        if "resumeLineage" in payload["inputs"]
    )
    assert resumed_start["inputs"]["resumeLineage"] == {
        "resumeId": resume["resumeId"],
        "answeredAttempt": resume["answeredAttempt"],
        "questionIds": ["approvePlanChecksum"],
    }

    persisted = _assert_record_checksum(workflow_path / "result.json")
    assert persisted["status"] == "completed"
    assert persisted["workflowRun"]["status"] == "completed"
    assert load_agent_workflow(path, workflow_id).status == "completed"


def test_resume_does_not_mutate_constructor_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    constructor_config = AutomatedWorkflowConfig(primaryInitialCandidates=2)
    orchestrator.config = constructor_config

    completed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=paused.workflowRun.workflowRunId,
            answers={"approvePlanChecksum": _PLAN_CHECKSUM},
        )
    )

    assert completed.status == "completed"
    assert orchestrator.config == constructor_config


@pytest.mark.parametrize(
    ("answers", "expected_note"),
    [
        ({}, "Missing resume answer"),
        ({"approvePlanChecksum": "approve"}, "persisted plan checksum"),
        (
            {
                "approvePlanChecksum": _PLAN_CHECKSUM,
                "unexpectedAnswer": "ignored",
            },
            "Unknown resume answer key",
        ),
    ],
)
def test_invalid_resume_answers_retain_the_persisted_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, Any],
    expected_note: str,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId

    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
            answers=answers,
        )
    )

    assert result.status == "needsInput"
    assert result.currentStage == "preprocessing_plan"
    assert result.needsInput == paused.needsInput
    assert result.workflowRun is not None
    assert result.workflowRun.status == "running"
    assert any(expected_note in note for note in result.notes)
    workflow_path = path / "agents" / "orchestrations" / workflow_id
    assert (
        len(
            list(
                (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
            )
        )
        == 1
    )
    assert len(list((workflow_path / "resumes").glob("*.json"))) == 1


def test_interrupted_attempt_supersedes_a_historical_pause_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    record, store = orchestrator.load_request_for_resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )
    prefix = journal_module._ensure_orchestration_store(store)
    journal_module._start_attempt(
        store.zw,
        prefix,
        workflow_id,
        "preprocessing",
        record,
        [],
        inputs={"simulatedCrash": True},
    )

    with pytest.raises(ValueError, match="no active persisted questions"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                answers={"approvePlanChecksum": _PLAN_CHECKSUM},
            )
        )

    resumed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )

    assert resumed.status == "needsInput"
    assert not any("Missing resume answer" in note for note in resumed.notes)
    workflow_path = path / "agents" / "orchestrations" / workflow_id
    assert (
        len(
            list(
                (workflow_path / "stages" / "preprocessing_plan").glob("*/outcome.json")
            )
        )
        == 2
    )


def test_interrupted_answered_attempt_inherits_exact_persisted_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    record, store = orchestrator.load_request_for_resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )
    prefix = journal_module._ensure_orchestration_store(store)
    paused_outcome = journal_module._stage_outcomes(
        store.zw,
        prefix,
        workflow_id,
        "preprocessing_plan",
    )[0]
    first_resume = OrchestrationResumeRecord(
        workflowRunId=workflow_id,
        resumeId="resume-before-crash",
        createdAtNs=paused_outcome.completedAtNs + 1,
        answeredAttempt=journal_module._parent_link(paused_outcome),
        questionIds=["approvePlanChecksum"],
        answers={"approvePlanChecksum": _PLAN_CHECKSUM},
    )
    first_resume = first_resume.model_copy(
        update={"contentSha256": journal_module._record_checksum(first_resume)}
    )
    journal_module._write_model_once(
        store.zw,
        journal_module._resume_key(
            prefix,
            workflow_id,
            first_resume.resumeId,
        ),
        first_resume,
    )
    journal_module._start_attempt(
        store.zw,
        prefix,
        workflow_id,
        "preprocessing_plan",
        record,
        paused_outcome.parentAttempts,
        inputs={"planChecksum": _PLAN_CHECKSUM},
        resume_record=first_resume,
    )

    with pytest.raises(ValueError, match="Cannot change answers"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                answers={"approvePlanChecksum": "b" * 64},
            )
        )

    completed = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(path),
            workflowRunId=workflow_id,
        )
    )

    assert completed.status == "completed"
    resume_records = sorted(
        (path / "agents" / "orchestrations" / workflow_id / "resumes").glob("*.json")
    )
    assert len(resume_records) == 2
    inherited = next(
        json.loads(value.read_text())
        for value in resume_records
        if value.stem != first_resume.resumeId
    )
    assert inherited["answers"] == first_resume.answers
    assert inherited["answeredAttempt"] == first_resume.answeredAttempt.model_dump(
        mode="json"
    )
    assert inherited["questionIds"] == first_resume.questionIds


def test_resume_rejects_changed_dataset_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    root = zarr.open_group(str(path), mode="r+")
    root["RNA"].attrs["dataset_fingerprint"] = "changed-dataset"

    with pytest.raises(ValueError, match="dataset fingerprints"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                answers={"approvePlanChecksum": _PLAN_CHECKSUM},
            )
        )


def test_resume_rejects_request_envelope_and_destination_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    request_path = path / "agents" / "orchestrations" / workflow_id / "request.json"
    payload = json.loads(request_path.read_text())
    payload["request"]["studyContext"] = "Tampered study context"
    request_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="request checksum"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
            )
        )

    copied = tmp_path / "copied.zarr"
    shutil.copytree(path, copied)
    with pytest.raises(ValueError, match="zarrPath"):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(copied),
                workflowRunId=workflow_id,
            )
        )


def test_workspace_records_are_isolated_and_wrong_workspace_cannot_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(
        tmp_path,
        monkeypatch,
        workspace="workspace_a",
    )
    workflow_id = paused.workflowRun.workflowRunId
    root = zarr.open_group(str(path), mode="r+")
    root.create_group("workspace_b")

    assert (
        path
        / "workspace_a"
        / "agents"
        / "orchestrations"
        / workflow_id
        / "request.json"
    ).is_file()
    assert not (path / "agents").exists()
    assert not (path / "workspace_b" / "agents").exists()

    with pytest.raises(FileNotFoundError):
        orchestrator.resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow_id,
                workspace="workspace_b",
            )
        )


def test_cancel_finalizes_abandoned_and_prevents_future_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    workflow_id = paused.workflowRun.workflowRunId
    request = AutomatedWorkflowResumeRequest(
        zarrPath=str(path),
        workflowRunId=workflow_id,
    )

    cancelled = orchestrator.cancel(request, message="Stop this test workflow")

    assert cancelled.status == "abandoned"
    assert cancelled.workflowRun is not None
    assert cancelled.workflowRun.status == "abandoned"
    assert load_agent_workflow(path, workflow_id).status == "abandoned"
    result_path = path / "agents" / "orchestrations" / workflow_id / "result.json"
    persisted = _assert_record_checksum(result_path)
    assert persisted["status"] == "abandoned"
    assert persisted["notes"] == ["Stop this test workflow"]
    with pytest.raises(RuntimeError, match="Cannot resume"):
        orchestrator.resume(request)

    result_path.unlink()
    repaired = orchestrator.resume(request)
    assert repaired.status == "abandoned"
    assert repaired.workflowRun is not None
    assert repaired.workflowRun.status == "abandoned"
    _assert_record_checksum(result_path)
    with pytest.raises(RuntimeError, match="Cannot resume"):
        orchestrator.resume(request)


def test_failed_stage_preserves_completed_operation_journal(tmp_path: Path) -> None:
    path = create_store(tmp_path / "failure-journal.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="failure-journal")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A failed-stage journal test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "preprocessing",
        request_record,
        [],
    )
    operation = {
        "operation": "filter_cells",
        "attrs": ["RNA_nCounts"],
        "resetPrevious": True,
    }

    outcome = journal_module.finish_exception(
        store,
        prefix,
        workflow,
        started,
        RuntimeError("failure after filtering"),
        actions=["cell_qc_global:validated"],
        outputs={"operations": [operation]},
        notes=["The completed selection mutation is retained for audit."],
    )

    assert outcome.status == "failed"
    assert outcome.actions == ["cell_qc_global:validated"]
    assert outcome.outputs["operations"] == [operation]
    assert outcome.notes == ["The completed selection mutation is retained for audit."]
    persisted = journal_module._stage_outcomes(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "preprocessing",
    )
    assert persisted == [outcome]
    assert load_agent_workflow(store, workflow.workflowRunId).status == "failed"


@pytest.mark.parametrize("status_code", [429, 500, 503, 599])
def test_retryable_model_http_error_leaves_stage_interrupted(
    tmp_path: Path,
    status_code: int,
) -> None:
    path = create_store(tmp_path / f"retryable-model-{status_code}.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(
        store,
        workflow_run_id=f"retryable-model-{status_code}",
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A retryable model failure test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "experimental_context",
        request_record,
        [],
    )
    error = ModelHTTPError(status_code, "test-model", {"error": "transient"})

    with pytest.raises(ModelHTTPError) as raised:
        journal_module.finish_exception(
            store,
            prefix,
            workflow,
            started,
            error,
        )

    assert raised.value is error
    assert (
        journal_module._stage_outcomes(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "experimental_context",
        )
        == []
    )
    assert load_agent_workflow(store, workflow.workflowRunId).status == "running"


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_nonretryable_model_http_error_remains_terminal(
    tmp_path: Path,
    status_code: int,
) -> None:
    path = create_store(tmp_path / f"terminal-model-{status_code}.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(
        store,
        workflow_run_id=f"terminal-model-{status_code}",
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A terminal model failure test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "experimental_context",
        request_record,
        [],
    )

    outcome = journal_module.finish_exception(
        store,
        prefix,
        workflow,
        started,
        ModelHTTPError(status_code, "test-model", {"error": "terminal"}),
    )

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.startswith("ModelHTTPError:")
    assert load_agent_workflow(store, workflow.workflowRunId).status == "failed"


def test_failed_stage_links_report_committed_before_exception(tmp_path: Path) -> None:
    path = create_store(tmp_path / "failure-report-link.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="failure-report-link")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A report-link crash test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "data_enrichment",
        request_record,
        [],
        inputs={"studyContext": request_record.request.studyContext},
    )
    report = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    _saved, reference = journal_module._save_stage_report(
        store,
        started,
        report,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"studyContext": request_record.request.studyContext},
        ),
        expected_type=DataEnrichmentReport,
    )

    outcome = journal_module.finish_exception(
        store,
        prefix,
        workflow,
        started,
        RuntimeError("failure after report commit"),
    )

    assert outcome.reportReferences == [reference]
    assert load_agent_report(store, reference) == report


def test_orphan_report_recovery_uses_semantic_stage_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    store = DataStore(
        str(path),
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    started = WorkflowStageAttempt(
        workflowRunId=workflow_id,
        stage="data_enrichment",
        attemptId="crash-attempt-one",
        status="started",
        startedAtNs=1,
        requestSha256="1" * 64,
        configSha256="2" * 64,
        inputs={
            "effectiveContext": "same",
            "resumeLineage": {
                "resumeId": "first-resume",
                "answeredAttempt": WorkflowStageLink.get_example().model_dump(
                    mode="json"
                ),
                "questionIds": ["approvePlanChecksum"],
            },
        },
    )
    report = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": DataEnrichmentReport.get_example().runInfo.model_copy(
                update={"runId": uuid.uuid4().hex}
            )
        }
    )
    persisted_report, reference = journal_module._save_stage_report(
        store,
        started,
        report,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"effectiveContext": "same"},
        ),
        expected_type=DataEnrichmentReport,
    )
    assert persisted_report == report

    retried = started.model_copy(
        update={
            "attemptId": "crash-attempt-two",
            "startedAtNs": 2,
            "inputs": {
                "effectiveContext": "same",
                "resumeLineage": {
                    "resumeId": "second-resume",
                    "answeredAttempt": None,
                    "questionIds": [],
                },
            },
        }
    )
    assert journal_module._stage_execution_id(started) == (
        journal_module._stage_execution_id(retried)
    )
    changed_context = retried.model_copy(
        update={"inputs": {"effectiveContext": "different"}}
    )
    assert journal_module._stage_execution_id(started) != (
        journal_module._stage_execution_id(changed_context)
    )
    recovered = journal_module._recover_persisted_stage_report(
        store,
        retried,
        agent_name="data_enrichment",
        expected_type=DataEnrichmentReport,
    )

    assert recovered == (report, reference)
    matching = [
        value
        for value in list_agent_reports(
            store,
            workflow_id,
            agent_name="data_enrichment",
        )
        if value.agentRunId == reference.agentRunId
    ]
    assert matching == [reference]


def test_data_enrichment_recovers_report_after_outcome_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, paused, path = _start_paused_workflow(tmp_path, monkeypatch)
    assert paused.workflowRun is not None
    workflow_id = paused.workflowRun.workflowRunId
    store = DataStore(
        str(path),
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    prefix = journal_module._ensure_orchestration_store(store)
    request_record = journal_module._read_model(
        store.zw,
        journal_module._request_key(prefix, workflow_id),
        OrchestrationRequestRecord,
    )
    assert isinstance(request_record, OrchestrationRequestRecord)
    workflow = load_agent_workflow(store, workflow_id)
    calls = 0

    class CountingAgent:
        config = AgentRunConfig()

        def run(self, *_args: Any, **_kwargs: Any) -> DataEnrichmentReport:
            nonlocal calls
            calls += 1
            return DataEnrichmentReport.get_example().model_copy(
                update={
                    "runInfo": DataEnrichmentReport.get_example().runInfo.model_copy(
                        update={"runId": uuid.uuid4().hex}
                    )
                }
            )

    monkeypatch.setattr(
        context_module,
        "DataEnrichmentAgent",
        lambda *_args, **_kwargs: CountingAgent(),
    )
    save_outcome = journal_module._save_outcome
    crashed = False

    def crash_after_report(*args: Any, **kwargs: Any) -> None:
        nonlocal crashed
        outcome = args[2]
        if outcome.stage == "data_enrichment" and not crashed:
            crashed = True
            raise KeyboardInterrupt("simulated process crash")
        save_outcome(*args, **kwargs)

    monkeypatch.setattr(journal_module, "_save_outcome", crash_after_report)
    cell_selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    with pytest.raises(KeyboardInterrupt, match="simulated process crash"):
        orchestrator.data_enrichment_stage(
            store,
            workflow,
            request_record,
            [],
            cell_selection,
            {},
        )
    monkeypatch.setattr(journal_module, "_save_outcome", save_outcome)

    outcome, recovered = orchestrator.data_enrichment_stage(
        store,
        workflow,
        request_record,
        [],
        cell_selection,
        {},
    )

    assert outcome.status == "done"
    assert recovered.status == "done"
    assert calls == 1
    assert "recover_persisted_data_enrichment_report" in outcome.actions
