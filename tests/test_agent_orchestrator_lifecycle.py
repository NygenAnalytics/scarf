"""Journal-owned resume, input identity, and visible failure behavior."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResumeRequest,
    journal,
)
from scarf.agent.orchestrator.models import (
    AutomatedWorkflowResult,
    OrchestrationResumeRecord,
    WorkflowIdentity,
    WorkflowNeedsInput,
    WorkflowQuestion,
)
from tests.agent_journal_store import memory_journal
from tests.agent_orchestrator_store import create_store, save_request


def _saved_workflow(path: Path, *, workspace: str | None = None):
    create_store(path, workspace=workspace)
    orchestrator = AgentOrchestrator("test-model")
    request = AutomatedWorkflowRequest(
        sourcePath=str(path),
        zarrPath=str(path),
        workspace=workspace,
        primaryAssay="RNA",
        markerAssay="RNA",
        analysisAssays=["RNA"],
        studyContext="Independent donors in two conditions",
        studyObjective="Resolve stable populations",
    )
    store = orchestrator.open_store(str(path), request)
    store.cells.insert("condition", np.array(["a", "a", "b", "b"]))
    record = save_request(
        orchestrator, store, WorkflowIdentity("workflow-1", workspace), request
    )
    resume = AutomatedWorkflowResumeRequest(
        zarrPath=str(path), workspace=workspace, workflowRunId="workflow-1"
    )
    return orchestrator, store, record, resume


@pytest.mark.parametrize("workspace", [None, "analysis"])
def test_resume_preserves_selected_rna_workspace_and_original_metadata(
    tmp_path, workspace
) -> None:
    orchestrator, store, record, resume = _saved_workflow(
        tmp_path / "rna.zarr", workspace=workspace
    )
    reopened_record, reopened = orchestrator.load_request_for_resume(resume)
    assert reopened_record == record
    assert reopened.workspace == workspace
    assert reopened.summary().default_assay == "RNA"
    store.cells.insert("derived_agent_column", np.array([1, 1, 2, 2]))
    assert orchestrator.load_request_for_resume(resume)[0] == record
    store.cells.insert("condition", np.array(["a", "b", "b", "b"]), overwrite=True)
    with pytest.raises(ValueError, match="metadata changed"):
        orchestrator.load_request_for_resume(resume)


def test_request_identity_binds_persisted_dataset_and_user_metadata(
    tmp_path,
) -> None:
    _, store, record, _ = _saved_workflow(tmp_path / "rna.zarr")
    data = record.inputIdentity["data"]
    assert data["datasetFingerprint"] == store._ensure_dataset_fingerprint("RNA")
    assert "countsSha256" not in data
    assert "condition" in data["metadata"]
    assert {"ids", "RNA_nCounts", "RNA_nFeatures"}.isdisjoint(data["metadata"])
    assert "names" in data["featureMetadata"]
    assert {"ids", "nCells", "dropOuts"}.isdisjoint(data["featureMetadata"])


def test_resume_rejects_changed_dataset_model_and_execution_settings(
    tmp_path,
) -> None:
    orchestrator, store, record, resume = _saved_workflow(tmp_path / "rna.zarr")
    with pytest.raises(ValueError, match="model differs"):
        AgentOrchestrator("another-model").load_request_for_resume(resume)
    with pytest.raises(ValueError, match="execution settings differ"):
        AgentOrchestrator(
            "test-model", config=AutomatedWorkflowConfig(randomSeed=3)
        ).load_request_for_resume(resume)
    root = zarr.open_group(str(store.zarr_loc), mode="r+")
    # Rebuilding the dataset from different counts publishes a new identity.
    root["RNA"].attrs["dataset_fingerprint"] = "rebuilt-dataset"
    with pytest.raises(ValueError, match="Selected RNA data"):
        orchestrator.load_request_for_resume(resume)
    assert (
        journal.read_request(
            store.zw, journal._orchestration_prefix(store), record.workflowRunId
        )
        == record
    )


def test_resume_never_reads_an_unrelated_latest_workflow(tmp_path, monkeypatch) -> None:
    orchestrator, store, record, resume = _saved_workflow(tmp_path / "rna.zarr")
    other_request = record.request.model_copy(
        update={"studyObjective": "A different question"}
    )
    save_request(orchestrator, store, WorkflowIdentity("other-run"), other_request)
    seen = []

    def stop(request):
        seen.append(request.workflowRunId)
        return AutomatedWorkflowResult(
            status="abstained", workflowRunId=request.workflowRunId
        )

    monkeypatch.setattr(orchestrator, "resume", stop)
    assert (
        orchestrator._reuse_or_resume(record.request).workflowRunId
        == record.workflowRunId
    )
    assert seen == [record.workflowRunId]
    conflicting = AgentOrchestrator("different-model")
    with pytest.raises(ValueError, match="different model or execution settings"):
        conflicting._reuse_or_resume(record.request)


def test_older_request_reports_actionable_hard_break_without_rewriting(
    tmp_path,
) -> None:
    orchestrator, store, record, resume = _saved_workflow(tmp_path / "rna.zarr")
    path = Path(store.zarr_loc) / "agents/orchestrations/old/request.json"
    path.parent.mkdir()
    content = '{"formatVersion":2,"workflowRunId":"old"}'
    path.write_text(content)
    with pytest.raises(
        ValueError, match="older requests cannot be resumed or regenerated"
    ):
        journal.open_analysis_store(store.zarr_loc, "old")
    assert path.read_text() == content
    assert store.get_assay("RNA").rawData.shape == (4, 4)


def _pause_with_interrupted_answer(
    *, failed: bool = False, tuning: bool = False, interrupted: bool = True
):
    store, prefix, record = memory_journal()
    start = journal._start_attempt(
        store.zw, prefix, record.workflowRunId, "ingest", record, []
    )
    ingest = journal._complete_attempt(start, status="done")
    journal._save_outcome(store.zw, prefix, ingest)
    parents = [journal._parent_link(ingest)]
    if tuning:
        for stage in (
            "data_enrichment",
            "rna_quality_metrics",
            "experimental_context",
            "preprocessing_plan",
            "preprocessing",
        ):
            started = journal._start_attempt(
                store.zw, prefix, record.workflowRunId, stage, record, parents
            )
            completed = journal._complete_attempt(started, status="done")
            journal._save_outcome(store.zw, prefix, completed)
            parents = [journal._parent_link(completed)]
    stage = "parameter_tuning" if tuning else "data_enrichment"
    question_id = "parameter_tuning" if tuning else "enrichmentDirections"
    start = journal._start_attempt(
        store.zw, prefix, record.workflowRunId, stage, record, parents
    )
    paused = journal._complete_attempt(
        start,
        status="needsInput",
        needs_input=WorkflowNeedsInput(
            questions=[
                WorkflowQuestion(
                    questionId=question_id,
                    question="Inspect unresolved scientific evidence"
                    if tuning
                    else "Confirm RNA organism",
                )
            ]
        ),
    )
    journal._save_outcome(store.zw, prefix, paused)
    answers = {
        question_id: {
            "action": "defer",
            "selectedCandidateId": "baseline",
            "correctionNeed": "uncertain",
            "evidenceIds": ["candidate:baseline"],
            "quantitativeFindings": ["The saved comparison is incomplete."],
            "qualitativeFindings": ["Marker programs remain unresolved."],
            "comparisonConclusions": [],
            "plainLanguageSummary": "More evidence is required.",
            "objectivePreservation": "Preserve the observed populations.",
            "rationale": "More evidence is required",
        }
        if tuning
        else {"organism": "human"}
    }
    resume_record = OrchestrationResumeRecord(
        answeredAttempt=journal._parent_link(paused), answers=answers
    )
    if not interrupted:
        return store, record, answers, resume_record
    interrupted_start = journal._start_attempt(
        store.zw,
        prefix,
        record.workflowRunId,
        stage,
        record,
        parents,
        resume_record=resume_record,
    )
    if failed:
        outcome = journal._complete_attempt(
            interrupted_start, status="failed", error="Provider unavailable"
        )
        journal._save_outcome(store.zw, prefix, outcome)
    return store, record, answers, resume_record


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("tuning", [False, True])
def test_interrupted_answered_attempt_resumes_without_asking_or_spending_again(
    monkeypatch, failed, tuning
) -> None:
    store, record, answers, original = _pause_with_interrupted_answer(
        failed=failed, tuning=tuning
    )
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    captured = {}

    def continue_work(*args, **kwargs):
        captured.update(kwargs)
        return AutomatedWorkflowResult(
            status="abstained", notes=["Stopped at the resumed boundary"]
        )

    monkeypatch.setattr(orchestrator, "_continue", continue_work)
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr", workflowRunId=record.workflowRunId
        )
    )
    assert result.status == "abstained"
    assert captured["answers"] == answers
    assert captured["resume_record"].answeredAttempt == original.answeredAttempt


@pytest.mark.parametrize("tuning", [False, True])
def test_empty_answer_reenters_only_the_checkpointed_tuning_stage(
    monkeypatch, tuning
) -> None:
    store, record, _, _ = _pause_with_interrupted_answer(
        tuning=tuning, interrupted=False
    )
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    before = journal.analysis_snapshot(store, record.workflowRunId)
    assert before["status"] == "needsInput"
    assert before["stages"][-1]["stage"] == (
        "parameter_tuning" if tuning else "data_enrichment"
    )
    captured = {}

    def continue_work(*args, **kwargs):
        captured.update(kwargs)
        return AutomatedWorkflowResult(
            status="abstained", currentStage="parameter_tuning"
        )

    monkeypatch.setattr(orchestrator, "_continue", continue_work)
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr", workflowRunId=record.workflowRunId
        )
    )
    if tuning:
        assert result.status == "abstained"
        assert captured == {"answers": {}, "resume_record": None}
    else:
        assert result.status == "needsInput"
        assert captured == {}
    assert journal.analysis_snapshot(store, record.workflowRunId) == before


def test_explicit_tuning_answer_keeps_its_exact_paused_attempt(monkeypatch) -> None:
    store, record, answers, original = _pause_with_interrupted_answer(
        tuning=True, interrupted=False
    )
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    captured = {}

    def continue_work(*args, **kwargs):
        captured.update(kwargs)
        return AutomatedWorkflowResult(status="abstained")

    monkeypatch.setattr(orchestrator, "_continue", continue_work)
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr",
            workflowRunId=record.workflowRunId,
            answers=answers,
        )
    )
    assert result.status == "abstained"
    assert captured["answers"] == answers
    assert captured["resume_record"].answeredAttempt == original.answeredAttempt


def test_completed_resume_regenerates_report_without_reentering_analysis(
    monkeypatch,
) -> None:
    store, prefix, record = memory_journal()
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    monkeypatch.setattr(
        journal,
        "analysis_snapshot",
        lambda *_: {
            "status": "completed",
            "finalAnalysis": {"limitations": ["No donor replication"]},
        },
    )
    calls = []
    monkeypatch.setattr(
        AutomatedWorkflowResult, "report", lambda self: calls.append(self.workflowRunId)
    )
    monkeypatch.setattr(
        orchestrator,
        "_continue",
        lambda *_a, **_k: pytest.fail("Numerical workflow repeated"),
    )
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr", workflowRunId=record.workflowRunId
        )
    )
    assert result.status == "completed"
    assert result.limitations == ["No donor replication"]
    assert calls == [record.workflowRunId]


def test_nonstage_failure_keeps_last_stage_and_resume_address(monkeypatch) -> None:
    store, prefix, record = memory_journal("analysis")
    journal._start_attempt(
        store.zw, prefix, record.workflowRunId, "parameter_tuning", record, []
    )
    orchestrator = AgentOrchestrator("test-model")

    def fail(*args, progress, **kwargs):
        progress.extend(["data_enrichment", "parameter_tuning"])
        raise RuntimeError("The selected full-cohort candidate lacks markers")

    monkeypatch.setattr(orchestrator, "_execute_stages", fail)
    result = orchestrator._continue(
        store, WorkflowIdentity(record.workflowRunId, "analysis"), record, answers={}
    )
    assert result.currentStage == "parameter_tuning"
    assert result.workflowRunId == record.workflowRunId
    assert result.workspace == "analysis"
    assert "lacks markers" in result.notes[0]


def test_gene_annotations_are_part_of_completed_request_identity(tmp_path) -> None:
    orchestrator, store, record, resume = _saved_workflow(tmp_path / "rna.zarr")
    store.get_assay("RNA").feats.insert(
        "names", np.array(["GENE9", "RPS3", "GENE1", "GENE2"]), overwrite=True
    )
    with pytest.raises(ValueError, match="data or relevant metadata changed"):
        orchestrator.load_request_for_resume(resume)


def test_provider_configuration_distinguishes_identically_named_models() -> None:
    from scarf.agent.orchestrator.main import _model_identity

    first = SimpleNamespace(
        model_name="rna-model",
        system="openai",
        settings={"temperature": 0},
        provider=SimpleNamespace(name="server", base_url="https://first.example/v1"),
    )
    second = SimpleNamespace(
        **{
            **vars(first),
            "provider": SimpleNamespace(
                name="server", base_url="https://second.example/v1"
            ),
        }
    )
    assert _model_identity(first) != _model_identity(second)
    text_only = SimpleNamespace(**vars(first), supports_image_input=False)
    assert _model_identity(first) != _model_identity(text_only)
    assert _model_identity(text_only) == _model_identity(
        SimpleNamespace(**vars(first), profile={"supports_image_input": False})
    )


def test_corrected_answer_replaces_one_whose_answering_attempt_failed(
    monkeypatch,
) -> None:
    store, record, _, original = _pause_with_interrupted_answer(failed=True)
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    captured = {}

    def continue_work(*args, **kwargs):
        captured.update(kwargs)
        return AutomatedWorkflowResult(status="abstained")

    monkeypatch.setattr(orchestrator, "_continue", continue_work)
    corrected = {"enrichmentDirections": {"organism": "mouse"}}
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr",
            workflowRunId=record.workflowRunId,
            answers=corrected,
        )
    )
    assert result.status == "abstained"
    assert captured["answers"] == corrected
    assert captured["resume_record"].answeredAttempt == original.answeredAttempt


def test_failed_tuning_attempt_does_not_replace_its_committed_answer(
    monkeypatch,
) -> None:
    store, record, answers, _ = _pause_with_interrupted_answer(failed=True, tuning=True)
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    monkeypatch.setattr(
        orchestrator,
        "_continue",
        lambda *args, **kwargs: pytest.fail("A committed answer must not change"),
    )
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr",
            workflowRunId=record.workflowRunId,
            answers=answers,
        )
    )
    assert result.status == "failed"
    assert "exact pending stage" in result.notes[0]


def test_invalid_tuning_answer_is_rejected_before_any_attempt(monkeypatch) -> None:
    store, record, _, _ = _pause_with_interrupted_answer(tuning=True, interrupted=False)
    orchestrator = AgentOrchestrator("test-model")
    monkeypatch.setattr(
        orchestrator, "load_request_for_resume", lambda request: (record, store)
    )
    monkeypatch.setattr(
        orchestrator,
        "_continue",
        lambda *args, **kwargs: pytest.fail("An invalid answer must not run"),
    )
    prefix = journal._orchestration_prefix(store)
    before = journal._list_keys(store.zw, prefix)
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath="analysis.zarr",
            workflowRunId=record.workflowRunId,
            answers={"parameter_tuning": {"action": "accept"}},
        )
    )
    assert result.status == "failed"
    assert "not a valid assessment" in result.notes[0]
    assert journal._list_keys(store.zw, prefix) == before


def test_resume_failure_reports_the_latest_started_stage(tmp_path) -> None:
    orchestrator, store, record, resume = _saved_workflow(tmp_path / "rna.zarr")
    prefix = journal._orchestration_prefix(store)
    journal._start_attempt(
        store.zw, prefix, record.workflowRunId, "data_enrichment", record, []
    )
    result = orchestrator.resume(
        resume.model_copy(update={"answers": {"unexpected": "answer"}})
    )
    assert result.status == "failed"
    assert result.currentStage == "data_enrichment"
    assert "exact pending stage" in result.notes[0]


def test_unrelated_failure_leaves_an_earlier_orphan_start_open(monkeypatch) -> None:
    store, prefix, record = memory_journal()
    journal._start_attempt(store.zw, prefix, record.workflowRunId, "ingest", record, [])
    runner = AgentOrchestrator("test-model")

    def fail(*args, **kwargs):
        raise RuntimeError("Validation failed before any stage attempt")

    monkeypatch.setattr(runner, "_execute_stages", fail)
    result = runner._continue(
        store, WorkflowIdentity(record.workflowRunId, None), record, answers={}
    )
    assert result.status == "failed"
    assert (
        journal._stage_outcomes(store.zw, prefix, record.workflowRunId, "ingest") == []
    )


def test_request_is_committed_only_after_its_ingest_stage(
    tmp_path, monkeypatch
) -> None:
    path = create_store(tmp_path / "rna.zarr")
    request = AutomatedWorkflowRequest(
        sourcePath=str(path),
        studyContext="Independent donors in two conditions",
        studyObjective="Resolve stable populations",
        analysisAssays=["RNA"],
    )
    interrupted = AgentOrchestrator("test-model")

    def stop(*args, **kwargs):
        raise KeyboardInterrupt("Interrupted while recording ingest")

    monkeypatch.setattr(interrupted, "record_ingest_stage", stop)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run(request)
    root = zarr.open_group(str(path), mode="r")
    prefix = "agents/orchestrations"
    assert not [
        key for key in journal._list_keys(root, prefix) if key.endswith("/request.json")
    ]

    def stop_after_commit(self, store, workflow, record, **kwargs):
        return AutomatedWorkflowResult(
            status="abstained", workflowRunId=workflow.workflowRunId
        )

    monkeypatch.setattr(AgentOrchestrator, "_continue", stop_after_commit)
    result = AgentOrchestrator("test-model").run(request)
    assert result.status == "abstained"
    assert result.workflowRunId is not None
    root = zarr.open_group(str(path), mode="r")
    assert journal.read_request(root, prefix, result.workflowRunId)
    [ingest] = journal._stage_outcomes(root, prefix, result.workflowRunId, "ingest")
    assert ingest.status == "done"


def test_identities_are_checked_before_any_input_conversion(tmp_path) -> None:
    from scarf.agent.config import AgentRunConfig

    config = AutomatedWorkflowConfig(
        agentRunConfig=AgentRunConfig(extraModelSettings={"opaque": object()})
    )
    source = tmp_path / "study.h5ad"
    result = AgentOrchestrator("test-model", config=config).run(
        AutomatedWorkflowRequest(
            sourcePath=str(source),
            studyContext="Independent donors in two conditions",
            studyObjective="Resolve stable populations",
        )
    )
    assert result.status == "failed"
    assert "serialize" in result.notes[0]
    assert not (tmp_path / "study.zarr").exists()


def test_model_identity_keeps_json_digests_and_names_opaque_settings() -> None:
    import httpx2

    from scarf.agent.orchestrator.main import _model_identity

    assert _model_identity("openai:gpt-4o") == (
        "builtins.str:openai:gpt-4o:"
        "5368fcc89da9c6802c8b69174e48c5cf18c2e1946c06c256843b3e6516434145"
    )

    def opaque() -> SimpleNamespace:
        return SimpleNamespace(
            model_name="rna-model", settings={"timeout": httpx2.Timeout(30.0)}
        )

    assert _model_identity(opaque()) == _model_identity(opaque())
    assert _model_identity(opaque()) != _model_identity(
        SimpleNamespace(model_name="rna-model", settings={"timeout": 30.0})
    )
