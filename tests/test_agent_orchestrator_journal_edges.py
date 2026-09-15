"""Immutable evidence, integrity, and exact stage lineage contracts."""

import json
from types import SimpleNamespace

import pytest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from scarf.agent.orchestrator import journal
from scarf.agent.orchestrator.models import (
    OrchestrationResumeRecord,
    StageEvidenceReference,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
)
from scarf.agent.types import AgentDataModel
from tests.agent_journal_store import memory_journal


class Evidence(AgentDataModel):
    rationale: str
    nCells: int


def test_checkpoint_is_immutable_idempotent_and_input_bound() -> None:
    store, prefix, record = memory_journal("analysis")
    inputs = {"selection": "cells-1", "metadata": {"age": "continuous"}}
    value = {"decision": "retain", "evidence": [1, 2]}
    key = "parameter_tuning/sample0/eval0"
    assert (
        journal.save_checkpoint(store, prefix, record.workflowRunId, key, inputs, value)
        == value
    )
    before = journal._list_keys(store.zw, prefix)
    assert (
        journal.save_checkpoint(store, prefix, record.workflowRunId, key, inputs, value)
        == value
    )
    assert journal._list_keys(store.zw, prefix) == before
    assert (
        journal.load_checkpoint(store, prefix, record.workflowRunId, key, inputs)
        == value
    )
    checkpoint = journal.read_checkpoint(store, prefix, record.workflowRunId, key)
    assert checkpoint["inputs"] == inputs
    assert checkpoint["outputs"] == value
    with pytest.raises(ValueError, match="different scientific inputs"):
        journal.load_checkpoint(
            store, prefix, record.workflowRunId, key, {**inputs, "selection": "cells-2"}
        )
    with pytest.raises(ValueError, match="different outcome"):
        journal.save_checkpoint(
            store, prefix, record.workflowRunId, key, inputs, {"decision": "exclude"}
        )
    assert all(path.startswith("analysis/agents/orchestrations/") for path in before)


def test_checkpoint_corruption_is_not_silently_repaired() -> None:
    store, prefix, record = memory_journal()
    key = "preprocessing/decision"
    journal.save_checkpoint(
        store, prefix, record.workflowRunId, key, {"input": 1}, {"chosen": 2}
    )
    path = journal._checkpoint_key(prefix, record.workflowRunId, key)
    raw = journal.record_io.read_key(store.zw, path)
    data = json.loads(raw)
    data["outputs"]["chosen"] = 3
    altered = journal.record_io.display_json_bytes(data)
    sync(
        store.zw.store.set(path, default_buffer_prototype().buffer.from_bytes(altered))
    )
    with pytest.raises(ValueError, match="checksum"):
        journal.load_checkpoint(store, prefix, record.workflowRunId, key, None)
    assert journal.record_io.read_key(store.zw, path) == altered


@pytest.mark.parametrize("key", ["../escape", "x//y", "/root", "x/../y", "x\\y"])
def test_checkpoint_path_cannot_escape_owner(key: str) -> None:
    with pytest.raises(ValueError, match="path components"):
        journal._checkpoint_key("agents/orchestrations", "workflow-1", key)


def test_committed_evidence_survives_interruption_before_stage_outcome() -> None:
    store, prefix, request = memory_journal()
    first = journal._start_attempt(
        store.zw,
        prefix,
        request.workflowRunId,
        "experimental_context",
        request,
        [],
        inputs={"columns": ["condition"]},
    )
    report = Evidence(rationale="Condition crosses donors", nCells=100)
    _, reference = journal._save_stage_report(
        store, first, report, expected_type=Evidence
    )
    restarted = journal._start_attempt(
        store.zw,
        prefix,
        request.workflowRunId,
        "experimental_context",
        request,
        [],
        inputs=first.inputs,
    )
    recovered, recovered_ref = journal._recover_persisted_stage_report(
        store, restarted, expected_type=Evidence
    )
    assert recovered == report
    assert recovered_ref == reference
    changed = journal._start_attempt(
        store.zw,
        prefix,
        request.workflowRunId,
        "experimental_context",
        request,
        [],
        inputs={"columns": ["condition", "age"]},
    )
    assert (
        journal._recover_persisted_stage_report(store, changed, expected_type=Evidence)
        is None
    )
    assert journal.read_stage_evidence(store, reference) == report.model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="owned"):
        journal.read_stage_evidence(
            store,
            StageEvidenceReference.model_validate(
                {**reference.model_dump(), "stage": "preprocessing"}
            ),
        )
    assert not any(
        "/runs/" in key or "snapshot" in key or "verification" in key
        for key in journal._list_keys(store.zw, "agents")
    )


def test_stage_reuse_requires_exact_parent_and_resolving_evidence() -> None:
    store, prefix, request = memory_journal()
    start = journal._start_attempt(
        store.zw, prefix, request.workflowRunId, "ingest", request, []
    )
    parent = journal._complete_attempt(start, status="done")
    journal._save_outcome(store.zw, prefix, parent)
    link = journal._parent_link(parent)
    child = journal._start_attempt(
        store.zw, prefix, request.workflowRunId, "data_enrichment", request, [link]
    )
    _, ref = journal._save_stage_report(
        store,
        child,
        Evidence(rationale="RNA identified", nCells=100),
        expected_type=Evidence,
    )
    outcome = journal._complete_attempt(child, status="done", report_references=[ref])
    journal._save_outcome(store.zw, prefix, outcome)
    assert (
        journal._validated_done_outcome(
            store, prefix, request.workflowRunId, "data_enrichment", request, [link]
        )
        == outcome
    )
    assert (
        journal._validated_done_outcome(
            store, prefix, request.workflowRunId, "data_enrichment", request, []
        )
        is None
    )
    sync(
        store.zw.store.delete(
            journal._checkpoint_key(prefix, request.workflowRunId, ref.key)
        )
    )
    assert (
        journal._validated_done_outcome(
            store, prefix, request.workflowRunId, "data_enrichment", request, [link]
        )
        is None
    )
    with pytest.raises(ValueError, match="unresolved"):
        journal.analysis_snapshot(store, request.workflowRunId)


def test_resume_answers_are_committed_in_stage_inputs_without_another_ledger() -> None:
    store, prefix, request = memory_journal()
    start = journal._start_attempt(
        store.zw, prefix, request.workflowRunId, "preprocessing_plan", request, []
    )
    pause = journal._complete_attempt(
        start,
        status="needsInput",
        needs_input=WorkflowNeedsInput(
            questions=[
                WorkflowQuestion(
                    questionId="decision:cellQuality",
                    decisionId="cellQuality",
                    options=["keep"],
                    question="Which supported QC policy?",
                )
            ]
        ),
    )
    answers = {
        "decision:cellQuality": {
            "decisionId": "cellQuality",
            "optionId": "keep",
            "rationale": "Preserves replicated populations",
        }
    }
    assert journal._resume_answer_errors(pause, answers) == []
    assert journal._resume_answer_errors(pause, {"unrelated": 1})
    bad = {
        "decision:cellQuality": {
            **answers["decision:cellQuality"],
            "optionId": "invented",
        }
    }
    assert journal._resume_answer_errors(pause, bad)
    resume = OrchestrationResumeRecord(
        workflowRunId=request.workflowRunId,
        answeredAttempt=journal._parent_link(pause),
        answers=answers,
        questionIds=list(answers),
    )
    answered = journal._start_attempt(
        store.zw,
        prefix,
        request.workflowRunId,
        "preprocessing_plan",
        request,
        [],
        resume_record=resume,
    )
    replay = journal._start_attempt(
        store.zw,
        prefix,
        request.workflowRunId,
        "preprocessing_plan",
        request,
        [],
        resume_record=resume,
    )
    assert answered.inputs["resumeAnswers"] == answers
    assert journal._stage_execution_id(answered) == journal._stage_execution_id(replay)
    assert not any("/resumes/" in key for key in journal._list_keys(store.zw, prefix))


def test_stage_checksums_and_original_start_are_required() -> None:
    store, prefix, request = memory_journal()
    start = journal._start_attempt(
        store.zw, prefix, request.workflowRunId, "ingest", request, []
    )
    outcome = journal._complete_attempt(start, status="done")
    journal._save_outcome(store.zw, prefix, outcome)
    altered = outcome.model_copy(update={"inputs": {"different": True}})
    with pytest.raises(ValueError, match="started and outcome"):
        journal._stage_outcome_resolves(
            store, prefix, request.workflowRunId, request, altered
        )
    with pytest.raises(FileExistsError, match="exists"):
        journal._save_outcome(store.zw, prefix, outcome)
    no_listing = SimpleNamespace(store=SimpleNamespace(supports_listing=False))
    with pytest.raises(NotImplementedError, match="listing"):
        journal._list_keys(no_listing, prefix)


@pytest.mark.parametrize(
    "values",
    [
        {"status": "started", "completedAtNs": 1},
        {"status": "done", "startedAtNs": 2, "completedAtNs": 1},
        {"status": "needsInput"},
        {"status": "failed"},
    ],
)
def test_invalid_stage_lifecycle_is_rejected(values) -> None:
    with pytest.raises(ValueError):
        WorkflowStageAttempt(**values)


@pytest.mark.parametrize(
    "mismatch", [None, "missingMarker", "differentCohort", "unaccepted"]
)
def test_final_checkpoint_requires_exact_artifacts_cohort_and_acceptance(
    monkeypatch, mismatch
) -> None:
    from scarf.agent.orchestrator.models import FinalAnalysisHandoff, _STAGE_ORDER
    from scarf.agent.types import ArtifactReferenceModel

    class TuningProof(AgentDataModel):
        recommendedCandidateId: str
        finalClusterArtifact: ArtifactReferenceModel

    store, prefix, request = memory_journal()
    artifacts = {
        name: ArtifactReferenceModel(
            scope="datastore" if name == "cellSelection" else "assay",
            assay=None if name == "cellSelection" else "RNA",
            kind=kind,
            artifactId=f"{index:064x}",
        )
        for index, (name, kind) in enumerate(
            {
                "cellSelection": "cell_selection",
                "graph": "connectivity_map",
                "clusters": "cluster_labels",
                "umap": "embedding",
                "embeddingInitialization": "embedding_initialization",
                "markerFeatures": "feature_selection",
                "markers": "marker_table",
            }.items(),
            1,
        )
    }
    final = FinalAnalysisHandoff(
        workflowRunId=request.workflowRunId,
        primaryAssay="RNA",
        markerAssay="RNA",
        **artifacts,
    )
    inputs = {
        "preprocessedAssays": [
            {"cellSelection": artifacts["cellSelection"].model_dump(mode="json")}
        ]
    }
    if mismatch == "missingMarker":
        final = final.model_copy(update={"markers": None})
    if mismatch == "differentCohort":
        inputs["preprocessedAssays"][0]["cellSelection"]["artifactId"] = "e" * 64
    parents = []
    for stage in _STAGE_ORDER:
        start = journal._start_attempt(
            store.zw,
            prefix,
            request.workflowRunId,
            stage,
            request,
            parents,
            inputs=inputs if stage == "analysis_finalization" else {},
        )
        references = []
        if stage == "parameter_tuning":
            _, ref = journal._save_stage_report(
                store,
                start,
                TuningProof(
                    recommendedCandidateId="selected",
                    finalClusterArtifact=artifacts["clusters"],
                ),
                expected_type=TuningProof,
            )
            references.append(ref)
        outcome = journal._complete_attempt(
            start,
            status="done",
            report_references=references,
            artifacts=artifacts if stage == "analysis_finalization" else {},
            outputs={"finalAnalysis": final.model_dump(mode="json")}
            if stage == "analysis_finalization"
            else {},
        )
        journal._save_outcome(store.zw, prefix, outcome)
        parents = [journal._parent_link(outcome)]
    monkeypatch.setattr(
        journal,
        "_analysis_review_views",
        lambda *args: [
            {
                "scope": "full",
                "action": "defer" if mismatch == "unaccepted" else "accept",
                "selectedCandidateId": "selected",
            }
        ],
    )
    if mismatch:
        with pytest.raises(ValueError, match="Final analysis"):
            journal.analysis_snapshot(store, request.workflowRunId)
    else:
        snapshot = journal.analysis_snapshot(store, request.workflowRunId)
        assert snapshot["status"] == "completed"
        assert snapshot["finalAnalysis"] == final.model_dump(mode="json")
