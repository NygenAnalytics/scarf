"""Saved invocation usage survives failure and is counted once in derived views."""

from typing import Any

import pytest

from scarf.agent.orchestrator import AgentOrchestrator, journal
from scarf.agent.orchestrator.models import AutomatedWorkflowRequest, WorkflowIdentity
from scarf.agent.types import AgentDataModel, AgentRunInfo, AgentUsageInfo
from tests.agent_journal_store import memory_journal


class MeasuredReport(AgentDataModel):
    runInfo: AgentRunInfo


def test_snapshot_counts_failed_and_completed_invocations_without_report_duplicates() -> (
    None
):
    store, prefix, record = memory_journal("analysis")
    inputs = {"selection": "cells", "metadata": {"age": "continuous"}}
    save = journal.model_attempt_callback(
        store, prefix, record.workflowRunId, "ingest/review", inputs
    )
    complete = AgentRunInfo(
        runId="complete",
        agentName="test",
        status="done",
        usage=AgentUsageInfo(
            requests=2,
            inputTokens=20,
            outputTokens=4,
            totalTokens=24,
            availability="reported",
        ),
    )
    failed = AgentRunInfo(
        runId="failed",
        agentName="test",
        status="failed",
        usage=AgentUsageInfo(
            requests=1,
            inputTokens=7,
            outputTokens=1,
            totalTokens=8,
            availability="partial",
        ),
        error="RuntimeError: Provider outage after a measured response",
    )
    unavailable = AgentRunInfo(
        runId="no-response",
        agentName="test",
        status="failed",
        usage=AgentUsageInfo(availability="unavailable"),
    )
    for info in (failed, complete, unavailable, complete):
        save(info)
    started = journal._start_attempt(
        store.zw, prefix, record.workflowRunId, "ingest", record, [], inputs=inputs
    )
    _, reference = journal._save_stage_report(
        store, started, MeasuredReport(runInfo=complete), expected_type=MeasuredReport
    )
    outcome = journal._complete_attempt(
        started,
        status="done",
        report_references=[reference],
        outputs={"runInfo": complete.model_dump(mode="json")},
    )
    journal._save_outcome(store.zw, prefix, outcome)
    snapshot = journal.analysis_snapshot(store, record.workflowRunId)
    assert len(snapshot["modelAttempts"]) == 4
    usage = snapshot["modelUsage"]
    assert usage["invocations"] == 3
    assert usage["failedInvocations"] == 2
    assert usage["requests"] == 3
    assert usage["inputTokens"] == 27
    assert usage["totalTokens"] == 32
    assert usage["availability"] == "partial"
    assert usage["partialUsageInvocations"] == 1
    assert usage["unavailableUsageInvocations"] == 1
    assert snapshot == journal.analysis_snapshot(store, record.workflowRunId)


def test_usage_without_reported_availability_is_not_assumed_complete() -> None:
    legacy = AgentRunInfo(
        modelName="legacy-model",
        usage=AgentUsageInfo(inputTokens=17, totalTokens=17),
    ).model_dump(mode="json")
    views = [{"runInfo": legacy}, {"report": {"runInfo": legacy}}, {"runInfo": {}}]
    usage = journal._model_usage([], views)
    assert usage["invocations"] == 1
    assert usage["inputTokens"] == 17
    assert usage["availability"] == "partial"
    assert usage["unspecifiedUsageInvocations"] == 1
    assert journal._model_usage([], [])["availability"] == "unavailable"


def test_uncaught_stage_failure_preserves_cause_and_invocation_usage(
    monkeypatch: Any,
) -> None:
    store, prefix, record = memory_journal()
    started = journal._start_attempt(
        store.zw, prefix, record.workflowRunId, "ingest", record, []
    )
    info = AgentRunInfo(
        runId="failed-call",
        status="failed",
        usage=AgentUsageInfo(inputTokens=9, totalTokens=9, availability="partial"),
    )
    error = RuntimeError("The bounded decision could not be completed")
    error.__cause__ = ValueError("Required marker comparison is missing")
    error.agent_run_info = info

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise error

    runner = AgentOrchestrator("test-model")
    monkeypatch.setattr(runner, "_execute_stages", fail)
    result = runner._continue(
        store, WorkflowIdentity(record.workflowRunId, None), record, answers={}
    )
    assert result.status == "failed"
    assert result.currentStage == "ingest"
    assert "Required marker comparison is missing" in " ".join(result.notes)
    assert result.workflowRunId == record.workflowRunId
    outcome = journal._stage_outcomes(store.zw, prefix, record.workflowRunId, "ingest")[
        0
    ]
    assert outcome.attemptId == started.attemptId
    assert outcome.outputs["runInfo"] == info.model_dump(mode="json")
    snapshot = journal.analysis_snapshot(store, record.workflowRunId)
    assert snapshot["modelUsage"]["inputTokens"] == 9


@pytest.mark.parametrize("persistence_fails", [False, True])
def test_structured_failure_preserves_root_cause_before_persistence(
    monkeypatch: Any, persistence_fails: bool
) -> None:
    runner = AgentOrchestrator("test-model")
    request = AutomatedWorkflowRequest(
        sourcePath="missing.h5ad",
        studyContext="Human cells",
        studyObjective="Assess populations",
    )
    error = RuntimeError("Could not open source")
    error.__cause__ = OSError("Exact missing file")
    if persistence_fails:
        error.add_note("Saving failure also failed")

    def fail(_request: Any) -> None:
        raise error

    monkeypatch.setattr(runner, "_run", fail)
    result = runner.run(request)
    assert result.status == "failed"
    assert "OSError: Exact missing file" in result.notes[0]
    if persistence_fails:
        assert "Saving failure also failed" in result.notes[0]


def test_report_shows_saved_failed_usage_without_model_or_numerical_calls() -> None:
    from scarf.agent.report.artifacts import scientific_summary
    from scarf.agent.report.rendering import render_analysis_document
    from tests.test_agent_report import display_payload, snapshot

    state = snapshot()
    state["modelUsage"] = {
        "invocations": 4,
        "failedInvocations": 1,
        "requests": 8,
        "validationRetries": 2,
        "inputTokens": 1234,
        "outputTokens": 56,
        "availability": "partial",
    }
    rendered = render_analysis_document(scientific_summary(state) | display_payload())
    assert "4 invocations, 1 failed" in rendered
    assert "1,234 input and 56 output" in rendered
    assert "missing usage is not zero" in rendered
    assert "2 validation corrections" in rendered
