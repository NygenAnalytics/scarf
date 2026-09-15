"""Tuning resumes numerical evidence without overwriting prior attempt reports."""

from types import SimpleNamespace

import numpy as np
import pytest

from scarf.agent.experimental_context import ExperimentalContextResult
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.orchestrator import journal, rna_tuning
from scarf.agent.orchestrator.models import (
    AutomatedPreprocessingPlan,
    PreprocessedAssayHandoff,
    WorkflowIdentity,
)
from scarf.agent.orchestrator.tuning import TuningStagesMixin
from scarf.agent.parameter_tuning import ParameterTuningReport
from scarf.agent.parameter_tuning.contracts import ParameterTuningNeedsInput
from tests.agent_examples import example
from tests.agent_journal_store import memory_journal


def test_tuning_reports_survive_pause_failure_and_completion(monkeypatch) -> None:
    store, prefix, request = memory_journal("analysis")
    feature_values = {
        "ids": np.asarray(["gene-1", "gene-2", "gene-3"]),
        "names": np.asarray(["A", "B", "C"]),
    }
    features = SimpleNamespace(
        N=3,
        _get_array=feature_values.__getitem__,
        default_block_rows=lambda column: 2,
    )
    store.get_assay = lambda assay: SimpleNamespace(feats=features)
    workflow = WorkflowIdentity(request.workflowRunId, request.request.workspace)
    plan = example(AutomatedPreprocessingPlan)
    handoff = example(PreprocessedAssayHandoff)
    paused = ParameterTuningReport(
        status="needsInput",
        fromAssay="RNA",
        cellSelection=handoff.cellSelection,
        rationale="The provider did not return a valid assessment.",
        needsInput=ParameterTuningNeedsInput(question="Retry the assessment?"),
    )
    completed = example(ParameterTuningReport)
    responses = [paused, RuntimeError("Assessment provider unavailable"), completed]
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            response = responses[len(calls)]
            calls.append(response)
            if isinstance(response, Exception):
                raise response
            return response, self.summary()

        def summary(self):
            return {"history": []}

    monkeypatch.setattr(rna_tuning, "RnaTuningRun", Runner)
    owner = TuningStagesMixin()

    def run_stage():
        return owner.parameter_tuning_stage(
            store,
            workflow,
            request,
            [],
            plan,
            [handoff],
            ExperimentalContextResult.get_blank(),
            None,
            None,
            {},
            study_contract=StudyContract.get_blank(),
        )

    first, first_report = run_stage()
    assert first.status == "needsInput"
    assert first_report == paused
    first_start = journal._stage_starts(
        store.zw, prefix, workflow.workflowRunId, "parameter_tuning"
    )[0]
    # Earlier runs used this execution-owned key for an incomplete report.
    _, historical_ref = journal._save_stage_report(
        store, first_start, paused, expected_type=ParameterTuningReport
    )
    historical_path = journal._checkpoint_key(
        prefix, workflow.workflowRunId, historical_ref.key
    )
    historical_bytes = journal.record_io.read_key(store.zw, historical_path)

    failed, _ = run_stage()
    assert failed.status == "failed"
    assert "Assessment provider unavailable" in failed.error
    assert failed.reportReferences == []
    done, done_report = run_stage()
    assert done.status == "done", done.error
    assert done_report == completed
    assert len({first.attemptId, failed.attemptId, done.attemptId}) == 3
    assert (
        len(
            {
                historical_ref.key,
                first.reportReferences[0].key,
                done.reportReferences[0].key,
            }
        )
        == 3
    )
    assert journal.read_stage_evidence(store, historical_ref) == paused.model_dump(
        mode="json"
    )
    assert journal.record_io.read_key(store.zw, historical_path) == historical_bytes
    for outcome, report in ((first, paused), (done, completed)):
        assert (
            journal.load_stage_report(store, outcome, ParameterTuningReport) == report
        )

    # A completed stage still resolves its exact saved reports and artifacts.
    reused, reused_report = run_stage()
    assert reused == done
    assert reused_report == completed
    assert len(calls) == 3
    handoff.nCells += 1
    with pytest.raises(ValueError, match="Tuning inputs changed"):
        run_stage()
    assert len(calls) == 3


def test_attempt_owned_report_is_immutable_and_input_bound() -> None:
    store, prefix, request = memory_journal()
    started = journal._start_attempt(
        store.zw,
        prefix,
        request.workflowRunId,
        "parameter_tuning",
        request,
        [],
        inputs={"selection": "cells-1"},
    )
    report = ParameterTuningReport(status="needsInput", rationale="Awaiting evidence")

    def save(attempt, value):
        return journal._save_stage_report(
            store,
            attempt,
            value,
            expected_type=ParameterTuningReport,
            attempt_owned=True,
        )

    _, reference = save(started, report)
    assert save(started, report)[1] == reference
    with pytest.raises(ValueError, match="different outcome"):
        save(started, report.model_copy(update={"status": "done"}))
    with pytest.raises(ValueError, match="different scientific inputs"):
        save(started.model_copy(update={"inputs": {"selection": "cells-2"}}), report)
    assert journal.read_stage_evidence(store, reference) == report.model_dump(
        mode="json"
    )
