"""Recovery acceptance tests with scripted models and a controlled science boundary."""

import json
from typing import Any
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna, resume_rna
from scarf.agent import api, evidence, workflow
from scarf.agent.models import Candidate
from scarf.agent.records import RunRecords
from scarf.agent.result import AnalysisRun


@dataclass(frozen=True)
class Ref:
    name: str

    def to_dict(self) -> Any:
        return {"artifactId": self.name}


class Run(dict[str, Ref]):
    def __init__(self, run_id: Any = "run-1", *, final: Any = False) -> None:
        super().__init__(
            analysis_cell_selection=Ref("same-cells"),
            clusters=Ref("same-clusters"),
            markers=Ref("same-markers"),
        )
        if final:
            self["umap"] = Ref("final-umap")
        self.run_id = run_id
        self.status = "completed"


def _study() -> Any:
    return Study(
        context="One donor, published filtered RNA", objective="Describe populations"
    )


def _candidate() -> Any:
    return Candidate(candidateId="c0", hvgCount=20, pcaDims=4, neighborsK=7)


@pytest.fixture
def records(tmp_path: Any) -> Any:
    return RunRecords.create(tmp_path / "records", {"runId": "recovery-test"})


def _planned(records: Any) -> Any:
    records.append(
        "pipelinePlanned",
        operation="screen_c0",
        label="prior-attempt",
        process={"pid": 1234},
    )


def _pipeline(records: Any, store: Any, tmp_path: Any) -> Any:
    return workflow._pipeline(
        records,
        tmp_path,
        store,
        {},
        _study(),
        AnalysisConfig(),
        RuntimeConfig(),
        _candidate(),
        operation="screen_c0",
    )


def test_completed_by_label_recovers_before_checking_old_process(
    records: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    _planned(records)
    completed = Run()
    store = SimpleNamespace(pipeline=SimpleNamespace(open=Mock(return_value=completed)))
    execute = Mock(side_effect=AssertionError("Completed work must not execute again"))
    alive = Mock(side_effect=AssertionError("Completed labels precede process checks"))
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    monkeypatch.setattr(workflow, "process_alive", alive)
    monkeypatch.setattr(workflow, "verify_source", Mock())
    assert _pipeline(records, store, tmp_path) is completed
    store.pipeline.open.assert_called_once_with(label="prior-attempt")
    assert records.latest("pipelineCompleted")["recovered"] is True
    execute.assert_not_called()


def test_unfinished_live_process_refuses_retry(
    records: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    _planned(records)
    store = SimpleNamespace(
        pipeline=SimpleNamespace(open=Mock(side_effect=KeyError("unpublished")))
    )
    execute = Mock()
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    monkeypatch.setattr(workflow, "process_alive", lambda identity: True)
    monkeypatch.setattr(workflow, "verify_source", Mock())
    with pytest.raises(RuntimeError, match="still be alive"):
        _pipeline(records, store, tmp_path)
    execute.assert_not_called()
    assert len(records.events()) == 1


def test_stopped_process_retries_as_new_labeled_pipeline(
    records: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    _planned(records)
    store = SimpleNamespace(
        pipeline=SimpleNamespace(open=Mock(side_effect=KeyError("unpublished")))
    )
    replacement = Run("retry-run")
    execute = Mock(return_value=replacement)
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    monkeypatch.setattr(workflow, "process_alive", lambda identity: False)
    monkeypatch.setattr(workflow, "verify_source", Mock())
    assert _pipeline(records, store, tmp_path) is replacement
    assert execute.call_args.kwargs["label"] == "agent_recovery-test_screen_c0_1"
    plans = [event for event in records.events() if event["kind"] == "pipelinePlanned"]
    assert len(plans) == 2
    assert plans[0]["label"] != plans[1]["label"]
    assert records.latest("pipelineCompleted")["runId"] == "retry-run"


def test_corrupt_completed_label_does_not_start_replacement(
    records: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    _planned(records)
    store = SimpleNamespace(
        pipeline=SimpleNamespace(open=Mock(side_effect=ValueError("duplicated label")))
    )
    execute = Mock()
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    monkeypatch.setattr(workflow, "verify_source", Mock())
    with pytest.raises(ValueError, match="duplicated label"):
        _pipeline(records, store, tmp_path)
    execute.assert_not_called()


@pytest.fixture
def science(tmp_path: Any, monkeypatch: Any) -> Any:
    """Keep real decisions/records/stages while substituting numerical computation."""
    source = tmp_path / "prepared-source"
    source.mkdir()
    runs: dict[str, Run] = {}
    labels: dict[str, Run] = {}
    calls: list[dict[str, Any]] = []

    def open_pipeline(*, run_id: Any = None, label: Any = None) -> Any:
        return runs[run_id] if run_id else labels[label]

    store = SimpleNamespace(
        pipeline=SimpleNamespace(open=Mock(side_effect=open_pipeline)),
        inspect_artifact=Mock(return_value=SimpleNamespace(complete=True)),
        list_artifacts=Mock(return_value=[]),
    )
    prepared = {
        "assay": "RNA",
        "source": str(source),
        "fingerprint": "frozen-source",
        "inputCells": 40,
        "retainedCells": 40,
        "availableFeatures": 30,
        "correctionEligible": False,
        "contextEvidence": {"columns": {}, "references": []},
        "limitations": ["Synthetic numerical adapter for recovery tests"],
    }
    inspect = Mock(return_value=prepared)
    verify = Mock()

    def execute(
        store: Any, prepared: Any, candidate: Any, config: Any, **kwargs: Any
    ) -> Any:
        calls.append(kwargs)
        run = Run(f"run-{len(calls)}", final=kwargs["final"])
        runs[run.run_id] = run
        labels[kwargs["label"]] = run
        return run

    def summarize(
        store: Any, run: Any, candidate: Any, prepared: Any, config: Any
    ) -> Any:
        return {
            "candidateId": candidate.candidateId,
            "runId": run.run_id,
            "parameters": candidate.model_dump(exclude={"candidateId", "parentId"}),
            "selection": run["analysis_cell_selection"].to_dict(),
            "partitions": [
                {
                    "optionId": f"{candidate.candidateId}:r{resolution:g}",
                    "candidateId": candidate.candidateId,
                    "resolution": resolution,
                    "score": 0.6,
                }
                for resolution in config.resolutions
            ],
        }

    def finalist(
        store: Any, run: Any, candidate: Any, prepared: Any, config: Any
    ) -> Any:
        return {
            "candidateId": candidate.candidateId,
            "runId": run.run_id,
            "clusters": [
                {
                    "clusterId": "0",
                    "count": 40,
                    "markers": [],
                    "weakMarkers": [],
                    "qualifyingMarkerCount": 0,
                }
            ],
            "metrics": {},
        }

    for module in (workflow, evidence):
        monkeypatch.setattr(module, "verify_source", verify)
        monkeypatch.setattr(module, "open_store", lambda *args, **kwargs: store)
    monkeypatch.setattr(workflow, "inspect_source", inspect)
    monkeypatch.setattr(workflow, "prepare_context", lambda inspected, *args: inspected)
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    monkeypatch.setattr(workflow, "summarize_candidate", summarize)
    monkeypatch.setattr(workflow, "finalist_evidence", finalist)
    # Recovery tests model durable calls, not alternative numerical graphs.
    # Probe execution and comparisons have separate real-pipeline coverage.
    monkeypatch.setattr(
        workflow,
        "native_probe_options",
        lambda *args: {"hvgCount": {}, "pcaDims": {}, "neighborsK": {}},
    )
    monkeypatch.setattr(workflow, "compare_candidates", lambda *args: {})
    # Other implementation work can run concurrently; this test isolates run
    # recovery from implementation-identity validation covered separately.
    monkeypatch.setattr(api, "procedure_identity", lambda: "test-procedure")
    monkeypatch.setattr(workflow, "procedure_identity", lambda: "test-procedure")
    return SimpleNamespace(
        source=source,
        calls=calls,
        store=store,
        inspect=inspect,
        verify=verify,
        runs=runs,
    )


def _model(
    observed: Any,
    *,
    context_question: Any = None,
    model_name: Any = "scripted-replacement",
    max_tokens: Any = 1024,
) -> Any:
    async def respond(messages: Any, info: Any) -> Any:
        payload = json.loads(messages[-1].parts[-1].content)
        measured = payload["evidence"]
        observed.append({"payload": payload, "settings": info.model_settings})
        schema = info.output_tools[0].parameters_json_schema["title"]
        answer: dict[str, Any]
        if schema == "ContextDecision":
            answer = {"rationale": "Use supplied context", "question": context_question}
            if context_question:
                answer.update(
                    deferralReason="uncertainMetadata", evidenceIds=["source:summary"]
                )
        elif schema == "AnnotationDecision":
            answer = {
                "annotations": [
                    {
                        "clusterId": row["clusterId"],
                        "identity": "unassigned",
                        "rationale": "Markers do not support a specific lineage",
                    }
                    for row in measured["clusters"]
                ]
            }
        elif "eligibleOptions" in measured:
            answer = {
                "action": "choose",
                "optionIds": [measured["eligibleOptions"][0]],
                "rationale": "A complete measured native finalist",
                "evidenceIds": [measured["eligibleOptions"][0]],
            }
        else:
            answer = {
                "action": "shortlist",
                "optionIds": ["c0:r0.5"],
                "rationale": "The bounded native baseline is sufficient",
                "evidenceIds": ["c0:r0.5"],
            }
        return ModelResponse(parts=[ToolCallPart("decision", answer)])

    return FunctionModel(
        respond, model_name=model_name, settings={"max_tokens": max_tokens}
    )


def _analyze(science: Any, tmp_path: Any, model: Any, *, runtime: Any = None) -> Any:
    return analyze_rna(
        science.source,
        run_dir=tmp_path / "analysis",
        model=model,
        study=_study(),
        config=AnalysisConfig(
            hvgCount=20,
            pcaDims=4,
            neighborsK=7,
            resolutions=(0.5, 0.75),
            maxCandidates=4,
            interactionMode="strict",
        ),
        runtime=runtime,
    )


def test_strict_context_question_resumes_with_its_original_question(
    science: Any, tmp_path: Any
) -> None:
    initial: list[dict[str, Any]] = []
    result = _analyze(
        science,
        tmp_path,
        _model(initial, context_question="Is this cohort restricted to immune cells?"),
    )
    assert result.status == "needsInput"
    assert len(initial) == 1
    assert science.calls == []
    question = result.pending_questions[0]
    observed: list[dict[str, Any]] = []
    replacement = _model(observed, max_tokens=512)
    waiting = resume_rna(result.run_dir, model=replacement)
    assert waiting.status == "needsInput" and observed == []
    with pytest.raises(ValueError, match="exact pending questionIds"):
        resume_rna(result.run_dir, model=replacement, answers={"wrong-id": "yes"})
    resumed = resume_rna(
        result.run_dir, model=replacement, answers={question["questionId"]: "yes"}
    )
    assert resumed.status == "completed"
    supplied_answer = observed[0]["payload"]["evidence"]["answers"][
        question["questionId"]
    ]
    assert supplied_answer["question"] == question["question"]
    assert supplied_answer["stage"] == "context"
    assert supplied_answer["answer"] == "yes"
    assert observed[0]["settings"]["max_tokens"] == 512
    assert science.inspect.call_count == 1
    assert len(science.calls) == 3
    assert resumed.pending_questions == []
    events = RunRecords(result.run_dir).events()
    accepted = [
        row
        for row in events
        if row["kind"] == "decisionAccepted" and row["stage"] == "context"
    ]
    assert len(accepted) == 2 and accepted[0]["decisionId"] != accepted[1]["decisionId"]


def test_provider_replacement_keeps_lifetime_request_budget(
    science: Any, tmp_path: Any
) -> None:
    async def denied(messages: Any, info: Any) -> Any:
        raise ModelHTTPError(
            401, "first-model", body={"error": {"code": "unauthorized"}}
        )

    result = _analyze(
        science, tmp_path, FunctionModel(denied), runtime=RuntimeConfig(maxRequests=1)
    )
    assert result.status == "failed"
    observed: list[dict[str, Any]] = []
    replacement = _model(observed)
    blocked = resume_rna(result.run_dir, model=replacement)
    assert blocked.status == "failed" and observed == []
    completed = resume_rna(
        result.run_dir, model=replacement, runtime=RuntimeConfig(maxRequests=5)
    )
    assert completed.status == "completed"
    assert len(observed) == 4
    records = RunRecords(result.run_dir)
    requests = [event for event in records.events() if event["kind"] == "modelRequest"]
    assert len(requests) == 5
    identities = [
        records.read_json(row["requestPath"])["model"]["model"] for row in requests
    ]
    assert identities[-1] == "scripted-replacement"
    invocation = records.latest("invocationStarted")
    assert invocation is not None
    assert invocation["runtime"] == {"maxRequests": 5}


def test_interruption_after_numerical_completion_reuses_completed_label(
    science: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    original = RunRecords.append
    interrupted = False

    def fail_once(records: Any, kind: Any, **payload: Any) -> Any:
        nonlocal interrupted
        if kind == "pipelineCompleted" and not interrupted:
            interrupted = True
            raise OSError("Simulated interruption after numerical completion")
        return original(records, kind, **payload)

    monkeypatch.setattr(RunRecords, "append", fail_once)
    observed: list[dict[str, Any]] = []
    result = _analyze(science, tmp_path, _model(observed))
    assert result.status == "failed"
    assert len(science.calls) == 1
    resumed = resume_rna(result.run_dir, model=_model(observed))
    assert resumed.status == "completed"
    assert len(science.calls) == 3
    assert len(observed) == 4
    records = RunRecords(result.run_dir)
    recovered = [
        row
        for row in records.events()
        if row["kind"] == "pipelineCompleted" and row["recovered"]
    ]
    assert len(recovered) == 1 and recovered[0]["operation"] == "screen_c0"


def test_report_failure_does_not_downgrade_scientific_completion(
    science: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        AnalysisRun, "report", Mock(side_effect=OSError("Render failed"))
    )
    result = _analyze(science, tmp_path, _model([]))
    assert result.status == "completed"
    records = RunRecords(result.run_dir)
    report_error = records.latest("reportError")
    status = records.latest("status")
    assert report_error is not None and status is not None
    assert report_error["errorType"] == "OSError"
    assert status["status"] == "completed"
    assert records.read_json("evidence/annotate.json")["annotations"]


def test_completed_resume_checks_artifact_completeness_before_return(
    science: Any, tmp_path: Any
) -> None:
    observed: list[dict[str, Any]] = []
    result = _analyze(science, tmp_path, _model(observed))
    assert result.status == "completed"
    science.store.inspect_artifact.return_value = SimpleNamespace(complete=False)
    previous_count = len(observed)
    with pytest.raises(ValueError, match="incomplete artifacts"):
        resume_rna(result.run_dir, model=_model(observed))
    assert len(observed) == previous_count
    assert len(science.calls) == 3
