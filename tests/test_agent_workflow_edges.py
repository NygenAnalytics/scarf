"""Workflow acceptance gates, resumable inputs, and exact numerical lineage."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import AnalysisConfig, analyze_rna, open_analysis, resume_rna
from scarf.agent import api, workflow
from scarf.agent.models import AnalysisInputError, Candidate, NeedsInput
from scarf.agent.records import RunRecords
from tests.test_agent_recovery import Ref, Run, _analyze, _model, _pipeline, _study
from tests.test_agent_recovery import records as records
from tests.test_agent_recovery import science as science


@pytest.mark.parametrize("invalid", ["missing-source", "nested-run", "missing-model"])
def test_invalid_entry_arguments_do_not_create_run_records(
    science: Any, tmp_path: Path, invalid: str
) -> None:
    source = tmp_path / "absent" if invalid == "missing-source" else science.source
    destination = source / "analysis" if invalid == "nested-run" else tmp_path / "new"
    with pytest.raises(ValueError):
        analyze_rna(
            source,
            run_dir=destination,
            model=None if invalid == "missing-model" else _model([]),
            study=_study(),
        )
    assert not destination.exists()
    assert science.inspect.call_count == 0
    assert science.calls == []


def test_resume_refuses_changed_procedure_before_any_new_work(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _analyze(science, tmp_path, _model([]))
    before = RunRecords(result.run_dir).events()
    monkeypatch.setattr(api, "procedure_identity", lambda: "changed-prompts")
    with pytest.raises(ValueError, match="implementation or prompts changed"):
        resume_rna(result.run_dir, model=_model([]))
    assert RunRecords(result.run_dir).events() == before
    assert len(science.calls) == 3


@pytest.mark.parametrize(
    ("field", "answer"),
    [("assay", "RNA"), ("organism", "mouse"), ("captureColumn", "capture")],
)
def test_missing_initial_input_is_resolved_and_frozen_before_execution(
    science: Any, tmp_path: Path, field: str, answer: str
) -> None:
    prepared = science.inspect.return_value
    science.inspect.side_effect = [NeedsInput(f"Supply {field}", field=field), prepared]
    result = _analyze(science, tmp_path, _model([]))
    assert result.status == "needsInput"
    assert science.calls == []
    question = result.pending_questions[0]
    assert question["stage"] == "inspect" and question["field"] == field
    for empty in (None, "  "):
        with pytest.raises(ValueError, match="must not be empty"):
            resume_rna(
                result.run_dir,
                model=_model([]),
                answers={question["questionId"]: empty},
            )
    result = resume_rna(
        result.run_dir, model=_model([]), answers={question["questionId"]: answer}
    )
    assert result.status == "completed"
    resolved = RunRecords(result.run_dir).latest("inputsResolved")
    container = "config" if field == "assay" else "study"
    assert resolved[container][field] == answer
    observed = science.inspect.call_args.args[2 if field == "assay" else 1]
    assert getattr(observed, field) == answer
    assert resume_rna(result.run_dir, model=None).status == "completed"
    assert len(science.calls) == 3


def test_initial_answer_cannot_replace_an_existing_scientific_policy(
    science: Any, tmp_path: Path
) -> None:
    science.inspect.side_effect = NeedsInput(
        "Change the neighbor policy", field="neighborsK"
    )
    result = _analyze(science, tmp_path, _model([]))
    question = result.pending_questions[0]
    before = RunRecords(result.run_dir).events()
    with pytest.raises(ValueError, match="change scientific inputs"):
        resume_rna(
            result.run_dir, model=_model([]), answers={question["questionId"]: 21}
        )
    assert RunRecords(result.run_dir).events() == before
    assert not science.calls


def test_verified_relocation_is_saved_and_completed_resume_needs_no_provider(
    science: Any, tmp_path: Path
) -> None:
    result = _analyze(science, tmp_path, _model([]))
    moved = tmp_path / "relocated-source"
    moved.mkdir()
    resumed = resume_rna(result.run_dir, model=None, source=moved)
    assert resumed.status == "completed"
    assert science.verify.call_args.args[0] == moved
    assert open_analysis(result.run_dir).source == moved
    rebound = RunRecords(result.run_dir).latest("sourceRebound")
    assert rebound["fingerprint"] == "frozen-source"
    assert len(science.calls) == 3


def test_failed_run_requires_a_model_for_unfinished_decisions(
    science: Any, tmp_path: Path
) -> None:
    science.inspect.side_effect = RuntimeError("Controlled inspection failure")
    result = _analyze(science, tmp_path, _model([]))
    assert result.status == "failed"
    with pytest.raises(ValueError, match="model to resume unfinished"):
        resume_rna(result.run_dir, model=None)
    assert not science.calls


def test_completed_status_without_final_record_is_not_trusted(
    science: Any, tmp_path: Path
) -> None:
    science.inspect.side_effect = RuntimeError("Incomplete inspection")
    result = _analyze(science, tmp_path, _model([]))
    RunRecords(result.run_dir).append("status", status="completed", stage="annotate")
    with pytest.raises(ValueError, match="no final pipeline record"):
        resume_rna(result.run_dir, model=None)
    assert not science.calls


def test_stage_cannot_complete_without_its_prerequisites(records: RunRecords) -> None:
    with pytest.raises(RuntimeError, match="missing its prerequisites"):
        workflow._complete(records, "finalize", {"runId": "invented"})
    assert records.events() == []
    assert not (records.path / "evidence/finalize.json").exists()


def test_saved_completed_pipeline_is_rechecked_on_reuse(
    records: RunRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records.append("pipelineCompleted", operation="screen_c0", runId="saved")
    run = Run("saved")
    run.status = "failed"
    store = SimpleNamespace(pipeline=SimpleNamespace(open=Mock(return_value=run)))
    monkeypatch.setattr(workflow, "verify_source", Mock())
    execute = Mock()
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    with pytest.raises(AnalysisInputError, match="no longer complete"):
        _pipeline(records, store, tmp_path)
    execute.assert_not_called()


def test_failed_pipeline_retries_with_a_new_label_and_same_candidate_slot(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = workflow.execute_pipeline
    fault = RuntimeError("private transport data must not be logged")
    monkeypatch.setattr(workflow, "execute_pipeline", Mock(side_effect=fault))
    result = _analyze(science, tmp_path, _model([]))
    records = RunRecords(result.run_dir)
    assert result.status == "failed"
    assert records.latest("pipelineFailed")["errorType"] == "RuntimeError"
    assert "private transport" not in json.dumps(records.events())
    first_label = records.latest("pipelinePlanned")["label"]
    monkeypatch.setattr(workflow, "execute_pipeline", execute)
    result = resume_rna(result.run_dir, model=_model([]))
    assert result.status == "completed"
    plans = [row for row in records.events() if row["kind"] == "pipelinePlanned"]
    assert plans[1]["label"] != first_label
    assert sum(row["kind"] == "candidateAdmitted" for row in records.events()) == 1
    assert len(science.calls) == 3


@pytest.mark.parametrize("stage", ["explore", "finalists"])
def test_deferred_selection_resumes_with_question_and_reuses_measurements(
    science: Any, tmp_path: Path, stage: str
) -> None:
    ordinary = _model([])

    async def defer(messages: Any, info: Any) -> ModelResponse:
        payload = json.loads(messages[-1].parts[-1].content)
        if payload["stage"] == stage:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "decision",
                        {
                            "action": "defer",
                            "question": "Which compartment is the intended focus?",
                            "rationale": "The objective needs one essential clarification",
                        },
                    )
                ]
            )
        return await ordinary.function(messages, info)

    result = _analyze(science, tmp_path, FunctionModel(defer))
    assert result.status == "needsInput"
    assert len(science.calls) == (1 if stage == "explore" else 2)
    question = result.pending_questions[0]
    assert question["stage"] == stage
    observed: list[Any] = []
    result = resume_rna(
        result.run_dir,
        model=_model(observed),
        answers={question["questionId"]: "all cells"},
    )
    assert result.status == "completed"
    assert (
        observed[0]["payload"]["evidence"]["answers"][question["questionId"]][
            "question"
        ]
        == question["question"]
    )
    assert len(science.calls) == 3


def test_no_valid_partition_returns_a_reportable_question(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarize = workflow.summarize_candidate
    monkeypatch.setattr(
        workflow,
        "summarize_candidate",
        lambda *args: {**summarize(*args), "partitions": []},
    )
    observed: list[Any] = []
    result = _analyze(science, tmp_path, _model(observed))
    assert result.status == "needsInput"
    assert "No valid partitions" in result.pending_questions[0]["question"]
    assert len(science.calls) == 1 and len(observed) == 1
    assert result.report().exists()


@pytest.mark.parametrize(
    "broken", ["analysis_cell_selection", "clusters", "markers", "incomplete"]
)
def test_finalization_rejects_changed_lineage_before_annotation(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    execute = workflow.execute_pipeline

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        run = execute(*args, **kwargs)
        if kwargs["final"]:
            if broken == "incomplete":
                science.store.inspect_artifact.return_value = SimpleNamespace(
                    complete=False
                )
            else:
                run[broken] = Ref("unaccepted-artifact")
        return run

    monkeypatch.setattr(workflow, "execute_pipeline", corrupt)
    observed: list[Any] = []
    result = _analyze(science, tmp_path, _model(observed))
    assert result.status == "failed"
    records = RunRecords(result.run_dir)
    status = records.latest("status")
    assert status["stage"] == "finalize"
    assert (
        "incomplete artifacts" in status["message"]
        if broken == "incomplete"
        else broken in status["message"]
    )
    assert workflow.stage_result(records, "annotate") is None
    assert len(observed) == 3 and result.annotations == []


def test_completed_resume_rejects_changed_final_artifact_mapping(
    science: Any, tmp_path: Path
) -> None:
    result = _analyze(science, tmp_path, _model([]))
    science.runs["run-3"]["clusters"] = Ref("replacement")
    with pytest.raises(AnalysisInputError, match="saved artifact mapping"):
        resume_rna(result.run_dir, model=None)
    assert len(science.calls) == 3


@pytest.mark.parametrize("missing", ["native", "resolution", "capacity"])
def test_corrected_shortlist_requires_exact_native_control_within_budget(
    missing: str,
) -> None:
    native = Candidate(candidateId="c0", hvgCount=20, pcaDims=4, neighborsK=7)
    corrected = native.model_copy(
        update={"candidateId": "c1", "parentId": "c0", "useHarmony": True}
    )
    candidates = {"c0": native, "c1": corrected}
    partitions = {
        "c0:r0.5": {"candidateId": "c0", "resolution": 0.5},
        "c1:r0.5": {"candidateId": "c1", "resolution": 0.5},
    }
    if missing == "native":
        del candidates["c0"]
    elif missing == "resolution":
        partitions["c0:r0.5"]["resolution"] = 1.0
    config = AnalysisConfig(maxFinalists=1 if missing == "capacity" else 2)
    with pytest.raises(AnalysisInputError, match="budget|matched native"):
        workflow._shortlist(["c1:r0.5"], partitions, candidates, config)


@pytest.mark.parametrize("scenario", ["accepted", "marker-harm", "different-cohort"])
def test_correction_selection_uses_matched_measured_evidence(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Exercise admission, matched shortlisting, acceptance, finalization and replay."""
    science.inspect.return_value["correctionEligible"] = True
    original_summary = workflow.summarize_candidate
    original_finalist = workflow.finalist_evidence

    def parameters(candidate: Candidate) -> dict[str, Any]:
        return candidate.model_dump(exclude={"candidateId", "parentId"})

    def summarize(*args: Any) -> dict[str, Any]:
        candidate = args[2]
        summary = original_summary(*args)
        summary["parameters"] = parameters(candidate)
        summary["partitions"][0].update(
            candidateId=candidate.candidateId,
            optionId=f"{candidate.candidateId}:r0.5",
        )
        if scenario == "different-cohort" and candidate.useHarmony:
            summary["selection"] = {"artifactId": "different-cells"}
        return summary

    def finalist(*args: Any) -> dict[str, Any]:
        candidate = args[2]
        result = original_finalist(*args)
        result.update(
            parameters={**parameters(candidate), "resolution": 0.5},
            selection={"artifactId": "same-cells"},
            features={"artifactId": "same-features"},
            diagnosticScope={"sample": "same-cells", "method": "bounded"},
            requiredSampleSupport=True,
            metrics={
                "mixing": {"batch": 0.8 if candidate.useHarmony else 0.5},
                "protection": {"condition": {"cLISI": 0.9, "graphConnectivity": 0.9}},
                "markerCoherence": 0.6
                if scenario == "marker-harm" and candidate.useHarmony
                else 0.9,
                "markerSpecificityMedian": 0.8,
                "crossUnitSupport": 1.0,
                "doubletHighScoreConcentration": 0.1,
            },
        )
        return result

    monkeypatch.setattr(workflow, "summarize_candidate", summarize)
    monkeypatch.setattr(workflow, "finalist_evidence", finalist)
    ordinary = _model([])
    selection_evidence: list[dict[str, Any]] = []

    async def choose(messages: Any, info: Any) -> ModelResponse:
        payload = json.loads(messages[-1].parts[-1].content)
        measured = payload["evidence"]
        if payload["stage"] == "explore":
            if len(measured["candidates"]) == 1:
                options = [
                    key
                    for key, value in measured["experiments"].items()
                    if value["useHarmony"]
                ]
                assert len(options) == 1
                answer = {
                    "action": "experiment",
                    "optionIds": options,
                    "rationale": "Measure the matched correction",
                }
            else:
                answer = {
                    "action": "shortlist",
                    "optionIds": ["c1:r0.5"],
                    "rationale": "Assess correction against its required native control",
                }
            return ModelResponse(parts=[ToolCallPart("decision", answer)])
        if payload["stage"] == "finalists":
            selection_evidence.append(measured)
            desired = "c1:r0.5" if scenario == "accepted" else "c0:r0.5"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "decision",
                        {
                            "action": "choose",
                            "optionIds": [desired],
                            "rationale": "Choose only an admissible measured finalist",
                            "evidenceIds": [desired],
                        },
                    )
                ]
            )
        return await ordinary.function(messages, info)

    result = analyze_rna(
        science.source,
        run_dir=tmp_path / "correction",
        model=FunctionModel(choose),
        study=_study(),
        config=AnalysisConfig(
            hvgCount=20,
            pcaDims=4,
            neighborsK=7,
            resolutions=(0.5,),
            maxCandidates=2,
            scoreDoublets=True,
        ),
    )
    records = RunRecords(result.run_dir)
    if scenario == "different-cohort":
        assert result.status == "failed"
        assert "exact frozen cohort" in records.latest("status")["message"]
        assert len(science.calls) == 2 and selection_evidence == []
        assert len(result.candidates) == 1
        return
    assert result.status == "completed", records.latest("status")
    assert records.read_json("evidence/explore.json")["shortlist"] == [
        "c1:r0.5",
        "c0:r0.5",
    ]
    selection = selection_evidence[0]
    if scenario == "marker-harm":
        assert selection["eligibleOptions"] == ["c0:r0.5"]
        assert "markerCoherence" in " ".join(selection["rejectedOptions"]["c1:r0.5"])
    else:
        assert selection["eligibleOptions"] == ["c0:r0.5", "c1:r0.5"]
        assert selection["rejectedOptions"] == {}
    expected = "c1:r0.5" if scenario == "accepted" else "c0:r0.5"
    assert records.read_json("evidence/finalize.json")["selected"] == expected
    assert len(science.calls) == 5
    assert [call["markers"] for call in science.calls] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert [call["final"] for call in science.calls] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert len(result.annotations) == 1
    replayed = result.replay_decisions()
    assert len(replayed) == 5
    assert all(row["valid"] for row in replayed)
