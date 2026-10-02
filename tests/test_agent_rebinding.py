"""Relocation admits the saved numerical history before changing the run locator."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import (
    AnalysisConfig,
    RuntimeConfig,
    Study,
    analyze_rna,
    open_analysis,
    resume_rna,
)
from scarf.agent import evidence, workflow
from scarf.agent.models import AnalysisInputError, NeedsInput
from scarf.agent.records import RunRecords
from scarf.storage.artifacts import ArtifactRef
from tests.test_agent_recovery import Ref, Run, _analyze, _model
from tests.test_agent_recovery import science as science
from tests.test_agent_workflow import scripted_model


def _replacement_store(
    science: Any,
    replacement: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    replacement.mkdir()
    runs = {key: Run(key) for key in science.runs}
    for key, original in science.runs.items():
        runs[key].clear()
        runs[key].update(original)

    def open_pipeline(*, run_id: str) -> Any:
        return runs[run_id]

    store = SimpleNamespace(
        pipeline=SimpleNamespace(open=Mock(side_effect=open_pipeline)),
        inspect_artifact=Mock(return_value=SimpleNamespace(complete=True)),
        list_artifacts=Mock(return_value=[]),
    )

    def open_store(source: Path, *args: Any, **kwargs: Any) -> Any:
        assert not kwargs.get("writable", False)
        return store if source == replacement else science.store

    monkeypatch.setattr(evidence, "open_store", open_store)
    return SimpleNamespace(store=store, runs=runs)


@pytest.mark.parametrize("fault", ["missing", "unfinished", "artifact", "mapping"])
def test_rejected_completed_relocation_preserves_the_previous_binding(
    science: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    observed: list[Any] = []
    result = _analyze(science, tmp_path, _model(observed))
    assert result.status == "completed"
    records = RunRecords(result.run_dir)
    before = records.events()
    request_count = len(observed)
    numerical_count = len(science.calls)
    replacement = tmp_path / "replacement"
    alternate = _replacement_store(science, replacement, monkeypatch)
    saved = [row for row in before if row["kind"] == "pipelineCompleted"]
    earlier_id, final_id = saved[0]["runId"], saved[-1]["runId"]
    if fault == "missing":
        del alternate.runs[earlier_id]
    elif fault == "unfinished":
        alternate.runs[earlier_id].status = "interrupted"
    elif fault == "artifact":
        # The final run remains intact; a cached candidate's output is incomplete.
        incomplete = Ref("incomplete-candidate-output")
        alternate.runs[earlier_id]["candidate_diagnostic"] = incomplete
        alternate.store.inspect_artifact.side_effect = lambda ref: SimpleNamespace(
            complete=ref != incomplete
        )
    else:
        alternate.runs[final_id]["clusters"] = Ref("different-final-clusters")
    with pytest.raises(ValueError, match="Replacement source|Final pipeline"):
        resume_rna(result.run_dir, source=replacement, model=_model(observed))
    assert records.events() == before
    assert open_analysis(result.run_dir).source == science.source
    assert len(observed) == request_count
    assert len(science.calls) == numerical_count
    science.verify.assert_called_with(
        replacement,
        records.read_json("evidence/inspect.json"),
        Study.model_validate(records.manifest["study"]),
        AnalysisConfig.model_validate(records.manifest["config"]),
        RuntimeConfig.model_validate(records.latest("invocationStarted")["runtime"]),
    )
    assert open_analysis(result.run_dir).pipeline is science.runs[final_id]


def test_missing_cached_finalist_rejects_relocation_before_recording_answers(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = _model([])

    async def ask_about_finalists(messages: Any, info: Any) -> ModelResponse:
        payload = json.loads(messages[-1].parts[-1].content)
        if payload["stage"] == "explore":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "decision",
                        {
                            "action": "shortlist",
                            "optionIds": ["c0:r0.5", "c0:r0.75"],
                            "evidenceIds": ["c0:r0.5", "c0:r0.75"],
                            "rationale": "Compare both measured granularities",
                        },
                    )
                ]
            )
        if payload["stage"] == "finalists":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "decision",
                        {
                            "action": "defer",
                            "question": "Which compartment is the intended focus?",
                            "rationale": "Both measured granularities remain acceptable",
                            "deferralReason": "ambiguousSelection",
                            "acceptableOptionIds": payload["evidence"][
                                "eligibleOptions"
                            ],
                            "evidenceIds": payload["evidence"]["eligibleOptions"],
                        },
                    )
                ]
            )
        return await ordinary.function(messages, info)

    result = _analyze(science, tmp_path, FunctionModel(ask_about_finalists))
    assert result.status == "needsInput"
    assert len(science.calls) == 3
    question = result.pending_questions[0]
    records = RunRecords(result.run_dir)
    before = records.events()
    replacement = tmp_path / "replacement"
    alternate = _replacement_store(science, replacement, monkeypatch)
    cached_finalist = records.latest("pipelineCompleted")["runId"]
    assert workflow.stage_result(records, "finalize") is None
    del alternate.runs[cached_finalist]
    observed: list[Any] = []
    with pytest.raises(ValueError, match="Replacement source"):
        resume_rna(
            result.run_dir,
            source=replacement,
            model=_model(observed),
            answers={question["questionId"]: "All cells"},
        )
    assert records.events() == before
    assert records.latest("answers") is None
    assert records.latest("inputsResolved") is None
    reopened = open_analysis(result.run_dir)
    assert reopened.source == science.source
    assert reopened.pending_questions == [question]
    assert observed == []
    assert len(science.calls) == 3


def test_context_paused_relocation_rejects_old_artifacts_before_saving_answers(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _analyze(
        science,
        tmp_path,
        _model([], context_question="Which tissue was sampled?"),
    )
    assert result.status == "needsInput"
    question = result.pending_questions[0]
    assert question["stage"] == "context"
    records = RunRecords(result.run_dir)
    assert records.latest("pipelinePlanned") is None
    before = records.events()
    replacement = tmp_path / "replacement"
    alternate = _replacement_store(science, replacement, monkeypatch)
    normalized = ArtifactRef("assay", "normalized", "a" * 64, "RNA")
    alternate.store.list_artifacts.return_value = [normalized]
    alternate.store.inspect_artifact.return_value = SimpleNamespace(
        complete=True, operation="run_normalization"
    )
    observed: list[Any] = []
    with pytest.raises(AnalysisInputError, match="Prior complete numerical"):
        resume_rna(
            result.run_dir,
            source=replacement,
            model=_model(observed),
            answers={question["questionId"]: "Peripheral blood"},
        )
    assert records.events() == before
    assert records.latest("answers") is None
    assert records.latest("inputsResolved") is None
    assert open_analysis(result.run_dir).source == science.source
    assert open_analysis(result.run_dir).pending_questions == [question]
    assert science.verify.call_args.args[0] == replacement
    alternate.store.list_artifacts.assert_called_once_with(
        from_assay="RNA", complete_only=True
    )
    assert observed == []
    assert science.calls == []


@pytest.mark.parametrize("inspection_fails", [False, True])
def test_initial_assay_answer_rebinds_only_after_successful_source_inspection(
    science: Any, tmp_path: Path, inspection_fails: bool
) -> None:
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    prepared = {**science.inspect.return_value, "source": str(replacement)}
    science.inspect.side_effect = [
        NeedsInput("Choose one RNA assay", field="assay"),
        RuntimeError("Replacement inspection failed") if inspection_fails else prepared,
    ]
    result = _analyze(science, tmp_path, _model([]))
    records = RunRecords(result.run_dir)
    question = result.pending_questions[0]
    assert question["stage"] == "inspect"
    assert workflow.stage_result(records, "inspect") is None
    observed: list[Any] = []
    resumed = resume_rna(
        result.run_dir,
        source=replacement,
        model=_model(observed),
        answers={question["questionId"]: "RNA"},
    )
    assert science.inspect.call_args.args[0] == replacement
    assert science.inspect.call_args.args[2].assay == "RNA"
    if inspection_fails:
        assert resumed.status == "failed"
        assert workflow.stage_result(records, "inspect") is None
        assert records.latest("sourceRebound") is None
        assert open_analysis(result.run_dir).source == science.source
        assert observed == []
        assert science.calls == []
    else:
        assert resumed.status == "completed"
        rebound = records.latest("sourceRebound")
        inspected = next(
            row
            for row in records.events()
            if row["kind"] == "stageCompleted" and row["stage"] == "inspect"
        )
        assert rebound["sequence"] > inspected["sequence"]
        assert rebound["fingerprint"] == prepared["fingerprint"]
        reopened = open_analysis(result.run_dir)
        assert reopened.source == replacement
        assert reopened.pipeline is science.runs["run-3"]
        assert len(science.calls) == 3


def test_real_fresh_mount_cannot_replace_history_but_a_complete_copy_can(
    agent_rna_source: Path, tmp_path: Path
) -> None:
    from scarf import DataStore, mount_datastore

    observed: list[Any] = []
    result = analyze_rna(
        agent_rna_source,
        run_dir=tmp_path / "analysis",
        model=scripted_model(observed),
        study=Study(context="One donor", objective="Describe major populations"),
        config=AnalysisConfig(
            hvgCount=40,
            pcaDims=4,
            neighborsK=7,
            resolutions=(0.5,),
            maxCandidates=4,
        ),
        runtime=RuntimeConfig(nthreads=2, memBudget="512M"),
    )
    records = RunRecords(result.run_dir)
    assert result.status == "completed", records.latest("status")
    original_pipeline = result.pipeline
    original_refs = dict(original_pipeline)
    before = records.events()
    requests = len(observed)
    mount_path = tmp_path / "fresh-mount.zarr"
    mounted = mount_datastore(
        str(agent_rna_source),
        str(mount_path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="512M",
    )
    assert mounted.pipeline.list_runs() == ()
    assert all(mounted.inspect_artifact(ref).complete for ref in original_refs.values())
    with pytest.raises(ValueError, match="Replacement source"):
        resume_rna(result.run_dir, model=None, source=mount_path)
    assert records.events() == before
    assert open_analysis(result.run_dir).source == agent_rna_source
    assert open_analysis(result.run_dir).pipeline.run_id == original_pipeline.run_id

    relocated = tmp_path / "complete-copy.zarr"
    shutil.copytree(agent_rna_source, relocated)
    resumed = resume_rna(result.run_dir, model=None, source=relocated)
    assert resumed.status == "completed"
    assert open_analysis(result.run_dir).source == relocated
    assert dict(resumed.pipeline) == original_refs
    assert len(observed) == requests
    appended = records.events()[len(before) :]
    assert [row["kind"] for row in appended] == ["sourceRebound"]
    copied = DataStore(str(relocated), zarr_mode="r", nthreads=2, mem_budget="512M")
    assert len(copied.pipeline.list_runs()) == 5
