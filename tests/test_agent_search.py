"""A nondefault graph search and interrupted multi-batch annotation."""

import asyncio
import json
from typing import Any

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
from scarf.agent import workflow
from scarf.agent.records import RunRecords
from tests.test_agent_recovery import _analyze, _model
from tests.test_agent_recovery import science as science


def test_registered_neighbor_experiment_keeps_graph_partitions_separate(
    agent_rna_source: Any, tmp_path: Any
) -> None:
    from scarf import DataStore

    searches: list[dict[str, Any]] = []

    async def select(messages: Any, info: Any) -> Any:
        evidence = json.loads(messages[-1].parts[-1].content)["evidence"]
        output = info.output_tools[0].parameters_json_schema["title"]
        answer: dict[str, Any]
        if output == "ContextDecision":
            answer = {"rationale": "One donor supports descriptive discovery"}
        elif output == "AnnotationDecision":
            answer = {
                "annotations": [
                    {
                        "clusterId": row["clusterId"],
                        "identity": "unassigned",
                        "rationale": "This test does not assert a biological label",
                    }
                    for row in evidence["clusters"]
                ]
            }
        elif "eligibleOptions" in evidence:
            assert evidence["eligibleOptions"] == ["c3:r1"]
            answer = {
                "action": "choose",
                "optionIds": ["c3:r1"],
                "rationale": "Use the measured alternative finalist",
                "evidenceIds": ["c3:r1"],
            }
        else:
            searches.append(evidence)
            if evidence["decisionKind"] == "pcProbe":
                options = {
                    key: row
                    for key, row in evidence["experiments"].items()
                    if row["pcaDims"] == 10
                }
                assert len(options) == 1
                answer = {
                    "action": "experiment",
                    "optionIds": list(options),
                    "rationale": "Measure the registered PC probe independently of the neighbor probe",
                    "evidenceIds": ["c0"],
                }
            else:
                assert evidence["experiments"] == {}
                answer = {
                    "action": "shortlist",
                    "optionIds": ["c3:r1"],
                    "rationale": "Assess markers on the alternative graph at resolution 1",
                    "evidenceIds": ["c3:r1"],
                }
        return ModelResponse(parts=[ToolCallPart("decision", answer)])

    result = analyze_rna(
        agent_rna_source,
        run_dir=tmp_path / "neighbor-search",
        model=FunctionModel(select),
        study=Study(
            context="One donor with three synthetic populations",
            objective="Describe populations",
        ),
        config=AnalysisConfig(
            hvgCount=40,
            pcaDims=4,
            neighborsK=7,
            resolutions=(0.5, 1.0),
            maxCandidates=4,
        ),
        runtime=RuntimeConfig(nthreads=2, memBudget="512M"),
    )
    records = RunRecords(result.run_dir)
    assert result.status == "completed", records.events()[-5:]
    assert len(searches) == 2
    candidates = searches[-1]["candidates"]
    assert [row["candidateId"] for row in candidates] == ["c0", "c2", "c3"]
    assert [row["parameters"]["neighborsK"] for row in candidates] == [7, 7, 21]
    assert all(row["selection"] == candidates[0]["selection"] for row in candidates)
    for candidate in candidates:
        assert {row["resolution"] for row in candidate["partitions"]} == {0.5, 1.0}
        assert all(
            row["candidateId"] == candidate["candidateId"]
            for row in candidate["partitions"]
        )
        assert all(
            row["optionId"].startswith(candidate["candidateId"] + ":")
            for row in candidate["partitions"]
        )
    admissions = [
        row["candidate"]
        for row in records.events()
        if row["kind"] == "candidateAdmitted"
    ]
    assert len(admissions) == 3
    assert admissions[2]["parentId"] == "c0"
    assert admissions[2]["neighborsK"] == 21
    calls = {
        row["operation"]: row
        for row in records.events()
        if row["kind"] == "pipelineCompleted"
    }
    assert set(calls) == {"screen_c0", "screen_c2", "screen_c3", "finalist_0", "final"}
    store = DataStore(
        str(agent_rna_source), zarr_mode="r", nthreads=2, mem_budget="512M"
    )
    baseline = store.pipeline.open(run_id=calls["screen_c0"]["runId"])
    changed = store.pipeline.open(run_id=calls["screen_c3"]["runId"])
    finalist = store.pipeline.open(run_id=calls["finalist_0"]["runId"])
    final = result.pipeline
    assert baseline["pca"] == changed["pca"]
    assert baseline["neighbors"] != changed["neighbors"]
    assert baseline["connectivity_map"] != changed["connectivity_map"]
    assert changed["neighbors"] == finalist["neighbors"] == final["neighbors"]
    assert changed["connectivity_map"] == final["connectivity_map"]
    selected_partition = next(
        key
        for key in changed
        if key.startswith("leiden_") and float(key.removeprefix("leiden_")) == 1.0
    )
    assert changed[selected_partition] == finalist["clusters"] == final["clusters"]
    assert finalist["markers"] == final["markers"]
    assert records.read_json("evidence/finalize.json")["selected"] == "c3:r1"


def test_cancelled_second_annotation_batch_resumes_without_repeating_first(
    science: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    clusters = [
        {
            "clusterId": str(index),
            "count": index + 1,
            "markers": [],
            "weakMarkers": [],
            "qualifyingMarkerCount": 0,
        }
        for index in range(17)
    ]

    def finalists(
        store: Any, run: Any, candidate: Any, prepared: Any, config: Any
    ) -> Any:
        return {
            "candidateId": candidate.candidateId,
            "runId": run.run_id,
            "clusters": clusters,
            "metrics": {},
        }

    monkeypatch.setattr(workflow, "finalist_evidence", finalists)
    ordinary = _model([])
    attempted_batches: list[list[str]] = []

    async def interrupt(messages: Any, info: Any) -> Any:
        evidence = json.loads(messages[-1].parts[-1].content)["evidence"]
        if info.output_tools[0].parameters_json_schema["title"] == "AnnotationDecision":
            attempted_batches.append([row["clusterId"] for row in evidence["clusters"]])
            if len(attempted_batches) == 2:
                raise asyncio.CancelledError
        return await ordinary.function(messages, info)

    with pytest.raises(asyncio.CancelledError):
        _analyze(science, tmp_path, FunctionModel(interrupt))
    run_dir = tmp_path / "analysis"
    assert open_analysis(run_dir).status == "interrupted"
    assert list(map(len, attempted_batches)) == [8, 8]
    assert len(science.calls) == 3
    observations: list[dict[str, Any]] = []
    result = resume_rna(run_dir, model=_model(observations))
    assert result.status == "completed"
    assert len(result.annotations) == 17
    assert {row["clusterId"] for row in result.annotations} == {
        str(index) for index in range(17)
    }
    assert [len(row["payload"]["evidence"]["clusters"]) for row in observations] == [
        8,
        1,
    ]
    assert len(science.calls) == 3
    accepted = [
        row
        for row in RunRecords(run_dir).events()
        if row["kind"] == "decisionAccepted" and row["stage"] == "annotate"
    ]
    assert len(accepted) == 3
