"""Interrupted multi-batch annotation resumes without repeating finished batches."""

import asyncio
import json
from typing import Any

import pytest
from pydantic_ai.models.function import FunctionModel

from scarf.agent import (
    open_analysis,
    resume_rna,
)
from scarf.agent import workflow
from scarf.agent.records import RunRecords
from tests.test_agent_recovery import _analyze, _model
from tests.test_agent_recovery import science as science


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
