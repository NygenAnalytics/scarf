"""End-to-end scripted decisions execute real Scarf numerical pipelines."""

from typing import Any

import json

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
from scarf.agent.records import RunRecords


def test_finalist_prompt_keeps_measurements_with_bounded_cluster_detail() -> None:
    from scarf.agent.workflow import _selection_evidence

    source = {
        "metrics": {"markerCoherence": 0.8},
        "clusters": [
            {
                "clusterId": str(index),
                "count": index + 1,
                "qualifyingMarkerCount": index % 5,
                "markers": [{"gene": f"G{gene}"} for gene in range(20)],
                "weakMarkers": [{"gene": f"W{gene}"} for gene in range(7)],
            }
            for index in range(100)
        ],
    }
    compact = _selection_evidence(source)
    assert compact["metrics"] == source["metrics"]
    assert 0 < len(compact["clusters"]) <= 12
    assert compact["omittedClusterCount"] == 100 - len(compact["clusters"])
    assert all(len(row["markers"]) <= 6 for row in compact["clusters"])
    assert all(len(row["weakMarkers"]) <= 3 for row in compact["clusters"])


def scripted_model(observed: Any) -> Any:
    async def choose(messages: Any, info: Any) -> Any:
        payload = json.loads(messages[-1].parts[-1].content)
        evidence = payload["evidence"]
        observed.append(evidence)
        name = info.output_tools[0].parameters_json_schema.get("title")
        answer: dict[str, Any]
        if name == "ContextDecision":
            answer = {
                "rationale": "Supplied study supports descriptive discovery",
                "columnRoles": {},
                "excludeFeatures": [],
            }
        elif name == "AnnotationDecision":
            answer = {
                "annotations": [
                    {
                        "clusterId": row["clusterId"],
                        "identity": "unassigned",
                        "rationale": "Provisional marker evidence requires independent review",
                    }
                    for row in evidence["clusters"]
                ]
            }
        elif "eligibleOptions" in evidence:
            assert all(row["clusters"] for row in evidence["finalists"].values())
            answer = {
                "action": "choose",
                "optionIds": [evidence["eligibleOptions"][0]],
                "rationale": "Selected a complete native finalist",
                "evidenceIds": [evidence["eligibleOptions"][0]],
            }
        elif evidence.get("decisionKind") == "pcProbe":
            answer = {
                "action": "experiment",
                "optionIds": [next(iter(evidence["experiments"]))],
                "rationale": "Measure the registered PC probe",
                "evidenceIds": ["c0"],
            }
        else:
            partitions = evidence["candidates"][0]["partitions"]
            answer = {
                "action": "shortlist",
                "optionIds": [partitions[0]["optionId"]],
                "rationale": "Bounded descriptive baseline",
                "evidenceIds": [partitions[0]["optionId"]],
            }
        return ModelResponse(parts=[ToolCallPart("decision", answer)])

    return FunctionModel(choose)


def test_real_pipeline_annotations_reuse_and_readonly_export(
    agent_rna_source: Any, tmp_path: Any
) -> None:
    from scarf import DataStore

    before = DataStore(
        str(agent_rna_source), zarr_mode="r", nthreads=2, mem_budget="512M"
    )
    columns = list(before.cells.columns)
    initial = before.cells.to_pandas_dataframe(columns, key=None)
    observed: list[dict[str, Any]] = []
    result = analyze_rna(
        agent_rna_source,
        run_dir=tmp_path / "analysis",
        model=scripted_model(observed),
        study=Study(
            context="One donor, no hypothesis testing",
            objective="Describe major populations",
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
    assert result.annotations
    assert set(row["identity"] for row in result.annotations) == {"unassigned"}
    assert len(observed) >= 4
    assert "HELD_OUT_ALPHA" not in json.dumps(observed)
    assert "doublets" not in result.artifacts
    calls = [row for row in records.events() if row["kind"] == "pipelineCompleted"]
    assert len(calls) == 5
    run = result.pipeline
    finalist_id = next(
        row["runId"] for row in calls if row["operation"] == "finalist_0"
    )
    finalist = before.pipeline.open(run_id=finalist_id)
    assert run["markers"] == finalist["markers"]
    assert run["clusters"] == finalist["clusters"]
    after = DataStore(
        str(agent_rna_source), zarr_mode="r", nthreads=2, mem_budget="512M"
    )
    assert list(after.cells.columns) == columns
    assert after.cells.to_pandas_dataframe(columns, key=None).equals(initial)
    assert result.report().exists()
    for name in ("umap_clusters.png", "marker_dotplot.png"):
        assert (result.run_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert records.latest("reportError") is None
    result.export(tmp_path / "export")
    import pandas as pd

    clusters = pd.read_csv(tmp_path / "export" / "clusters.csv")
    embedding = pd.read_csv(tmp_path / "export" / "umap.csv")
    assert len(clusters) == 120
    assert clusters.iloc[:, 0].tolist() == embedding.iloc[:, 0].tolist()
    count = len(observed)
    resumed = resume_rna(result.run_dir, model=scripted_model(observed))
    assert resumed.status == "completed"
    assert len(observed) == count
    assert open_analysis(result.run_dir).status == "completed"
    assert [row["candidateId"] for row in result.candidates] == ["c0", "c2", "c3"]
    replayed = result.replay_decisions()
    assert len(replayed) == len(observed)
    assert all(row["valid"] for row in replayed)


def test_sync_api_rejects_notebook_loop_before_creating_files(tmp_path: Any) -> None:
    import asyncio

    async def call() -> Any:
        with pytest.raises(RuntimeError, match="event loop"):
            analyze_rna(
                tmp_path,
                run_dir=tmp_path / "run",
                model="unused",
                study=Study(context="Study", objective="Discovery"),
            )

    asyncio.run(call())
    assert not (tmp_path / "run").exists()


def test_provider_failure_is_inspectable_without_store_results(
    agent_rna_source: Any, tmp_path: Any
) -> None:
    from pydantic_ai.exceptions import ModelHTTPError

    async def failure(messages: Any, info: Any) -> Any:
        raise ModelHTTPError(401, "offline-model", {"error": "unauthorized"})

    result = analyze_rna(
        agent_rna_source,
        run_dir=tmp_path / "failed",
        model=FunctionModel(failure),
        study=Study(context="Study", objective="Discovery"),
        config=AnalysisConfig(hvgCount=40, pcaDims=4, neighborsK=7),
        runtime=RuntimeConfig(nthreads=2, memBudget="512M"),
    )
    assert result.status == "failed"
    assert result.report().exists()
    assert (
        len(
            [
                row
                for row in RunRecords(result.run_dir).events()
                if row["kind"] == "modelRequest"
            ]
        )
        == 1
    )
