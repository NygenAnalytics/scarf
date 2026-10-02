"""Mandatory exploration and visible conservative resolutions."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna
from scarf.agent import workflow
from scarf.agent.models import Choice, ContextDecision
from scarf.agent.records import RunRecords, procedure_identity
from tests.test_agent_recovery import science as science


def _provider(
    observed: list[dict[str, Any]], *, ambiguous: bool = False
) -> FunctionModel:
    async def respond(messages: Any, info: Any) -> ModelResponse:
        evidence = json.loads(messages[-1].parts[-1].content)["evidence"]
        observed.append(evidence)
        schema = info.output_tools[0].parameters_json_schema["title"]
        if schema == "ContextDecision":
            assert "source:summary" in evidence["evidenceIds"]
            answer = {"rationale": "Retain the supplied cohort and declared roles."}
        elif schema == "AnnotationDecision":
            answer = {
                "annotations": [
                    {
                        "clusterId": row["clusterId"],
                        "identity": "unassigned",
                        "rationale": "Synthetic evidence does not establish a biological identity.",
                    }
                    for row in evidence["clusters"]
                ]
            }
        elif evidence.get("decisionKind") == "pcProbe":
            assert evidence["allowedActions"] == ["experiment", "defer"]
            answer = {
                "action": "experiment",
                "optionIds": [
                    next(
                        key
                        for key, row in evidence["experiments"].items()
                        if row["pcaDims"] == 30
                    )
                ],
                "evidenceIds": ["c0"],
                "rationale": "Measure the higher-rank registered probe.",
            }
        elif "eligibleOptions" in evidence:
            assert evidence["allowedActions"] == ["choose", "defer"]
            assert set(evidence["eligibleOptions"]) <= set(evidence["evidenceIds"])
            answer = {
                "action": "choose",
                "optionIds": [evidence["eligibleOptions"][0]],
                "evidenceIds": evidence["eligibleOptions"],
                "rationale": "Select a measured native population partition.",
            }
            if ambiguous and len(evidence["eligibleOptions"]) >= 2:
                answer.update(
                    action="defer",
                    optionIds=[],
                    acceptableOptionIds=evidence["eligibleOptions"],
                    deferralReason="ambiguousSelection",
                    question="Both measured partitions support descriptive discovery; which should be presented?",
                )
        else:
            assert evidence["decisionKind"] == "nativeShortlist"
            assert evidence["allowedActions"] == ["shortlist", "defer"]
            assert all(
                row["status"] != "pending"
                for row in evidence["explorationCoverage"]["slots"]
            )
            answer = {
                "action": "shortlist",
                "optionIds": ["c0:r0.5", "c0:r0.75"],
                "evidenceIds": ["c0:r0.5", "c0:r0.75"],
                "rationale": "Compare measured baseline resolutions after all independent probes.",
            }
        return ModelResponse(parts=[ToolCallPart("decision", answer)])

    return FunctionModel(respond)


@pytest.fixture
def full_panel_source(tmp_path: Path) -> Path:
    from scipy.sparse import csr_matrix
    from scarf import DataStore
    from scarf.writers import SparseToZarr

    source = tmp_path / "counts.zarr"
    rng = np.random.default_rng(39)
    counts = rng.poisson(1, size=(120, 2102)).astype(np.uint32)
    for group in range(3):
        counts[group * 40 : (group + 1) * 40, 2 + group * 40 : 42 + group * 40] += (
            rng.poisson(4, size=(40, 40)).astype(np.uint32)
        )
    names = ["MT-CO1", "RPL3", *[f"GENE{i}" for i in range(2100)]]
    writer = SparseToZarr(
        csr_matrix(counts),
        str(source),
        [f"cell{i}" for i in range(120)],
        names,
        mem_budget="512M",
        nthreads=1,
    )
    writer.dump()
    store = DataStore(
        str(source),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        mem_budget="512M",
    )
    store.cells.insert("donor", np.tile(["donorA", "donorB"], 60))
    store.cells.insert(
        "author_annotation", np.repeat(["SECRET_A", "SECRET_B", "SECRET_C"], 40)
    )
    return source


def test_real_four_representation_panel_and_recorded_lenient_choice(
    full_panel_source: Path, tmp_path: Path
) -> None:
    from scarf import DataStore

    before = DataStore(
        str(full_panel_source), zarr_mode="r", nthreads=1, mem_budget="512M"
    )
    columns = list(before.cells.columns)
    selected = before.cells.fetch_all("I").copy()
    seen: list[dict[str, Any]] = []
    result = analyze_rna(
        full_panel_source,
        run_dir=tmp_path / "analysis",
        model=_provider(seen, ambiguous=True),
        study=Study(
            context="Synthetic RNA with two explicitly declared donors.",
            objective="Describe populations",
            sampleColumn="donor",
        ),
        config=AnalysisConfig(maxCandidates=4),
        runtime=RuntimeConfig(nthreads=1, memBudget="512M"),
    )
    records = RunRecords(result.run_dir)
    assert result.status == "completed", records.events()[-5:]
    coverage = result.exploration_coverage
    assert coverage["nativeComplete"] is True
    assert [row["status"] for row in coverage["slots"]] == ["measured"] * 4
    candidates = result.candidates
    assert len(candidates) == 4
    baseline = candidates[0]["parameters"]
    for row, axis in zip(
        candidates[1:], ["hvgCount", "pcaDims", "neighborsK"], strict=True
    ):
        assert {key for key in baseline if baseline[key] != row["parameters"][key]} == {
            axis
        }
        assert row["selection"] == candidates[0]["selection"]
        assert len(row["comparisons"]) == 4
    assert [row["actualHvgCount"] for row in candidates[:2]] == [1000, 2000]
    assert len(result.pipeline_runs) == 7
    assert len(result.decision_resolutions) == 1
    assert result.decision_resolutions[0]["resolved"]["optionIds"] == ["c0:r0.5"]
    assert any(row["source"] == "policy" for row in result.replay_decisions())
    finalist = records.read_json("evidence/finalist_0.json")
    assert (
        "markerSupportFraction" in finalist["metrics"]
        and "markerCoherence" not in finalist["metrics"]
    )
    assessed_run = before.pipeline.open(run_id=finalist["runId"])
    assert assessed_run["markers"] == result.pipeline["markers"]
    assert not any("SECRET_" in json.dumps(evidence) for evidence in seen)
    after = DataStore(
        str(full_panel_source), zarr_mode="r", nthreads=1, mem_budget="512M"
    )
    assert list(after.cells.columns) == columns
    np.testing.assert_array_equal(after.cells.fetch_all("I"), selected)


def test_model_cannot_shortlist_instead_of_the_pc_probe() -> None:
    config = AnalysisConfig()
    evidence = {
        "decisionKind": "pcProbe",
        "experiments": {"c0:pcaDims:30": {}},
        "candidates": [],
        "evidenceIds": ["c0"],
    }
    with pytest.raises(ValueError, match="action must"):
        workflow._validate_exploration(
            Choice(
                action="shortlist",
                optionIds=["c0:pcaDims:30"],
                evidenceIds=["c0"],
                rationale="Skip investigation",
            ),
            evidence,
            config,
        )


@pytest.mark.parametrize(
    "cause,stage,accepted",
    [
        (np.linalg.LinAlgError("failed"), "pca", True),
        (ValueError("corrupt"), "pca", False),
        (np.linalg.LinAlgError("failed"), "markers", False),
        (MemoryError(), "pca", False),
    ],
)
def test_numerical_fallback_is_narrow(
    cause: BaseException, stage: str, accepted: bool
) -> None:
    from scarf.datastore.pipeline_run import PipelineExecutionError

    error = PipelineExecutionError("failed-run", stage, cause)
    error.__cause__ = cause
    assert bool(workflow._numerical_failure(error)) is accepted


def test_optional_context_question_resolves_without_forging_model_answer(
    tmp_path: Path,
) -> None:
    from scarf.agent.provider import decide
    from scarf.agent.choices import validate_context
    from scarf.agent.prompts import CONTEXT_INSTRUCTIONS

    study = Study(context="Unknown donor roles", objective="Describe populations")
    config = AnalysisConfig()
    records = RunRecords.create(
        tmp_path / "records",
        {
            "procedureIdentity": procedure_identity(),
            "study": study.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
        },
    )
    value = ContextDecision(
        rationale="The donor role is uncertain",
        evidenceIds=["source:summary"],
        question="Does donor mean capture?",
        deferralReason="uncertainMetadata",
    )

    def respond(messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("decision", value.model_dump(mode="json"))]
        )

    accepted = asyncio.run(
        decide(
            FunctionModel(respond),
            decision_id="context",
            stage="context",
            evidence={
                "observations": {"columns": {}},
                "optionOrder": [],
                "unresolvedFactIds": [],
            },
            output_type=ContextDecision,
            validate=lambda row: validate_context(row, columns=set(), study=study),
            records=records,
            runtime=RuntimeConfig(),
            instructions=CONTEXT_INSTRUCTIONS,
        )
    )
    resolved = workflow._resolve_decision(
        records, accepted, config, key="context", stage="context", option_order=[]
    )
    assert resolved.question is None
    assert records.latest("decisionAccepted")["output"]["question"] == value.question
    assert records.latest("decisionResolved")["resolved"] == {
        "action": "retainDeclaredRoles"
    }
    original_count = len(records.events())
    workflow._resolve_decision(
        records, accepted, config, key="context", stage="context", option_order=[]
    )
    assert len(records.events()) == original_count
    replay = workflow.replay(records)
    assert [row["source"] for row in replay] == ["model", "policy"]


def test_optional_prompt_compression_preserves_choices_and_measurement_gates() -> None:
    evidence = {
        "experiments": {"x": {"pcaDims": 30}},
        "eligibleOptions": ["a"],
        "finalists": {
            "a": {
                "metrics": {"markerSupportFraction": 1.0},
                "covariateAssociations": [{"column": "long" * 1000}] * 40,
            }
        },
    }
    result = workflow._bounded_evidence(evidence, 20000)
    assert result["experiments"] == evidence["experiments"]
    assert result["eligibleOptions"] == ["a"]
    assert result["finalists"]["a"]["metrics"] == evidence["finalists"]["a"]["metrics"]
    assert result["promptOmissions"]
    assert evidence["finalists"]["a"]["covariateAssociations"]


@pytest.mark.parametrize("stage", ["explore", "finalists", "annotate"])
def test_realistic_measured_payloads_fit_the_complete_default_request(
    tmp_path: Path, stage: str
) -> None:
    from scarf.agent.choices import validate_annotations, validate_choice
    from scarf.agent.models import AnnotationDecision
    from scarf.agent.prompts import (
        ANNOTATE_INSTRUCTIONS,
        EXPLORE_INSTRUCTIONS,
        SELECT_INSTRUCTIONS,
    )

    # Shapes reflect the mounted studies: four panels with up to 33 clusters,
    # three baseline probes and dense multi-donor cluster summaries.
    candidates = []
    for index in range(4):
        candidate = {
            "candidateId": f"c{index}",
            "covariateAssociations": [
                {
                    "column": f"covariate_{column}",
                    "component": component,
                    "association": (column + 1) / 50,
                    "kind": "spearman",
                    "rowsUsed": 10000,
                    "rowsMissing": 0,
                }
                for column in range(48)
                for component in range(1, 31)
            ],
            "partitions": [
                {
                    "optionId": f"c{index}:r{resolution}",
                    "clusterCounts": {str(group): 1000 for group in range(size)},
                    "clusterCount": size,
                    "resolution": resolution,
                }
                for resolution, size in zip(
                    [0.5, 0.75, 1.0, 1.25], [20, 25, 30, 33], strict=True
                )
            ],
            "comparisons": [],
        }
        if index:
            for resolution, size in zip(
                [0.5, 0.75, 1.0, 1.25], [20, 25, 30, 33], strict=True
            ):
                overlap = [
                    {
                        "clusterId": str(group),
                        "matchedClusterId": str(group),
                        "sourceCells": 1000,
                        "intersectionCells": 500 + group,
                        "fraction": (500 + group) / 1000,
                    }
                    for group in range(size)
                ]
                candidate["comparisons"].append(
                    {
                        "resolution": resolution,
                        "adjustedRandIndex": 0.8,
                        "parentToCandidate": overlap,
                        "candidateToParent": overlap,
                        "selection": {"artifact_id": "frozen-cells"},
                    }
                )
        candidates.append(candidate)
    markers = [
        {
            "gene": f"GENE{index}",
            "score": 0.8,
            "fracExp": 0.7,
            "fracExpRest": 0.2,
            "fracExpDelta": 0.5,
        }
        for index in range(12)
    ]
    clusters = [
        {
            "clusterId": str(index),
            "count": 1000,
            "markers": markers,
            "weakMarkers": [],
            "qualifyingMarkerCount": 12,
            "groupComposition": {
                column: {
                    "levels": [
                        {
                            "value": f"observed-{column}-group-{group}",
                            "count": 10,
                            "fraction": 0.01,
                        }
                        for group in range(64)
                    ],
                    "missing": 360,
                    "unique": 64,
                    "omittedLevels": 0,
                    "dominantFraction": 0.01,
                }
                for column in ["donor", "capture", "condition"]
            },
        }
        for index in range(24)
    ]
    if stage == "explore":
        evidence = {
            "candidates": candidates,
            "experiments": {},
            "eligibleOptions": ["c0:r0.5"],
            "diagnosticPriorityColumns": ["covariate_0"],
        }
        output_type = Choice
        instructions = EXPLORE_INSTRUCTIONS
        value = Choice(
            action="shortlist",
            optionIds=["c0:r0.5"],
            rationale="Measured comparison",
            evidenceIds=["c0"],
        )

        def validator(result: Any) -> None:
            validate_choice(
                result, options={"c0:r0.5"}, actions={"shortlist"}, evidence_ids={"c0"}
            )
    elif stage == "finalists":
        evidence = {
            "eligibleOptions": ["a", "b"],
            "finalists": {
                key: workflow._selection_evidence(
                    {"clusters": clusters, "metrics": {"markerSupportFraction": 1.0}}
                )
                for key in ["a", "b"]
            },
        }
        output_type = Choice
        instructions = SELECT_INSTRUCTIONS
        value = Choice(
            action="choose",
            optionIds=["a"],
            rationale="Measured comparison",
            evidenceIds=["a"],
        )

        def validator(result: Any) -> None:
            validate_choice(
                result, options={"a", "b"}, actions={"choose"}, evidence_ids={"a", "b"}
            )
    else:
        evidence = {"clusters": clusters[:8], "context": "Descriptive RNA study"}
        output_type = AnnotationDecision
        instructions = ANNOTATE_INSTRUCTIONS
        value = AnnotationDecision(
            annotations=[
                {
                    "clusterId": str(index),
                    "identity": "unassigned",
                    "rationale": "Synthetic markers do not establish a biological identity.",
                }
                for index in range(8)
            ]
        )

        def validator(result: Any) -> None:
            validate_annotations(result, clusters=clusters[:8])

    original = json.dumps(evidence, sort_keys=True)
    observed = []

    def respond(messages: Any, info: Any) -> ModelResponse:
        observed.append(json.loads(messages[-1].parts[-1].content)["evidence"])
        return ModelResponse(
            parts=[ToolCallPart("decision", value.model_dump(mode="json"))]
        )

    records = RunRecords.create(tmp_path / stage, {})
    asyncio.run(
        workflow._decision(
            FunctionModel(respond),
            records,
            RuntimeConfig(),
            stage=stage,
            key="bounded",
            evidence=evidence,
            output_type=output_type,
            validate=validator,
            instructions=instructions,
        )
    )
    assert json.dumps(evidence, sort_keys=True) == original
    request = records.read_json(records.latest("modelRequest")["requestPath"])
    assert (
        len(
            json.dumps(
                request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        <= 65536
    )
    assert observed[0]["promptOmissions"]
    if stage == "explore":
        assert observed[0]["eligibleOptions"] == ["c0:r0.5"]
        for candidate in observed[0]["candidates"]:
            assert candidate["covariateAssociations"]
            for comparison in candidate["comparisons"]:
                assert comparison["adjustedRandIndex"] == 0.8
                assert comparison["selection"] == {"artifact_id": "frozen-cells"}
                for key in ["parentToCandidate", "candidateToParent"]:
                    assert len(comparison[key]) <= 4
                    if comparison[key]:
                        assert comparison[key][0]["fraction"] == 0.5
    elif stage == "finalists":
        assert observed[0]["eligibleOptions"] == ["a", "b"]
        assert observed[0]["finalists"]["a"]["metrics"]["markerSupportFraction"] == 1.0
    else:
        assert len(observed[0]["clusters"]) == 8
        assert all(cluster["markers"] == markers for cluster in observed[0]["clusters"])


@pytest.mark.parametrize(
    "mode,cause,expected",
    [
        ("lenient", np.linalg.LinAlgError("no convergence"), "completed"),
        ("strict", np.linalg.LinAlgError("no convergence"), "failed"),
        ("lenient", ValueError("invalid lineage"), "failed"),
        ("lenient", MemoryError(), "failed"),
    ],
)
def test_failed_probe_continuation_consumes_slot_without_repair(
    science: Any,
    tmp_path: Path,
    monkeypatch: Any,
    mode: str,
    cause: BaseException,
    expected: str,
) -> None:
    from scarf.agent import resume_rna
    from scarf.agent.choices import native_probe_options
    from scarf.datastore.pipeline_run import PipelineExecutionError
    from tests.test_agent_recovery import Run, _model, _study

    monkeypatch.setattr(workflow, "native_probe_options", native_probe_options)
    monkeypatch.setattr(workflow, "compare_candidates", lambda *args: [])
    execute = workflow.execute_pipeline
    attempted = []

    def fail_probe(
        store: Any, prepared: Any, candidate: Any, config: Any, **kwargs: Any
    ) -> Any:
        attempted.append(candidate.candidateId)
        if candidate.candidateId == "c2":
            failed = Run("numerical-failure")
            failed.status = "failed"
            science.runs[failed.run_id] = failed
            raise PipelineExecutionError(failed.run_id, "pca", cause) from cause
        return execute(store, prepared, candidate, config, **kwargs)

    monkeypatch.setattr(workflow, "execute_pipeline", fail_probe)
    config = AnalysisConfig(
        hvgCount=20,
        pcaDims=4,
        neighborsK=7,
        resolutions=(0.5,),
        maxCandidates=4,
        interactionMode=mode,
    )
    run = analyze_rna(
        science.source,
        run_dir=tmp_path / "analysis",
        model=_model([]),
        study=_study(),
        config=config,
    )
    assert run.status == expected
    records = RunRecords(run.run_dir)
    assert attempted.count("c2") == 1
    if expected == "completed":
        assert [row["status"] for row in run.exploration_coverage["slots"]] == [
            "measured",
            "infeasible",
            "failed",
            "measured",
        ]
        assert run.exploration_coverage["nativeComplete"] is False
        assert records.latest("candidateFailed")["reason"] == "LinAlgError"
        before = list(attempted)
        assert resume_rna(run.run_dir, model=_model([])).status == "completed"
        assert attempted == before
        # A same-procedure interruption after the exploration stage reuses its
        # terminal failure as well as all completed pipeline invocations.
        monkeypatch.setattr(
            workflow, "execute_pipeline", Mock(side_effect=AssertionError("must reuse"))
        )
        assert run.replay_decisions()
    else:
        assert records.latest("candidateFailed") is None
