"""Mandatory exploration and visible conservative resolutions."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna
from scarf.agent import workflow
from scarf.agent.models import Choice, ContextDecision
from scarf.agent.records import RunRecords, procedure_identity
from tests.test_agent_recovery import science as science


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


def test_prompt_trimming_empties_optional_tables_only_until_the_budget_fits() -> None:
    evidence = {
        "byGroup": {f"group{index:02d}": "x" * 40 for index in range(80)},
        "clusterCounts": {str(index): index for index in range(30)},
        "eligibleOptions": ["c0:r0.5"],
    }
    original = json.dumps(evidence, sort_keys=True)
    # The budget reserves 16 KiB of the limit for instructions and schema.
    result = workflow._bounded_evidence(evidence, 16_384 + 700)
    assert result == {
        "byGroup": {},
        "clusterCounts": evidence["clusterCounts"],
        "eligibleOptions": ["c0:r0.5"],
        "promptOmissions": [{"path": "evidence.byGroup", "omitted": 80}],
    }
    assert len(json.dumps(result).encode()) <= 700
    assert json.dumps(evidence, sort_keys=True) == original


def test_policy_resolution_requires_its_saved_decision_and_frozen_order(
    tmp_path: Path,
) -> None:
    from scarf.agent.records import RecordError

    records = RunRecords.create(tmp_path / "records", {})
    deferred = Choice(
        action="defer",
        acceptableOptionIds=["c0:r0.5", "c0:r0.75"],
        evidenceIds=["c0:r0.5", "c0:r0.75"],
        deferralReason="ambiguousSelection",
        question="Which measured granularity should be presented?",
        rationale="Both measured partitions remain acceptable",
    )
    config = AnalysisConfig()
    options = {"key": "choose_finalist", "stage": "finalists"}
    with pytest.raises(RecordError, match="requires its saved model decision"):
        workflow._resolve_decision(
            records, deferred, config, option_order=["c0:r0.5", "c0:r0.75"], **options
        )
    assert records.events() == []
    records.append(
        "decisionAccepted",
        decisionId="choose_finalist",
        stage="finalists",
        evidenceDigest="frozen-evidence",
        output=deferred.model_dump(mode="json"),
    )
    resolved = workflow._resolve_decision(
        records, deferred, config, option_order=["c0:r0.75", "c0:r0.5"], **options
    )
    assert resolved.action == "choose"
    assert resolved.optionIds == ["c0:r0.75"]
    assert resolved.question is None and resolved.acceptableOptionIds == []
    saved = records.latest("decisionResolved")
    assert saved["acceptedSequence"] == 1
    assert saved["evidenceDigest"] == "frozen-evidence"
    assert saved["rule"] == "orderedAcceptableOption"
    before = records.events()
    # A resumed resolution cannot silently change under another frozen order.
    with pytest.raises(RecordError, match="no longer matches frozen evidence"):
        workflow._resolve_decision(
            records, deferred, config, option_order=["c0:r0.5", "c0:r0.75"], **options
        )
    assert records.events() == before


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


def _fail_second_probe(
    science: Any,
    monkeypatch: Any,
    cause: BaseException,
    *,
    failed_status: str = "failed",
    damage_parent: bool = False,
) -> list[str]:
    """Run the registered probes and fail the PC probe (c2) at its PCA stage."""
    from scarf.agent.choices import native_probe_options
    from scarf.datastore.pipeline_run import PipelineExecutionError
    from tests.test_agent_recovery import Run

    monkeypatch.setattr(workflow, "native_probe_options", native_probe_options)
    monkeypatch.setattr(workflow, "compare_candidates", lambda *args: [])
    execute = workflow.execute_pipeline
    attempted: list[str] = []

    def fail_probe(
        store: Any, prepared: Any, candidate: Any, config: Any, **kwargs: Any
    ) -> Any:
        attempted.append(candidate.candidateId)
        if candidate.candidateId == "c2":
            failed = Run("numerical-failure")
            failed.status = failed_status
            science.runs[failed.run_id] = failed
            if damage_parent:
                science.store.inspect_artifact.return_value = SimpleNamespace(
                    complete=False
                )
            raise PipelineExecutionError(failed.run_id, "pca", cause) from cause
        return execute(store, prepared, candidate, config, **kwargs)

    monkeypatch.setattr(workflow, "execute_pipeline", fail_probe)
    return attempted


def _probe_config(mode: str = "lenient") -> AnalysisConfig:
    return AnalysisConfig(
        hvgCount=20,
        pcaDims=4,
        neighborsK=7,
        resolutions=(0.5,),
        maxCandidates=4,
        interactionMode=mode,
    )


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
    from tests.test_agent_recovery import _model, _study

    attempted = _fail_second_probe(science, monkeypatch, cause)
    run = analyze_rna(
        science.source,
        run_dir=tmp_path / "analysis",
        model=_model([]),
        study=_study(),
        config=_probe_config(mode),
    )
    assert run.status == expected
    records = RunRecords(run.run_dir)
    assert attempted.count("c2") == 1
    failure = records.latest("pipelineFailed")
    assert failure["operation"] == "screen_c2"
    assert failure["errorType"] == "PipelineExecutionError"
    numerical = isinstance(cause, np.linalg.LinAlgError)
    assert failure.get("numericalFailure") == ("LinAlgError" if numerical else None)
    if expected == "completed":
        assert [row["status"] for row in run.exploration_coverage["slots"]] == [
            "measured",
            "infeasible",
            "failed",
            "measured",
        ]
        assert run.exploration_coverage["slots"][2]["reason"] == (
            "Recognized numerical failure; no automatic retry"
        )
        assert run.exploration_coverage["nativeComplete"] is False
        skipped = records.latest("candidateFailed")
        assert skipped["candidateId"] == "c2"
        assert skipped["reason"] == "LinAlgError"
        assert skipped["failureSequence"] == failure["sequence"]
        assert attempted == ["c0", "c2", "c3", "c0", "c0"]
        assert resume_rna(run.run_dir, model=_model([])).status == "completed"
        assert attempted == ["c0", "c2", "c3", "c0", "c0"]
    else:
        # Strict mode and non-numerical causes stop instead of skipping a slot.
        status = records.latest("status")
        assert status["stage"] == "explore"
        assert status["errorType"] == "PipelineExecutionError"
        assert records.latest("candidateFailed") is None
        assert attempted == ["c0", "c2"]


def test_interrupted_exploration_resumes_with_the_saved_probe_failure(
    science: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    from scarf.agent import resume_rna
    from tests.test_agent_recovery import _model, _study

    attempted = _fail_second_probe(
        science, monkeypatch, np.linalg.LinAlgError("no convergence")
    )
    decision = workflow._decision

    async def stop_before_shortlist(*args: Any, **kwargs: Any) -> Any:
        if kwargs["key"] == "shortlist_native":
            raise asyncio.CancelledError
        return await decision(*args, **kwargs)

    monkeypatch.setattr(workflow, "_decision", stop_before_shortlist)
    with pytest.raises(asyncio.CancelledError):
        analyze_rna(
            science.source,
            run_dir=tmp_path / "analysis",
            model=_model([]),
            study=_study(),
            config=_probe_config(),
        )
    records = RunRecords(tmp_path / "analysis")
    assert records.latest("candidateFailed")["candidateId"] == "c2"
    assert workflow.stage_result(records, "explore") is None
    assert attempted == ["c0", "c2", "c3"]
    monkeypatch.setattr(workflow, "_decision", decision)
    resumed = resume_rna(records.path, model=_model([]))
    assert resumed.status == "completed"
    # The terminal failure and both measurements are reused; only the finalist
    # and final pipelines are new.
    assert attempted == ["c0", "c2", "c3", "c0", "c0"]
    kinds = [row["kind"] for row in records.events()]
    assert kinds.count("candidateFailed") == 1
    assert [
        row["operation"] for row in records.events() if row["kind"] == "pipelinePlanned"
    ] == ["screen_c0", "screen_c2", "screen_c3", "finalist_0", "final"]
    assert [row["status"] for row in resumed.exploration_coverage["slots"]] == [
        "measured",
        "infeasible",
        "failed",
        "measured",
    ]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (
            "unfinished",
            "A skipped numerical attempt must have a durable failed outcome",
        ),
        ("parent", "A numerical fallback requires intact completed parent artifacts"),
    ],
)
def test_numerical_fallback_requires_a_durable_failure_and_intact_parent(
    science: Any, tmp_path: Path, monkeypatch: Any, damage: str, message: str
) -> None:
    from tests.test_agent_recovery import _model, _study

    attempted = _fail_second_probe(
        science,
        monkeypatch,
        np.linalg.LinAlgError("no convergence"),
        failed_status="running" if damage == "unfinished" else "failed",
        damage_parent=damage == "parent",
    )
    run = analyze_rna(
        science.source,
        run_dir=tmp_path / "analysis",
        model=_model([]),
        study=_study(),
        config=_probe_config(),
    )
    assert run.status == "failed"
    records = RunRecords(run.run_dir)
    status = records.latest("status")
    assert status["stage"] == "explore"
    assert status["errorType"] == "AnalysisInputError"
    assert status["message"] == message
    assert records.latest("pipelineFailed")["numericalFailure"] == "LinAlgError"
    assert records.latest("candidateFailed") is None
    assert attempted == ["c0", "c2"]


def test_probe_skip_requires_a_numerical_failure_of_its_latest_attempt(
    science: Any, tmp_path: Path
) -> None:
    from scarf.agent.models import Candidate
    from scarf.agent.records import RecordError
    from tests.test_agent_recovery import Run, _study

    records = RunRecords.create(tmp_path / "records", {"runId": "skip-test"})
    baseline = Candidate(candidateId="c0", hvgCount=20, pcaDims=4, neighborsK=7)
    probe = baseline.model_copy(
        update={"candidateId": "c2", "parentId": "c0", "pcaDims": 10}
    )
    science.runs["baseline-run"] = Run("baseline-run")
    failed = Run("failed-attempt")
    failed.status = "failed"
    science.runs[failed.run_id] = failed
    parent = {"runId": "baseline-run", "selection": {"artifactId": "same-cells"}}
    records.append("pipelinePlanned", operation="screen_c2", label="first")
    assert workflow._skippable_failure(records, science.store, probe, parent) is None
    records.append(
        "pipelineFailed",
        operation="screen_c2",
        label="first",
        runId="failed-attempt",
        errorType="PipelineExecutionError",
        numericalFailure="LinAlgError",
        numericalStage="pca",
    )
    failure = workflow._skippable_failure(records, science.store, probe, parent)
    assert failure is not None and failure["runId"] == "failed-attempt"
    # The baseline has no parent to fall back to.
    assert workflow._skippable_failure(records, science.store, baseline, None) is None
    # A newer unfinished attempt supersedes the earlier numerical failure.
    records.append("pipelinePlanned", operation="screen_c2", label="retry")
    assert workflow._skippable_failure(records, science.store, probe, parent) is None
    records.append(
        "candidateFailed",
        candidateId="c2",
        failureSequence=failure["sequence"],
        reason="LinAlgError",
        stage="pca",
    )
    with pytest.raises(RecordError, match="saved numerical skip lacks its failed"):
        workflow._measure_candidate(
            records,
            science.source,
            science.store,
            science.inspect.return_value,
            _study(),
            _probe_config(),
            RuntimeConfig(),
            probe,
            parent=parent,
            parent_candidate=baseline,
        )
    assert science.calls == []


def test_hvg_probe_matching_the_baseline_selection_is_infeasible(
    science: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    from scarf.agent.choices import native_probe_options
    from tests.test_agent_recovery import _model, _study

    monkeypatch.setattr(workflow, "native_probe_options", native_probe_options)
    monkeypatch.setattr(workflow, "compare_candidates", lambda *args: [])
    science.inspect.return_value["availableFeatures"] = 2500
    summarize = workflow.summarize_candidate
    monkeypatch.setattr(
        workflow,
        "summarize_candidate",
        lambda *args: {**summarize(*args), "hvgSelectionDigest": "same-genes"},
    )
    run = analyze_rna(
        science.source,
        run_dir=tmp_path / "analysis",
        model=_model([]),
        study=_study(),
        config=_probe_config(),
    )
    assert run.status == "completed"
    coverage = run.exploration_coverage
    assert [(row["axis"], row["status"]) for row in coverage["slots"]] == [
        ("baseline", "measured"),
        ("hvgCount", "infeasible"),
        ("pcaDims", "measured"),
        ("neighborsK", "measured"),
    ]
    assert coverage["slots"][1]["parameters"]["hvgCount"] == 2000
    assert coverage["slots"][1]["reason"] == (
        "Measured HVG selection is identical to the baseline; "
        "no distinct sensitivity comparison"
    )
    assert coverage["nativeComplete"] is False
    # The probe was measured before it was classified; nothing was skipped.
    assert [row["candidateId"] for row in run.candidates] == ["c0", "c1", "c2", "c3"]
    assert RunRecords(run.run_dir).latest("candidateFailed") is None
    assert all(row["valid"] for row in run.replay_decisions())
