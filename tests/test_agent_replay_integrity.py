"""Offline replay rejects inconsistent scientific histories without provider calls."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna
from scarf.agent import workflow
from scarf.agent.choices import native_probe_options, validate_context
from scarf.agent.models import ContextDecision
from scarf.agent.prompts import CONTEXT_INSTRUCTIONS
from scarf.agent.provider import decide
from scarf.agent.records import RecordError, RunRecords
from tests.test_agent_recovery import science as science
from tests.test_agent_workflow import scripted_model


@pytest.fixture
def measured_history(
    science: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Use recorded decisions and independent probes with a controlled numerical boundary."""
    science.inspect.return_value["availableFeatures"] = 80
    monkeypatch.setattr(workflow, "native_probe_options", native_probe_options)
    observed: list[dict[str, Any]] = []
    result = analyze_rna(
        science.source,
        run_dir=tmp_path / "analysis",
        model=scripted_model(observed),
        study=Study(
            context="Prepared synthetic study", objective="Describe populations"
        ),
        config=AnalysisConfig(hvgCount=40, pcaDims=4, neighborsK=7, maxCandidates=4),
    )
    records = RunRecords(result.run_dir)
    assert result.status == "completed", records.latest("status")
    assert all(row["valid"] for row in result.replay_decisions())
    return result, records, observed


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing-slot", "four native slots"),
        ("seed", "exploration policy changed"),
        ("baseline", "baseline differs"),
        ("parent", "different parent or position"),
        ("omitted-probe", "feasible probe was omitted"),
        ("changed-pc-choice", "registered deterministic or model choice"),
    ],
)
def test_replay_rejects_edited_probe_plan_without_new_work(
    measured_history: Any,
    fault: str,
    message: str,
) -> None:
    result, records, observed = measured_history
    relative = records.latest("explorationPlanned")["evidence"]
    plan = records.read_json(relative)
    if fault == "missing-slot":
        plan["slots"].pop()
    elif fault == "seed":
        plan["seed"] += 1
    elif fault == "baseline":
        plan["slots"][0]["parameters"]["hvgCount"] += 1
    elif fault == "parent":
        plan["slots"][2]["parentId"] = "c3"
    elif fault == "omitted-probe":
        plan["slots"][2]["parameters"] = None
    else:
        assert plan["slots"][2]["parameters"]["pcaDims"] == 10
        plan["slots"][2]["parameters"]["pcaDims"] = 30
    # Simulate an externally edited record, without bypassing replay validation.
    (records.path / relative).write_text(json.dumps(plan))
    before = records.events()
    requests = len(observed)
    with pytest.raises(RecordError, match=message):
        result.replay_decisions()
    assert records.events() == before
    assert len(observed) == requests


def test_replay_rejects_measured_parameters_that_disagree_with_admission(
    measured_history: Any,
) -> None:
    result, records, observed = measured_history
    event = next(
        row
        for row in records.events()
        if row["kind"] == "candidateMeasured" and row["candidateId"] == "c2"
    )
    evidence = records.read_json(event["evidence"])
    evidence["parameters"]["pcaDims"] = 30
    (records.path / event["evidence"]).write_text(json.dumps(evidence))
    requests = len(observed)
    with pytest.raises(RecordError, match="Measured candidate parameters differ"):
        result.replay_decisions()
    assert len(observed) == requests


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("recipe", "coverage differs from the frozen candidate recipe"),
        ("unmeasured", "claims an unmeasured candidate"),
        ("failed", "claims an unrecorded failure"),
        ("pending", "unfinished required probe"),
    ],
)
def test_completed_replay_rejects_inconsistent_exploration_coverage(
    measured_history: Any,
    fault: str,
    message: str,
) -> None:
    result, records, _ = measured_history
    coverage = records.latest("explorationCoverage")["coverage"]
    if fault == "recipe":
        coverage["slots"][2]["parameters"]["pcaDims"] = 30
    elif fault == "unmeasured":
        assert coverage["slots"][1]["parameters"] is None
        coverage["slots"][1]["status"] = "measured"
    elif fault == "failed":
        coverage["slots"][2]["status"] = "failed"
    else:
        coverage["slots"][2]["status"] = "pending"
    records.append("explorationCoverage", coverage=coverage)
    before = records.events()
    with pytest.raises(RecordError, match=message):
        result.replay_decisions()
    assert records.events() == before


def test_old_procedure_remains_readable_but_cannot_claim_current_replay(
    measured_history: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, records, observed = measured_history
    before = records.events()
    requests = len(observed)
    monkeypatch.setattr(workflow, "procedure_identity", lambda: "different-procedure")
    assert result.status == "completed"
    assert result.annotations
    with pytest.raises(ValueError, match="original implementation and prompts"):
        result.replay_decisions()
    assert records.events() == before
    assert len(observed) == requests


@pytest.fixture
def resolved_history(tmp_path: Path) -> RunRecords:
    study = Study(context="Unknown sample roles", objective="Describe populations")
    config = AnalysisConfig()
    records = RunRecords.create(
        tmp_path / "records",
        {
            "procedureIdentity": workflow.procedure_identity(),
            "study": study.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
        },
    )
    value = ContextDecision(
        rationale="Capture identity is optional",
        evidenceIds=["source:summary"],
        question="Does donor identify a capture?",
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
    workflow._resolve_decision(
        records, accepted, config, key="context", stage="context", option_order=[]
    )
    assert [row["source"] for row in workflow.replay(records)] == ["model", "policy"]
    return records


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("acceptedSequence", -1, "does not match an accepted decision"),
        ("optionOrder", ["invented"], "frozen preference or reason"),
        ("resolved", {"action": "inventRoles"}, "does not reproduce its frozen choice"),
    ],
)
def test_replay_rejects_policy_resolutions_disconnected_from_accepted_evidence(
    resolved_history: RunRecords,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    records = resolved_history
    original = records.latest("decisionResolved")
    payload = {
        key: original[key]
        for key in (
            "stage",
            "acceptedSequence",
            "acceptedDecisionId",
            "evidenceDigest",
            "reason",
            "rule",
            "optionOrder",
            "resolved",
            "limitation",
        )
    }
    payload[field] = replacement
    records.append("decisionResolved", **payload)
    before = records.events()
    with pytest.raises(RecordError, match=message):
        workflow.replay(records)
    assert records.events() == before
