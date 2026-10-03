"""Offline replay rejects inconsistent scientific histories without provider calls."""

import asyncio
import json
import shutil
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
from scarf.agent.result import AnalysisRun
from tests.test_agent_recovery import install_science
from tests.test_agent_workflow import scripted_model


@pytest.fixture(scope="module")
def measured_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Record one history with independent probes and a controlled numerical boundary."""
    from pydantic_ai import models

    base = tmp_path_factory.mktemp("measured-history")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
        science = install_science(base / "prepared-source", patch)
        science.inspect.return_value["availableFeatures"] = 80
        patch.setattr(workflow, "native_probe_options", native_probe_options)
        result = analyze_rna(
            science.source,
            run_dir=base / "analysis",
            model=scripted_model([]),
            study=Study(
                context="Prepared synthetic study", objective="Describe populations"
            ),
            config=AnalysisConfig(
                hvgCount=40, pcaDims=4, neighborsK=7, maxCandidates=4
            ),
        )
        records = RunRecords(result.run_dir)
        assert result.status == "completed", records.latest("status")
        assert all(row["valid"] for row in result.replay_decisions())
    return result.run_dir


@pytest.fixture
def measured_history(
    measured_template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AnalysisRun, RunRecords]:
    """A private copy of the shared history; replay never opens the store."""
    run_dir = tmp_path / "analysis"
    shutil.copytree(measured_template, run_dir)
    monkeypatch.setattr(workflow, "procedure_identity", lambda: "test-procedure")
    return AnalysisRun(run_dir), RunRecords(run_dir)


def _files(path: Path) -> dict[str, bytes]:
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


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
    measured_history: tuple[AnalysisRun, RunRecords],
    fault: str,
    message: str,
) -> None:
    result, records = measured_history
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
    before = _files(records.path)
    with pytest.raises(RecordError, match=message):
        result.replay_decisions()
    assert _files(records.path) == before


def test_replay_rejects_measured_parameters_that_disagree_with_admission(
    measured_history: tuple[AnalysisRun, RunRecords],
) -> None:
    result, records = measured_history
    event = next(
        row
        for row in records.events()
        if row["kind"] == "candidateMeasured" and row["candidateId"] == "c2"
    )
    evidence = records.read_json(event["evidence"])
    evidence["parameters"]["pcaDims"] = 30
    (records.path / event["evidence"]).write_text(json.dumps(evidence))
    before = _files(records.path)
    with pytest.raises(RecordError, match="Measured candidate parameters differ"):
        result.replay_decisions()
    assert _files(records.path) == before


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
    measured_history: tuple[AnalysisRun, RunRecords],
    fault: str,
    message: str,
) -> None:
    result, records = measured_history
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
    before = _files(records.path)
    with pytest.raises(RecordError, match=message):
        result.replay_decisions()
    assert _files(records.path) == before


@pytest.mark.parametrize(
    ("kind", "identifier", "message"),
    [
        ("stageCompleted", "preprocess", "missing its preprocessing evidence"),
        ("candidateMeasured", "c0", "Probe plan lacks its measured baseline"),
        ("decisionAccepted", "choose_pc_probe", "lacks its PC direction decision"),
    ],
)
def test_replay_rejects_a_probe_plan_recorded_before_its_prerequisites(
    measured_history: tuple[AnalysisRun, RunRecords],
    kind: str,
    identifier: str,
    message: str,
) -> None:
    result, records = measured_history
    plan = records.latest("explorationPlanned")
    events = records.events()
    first = next(
        row["sequence"]
        for row in events
        if row["kind"] == kind
        and identifier
        in {row.get("stage"), row.get("candidateId"), row.get("decisionId")}
    )
    # Truncating the journal keeps a valid hash chain; the plan is then
    # re-recorded without the prerequisite that the live procedure saves first.
    for row in events[first - 1 :]:
        (records.path / "events" / f"{row['sequence']:06d}.json").unlink()
    records.append("explorationPlanned", evidence=plan["evidence"])
    with pytest.raises(RecordError, match=message):
        result.replay_decisions()


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (
            {"candidateId": "c2", "pcaDims": 30},
            "Admitted candidate differs from the frozen probe plan",
        ),
        (
            {"candidateId": "c4", "parentId": "c0", "useHarmony": True},
            "Corrected candidate lacks its permitted native parent",
        ),
    ],
)
def test_replay_rejects_admissions_outside_the_frozen_probe_plan(
    measured_history: tuple[AnalysisRun, RunRecords],
    candidate: dict[str, Any],
    message: str,
) -> None:
    result, records = measured_history
    plan = records.read_json(records.latest("explorationPlanned")["evidence"])
    baseline = plan["slots"][0]["parameters"]
    if candidate["candidateId"] == "c2":
        recipe = plan["slots"][2]["parameters"]
        assert recipe["pcaDims"] == 10
    else:
        # Four-candidate procedures never permit a corrected probe.
        assert plan["harmonyPermitted"] is False
        recipe = baseline
    admitted = {**recipe, **candidate}
    records.append(
        "candidateAdmitted", candidateId=admitted["candidateId"], candidate=admitted
    )
    before = _files(records.path)
    with pytest.raises(RecordError, match=message):
        result.replay_decisions()
    assert _files(records.path) == before


def test_old_procedure_remains_readable_but_cannot_claim_current_replay(
    measured_history: tuple[AnalysisRun, RunRecords],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, records = measured_history
    before = _files(records.path)
    monkeypatch.setattr(workflow, "procedure_identity", lambda: "different-procedure")
    assert result.status == "completed"
    assert [row["identity"] for row in result.annotations] == ["unassigned"]
    with pytest.raises(ValueError, match="original implementation and prompts"):
        result.replay_decisions()
    assert _files(records.path) == before


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
