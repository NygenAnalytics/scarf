"""QC decision ownership, human answers and replay retain exact verified evidence."""

from types import SimpleNamespace
from typing import Any
import json

import pytest

from scarf.agent.decisions.kernel import DecisionSelection
from scarf.agent.decisions.rna import build_qc_grouping_decision
from scarf.agent.orchestrator import decisions
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401
from tests.test_agent_rna_decisions import _bundle


@pytest.fixture
def decision_case(request: pytest.FixtureRequest) -> Any:
    request.getfixturevalue("memory_checkpoints")
    definition = build_qc_grouping_decision(
        evidence_bundle_id="bundle:qc",
        physical_capture_eligible=True,
        pooled_reference_eligible=False,
    )
    evidence = _bundle("qcGrouping", "bundle:qc", ["qualityControl", "design"])
    selection = DecisionSelection(
        selectedOptionId="qcGrouping:global",
        evidenceIds=[item.evidenceId for item in evidence.evidence],
        rationale="Global thresholds preserve the measured groups without losing a capture.",
        confidence="medium",
    )
    request = SimpleNamespace(
        workflowRunId="workflow", requestSha256="a" * 64, configSha256="b" * 64
    )
    owner = decisions.DecisionStagesMixin()
    return owner, request, definition, evidence, selection


@pytest.mark.parametrize("owner_name", ["rule", "agent", "human"])
@pytest.mark.parametrize("defer", [False, True])
def test_committed_qc_choice_replays_exact_owner_and_pending_outcome(
    decision_case: Any,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    defer: bool,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    owner, request, definition, evidence, selection = decision_case
    if defer:
        selection = selection.model_copy(
            update={"selectedOptionId": "qcGrouping:defer"}
        )
    answers, kwargs = {}, {}
    if owner_name == "human":
        answers["decision:qcGrouping"] = {
            "decisionId": "qcGrouping",
            "optionId": selection.selectedOptionId,
            "rationale": selection.rationale,
        }
    else:
        kwargs[f"{owner_name}_selection"] = selection
    monkeypatch.setattr(
        decisions,
        "run_agent_sync",
        lambda **kw: pytest.fail(
            "Committed or explicitly owned decisions must not call a model"
        ),
    )
    result = owner._resolve_rna_decision(
        None, request, definition, evidence, answers, **kwargs
    )
    assert result.record.source == owner_name
    assert (result.pending is not None) == defer
    assert (result.compiled is None) == defer
    replay = owner._resolve_rna_decision(
        None, request, definition, evidence, answers, **kwargs
    )
    assert result == replay
    if defer:
        assert owner._pending_decision_question(replay, definition).options == [
            item.optionId for item in definition.spec.options
        ]
    else:
        with pytest.raises(ValueError, match="no pending question"):
            owner._pending_decision_question(replay, definition)
    next(iter(saved.values()))["outputs"]["checks"] = []
    with pytest.raises(ValueError, match="checks differ"):
        owner._resolve_rna_decision(
            None, request, definition, evidence, answers, **kwargs
        )


@pytest.mark.parametrize(
    "failure",
    [
        "twoOwners",
        "ruleAnswer",
        "agentAnswer",
        "notMapping",
        "wrongDecision",
        "unoffered",
        "wrongIdentity",
    ],
)
def test_ambiguous_or_unbound_decision_requests_fail_before_persistence(
    decision_case: Any, request: pytest.FixtureRequest, failure: str
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    owner, request, definition, evidence, selection = decision_case
    answer = {
        "decisionId": "qcGrouping",
        "optionId": "qcGrouping:global",
        "rationale": selection.rationale,
    }
    answers, kwargs = {}, {}
    if failure == "twoOwners":
        kwargs = {"rule_selection": selection, "agent_selection": selection}
        reason = "two supplied owners"
    elif failure in {"ruleAnswer", "agentAnswer"}:
        kwargs = {f"{failure[:-6]}_selection": selection}
        answers["decision:qcGrouping"] = answer
        reason = "cannot accept a human answer"
    elif failure == "notMapping":
        answers["decision:qcGrouping"] = "global"
        reason = "must be a mapping"
    elif failure == "wrongDecision":
        answers["decision:qcGrouping"] = {**answer, "decisionId": "other"}
        reason = "must name this decisionId"
    elif failure == "unoffered":
        answers["decision:qcGrouping"] = {**answer, "optionId": "global"}
        reason = "offered option"
    else:
        evidence = evidence.model_copy(update={"decisionId": "other"})
        reason = "exact evidence identities differ"
    with pytest.raises(ValueError, match=reason):
        owner._resolve_rna_decision(
            None, request, definition, evidence, answers, **kwargs
        )
    assert not saved


def test_programmatic_record_creation_requires_digest_and_exact_option(
    decision_case: Any,
) -> None:
    _, _, definition, evidence, selection = decision_case
    with pytest.raises(ValueError, match="content digest"):
        decisions._record_from_selection(
            definition, evidence, selection, "rule", None, None, 0
        )
    selection = selection.model_copy(update={"selectedOptionId": "global"})
    with pytest.raises(ValueError, match="not offered"):
        decisions._record_from_selection(
            definition, evidence.with_content_sha256(), selection, "rule", None, None, 0
        )


def test_qc_model_and_checkpoint_receive_exact_policy_evidence(
    decision_case: Any, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.agent.config import AgentRunConfig

    saved = request.getfixturevalue("memory_checkpoints")
    owner, request, definition, evidence, selection = decision_case
    owner.model = object()
    request.config = SimpleNamespace(agentRunConfig=AgentRunConfig())
    request.request = SimpleNamespace(
        studyContext="Observed captures", studyObjective="Preserve the rare joint group"
    )
    policy_evidence = {
        "policies": [
            {
                "evidenceId": evidence.evidence[0].evidenceId,
                "resolvedBounds": {"mitochondrial": {"upper": 7.125}},
                "retainedCellsByCombination": {"condition/sex": {"treated/F": 12}},
            }
        ]
    }
    calls = []

    def provider(**kwargs: Any) -> Any:
        payload = json.loads(kwargs["user_prompt"])
        assert payload["qcPolicyEvidence"] == policy_evidence
        assert "exact thresholds" in kwargs["system_prompt"]
        calls.append(payload)
        return SimpleNamespace(
            output=kwargs["output_validator"](selection),
            runInfo=SimpleNamespace(modelName="offline-reviewer"),
        )

    monkeypatch.setattr(decisions, "run_agent_sync", provider)
    result = owner._resolve_rna_decision(
        None, request, definition, evidence, {}, qc_evidence=policy_evidence
    )
    checkpoint = next(iter(saved.values()))
    assert checkpoint["inputs"]["qcPolicyEvidence"] == policy_evidence
    assert result.record.rationale == selection.rationale
    assert (
        owner._resolve_rna_decision(
            None, request, definition, evidence, {}, qc_evidence=policy_evidence
        )
        == result
    )
    assert len(calls) == 1
