"""Context evidence reads fit the bounded run without expanding scientific work."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent.config import AgentRunConfig
from scarf.agent.experimental_context import ExperimentalContextAgent, tools
from scarf.agent.experimental_context.contracts import (
    CovariateCharacterization,
    CovariateEvidence,
    CovariateProposal,
    ExperimentalContextDependencies,
)
from tests.test_agent_experimental_context import _Store, _design_decision


@pytest.mark.parametrize("tool_limit", [None, 6])
def test_context_can_read_required_saved_details_with_bounded_usage(
    monkeypatch, tool_limit
):
    store = _Store()
    decision = _design_decision()
    calls = []
    snapshots = {}
    attempts = []
    characterize = tools.characterize_covariates

    def measured(*args, **kwargs):
        calls.append(kwargs["directions"])
        return characterize(*args, **kwargs)

    monkeypatch.setattr(tools, "characterize_covariates", measured)
    actions = [
        ("inspect_cell_covariates", {}),
        (
            "analyze_experimental_design",
            {
                "column_domains": decision.columnDomains,
                "coefficients_of_interest": decision.coefficientsOfInterest,
                "units_of_inference": {
                    name: unit.model_dump()
                    for name, unit in decision.unitsOfInference.items()
                },
                "batch_columns": decision.batchCorrection.batchColumns,
            },
        ),
        # Reproduce the notebook's malformed lookup and bounded tool repair.
        (
            "inspect_context_evidence",
            {"section": "confounding", "record_id": "confounding:disease:batch"},
        ),
        (
            "inspect_context_evidence",
            {"section": "confounding", "record_id": "0"},
        ),
        (
            "inspect_context_evidence",
            {"section": "coefficient", "record_id": "disease"},
        ),
        *[
            ("inspect_context_evidence", {"section": "column", "record_id": name})
            for name in ("batch", "sample", "donor")
        ],
    ]
    requests = 0
    measured_before_details = None

    async def reply(_messages, info):
        nonlocal requests, measured_before_details
        index = requests
        requests += 1
        if index == 2:
            measured_before_details = len(calls)
            assert measured_before_details > 0
        if index >= 2:
            assert len(calls) == measured_before_details
        if index < len(actions):
            name, args = actions[index]
        else:
            name, args = info.output_tools[0].name, decision.model_dump()
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])

    def checkpoint_write(key, value):
        snapshots[key] = deepcopy(value)

    agent = ExperimentalContextAgent(
        FunctionModel(reply),
        config=AgentRunConfig(toolCallLimit=tool_limit) if tool_limit else None,
    )
    kwargs = {
        "study_context": "Case-control study with samples nested in donors.",
        "cell_selection": store.cell_selection,
        "checkpoint_write": checkpoint_write,
        "on_attempt": attempts.append,
    }
    if tool_limit is not None:
        with pytest.raises(UsageLimitExceeded, match="tool_calls_limit of 6"):
            agent.run(store, **kwargs)
        assert attempts[-1].status == "failed"
        assert "result" not in snapshots
    else:
        result = agent.run(store, **kwargs)
        assert result.status == "done"
        assert result.decision.batchCorrection.action == "unsafe"
        assert len(result.runInfo.toolCalls) == 8
        assert len(result.runInfo.validationRetries) == 1
        assert result.runInfo.usage.requests == 9
        assert len(calls) == measured_before_details
    assert requests <= 10
    assert snapshots["design1"]["state"]["designRounds"] == 1
    assert "design2" not in snapshots

    # More room to inspect evidence does not permit a fifth follow-up proposal.
    deps = ExperimentalContextDependencies.model_validate(snapshots["design1"]["state"])
    saved = deps.model_dump(mode="json")
    with pytest.raises(ModelRetry, match="eight initial and four follow-up"):
        asyncio.run(
            tools.analyze_experimental_design(
                SimpleNamespace(deps=deps),
                column_domains=decision.columnDomains,
                coefficients_of_interest=decision.coefficientsOfInterest,
                units_of_inference=decision.unitsOfInference,
                batch_columns=decision.batchCorrection.batchColumns,
                proposals=[
                    CovariateProposal(
                        response="disease",
                        explanatoryColumns=["batch"],
                        observationUnit="sample",
                        rationale="Check batch confounding across observation units.",
                    )
                ]
                * 5,
            )
        )
    assert deps.model_dump(mode="json") == saved


def test_context_summary_offers_exact_confounding_lookup_without_mutation():
    characterization = CovariateCharacterization(
        status="done",
        confounding=[
            {"coefficient": "disease", "reason": "Disease aliases the batch."},
            {"coefficient": "sex", "reason": "Sex is incompletely crossed."},
        ],
    )
    evidence = CovariateEvidence(characterization=characterization)
    original = evidence.model_dump(mode="json")
    view = tools.compact_context_evidence(evidence)
    ctx = SimpleNamespace(
        deps=ExperimentalContextDependencies(characterization=characterization)
    )
    for index, row in enumerate(view["characterization"]["confounding"]):
        assert row["details"] == {"section": "confounding", "record_id": str(index)}
        detail = asyncio.run(tools.inspect_context_evidence(ctx, **row["details"]))
        assert detail == characterization.confounding[index]
        detail["reason"] = "Returned copies cannot rewrite committed evidence."
    with pytest.raises(ModelRetry) as exc_info:
        asyncio.run(
            tools.inspect_context_evidence(
                ctx, "confounding", "confounding:disease:batch"
            )
        )
    assert "record_id" in str(exc_info.value)
    assert '["0", "1"]' in str(exc_info.value)
    assert evidence.model_dump(mode="json") == original
