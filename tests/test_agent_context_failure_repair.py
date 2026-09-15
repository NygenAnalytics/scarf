"""Unmeasured study questions are repaired before the design allowance closes."""

import asyncio

import pytest
from pydantic_ai import ModelRetry

from scarf.agent.experimental_context import tools, validation
from tests.test_agent_experimental_context import _Store, _context, _design_decision


def test_pending_joint_question_requests_followup_then_remains_unresolved(monkeypatch):
    context = _context(_Store())
    context.deps.studyContext = "Assess individual and combined covariates."
    context.deps.studyObjective = "Describe disease populations."
    decision = _design_decision()
    asyncio.run(tools.inspect_cell_covariates(context))
    arguments = {
        "column_domains": decision.columnDomains,
        "coefficients_of_interest": decision.coefficientsOfInterest,
        "units_of_inference": decision.unitsOfInference,
        "batch_columns": decision.batchCorrection.batchColumns,
    }
    asyncio.run(tools.analyze_experimental_design(context, **arguments))
    monkeypatch.setattr(
        tools,
        "characterize_covariates",
        lambda *a, **k: pytest.fail(
            "Output repair must reuse measured characterization"
        ),
    )
    with pytest.raises(ModelRetry, match="remaining analyze_experimental_design round"):
        validation.validate_experimental_context(decision, context.deps)
    assert context.deps.designRounds == 1
    # Repeating marginal evidence cannot answer the pending combined question.
    asyncio.run(tools.analyze_experimental_design(context, **arguments))
    result = validation.validate_experimental_context(decision, context.deps)
    assert context.deps.designRounds == 2
    assert any("combined covariates" in question for question in result.needsInput)
    assert result.batchCorrection.action == "unsafe"
