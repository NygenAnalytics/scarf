"""Capture provenance repair reuses measured designs without reopening comparisons."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.tools import ToolDefinition

from scarf.agent.experimental_context import tools
from scarf.agent.experimental_context.contracts import (
    CaptureProposal,
    CellQcProfileEvidence,
    CovariateComparison,
    InferenceUnit,
)
from scarf.agent.types import BatchSafetyEvidence
from tests.test_agent_design_comparisons import _deps, _design, _proposal


def _repair_context(monkeypatch):
    cells, characterization = _design()
    characterization.columns.append(
        {"name": "batch", "kind": "categorical", "domain": "technical"}
    )
    deps = _deps(cells)
    deps.studyContext = "sample identifies the physical capture."
    deps.characterization = characterization
    deps.designRounds = 2
    deps.toolCalls = [
        "inspect_cell_covariates",
        "analyze_experimental_design",
        "analyze_experimental_design",
    ]
    measured = CovariateComparison(
        proposal=_proposal(), status="computed", evidenceId="measured-comparison"
    )
    deps.comparisons = [measured]
    characterization.comparisons = [measured]
    deps.batchSafety = {
        "measured-safety": BatchSafetyEvidence(
            coefficient="response",
            batchColumns=["batch"],
            status="unsafe",
            evidenceId="measured-safety",
        )
    }
    arguments = {
        "column_domains": {"sample": "design", "batch": "technical"},
        "coefficients_of_interest": [],
        "units_of_inference": {},
        "batch_columns": ["batch"],
        "proposals": [],
        "capture_proposal": CaptureProposal(
            column="sample", provenanceQuote=deps.studyContext
        ),
    }
    deps.characterizationInputs = {
        "directions": {
            "columnDomains": dict(arguments["column_domains"]),
            "coefficientsOfInterest": [],
            "unitsOfInference": {},
        },
        "metadata": {"sample": "unchanged"},
    }
    saved = {}

    def write(key, value):
        assert key not in saved, "Committed measurements must remain immutable"
        saved[key] = deepcopy(value)

    deps.checkpointWrite = write
    deps.checkpointRead = saved.get
    tools.persist_context_evidence(deps, "design2")
    qc_calls = []

    def qc(current, design):
        qc_calls.append(current.captureProposal)
        assert design is current.characterization
        profile = CellQcProfileEvidence(
            profileId="capture-supported",
            action="sampleMad",
            sampleColumn="sample",
            attributes=["counts"],
        )
        current.qcProfiles = {profile.profileId: profile}
        return [profile]

    monkeypatch.setattr(tools, "_offered_qc_profiles", qc)
    for name in (
        "characterize_context",
        "evaluate_proposals",
        "_batch_safety_evidence",
    ):
        monkeypatch.setattr(
            tools,
            name,
            lambda *a, **k: pytest.fail("Capture repair must reuse measured design"),
        )
    return SimpleNamespace(deps=deps), arguments, saved, qc_calls


def test_capture_repair_preserves_design_and_immutable_checkpoints(monkeypatch):
    context, arguments, saved, qc_calls = _repair_context(monkeypatch)
    before = deepcopy(saved["design2"])
    result = asyncio.run(tools.analyze_experimental_design(context, **arguments))
    assert context.deps.designRounds == 2
    assert len(context.deps.comparisons) == 1
    assert result.characterization.captureProvenance == arguments["capture_proposal"]
    assert result.batchSafety[0].status == "unsafe"
    assert result.batchSafety[0].batchColumns == ["batch"]
    assert len(qc_calls) == 1
    assert set(saved) == {"design2", "capture"}
    assert saved["design2"] == before
    assert (
        saved["capture"]["characterizationInputs"] == before["characterizationInputs"]
    )
    restored = _deps(context.deps.cells)
    restored.checkpointRead = saved.get
    assert tools.restore_context_evidence(restored)
    assert restored.captureProposal == arguments["capture_proposal"]
    assert restored.designRounds == 2
    assert restored.comparisons == context.deps.comparisons
    assert list(restored.qcProfiles) == ["capture-supported"]
    assert len(qc_calls) == 1
    definition = ToolDefinition(name="analyze_experimental_design")
    assert (
        tools._prepare_experimental_context_tool(
            SimpleNamespace(deps=restored), definition
        )
        is None
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"capture_proposal": None}, "two evidence rounds"),
        ({"proposals": [_proposal()]}, "two evidence rounds"),
        ({"column_domains": {"sample": "technical"}}, "exact saved domains"),
        ({"coefficients_of_interest": ["response"]}, "exact saved domains"),
        (
            {
                "units_of_inference": {
                    "response": InferenceUnit(observationUnit="sample")
                }
            },
            "exact saved domains",
        ),
        ({"batch_columns": []}, "unchanged assessed batch"),
        (
            {
                "capture_proposal": CaptureProposal(
                    column="sample", provenanceQuote="sample is an inferred capture"
                )
            },
            "exact study quote",
        ),
    ],
)
def test_capture_repair_rejects_new_work_or_unverified_provenance(
    monkeypatch, change, reason
):
    context, arguments, saved, qc_calls = _repair_context(monkeypatch)
    arguments.update(change)
    with pytest.raises(ModelRetry, match=reason):
        asyncio.run(tools.analyze_experimental_design(context, **arguments))
    assert context.deps.designRounds == 2
    assert context.deps.captureProposal is None
    assert set(saved) == {"design2"}
    assert not qc_calls


def test_capture_repair_rejects_ambiguous_historical_batch_designs(monkeypatch):
    context, arguments, saved, qc_calls = _repair_context(monkeypatch)
    context.deps.batchSafety["alternative"] = BatchSafetyEvidence(
        coefficient="response", batchColumns=["other"], status="safe"
    )
    with pytest.raises(ModelRetry, match="Alternative designs cannot be combined"):
        asyncio.run(tools.analyze_experimental_design(context, **arguments))
    assert not qc_calls
    assert set(saved) == {"design2"}


def test_failed_capture_qc_refresh_does_not_authorize_incomplete_evidence(monkeypatch):
    context, arguments, saved, _ = _repair_context(monkeypatch)

    def failed(*args):
        raise RuntimeError("Capture quality measurements unavailable")

    monkeypatch.setattr(tools, "_offered_qc_profiles", failed)
    with pytest.raises(RuntimeError, match="quality measurements unavailable"):
        asyncio.run(tools.analyze_experimental_design(context, **arguments))
    assert context.deps.captureProposal is None
    assert context.deps.characterization.captureProvenance is None
    assert context.deps.designRounds == 2
    assert set(saved) == {"design2"}


def test_model_summary_exposes_provenance_repair_after_exhausted_design_rounds(
    monkeypatch,
):
    context, arguments, _, _ = _repair_context(monkeypatch)
    definition = ToolDefinition(name="analyze_experimental_design")
    assert tools._prepare_experimental_context_tool(context, definition) is definition
    from scarf.agent.experimental_context.contracts import CovariateEvidence

    async def measured(ctx):
        return CovariateEvidence(characterization=ctx.deps.characterization)

    before = asyncio.run(tools.model_evidence_tool(measured)(context))
    assert before["captureRepairAvailable"] is True
    assert "proposals=[]" in before["captureRepairInstructions"]
    assert before["captureRepairInputs"] == {
        key: value for key, value in arguments.items() if key != "capture_proposal"
    }
    asyncio.run(tools.analyze_experimental_design(context, **arguments))
    after = asyncio.run(tools.model_evidence_tool(measured)(context))
    assert after["captureRepairAvailable"] is False
    assert "captureRepairInstructions" not in after


@pytest.mark.parametrize(
    "quote,purpose",
    [
        ("Assess combined covariates.", "designCoverage"),
        ("Assess joint association of covariates.", "association"),
        ("Estimate joint effects of covariates.", "effectEstimation"),
    ],
)
def test_requested_question_summary_preserves_its_required_purpose(
    monkeypatch, quote, purpose
):
    from scarf.agent.experimental_context.contracts import CovariateEvidence

    context, _, _, _ = _repair_context(monkeypatch)
    context.deps.studyObjective = quote

    async def measured(ctx):
        return CovariateEvidence(characterization=ctx.deps.characterization)

    payload = asyncio.run(tools.model_evidence_tool(measured)(context))
    question = payload["requestedComparisons"][0]
    assert question["purpose"] == purpose
    assert question["essential"] is True
    assert question["completionEvidence"] == "evidenceRequirements and evidenceCoverage"
