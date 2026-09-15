"""The documentation's offline provider follows the actual comparison protocol."""

import ast
import json
import re
from pathlib import Path

from pydantic_ai import Agent

from scarf.agent.experimental_context.contracts import ExperimentalContextDecision
from scarf.agent.orchestrator.rna_tuning import _assessment_output_type
from scarf.agent.parameter_tuning.comparisons import validate_comparison_review
from tests.agent_comparison_examples import comparison_review


def test_teaching_provider_nominates_combines_and_assesses_actual_evidence() -> None:
    source = (
        Path(__file__).parents[1] / "docs/source/tutorials/agent_workflow.md"
    ).read_text()
    hidden = re.search(
        r"```\{code-cell\} ipython3\n:tags: \[remove-cell\]\n(.*?)\n```", source, re.S
    )
    assert hidden is not None
    parsed = ast.parse(hidden[1])
    definitions = ast.Module(
        body=[
            node
            for node in parsed.body
            if isinstance(
                node,
                ast.Import | ast.ImportFrom | ast.FunctionDef | ast.AsyncFunctionDef,
            )
        ],
        type_ignores=[],
    )
    namespace = {}
    # Only imports and definitions run; no notebook setup, dataset, or analysis cells.
    exec(compile(definitions, "agent_workflow teaching fixture", "exec"), namespace)
    model, state = namespace["_scripted_workflow_model"]()
    review = comparison_review()
    review["comparisonCoverage"]["candidateSettings"]["resolution-half"][
        "metrics"
    ].update(seedStability=0.91, markerCoherence=0.99)
    for item in review["comparisonCoverage"]["candidateSettings"].values():
        item["metrics"]["topMarkerGenes"] = {"0": ["MS4A1"], "1": []}
    evidence = {
        "currentCandidateId": "baseline",
        "candidates": [
            {**item, "status": "done", "eligible": True}
            for item in review["candidates"]
        ],
        "comparisonCoverage": review["comparisonCoverage"],
        "imageHashes": {},
        "featureEvidence": {
            "baseline": {
                "families": {"hla": {"selectedGenes": 6, "selectedExamples": ["HLA-A"]}}
            }
        },
        "experiments": {
            "excludeFamily:hla": {
                "parameter": "excludeFamily",
                "value": "hla",
                "affectedEligibleGenes": 12,
            }
        },
    }
    identities = tuple(item["candidateId"] for item in evidence["candidates"])
    policy = next(
        row
        for row in evidence["comparisonCoverage"]["comparisons"]
        if row["axis"] == "featurePolicy"
    )
    for phase, expected in (
        ("sensitivity", "experiment"),
        ("sensitivity", "combine"),
        ("validation", "accept"),
    ):
        evidence["comparisonCoverage"]["phase"] = phase
        policy["status"] = "pending" if expected == "experiment" else "notApplicable"
        output_type = _assessment_output_type(
            identities, ("excludeFamily:hla",), scope="full", phase=phase
        )
        result = (
            Agent(model, output_type=output_type).run_sync(json.dumps(evidence)).output
        )
        assert result.action == expected
        validate_comparison_review(
            evidence["comparisonCoverage"], result.model_dump(mode="json")
        )
        assert len(result.comparisonConclusions) == 6
        assert any(row.tradeoffs for row in result.comparisonConclusions)
        if expected == "accept":
            assert (
                result.selectedCandidateId
                in evidence["comparisonCoverage"]["resolutionCandidateIds"]
            )
            assert result.populationConcerns[0].clusterId == "1"
        elif expected == "combine":
            assert result.combinedSettings is not None
        else:
            assert result.experimentId == "excludeFamily:hla"
            assert "6 selected genes" in result.concern
    assert state["requests"] == 3
    assert len(state["assessments"][-1]["alternatives"]) == 4

    async def inspect_cell_covariates() -> dict:
        return {"characterization": {}, "qcProfiles": []}

    async def analyze_experimental_design(
        column_domains: dict,
        coefficients_of_interest: list,
        units_of_inference: dict,
        batch_columns: list,
    ) -> dict:
        return {
            "qcProfiles": [{"action": "skip", "evidenceId": "qc:retention"}],
            "captureDesignSafety": {},
            "requestedComparisons": [],
        }

    context_model, _ = namespace["_scripted_workflow_model"]()
    context = Agent(
        context_model,
        output_type=ExperimentalContextDecision,
        tools=[inspect_cell_covariates, analyze_experimental_design],
    ).run_sync("Assess the available context")
    assert context.output.batchCorrection.action == "skip"
    assert context.output.evidenceIds == ["qc:retention"]
