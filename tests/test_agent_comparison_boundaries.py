"""Corrupt, incomplete and mismatched measurements cannot authorize acceptance."""

from copy import deepcopy
from typing import Any

import pytest

from scarf.agent.orchestrator.rna_tuning import validate_completed_comparison_evidence
from scarf.agent.parameter_tuning.comparisons import validate_comparison_review
from tests.agent_comparison_examples import comparison_review


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("phase", "unmeasured", "Unknown RNA comparison phase"),
        ("population", "unknown", "population must identify"),
        ("candidateSettings", [], "exact settings and rows"),
        ("comparisons", {}, "exact settings and rows"),
        ("resolutionCandidateIds", [], "four resolutions"),
    ],
)
def test_review_rejects_incomplete_coverage(
    field: str, value: Any, reason: str
) -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    coverage[field] = value
    with pytest.raises(ValueError, match=reason):
        validate_comparison_review(coverage, review)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("duplicate", "IDs must be unique"),
        ("missingBaseline", "baseline is not completed"),
        ("unfinishedBaseline", "baseline is not completed"),
        ("missingAlternative", "lacks a completed alternative"),
        ("unfinishedAlternative", "complete on the exact same cells"),
        ("differentCells", "complete on the exact same cells"),
        ("missingReason", "observed eligibility reason"),
        ("inventedAlternative", "cannot claim a different"),
        ("unsupportedEquivalence", "valid observed equivalence"),
        ("missingConclusion", "explicit comparison conclusions"),
        ("unaddressedAlternative", "all its observed alternatives"),
    ],
)
def test_each_comparison_requires_complete_attributable_evidence(
    failure: str, reason: str
) -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    rows, settings = coverage["comparisons"], coverage["candidateSettings"]
    if failure == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif failure == "missingBaseline":
        rows[0]["baselineCandidateId"] = "absent"
    elif failure == "unfinishedBaseline":
        settings["baseline"]["status"] = "failed"
    elif failure == "missingAlternative":
        rows[0]["alternativeCandidateId"] = "absent"
    elif failure == "unfinishedAlternative":
        settings["resolution-half"]["status"] = "failed"
    elif failure == "differentCells":
        settings["resolution-half"]["cellSelection"]["artifactId"] = "d" * 64
    elif failure == "missingReason":
        rows[-1]["reason"] = " "
    elif failure == "inventedAlternative":
        rows[-1]["alternativeCandidateId"] = "genes-two"
    elif failure == "unsupportedEquivalence":
        rows[-1]["observedProof"]["meaningfulPermittedInterventions"] = 1
    elif failure == "missingConclusion":
        review.pop("comparisonConclusions")
    elif failure == "unaddressedAlternative":
        review["comparisonConclusions"][0]["candidateIds"] = ["baseline"]
    with pytest.raises(ValueError, match=reason):
        validate_comparison_review(coverage, review)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("absent", "candidate is unavailable"),
        ("failed", "completed on the same cells"),
        ("cells", "completed on the same cells"),
        ("representation", "changed the combined representation"),
        ("resolution", "coverage is incomplete"),
        ("missingSelected", "lack exact completed evidence"),
        ("missingClusterMarkers", "every selected cluster"),
        ("inventedConcern", "must cite supplied candidate evidence"),
        ("unvalidatedRepair", "one targeted full repair"),
    ],
)
def test_final_panel_requires_exact_selected_combination(
    failure: str, reason: str
) -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    settings = coverage["candidateSettings"]
    replacement = deepcopy(settings["resolution-half"])
    settings["final-half"] = replacement
    coverage["resolutionCandidateIds"][0] = "final-half"
    if failure == "absent":
        del settings["final-half"]
    elif failure == "failed":
        replacement["status"] = "failed"
    elif failure == "cells":
        replacement["cellSelection"]["artifactId"] = "a" * 64
    elif failure == "representation":
        replacement["parameters"]["dimensions"] = 30
    elif failure == "resolution":
        replacement["parameters"]["leidenResolution"] = 0.9
    elif failure == "missingSelected":
        review["selectedCandidateId"] = "missing"
    elif failure == "missingClusterMarkers":
        settings["candidate-two"]["metrics"]["nClusters"] = 3
    elif failure == "inventedConcern":
        review["populationConcerns"] = [
            {
                "candidateId": "candidate-two",
                "clusterId": "0",
                "status": "nonEssentialLimitation",
                "evidenceIds": ["invented-marker-proof"],
                "explanation": "A proposed interpretation needs observed support.",
            }
        ]
    elif failure == "unvalidatedRepair":
        repaired = deepcopy(settings["baseline"])
        repaired["parameters"]["dimensions"] = 30
        repaired["parameters"]["neighborsK"] = 41
        settings["unvalidated"] = repaired
        review["selectedCandidateId"] = "unvalidated"
    with pytest.raises(ValueError, match=reason):
        validate_comparison_review(coverage, review)


@pytest.mark.parametrize("failure", ["coverage", "candidate", "measurement"])
def test_report_cannot_detach_accepted_review_from_saved_measurements(
    failure: str,
) -> None:
    review = comparison_review()
    if failure == "coverage":
        review["comparisonCoverage"] = None
        reason = "mandatory comparison evidence"
    elif failure == "candidate":
        review["candidates"][0]["cellSelection"] = None
        reason = "exact reviewed candidate"
    else:
        review["candidates"][0] = deepcopy(review["candidates"][0])
        review["candidates"][0]["metrics"]["seedStability"] = 1.0
        reason = "measurements differ"
    with pytest.raises(ValueError, match=reason):
        validate_completed_comparison_evidence(review)


def test_duplicate_or_fabricated_tradeoffs_cannot_justify_preference() -> None:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    coverage["candidateSettings"]["genes-two"]["metrics"]["seedStability"] = 0.99
    conclusion = next(
        c for c in review["comparisonConclusions"] if c["axis"] == "hvgCount"
    )
    tradeoff = {
        "alternativeCandidateId": "genes-two",
        "metric": "seedStability",
        "preferredValue": 0.92,
        "alternativeValue": 0.99,
        "interpretation": "The measured stability advantage is weighed against marker preservation.",
    }
    conclusion["tradeoffs"] = [tradeoff, deepcopy(tradeoff)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_comparison_review(coverage, review)
    conclusion["tradeoffs"] = [{**tradeoff, "alternativeValue": 1.0}]
    with pytest.raises(ValueError, match="exact preferred and alternative"):
        validate_comparison_review(coverage, review)
