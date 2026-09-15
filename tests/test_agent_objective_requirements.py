"""Measured design coverage is required before an objective can be completed."""

from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from scarf.agent.experimental_context.comparisons import compare_covariates
from scarf.agent.experimental_context.contracts import (
    CovariateCharacterization,
    CovariateProposal,
    ExperimentalContextDecision,
    InferenceUnit,
)
from scarf.agent.experimental_context.study import (
    StudyContract,
    build_study_contract,
    validate_objective_evidence,
)


def _repeated_design(*, purpose: str = "designCoverage", essential: bool = True):
    observations = pd.DataFrame(
        {
            "sample": ["a0", "b0", "a1", "b2", "a3", "b3"],
            "donor": ["d0", "d0", "d1", "d2", "d3", "d3"],
            "tissue": ["A", "B", "A", "B", "A", "B"],
            "condition": ["yes", "yes", "no", "no", "yes", "yes"],
        }
    )
    cells = observations.loc[observations.index.repeat(3)].reset_index(drop=True)
    characterization = CovariateCharacterization(
        status="done",
        columns=[
            {
                "name": name,
                "kind": "categorical",
                "domain": "design" if name == "sample" else "biological",
            }
            for name in cells.columns
        ],
        coefficients=[
            {
                "name": "tissue",
                "scope": "betweenUnit",
                "kind": "categorical",
                "observationUnit": "sample",
                "independentUnit": "donor",
                "groupOrder": ["A", "B"],
                "unitLevelCounts": {
                    "observationUnit": {"levels": 6},
                    "independentUnit": {"levels": 4},
                },
                "replication": {"sufficient": True, "minimumPerGroup": 3},
                "pairedCoverage": {
                    "design": "mixedOrIncomplete",
                    "completePairs": 2,
                    "incompletePairs": 2,
                },
            }
        ],
    )
    proposal = CovariateProposal.model_validate(
        {
            "response": "tissue",
            "explanatoryColumns": ["condition"],
            "observationUnit": "sample",
            "independentUnit": "donor",
            "rationale": "Assess observed tissue and condition support across donors.",
            "purpose": purpose,
            "essential": essential,
            "objectiveQuote": "Describe supported populations",
        }
    )
    comparison = compare_covariates(
        SimpleNamespace(
            columns=list(cells.columns), fetch=lambda name: cells[name].to_numpy()
        ),
        characterization,
        proposal,
        selection_identity={"cells": "fixed"},
    )
    characterization.comparisons = [comparison]
    result = SimpleNamespace(
        status="done",
        characterization=characterization,
        batchSafety=[],
        notes=[],
        decision=ExperimentalContextDecision(
            coefficientsOfInterest=["tissue"],
            unitsOfInference={
                "tissue": InferenceUnit(
                    observationUnit="sample", independentUnit="donor"
                )
            },
        ),
    )
    return result


def _contract(result):
    return build_study_contract(
        study_context="The observations contain repeated donors and incomplete pairing.",
        study_objective="Describe supported populations",
        experimental_result=result,
    )


def test_repeated_donors_have_descriptive_coverage_without_an_association():
    result = _repeated_design()
    comparison = result.characterization.comparisons[0]
    assert comparison.status == "unsupported"
    assert comparison.reasons == ["withinIndependentUnitComparisonsAreUnsupported"]
    evidence = comparison.evidence["descriptiveDesign"]
    assert evidence["observationUnits"] == 6
    assert evidence["independentUnits"] == 4
    assert evidence["groupSupport"]["tissue"] == [
        {"group": "A", "observationUnits": 3, "independentUnits": 3},
        {"group": "B", "observationUnits": 3, "independentUnits": 3},
    ]
    assert evidence["pairedCoverage"]["tissue"]["completePairs"] == 2
    assert evidence["pairedCoverage"]["tissue"]["incompletePairs"] == 2
    assert evidence["sharedIndependentUnits"]["tissue"]["pairs"] == [
        {"groups": ["A", "B"], "independentUnits": 2},
    ]
    assert "singleAssociations" not in comparison.evidence
    contract = _contract(result)
    validate_objective_evidence(contract, result)
    assert [item.status for item in contract.evidenceCoverage] == [
        "computed",
        "computed",
    ]
    assert any(
        "association method remains unsupported" in text
        for text in contract.limitations
    )


@pytest.mark.parametrize("purpose", ["association", "effectEstimation"])
def test_descriptive_counts_cannot_satisfy_an_essential_effect_or_association(purpose):
    result = _repeated_design(purpose=purpose)
    contract = _contract(result)
    assert contract.evidenceCoverage[1].status == "unsupported"
    with pytest.raises(ValueError, match="Essential objective evidence is unresolved"):
        validate_objective_evidence(contract, result)


def test_optional_unsupported_question_remains_an_explicit_limitation():
    result = _repeated_design(purpose="association", essential=False)
    result.characterization.comparisons[
        0
    ].proposal.objectiveQuote = (
        "The observations contain repeated donors and incomplete pairing."
    )
    contract = _contract(result)
    validate_objective_evidence(contract, result)
    assert contract.evidenceCoverage[1].status == "unsupported"
    assert any(
        "no supported association or absence finding" in text
        for text in contract.limitations
    )


def test_missing_replication_blocks_completion_even_without_proposals():
    result = _repeated_design()
    result.characterization.comparisons = []
    result.characterization.coefficients[0]["replication"] = {}
    contract = _contract(result)
    with pytest.raises(ValueError, match="replication evidence is unavailable"):
        validate_objective_evidence(contract, result)


def test_nonidentifiable_exact_batch_design_answers_design_question_only():
    result = _repeated_design()
    result.batchSafety = [
        SimpleNamespace(
            coefficient="tissue",
            batchColumns=["batch"],
            status="unsafe",
            evidenceId="batchSafety:tissue:batch",
        )
    ]
    result.characterization.columns.append(
        {"name": "batch", "domain": "technical", "kind": "categorical"}
    )
    contract = _contract(result)
    assert contract.evidenceCoverage[0].status == "nonIdentifiable"
    validate_objective_evidence(contract, result)
    assert contract.correctionLicense == "unsafeConfounded"


def test_recomputed_context_coverage_rejects_tampered_or_stale_answer():
    result = _repeated_design()
    contract = _contract(result)
    result.characterization.coefficients[0]["replication"] = {}
    with pytest.raises(ValueError, match="differs from its measured context report"):
        validate_objective_evidence(contract, result)


@pytest.mark.parametrize("field", ["evidenceRequirements", "evidenceCoverage"])
def test_old_contracts_without_objective_evidence_are_incompatible(field):
    values = _contract(_repeated_design()).model_dump(mode="json")
    del values[field]
    with pytest.raises(ValidationError, match=field):
        StudyContract.model_validate(values)


def test_empty_requirements_and_fabricated_citations_cannot_bypass_gate():
    contract = _contract(_repeated_design())
    with pytest.raises(ValueError):
        validate_objective_evidence(
            contract.model_copy(update={"evidenceRequirements": []})
        )
    contract.evidenceCoverage[0].evidenceIds.append("unmeasured:claim")
    with pytest.raises(ValueError, match="outside the study contract"):
        validate_objective_evidence(contract)


def test_context_role_change_invalidates_previously_computed_design_question():
    result = _repeated_design()
    result.characterization.columns[1]["domain"] = "technical"
    contract = _contract(result)
    assert contract.evidenceCoverage[1].status == "unsupported"
    with pytest.raises(ValueError, match="different column roles"):
        validate_objective_evidence(contract, result)
