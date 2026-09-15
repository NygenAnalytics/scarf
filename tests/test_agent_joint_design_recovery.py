"""Saved descriptive joint measurements answer design questions, not effects."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from scarf.agent.experimental_context.comparisons import compare_covariates
from scarf.agent.experimental_context.contracts import CovariateProposal
from scarf.agent.experimental_context.requirements import (
    objective_evidence,
    unmet_objective_requirements,
)
from tests.test_agent_objective_requirements import _repeated_design

CONTEXT = "The observations contain repeated donors and incomplete pairing."
REQUEST = "Assess individual and combined covariates, replication, and confounding"


def _joint_result(*, essential=False):
    result = _repeated_design()
    observations = pd.DataFrame(
        {
            "sample": ["a0", "b0", "a1", "b2", "a3", "b3"],
            "donor": ["d0", "d0", "d1", "d2", "d3", "d3"],
            "tissue": ["A", "B", "A", "B", "A", "B"],
            "condition": ["yes", "yes", "no", "no", "yes", "yes"],
            "sex": ["female", "female", "female", "male", "female", "female"],
        }
    )
    cells = observations.loc[observations.index.repeat(3)].reset_index(drop=True)
    result.characterization.columns.append(
        {"name": "sex", "kind": "categorical", "domain": "biological"}
    )
    proposal = CovariateProposal(
        response="tissue",
        explanatoryColumns=["condition"],
        conditionedOn="sex",
        observationUnit="sample",
        independentUnit="donor",
        rationale="Assess joint tissue and condition support within sex groups.",
        purpose="designCoverage",
        essential=essential,
        objectiveQuote=CONTEXT,
    )
    result.characterization.comparisons = [
        compare_covariates(
            SimpleNamespace(
                columns=list(cells.columns), fetch=lambda name: cells[name].to_numpy()
            ),
            result.characterization,
            proposal,
            selection_identity={"cells": "fixed"},
        )
    ]
    return result


def _evidence(result, request=REQUEST):
    return objective_evidence(
        study_context=f"{CONTEXT} {request}",
        study_objective="Describe supported populations",
        experimental_result=result,
    )


@pytest.mark.parametrize("essential", [False, True])
def test_joint_counts_recover_descriptive_requirement_without_erasing_limitations(
    essential,
):
    result = _joint_result(essential=essential)
    comparison = result.characterization.comparisons[0]
    original = comparison.model_dump(mode="json")
    assert comparison.status == "unsupported"
    assert comparison.evidence["descriptiveDesign"]["observationUnits"] == 6
    assert comparison.evidence["descriptiveDesign"]["independentUnits"] == 4
    assert comparison.evidence["missingCells"] == 0
    assert (
        comparison.evidence["descriptiveDesign"]["pairedCoverage"]["tissue"][
            "completePairs"
        ]
        == 2
    )

    requirements, coverage = _evidence(result)

    assert len(requirements) == 2
    assert requirements[1].essential
    assert requirements[1].kind == "designCoverage"
    assert coverage[1].status == "computed"
    assert coverage[1].evidenceIds == [comparison.evidenceId]
    assert "withinIndependentUnitComparisonsAreUnsupported" in coverage[1].reasons
    assert any(
        "association method remains unsupported" in x for x in coverage[1].reasons
    )
    assert not unmet_objective_requirements(requirements, coverage)
    assert comparison.model_dump(mode="json") == original


@pytest.mark.parametrize(
    "damage",
    [
        "missingTable",
        "emptyTable",
        "notComputed",
        "missingColumn",
        "missingGroup",
        "invalidUnitCount",
        "fractionalUnitCount",
        "changedKind",
        "changedDomain",
        "missingKinds",
    ],
)
def test_joint_descriptive_recovery_requires_exact_completed_measurements(damage):
    result = _joint_result()
    comparison = result.characterization.comparisons[0]
    table = comparison.evidence["descriptiveDesign"]
    if damage == "missingTable":
        table.pop("jointGroupSupport")
    elif damage == "emptyTable":
        table["jointGroupSupport"] = []
    elif damage == "notComputed":
        table["status"] = "unsupported"
    elif damage == "missingColumn":
        for row in table["jointGroupSupport"]:
            row["groups"].pop("sex")
    elif damage == "missingGroup":
        table["jointGroupSupport"].pop()
    elif damage == "invalidUnitCount":
        table["jointGroupSupport"][0]["independentUnits"] = 100
    elif damage == "fractionalUnitCount":
        table["jointGroupSupport"][0]["independentUnits"] = 1.5
    elif damage in {"changedKind", "changedDomain"}:
        field, value = (
            ("kind", "continuous") if damage == "changedKind" else ("domain", "ignore")
        )
        result.characterization.columns[-1][field] = value
    else:
        comparison.evidence.pop("columnKinds")

    requirements, coverage = _evidence(result)

    pending = [
        r for r in requirements if r.requirementId.startswith("requestedDesign:")
    ]
    assert len(pending) == 1
    assert pending[0].essential
    assert unmet_objective_requirements(requirements, coverage)


def test_marginal_counts_cannot_replace_a_joint_design():
    requirements, coverage = _evidence(_repeated_design())
    assert any(r.requirementId.startswith("requestedDesign:") for r in requirements)
    assert unmet_objective_requirements(requirements, coverage)


@pytest.mark.parametrize(
    "question, purpose",
    [
        (
            "Assess joint associations of tissue and condition within sex",
            "designCoverage",
        ),
        ("Assess joint associations of tissue and condition within sex", "association"),
        (
            "Estimate combined effects of tissue and condition within sex",
            "designCoverage",
        ),
        (
            "Estimate combined effects of tissue and condition within sex",
            "effectEstimation",
        ),
    ],
)
def test_descriptive_recovery_does_not_answer_an_association_or_effect(
    question, purpose
):
    result = _joint_result(essential=True)
    result.characterization.comparisons[0].proposal.purpose = purpose
    requirements, coverage = _evidence(result, question)
    assert unmet_objective_requirements(requirements, coverage)


def test_joint_descriptive_recovery_does_not_ignore_a_requested_column():
    result = _joint_result()
    result.characterization.columns.append(
        {"name": "age", "kind": "continuous", "domain": "biological"}
    )
    requirements, coverage = _evidence(
        result, "Assess tissue and age jointly within sex groups"
    )
    assert any(r.requirementId.startswith("requestedDesign:") for r in requirements)
    assert unmet_objective_requirements(requirements, coverage)


def test_reusing_joint_measurements_does_not_duplicate_bounded_requirements():
    result = _joint_result()
    comparison = result.characterization.comparisons[0]
    result.characterization.comparisons = []
    quotes = [f"Optional observation {i}" for i in range(12)]
    for index, quote in enumerate(quotes):
        item = deepcopy(comparison)
        item.proposal.objectiveQuote = quote
        item.evidenceId = f"measured:{index}"
        result.characterization.comparisons.append(item)
    requirements, coverage = _evidence(result, ". ".join([*quotes, REQUEST]))
    assert len(requirements) == len(coverage) == 13
    assert all(r.essential for r in requirements)
    assert not unmet_objective_requirements(requirements, coverage)
