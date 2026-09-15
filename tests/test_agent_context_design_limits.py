"""Repeated-unit aggregation and sparse design limits remain explicit evidence."""

import numpy as np
import pandas as pd
import pytest

from scarf.agent.experimental_context.comparisons import (
    compare_covariates,
    combination_labels,
)
from scarf.agent.experimental_context.contracts import (
    CovariateCharacterization,
    CovariateProposal,
)
from tests.test_agent_design_comparisons import _Cells


def _comparison(frame, *, conditioned=False, joint=False):
    kinds = {
        name: "continuous"
        if name == "response" and frame[name].dtype.kind == "f"
        else "categorical"
        for name in frame.columns
    }
    known = CovariateCharacterization(
        status="done",
        columns=[
            {
                "name": name,
                "kind": kinds[name],
                "domain": "design" if name in {"sample", "donor"} else "biological",
            }
            for name in frame.columns
        ],
        coefficients=[
            {
                "name": "response",
                "observationUnit": "sample",
                "independentUnit": "donor",
            }
        ],
    )
    proposal = CovariateProposal(
        response="response",
        explanatoryColumns=["treatment", "stratum"] if joint else ["treatment"],
        conditionedOn="stratum" if conditioned else None,
        observationUnit="sample",
        independentUnit="donor",
        rationale="Check the supported comparison across independent donors.",
    )
    return compare_covariates(
        _Cells(frame), known, proposal, selection_identity={"cells": "frozen"}
    )


def test_continuous_response_uses_observation_medians_then_independent_donors():
    frame = pd.DataFrame(
        [
            {
                "sample": f"s{i}-{j}",
                "donor": f"d{i}",
                "treatment": str(i % 2),
                "response": float(i * 10 + j + c),
            }
            for i in range(8)
            for j in range(2)
            for c in range(3)
        ]
    )
    result = _comparison(frame)
    assert result.status == "computed"
    assert result.evidence["responseAggregation"] == "medianPerObservationUnit"
    assert result.evidence["independentAggregation"] == "medianOfObservationMedians"
    assert result.evidence["observationUnits"] == 16
    assert result.evidence["independentUnits"] == 8
    assert (
        result.evidence["descriptiveDesign"]["continuousSummaries"]["response"][
            "minimum"
        ]
        == 1.0
    )


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("missing", "noCompleteObservations"),
        ("cellIdentity", "observationUnitIsCellIdentifier"),
        (
            "inconsistentObservation",
            "explanatoryColumnsMustBeConstantWithinObservationUnit",
        ),
        ("fewDonors", "fewerThanFourIndependentUnits"),
    ],
)
def test_unsupported_designs_cannot_be_relabelled_negative_associations(damage, reason):
    frame = pd.DataFrame(
        [
            {
                "sample": f"s{i}",
                "donor": f"d{i}",
                "treatment": str(i % 2),
                "response": float(i + c),
            }
            for i in range(8)
            for c in range(3)
        ]
    )
    if damage == "missing":
        frame["response"] = np.nan
    elif damage == "cellIdentity":
        frame["sample"] = [str(i) for i in range(len(frame))]
    elif damage == "inconsistentObservation":
        frame.loc[0, "treatment"] = "different"
    else:
        frame = frame.iloc[:9]
    result = _comparison(frame)
    assert result.status == "unsupported"
    assert reason in result.reasons
    assert "singleAssociations" not in result.evidence


@pytest.mark.parametrize(
    "case,reason",
    [
        ("levels", "moreThanThirtyTwoCategoricalLevels"),
        ("strata", "moreThanSixteenConditioningStrata"),
        ("sparseStrata", "unsupportedConditioningStrata"),
        ("joint", "moreThanThirtyTwoJointGroups"),
    ],
)
def test_large_or_sparse_combinations_preserve_the_unsupported_reason(case, reason):
    n = (
        68
        if case == "strata"
        else 49
        if case == "joint"
        else 34
        if case == "levels"
        else 8
    )
    rows = [
        {
            "sample": f"s{i}",
            "donor": f"d{i}",
            "treatment": str(
                i if case == "levels" else i % 7 if case == "joint" else i % 2
            ),
            "stratum": str(
                i // 7 if case == "joint" else i // 4 if case == "strata" else i // 2
            ),
            "response": str(i % 2),
        }
        for i in range(n)
    ]
    frame = pd.DataFrame(rows).loc[np.repeat(np.arange(n), 3)].reset_index(drop=True)
    result = _comparison(
        frame, conditioned=case in {"strata", "sparseStrata"}, joint=case == "joint"
    )
    assert result.status == "unsupported"
    assert reason in result.reasons
    assert result.evidence["independentUnits"] == n
    if case == "sparseStrata":
        assert all(
            row["association"]["reason"] == "fewerThanFourIndependentUnits"
            for row in result.evidence["strata"]
        )


def test_combination_identity_accepts_utf8_and_rejects_misaligned_columns():
    cells = _Cells(pd.DataFrame({"one": [b"alpha", b"beta"], "two": ["yes", "no"]}))
    labels = combination_labels(cells, ["one", "two"])
    assert '"alpha"' in labels[0]
    with pytest.raises(ValueError, match="two distinct"):
        combination_labels(cells, ["one", "one"])
    cells.fetch = lambda name: np.asarray(["a", "b"] if name == "one" else ["c"])
    with pytest.raises(ValueError, match="align"):
        combination_labels(cells, ["one", "two"])
