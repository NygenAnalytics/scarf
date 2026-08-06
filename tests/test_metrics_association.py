"""Tests for association and estimability helpers."""

import numpy as np
import pandas as pd
import pytest

from scarf.metrics.association import (
    association_pair,
    coefficient_estimability,
    cramers_v,
    directional_mapping,
    eta_squared,
    report_confounding,
    report_technical_nesting,
    spearman_rho,
)


def _categorical_table(counts: dict[tuple[str, str], int]) -> tuple[np.ndarray, ...]:
    left: list[str] = []
    right: list[str] = []
    for (level_left, level_right), count in counts.items():
        left.extend([level_left] * count)
        right.extend([level_right] * count)
    return np.array(left), np.array(right)


def test_cramers_v_exact_mapping_is_one() -> None:
    left = np.array(["a", "a", "b", "b", "c", "c"])
    right = np.array(["x", "x", "y", "y", "z", "z"])
    result = cramers_v(left, right)
    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(1.0)
    assert result["directionalMapping"]["nesting"] == "equivalent"


def test_cramers_v_matches_hand_computed_chi_square() -> None:
    # 2x2 table [[30, 20], [20, 30]] has chi-square 4 at n=100, so V = 0.2.
    left, right = _categorical_table(
        {("a", "x"): 30, ("a", "y"): 20, ("b", "x"): 20, ("b", "y"): 30}
    )
    result = cramers_v(left, right)
    assert result["valueUncorrected"] == pytest.approx(0.2)
    # The bias correction shrinks the estimate toward zero.
    assert result["value"] < result["valueUncorrected"]


def test_cramers_v_is_degenerate_when_levels_reach_rows() -> None:
    # Disease is perfectly determined by donor, but with one donor per row the
    # bias correction has no residual degrees of freedom left to divide by.
    disease = np.array(["case"] * 3 + ["ctrl"] * 3)
    donor = np.array([f"d{index}" for index in range(6)])
    result = cramers_v(disease, donor)
    assert result["status"] == "notComputed"
    assert result["reason"] == "degenerateCorrection"
    assert result["valueUncorrected"] == pytest.approx(1.0)
    assert result["directionalMapping"]["nesting"] == "rightInLeft"


def test_cramers_v_constant_is_not_computed() -> None:
    left = np.array(["a", "a", "a", "a"])
    right = np.array(["x", "y", "x", "y"])
    result = cramers_v(left, right)
    assert result["status"] == "notComputed"
    assert result["reason"] == "constantOrSingleLevel"


def test_directional_mapping_left_in_right() -> None:
    left = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
    right = np.array(["b1", "b1", "b1", "b1", "b2", "b2"])
    mapping = directional_mapping(left, right)
    assert mapping["leftMapsToRight"] is True
    assert mapping["rightMapsToLeft"] is False
    assert mapping["nesting"] == "leftInRight"


def test_directional_mapping_ignores_constant_column() -> None:
    # Everything is trivially nested inside a constant, which is not a finding.
    mapping = directional_mapping(
        np.array(["a", "a", "a", "a"]),
        np.array(["x", "y", "x", "y"]),
    )
    assert mapping["nesting"] == "none"
    assert mapping["reason"] == "constantColumn"


def test_directional_mapping_excludes_missing_rows() -> None:
    left = np.array(["s1", "s1", "s2", "s2", "s3", None], dtype=object)
    right = np.array(["b1", "b1", "b1", "b1", "b2", "b2"], dtype=object)
    mapping = directional_mapping(left, right)
    assert mapping["rowsUsed"] == 5
    assert mapping["rowsMissing"] == 1
    assert mapping["nesting"] == "leftInRight"


def test_eta_squared_matches_hand_computed_sums_of_squares() -> None:
    # Grand mean 4, SS total 20, SS between 16.
    result = eta_squared(
        np.array([1.0, 3.0, 5.0, 7.0]),
        np.array(["a", "a", "b", "b"]),
    )
    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(0.8)
    assert result["saturated"] is False


def test_eta_squared_flags_singleton_groups() -> None:
    # One observation per level forces the ratio to 1 with no real effect.
    result = eta_squared(np.array([1.0, 2.0, 3.0]), np.array(["a", "b", "c"]))
    assert result["value"] == pytest.approx(1.0)
    assert result["saturated"] is True


def test_eta_squared_rejects_non_numeric_values() -> None:
    result = eta_squared(np.array(["low", "high", "low"]), np.array(["a", "b", "a"]))
    assert result["status"] == "notComputed"
    assert result["reason"] == "nonNumeric"


def test_spearman_rho_perfect_monotone() -> None:
    left = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    right = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = spearman_rho(left, right)
    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(1.0)
    assert result["tiedLeft"] is False


def test_spearman_rho_reports_ties_and_missing() -> None:
    left = np.array([1.0, 2.0, 2.0, 3.0, np.nan])
    right = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = spearman_rho(left, right)
    assert result["rowsUsed"] == 4
    assert result["rowsMissing"] == 1
    assert result["tiedLeft"] is True
    assert result["tiedRight"] is False


def test_spearman_rho_rejects_non_numeric_values() -> None:
    result = spearman_rho(np.array(["a", "b", "c"]), np.array([1.0, 2.0, 3.0]))
    assert result["status"] == "notComputed"
    assert result["reason"] == "nonNumeric"


def test_association_pair_dispatches_kinds() -> None:
    categorical = np.array(["a", "a", "b", "b"])
    continuous = np.array([1.0, 1.2, 8.0, 8.5])
    assert (
        association_pair(
            continuous,
            categorical,
            leftKind="continuous",
            rightKind="categorical",
        )["measure"]
        == "etaSquared"
    )
    assert (
        association_pair(
            categorical,
            continuous,
            leftKind="categorical",
            rightKind="continuous",
        )["measure"]
        == "etaSquared"
    )
    assert (
        association_pair(
            continuous,
            continuous,
            leftKind="continuous",
            rightKind="continuous",
        )["measure"]
        == "spearmanRho"
    )


def test_report_technical_nesting_finds_nested_batch() -> None:
    columns = {
        "sample": np.array(["s1", "s1", "s2", "s2", "s3", "s3"]),
        "batch": np.array(["b1", "b1", "b1", "b1", "b2", "b2"]),
        "chemistry": np.array(["v3"] * 6),
    }
    reports = report_technical_nesting(columns)
    assert len(reports) == 1
    assert (reports[0]["left"], reports[0]["right"]) == ("sample", "batch")
    assert reports[0]["nesting"] == "leftInRight"


def test_coefficient_estimability_detects_alias() -> None:
    # Coefficient equals technical batch: not estimable after technicals.
    batch = np.array(["b1", "b1", "b2", "b2", "b3", "b3"])
    disease = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])
    result = coefficient_estimability(
        disease,
        coefficientKind="categorical",
        technicals={"batch": batch},
        technicalKinds={"batch": "categorical"},
    )
    assert result["status"] == "ok"
    assert result["coefficientEstimable"] is False
    assert result["rankDeficient"] is True


def test_coefficient_estimability_detects_complementary_alias() -> None:
    # Level order is reversed between the two factors. Without an intercept in
    # the model matrix the indicators look independent and this alias is missed.
    batch = np.array(["b1", "b1", "b2", "b2"])
    disease = np.array(["ctrl", "ctrl", "case", "case"])
    result = coefficient_estimability(
        disease,
        coefficientKind="categorical",
        technicals={"batch": batch},
        technicalKinds={"batch": "categorical"},
    )
    assert result["coefficientEstimable"] is False
    assert result["rankTechnical"] == result["rankWithCoefficient"] == 2


def test_coefficient_estimability_keeps_crossed_factor() -> None:
    batch = np.array(["b1", "b1", "b2", "b2"])
    disease = np.array(["case", "ctrl", "case", "ctrl"])
    result = coefficient_estimability(
        disease,
        coefficientKind="categorical",
        technicals={"batch": batch},
        technicalKinds={"batch": "categorical"},
    )
    assert result["coefficientEstimable"] is True
    assert result["rankWithCoefficient"] == result["rankTechnical"] + 1


def test_coefficient_estimability_detects_continuous_alias() -> None:
    # A continuous coefficient that is a per-batch constant lies in the span of
    # the intercept plus the batch indicator.
    batch = np.array(["b1", "b1", "b2", "b2"])
    dose = np.array([1.0, 1.0, 2.0, 2.0])
    result = coefficient_estimability(
        dose,
        coefficientKind="continuous",
        technicals={"batch": batch},
        technicalKinds={"batch": "categorical"},
    )
    assert result["coefficientEstimable"] is False


def test_coefficient_estimability_without_technicals() -> None:
    result = coefficient_estimability(
        np.array(["case", "case", "ctrl", "ctrl"]),
        coefficientKind="categorical",
        technicals={},
        technicalKinds={},
    )
    assert result["coefficientEstimable"] is True


def test_coefficient_estimability_gates_when_columns_reach_rows() -> None:
    # Three technical factors with many levels relative to six rows. Rank loss
    # still proves the coefficient is not estimable, so the result is reported
    # instead of deferred as notComputed.
    rows = 6
    technicals = {
        "t1": np.array([f"a{i}" for i in range(rows)]),
        "t2": np.array([f"b{i}" for i in range(rows)]),
        "t3": np.array([f"c{i}" for i in range(rows)]),
    }
    result = coefficient_estimability(
        np.array(["x", "x", "y", "y", "z", "z"]),
        coefficientKind="categorical",
        technicals=technicals,
        technicalKinds={name: "categorical" for name in technicals},
    )
    assert result["status"] == "ok"
    assert result["coefficientEstimable"] is False
    assert result["rankDeficient"] is True
    assert result["encodedColumns"] >= result["rowsUsed"]


def test_report_confounding_on_design_table() -> None:
    design = pd.DataFrame(
        {
            "disease": ["case", "case", "ctrl", "ctrl"],
            "batch": ["b1", "b1", "b2", "b2"],
            "chemistry": ["v2", "v3", "v2", "v3"],
        }
    )
    report = report_confounding(
        design,
        coefficient="disease",
        technicalColumns=["batch", "chemistry"],
        columnKinds={
            "disease": "categorical",
            "batch": "categorical",
            "chemistry": "categorical",
        },
        associationFloor=0.1,
    )
    assert report["coefficient"] == "disease"
    assert report["nRows"] == 4
    selected = {pair["technical"]: pair["selected"] for pair in report["pairs"]}
    assert selected == {"batch": True, "chemistry": False}
    assert report["estimability"]["coefficientEstimable"] is False


def test_report_confounding_selects_deterministic_pair_without_effect_size() -> None:
    # The corrected effect size is unavailable here, so selection has to fall
    # back on the exact mapping. Estimability still reports non-estimable even
    # though the encoded design saturates the row count.
    design = pd.DataFrame(
        {
            "disease": ["case"] * 3 + ["ctrl"] * 3,
            "donor": [f"d{index}" for index in range(6)],
        }
    )
    report = report_confounding(
        design,
        coefficient="disease",
        technicalColumns=["donor"],
        columnKinds={"disease": "categorical", "donor": "categorical"},
    )
    pair = report["pairs"][0]
    assert pair["association"]["status"] == "notComputed"
    assert pair["selected"] is True
    assert report["estimability"]["status"] == "ok"
    assert report["estimability"]["coefficientEstimable"] is False
    assert report["estimability"]["rankDeficient"] is True


def test_coefficient_estimability_partially_estimable_categorical() -> None:
    # Three disease levels, one contrast recoverable against batch, one aliased.
    batch = np.array(["y", "y", "x", "x", "y", "y", "y", "y"])
    disease = np.array(["A", "A", "B", "B", "C", "C", "C", "A"])
    result = coefficient_estimability(
        disease,
        coefficientKind="categorical",
        technicals={"batch": batch},
        technicalKinds={"batch": "categorical"},
    )
    assert result["status"] == "ok"
    assert result["coefficientEstimable"] is False
    assert result["partiallyEstimable"] is True
    assert result["coefficientDf"] == 2
    assert result["estimableDf"] == 1


def test_report_confounding_estimability_uses_all_technicals() -> None:
    # Neither technical passes the pairwise threshold, but together they exactly
    # span the coefficient. Estimability must therefore use both unselected terms.
    design = pd.DataFrame(
        {
            "batch": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "site": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    design["response"] = design["batch"] + design["site"]
    report = report_confounding(
        design,
        coefficient="response",
        technicalColumns=["batch", "site"],
        columnKinds={
            "response": "continuous",
            "batch": "continuous",
            "site": "continuous",
        },
        associationFloor=0.99,
    )
    assert all(pair["selected"] is False for pair in report["pairs"])
    assert report["estimability"]["status"] == "ok"
    assert report["estimability"]["coefficientEstimable"] is False
    assert report["estimability"]["rankDeficient"] is True


def test_coefficient_estimability_saturated_but_full_is_not_computed() -> None:
    # Fully recovered coefficient with no residual df reports notComputed.
    result = coefficient_estimability(
        np.array(["a", "b"]),
        coefficientKind="categorical",
        technicals={},
        technicalKinds={},
    )
    assert result["status"] == "notComputed"
    assert result["reason"] == "encodedColumnsReachRows"
    assert result["estimableDf"] == result["coefficientDf"]
    assert result["residualDf"] == 0


def test_report_confounding_requires_present_columns() -> None:
    design = pd.DataFrame({"disease": ["case", "ctrl"]})
    with pytest.raises(KeyError):
        report_confounding(
            design,
            coefficient="missing",
            technicalColumns=[],
            columnKinds={"missing": "categorical"},
        )
    with pytest.raises(KeyError):
        report_confounding(
            design,
            coefficient="disease",
            technicalColumns=["batch"],
            columnKinds={"disease": "categorical", "batch": "categorical"},
        )
