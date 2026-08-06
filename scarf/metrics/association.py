"""Association and estimability helpers for covariate characterization."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "association_pair",
    "coefficient_estimability",
    "cramers_v",
    "directional_mapping",
    "eta_squared",
    "report_confounding",
    "report_technical_nesting",
    "spearman_rho",
]

ColumnKind = Literal["categorical", "continuous"]


def _as_1d(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return array


def _to_float(values: np.ndarray) -> np.ndarray | None:
    """Return values as float, or None when the column is not numeric."""
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None


def _pairwise_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) != len(right):
        raise ValueError("paired arrays must have the same length")
    left_ok = np.asarray(~pd.isna(left), dtype=bool)
    right_ok = np.asarray(~pd.isna(right), dtype=bool)
    return cast(np.ndarray, left_ok & right_ok)


def _not_computed(
    reason: str,
    *,
    rowsUsed: int = 0,
    rowsMissing: int = 0,
) -> dict[str, Any]:
    return {
        "status": "notComputed",
        "reason": reason,
        "rowsUsed": int(rowsUsed),
        "rowsMissing": int(rowsMissing),
    }


def directional_mapping(
    left: Any,
    right: Any,
) -> dict[str, Any]:
    """Report exact directional mapping between two categorical columns."""
    left_values = _as_1d(left)
    right_values = _as_1d(right)
    mask = _pairwise_mask(left_values, right_values)
    rows_used = int(mask.sum())
    rows_missing = int((~mask).sum())
    blank = {
        "leftMapsToRight": False,
        "rightMapsToLeft": False,
        "nesting": "none",
        "rowsUsed": rows_used,
        "rowsMissing": rows_missing,
    }
    if rows_used == 0:
        return blank
    frame = pd.DataFrame({"left": left_values[mask], "right": right_values[mask]})
    n_left = int(frame["left"].nunique(dropna=False))
    n_right = int(frame["right"].nunique(dropna=False))
    # A constant column is nested inside everything, which is true but useless.
    if n_left < 2 or n_right < 2:
        return {**blank, "reason": "constantColumn"}
    left_maps = bool(frame.groupby("left", dropna=False)["right"].nunique().le(1).all())
    right_maps = bool(
        frame.groupby("right", dropna=False)["left"].nunique().le(1).all()
    )
    if left_maps and right_maps:
        nesting = "equivalent"
    elif left_maps:
        nesting = "leftInRight"
    elif right_maps:
        nesting = "rightInLeft"
    else:
        nesting = "none"
    return {
        "leftMapsToRight": left_maps,
        "rightMapsToLeft": right_maps,
        "nesting": nesting,
        "rowsUsed": rows_used,
        "rowsMissing": rows_missing,
    }


def cramers_v(left: Any, right: Any) -> dict[str, Any]:
    """Bias-corrected Cramér's V for two categorical columns.

    ``value`` uses the Bergsma correction, which subtracts the expected
    chi-square under independence. On the small design tables this module
    targets, that correction can shrink even a perfect association to zero, so
    ``valueUncorrected`` and ``directionalMapping`` are reported alongside it.
    """
    left_values = _as_1d(left)
    right_values = _as_1d(right)
    mask = _pairwise_mask(left_values, right_values)
    rows_used = int(mask.sum())
    rows_missing = int((~mask).sum())
    if rows_used < 2:
        return _not_computed(
            "insufficientRows",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    contingency = pd.crosstab(left_values[mask], right_values[mask])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return _not_computed(
            "constantOrSingleLevel",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    chi2 = float(stats.chi2_contingency(contingency.to_numpy(), correction=False)[0])
    n = float(rows_used)
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denominator = min(r_corr - 1.0, k_corr - 1.0)
    mapping = directional_mapping(left_values[mask], right_values[mask])
    common = {
        "rowsUsed": rows_used,
        "rowsMissing": rows_missing,
        "nLevelsLeft": int(r),
        "nLevelsRight": int(k),
        "valueUncorrected": float(np.sqrt(min(phi2 / (min(r, k) - 1), 1.0))),
        "directionalMapping": mapping,
    }
    # One level per row leaves the correction no residual degrees of freedom.
    if denominator <= 0:
        return {"status": "notComputed", "reason": "degenerateCorrection", **common}
    return {
        "status": "ok",
        "measure": "cramersV",
        "value": float(np.sqrt(phi2_corr / denominator)),
        **common,
    }


def eta_squared(continuous: Any, categorical: Any) -> dict[str, Any]:
    """Correlation ratio η² for continuous values grouped by a categorical column."""
    raw = _as_1d(continuous)
    groups = _as_1d(categorical)
    values = _to_float(raw)
    if values is None:
        return _not_computed("nonNumeric", rowsMissing=len(raw))
    mask = _pairwise_mask(values, groups) & np.isfinite(values)
    rows_used = int(mask.sum())
    rows_missing = int(len(values) - rows_used)
    if rows_used < 2:
        return _not_computed(
            "insufficientRows",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    frame = pd.DataFrame({"y": values[mask], "g": groups[mask]})
    n_levels = int(frame["g"].nunique(dropna=False))
    if n_levels < 2:
        return _not_computed(
            "constantOrSingleLevel",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    grand_mean = float(frame["y"].mean())
    ss_total = float(((frame["y"] - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return _not_computed(
            "zeroVariance",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    group_means = frame.groupby("g", dropna=False)["y"].transform("mean")
    ss_between = float(((group_means - grand_mean) ** 2).sum())
    return {
        "status": "ok",
        "measure": "etaSquared",
        "value": float(ss_between / ss_total),
        "rowsUsed": rows_used,
        "rowsMissing": rows_missing,
        "nLevels": n_levels,
        # One observation per level forces η² to 1 regardless of any real effect.
        "saturated": bool(n_levels >= rows_used),
    }


def spearman_rho(left: Any, right: Any) -> dict[str, Any]:
    """Spearman correlation for two continuous columns."""
    raw_left = _as_1d(left)
    raw_right = _as_1d(right)
    left_values = _to_float(raw_left)
    right_values = _to_float(raw_right)
    if left_values is None or right_values is None:
        return _not_computed("nonNumeric", rowsMissing=len(raw_left))
    mask = (
        _pairwise_mask(left_values, right_values)
        & np.isfinite(left_values)
        & np.isfinite(right_values)
    )
    rows_used = int(mask.sum())
    rows_missing = int(len(left_values) - rows_used)
    if rows_used < 3:
        return _not_computed(
            "insufficientRows",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    x = left_values[mask]
    y = right_values[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return _not_computed(
            "zeroVariance",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    rho = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(rho):
        return _not_computed(
            "undefined",
            rowsUsed=rows_used,
            rowsMissing=rows_missing,
        )
    return {
        "status": "ok",
        "measure": "spearmanRho",
        "value": rho,
        "rowsUsed": rows_used,
        "rowsMissing": rows_missing,
        "tiedLeft": bool(np.unique(x).size < rows_used),
        "tiedRight": bool(np.unique(y).size < rows_used),
    }


def association_pair(
    left: Any,
    right: Any,
    *,
    leftKind: ColumnKind,
    rightKind: ColumnKind,
) -> dict[str, Any]:
    """Dispatch the MVP association measure for a typed column pair."""
    if leftKind == "categorical" and rightKind == "categorical":
        return cramers_v(left, right)
    if leftKind == "continuous" and rightKind == "categorical":
        return eta_squared(left, right)
    if leftKind == "categorical" and rightKind == "continuous":
        return eta_squared(right, left)
    return spearman_rho(left, right)


def report_technical_nesting(
    columns: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Directional nesting among categorical technical columns."""
    names = list(columns)
    reports: list[dict[str, Any]] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            mapping = directional_mapping(columns[left_name], columns[right_name])
            if mapping["nesting"] == "none":
                continue
            reports.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "nesting": mapping["nesting"],
                    "directionalMapping": mapping,
                }
            )
    return reports


def _one_hot(values: np.ndarray) -> np.ndarray:
    """Treatment-coded indicators with the first level dropped."""
    dummies = pd.get_dummies(pd.Series(values), dummy_na=False, dtype=float)
    if dummies.shape[1] > 1:
        dummies = dummies.iloc[:, 1:]
    elif dummies.shape[1] == 1:
        return np.zeros((len(values), 0), dtype=float)
    return cast(np.ndarray, dummies.to_numpy(dtype=float))


def coefficient_estimability(
    coefficient: Any,
    *,
    coefficientKind: ColumnKind,
    technicals: Mapping[str, Any],
    technicalKinds: Mapping[str, ColumnKind],
) -> dict[str, Any]:
    """Minimal model-matrix rank check for a coefficient among technical columns.

    The matrix carries an explicit intercept. Without it, treatment-coded
    indicators of two perfectly complementary factors look linearly
    independent and a fully aliased coefficient is reported as estimable.
    """
    coeff = _as_1d(coefficient)
    n = len(coeff)
    if n == 0:
        return _not_computed("emptyInput")

    mask = np.asarray(~pd.isna(coeff), dtype=bool)
    coeff_float: np.ndarray | None = None
    if coefficientKind == "continuous":
        coeff_float = _to_float(coeff)
        if coeff_float is None:
            return _not_computed("nonNumeric", rowsMissing=n)
        mask &= np.isfinite(coeff_float)
    for values in technicals.values():
        mask &= np.asarray(~pd.isna(_as_1d(values)), dtype=bool)
    rows_used = int(mask.sum())
    if rows_used < 2:
        return _not_computed(
            "insufficientRows",
            rowsUsed=rows_used,
            rowsMissing=n - rows_used,
        )

    blocks: list[np.ndarray] = [np.ones((rows_used, 1), dtype=float)]
    for name, values in technicals.items():
        subset = _as_1d(values)[mask]
        if technicalKinds[name] == "continuous":
            column = _to_float(subset)
            if column is None or float(np.nanstd(column)) == 0.0:
                continue
            blocks.append(column.reshape(-1, 1))
            continue
        indicators = _one_hot(subset)
        if indicators.shape[1]:
            blocks.append(indicators)
    technical_matrix = np.concatenate(blocks, axis=1)

    if coefficientKind == "continuous":
        assert coeff_float is not None
        coeff_block = coeff_float[mask].reshape(-1, 1)
    else:
        coeff_block = _one_hot(coeff[mask])
        if coeff_block.shape[1] == 0:
            return _not_computed("constantCoefficient", rowsUsed=rows_used)

    encoded = int(technical_matrix.shape[1] + coeff_block.shape[1])
    coefficient_df = int(coeff_block.shape[1])
    rank_technical = int(np.linalg.matrix_rank(technical_matrix))
    rank_full = int(
        np.linalg.matrix_rank(np.concatenate([technical_matrix, coeff_block], axis=1))
    )
    estimable_df = int(rank_full - rank_technical)
    residual_df = int(rows_used - rank_full)

    if coefficientKind == "continuous":
        fully_estimable = estimable_df == 1
        partially_estimable = False
        required_df = 1
    else:
        fully_estimable = estimable_df == coefficient_df
        partially_estimable = 0 < estimable_df < coefficient_df
        required_df = coefficient_df

    # A saturated design with a fully recovered coefficient cannot be audited
    # for residual variation. Rank loss still reports absent/partial estimability.
    if encoded >= rows_used and fully_estimable and residual_df <= 0:
        return {
            "status": "notComputed",
            "reason": "encodedColumnsReachRows",
            "rowsUsed": rows_used,
            "encodedColumns": encoded,
            "rankTechnical": rank_technical,
            "rankWithCoefficient": rank_full,
            "coefficientDf": required_df,
            "estimableDf": estimable_df,
            "residualDf": residual_df,
        }

    return {
        "status": "ok",
        "coefficientEstimable": fully_estimable,
        "partiallyEstimable": partially_estimable,
        "rankDeficient": estimable_df < required_df,
        "rankTechnical": rank_technical,
        "rankWithCoefficient": rank_full,
        "coefficientDf": required_df,
        "estimableDf": estimable_df,
        "residualDf": residual_df,
        "rowsUsed": rows_used,
        "encodedColumns": encoded,
    }


def report_confounding(
    design: pd.DataFrame,
    *,
    coefficient: str,
    technicalColumns: Sequence[str],
    columnKinds: Mapping[str, ColumnKind],
    associationFloor: float = 0.1,
) -> dict[str, Any]:
    """Association of one coefficient with technical columns on a design table."""
    if coefficient not in design.columns:
        raise KeyError(f"coefficient column {coefficient!r} missing from design")
    missing = [name for name in technicalColumns if name not in design.columns]
    if missing:
        raise KeyError(f"technical columns missing from design: {missing}")
    coeff_kind = columnKinds[coefficient]
    pairs: list[dict[str, Any]] = []
    for name in technicalColumns:
        result = association_pair(
            design[coefficient].to_numpy(),
            design[name].to_numpy(),
            leftKind=coeff_kind,
            rightKind=columnKinds[name],
        )
        mapping = result.get("directionalMapping") or {}
        # Bias-corrected V collapses toward zero on small design tables, so a
        # deterministic mapping selects the pair even when the effect size does not.
        selected = (
            result.get("status") == "ok"
            and abs(float(result["value"])) >= associationFloor
        ) or mapping.get("nesting", "none") != "none"
        pairs.append({"technical": name, "association": result, "selected": selected})

    return {
        "coefficient": coefficient,
        "nRows": int(len(design)),
        "pairs": pairs,
        "estimability": coefficient_estimability(
            design[coefficient].to_numpy(),
            coefficientKind=coeff_kind,
            technicals={name: design[name].to_numpy() for name in technicalColumns},
            technicalKinds={name: columnKinds[name] for name in technicalColumns},
        ),
    }
