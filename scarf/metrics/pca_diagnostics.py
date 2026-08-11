"""Pure helpers for PCA branch diagnostics."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .association import association_pair

__all__ = [
    "branch_nuisance_summary",
    "family_loading_concentration",
    "per_pc_covariate_associations",
]

ColumnKind = Literal["categorical", "continuous"]
_TOP_LOADINGS = 8
_ASSOCIATION_FLOOR = 0.1


def per_pc_covariate_associations(
    coordinates: Any,
    covariates: Mapping[str, Any],
    *,
    columnKinds: Mapping[str, ColumnKind],
    associationFloor: float = _ASSOCIATION_FLOOR,
) -> list[dict[str, Any]]:
    """Associate every PC with typed covariates.

    Returns one record per PC and covariate. Records below
    ``associationFloor`` keep their values but set ``flagged`` to False.
    """
    shape = getattr(coordinates, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    n_cells, n_dims = map(int, shape)
    reports: list[dict[str, Any]] = []
    for pc_index in range(n_dims):
        pc_values = np.asarray(coordinates[:, pc_index])
        for name, values in covariates.items():
            kind = columnKinds.get(name)
            if kind is None:
                raise KeyError(f"columnKinds missing entry for {name!r}")
            values_array = np.asarray(values)
            if values_array.shape != (n_cells,):
                raise ValueError(
                    f"covariate {name!r} must have length {n_cells}, "
                    f"got {values_array.shape}"
                )
            association = association_pair(
                pc_values,
                values_array,
                leftKind="continuous",
                rightKind=kind,
            )
            strength = _association_strength(association)
            flagged = strength is not None and strength >= associationFloor
            reports.append(
                {
                    "pc": int(pc_index + 1),
                    "covariate": name,
                    "kind": kind,
                    "association": association,
                    "strength": strength,
                    "flagged": bool(flagged),
                }
            )
    return reports


def _association_strength(association: Mapping[str, Any]) -> float | None:
    """Return unsigned association strength for branch comparison.

    Spearman is signed; eta-squared and Cramér's V are already non-negative.
    """
    if association.get("status") != "ok":
        return None
    value = association.get("value")
    if value is None:
        return None
    score = float(value)
    if association.get("measure") == "spearmanRho":
        return abs(score)
    return score


def family_loading_concentration(
    loadings: np.ndarray,
    *,
    featureIndexes: Sequence[int],
    familyIndexes: Mapping[str, Sequence[int]],
    topN: int = _TOP_LOADINGS,
    featureNames: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Report family concentration and top loadings for every PC."""
    matrix = np.asarray(loadings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("loadings must be a two-dimensional array")
    n_features, n_dims = matrix.shape
    feature_indexes = np.asarray(list(featureIndexes), dtype=np.int64)
    if feature_indexes.shape != (n_features,):
        raise ValueError("featureIndexes length must match the loadings feature axis")
    names: Sequence[str] | None = None
    if featureNames is not None:
        names = list(featureNames)
        if len(names) != n_features:
            raise ValueError("featureNames length must match loadings rows")

    position = {int(index): offset for offset, index in enumerate(feature_indexes)}
    family_positions: dict[str, np.ndarray] = {}
    for family, indexes in familyIndexes.items():
        matched = [position[int(index)] for index in indexes if int(index) in position]
        family_positions[family] = np.asarray(matched, dtype=np.int64)

    reports: list[dict[str, Any]] = []
    for pc_index in range(n_dims):
        weights = np.abs(matrix[:, pc_index])
        total = float(weights.sum())
        order = np.argsort(weights)[::-1]
        top_limit = max(0, min(int(topN), n_features))
        top_rows = order[:top_limit]
        top_loadings: list[dict[str, Any]] = []
        for row in top_rows:
            entry: dict[str, Any] = {
                "featureIndex": int(feature_indexes[row]),
                "loading": float(matrix[row, pc_index]),
                "absLoading": float(weights[row]),
            }
            if names is not None:
                entry["featureName"] = str(names[row])
            top_loadings.append(entry)
        family_shares: dict[str, float] = {}
        for family, family_rows in family_positions.items():
            if total <= 0 or family_rows.size == 0:
                family_shares[family] = 0.0
            else:
                family_shares[family] = float(weights[family_rows].sum() / total)
        reports.append(
            {
                "pc": int(pc_index + 1),
                "familyShares": family_shares,
                "topLoadings": top_loadings,
            }
        )
    return reports


def branch_nuisance_summary(
    associations: Sequence[Mapping[str, Any]],
    *,
    technicalCovariates: Sequence[str],
    nuisanceCovariates: Sequence[str] = (),
    protectedCovariates: Sequence[str] = (),
    associationFloor: float = _ASSOCIATION_FLOOR,
) -> dict[str, Any]:
    """Aggregate per-PC associations into branch-level nuisance summaries.

    Does not align or compare individual PC indexes across branches.
    """
    technical = set(technicalCovariates)
    nuisance = set(nuisanceCovariates)
    protected = set(protectedCovariates)

    def _collect(names: set[str]) -> dict[str, Any]:
        values: list[float] = []
        flagged_pcs: set[int] = set()
        by_covariate: dict[str, list[float]] = {name: [] for name in names}
        for record in associations:
            covariate = str(record["covariate"])
            if covariate not in names:
                continue
            strength = record.get("strength")
            if strength is None:
                association = record.get("association") or {}
                strength = _association_strength(association)
            if strength is None:
                continue
            score = float(strength)
            values.append(score)
            by_covariate[covariate].append(score)
            if score >= associationFloor:
                flagged_pcs.add(int(record["pc"]))
        mean_value = float(np.mean(values)) if values else 0.0
        max_value = float(np.max(values)) if values else 0.0
        return {
            "nAssociations": len(values),
            "meanAssociation": mean_value,
            "maxAssociation": max_value,
            "nFlaggedPcs": len(flagged_pcs),
            "flaggedPcs": sorted(flagged_pcs),
            "byCovariate": {
                name: {
                    "nAssociations": len(scores),
                    "meanAssociation": (float(np.mean(scores)) if scores else 0.0),
                    "maxAssociation": (float(np.max(scores)) if scores else 0.0),
                }
                for name, scores in by_covariate.items()
            },
        }

    return {
        "technical": _collect(technical),
        "nuisance": _collect(nuisance),
        "protected": _collect(protected),
        "associationFloor": float(associationFloor),
    }
