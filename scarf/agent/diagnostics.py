"""Bounded descriptive diagnostics over frozen metadata and public artifacts."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .models import AnalysisInputError


MAX_DIAGNOSTIC_CELLS = 10_000
MAX_DIAGNOSTIC_COLUMNS = 48
MAX_GROUP_LEVELS = 64


def family_audit(names: np.ndarray, blacklist: str) -> dict[str, Any]:
    """Compare the executed name policy with the unchanged core registry."""
    from scarf.features.gene_families import GENE_FAMILY_PATTERNS, gene_family_mask
    from scarf.features.variability import DEFAULT_HVG_BLACKLIST
    from scarf.utils.arrays import regex_match_mask

    standard = regex_match_mask(names, DEFAULT_HVG_BLACKLIST)
    executed = regex_match_mask(names, blacklist)
    families = []
    for family, pattern in GENE_FAMILY_PATTERNS.items():
        matched = gene_family_mask(names, family)
        families.append(
            {
                "family": family,
                "pattern": pattern,
                "matchedFeatures": int(matched.sum()),
                "excludedFeatures": int((matched & executed).sum()),
                "standardExcludedFeatures": int((matched & standard).sum()),
                "matchedNames": [str(name) for name in names[matched][:12]],
                "omittedMatchedNames": max(0, int(matched.sum()) - 12),
            }
        )
    return {
        "standardBlacklist": DEFAULT_HVG_BLACKLIST,
        "executedBlacklist": blacklist,
        "featureCount": len(names),
        "standardExcludedFeatures": int(standard.sum()),
        "excludedFeatures": int(executed.sum()),
        "standardOnlyExcludedFeatures": int((standard & ~executed).sum()),
        "families": families,
        "limitations": [
            "Families are name-pattern matches from the Scarf registry, not complete gene programs or species-aware reference mappings."
        ],
    }


def resolved_roles(
    study: Any,
    suggestions: Mapping[str, str] | None = None,
    evidence_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Separate supplied experimental authority from model role suggestions."""
    supplied: set[tuple[str, str]] = set()
    for column in study.technicalBatchColumns:
        supplied.add((column, "technical"))
    for column in study.protectedColumns:
        supplied.add((column, "protected"))
    if study.sampleColumn:
        supplied.add((study.sampleColumn, "sample"))
    if study.captureColumn:
        supplied.add((study.captureColumn, "capture"))
    roles = supplied | set((suggestions or {}).items())
    return [
        {
            "column": column,
            "role": role,
            "source": "supplied" if (column, role) in supplied else "inferred",
            "authority": "confirmed"
            if (column, role) in supplied
            else "diagnosticOnly",
            "evidenceIds": [f"study:{role}"]
            if (column, role) in supplied
            else list(dict.fromkeys(evidence_ids or [])),
        }
        for column, role in sorted(roles)
    ]


def group_columns(roles: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            row["column"]
            for row in roles
            if row["role"] in {"technical", "protected", "sample", "capture"}
        )
    )


def categorical_summary(series: pd.Series) -> dict[str, Any]:
    """Keep missingness distinct from string-valued labels and bound output."""
    missing = series.isna() | series.astype(str).str.strip().eq("")
    counts = series.loc[~missing].astype(str).value_counts()
    present = int((~missing).sum())
    return {
        "levels": [
            {
                "value": str(value),
                "count": int(count),
                "fraction": float(count / len(series)),
            }
            for value, count in counts.head(MAX_GROUP_LEVELS).items()
        ],
        "missing": int(missing.sum()),
        "unique": len(counts),
        "omittedLevels": max(0, len(counts) - MAX_GROUP_LEVELS),
        "dominantFraction": float(counts.iloc[0] / len(series)) if present else None,
    }


def design_diagnostics(
    metadata: Any, retained: np.ndarray, roles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe crossed and confounded labels without enabling correction."""
    columns = group_columns(roles)
    frames = {
        column: metadata.to_pandas_dataframe([column])[column]
        .loc[retained]
        .reset_index(drop=True)
        for column in columns
    }
    pairs = [
        (left, right)
        for index, left in enumerate(columns)
        for right in columns[index + 1 :]
    ]
    tables = []
    limitations = []
    if len(pairs) > 24:
        limitations.append(f"Design cross-tabs show 24 of {len(pairs)} column pairs.")
    for left, right in pairs[:24]:
        frame = pd.DataFrame({"left": frames[left], "right": frames[right]})
        present = frame.notna().all(axis=1) & frame.astype(str).apply(
            lambda x: x.str.strip().ne("")
        ).all(axis=1)
        values = frame.loc[present].astype(str)
        levels = [values[key].nunique() for key in ("left", "right")]
        row: dict[str, Any] = {
            "leftColumn": left,
            "rightColumn": right,
            "rowsUsed": int(present.sum()),
            "rowsMissing": int((~present).sum()),
            "leftLevels": int(levels[0]),
            "rightLevels": int(levels[1]),
        }
        if max(levels, default=0) > MAX_GROUP_LEVELS:
            row.update(
                {
                    "available": False,
                    "reason": "More than 64 levels; cross-tab omitted.",
                }
            )
        else:
            counts = values.value_counts(sort=False)
            row.update(
                {
                    "available": bool(len(values)),
                    "fullyCrossed": bool(
                        len(values) and len(counts) == levels[0] * levels[1]
                    ),
                    "counts": [
                        {"left": a, "right": b, "count": int(count)}
                        for (a, b), count in counts.items()
                    ],
                }
            )
        tables.append(row)
    return {"crossTabs": tables, "roles": roles, "limitations": limitations}


def diagnostic_indices(size: int, seed: int) -> np.ndarray:
    if size <= MAX_DIAGNOSTIC_CELLS:
        return np.arange(size)
    return np.sort(
        np.random.default_rng(seed).choice(size, MAX_DIAGNOSTIC_CELLS, replace=False)
    )


def _rows_columns(array: Any, rows: np.ndarray, columns: slice) -> np.ndarray:
    """Read bounded coordinates through the public artifact array interface."""
    if isinstance(array, np.ndarray):
        return np.asarray(array[rows, columns], dtype=float)
    return np.asarray(array.get_orthogonal_selection((rows, columns)), dtype=float)


def representation_diagnostics(
    store: Any, run: Any, prepared: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Measure PC associations, selected families and bounded loading summaries."""
    from scarf.features.gene_families import GENE_FAMILY_PATTERNS, gene_family_mask
    from scarf.metrics import eta_squared, spearman_rho

    coordinates = store.load_artifact(run["pca"])["data"]
    rows = diagnostic_indices(coordinates.shape[0], seed)
    dimensions = min(30, coordinates.shape[1])
    values = _rows_columns(coordinates, rows, slice(0, dimensions))
    if not np.isfinite(values).all():
        raise AnalysisInputError("PCA diagnostic coordinates are non-finite")
    associations = []
    limitations = []
    columns = prepared.get("diagnosticColumns", [])[:MAX_DIAGNOSTIC_COLUMNS]
    for column in columns:
        series = (
            run.cells.to_pandas_dataframe([column])[column]
            .iloc[rows]
            .reset_index(drop=True)
        )
        numeric = pd.api.types.is_numeric_dtype(
            series
        ) and not pd.api.types.is_bool_dtype(series)
        present = series.notna() & series.astype(str).str.strip().ne("")
        if numeric:
            present &= np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        observed = series.loc[present]
        if not numeric and observed.nunique() > MAX_GROUP_LEVELS:
            limitations.append(
                f"PC association for {column} omitted: more than 64 categorical levels."
            )
            continue
        measured = (
            series.to_numpy(dtype=float, na_value=np.nan)
            if numeric
            else series.astype(object).where(present, None).to_numpy()
        )
        for index in range(dimensions):
            metric = (
                spearman_rho(values[:, index], measured)
                if numeric
                else eta_squared(values[:, index], measured)
            )
            association = metric.get("value") if metric["status"] == "ok" else None
            status = metric["status"]
            reason = metric.get("reason")
            if metric.get("saturated"):
                association, status, reason = None, "notComputed", "saturatedCategories"
            elif association is not None and not np.isfinite(association):
                association, status, reason = (
                    None,
                    "notComputed",
                    "nonfiniteAssociation",
                )
            associations.append(
                {
                    "column": column,
                    "kind": "spearman" if numeric else "etaSquared",
                    "component": index + 1,
                    "association": association,
                    "rowsUsed": metric["rowsUsed"],
                    "rowsMissing": metric["rowsMissing"],
                    "status": status,
                    "saturated": metric.get("saturated", False),
                    "limitation": reason,
                }
            )
    mask = np.asarray(
        store.load_artifact(run["highly_variable_features"])["values"][:], dtype=bool
    )
    # run.features is aligned to the frozen feature universe, not live feature I.
    feature_names = np.asarray(run.features.fetch("names")).astype(str)
    if mask.shape != feature_names.shape:
        raise AnalysisInputError(
            "HVG selection does not align with the frozen feature axis"
        )
    names = feature_names[mask]
    hvg = family_audit(names, prepared["blacklist"])
    loadings_array = store.load_artifact(run["pca"])["loadings"]
    if loadings_array.shape[0] != len(names):
        raise AnalysisInputError("PCA loadings do not align with the selected features")
    loading_rows = []
    # Only ten loading vectors are read, never a cell-by-gene matrix.
    for index in range(min(10, loadings_array.shape[1])):
        loading = np.asarray(loadings_array[:, index], dtype=float)
        if not np.isfinite(loading).all():
            raise AnalysisInputError("PCA loadings contain non-finite values")
        top = np.argsort(-np.abs(loading), kind="stable")[:20]
        top_names = names[top]
        loading_rows.append(
            {
                "component": index + 1,
                "topGenes": [
                    {"gene": str(names[i]), "loading": float(loading[i])} for i in top
                ],
                "families": {
                    family: int(gene_family_mask(top_names, family).sum())
                    for family in GENE_FAMILY_PATTERNS
                },
            }
        )
    ids = [str(value) for value in run.cells.fetch("ids")[rows]]
    return {
        "covariateAssociations": associations,
        "pcaDiagnosticScope": {
            "method": "sampledPcaAssociations",
            "sampleCells": len(rows),
            "populationCells": coordinates.shape[0],
            "components": dimensions,
            "columns": columns,
            "seed": seed,
            "cellIdsSha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "limitations": limitations,
        },
        "hvgAudit": hvg,
        "loadingFamilies": loading_rows,
        "actualHvgCount": len(names),
        "hvgSelectionDigest": hashlib.sha256(
            np.packbits(mask).tobytes() + str(mask.shape).encode()
        ).hexdigest(),
    }


def finalist_composition(
    run: Any, prepared: dict[str, Any], labels: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Describe all finalist cells by sample/group and existing QC columns."""
    groups = group_columns(prepared.get("resolvedRoles", []))
    qc_columns = [
        column
        for column in prepared.get("diagnosticColumns", [])
        if column in prepared.get("qcFlags", {})
    ]
    frames = {
        column: run.cells.to_pandas_dataframe([column])[column].reset_index(drop=True)
        for column in dict.fromkeys([*groups, *qc_columns])
    }
    if any(len(series) != len(labels) for series in frames.values()):
        raise AnalysisInputError(
            "Frozen diagnostic metadata does not align with finalist cells"
        )
    result = {}
    for label in np.unique(labels):
        selected = labels == label
        qc = {}
        for column in qc_columns:
            values = frames[column].loc[selected].to_numpy(dtype=float, na_value=np.nan)
            valid = values[np.isfinite(values)]
            qc[column] = {
                "median": float(np.median(valid)) if len(valid) else None,
                "q10": float(np.quantile(valid, 0.1)) if len(valid) else None,
                "q90": float(np.quantile(valid, 0.9)) if len(valid) else None,
                "missing": int(len(values) - len(valid)),
            }
        result[str(label)] = {
            "groupComposition": {
                column: categorical_summary(frames[column].loc[selected])
                for column in groups
            },
            "qc": qc,
        }
    return result
