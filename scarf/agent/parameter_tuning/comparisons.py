"""Exact RNA sensitivity comparisons and their review requirements."""

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, get_args

import numpy as np
from pydantic import Field

from ...storage.refs import ArtifactRef
from ..types import AgentDataModel
from .contracts import ParameterCandidateEvaluation

type ComparisonAxis = Literal[
    "hvgCount", "hvgRanking", "featurePolicy", "pca", "neighbors", "partition"
]


class ComparisonConclusion(AgentDataModel):
    """Explain an observed comparison against the scientific objective."""

    axis: ComparisonAxis
    candidateIds: list[str] = Field(min_length=1)
    preferredCandidateId: str
    quantitativeReason: str = Field(min_length=1)
    biologicalReason: str = Field(min_length=1)
    plainLanguageSummary: str = Field(min_length=1)
    tradeoffs: list["ComparisonTradeoff"] = Field(default_factory=list)


class ComparisonTradeoff(AgentDataModel):
    """A measured advantage of an alternative that the preference must explain."""

    alternativeCandidateId: str
    metric: Literal[
        "seedStability",
        "subsampleStability",
        "markerCoherence",
        "markerSpecificityMedian",
        "macroF1",
    ]
    preferredValue: float = Field(allow_inf_nan=False)
    alternativeValue: float = Field(allow_inf_nan=False)
    interpretation: str = Field(min_length=1)


def comparison_advantages(coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate exact measured counterevidence for each observed preference."""
    settings = coverage["candidateSettings"]
    axes: dict[str, set[str]] = {}
    for comparison in coverage["comparisons"]:
        identifiers = axes.setdefault(comparison["axis"], set())
        identifiers.add(comparison["baselineCandidateId"])
        if comparison["status"] == "completed":
            identifiers.add(comparison["alternativeCandidateId"])
    rows = []
    metrics = get_args(ComparisonTradeoff.model_fields["metric"].annotation)
    for axis, identifiers in sorted(axes.items()):
        for preferred_id in sorted(identifiers):
            preferred = settings[preferred_id]["metrics"]
            for alternative_id in sorted(identifiers - {preferred_id}):
                alternative = settings[alternative_id]["metrics"]
                for metric in metrics:
                    left, right = preferred.get(metric), alternative.get(metric)
                    if (
                        isinstance(left, (int, float))
                        and not isinstance(left, bool)
                        and isinstance(right, (int, float))
                        and not isinstance(right, bool)
                        and np.isfinite(left)
                        and np.isfinite(right)
                        and right > left
                    ):
                        rows.append(
                            {
                                "axis": axis,
                                "preferredCandidateId": preferred_id,
                                "alternativeCandidateId": alternative_id,
                                "metric": metric,
                                "preferredValue": float(left),
                                "alternativeValue": float(right),
                                "difference": float(right - left),
                            }
                        )
    return rows


def bind_comparison_measurements(
    coverage: Mapping[str, Any], action: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach measured values to model-authored interpretations, never rationales."""
    inventory = {
        (
            row["axis"],
            row["preferredCandidateId"],
            row["alternativeCandidateId"],
            row["metric"],
        ): row
        for row in comparison_advantages(coverage)
    }
    conclusions = []
    errors = []
    permitted = {}
    for conclusion in action.get("comparisonConclusions", []):
        preference = (conclusion["axis"], conclusion["preferredCandidateId"])
        required = {key: row for key, row in inventory.items() if key[:2] == preference}
        permitted.update(required)
        prefix = (
            f"comparisonConclusions[axis={preference[0]!r}, "
            f"preferredCandidateId={preference[1]!r}].tradeoffs"
        )
        tradeoffs = []
        supplied = set()
        for interpretation in conclusion.get("tradeoffs", []):
            key = (
                *preference,
                interpretation["alternativeCandidateId"],
                interpretation["metric"],
            )
            if key in supplied:
                errors.append(f"{prefix}: duplicate tradeoff entry {key[2:]!r}")
            supplied.add(key)
            measured = inventory.get(key)
            if measured is None:
                settings = coverage["candidateSettings"]
                left = settings.get(preference[1], {}).get("metrics", {}).get(key[3])
                right = settings.get(key[2], {}).get("metrics", {}).get(key[3])
                errors.append(
                    f"{prefix}: {key[2:]!r} does not identify an observed advantage "
                    f"(preferred={left!r}, alternative={right!r}). "
                    "Only permittedTradeoffs rows belong here; comparisons favoring "
                    "the preference and ties belong in quantitativeReason."
                )
                continue
            values = {
                field: measured[field]
                for field in ("preferredValue", "alternativeValue")
            }
            if any(
                interpretation.get(field) is not None and interpretation[field] != value
                for field, value in values.items()
            ):
                errors.append(
                    f"{prefix}: use exact preferred and alternative measurements "
                    f"for {key[2:]!r}: {values!r}"
                )
            tradeoffs.append({**interpretation, **values})
        if action.get("action") in {"accept", "combine"}:
            for key in sorted(required.keys() - supplied):
                errors.append(
                    f"{prefix}: add alternativeCandidateId={key[2]!r}, "
                    f"metric={key[3]!r} and an interpretation of its measured advantage"
                )
        conclusions.append({**conclusion, "tradeoffs": tradeoffs})
    if errors:
        raise ValueError(
            "Repair all comparisonConclusions tradeoffs errors together. "
            "permittedTradeoffs contains the exact allowed rows for your current "
            "axis and preferredCandidateId choices. For accept or combine, explain "
            "every matching row using alternativeCandidateId, metric and interpretation. "
            "Do not include other comparisons, worse alternatives or ties in tradeoffs; "
            "put general comparisons in quantitativeReason. Expanding quantitativeReason "
            "or biologicalReason does not replace required entries. Scarf supplies "
            "preferredValue and alternativeValue; do not transcribe them. "
            "Preserve unaffected choices and explanations. If you reconsider a "
            "preference, update its conclusion and combinedSettings consistently and "
            "use comparisonAdvantages for the new preference. Required repairs: "
            + json.dumps(
                {"errors": errors, "permittedTradeoffs": list(permitted.values())},
                sort_keys=True,
            )
        )
    return {**action, "comparisonConclusions": conclusions}


class PopulationConcern(AgentDataModel):
    """Keep unsupported population interpretations explicit in acceptance."""

    candidateId: str
    clusterId: str
    status: Literal["nonEssentialLimitation", "unresolvedEssential"]
    evidenceIds: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class CombinedSettings(AgentDataModel):
    """Propose one combination using settings from actual completed candidates."""

    hvgCountCandidateId: str
    hvgRankingCandidateId: str
    featurePolicyCandidateId: str
    pcaCandidateId: str
    neighborsCandidateId: str


_REQUIRED_COMPARISONS = {
    "defaultResolution:0.5",
    "defaultResolution:0.75",
    "defaultResolution:1.25",
    "hvgCount:2000",
    "hvgCount:4000",
    "dimensions:10",
    "dimensions:30",
    "neighborsK:21",
    "neighborsK:41",
    "hvgRanking",
    "featurePolicy",
}
_CHOICE_AXES = {
    "hvgCountCandidateId": "hvgCount",
    "hvgRankingCandidateId": "hvgRanking",
    "featurePolicyCandidateId": "featurePolicy",
    "pcaCandidateId": "pca",
    "neighborsCandidateId": "neighbors",
}


def partition_comparison_evidence(
    store: Any,
    left: ParameterCandidateEvaluation,
    right: ParameterCandidateEvaluation,
) -> dict[str, Any]:
    """Describe bounded observed splits and merges on exactly matched cells."""
    if left.cellSelection is None or left.cellSelection != right.cellSelection:
        raise ValueError("Partition comparisons require the same frozen cells")
    arrays = []
    for candidate in (left, right):
        reference = candidate.artifacts["clusters"]
        ref = ArtifactRef(
            scope=reference.scope,
            assay=reference.assay,
            kind=reference.kind,
            artifact_id=reference.artifactId,
        )
        status = store.inspect_artifact(ref)
        cells = left.cellSelection
        cell_ref = ArtifactRef(
            scope=cells.scope,
            assay=cells.assay,
            kind=cells.kind,
            artifact_id=cells.artifactId,
        )
        if (
            not status.complete
            or status.inputs.get("cell_selection") != cell_ref.to_dict()
        ):
            raise ValueError(
                "Partition evidence does not bind its exact selected cells"
            )
        arrays.append(store.load_artifact(ref)["values"])
    if len(arrays[0].shape) != 1 or arrays[0].shape != arrays[1].shape:
        raise ValueError("Partition labels do not align")
    counts: Counter[tuple[str, str]] = Counter()
    for start in range(0, arrays[0].shape[0], 65_536):
        a, b = (
            np.asarray(values[start : start + 65_536]).astype(str) for values in arrays
        )
        pairs, numbers = np.unique(np.column_stack((a, b)), axis=0, return_counts=True)
        counts.update(
            {
                (str(pair[0]), str(pair[1])): int(number)
                for pair, number in zip(pairs, numbers, strict=True)
            }
        )

    def summarize(reverse: bool) -> list[dict[str, Any]]:
        by_source: dict[str, Counter[str]] = {}
        for (a, b), count in counts.items():
            source, target = (b, a) if reverse else (a, b)
            by_source.setdefault(source, Counter())[target] += count
        source_metrics = right.metrics if reverse else left.metrics
        target_metrics = left.metrics if reverse else right.metrics
        rows: list[dict[str, Any]] = []
        for source, targets in by_source.items():
            total = sum(targets.values())
            ordered = targets.most_common()
            rows.append(
                {
                    "cluster": source,
                    "cells": total,
                    "fractionOutsideLargestMatch": 1 - ordered[0][1] / total,
                    "markerGenes": source_metrics.topMarkerGenes.get(source, [])[:5],
                    "matches": [
                        {
                            "cluster": target,
                            "cells": number,
                            "fractionOfSource": number / total,
                            "markerGenes": target_metrics.topMarkerGenes.get(
                                target, []
                            )[:5],
                        }
                        for target, number in ordered[:3]
                    ],
                    "omittedMatches": max(0, len(ordered) - 3),
                    "omittedCells": sum(number for _, number in ordered[3:]),
                }
            )
        return sorted(
            rows, key=lambda row: (-row["fractionOutsideLargestMatch"], row["cells"])
        )[:5]

    return {
        "matchedCells": arrays[0].shape[0],
        "splits": summarize(False),
        "merges": summarize(True),
        "interpretation": "Observed same-cell partition overlap, with up to five split and merge examples and three matches per example. Marker names are bounded summaries; this is not independent stability or a validated cell-type identity.",
    }


def setting_changes(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Describe logical interventions, allowing the resulting gene set to change."""
    changes = {}
    if left["hvgCount"] != right["hvgCount"] and left["features"] == right["features"]:
        raise ValueError(
            "One frozen feature artifact cannot have different selected-gene counts"
        )
    for field, axis in (
        ("dimensions", "pca"),
        ("neighborsK", "neighbors"),
        ("leidenResolution", "partition"),
        ("useHarmony", "correction"),
    ):
        a, b = left["parameters"][field], right["parameters"][field]
        if a != b:
            changes[axis] = {"current": a, "alternative": b}
    for field, axis in (
        ("hvgCount", "hvgCount"),
        ("eligibleFeatures", "featurePolicy"),
    ):
        if left[field] != right[field]:
            changes[axis] = {"current": left[field], "alternative": right[field]}
    a = (left["ranking"], left["rankingColumn"])
    b = (right["ranking"], right["rankingColumn"])
    if a != b:
        changes["hvgRanking"] = {"current": list(a), "alternative": list(b)}
    if left["features"] != right["features"] and not {
        "hvgCount",
        "hvgRanking",
        "featurePolicy",
    }.intersection(changes):
        changes["unexplainedFeatures"] = {
            "current": left["features"],
            "alternative": right["features"],
        }
    return changes


def validate_comparison_review(
    coverage: Mapping[str, Any], action: Mapping[str, Any]
) -> None:
    """Validate saved or new conclusions against exact completed comparison inputs."""
    required = {
        "phase",
        "population",
        "comparisons",
        "candidateSettings",
        "combinedCandidateId",
        "resolutionCandidateIds",
    }
    if not required.issubset(coverage):
        raise ValueError(
            "Saved analysis lacks mandatory comparison coverage; start a new workflow"
        )
    if coverage["phase"] not in {"sensitivity", "combined", "validation"}:
        raise ValueError("Unknown RNA comparison phase")
    if coverage["population"] not in {"subset", "allCells"}:
        raise ValueError("Comparison population must identify sampled or all cells")
    settings = coverage["candidateSettings"]
    rows = coverage["comparisons"]
    if not isinstance(settings, Mapping) or not isinstance(rows, list):
        raise ValueError("Comparison coverage must contain exact settings and rows")
    if len({row["comparisonId"] for row in rows}) != len(rows):
        raise ValueError("Comparison IDs must be unique")
    if not _REQUIRED_COMPARISONS.issubset({row["comparisonId"] for row in rows}):
        raise ValueError("Required RNA sensitivity comparisons are missing")
    axis_candidates: dict[str, set[str]] = {}
    for row in rows:
        axis = row["axis"]
        left_id, right_id = row["baselineCandidateId"], row["alternativeCandidateId"]
        if left_id not in settings or settings[left_id]["status"] != "done":
            raise ValueError("A comparison baseline is not completed evidence")
        axis_candidates.setdefault(axis, set()).add(left_id)
        if row["status"] == "pending" and axis == "featurePolicy":
            if action["action"] in {"combine", "accept"}:
                raise ValueError(
                    "A feature-policy comparison still needs an evidence-based nomination and execution"
                )
            continue
        if row["status"] == "notApplicable":
            if not str(row.get("reason", "")).strip():
                raise ValueError(
                    "An unavailable comparison needs its observed eligibility reason"
                )
            if right_id is not None and right_id != left_id:
                raise ValueError(
                    "An unavailable comparison cannot claim a different evaluated alternative"
                )
            proof = row.get("observedProof", {})
            baseline = settings[left_id]
            kind = proof.get("kind")
            if proof.get("baselineFeatures") != baseline["features"]:
                raise ValueError(
                    "Unavailable comparisons require observed proof bound to the exact baseline features"
                )
            valid = False
            if axis in {"pca", "neighbors"}:
                requested = int(row["comparisonId"].split(":")[1])
                field = "dimensions" if axis == "pca" else "neighborsK"
                bound = (
                    min(baseline["hvgCount"], baseline["nCells"])
                    if axis == "pca"
                    else baseline["nCells"]
                )
                valid = kind == "numericalBound" and (
                    requested >= bound or requested == baseline["parameters"][field]
                )
            elif kind == "identicalSelectedGenes":
                alternative = proof.get("alternativeSetting", {})
                differences = setting_changes(
                    baseline, {**alternative, "features": baseline["features"]}
                )
                valid = (
                    set(differences) <= {axis}
                    and alternative["hvgCount"] == baseline["hvgCount"]
                    and proof.get("verifiedEqualMasks") is True
                )
            elif axis == "hvgRanking" and kind == "insufficientTechnicalGroups":
                groups = proof.get("eligibleGroupsByColumn")
                valid = isinstance(groups, dict) and all(
                    isinstance(count, int) and count < 2 for count in groups.values()
                )
            elif axis == "featurePolicy" and kind == "noPermittedPolicy":
                valid = proof.get(
                    "meaningfulPermittedInterventions"
                ) == 0 and isinstance(proof.get("registeredFamilies"), list)
            elif axis == "featurePolicy" and kind == "insufficientEligibleGenes":
                valid = (
                    isinstance(proof.get("eligibleFeatureCount"), int)
                    and proof["eligibleFeatureCount"] < baseline["hvgCount"]
                )
            if not valid:
                raise ValueError(
                    "An unavailable comparison needs a valid observed equivalence or infeasibility proof for its exact axis"
                )
            continue
        if row["status"] != "completed" or right_id not in settings:
            raise ValueError("A required comparison lacks a completed alternative")
        left, right = settings[left_id], settings[right_id]
        if right["status"] != "done" or left["cellSelection"] != right["cellSelection"]:
            raise ValueError(
                "Comparison candidates must be complete on the exact same cells"
            )
        differences = setting_changes(left, right)
        if set(differences) != {axis}:
            raise ValueError(
                "A sensitivity comparison must change only its declared setting"
            )
        axis_candidates[axis].add(right_id)
    if (
        "comparisonConclusions" not in action
        or not str(action.get("plainLanguageSummary", "")).strip()
    ):
        raise ValueError(
            "Analysis review lacks explicit comparison conclusions and reader summary"
        )
    conclusions = [
        ComparisonConclusion.model_validate(row)
        for row in action["comparisonConclusions"]
    ]
    if action["action"] in {"accept", "combine"}:
        by_axis: dict[str, ComparisonConclusion] = {
            row.axis: row for row in conclusions
        }
        if len(by_axis) != len(conclusions) or set(by_axis) != set(axis_candidates):
            raise ValueError(
                "Conclude every required comparison axis before combining or accepting"
            )
        for axis, ids in axis_candidates.items():
            conclusion = by_axis[axis]
            if not ids.issubset(conclusion.candidateIds) or not set(
                conclusion.candidateIds
            ).issubset(settings):
                raise ValueError(
                    "A conclusion must address the actual baseline and all its observed alternatives"
                )
            if conclusion.preferredCandidateId not in ids:
                raise ValueError(
                    "A comparison preference must name its observed candidate"
                )
        bind_comparison_measurements(coverage, action)
    if action["action"] == "combine":
        if coverage["phase"] != "sensitivity":
            raise ValueError(
                "Only the sensitivity assessment may propose combined settings"
            )
        proposal = CombinedSettings.model_validate(action.get("combinedSettings"))
        for field, axis in _CHOICE_AXES.items():
            if getattr(proposal, field) not in axis_candidates[axis]:
                raise ValueError(
                    "Combined settings must use values from the matching observed sensitivity axis"
                )
            if getattr(proposal, field) != by_axis[axis].preferredCandidateId:
                raise ValueError(
                    "The proposed combination must agree with its comparison conclusions"
                )
    if action["action"] == "accept":
        if (
            coverage["phase"] == "sensitivity"
            or coverage["combinedCandidateId"] not in settings
        ):
            raise ValueError(
                "Acceptance requires execution of the proposed combined settings"
            )
        combined = settings[coverage["combinedCandidateId"]]
        partition_ids = coverage["resolutionCandidateIds"]
        if not isinstance(partition_ids, list) or len(partition_ids) != 4:
            raise ValueError(
                "Acceptance requires four resolutions on the combined representation"
            )
        resolutions = set()
        for identifier in partition_ids:
            if identifier not in settings:
                raise ValueError("A final resolution candidate is unavailable")
            row = settings[identifier]
            if (
                row["status"] != "done"
                or row["cellSelection"] != combined["cellSelection"]
            ):
                raise ValueError(
                    "Final resolutions must be completed on the same cells"
                )
            if set(setting_changes(combined, row)) - {"partition", "correction"}:
                raise ValueError(
                    "Final resolution evidence changed the combined representation"
                )
            resolutions.add(row["parameters"]["leidenResolution"])
        if resolutions != {0.5, 0.75, 1.0, 1.25}:
            raise ValueError("Final resolution coverage is incomplete")
        selected = settings.get(action.get("selectedCandidateId"))
        if selected is None:
            raise ValueError("Accepted settings lack exact completed evidence")
        concerns = [
            PopulationConcern.model_validate(row)
            for row in action.get("populationConcerns", [])
        ]
        if any(row.status == "unresolvedEssential" for row in concerns):
            raise ValueError(
                "An essential population interpretation remains unresolved"
            )
        missing_markers = {
            str(cluster)
            for cluster, genes in selected["metrics"].get("topMarkerGenes", {}).items()
            if not genes
        }
        if selected["metrics"].get("nClusters") != len(
            selected["metrics"].get("topMarkerGenes", {})
        ):
            raise ValueError(
                "Acceptance needs marker support or explicit missing-marker evidence for every selected cluster"
            )
        addressed = {
            row.clusterId
            for row in concerns
            if row.candidateId == action["selectedCandidateId"]
        }
        if not missing_markers <= addressed:
            raise ValueError(
                "Acceptance must explicitly resolve each selected population without qualifying markers against the objective"
            )
        for concern in concerns:
            if concern.candidateId not in settings or not set(
                concern.evidenceIds
            ) <= set(action["evidenceIds"]):
                raise ValueError(
                    "Population concerns must cite supplied candidate evidence used by the assessment"
                )
        if not any(
            set(setting_changes(selected, settings[identifier])) <= {"correction"}
            for identifier in partition_ids
        ):
            repair = coverage.get("fullRepair")
            if (
                coverage["phase"] != "validation"
                or not isinstance(repair, Mapping)
                or repair.get("selectedCandidateId") != action["selectedCandidateId"]
                or repair.get("baselineCandidateId") not in settings
                or len(
                    setting_changes(settings[repair["baselineCandidateId"]], selected)
                )
                != 1
            ):
                raise ValueError(
                    "Accepted settings were not validated by the final comparison panel or its one targeted full repair"
                )
