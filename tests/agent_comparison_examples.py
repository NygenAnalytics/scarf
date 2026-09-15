"""Exact, completed RNA comparison evidence for contract and report tests."""

from copy import deepcopy


def observed_action(evidence: dict, *, selected: str | None = None) -> dict:
    """A deterministic synthetic-study reviewer grounded in its supplied rows."""
    coverage = evidence["comparisonCoverage"]
    rows = coverage["comparisons"]
    settings = coverage["candidateSettings"]
    baseline = rows[0]["baselineCandidateId"]
    chosen = selected or evidence["currentCandidateId"]
    combining = coverage["phase"] == "sensitivity"
    pending = next((row for row in rows if row["status"] == "pending"), None)
    experiment = (
        next(
            (
                key
                for key, value in evidence.get("experiments", {}).items()
                if value["parameter"]
                in {
                    "includeFamily",
                    "excludeFamily",
                    "includeFeature",
                    "excludeFeature",
                }
            ),
            None,
        )
        if pending
        else None
    )
    conclusions = []
    for axis in dict.fromkeys(row["axis"] for row in rows):
        ids = list(
            dict.fromkeys(
                identifier
                for row in rows
                if row["axis"] == axis
                for identifier in (
                    row["baselineCandidateId"],
                    row["alternativeCandidateId"],
                )
                if identifier
            )
        )
        preferred = (
            chosen
            if axis == "partition" and chosen in ids and not combining
            else baseline
        )
        metrics = settings[preferred]["metrics"]
        tradeoffs = []
        for identifier in ids:
            if identifier == preferred:
                continue
            for metric in (
                "seedStability",
                "subsampleStability",
                "markerCoherence",
                "markerSpecificityMedian",
                "macroF1",
            ):
                value = settings[identifier]["metrics"].get(metric)
                current = metrics.get(metric)
                if (
                    isinstance(value, (int, float))
                    and isinstance(current, (int, float))
                    and value > current
                ):
                    tradeoffs.append(
                        {
                            "alternativeCandidateId": identifier,
                            "metric": metric,
                            "preferredValue": current,
                            "alternativeValue": value,
                            "interpretation": "The synthetic reference's marker program and exact population partition remain the objective; this measured improvement alone does not justify replacing that partition.",
                        }
                    )
        conclusions.append(
            {
                "axis": axis,
                "candidateIds": ids,
                "preferredCandidateId": preferred,
                "quantitativeReason": f"Compare the actual stability and marker measurements for {len(ids)} observed settings.",
                "biologicalReason": "Preserve the synthetic reference marker programs and population membership.",
                "plainLanguageSummary": "The reference population remains represented with its marker program.",
                "tradeoffs": tradeoffs,
            }
        )
    evidence_ids = [f"candidate:{chosen}", *evidence.get("imageHashes", {})]
    return {
        "action": "experiment" if pending else "combine" if combining else "accept",
        "selectedCandidateId": chosen,
        "experimentId": experiment,
        "combinedSettings": {
            field: baseline
            for field in (
                "hvgCountCandidateId",
                "hvgRankingCandidateId",
                "featurePolicyCandidateId",
                "pcaCandidateId",
                "neighborsCandidateId",
            )
        }
        if combining and not pending
        else None,
        "correctionNeed": "notApplicable",
        "evidenceIds": evidence_ids,
        "quantitativeFindings": [
            "Use the supplied same-cell marker and stability measurements."
        ],
        "qualitativeFindings": [
            "Interpret the observed marker names against the known synthetic population program."
        ],
        "comparisonConclusions": conclusions,
        "populationConcerns": [
            {
                "candidateId": chosen,
                "clusterId": cluster,
                "status": "nonEssentialLimitation",
                "evidenceIds": evidence_ids,
                "explanation": "No cell identity is assigned to this marker-poor synthetic background group; the reference marker-defined group remains supported.",
            }
            for cluster, genes in settings[chosen]["metrics"]
            .get("topMarkerGenes", {})
            .items()
            if not genes
        ],
        "plainLanguageSummary": "The observed reference population and its marker genes are retained.",
        "concern": "Test the nominated family contribution to the selected population program."
        if pending
        else "",
        "expectedImprovement": "Determine whether this exact gene policy changes marker-supported separation."
        if pending
        else "",
        "objectivePreservation": "Retain the known synthetic population and marker program.",
        "rationale": "Use the measured comparisons and preserve the explicitly defined synthetic reference population.",
    }


def comparison_review(scope: str = "full") -> dict:
    cells = {
        "scope": "datastore",
        "assay": None,
        "kind": "cell_selection",
        "artifactId": "c" * 64,
        "type": "artifact",
    }
    features = {
        "scope": "assay",
        "assay": "RNA2",
        "kind": "feature_selection",
        "artifactId": "4" * 64,
        "type": "artifact",
    }
    baseline = {
        "scope": scope,
        "status": "done",
        "nCells": 621200,
        "cellSelection": cells,
        "features": features,
        "eligibleFeatures": {**features, "artifactId": "6" * 64},
        "hvgCount": 1000,
        "ranking": "global",
        "rankingColumn": None,
        "parameters": {
            "dimensions": 20,
            "neighborsK": 15,
            "leidenResolution": 1.0,
            "useHarmony": False,
        },
        "metrics": {
            "nClusters": 2,
            "minClusterCells": 1200,
            "seedStability": 0.92,
            "subsampleStability": 0.88,
            "markerCoherence": 0.84,
            "topMarkerGenes": {"0": ["MS4A1"], "1": ["CD3D"]},
        },
    }
    settings = {"baseline": baseline}
    rows = []
    for comparison_id, axis, identity, field, value in (
        (
            "defaultResolution:0.5",
            "partition",
            "resolution-half",
            "leidenResolution",
            0.5,
        ),
        (
            "defaultResolution:0.75",
            "partition",
            "candidate-two",
            "leidenResolution",
            0.75,
        ),
        (
            "defaultResolution:1.25",
            "partition",
            "resolution-high",
            "leidenResolution",
            1.25,
        ),
        ("hvgCount:2000", "hvgCount", "genes-two", "hvgCount", 2000),
        ("hvgCount:4000", "hvgCount", "genes-four", "hvgCount", 4000),
        ("dimensions:10", "pca", "dimensions-ten", "dimensions", 10),
        ("dimensions:30", "pca", "dimensions-thirty", "dimensions", 30),
        ("neighborsK:21", "neighbors", "neighbors-twenty-one", "neighborsK", 21),
        ("neighborsK:41", "neighbors", "neighbors-forty-one", "neighborsK", 41),
    ):
        setting = deepcopy(baseline)
        target = setting if field == "hvgCount" else setting["parameters"]
        target[field] = value
        if field == "hvgCount":
            setting["features"] = {
                **features,
                "artifactId": ("8" if value == 2000 else "9") * 64,
            }
        settings[identity] = setting
        rows.append(
            {
                "comparisonId": comparison_id,
                "axis": axis,
                "status": "completed",
                "baselineCandidateId": "baseline",
                "alternativeCandidateId": identity,
                "reason": "Observed sensitivity comparison",
            }
        )
    for axis, reason in (
        ("hvgRanking", "No technical grouping supports a batch-specific ranking."),
        (
            "featurePolicy",
            "No observed unprotected gene family supports an exclusion comparison.",
        ),
    ):
        rows.append(
            {
                "comparisonId": axis,
                "axis": axis,
                "status": "notApplicable",
                "baselineCandidateId": "baseline",
                "alternativeCandidateId": None,
                "reason": reason,
                "observedProof": {
                    "baselineFeatures": features,
                    **(
                        {
                            "kind": "insufficientTechnicalGroups",
                            "eligibleGroupsByColumn": {},
                        }
                        if axis == "hvgRanking"
                        else {
                            "kind": "noPermittedPolicy",
                            "meaningfulPermittedInterventions": 0,
                            "registeredFamilies": [],
                        }
                    ),
                },
            }
        )
    conclusions = []
    for axis in dict.fromkeys(row["axis"] for row in rows):
        identities = [
            "baseline",
            *[
                row["alternativeCandidateId"]
                for row in rows
                if row["axis"] == axis and row["alternativeCandidateId"]
            ],
        ]
        conclusions.append(
            {
                "axis": axis,
                "candidateIds": identities,
                "preferredCandidateId": "candidate-two"
                if axis == "partition"
                else "baseline",
                "quantitativeReason": "Repeat agreement was 0.92 and 84% of clusters had qualifying markers.",
                "biologicalReason": "The recorded marker programs remain available for interpretation.",
                "plainLanguageSummary": "Resolution 0.75 retains a small population with clear markers."
                if axis == "partition"
                else f"The observed {axis} alternatives did not justify changing this setting.",
            }
        )
    return {
        "scope": scope,
        "action": "accept",
        "selectedCandidateId": "candidate-two",
        "experimentId": None,
        "correctionNeed": "notApplicable",
        "evidenceIds": ["candidate:candidate-two:clusters"],
        "quantitativeFindings": ["Repeat agreement was 0.92."],
        "qualitativeFindings": ["MS4A1 and CD79A support the same marker program."],
        "objectivePreservation": "Retain the small marker-supported population.",
        "rationale": "Original detailed model reasoning retained in the journal.",
        "plainLanguageSummary": "The selected settings retain a small population with clear marker genes.",
        "comparisonConclusions": conclusions,
        "evidenceMode": "structured",
        "visualInspection": "unavailable",
        "coverage": {
            "populationCells": 621200,
            "screeningCells": 621200 if scope == "full" else 50000,
        },
        "comparisonCoverage": {
            "phase": "validation" if scope == "full" else "combined",
            "population": "allCells" if scope == "full" else "subset",
            "comparisons": rows,
            "candidateSettings": settings,
            "combinedCandidateId": "baseline",
            "resolutionCandidateIds": [
                "resolution-half",
                "candidate-two",
                "baseline",
                "resolution-high",
            ],
        },
        "settings": {
            identity: deepcopy(setting) for identity, setting in settings.items()
        },
        "featureEvidence": {
            identity: {
                "selectedGenes": setting["hvgCount"],
                "eligibleGenes": 12000,
                "families": {"hla": {"eligibleGenes": 12, "selectedGenes": 6}},
            }
            for identity, setting in settings.items()
        },
        "candidates": [
            {
                "candidateId": identity,
                "parameters": setting["parameters"],
                "metrics": setting["metrics"],
                "cellSelection": cells,
                "artifacts": {"graphFeatures": setting["features"]},
            }
            for identity, setting in settings.items()
        ],
    }
