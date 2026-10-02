"""Exploration and conservative resolutions extend the existing offline report."""

from pathlib import Path
from typing import Any

import pytest

from scarf.agent.rendering import summary
from scarf.agent.result import AnalysisRun

from .test_agent_report import _complete, _Document, _rich_records


def test_legacy_coverage_is_unknown_and_policy_resolutions_are_empty(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    result = AnalysisRun(records.path)
    assert result.exploration_coverage is None
    assert result.decision_resolutions == []
    page = result.report().read_text()
    assert "Exploration coverage was not recorded" in page
    assert "Marker coherence (marker support)" in page


def test_saved_coverage_and_resolutions_remain_separate_from_model_decisions(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    coverage = {
        "slots": [
            {
                "candidateId": "c0",
                "axis": "baseline",
                "parentId": None,
                "status": "measured",
                "reason": "",
            },
            {
                "candidateId": "c1",
                "axis": "hvgCount",
                "parentId": "c0",
                "status": "failed",
                "reason": "Numerical convergence failed.",
            },
        ],
        "nativeComplete": False,
    }
    records.append("explorationCoverage", coverage=coverage)
    records.append(
        "decisionResolved",
        stage="finalists",
        acceptedDecisionId="select",
        reason="ambiguousSelection",
        rule="preferNativeThenTrialOrder",
        resolved={"action": "choose", "optionIds": ["c0:r0.5"]},
        limitation="The selected option is not demonstrated to be superior.",
    )
    before = {path: path.read_bytes() for path in records.path.rglob("*.json")}
    result = AnalysisRun(records.path)
    assert result.exploration_coverage == coverage
    assert len(result.decision_resolutions) == 1
    saved = summary(records)
    assert len(saved["decisionResolutions"]) == 1
    assert not any(row.get("rule") for row in saved["decisions"])
    document = _Document(result.report().read_text())
    assert "Numerical convergence failed." in " ".join(document.sections["exploration"])
    assert "Native sensitivity coverage is incomplete" in " ".join(
        document.sections["exploration"]
    )
    assert "Automatic conservative resolution" in " ".join(
        document.sections["selection"]
    )
    assert "not demonstrated to be superior" in " ".join(document.sections["selection"])
    assert all(path.read_bytes() == value for path, value in before.items())


def test_completed_coverage_is_read_from_saved_exploration(tmp_path: Path) -> None:
    records = _rich_records(tmp_path)
    coverage = {"slots": [], "nativeComplete": True}
    _complete(records, "explore", "coverage-final", {"explorationCoverage": coverage})
    result = AnalysisRun(records.path)
    assert result.exploration_coverage == coverage
    assert (
        "All planned native comparisons were measured." in result.report().read_text()
    )


@pytest.mark.parametrize(
    "value, expected", [(0, "0.0%"), (None, "Not recorded"), (0.75, "75.0%")]
)
def test_new_marker_support_takes_precedence_even_when_zero_or_missing(
    tmp_path: Path, value: Any, expected: str
) -> None:
    records = _rich_records(tmp_path)
    finalist = records.read_json("evidence/selected-finalist.json")
    finalist["metrics"]["markerSupportFraction"] = value
    records.append(
        "finalistMeasured",
        optionId="c1:r0.5",
        evidence=records.write_evidence("new-support", finalist),
    )
    document = _Document(AnalysisRun(records.path).report().read_text())
    row = next(
        row
        for row in document.rows
        if row[0].startswith("Alternative 1") and len(row) == 4
    )
    assert row[1] == expected
    assert "partition" in " ".join(document.sections["selection"])


def test_new_diagnostics_extend_existing_steps_without_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.evidence as evidence

    records = _rich_records(tmp_path)
    prepared = records.read_json("evidence/prepared.json")
    prepared.update(
        resolvedRoles=[
            {
                "column": "donor_id",
                "role": "sample",
                "source": "inferred",
                "authority": "diagnosticOnly",
            }
        ],
        designDiagnostics={
            "crossTabs": [
                {
                    "leftColumn": "donor_id",
                    "rightColumn": "condition",
                    "rowsMissing": 3,
                    "counts": [{"left": "A", "right": "healthy", "count": 100}],
                }
            ],
            "limitations": ["Inference is descriptive."],
        },
        qcProjections=[
            {
                "policy": "gentleMad5",
                "retainedCells": 11950,
                "removedCells": 50,
                "executed": False,
                "available": True,
                "limitations": ["Global thresholds."],
                "bounds": {"RNA_nFeatures": [200, None]},
                "byGroup": {
                    "donor_id": {
                        "levels": [
                            {
                                "value": "A",
                                "inputCells": 100,
                                "retainedCells": 90,
                                "removedCells": 10,
                            }
                        ],
                        "missingInputCells": 1,
                        "missingRetainedCells": 1,
                        "omittedLevels": 0,
                    }
                },
            }
        ],
        featureAudit={
            "families": [
                {
                    "family": "ribosomal",
                    "matchedFeatures": 40,
                    "excludedFeatures": 0,
                    "standardExcludedFeatures": 40,
                }
            ],
            "limitations": ["Feature names can be incomplete."],
        },
    )
    _complete(records, "preprocess", "enriched-preparation", prepared)
    candidate = records.read_json("evidence/candidate.json")
    candidate.update(
        actualHvgCount=1000,
        covariateAssociations=[
            {
                "column": "RNA_nCounts",
                "kind": "continuous",
                "component": 1,
                "association": 0.8,
                "rowsUsed": 1000,
                "rowsMissing": 0,
            }
        ],
        hvgAudit={
            "families": [{"family": "ribosomal", "matchedFeatures": 4}],
            "limitations": ["Gene names are required."],
        },
        loadingFamilies=[
            {
                "component": 1,
                "topGenes": [{"gene": "RPL3", "loading": 0.2}],
                "families": {"ribosomal": 1},
            }
        ],
        comparisons=[
            {
                "parentCandidateId": "c0",
                "candidateId": "c1",
                "resolution": 0.5,
                "adjustedRandIndex": 0.82,
                "cellCount": 12000,
                "parentToCandidate": [
                    {
                        "clusterId": "1",
                        "matchedClusterId": "3",
                        "sourceCells": 8000,
                        "intersectionCells": 6000,
                        "fraction": 0.75,
                    }
                ],
                "candidateToParent": [
                    {
                        "clusterId": "3",
                        "matchedClusterId": "1",
                        "sourceCells": 6000,
                        "intersectionCells": 6000,
                        "fraction": 1.0,
                    }
                ],
            }
        ],
    )
    records.append(
        "candidateMeasured",
        candidateId="c1",
        evidence=records.write_evidence("enriched-candidate", candidate),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Report must not reopen the numerical source")

    monkeypatch.setattr(evidence, "open_store", forbidden)
    document = _Document(AnalysisRun(records.path).report().read_text())
    assert list(document.sections) == [
        "overview",
        "quality",
        "exploration",
        "selection",
        "results",
        "populations",
        "provenance",
    ]
    assert "Diagnostic only" in " ".join(document.sections["overview"])
    assert "healthy" in " ".join(document.sections["overview"])
    quality = " ".join(document.sections["quality"])
    assert "Projection only" in quality
    assert "11,950" in quality
    assert "ribosomal" in quality
    assert ["A", "100", "90", "10"] in document.rows
    assert "0.8" in " ".join(document.sections["exploration"])
    assert "0.82" in " ".join(document.sections["exploration"])
    assert "RPL3" in " ".join(document.sections["exploration"])
    assert ["1", "3", "8,000", "6,000", "75.0%"] in document.rows
    assert not any(tag == "table" for tag, _ in document.section_tags["results"])


def test_final_cluster_quality_and_group_evidence_stay_with_identity(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    assessed = records.read_json("evidence/assessed.json")
    cluster = assessed["finalists"]["c0:r0.5"]["clusters"][0]
    cluster["qc"] = {
        "RNA_percentMito": {"median": 3.5, "q10": 1.0, "q90": 8.0, "missing": 0}
    }
    cluster["groupComposition"] = {
        "donor_id": {
            "levels": [{"value": "A", "count": 4000, "fraction": 0.5}],
            "missing": 0,
            "omittedLevels": 0,
        }
    }
    _complete(records, "finalists", "assessed-grouped", assessed)
    document = _Document(AnalysisRun(records.path).report().read_text())
    assert ["A", "4,000", "50.0%"] in document.rows
    assert "3.5" in " ".join(document.sections["populations"])
    assert "does not establish independent replication" in " ".join(
        document.sections["populations"]
    )
    assert not any(tag == "table" for tag, _ in document.section_tags["results"])


def test_new_evidence_text_is_escaped_in_both_report_formats(tmp_path: Path) -> None:
    records = _rich_records(tmp_path)
    records.append(
        "explorationCoverage",
        coverage={
            "nativeComplete": False,
            "slots": [
                {
                    "candidateId": "c1",
                    "axis": "<script>axis</script>",
                    "status": "failed",
                    "reason": "<img src=x onerror=alert(1)>",
                }
            ],
        },
    )
    records.append(
        "decisionResolved",
        stage="context",
        reason="<script>reason</script>",
        rule="safe",
        resolved={"action": "retainDeclaredRoles"},
        limitation="<img src=x>",
    )
    page = AnalysisRun(records.path).report().read_text()
    document = _Document(page)
    assert not any(tag == "script" for tag, _ in document.tags)
    assert "&lt;script&gt;axis&lt;/script&gt;" in page
    markdown = (records.path / "report.md").read_text()
    assert "&lt;img src=x onerror=alert(1)&gt;" in markdown
