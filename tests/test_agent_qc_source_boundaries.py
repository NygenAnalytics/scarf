"""QC policies cannot acquire invented metric, capture, or replication support."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.experimental_context import qc_evidence as q
from scarf.agent.experimental_context.contracts import (
    CaptureProposal,
    NamedArtifactSource,
)
from scarf.agent.tools import artifact_reference
from scarf.storage.refs import ArtifactRef
from tests.test_agent_context_computation_reuse import _capture_context


@pytest.mark.parametrize(
    "directions",
    [
        {"physicalCaptureColumn": 0},
        {
            "physicalCaptureColumn": "capture",
            "cellQc": {"physicalCaptureColumn": "donor"},
        },
        {"physicalCaptureColumn": "absent"},
        {"physicalCaptureColumn": "capture", "pooledReferenceCaptures": ["d1", "d1"]},
        {"physicalCaptureColumn": "capture", "pooledReferenceCaptures": [1, 2]},
        {"pooledReferenceCaptures": ["d1", "d2"]},
    ],
)
def test_unproven_or_conflicting_capture_scope_cannot_create_qc_policies(
    directions: dict[str, Any],
) -> None:
    deps, characterized = _capture_context()
    deps.directions = directions
    with pytest.raises(ValueError):
        q._offered_qc_profiles(deps, characterized)
    assert deps.qcDesignData is None


def test_reference_proposal_cannot_replace_explicit_caller_references() -> None:
    deps, _ = _capture_context()
    deps.directions["pooledReferenceCaptures"] = ["d1", "d2"]
    deps.captureProposal = CaptureProposal(
        column="capture",
        provenanceQuote="capture is a physical capture",
        referenceCaptures=["d2", "d3"],
        referenceProvenanceQuote="d2 and d3 are reference captures",
    )
    with pytest.raises(ValueError, match="conflicts"):
        q._directed_pooled_reference_captures(deps)


def test_nonnumeric_counts_remain_missing_for_every_capture() -> None:
    deps, characterized = _capture_context()
    deps.store.cells._values["RNA_nCounts"] = np.asarray(["unavailable"] * 24)
    profiles = q._offered_qc_profiles(deps, characterized)
    assert profiles
    for profile in profiles:
        source = next(
            item for item in profile.metricSources if item.metricName == "RNA_nCounts"
        )
        assert not source.usableForFiltering
        assert source.missingCellsByCapture == {"d1": 8, "d2": 8, "d3": 8}
        assert "RNA_nCounts" not in profile.attributes
        if isinstance(profile.resolvedBounds, dict):
            assert "RNA_nCounts" not in profile.resolvedBounds
        else:
            assert all(
                bound.get("metric") != "RNA_nCounts" for bound in profile.resolvedBounds
            )


@pytest.mark.parametrize(
    "damage", ["metadataShape", "artifactShape", "duplicateMetric"]
)
def test_competing_qc_sources_require_exact_axes_and_unique_execution_names(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    deps, _ = _capture_context()
    if damage == "metadataShape":
        monkeypatch.setattr(q, "_active_cell_count", lambda _: 23)
    else:
        source = NamedArtifactSource(
            name="externalCounts",
            artifact=artifact_reference(
                ArtifactRef("assay", "quality_metric", "a" * 64, "RNA")
            ),
        )
        deps.qualityMetricArtifacts = (
            [source] if damage == "artifactShape" else [source, source]
        )
        monkeypatch.setattr(
            q,
            "_resolved_artifact_values",
            lambda *_args, **_kwargs: np.ones(23 if damage == "artifactShape" else 24),
        )
        monkeypatch.setattr(
            q,
            "inspect_artifact",
            lambda *_: SimpleNamespace(operation="externalQuality", inputs={}),
        )
    with pytest.raises(ValueError, match="align|not unique"):
        q._qc_metric_sources(deps, ("RNA", "RNA"))


def test_qc_artifact_provenance_keeps_valid_refs_without_promoting_malformed_records() -> (
    None
):
    valid = ArtifactRef("assay", "feature_selection", "a" * 64, "RNA")
    malformed = {
        "scope": "assay",
        "kind": "feature_selection",
        "artifact_id": "missing-fields",
    }
    refs = q._artifact_input_references(
        {"untrusted": malformed, "valid": valid.to_dict(), "duplicate": [valid]}
    )
    assert refs == [artifact_reference(valid)]


@pytest.mark.parametrize("column", ["condition", "donor"])
def test_capture_safety_rejects_misaligned_condition_or_independent_unit_values(
    column: str,
) -> None:
    deps, characterized = _capture_context()
    deps.qcDesignData = q._QcDesignData(deps.cells)
    deps.qcDesignData.values[column] = np.asarray(["value"] * 23)
    with pytest.raises(ValueError, match="align"):
        q._capture_design_safety(deps, characterized, deps.cells.fetch("capture"), "d1")
    with pytest.raises(ValueError, match="align"):
        q._design_retention(
            deps, characterized, np.ones(24, dtype=bool), np.ones(24, dtype=bool)
        )


def test_capture_exclusion_preserves_margins_but_cannot_claim_complete_repeated_donor_pairs() -> (
    None
):
    deps, characterized = _capture_context()
    deps.store.cells._values["condition"][16:] = "case"
    rows, conditions, independent = q._capture_design_safety(
        deps, characterized, deps.cells.fetch("capture"), "d1"
    )
    assert conditions and not independent
    assert rows[0]["completePairsAfterExclusion"] == 1
    assert rows[0]["incompletePairsAfterExclusion"] == 1
    characterized.coefficients[0]["scope"] = "withinCell"
    rows, conditions, independent = q._capture_design_safety(
        deps, characterized, deps.cells.fetch("capture"), "d1"
    )
    assert rows == [] and not conditions and not independent


def test_core_default_failure_is_reported_without_claiming_that_policy_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps, characterized = _capture_context()
    original = q.project_auto_filter_profile

    def reject(action: str, **kwargs: Any) -> Any:
        if action == "globalGaussian":
            raise ValueError("core projection unavailable")
        return original(action, **kwargs)

    monkeypatch.setattr(q, "project_auto_filter_profile", reject)
    profiles = q._offered_qc_profiles(deps, characterized)
    assert profiles and not any(
        profile.action == "globalGaussian" for profile in profiles
    )
    assert any(
        "core projection unavailable" in note
        for profile in profiles
        for note in profile.notes
    )
