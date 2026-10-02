from typing import Any

import pytest

from scarf.agent.choices import (
    validate_annotations,
    validate_choice,
    validate_context,
)
from scarf.agent.models import (
    AnnotationDecision,
    Choice,
    ContextDecision,
    Study,
)


def test_changed_graph_partition_ids_are_local_to_offered_set() -> None:
    value = Choice(
        action="shortlist",
        optionIds=["c1:r0.5"],
        rationale="Measured marker coherence",
        evidenceIds=["c1:r0.5"],
    )
    validate_choice(value, options={"c1:r0.5", "c1:r1.0"}, actions={"shortlist"})
    with pytest.raises(ValueError, match="unknown optionIds"):
        validate_choice(value, options={"c0:r0.5"}, actions={"shortlist"})


def test_context_does_not_authorize_invented_exclusions_or_heldout_columns() -> None:
    study = Study(context="A descriptive study", objective="Identify populations")
    value = ContextDecision(
        columnRoles={"cell_type": "protected"},
        excludeFeatures=["XIST"],
        rationale="Invented",
    )
    with pytest.raises(ValueError, match="held-out.*feature exclusions"):
        validate_context(value, columns={"batch"}, study=study)


def test_annotations_require_specific_observed_positive_markers() -> None:
    clusters = [
        {
            "clusterId": "0",
            "markers": [
                {"gene": "A", "score": 0.8, "fracExp": 0.5},
                {"gene": "B", "score": -0.1, "fracExp": 0.5},
            ],
        }
    ]
    value = AnnotationDecision.model_validate(
        dict(
            annotations=[
                {
                    "clusterId": "0",
                    "identity": "T cell",
                    "supportingMarkers": ["A", "B"],
                    "rationale": "Too weak",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="two observed positive"):
        validate_annotations(value, clusters=clusters)
    unassigned = AnnotationDecision.model_validate(
        dict(
            annotations=[
                {
                    "clusterId": "0",
                    "identity": "unassigned",
                    "rationale": "Markers do not establish a lineage",
                }
            ]
        )
    )
    validate_annotations(unassigned, clusters=clusters)


def test_annotations_cover_every_cluster_once() -> None:
    value = AnnotationDecision.model_validate(dict(annotations=[]))
    with pytest.raises(ValueError, match="exactly these clusterIds"):
        validate_annotations(value, clusters=[{"clusterId": "0", "markers": []}])


def test_context_rejects_all_role_conflicts_before_acceptance() -> None:
    study = Study(
        context="Technical batches cross biological conditions",
        objective="Describe populations",
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
    )
    value = ContextDecision(
        columnRoles={
            "batch": "protected",
            "condition": "ignore",
            "sample": "technical",
        },
        evidenceIds=["fabricated:proof"],
        rationale="Invalid roles and unsupported evidence",
    )
    with pytest.raises(ValueError) as error:
        validate_context(value, columns={"batch", "condition", "sample"}, study=study)
    message = str(error.value)
    assert "cannot change a supplied technical role" in message
    assert "cannot remove a supplied protected role" in message
    assert "cannot authorize a new technical batch column" in message
    assert "unknown evidenceIds" in message


def test_context_preserves_declared_roles_and_accepts_grounded_protection() -> None:
    study = Study(
        context="Batches cross conditions",
        objective="Describe populations",
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
        featureExclusions=["LOCAL_REPORTER"],
    )
    value = ContextDecision(
        columnRoles={
            "batch": "technical",
            "condition": "protected",
            "time": "protected",
        },
        excludeFeatures=["LOCAL_REPORTER"],
        evidenceIds=["column:time", "source:summary"],
        rationale="Study time is biological",
    )
    validate_context(value, columns={"batch", "condition", "time"}, study=study)


def test_context_optional_evidence_inventory_restricts_citations() -> None:
    study = Study(context="Study", objective="Describe populations")
    value = ContextDecision(
        rationale="Unsupported reference", evidenceIds=["reference:0"]
    )
    with pytest.raises(ValueError, match="unknown evidenceIds"):
        validate_context(value, columns=set(), study=study)
    validate_context(value, columns=set(), study=study, evidence_ids={"reference:0"})


def test_annotation_can_cite_observed_weak_markers_as_contradictory() -> None:
    clusters = [
        {
            "clusterId": "0",
            "markers": [
                {"gene": "A", "score": 0.25, "fracExp": 0.2},
                {"gene": "B", "score": 0.9, "fracExp": 0.7},
            ],
            "weakMarkers": [{"gene": "C", "score": -0.2, "fracExp": 0.01}],
        }
    ]
    value = AnnotationDecision.model_validate(
        dict(
            annotations=[
                {
                    "clusterId": "0",
                    "identity": "provisional lineage",
                    "supportingMarkers": ["A", "B"],
                    "contradictingMarkers": ["C"],
                    "rationale": "A and B support the provisional identity; C has weak support",
                }
            ]
        )
    )
    validate_annotations(value, clusters=clusters)


@pytest.mark.parametrize(
    ("score", "fraction"),
    [
        (0.24, 0.8),
        (0.8, 0.19),
        (None, 0.8),
        (0.8, None),
        (float("nan"), 0.8),
        (float("inf"), 0.8),
    ],
)
def test_named_annotation_requires_two_supported_markers(
    score: Any, fraction: Any
) -> None:
    clusters = [
        {
            "clusterId": "0",
            "markers": [
                {"gene": "A", "score": 0.9, "fracExp": 0.8},
                {"gene": "B", "score": score, "fracExp": fraction},
            ],
        }
    ]
    value = AnnotationDecision.model_validate(
        dict(
            annotations=[
                {
                    "clusterId": "0",
                    "identity": "provisional lineage",
                    "supportingMarkers": ["A", "B"],
                    "rationale": "Insufficient measured support",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="score >= 0.25 and fracExp >= 0.2"):
        validate_annotations(value, clusters=clusters)


def test_annotation_rejects_fabricated_marker_in_both_evidence_lists() -> None:
    value = AnnotationDecision.model_validate(
        dict(
            annotations=[
                {
                    "clusterId": "0",
                    "identity": "unassigned",
                    "contradictingMarkers": ["INVENTED"],
                    "rationale": "Unsupported citation",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="markers must be observed"):
        validate_annotations(
            value, clusters=[{"clusterId": "0", "markers": [], "weakMarkers": []}]
        )


def test_choice_reports_all_action_cardinality_citation_and_option_errors() -> None:
    value = Choice(
        action="experiment",
        optionIds=["invented", "invented"],
        evidenceIds=["unmeasured"],
        rationale="Unsupported experiment",
    )
    with pytest.raises(ValueError) as error:
        validate_choice(
            value,
            options={"c0:r0.5"},
            actions={"shortlist"},
            evidence_ids={"c0"},
        )
    message = str(error.value)
    assert "action must be one of ['shortlist']" in message
    assert "select between one and 1 optionIds" in message
    assert "unknown optionIds: ['invented']; offered: ['c0:r0.5']" in message
    assert "optionIds must not contain duplicates" in message
    assert "unknown evidenceIds: ['unmeasured']" in message


def test_deferred_choice_requires_an_actionable_question_and_no_selection() -> None:
    value = Choice(
        action="defer", optionIds=["c0:r0.5"], question=" \n ", rationale="Missing fact"
    )
    with pytest.raises(ValueError) as error:
        validate_choice(value, options={"c0:r0.5"}, actions={"defer"})
    assert "must not select options" in str(error.value)
    assert "requires an actionable question" in str(error.value)

    question = Choice(
        action="defer",
        question="Which assay contains RNA counts?",
        evidenceIds=["source:summary"],
        rationale="Multiple assays are available",
        deferralReason="missingEssentialInput",
    )
    validate_choice(
        question,
        options=set(),
        actions={"defer"},
        evidence_ids={"source:summary"},
        unresolved_fact_ids={"source:summary"},
    )


def test_annotation_aggregates_cluster_marker_conflicts_and_blank_identity() -> None:
    value = AnnotationDecision.model_validate(
        {
            "annotations": [
                {
                    "clusterId": "unknown",
                    "rationale": "An invented cluster cannot receive an identity",
                },
                {
                    "clusterId": "0",
                    "identity": " \t ",
                    "supportingMarkers": ["A"],
                    "contradictingMarkers": ["A"],
                    "rationale": "The same observation cannot support and contradict",
                },
            ]
        }
    )
    with pytest.raises(ValueError) as error:
        validate_annotations(
            value,
            clusters=[
                {
                    "clusterId": "0",
                    "markers": [{"gene": "A", "score": 0.8, "fracExp": 0.7}],
                }
            ],
        )
    message = str(error.value)
    assert "annotate exactly these clusterIds once each" in message
    assert "supporting and contradicting markers must differ" in message
    assert "at least two observed positive markers" in message
    assert "identity must not be blank" in message


def test_unassigned_identity_is_case_insensitive_but_requires_low_confidence() -> None:
    value = AnnotationDecision.model_validate(
        {
            "annotations": [
                {
                    "clusterId": "0",
                    "identity": " Unassigned ",
                    "confidence": "high",
                    "rationale": "Identity is unresolved",
                }
            ]
        }
    )
    with pytest.raises(
        ValueError, match="unassigned identity must have low confidence"
    ):
        validate_annotations(value, clusters=[{"clusterId": "0", "markers": []}])
