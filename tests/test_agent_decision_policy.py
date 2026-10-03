"""Independent probes and lenient decisions preserve the frozen evidence bounds."""

import pytest

from scarf.agent.choices import (
    native_probe_options,
    resolve_deferral,
    validate_choice,
    validate_context,
)
from scarf.agent.models import Candidate, Choice, ContextDecision, Study


def test_native_probes_are_ordered_independent_changes_from_the_baseline() -> None:
    baseline = Candidate(candidateId="c0", hvgCount=1000, pcaDims=21, neighborsK=11)
    options = native_probe_options(
        baseline, {"retainedCells": 100, "availableFeatures": 5000}
    )
    assert list(options) == ["hvgCount", "pcaDims", "neighborsK"]
    assert [row.hvgCount for row in options["hvgCount"].values()] == [2000, 4000]
    assert [row.pcaDims for row in options["pcaDims"].values()] == [10, 30]
    assert [row.neighborsK for row in options["neighborsK"].values()] == [21, 41]
    for axis, offered in options.items():
        for option_id, row in offered.items():
            assert row.candidateId == option_id
            assert row.parentId == "c0"
            assert row.useHarmony is False
            assert {
                field
                for field in ("hvgCount", "pcaDims", "neighborsK")
                if getattr(row, field) != getattr(baseline, field)
            } == {axis}


def test_native_probes_skip_impossible_axes_and_retain_registered_fallbacks() -> None:
    baseline = Candidate(
        candidateId="baseline", hvgCount=2000, pcaDims=10, neighborsK=21
    )
    options = native_probe_options(
        baseline, {"retainedCells": 25, "availableFeatures": 2000}
    )
    assert list(options["hvgCount"]) == ["baseline:hvgCount:1000"]
    assert options["pcaDims"] == {}
    assert list(options["neighborsK"]) == ["baseline:neighborsK:11"]
    with pytest.raises(ValueError, match="uncorrected baseline"):
        native_probe_options(
            baseline.model_copy(update={"useHarmony": True}),
            {"retainedCells": 25, "availableFeatures": 2000},
        )


def test_pc_probe_feasibility_uses_measured_hvgs_not_the_requested_count() -> None:
    baseline = Candidate(candidateId="c0", hvgCount=1000, pcaDims=21, neighborsK=11)
    options = native_probe_options(
        baseline,
        {"retainedCells": 100, "availableFeatures": 5000, "actualHvgCount": 25},
    )
    assert list(options["pcaDims"]) == ["c0:pcaDims:10"]
    assert [row.hvgCount for row in options["hvgCount"].values()] == [2000, 4000]
    assert [row.neighborsK for row in options["neighborsK"].values()] == [21, 41]


def ambiguous_choice() -> Choice:
    return Choice(
        action="defer",
        deferralReason="ambiguousSelection",
        acceptableOptionIds=["c1:r0.5", "c0:r0.75"],
        evidenceIds=["c1:r0.5", "c0:r0.75"],
        question="Which supported population granularity is preferred?",
        rationale="Both measured partitions remain acceptable for this objective",
    )


def test_lenient_tie_uses_frozen_order_only_among_acceptable_options() -> None:
    value = ambiguous_choice()
    options = {"c0:r0.5", "c0:r0.75", "c1:r0.5"}
    validate_choice(value, options=options, actions={"defer"}, evidence_ids=options)
    order = ["c0:r0.5", "c0:r0.75", "c1:r0.5"]
    assert resolve_deferral(
        value, interaction_mode="lenient", option_order=order, unresolved_fact_ids=set()
    ) == {"action": "choose", "optionIds": ["c0:r0.75"]}
    assert (
        resolve_deferral(
            value,
            interaction_mode="strict",
            option_order=order,
            unresolved_fact_ids=set(),
        )
        is None
    )


@pytest.mark.parametrize(
    "acceptable", [[], ["c0:r0.75"], ["unknown", "c0:r0.75"], ["c0:r0.75", "c0:r0.75"]]
)
def test_ambiguous_selection_cannot_expand_or_duplicate_eligible_options(
    acceptable: list[str],
) -> None:
    value = ambiguous_choice().model_copy(update={"acceptableOptionIds": acceptable})
    with pytest.raises(ValueError, match="distinct eligible"):
        validate_choice(value, options={"c0:r0.75", "c1:r0.5"}, actions={"defer"})


def test_missing_essential_fact_requires_deterministic_evidence_and_never_resolves() -> (
    None
):
    value = Choice(
        action="defer",
        deferralReason="missingEssentialInput",
        question="Choose an assay?",
        evidenceIds=["input:assay"],
        rationale="Two prepared RNA assays are available",
    )
    with pytest.raises(ValueError, match="deterministic unresolved fact"):
        validate_choice(
            value, options=set(), actions={"defer"}, evidence_ids={"input:assay"}
        )
    validate_choice(
        value,
        options=set(),
        actions={"defer"},
        evidence_ids={"input:assay"},
        unresolved_fact_ids={"input:assay"},
    )
    assert (
        resolve_deferral(
            value,
            interaction_mode="lenient",
            option_order=[],
            unresolved_fact_ids={"input:assay"},
        )
        is None
    )


def test_normal_choice_requires_grounded_citations_and_no_deferral_fields() -> None:
    value = Choice(action="choose", optionIds=["c0:r0.5"], rationale="Adequate")
    with pytest.raises(ValueError, match="require supplied evidenceIds"):
        validate_choice(value, options={"c0:r0.5"}, actions={"choose"})
    value = value.model_copy(
        update={"evidenceIds": ["c0:r0.5"], "question": "Continue?"}
    )
    with pytest.raises(ValueError, match="must not contain deferral fields"):
        validate_choice(value, options={"c0:r0.5"}, actions={"choose"})


def test_context_fallback_preserves_declared_roles_without_promoting_guesses() -> None:
    study = Study(
        context="Prepared study", objective="Describe populations", sampleColumn="donor"
    )
    value = ContextDecision(
        columnRoles={"donor": "sample"},
        question="What identifies a capture?",
        deferralReason="uncertainMetadata",
        evidenceIds=["column:donor"],
        rationale="Capture identity is optional and unknown",
    )
    validate_context(value, columns={"donor"}, study=study)
    assert resolve_deferral(
        value, interaction_mode="lenient", option_order=[], unresolved_fact_ids=set()
    ) == {"action": "retainDeclaredRoles"}
    validate_context(
        ContextDecision(rationale="No additional roles are proposed"),
        columns={"donor"},
        study=study,
    )


def test_context_rejects_multiple_units_and_overriding_declared_roles() -> None:
    study = Study(
        context="Prepared study",
        objective="Describe populations",
        sampleColumn="donor",
        captureColumn="library",
        technicalBatchColumns=["batch"],
    )
    value = ContextDecision(
        columnRoles={
            "donor": "ignore",
            "library": "protected",
            "batch": "ignore",
            "a": "sample",
            "b": "sample",
        },
        evidenceIds=["source:summary"],
        rationale="Conflicting role declarations",
    )
    with pytest.raises(ValueError) as error:
        validate_context(
            value, columns={"donor", "library", "batch", "a", "b"}, study=study
        )
    message = str(error.value)
    assert "at most one column may have the sample role" in message
    assert "cannot replace the supplied sample column" in message
    assert "cannot change a supplied sample role" in message
    assert "cannot change a supplied capture role" in message
    assert "cannot change a supplied technical role" in message


@pytest.mark.parametrize("role", ["sample", "capture", "protected"])
def test_context_allows_overlapping_explicit_roles_without_reassigning_them(
    role: str,
) -> None:
    study = Study(
        context="One capture for each donor",
        objective="Describe populations",
        sampleColumn="donor",
        captureColumn="donor",
        protectedColumns=["donor"],
    )
    value = ContextDecision.model_validate(
        {
            "columnRoles": {"donor": role},
            "evidenceIds": ["column:donor"],
            "rationale": "All declared donor roles remain preserved",
        }
    )
    validate_context(value, columns={"donor"}, study=study)


def test_metadata_uncertainty_cannot_be_reused_as_permission_to_choose_a_partition() -> (
    None
):
    value = ambiguous_choice().model_copy(
        update={"deferralReason": "uncertainMetadata", "acceptableOptionIds": []}
    )
    with pytest.raises(ValueError, match="context deferral"):
        validate_choice(value, options={"c0:r0.75", "c1:r0.5"}, actions={"defer"})
    with pytest.raises(ValueError, match="Choices can resolve only ambiguousSelection"):
        resolve_deferral(
            value,
            interaction_mode="lenient",
            option_order=["c0:r0.75"],
            unresolved_fact_ids=set(),
        )


def test_selection_uncertainty_cannot_erase_context_roles() -> None:
    value = ContextDecision(
        question="Which clustering should be presented?",
        deferralReason="ambiguousSelection",
        evidenceIds=["source:summary"],
        rationale="This question concerns selection, not metadata roles",
    )
    with pytest.raises(ValueError, match="selection deferral"):
        validate_context(
            value,
            columns=set(),
            study=Study(context="Study", objective="Describe populations"),
        )
    with pytest.raises(ValueError, match="Context can resolve only uncertainMetadata"):
        resolve_deferral(
            value,
            interaction_mode="lenient",
            option_order=[],
            unresolved_fact_ids=set(),
        )


def test_fallback_rechecks_acceptable_options_after_eligibility_narrows() -> None:
    value = ambiguous_choice()
    validate_choice(value, options=set(value.acceptableOptionIds), actions={"defer"})
    with pytest.raises(ValueError, match="distinct eligible options"):
        resolve_deferral(
            value,
            interaction_mode="lenient",
            option_order=["c0:r0.75"],
            unresolved_fact_ids=set(),
        )


def test_choice_cannot_smuggle_fallback_options_into_a_nonambiguity_outcome() -> None:
    ordinary = Choice(
        action="choose",
        optionIds=["c0:r0.75"],
        acceptableOptionIds=["c1:r0.5"],
        evidenceIds=["c0:r0.75"],
        rationale="A final choice cannot contain another fallback",
    )
    with pytest.raises(ValueError, match="must not contain acceptableOptionIds"):
        validate_choice(ordinary, options={"c0:r0.75"}, actions={"choose"})
    with pytest.raises(ValueError, match="Only a deferred choice"):
        resolve_deferral(
            ordinary,
            interaction_mode="lenient",
            option_order=["c0:r0.75"],
            unresolved_fact_ids=set(),
        )
    unsupported = ambiguous_choice().model_copy(
        update={"deferralReason": "unsupportedObjective"}
    )
    with pytest.raises(ValueError, match="only valid for ambiguousSelection"):
        validate_choice(
            unsupported, options=set(unsupported.acceptableOptionIds), actions={"defer"}
        )


def test_supplied_role_evidence_is_citable_and_held_out_annotations_are_absent() -> (
    None
):
    from scarf.agent.choices import context_evidence_ids
    from scarf.agent.diagnostics import resolved_roles

    study = Study(
        context="Technical library batches cross the biological conditions",
        objective="Describe populations",
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
        sampleColumn="donor",
        captureColumn="library",
        excludedColumns=["author_cell_type"],
        referenceFiles=["local-reference.txt"],
    )
    columns = {"batch", "condition", "donor", "library"}
    offered = context_evidence_ids(columns, study)
    cited = {
        identifier for row in resolved_roles(study) for identifier in row["evidenceIds"]
    }
    assert cited == {
        "study:technical",
        "study:protected",
        "study:sample",
        "study:capture",
    }
    assert cited <= offered
    assert "column:author_cell_type" not in offered
    assert "reference:0" in offered and "reference:1" not in offered
    value = ContextDecision(
        columnRoles={
            "batch": "technical",
            "condition": "protected",
            "donor": "sample",
            "library": "capture",
        },
        evidenceIds=sorted(cited),
        rationale="Preserve the supplied roles described in the measured context",
    )
    validate_context(value, columns=columns, study=study)


def test_context_role_citations_require_supplied_visible_declarations() -> None:
    from scarf.agent.choices import context_evidence_ids

    study = Study(
        context="Unspecified donor identity", objective="Describe populations"
    )
    assert not any(
        identifier.startswith("study:")
        for identifier in context_evidence_ids({"donor"}, study)
    )
    supplied_but_hidden = study.model_copy(update={"sampleColumn": "author_identity"})
    assert "study:sample" not in context_evidence_ids({"donor"}, supplied_but_hidden)
    value = ContextDecision(
        columnRoles={"author_identity": "protected", "donor": "technical"},
        excludeFeatures=["NOT_SUPPLIED"],
        evidenceIds=["study:sample", "fabricated:context"],
        rationale="Invalid interpretation must report all repairable conflicts",
    )
    with pytest.raises(ValueError) as error:
        validate_context(value, columns={"donor"}, study=study)
    message = str(error.value)
    assert "unknown or held-out columns" in message
    assert "cannot authorize a new technical batch column" in message
    assert "feature exclusions must come from the supplied inventory" in message
    assert "unknown evidenceIds" in message
    assert "source:summary" in message and "column:donor" in message


def test_policy_resolution_requires_a_known_mode_and_a_complete_deferral() -> None:
    order = ["c0:r0.75", "c1:r0.5"]
    with pytest.raises(
        ValueError, match="^interaction_mode must be strict or lenient$"
    ):
        resolve_deferral(
            ambiguous_choice(),
            interaction_mode="automatic",
            option_order=order,
            unresolved_fact_ids=set(),
        )
    # A stored deferral is rechecked before any policy can resolve it.
    incomplete = ambiguous_choice().model_copy(
        update={"question": " ", "evidenceIds": []}
    )
    with pytest.raises(ValueError) as error:
        resolve_deferral(
            incomplete,
            interaction_mode="lenient",
            option_order=order,
            unresolved_fact_ids=set(),
        )
    assert str(error.value) == (
        "defer requires an actionable question; defer requires supplied evidenceIds"
    )
