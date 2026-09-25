from collections.abc import Mapping, Sequence
from typing import Any

from ...storage.refs import ArtifactRef
from ..tools import core_artifact_reference
from ..types import ExperimentalTuningHandoff
from .contracts import (
    _CANDIDATE_ID,
    ParameterCandidate,
    ParameterTuningDependencies,
)
from .execution import normalized_artifact_shape
from .prompts import (
    build_initial_parameter_candidates,
    get_default_parameter_candidates,
)

_MAX_CANDIDATES_OFFERED = 25


def _resolve_experimental_tuning_handoff(
    *,
    normalized_cell_selection: ArtifactRef,
    batch_columns: Sequence[str],
    preservation_columns: Sequence[str],
    experimental_handoff: ExperimentalTuningHandoff | None,
) -> tuple[ArtifactRef, list[str], list[str]]:
    resolved_batch_columns = list(batch_columns)
    resolved_preservation_columns = list(preservation_columns)
    if experimental_handoff is None:
        return (
            normalized_cell_selection,
            resolved_batch_columns,
            resolved_preservation_columns,
        )

    handoff_batch_columns = list(experimental_handoff.batchColumns)
    canonical_batch_columns = sorted(set(handoff_batch_columns))
    if len(canonical_batch_columns) != len(handoff_batch_columns):
        raise ValueError("experimental_handoff batch columns must be unique")
    handoff_cell_selection = core_artifact_reference(experimental_handoff.cellSelection)
    if not isinstance(handoff_cell_selection, ArtifactRef):
        raise ValueError("experimental_handoff lacks an exact cell selection")
    if handoff_cell_selection != normalized_cell_selection:
        raise ValueError("normalized selection conflicts with experimental_handoff")
    if resolved_batch_columns and sorted(resolved_batch_columns) != (
        canonical_batch_columns
    ):
        raise ValueError("batch_columns conflict with experimental_handoff")
    if resolved_preservation_columns and resolved_preservation_columns != list(
        experimental_handoff.preservationColumns
    ):
        raise ValueError("preservation_columns conflict with experimental_handoff")
    if experimental_handoff.batchAction == "needsInput":
        raise ValueError("Experimental Context requires input before tuning")
    if experimental_handoff.batchAction == "skip" and experimental_handoff.batchColumns:
        raise ValueError("A skip handoff must not contain batch columns")
    if experimental_handoff.batchAction == "evaluateHarmony":
        expected_coefficients = set(experimental_handoff.coefficientsOfInterest)
        safe_coefficients = {
            item.coefficient
            for item in experimental_handoff.batchSafety
            if item.status == "safe" and item.batchColumns == canonical_batch_columns
        }
        if (
            not expected_coefficients
            or not canonical_batch_columns
            or safe_coefficients != expected_coefficients
        ):
            raise ValueError(
                "Harmony handoff lacks safe evidence for every coefficient"
            )
    if experimental_handoff.batchAction == "unsafe":
        expected_coefficients = set(experimental_handoff.coefficientsOfInterest)
        exact_safety = [
            item
            for item in experimental_handoff.batchSafety
            if item.batchColumns == canonical_batch_columns
            and item.coefficient in expected_coefficients
        ]
        if (
            not expected_coefficients
            or {item.coefficient for item in exact_safety} != expected_coefficients
            or any(item.status == "notComputed" for item in exact_safety)
            or not any(item.status == "unsafe" for item in exact_safety)
        ):
            raise ValueError("Unsafe handoff lacks exact unsafe batch evidence")
    if any(
        item.evidenceId not in experimental_handoff.evidenceIds
        for item in experimental_handoff.batchSafety
    ):
        raise ValueError("Experimental handoff does not cite its batch evidence")
    return (
        normalized_cell_selection,
        canonical_batch_columns,
        list(experimental_handoff.preservationColumns),
    )


def prepare_parameter_tuning_dependencies(
    store: Any,
    *,
    normalized: ArtifactRef,
    candidates: Sequence[ParameterCandidate] | None = None,
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    experimental_handoff: ExperimentalTuningHandoff | None = None,
    max_candidates: int = 5,
    max_refined_candidates: int = 0,
    allow_harmony_refinement: bool = True,
    pair_harmony_candidates: bool | None = None,
    min_cluster_cells: int = 20,
    identity_feature_limit: int = 64,
) -> tuple[ParameterTuningDependencies, list[str]]:
    """Validate one assay request and construct branch-safe dependencies."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least one")
    if max_refined_candidates < 0:
        raise ValueError("max_refined_candidates must be non-negative")
    if pair_harmony_candidates is not None and not isinstance(
        pair_harmony_candidates,
        bool,
    ):
        raise TypeError("pair_harmony_candidates must be a boolean or None")
    if min_cluster_cells < 1:
        raise ValueError("min_cluster_cells must be at least one")
    if identity_feature_limit < 2:
        raise ValueError("identity_feature_limit must be at least two")
    normalized = core_artifact_reference(normalized)
    if not isinstance(normalized, ArtifactRef) or normalized.kind != "normalized":
        raise TypeError("normalized must be a normalized ArtifactRef")
    if normalized.assay is None:
        raise ValueError("normalized artifact has no assay")
    normalized_status = store.inspect_artifact(normalized)
    if not getattr(normalized_status, "exists", True):
        raise ValueError("normalized artifact does not exist")
    if not getattr(normalized_status, "complete", False):
        raise ValueError("normalized artifact is incomplete")
    raw_cell_selection = (getattr(normalized_status, "inputs", None) or {}).get(
        "cell_selection"
    )
    if not isinstance(raw_cell_selection, Mapping):
        raise ValueError("normalized artifact has no cell-selection input")
    normalized_cell_selection = ArtifactRef.from_dict(dict(raw_cell_selection))
    if (
        normalized_cell_selection.scope != "datastore"
        or normalized_cell_selection.kind != "cell_selection"
        or normalized_cell_selection.assay is not None
    ):
        raise ValueError("normalized artifact has an invalid cell-selection input")
    from_assay = normalized.assay
    (
        resolved_cell_selection,
        resolved_batch_columns,
        resolved_preservation_columns,
    ) = _resolve_experimental_tuning_handoff(
        normalized_cell_selection=normalized_cell_selection,
        batch_columns=batch_columns,
        preservation_columns=preservation_columns,
        experimental_handoff=experimental_handoff,
    )
    if len(set(resolved_batch_columns)) != len(resolved_batch_columns):
        raise ValueError("batch_columns must be unique")
    seed_candidates = (
        get_default_parameter_candidates() if candidates is None else list(candidates)
    )
    if not seed_candidates:
        raise ValueError("candidates must be non-empty")
    if len(seed_candidates) > max_candidates:
        raise ValueError(
            f"Initial candidate count exceeds max_candidates={max_candidates}"
        )
    pair_harmony = (
        (
            experimental_handoff is not None
            and experimental_handoff.batchAction == "evaluateHarmony"
        )
        if pair_harmony_candidates is None
        else pair_harmony_candidates
    )
    candidate_values = build_initial_parameter_candidates(
        seed_candidates,
        pair_harmony=pair_harmony,
    )
    if len(candidate_values) + max_refined_candidates > _MAX_CANDIDATES_OFFERED:
        raise ValueError(
            "Initial and refined candidates may contain at most "
            f"{_MAX_CANDIDATES_OFFERED} values"
        )
    candidate_map: dict[str, ParameterCandidate] = {}
    for candidate in candidate_values:
        if not _CANDIDATE_ID.fullmatch(candidate.candidateId):
            raise ValueError(
                "candidateId must contain only ASCII letters, numbers, and underscores"
            )
        if candidate.candidateId in candidate_map:
            raise ValueError(f"Duplicate candidateId {candidate.candidateId!r}")
        if candidate.useHarmony and not resolved_batch_columns:
            raise ValueError(
                f"Candidate {candidate.candidateId!r} requires batch_columns"
            )
        if (
            candidate.useHarmony
            and experimental_handoff is not None
            and experimental_handoff.batchAction != "evaluateHarmony"
        ):
            raise ValueError(
                f"Candidate {candidate.candidateId!r} is not authorized for Harmony"
            )
        candidate_map[candidate.candidateId] = candidate
    harmony_authorized = (
        allow_harmony_refinement
        and bool(resolved_batch_columns)
        and (
            experimental_handoff is None
            or experimental_handoff.batchAction == "evaluateHarmony"
        )
    )
    normalized_shape = normalized_artifact_shape(store, normalized)
    deps = ParameterTuningDependencies(
        store=store,
        normalized=normalized,
        cellSelection=resolved_cell_selection,
        normalizedShape=normalized_shape,
        fromAssay=from_assay,
        candidates=candidate_map,
        candidatePhases={candidate_id: "initial" for candidate_id in candidate_map},
        batchColumns=tuple(resolved_batch_columns),
        preservationColumns=tuple(resolved_preservation_columns),
        harmonyAuthorized=harmony_authorized,
        maxCandidates=len(candidate_values) + max_refined_candidates,
        minClusterCells=min_cluster_cells,
        identityFeatureLimit=identity_feature_limit,
    )
    return deps, list(candidate_map)
