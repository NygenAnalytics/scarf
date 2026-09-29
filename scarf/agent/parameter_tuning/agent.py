from collections.abc import Mapping, Sequence
from typing import Any

from ...storage.refs import ArtifactRef
from ..tools import core_artifact_reference
from .contracts import (
    _CANDIDATE_ID,
    ParameterCandidate,
    ParameterTuningDependencies,
)
from .execution import normalized_artifact_shape


def prepare_parameter_tuning_dependencies(
    store: Any,
    *,
    normalized: ArtifactRef,
    candidates: Sequence[ParameterCandidate],
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    max_candidates: int = 5,
    min_cluster_cells: int = 20,
) -> tuple[ParameterTuningDependencies, list[str]]:
    """Validate one assay request and construct branch-safe dependencies."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least one")
    if min_cluster_cells < 1:
        raise ValueError("min_cluster_cells must be at least one")
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
    resolved_batch_columns = list(batch_columns)
    if len(set(resolved_batch_columns)) != len(resolved_batch_columns):
        raise ValueError("batch_columns must be unique")
    seed_candidates = list(candidates)
    if not seed_candidates:
        raise ValueError("candidates must be non-empty")
    if len(seed_candidates) > max_candidates:
        raise ValueError(
            f"Initial candidate count exceeds max_candidates={max_candidates}"
        )
    candidate_map: dict[str, ParameterCandidate] = {}
    for candidate in seed_candidates:
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
        candidate_map[candidate.candidateId] = candidate
    normalized_shape = normalized_artifact_shape(store, normalized)
    deps = ParameterTuningDependencies(
        store=store,
        normalized=normalized,
        cellSelection=normalized_cell_selection,
        normalizedShape=normalized_shape,
        fromAssay=normalized.assay,
        candidates=candidate_map,
        batchColumns=tuple(resolved_batch_columns),
        preservationColumns=tuple(preservation_columns),
        maxCandidates=len(candidate_map),
        minClusterCells=min_cluster_cells,
    )
    return deps, list(candidate_map)
