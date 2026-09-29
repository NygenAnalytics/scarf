"""Bounded parameter tuning over explicit Scarf analysis candidates."""

from .agent import prepare_parameter_tuning_dependencies
from .contracts import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
)
from .execution import (
    execute_parameter_candidate,
    normalized_artifact_shape,
    validate_parameter_candidate_rank,
)
from .selection import (
    finalize_parameter_tuning_selection,
    harmony_acceptance_gate,
    promote_parameter_candidate,
)

__all__ = [
    "ArtifactRecord",
    "execute_parameter_candidate",
    "finalize_parameter_tuning_selection",
    "normalized_artifact_shape",
    "ParameterCandidate",
    "ParameterCandidateEvaluation",
    "ParameterMetrics",
    "ParameterTuningDependencies",
    "ParameterTuningNeedsInput",
    "ParameterTuningReport",
    "harmony_acceptance_gate",
    "prepare_parameter_tuning_dependencies",
    "promote_parameter_candidate",
    "validate_parameter_candidate_rank",
]
