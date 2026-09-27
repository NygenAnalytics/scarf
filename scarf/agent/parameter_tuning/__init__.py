"""Bounded parameter tuning over explicit Scarf analysis candidates."""

from .agent import prepare_parameter_tuning_dependencies
from .contracts import (
    ArtifactRecord,
    CandidateComparison,
    FinalGraphComparison,
    FinalGraphNeedsInput,
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
)
from .execution import (
    execute_parameter_candidate,
    normalized_artifact_shape,
    run_candidate_reduction,
    validate_parameter_candidate_rank,
)
from .prompts import (
    build_initial_parameter_candidates,
    get_default_parameter_candidates,
)
from .selection import (
    annotate_candidate_dominance,
    finalize_parameter_tuning_selection,
    harmony_acceptance_gate,
    promote_parameter_candidate,
)

__all__ = [
    "annotate_candidate_dominance",
    "ArtifactRecord",
    "build_initial_parameter_candidates",
    "CandidateComparison",
    "execute_parameter_candidate",
    "FinalGraphComparison",
    "FinalGraphNeedsInput",
    "FinalGraphSelection",
    "finalize_parameter_tuning_selection",
    "IntegrationCandidateEvaluation",
    "IntegrationMetrics",
    "normalized_artifact_shape",
    "ParameterCandidate",
    "ParameterCandidateEvaluation",
    "ParameterMetrics",
    "ParameterSearchPlan",
    "ParameterTuningDependencies",
    "ParameterTuningNeedsInput",
    "ParameterTuningReport",
    "get_default_parameter_candidates",
    "harmony_acceptance_gate",
    "prepare_parameter_tuning_dependencies",
    "promote_parameter_candidate",
    "run_candidate_reduction",
    "validate_parameter_candidate_rank",
]
