"""Tool-driven experimental-design and batch-correction assessment."""

from ..cell_quality.profiles import RegisteredCellQcProfile
from ..types import BatchSafetyEvidence
from .agent import ExperimentalContextAgent
from .contracts import (
    BatchCorrectionPlan,
    CaptureFailureEvidence,
    CellQcPlan,
    CellQcProfileEvidence,
    ContrastPlan,
    CovariateEvidence,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    InferenceUnit,
    NamedArtifactSource,
    QcMetricSourceEvidence,
    QcSourceConcordance,
    RepresentationEvaluation,
)
from .tools import (
    analyze_experimental_design,
    contrast_plans_from_characterization,
    inspect_cell_covariates,
    score_current_representation,
)
from .validation import validate_experimental_context

__all__ = [
    "BatchCorrectionPlan",
    "BatchSafetyEvidence",
    "CellQcPlan",
    "CellQcProfileEvidence",
    "CaptureFailureEvidence",
    "ContrastPlan",
    "CovariateEvidence",
    "ExperimentalContextAgent",
    "ExperimentalContextDecision",
    "ExperimentalContextDependencies",
    "ExperimentalContextResult",
    "InferenceUnit",
    "NamedArtifactSource",
    "QcMetricSourceEvidence",
    "QcSourceConcordance",
    "RepresentationEvaluation",
    "RegisteredCellQcProfile",
    "analyze_experimental_design",
    "contrast_plans_from_characterization",
    "inspect_cell_covariates",
    "score_current_representation",
    "validate_experimental_context",
]
