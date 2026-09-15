"""Small entry point for the supported automated RNA analysis."""

from pathlib import Path
from typing import Any

from .main import AgentOrchestrator
from .models import (
    AnalysisError,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
)


def analyze_rna(
    source: str | Path,
    *,
    model: Any,
    study_context: str,
    study_objective: str,
    assay: str | None = None,
    zarr_path: str | Path | None = None,
    score_doublets: bool = False,
) -> AutomatedWorkflowResult:
    """Choose and explain settings for one RNA assay, then execute them.

    ``source`` is a supported input file or an existing Zarr store. ``assay``
    selects the RNA assay when the input contains more than one. The workflow
    runs unattended and raises ``AnalysisError`` if essential evidence remains
    unresolved or execution fails. A completed result provides ``plot_embedding()``,
    ``get_markers()``, and ``report()``.

    Work is bounded by the advanced orchestrator's screening and full-cohort
    limits. Use that interface for explicit workspaces, execution limits, and
    resumable pauses. An identical repeated call reuses or resumes exact work.

    ``score_doublets=False`` skips advisory doublet scoring when Harmony is
    unavailable or prohibited. Harmony-eligible runs retain the matched doublet
    diagnostics required by the correction acceptance gate. Scoring does not
    remove cells. Changing this option requires a new workflow destination.
    Advisory scoring defaults to disabled for new beginner calls. Pass
    ``score_doublets=True`` explicitly to resume a run that enabled scoring.
    """
    if model is None or isinstance(model, str) and not model.strip():
        raise ValueError(
            "model must be a configured model or a non-empty model identifier"
        )
    request = AutomatedWorkflowRequest(
        sourcePath=str(source),
        zarrPath=str(zarr_path) if zarr_path is not None else None,
        studyContext=study_context,
        studyObjective=study_objective,
        primaryAssay=assay,
        markerAssay=assay,
        analysisAssays=[assay] if assay is not None else [],
    )
    config = AutomatedWorkflowConfig(
        inputPolicy="unattended",
        scoreDoublets=score_doublets,
    )
    result = AgentOrchestrator(model, config=config).run(request)
    if result.status != "completed":
        raise AnalysisError(result)
    return result
